"""
TrafficVision End-to-End Processing Pipeline:
  - Video validation & frame ingestion
  - YOLO vehicle detection (car, motorcycle, bus, truck)
  - ByteTrack persistent multi-object tracking
  - Bidirectional virtual line counting
  - Perspective calibration & speed estimation
  - Multi-signal road ROI density & congestion classification
  - Explainable rule-based incident detection
  - Rich video annotation & structured JSON/CSV report generation
"""
import argparse
import os
import sys
from pathlib import Path

_mpl_cache = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.perspective import PerspectiveTransformer, example_calibration
from src.analytics import SpeedEstimator, LineCounter
from src.congestion import TrafficDensityAnalyzer, CongestionSnapshot
from src.incidents import IncidentDetector, IncidentEvent
from src.reporting import ReportGenerator


def load_transformer(calibration_path: Optional[str]) -> Tuple[PerspectiveTransformer, Optional[str]]:
    if calibration_path is None or not Path(calibration_path).exists():
        transformer = example_calibration()
        warning = (
            "No calibration file provided — speed and physical density use uncalibrated "
            "illustrative scale. Run tools/calibrate.py to measure real-world ground truth."
        )
        return transformer, warning

    transformer = PerspectiveTransformer.from_json(calibration_path)
    if not transformer.is_calibrated:
        warning = (
            f"Calibration is illustrative/unverified ({transformer.provenance}): "
            f"{transformer.validation_message}. Speed numbers are approximate."
        )
    else:
        warning = None
    return transformer, warning


class TrafficPipeline:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.35,
        calibration_path: Optional[str] = None,
        counting_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        road_roi: Optional[np.ndarray] = None,
        frame_skip: int = 1,
        stationary_time_sec: float = 3.0,
        free_flow_speed_kmh: float = 60.0,
        target_classes: Optional[List[str]] = None,
    ):
        self.weights = weights
        self.conf = conf
        self.calibration_path = calibration_path
        self.counting_line = counting_line
        self.road_roi = road_roi
        self.frame_skip = max(1, frame_skip)
        self.stationary_time_sec = stationary_time_sec
        self.free_flow_speed_kmh = free_flow_speed_kmh
        self.target_classes = target_classes

        # Initialize detector once (avoid repeated reload overhead)
        self.detector = VehicleDetector(
            weights_path=weights,
            conf_threshold=conf,
            target_classes=target_classes,
        )

    def process_video(
        self,
        source_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        generate_annotated_video: bool = True,
    ) -> Dict:
        """
        Processes a video file end-to-end and returns structured analytics.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Input video file does not exist: {source_path}")

        # Validate video format
        valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        if source.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported video format {source.suffix}. Expected {valid_extensions}")

        start_time = time.time()
        video_info = sv.VideoInfo.from_video_path(str(source))
        fps = video_info.fps if video_info.fps > 0 else 25.0
        h, w = video_info.height, video_info.width
        total_frames = video_info.total_frames or 1

        # Fallback counting line across bottom 60% of frame if not specified
        if self.counting_line is not None:
            line_start, line_end = self.counting_line
        else:
            line_start = (int(w * 0.05), int(h * 0.60))
            line_end = (int(w * 0.95), int(h * 0.60))

        # Default road ROI polygon if not specified: lower trapezoid
        if self.road_roi is not None:
            road_roi = self.road_roi
        else:
            road_roi = np.array([
                [int(w * 0.20), int(h * 0.40)],
                [int(w * 0.80), int(h * 0.40)],
                [int(w * 0.98), int(h * 0.95)],
                [int(w * 0.02), int(h * 0.95)],
            ], dtype=np.int32)

        # Initialize core components
        tracker = VehicleTracker(frame_rate=int(fps))
        transformer, calib_warning = load_transformer(self.calibration_path)
        speed_estimator = SpeedEstimator(transformer, fps=fps)
        line_counter = LineCounter(start=line_start, end=line_end)

        # Use calibrated ground area if genuinely available, else document nominal default
        if transformer.is_calibrated and transformer.calibrated_area_sq_m > 10.0:
            roi_area_m2 = transformer.calibrated_area_sq_m
            roi_source = "calibrated_ground_plane"
        else:
            roi_area_m2 = 400.0
            roi_source = "uncalibrated_nominal_default"

        congestion_analyzer = TrafficDensityAnalyzer(
            fps=fps,
            road_roi=road_roi,
            free_flow_speed_kmh=self.free_flow_speed_kmh,
            roi_area_sq_meters=roi_area_m2,
            roi_area_source=roi_source,
        )
        incident_detector = IncidentDetector(
            fps=fps,
            stationary_time_threshold_sec=self.stationary_time_sec,
        )

        # Annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.45, text_padding=3)
        trace_annotator = sv.TraceAnnotator(trace_length=int(fps * 2), thickness=2)
        line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.55)

        sink = None
        if generate_annotated_video and output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # Adjust output video info if frame_skip > 1
            out_info = sv.VideoInfo(
                width=w,
                height=h,
                fps=fps / self.frame_skip,
                total_frames=int(total_frames / self.frame_skip),
            )
            sink = sv.VideoSink(output_path, out_info)
            sink.__enter__()

        frames_processed = 0
        total_detections_count = 0
        unique_tids_seen = set()
        active_incident_banners: List[Tuple[str, int]] = []  # (text, expiry_frame)

        try:
            for frame_idx, frame in enumerate(sv.get_video_frames_generator(str(source))):
                if frame_idx % self.frame_skip != 0:
                    continue

                frames_processed += 1

                # 1. Vehicle Detection
                detections = self.detector.detect(frame)
                total_detections_count += len(detections)

                # 2. Tracking
                tracked = tracker.update(detections, frame_idx=frame_idx)
                if tracked.tracker_id is not None:
                    for tid in tracked.tracker_id:
                        unique_tids_seen.add(int(tid))

                # 3. Speed Estimation
                speeds = speed_estimator.update(tracked, frame_idx=frame_idx)

                # 4. Counting Line Trigger
                in_count, out_count = line_counter.update(
                    tracked,
                    class_name_getter=self.detector.class_name,
                    frame_idx=frame_idx,
                    speeds=speeds,
                )

                # 5. Congestion Analysis
                congestion_snap = congestion_analyzer.analyze_frame(
                    tracked,
                    speeds=speeds,
                    frame_idx=frame_idx,
                    frame_shape=(h, w),
                    line_crossing_total=in_count + out_count,
                )

                # 6. Incident Detection
                new_incidents = incident_detector.process_frame(
                    tracked,
                    speeds=speeds,
                    frame_idx=frame_idx,
                    tracker_history=tracker.history,
                    congestion_level=congestion_snap.congestion_level,
                )
                for inc in new_incidents:
                    active_incident_banners.append((f"ALERT: {inc.title} (#{','.join(map(str, inc.vehicle_ids))})", frame_idx + int(fps * 2)))

                # Progress reporting
                if progress_callback and frame_idx % 10 == 0:
                    pct = min(0.99, frame_idx / total_frames)
                    progress_callback(pct, f"Processing frame {frame_idx}/{total_frames}...")

                # 7. Render Annotated Video
                if sink is not None:
                    annotated = frame.copy()

                    # Draw Road ROI outline
                    cv2.polylines(annotated, [road_roi], isClosed=True, color=(180, 180, 0), thickness=2)

                    # Draw Line Zone
                    line_annotator.annotate(annotated, line_counter.line_zone)

                    # Build labels
                    labels = []
                    for i in range(len(tracked)):
                        cid = tracked.class_id[i] if tracked.class_id is not None else 0
                        tid = tracked.tracker_id[i] if tracked.tracker_id is not None else -1
                        cname = self.detector.class_name(int(cid))
                        spd = speeds.get(int(tid))

                        lbl = f"#{tid} {cname}"
                        if spd is not None and transformer.is_calibrated:
                            lbl += f" {spd:.0f}km/h"
                        elif spd is not None:
                            lbl += f" ~{spd:.0f}km/h*"
                        labels.append(lbl)

                    # Overlays
                    annotated = trace_annotator.annotate(annotated, tracked)
                    annotated = box_annotator.annotate(annotated, tracked)
                    annotated = label_annotator.annotate(annotated, tracked, labels=labels)

                    # HUD Overlay Panel (Top Bar)
                    hud_h = 55
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (0, 0), (w, hud_h), (25, 25, 28), -1)
                    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

                    # Status Texts
                    cv2.putText(
                        annotated, f"TRAFFICVISION | IN: {in_count}  OUT: {out_count}  ACTIVE: {len(tracked)}",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                    )

                    cong_color = {
                        "LOW": (0, 220, 0),
                        "MODERATE": (0, 215, 255),
                        "HIGH": (0, 140, 255),
                        "SEVERE": (0, 0, 240),
                    }.get(congestion_snap.congestion_level, (200, 200, 200))

                    cv2.putText(
                        annotated, f"CONGESTION: {congestion_snap.congestion_level} (Avg: {congestion_snap.avg_speed_kmh:.0f} km/h)",
                        (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cong_color, 2
                    )

                    # Active Incident Banner
                    active_banners = [b for b in active_incident_banners if b[1] >= frame_idx]
                    if active_banners:
                        banner_text = active_banners[-1][0]
                        banner_w = len(banner_text) * 12 + 20
                        cv2.rectangle(annotated, (w - banner_w - 15, 10), (w - 15, 45), (0, 0, 200), -1)
                        cv2.putText(
                            annotated, banner_text,
                            (w - banner_w - 5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                        )

                    sink.write_frame(annotated)

        finally:
            if sink is not None:
                sink.__exit__(None, None, None)

        proc_duration = max(0.01, time.time() - start_time)
        proc_fps = round(frames_processed / proc_duration, 1)

        if progress_callback:
            progress_callback(1.0, f"Completed processing {frames_processed} frames in {proc_duration:.1f}s.")

        # Aggregate Result Payload
        speed_summary = speed_estimator.get_summary_stats()
        congestion_summary = congestion_analyzer.get_summary()
        incident_summary = incident_detector.get_summary()
        breakdown = line_counter.get_breakdown()

        result = {
            "video_metadata": {
                "source_path": str(source),
                "filename": source.name,
                "fps": round(fps, 2),
                "width": w,
                "height": h,
                "total_frames": total_frames,
                "duration_sec": round(total_frames / fps, 2),
            },
            "processing_duration_sec": round(proc_duration, 2),
            "processing_fps": proc_fps,
            "frames_processed": frames_processed,
            "frame_skip": self.frame_skip,
            "vehicle_counts": {
                "total_detected_instances": total_detections_count,
                "unique_tracked": len(unique_tids_seen),
                "vehicles_in": line_counter.in_count,
                "vehicles_out": line_counter.out_count,
                "total_crossings": line_counter.total_count,
                "class_distribution": {
                    cls: breakdown["in"].get(cls, 0) + breakdown["out"].get(cls, 0)
                    for cls in set(list(breakdown["in"].keys()) + list(breakdown["out"].keys()))
                },
                "directional_breakdown": breakdown,
            },
            "speed_statistics": {
                "avg_speed_kmh": speed_summary["avg_speed_kmh"],
                "max_speed_kmh": speed_summary["max_speed_kmh"],
                "is_calibrated": transformer.is_calibrated,
                "calibration_status": transformer.status,
                "calibration_provenance": transformer.provenance,
                "speed_warning": calib_warning,
                "outliers_detected": speed_summary.get("outliers_detected", 0),
                "per_track_max_speed": {
                    tid: round(speed_estimator.get_max_speed(tid), 1)
                    for tid in unique_tids_seen
                    if speed_estimator.get_max_speed(tid) > 0
                },
                "per_track_avg_speed": {
                    tid: round(speed_estimator.get_avg_speed(tid), 1)
                    for tid in unique_tids_seen
                    if speed_estimator.get_avg_speed(tid) > 0
                },
            },
            "congestion_statistics": congestion_summary,
            "congestion_timeline": [
                {
                    "frame_idx": s.frame_idx,
                    "timestamp_sec": s.timestamp_sec,
                    "active_vehicles": s.active_vehicles,
                    "vehicles_in_roi": s.vehicles_in_roi,
                    "avg_speed_kmh": s.avg_speed_kmh,
                    "flow_rate_vpm": s.flow_rate_vpm,
                    "density_veh_per_sq_meter": s.density_veh_per_sq_meter,
                    "occupancy_ratio": s.occupancy_ratio,
                    "congestion_score": s.congestion_score,
                    "congestion_level": s.congestion_level,
                    "buildup_trend": s.buildup_trend,
                }
                for s in congestion_analyzer.snapshots
            ],
            "incidents": [e.to_dict() for e in incident_detector.get_events()],
            "incident_summary": incident_summary,
            "output_video_path": str(output_path) if output_path else None,
        }

        return result


def run(
    source_path: str,
    output_path: str = "outputs/annotated.mp4",
    weights: str = "yolo11n.pt",
    conf: float = 0.35,
    calibration_path: Optional[str] = None,
    frame_skip: int = 1,
) -> Dict:
    pipeline = TrafficPipeline(
        weights=weights,
        conf=conf,
        calibration_path=calibration_path,
        frame_skip=frame_skip,
    )
    result = pipeline.process_video(source_path, output_path=output_path)
    rep = ReportGenerator(result)
    rep.save_json(Path(output_path).with_suffix(".json"))
    rep.save_incident_csv(Path(output_path).with_suffix(".csv"))
    print(f"Completed processing {source_path}.")
    print(f"Results saved to {output_path}, {Path(output_path).with_suffix('.json')}, {Path(output_path).with_suffix('.csv')}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficVision AI Traffic Monitoring Pipeline")
    parser.add_argument("--source", required=True, help="Path to input video (mp4, avi, mov)")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="Path to save annotated video")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO model checkpoint")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--calibration", default=None, help="Path to calibration.json")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    run(
        source_path=args.source,
        output_path=args.output,
        weights=args.weights,
        conf=args.conf,
        calibration_path=args.calibration,
        frame_skip=args.frame_skip,
    )
