"""
Downstream analytics built on top of tracked detections:
  - per-vehicle speed estimation (using ground-plane positions over time)
  - unique vehicle counting via virtual line-crossing
  - directional traffic analysis (incoming vs outgoing)
"""
from collections import defaultdict, deque
import math
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import supervision as sv

from src.perspective import PerspectiveTransformer


class SpeedEstimator:
    def __init__(
        self,
        transformer: PerspectiveTransformer,
        fps: float,
        window_seconds: float = 1.0,
        max_speed_kmh: float = 200.0,
        ema_alpha: float = 0.4,
        max_gap_seconds: float = 1.5,
    ):
        """
        Args:
            transformer: maps pixel -> real-world meter coordinates.
            fps: video frame rate.
            window_seconds: duration of position history window.
            max_speed_kmh: threshold above which speeds are marked as outliers.
            ema_alpha: smoothing factor for Exponential Moving Average.
            max_gap_seconds: max time gap before considering a track discontinuous.
        """
        self.transformer = transformer
        self.fps = float(fps) if fps > 0 else 25.0
        self.is_calibrated = getattr(transformer, "is_calibrated", False)
        self.calibration_status = getattr(transformer, "status", "illustrative")
        self.window_size = max(2, int(window_seconds * self.fps))
        self.max_speed_kmh = float(max_speed_kmh)
        self.ema_alpha = float(ema_alpha)
        self.max_gap_frames = max(2, int(max_gap_seconds * self.fps))

        # tracker_id -> deque of (frame_idx, x_m, y_m)
        self.history: Dict[int, Deque[Tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        # tracker_id -> last observed frame index (for discontinuity detection)
        self.last_seen_frame: Dict[int, int] = {}

        # tracker_id -> latest smoothed reliable speed km/h
        self.latest_speeds: Dict[int, float] = {}
        # tracker_id -> list of reliable (frame_idx, speed_kmh)
        self.speed_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # tracker_id -> max reliable observed speed
        self.max_speeds: Dict[int, float] = {}

        # Raw diagnostic measurements (for audit, inspection, debugging)
        # tracker_id -> list of dicts with raw measurements and outlier flags
        self.raw_diagnostics: Dict[int, List[Dict]] = defaultdict(list)
        self.outliers_count = 0

    @property
    def outlier_count(self) -> int:
        return self.outliers_count

    def update(self, detections: sv.Detections, frame_idx: int) -> Dict[int, float]:
        """Feed current-frame tracked detections, get back
        {tracker_id: speed_kmh} for tracks with valid, non-outlier history."""
        if detections is None or detections.tracker_id is None or len(detections.tracker_id) == 0:
            return {}

        anchor_points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        ground_points = self.transformer.transform_points(anchor_points)

        current_frame_speeds: Dict[int, float] = {}
        for tracker_id, (x_m, y_m) in zip(detections.tracker_id, ground_points):
            tid = int(tracker_id)
            xm, ym = float(x_m), float(y_m)

            # Check track discontinuity: if track reappears after a large frame gap,
            # reset history so we do not measure false jumps across missing periods
            if tid in self.last_seen_frame:
                gap = frame_idx - self.last_seen_frame[tid]
                if gap > self.max_gap_frames:
                    self.history[tid].clear()

            self.last_seen_frame[tid] = frame_idx
            self.history[tid].append((frame_idx, xm, ym))
            hist = self.history[tid]

            if len(hist) >= 2:
                (f0, x0, y0), (f1, x1, y1) = hist[0], hist[-1]
                elapsed_frames = f1 - f0
                dt = elapsed_frames / self.fps

                if dt > 0:
                    dist_m = float(np.hypot(x1 - x0, y1 - y0))
                    raw_speed_mps = dist_m / dt
                    raw_speed_kmh = raw_speed_mps * 3.6

                    is_outlier = False
                    outlier_reason = None

                    # Check finite numbers
                    if not math.isfinite(raw_speed_kmh) or raw_speed_kmh < 0.0:
                        is_outlier = True
                        outlier_reason = "non_finite_or_negative"
                    elif raw_speed_kmh > self.max_speed_kmh:
                        is_outlier = True
                        outlier_reason = f"exceeds_plausible_max_{self.max_speed_kmh}kmh"

                    # Record raw diagnostic measurement
                    diag = {
                        "frame_idx": frame_idx,
                        "dt_sec": round(dt, 4),
                        "dist_m": round(dist_m, 3),
                        "raw_kmh": round(raw_speed_kmh, 2) if math.isfinite(raw_speed_kmh) else None,
                        "is_outlier": is_outlier,
                        "outlier_reason": outlier_reason,
                    }
                    self.raw_diagnostics[tid].append(diag)

                    if is_outlier:
                        self.outliers_count += 1
                        # DO NOT clamp silently into reliable statistics!
                        # Outlier speeds are excluded from reliable summary stats.
                        continue

                    # Apply EMA smoothing on valid, physically plausible measurements
                    if tid in self.latest_speeds:
                        prev = self.latest_speeds[tid]
                        smoothed = self.ema_alpha * raw_speed_kmh + (1.0 - self.ema_alpha) * prev
                    else:
                        smoothed = raw_speed_kmh

                    self.latest_speeds[tid] = smoothed
                    self.speed_history[tid].append((frame_idx, smoothed))
                    self.max_speeds[tid] = max(self.max_speeds.get(tid, 0.0), smoothed)
                    current_frame_speeds[tid] = smoothed

        return current_frame_speeds

    def get_speed(self, tracker_id: int) -> Optional[float]:
        return self.latest_speeds.get(tracker_id)

    def get_max_speed(self, tracker_id: int) -> float:
        return self.max_speeds.get(tracker_id, 0.0)

    def get_avg_speed(self, tracker_id: int) -> float:
        records = self.speed_history.get(tracker_id, [])
        if not records:
            return 0.0
        return float(np.mean([s for _, s in records]))

    def get_raw_diagnostics(self, tracker_id: int) -> List[Dict]:
        return list(self.raw_diagnostics.get(tracker_id, []))

    def get_summary_stats(self) -> Dict:
        """Global speed summary across all recorded reliable tracks."""
        reliable_speeds = list(self.max_speeds.values())
        if not reliable_speeds:
            return {
                "avg_speed_kmh": 0.0,
                "max_speed_kmh": 0.0,
                "active_speed_count": 0,
                "is_calibrated": self.is_calibrated,
                "calibration_status": self.calibration_status,
                "outliers_detected": self.outliers_count,
            }

        return {
            "avg_speed_kmh": round(float(np.mean(reliable_speeds)), 1),
            "max_speed_kmh": round(float(np.max(reliable_speeds)), 1),
            "active_speed_count": len(reliable_speeds),
            "is_calibrated": self.is_calibrated,
            "calibration_status": self.calibration_status,
            "outliers_detected": self.outliers_count,
        }


class LineCounter:
    """Counts unique tracker IDs crossing a virtual line, split by direction."""

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start = start
        self.end = end
        self.line_zone = sv.LineZone(start=sv.Point(*start), end=sv.Point(*end))
        self.crossed_in_ids: Set[int] = set()
        self.crossed_out_ids: Set[int] = set()
        self.crossing_events: List[Dict] = []
        self.class_counts_in: Dict[str, int] = defaultdict(int)
        self.class_counts_out: Dict[str, int] = defaultdict(int)

    def update(
        self,
        detections: sv.Detections,
        class_name_getter=None,
        frame_idx: int = 0,
        speeds: Optional[Dict[int, float]] = None,
    ) -> Tuple[int, int]:
        """
        Updates line zone and returns (count_in, count_out).
        Guarantees each tracker ID is counted uniquely per direction.
        """
        if detections is None or detections.tracker_id is None or len(detections.tracker_id) == 0:
            return self.in_count, self.out_count

        crossed_in_mask, crossed_out_mask = self.line_zone.trigger(detections)

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            cid = int(detections.class_id[i]) if detections.class_id is not None else 0
            cname = class_name_getter(cid) if class_name_getter else str(cid)
            speed = speeds.get(tid, None) if speeds else None

            if crossed_in_mask[i] and tid not in self.crossed_in_ids:
                self.crossed_in_ids.add(tid)
                self.class_counts_in[cname] += 1
                self.crossing_events.append({
                    "frame_idx": frame_idx,
                    "tracker_id": tid,
                    "direction": "in",
                    "class_name": cname,
                    "speed_kmh": speed,
                })

            if crossed_out_mask[i] and tid not in self.crossed_out_ids:
                self.crossed_out_ids.add(tid)
                self.class_counts_out[cname] += 1
                self.crossing_events.append({
                    "frame_idx": frame_idx,
                    "tracker_id": tid,
                    "direction": "out",
                    "class_name": cname,
                    "speed_kmh": speed,
                })

        return self.in_count, self.out_count

    @property
    def in_count(self) -> int:
        return len(self.crossed_in_ids) if self.crossed_in_ids else self.line_zone.in_count

    @property
    def out_count(self) -> int:
        return len(self.crossed_out_ids) if self.crossed_out_ids else self.line_zone.out_count

    @property
    def total_count(self) -> int:
        return self.in_count + self.out_count

    def get_breakdown(self) -> Dict[str, Dict[str, int]]:
        return {
            "in": dict(self.class_counts_in),
            "out": dict(self.class_counts_out),
            "total_in": self.in_count,
            "total_out": self.out_count,
            "total_vehicles": self.total_count,
        }
