"""
Traffic density and congestion analysis module.

Calculates:
  - Active vehicle count
  - Spatial vehicle density within a road Region of Interest (ROI)
  - Occupancy ratio (% of ROI road area occupied by vehicles)
  - Vehicle flow rate (vehicles per minute)
  - Average vehicle speed
  - Traffic buildup index (acceleration/deceleration of queue formation)
  - Multi-signal transparent rule-based congestion classification:
      LOW, MODERATE, HIGH, SEVERE

Rule-based logic combines density, speed, stopped ratio, and flow rate.
Camera position and road geometry can be configured via Road ROI and reference speed.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv


@dataclass
class CongestionSnapshot:
    frame_idx: int
    timestamp_sec: float
    active_vehicles: int
    vehicles_in_roi: int
    density_per_sq_meter: float
    occupancy_ratio: float
    avg_speed_kmh: float
    flow_rate_vpm: float
    buildup_trend: str  # "CLEARING", "STABLE", "BUILDING", "CRITICAL"
    congestion_level: str  # "LOW", "MODERATE", "HIGH", "SEVERE"
    congestion_score: float  # 0.0 to 1.0
    reason: str


class TrafficDensityAnalyzer:
    def __init__(
        self,
        fps: float = 25.0,
        road_roi: Optional[np.ndarray] = None,
        free_flow_speed_kmh: float = 60.0,
        roi_area_sq_meters: float = 400.0,
        flow_window_sec: float = 5.0,
        trend_window_sec: float = 4.0,
    ):
        """
        Args:
            fps: Video frames per second
            road_roi: (N, 2) integer or float array of road polygon vertices.
                      If None, entire frame is considered.
            free_flow_speed_kmh: Nominal speed limit or expected free-flow speed.
            roi_area_sq_meters: Calibrated or estimated ground area of the ROI (m²).
            flow_window_sec: Duration in seconds to compute flow rate.
            trend_window_sec: Duration in seconds to detect buildup trend.
        """
        self.fps = max(1.0, float(fps))
        self.road_roi = np.asarray(road_roi, dtype=np.int32) if road_roi is not None else None
        self.free_flow_speed_kmh = max(10.0, float(free_flow_speed_kmh))
        self.roi_area_sq_meters = max(10.0, float(roi_area_sq_meters))

        self.flow_window_frames = max(5, int(flow_window_sec * self.fps))
        self.trend_window_frames = max(5, int(trend_window_sec * self.fps))

        # Recent counts for flow and trend analysis
        # (frame_idx, in_roi_count, avg_speed)
        self.history: deque = deque(maxlen=self.flow_window_frames)
        self.snapshots: List[CongestionSnapshot] = []

    def set_roi(self, polygon: np.ndarray, area_sq_m: Optional[float] = None) -> None:
        """Dynamically update road ROI polygon."""
        self.road_roi = np.asarray(polygon, dtype=np.int32)
        if area_sq_m is not None:
            self.roi_area_sq_meters = max(10.0, float(area_sq_m))

    def analyze_frame(
        self,
        detections: sv.Detections,
        speeds: Dict[int, float],
        frame_idx: int,
        frame_shape: Optional[Tuple[int, int]] = None,
        line_crossing_total: int = 0,
    ) -> CongestionSnapshot:
        """
        Analyzes vehicle density and classifies congestion for the current frame.
        """
        timestamp_sec = frame_idx / self.fps
        num_active = len(detections) if detections is not None else 0

        # Determine which vehicles fall inside Road ROI
        in_roi_indices = []
        roi_boxes = []
        if detections is not None and num_active > 0:
            if self.road_roi is not None:
                # Use bottom-center anchor as vehicle ground contact point
                anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                for idx, pt in enumerate(anchors):
                    # pointPolygonTest returns >= 0 if inside or on edge
                    inside = cv2.pointPolygonTest(self.road_roi, (float(pt[0]), float(pt[1])), False) >= 0
                    if inside:
                        in_roi_indices.append(idx)
                        roi_boxes.append(detections.xyxy[idx])
            else:
                in_roi_indices = list(range(num_active))
                roi_boxes = list(detections.xyxy)

        vehicles_in_roi = len(in_roi_indices)

        # Compute occupancy ratio
        # Sum of vehicle box areas within ROI relative to ROI bounding box or frame area
        if frame_shape and self.road_roi is not None:
            roi_area_px = max(1.0, float(cv2.contourArea(self.road_roi)))
        elif frame_shape:
            roi_area_px = max(1.0, float(frame_shape[0] * frame_shape[1]))
        else:
            roi_area_px = 640.0 * 480.0

        vehicle_box_area_px = 0.0
        for box in roi_boxes:
            w = max(0.0, box[2] - box[0])
            h = max(0.0, box[3] - box[1])
            vehicle_box_area_px += (w * h)

        occupancy_ratio = min(1.0, vehicle_box_area_px / roi_area_px)
        density_per_sq_meter = vehicles_in_roi / self.roi_area_sq_meters

        # Calculate current average speed
        roi_speeds = []
        stopped_vehicles = 0
        if detections is not None and detections.tracker_id is not None:
            for idx in in_roi_indices:
                tid = int(detections.tracker_id[idx])
                if tid in speeds:
                    s = speeds[tid]
                    roi_speeds.append(s)
                    if s < 5.0:  # near stationary
                        stopped_vehicles += 1

        if roi_speeds:
            current_avg_speed = float(np.mean(roi_speeds))
        else:
            # If no active speeds measured yet, use free flow speed when empty, or estimate
            current_avg_speed = self.free_flow_speed_kmh if vehicles_in_roi == 0 else (self.free_flow_speed_kmh * 0.5)

        # Flow rate estimation (vehicles passing per minute)
        self.history.append((frame_idx, vehicles_in_roi, current_avg_speed, line_crossing_total))

        if len(self.history) >= 2:
            first = self.history[0]
            last = self.history[-1]
            dt_sec = max(0.1, (last[0] - first[0]) / self.fps)
            delta_crossed = max(0, last[3] - first[3])
            flow_rate_vpm = (delta_crossed / dt_sec) * 60.0
        else:
            flow_rate_vpm = 0.0

        # Traffic buildup trend
        if len(self.history) >= self.trend_window_frames:
            half = len(self.history) // 2
            past_counts = [h[1] for h in list(self.history)[:half]]
            recent_counts = [h[1] for h in list(self.history)[half:]]
            count_change = np.mean(recent_counts) - np.mean(past_counts)

            if count_change > 2.0:
                buildup_trend = "CRITICAL" if vehicles_in_roi > 6 else "BUILDING"
            elif count_change < -2.0:
                buildup_trend = "CLEARING"
            else:
                buildup_trend = "STABLE"
        else:
            buildup_trend = "STABLE"

        # Multi-signal Congestion Index calculation:
        # 1. Speed factor: slow speed -> high congestion score
        speed_factor = max(0.0, min(1.0, 1.0 - (current_avg_speed / self.free_flow_speed_kmh)))

        # 2. Density factor: high count in ROI -> high congestion score
        # Assume 8-12 vehicles in standard camera view is dense
        density_factor = min(1.0, vehicles_in_roi / 10.0)

        # 3. Stopped ratio
        stopped_ratio = (stopped_vehicles / max(1, vehicles_in_roi)) if vehicles_in_roi > 0 else 0.0

        # 4. Occupancy factor
        occupancy_factor = min(1.0, occupancy_ratio * 2.5)

        # Weighted combination of independent signals
        congestion_score = (
            0.40 * speed_factor +
            0.30 * density_factor +
            0.20 * stopped_ratio +
            0.10 * occupancy_factor
        )
        congestion_score = float(np.clip(congestion_score, 0.0, 1.0))

        # Congestion classification
        if vehicles_in_roi == 0:
            level = "LOW"
            reason = "Clear roadway: zero active vehicles in target ROI."
        elif congestion_score < 0.30:
            level = "LOW"
            reason = f"Free flowing: avg speed {current_avg_speed:.1f} km/h, light density ({vehicles_in_roi} vehicles)."
        elif congestion_score < 0.55:
            level = "MODERATE"
            reason = f"Moderate flow: avg speed {current_avg_speed:.1f} km/h, steady traffic ({vehicles_in_roi} vehicles)."
        elif congestion_score < 0.75:
            level = "HIGH"
            reason = f"High congestion: reduced speed {current_avg_speed:.1f} km/h, high density ({vehicles_in_roi} vehicles, {stopped_vehicles} stopped)."
        else:
            level = "SEVERE"
            reason = f"Severe congestion: crawling traffic ({current_avg_speed:.1f} km/h), queue forming ({stopped_vehicles} stopped, {vehicles_in_roi} vehicles in ROI)."

        snapshot = CongestionSnapshot(
            frame_idx=frame_idx,
            timestamp_sec=round(timestamp_sec, 2),
            active_vehicles=num_active,
            vehicles_in_roi=vehicles_in_roi,
            density_per_sq_meter=round(density_per_sq_meter, 4),
            occupancy_ratio=round(occupancy_ratio, 3),
            avg_speed_kmh=round(current_avg_speed, 1),
            flow_rate_vpm=round(flow_rate_vpm, 1),
            buildup_trend=buildup_trend,
            congestion_level=level,
            congestion_score=round(congestion_score, 3),
            reason=reason,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_summary(self) -> Dict:
        """Returns aggregate congestion metrics over the entire video."""
        if not self.snapshots:
            return {
                "average_congestion_score": 0.0,
                "dominant_level": "LOW",
                "peak_vehicles_in_roi": 0,
                "average_speed_kmh": 0.0,
                "congestion_level_distribution": {"LOW": 100.0, "MODERATE": 0.0, "HIGH": 0.0, "SEVERE": 0.0},
            }

        scores = [s.congestion_score for s in self.snapshots]
        speeds = [s.avg_speed_kmh for s in self.snapshots]
        counts = [s.vehicles_in_roi for s in self.snapshots]
        levels = [s.congestion_level for s in self.snapshots]

        level_counts = {lvl: levels.count(lvl) for lvl in ["LOW", "MODERATE", "HIGH", "SEVERE"]}
        total = max(1, len(levels))
        dist = {lvl: round((cnt / total) * 100.0, 1) for lvl, cnt in level_counts.items()}
        dominant = max(dist.items(), key=lambda x: x[1])[0]

        return {
            "average_congestion_score": round(float(np.mean(scores)), 3),
            "dominant_level": dominant,
            "peak_vehicles_in_roi": int(np.max(counts)),
            "average_speed_kmh": round(float(np.mean(speeds)), 1),
            "congestion_level_distribution": dist,
        }
