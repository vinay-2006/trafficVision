"""
Downstream analytics built on top of tracked detections:
  - per-vehicle speed estimation (using ground-plane positions over time)
  - unique vehicle counting via virtual line-crossing
  - directional traffic analysis (incoming vs outgoing)
"""
from collections import defaultdict, deque
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
        max_speed_kmh: float = 240.0,
        ema_alpha: float = 0.4,
    ):
        """
        Args:
            transformer: maps pixel -> real-world meter coordinates
            fps: video frame rate
            window_seconds: duration of position history window
            max_speed_kmh: ceiling for physically plausible speed on highway/city
            ema_alpha: smoothing factor for Exponential Moving Average
        """
        self.transformer = transformer
        self.fps = float(fps) if fps > 0 else 25.0
        self.is_calibrated = getattr(transformer, "is_calibrated", True)
        self.window_size = max(2, int(window_seconds * self.fps))
        self.max_speed_kmh = max_speed_kmh
        self.ema_alpha = ema_alpha

        # tracker_id -> deque of (frame_idx, x_m, y_m)
        self.history: Dict[int, Deque[Tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        # tracker_id -> latest smoothed speed km/h
        self.latest_speeds: Dict[int, float] = {}
        # tracker_id -> list of (frame_idx, speed_kmh)
        self.speed_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # tracker_id -> max observed speed
        self.max_speeds: Dict[int, float] = {}

    def update(self, detections: sv.Detections, frame_idx: int) -> Dict[int, float]:
        """Feed current-frame tracked detections, get back
        {tracker_id: speed_kmh} for tracks with sufficient history."""
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            return {}

        anchor_points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        ground_points = self.transformer.transform_points(anchor_points)

        current_frame_speeds: Dict[int, float] = {}
        for tracker_id, (x_m, y_m) in zip(detections.tracker_id, ground_points):
            tid = int(tracker_id)
            self.history[tid].append((frame_idx, float(x_m), float(y_m)))
            hist = self.history[tid]

            if len(hist) >= 2:
                (f0, x0, y0), (f1, x1, y1) = hist[0], hist[-1]
                dt = (f1 - f0) / self.fps
                if dt > 0:
                    dist_m = float(np.hypot(x1 - x0, y1 - y0))
                    raw_speed_mps = dist_m / dt
                    raw_speed_kmh = raw_speed_mps * 3.6

                    # Clamp to realistic physical range
                    clamped_kmh = min(raw_speed_kmh, self.max_speed_kmh)

                    # Smooth using EMA
                    if tid in self.latest_speeds:
                        prev = self.latest_speeds[tid]
                        smoothed = self.ema_alpha * clamped_kmh + (1.0 - self.ema_alpha) * prev
                    else:
                        smoothed = clamped_kmh

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

    def get_summary_stats(self) -> Dict[str, float]:
        """Global speed summary across all recorded tracks."""
        if not self.max_speeds:
            return {
                "avg_speed_kmh": 0.0,
                "max_speed_kmh": 0.0,
                "active_speed_count": 0,
                "is_calibrated": self.is_calibrated,
            }
        all_speeds = list(self.max_speeds.values())
        return {
            "avg_speed_kmh": round(float(np.mean(all_speeds)), 1),
            "max_speed_kmh": round(float(np.max(all_speeds)), 1),
            "active_speed_count": len(all_speeds),
            "is_calibrated": self.is_calibrated,
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
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
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
