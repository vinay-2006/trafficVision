"""
Multi-object tracking layer on top of raw detections.

Uses ByteTrack (via supervision) to assign a persistent ID to each
vehicle across frames, so downstream logic (counting, speed) can
reason about "this specific vehicle" rather than "a detection in frame N".
"""
import numpy as np
import supervision as sv


from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv


class VehicleTracker:
    def __init__(
        self,
        frame_rate: int = 25,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        max_history_len: int = 150,
    ):
        self.frame_rate = frame_rate
        self.max_history_len = max_history_len
        self.tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        # tracker_id -> deque of (frame_idx, center_x, center_y, bottom_x, bottom_y)
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_history_len))
        self.class_history: Dict[int, int] = {}

    def update(self, detections: sv.Detections, frame_idx: Optional[int] = None) -> sv.Detections:
        """Feed one frame's detections in, get back detections with
        a stable `tracker_id` field populated for each box."""
        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for i, tracker_id in enumerate(tracked.tracker_id):
                tid = int(tracker_id)
                box = tracked.xyxy[i]
                cx = float((box[0] + box[2]) / 2.0)
                cy = float((box[1] + box[3]) / 2.0)
                bx = float(cx)
                by = float(box[3])
                fidx = frame_idx if frame_idx is not None else len(self.history[tid])
                self.history[tid].append((fidx, cx, cy, bx, by))
                if tracked.class_id is not None and i < len(tracked.class_id):
                    self.class_history[tid] = int(tracked.class_id[i])

        return tracked

    def get_track_history(self, tracker_id: int) -> List[Tuple[int, float, float, float, float]]:
        """Return list of (frame_idx, cx, cy, bx, by) for given tracker_id."""
        return list(self.history.get(tracker_id, []))

    def reset(self):
        self.tracker.reset()
        self.history.clear()
        self.class_history.clear()
