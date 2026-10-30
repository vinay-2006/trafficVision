"""
Explainable rule-based incident detection module for traffic monitoring.

Detects potential anomalous vehicular events without neural network retraining:
  1. Prolonged stationary vehicle in active roadway
  2. Sudden abnormal deceleration
  3. Abnormal trajectory / wrong-way direction movement
  4. Potential road blockage / congestion bottleneck
  5. Possible collision indicator based on suspicious proximity & deceleration

Guarantees:
  - Deduplication via cooldown windows per vehicle & incident type
  - Temporal confirmation (requires persistent anomaly over multiple frames)
  - Clear human-verification disclaimers (potential alerts, never confirmed accidents)
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import supervision as sv


@dataclass
class IncidentEvent:
    event_id: str
    event_type: str
    title: str
    vehicle_ids: List[int]
    timestamp_sec: float
    frame_idx: int
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description: str
    evidence: str
    confidence: float
    human_verification_warning: str = (
        "Automated rule-based indicator for operator review. "
        "Does NOT constitute a confirmed collision or accident."
    )

    def to_dict(self) -> Dict:
        return asdict(self)


class IncidentDetector:
    def __init__(
        self,
        fps: float = 25.0,
        stationary_time_threshold_sec: float = 3.0,
        stationary_dist_max_px: float = 25.0,
        abnormal_decel_threshold_kmh: float = 25.0,
        abnormal_decel_window_sec: float = 1.5,
        proximity_threshold_px: float = 40.0,
        cooldown_sec: float = 12.0,
        temporal_confirm_frames: int = 4,
        dominant_direction_y: int = 1,  # 1 for top-to-bottom, -1 for bottom-to-top
    ):
        """
        Args:
            fps: Video frame rate.
            stationary_time_threshold_sec: Duration a vehicle must stay stationary to trigger alert.
            stationary_dist_max_px: Max pixel movement allowed during stationary state.
            abnormal_decel_threshold_kmh: Speed drop magnitude considered abnormal.
            abnormal_decel_window_sec: Window in seconds for sudden deceleration.
            proximity_threshold_px: Center-to-center distance threshold for suspicious proximity.
            cooldown_sec: Minimum seconds before a duplicate alert of same type/vehicle can fire.
            temporal_confirm_frames: Consecutive frames anomaly must be seen before emitting event.
            dominant_direction_y: Expected direction of travel along Y axis (1 or -1).
        """
        self.fps = max(1.0, float(fps))
        self.stationary_time_threshold_sec = float(stationary_time_threshold_sec)
        self.stationary_dist_max_px = float(stationary_dist_max_px)
        self.abnormal_decel_threshold_kmh = float(abnormal_decel_threshold_kmh)
        self.abnormal_decel_window_sec = float(abnormal_decel_window_sec)
        self.proximity_threshold_px = float(proximity_threshold_px)
        self.cooldown_sec = float(cooldown_sec)
        self.temporal_confirm_frames = max(1, int(temporal_confirm_frames))
        self.dominant_direction_y = dominant_direction_y

        self.incident_counter = 0
        self.events: List[IncidentEvent] = []

        # (event_type, vehicle_id_tuple) -> last_triggered_frame
        self.last_alert_frame: Dict[Tuple[str, Tuple[int, ...]], int] = {}

        # Tracking state counters for temporal confirmation
        # (event_type, vehicle_id) -> consecutive_frames_observed
        self.anomaly_streak: Dict[Tuple[str, int], int] = {}

    def _is_on_cooldown(self, event_type: str, vehicle_ids: List[int], frame_idx: int) -> bool:
        key = (event_type, tuple(sorted(vehicle_ids)))
        if key in self.last_alert_frame:
            last_frame = self.last_alert_frame[key]
            elapsed_sec = (frame_idx - last_frame) / self.fps
            if elapsed_sec < self.cooldown_sec:
                return True
        return False

    def _record_alert(self, event: IncidentEvent) -> None:
        key = (event.event_type, tuple(sorted(event.vehicle_ids)))
        self.last_alert_frame[key] = event.frame_idx
        self.events.append(event)

    def process_frame(
        self,
        detections: sv.Detections,
        speeds: Dict[int, float],
        frame_idx: int,
        tracker_history: Dict[int, List[Tuple[int, float, float, float, float]]],
        congestion_level: str = "LOW",
    ) -> List[IncidentEvent]:
        """
        Evaluates the current frame for all 5 categories of potential incidents.
        Returns newly triggered events in this frame.
        """
        if detections is None or detections.tracker_id is None or len(detections.tracker_id) == 0:
            return []

        timestamp_sec = round(frame_idx / self.fps, 2)
        new_events: List[IncidentEvent] = []
        active_tids = [int(tid) for tid in detections.tracker_id]

        # ---------------------------------------------------------------------
        # 1. Prolonged Stationary Vehicle in Roadway
        # ---------------------------------------------------------------------
        min_stationary_frames = int(self.stationary_time_threshold_sec * self.fps)
        stationary_tids = []

        for tid in active_tids:
            hist = list(tracker_history.get(tid, []))
            if len(hist) >= min_stationary_frames:
                recent_hist = hist[-min_stationary_frames:]
                first_pt = np.array([recent_hist[0][1], recent_hist[0][2]])
                last_pt = np.array([recent_hist[-1][1], recent_hist[-1][2]])
                displacement = float(np.linalg.norm(last_pt - first_pt))

                # Check if current speed is also near 0 (if available)
                curr_spd = speeds.get(tid, 0.0)

                if displacement <= self.stationary_dist_max_px and curr_spd < 6.0:
                    stationary_tids.append(tid)
                    streak_key = ("stationary", tid)
                    self.anomaly_streak[streak_key] = self.anomaly_streak.get(streak_key, 0) + 1

                    if self.anomaly_streak[streak_key] >= self.temporal_confirm_frames:
                        if not self._is_on_cooldown("suspicious_stationary_vehicle", [tid], frame_idx):
                            self.incident_counter += 1
                            duration = len(recent_hist) / self.fps
                            ev = IncidentEvent(
                                event_id=f"INC-{self.incident_counter:03d}",
                                event_type="suspicious_stationary_vehicle",
                                title="Suspicious Stationary Vehicle",
                                vehicle_ids=[tid],
                                timestamp_sec=timestamp_sec,
                                frame_idx=frame_idx,
                                severity="HIGH",
                                description=(
                                    f"Vehicle #{tid} has been stationary in the roadway "
                                    f"for {duration:.1f} seconds."
                                ),
                                evidence=(
                                    f"Displacement: {displacement:.1f}px over {duration:.1f}s "
                                    f"(Speed: {curr_spd:.1f} km/h)."
                                ),
                                confidence=0.88,
                            )
                            self._record_alert(ev)
                            new_events.append(ev)
                else:
                    self.anomaly_streak[("stationary", tid)] = 0
            else:
                self.anomaly_streak[("stationary", tid)] = 0

        # ---------------------------------------------------------------------
        # 2. Sudden Abnormal Deceleration
        # ---------------------------------------------------------------------
        decel_window_frames = max(2, int(self.abnormal_decel_window_sec * self.fps))
        for tid in active_tids:
            hist = list(tracker_history.get(tid, []))
            if len(hist) >= decel_window_frames:
                # Estimate speed drop from history if speeds are tracked over time
                # Or from spatial step progression
                t_window = hist[-decel_window_frames:]
                # Displacement in early half vs later half of window
                half = len(t_window) // 2
                disp_early = np.linalg.norm(
                    np.array([t_window[half][1] - t_window[0][1], t_window[half][2] - t_window[0][2]])
                )
                disp_late = np.linalg.norm(
                    np.array([t_window[-1][1] - t_window[half][1], t_window[-1][2] - t_window[half][2]])
                )

                dt_half = (half / self.fps)
                px_speed_early = disp_early / dt_half if dt_half > 0 else 0
                px_speed_late = disp_late / dt_half if dt_half > 0 else 0

                # Drop ratio
                if px_speed_early > 40.0 and (px_speed_early - px_speed_late) > 30.0:
                    streak_key = ("sudden_deceleration", tid)
                    self.anomaly_streak[streak_key] = self.anomaly_streak.get(streak_key, 0) + 1

                    if self.anomaly_streak[streak_key] >= 2:
                        if not self._is_on_cooldown("sudden_abnormal_deceleration", [tid], frame_idx):
                            self.incident_counter += 1
                            curr_spd = speeds.get(tid, 0.0)
                            ev = IncidentEvent(
                                event_id=f"INC-{self.incident_counter:03d}",
                                event_type="sudden_abnormal_deceleration",
                                title="Sudden Abnormal Deceleration",
                                vehicle_ids=[tid],
                                timestamp_sec=timestamp_sec,
                                frame_idx=frame_idx,
                                severity="MEDIUM",
                                description=(
                                    f"Vehicle #{tid} underwent rapid, unexpected deceleration."
                                ),
                                evidence=(
                                    f"Estimated velocity drop: -{(px_speed_early - px_speed_late):.1f} px/s "
                                    f"within {self.abnormal_decel_window_sec:.1f}s window."
                                ),
                                confidence=0.82,
                            )
                            self._record_alert(ev)
                            new_events.append(ev)
                else:
                    self.anomaly_streak[("sudden_deceleration", tid)] = 0

        # ---------------------------------------------------------------------
        # 3. Abnormal Trajectory or Direction Change
        # ---------------------------------------------------------------------
        direction_window_frames = max(4, min(12, int(0.5 * self.fps)))
        for tid in active_tids:
            hist = list(tracker_history.get(tid, []))
            if len(hist) >= direction_window_frames:
                recent = hist[-direction_window_frames:]
                dy = recent[-1][2] - recent[0][2]
                dx = recent[-1][1] - recent[0][1]
                net_disp = np.hypot(dx, dy)

                if net_disp > 20.0:
                    # Check if movement opposes dominant flow
                    opposing = (self.dominant_direction_y > 0 and dy < -15.0) or \
                               (self.dominant_direction_y < 0 and dy > 15.0)
                    # Check for extreme lateral drift
                    lateral_drift = abs(dx) > (abs(dy) * 2.2) and abs(dx) > 35.0

                    if opposing or lateral_drift:
                        streak_key = ("abnormal_trajectory", tid)
                        self.anomaly_streak[streak_key] = self.anomaly_streak.get(streak_key, 0) + 1

                        if self.anomaly_streak[streak_key] >= self.temporal_confirm_frames:
                            if not self._is_on_cooldown("abnormal_trajectory", [tid], frame_idx):
                                self.incident_counter += 1
                                desc = "Vehicle traveling against normal traffic flow direction" if opposing else "Vehicle exhibiting abrupt erratic lateral trajectory"
                                ev = IncidentEvent(
                                    event_id=f"INC-{self.incident_counter:03d}",
                                    event_type="abnormal_trajectory",
                                    title="Abnormal Vehicle Trajectory",
                                    vehicle_ids=[tid],
                                    timestamp_sec=timestamp_sec,
                                    frame_idx=frame_idx,
                                    severity="HIGH",
                                    description=f"Vehicle #{tid}: {desc}.",
                                    evidence=f"Vector (dx={dx:.1f}px, dy={dy:.1f}px), net displacement {net_disp:.1f}px.",
                                    confidence=0.85,
                                )
                                self._record_alert(ev)
                                new_events.append(ev)
                    else:
                        self.anomaly_streak[("abnormal_trajectory", tid)] = 0

        # ---------------------------------------------------------------------
        # 4. Potential Road Blockage
        # ---------------------------------------------------------------------
        # If 2 or more stationary vehicles or severe congestion with stopped cluster
        if len(stationary_tids) >= 2 or (congestion_level == "SEVERE" and len(stationary_tids) >= 1):
            if not self._is_on_cooldown("potential_road_blockage", stationary_tids, frame_idx):
                self.incident_counter += 1
                ev = IncidentEvent(
                    event_id=f"INC-{self.incident_counter:03d}",
                    event_type="potential_road_blockage",
                    title="Potential Road Blockage",
                    vehicle_ids=stationary_tids,
                    timestamp_sec=timestamp_sec,
                    frame_idx=frame_idx,
                    severity="HIGH" if len(stationary_tids) == 2 else "CRITICAL",
                    description=(
                        f"Detected multiple stopped vehicles ({len(stationary_tids)} units) "
                        f"indicating possible lane obstruction."
                    ),
                    evidence=(
                        f"Stationary vehicle IDs: {stationary_tids}. "
                        f"Current traffic congestion status: {congestion_level}."
                    ),
                    confidence=0.89,
                )
                self._record_alert(ev)
                new_events.append(ev)

        # ---------------------------------------------------------------------
        # 5. Possible Collision Indicator (Proximity & Abrupt Motion Anomaly)
        # ---------------------------------------------------------------------
        # Pairwise check between detected bounding boxes
        num_boxes = len(detections.xyxy)
        if num_boxes >= 2:
            centers = []
            for box in detections.xyxy:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                centers.append((cx, cy))

            for i in range(num_boxes):
                for j in range(i + 1, num_boxes):
                    id1 = active_tids[i]
                    id2 = active_tids[j]
                    dist_px = float(np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))

                    # If unusually close AND at least one was in deceleration or stationary
                    if dist_px < self.proximity_threshold_px:
                        spd1 = speeds.get(id1, 10.0)
                        spd2 = speeds.get(id2, 10.0)
                        either_slow = (spd1 < 5.0 or spd2 < 5.0)

                        if either_slow:
                            streak_key = ("possible_collision_indicator", tuple(sorted([id1, id2])))
                            self.anomaly_streak[streak_key] = self.anomaly_streak.get(streak_key, 0) + 1

                            if self.anomaly_streak[streak_key] >= 3:
                                if not self._is_on_cooldown("possible_collision_indicator", [id1, id2], frame_idx):
                                    self.incident_counter += 1
                                    ev = IncidentEvent(
                                        event_id=f"INC-{self.incident_counter:03d}",
                                        event_type="possible_collision_indicator",
                                        title="Possible Collision Indicator",
                                        vehicle_ids=[id1, id2],
                                        timestamp_sec=timestamp_sec,
                                        frame_idx=frame_idx,
                                        severity="CRITICAL",
                                        description=(
                                            f"Suspicious proximity detected between vehicle #{id1} and #{id2} "
                                            f"with immediate motion standstill."
                                        ),
                                        evidence=(
                                            f"Inter-vehicle centroid distance: {dist_px:.1f}px. "
                                            f"Observed speeds: #{id1}={spd1:.1f} km/h, #{id2}={spd2:.1f} km/h."
                                        ),
                                        confidence=0.76,
                                    )
                                    self._record_alert(ev)
                                    new_events.append(ev)

        return new_events

    def get_events(self) -> List[IncidentEvent]:
        return list(self.events)

    def get_summary(self) -> Dict:
        """Returns statistics of all detected incidents."""
        type_counts = {}
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

        for ev in self.events:
            type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1
            severity_counts[ev.severity] = severity_counts.get(ev.severity, 0) + 1

        return {
            "total_incidents": len(self.events),
            "incidents_by_type": type_counts,
            "incidents_by_severity": severity_counts,
            "disclaimer": "All logged alerts are potential rule-based indicators and require human operator confirmation.",
        }
