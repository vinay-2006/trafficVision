"""
Unit tests for IncidentDetector in src/incidents.py.

Verifies:
- Stationary vehicle detection with temporal confirmation.
- Sudden deceleration detection using configured parameters.
- Disambiguation of potential wrong-way movement vs abnormal lateral drift.
- Arbitrary 2D dominant direction vector support.
- Cooldown and temporal suppression.
- Stale anomaly streak pruning for disappeared tracks.
- Mandatory human-verification disclaimer.
"""
import numpy as np
import pytest
import supervision as sv

from src.incidents import IncidentDetector, IncidentEvent


def test_stationary_vehicle_detection_triggers_event():
    fps = 25.0
    detector = IncidentDetector(
        fps=fps,
        stationary_time_threshold_sec=1.0,
        stationary_dist_max_px=10.0,
        temporal_confirm_frames=2,
        cooldown_sec=5.0,
    )

    tid = 42
    # 30 frames with zero displacement
    history = {tid: [(f, 200.0, 300.0, 200.0, 320.0) for f in range(30)]}

    det = sv.Detections(
        xyxy=np.array([[180, 280, 220, 320]], dtype=np.float32),
        tracker_id=np.array([tid]),
        class_id=np.array([2]),
    )

    # Frame 1
    events1 = detector.process_frame(det, speeds={tid: 1.0}, frame_idx=25, tracker_history=history)
    # Frame 2 (reaches temporal_confirm_frames=2)
    events2 = detector.process_frame(det, speeds={tid: 1.0}, frame_idx=26, tracker_history=history)

    assert len(detector.get_events()) == 1
    ev = detector.get_events()[0]
    assert ev.event_type == "suspicious_stationary_vehicle"
    assert tid in ev.vehicle_ids
    assert ev.severity == "HIGH"
    assert "human_verification_warning" in ev.to_dict()


def test_cooldown_prevents_duplicate_events():
    fps = 25.0
    detector = IncidentDetector(
        fps=fps,
        stationary_time_threshold_sec=1.0,
        temporal_confirm_frames=1,
        cooldown_sec=5.0,
    )

    tid = 15
    history = {tid: [(f, 100.0, 100.0, 100.0, 110.0) for f in range(30)]}
    det = sv.Detections(
        xyxy=np.array([[90, 90, 110, 110]], dtype=np.float32),
        tracker_id=np.array([tid]),
        class_id=np.array([2]),
    )

    # Trigger first event at frame 25 (t=1.0s)
    events1 = detector.process_frame(det, speeds={tid: 0.0}, frame_idx=25, tracker_history=history)
    assert len(events1) == 1

    # Immediate next frame at frame 26 (t=1.04s, well within 5.0s cooldown)
    events2 = detector.process_frame(det, speeds={tid: 0.0}, frame_idx=26, tracker_history=history)
    assert len(events2) == 0  # blocked by cooldown

    # Total recorded alerts should still be 1
    assert len(detector.get_events()) == 1


def test_potential_wrong_way_movement_detection():
    fps = 25.0
    detector = IncidentDetector(
        fps=fps,
        temporal_confirm_frames=1,
        dominant_direction_y=1,  # normal is top to bottom (dy > 0)
    )

    tid = 99
    # Vehicle moving upwards against dominant downward flow
    history = {tid: [(f, 200.0, 400.0 - f * 8.0, 200.0, 420.0 - f * 8.0) for f in range(15)]}
    det = sv.Detections(
        xyxy=np.array([[180, 280, 220, 320]], dtype=np.float32),
        tracker_id=np.array([tid]),
        class_id=np.array([2]),
    )

    events = detector.process_frame(det, speeds={tid: 40.0}, frame_idx=15, tracker_history=history)
    assert len(events) >= 1
    assert events[0].event_type == "potential_wrong_way_movement"
    assert tid in events[0].vehicle_ids
    assert "Wrong-Way" in events[0].title


def test_abnormal_lateral_drift_detection():
    """Verify lateral perpendicular movement triggers abnormal lateral drift alert."""
    fps = 25.0
    detector = IncidentDetector(
        fps=fps,
        temporal_confirm_frames=1,
    )

    tid = 77
    # 15 frames with large horizontal displacement (+60px) and small forward progress (+10px)
    history_pts = []
    for f in range(15):
        x = 100.0 + (f / 14.0) * 60.0
        y = 100.0 + (f / 14.0) * 10.0
        history_pts.append((f, x, y, x, y + 20.0))
    history = {tid: history_pts}

    det = sv.Detections(
        xyxy=np.array([[150, 100, 170, 130]], dtype=np.float32),
        tracker_id=np.array([tid]),
        class_id=np.array([2]),
    )

    events = detector.process_frame(det, speeds={tid: 30.0}, frame_idx=15, tracker_history=history)
    drift_events = [e for e in events if e.event_type == "abnormal_lateral_drift"]
    assert len(drift_events) >= 1
    assert tid in drift_events[0].vehicle_ids
    assert "Lateral" in drift_events[0].title


def test_sudden_deceleration_uses_configured_parameters():
    """Verify sudden deceleration triggers when drop exceeds configured threshold."""
    fps = 25.0
    detector = IncidentDetector(
        fps=fps,
        abnormal_decel_threshold_kmh=20.0,
        abnormal_decel_window_sec=1.5,
        temporal_confirm_frames=1,
    )

    tid = 101
    det = sv.Detections(
        xyxy=np.array([[200, 200, 250, 250]], dtype=np.float32),
        tracker_id=np.array([tid]),
        class_id=np.array([2]),
    )
    # 40 frames total. Early 20 frames: moves from y=100 to y=200 (100px fast)
    # Late 20 frames: stays at y=200 (0px standstill)
    history_pts = []
    for f in range(20):
        history_pts.append((f, 200.0, 100.0 + f * 5.0, 200.0, 120.0 + f * 5.0))
    for f in range(20, 40):
        history_pts.append((f, 200.0, 200.0, 200.0, 220.0))
    history = {tid: history_pts}

    events = detector.process_frame(det, speeds={tid: 5.0}, frame_idx=40, tracker_history=history)
    decel_events = [e for e in events if e.event_type == "sudden_abnormal_deceleration"]
    assert len(decel_events) >= 1
    assert tid in decel_events[0].vehicle_ids
    assert decel_events[0].severity in ("MEDIUM", "HIGH")


def test_dominant_direction_2d_arbitrary_vector():
    """Verify 2D vector for arbitrary camera angle (e.g. horizontal left-to-right traffic)."""
    fps = 25.0
    # Road moves along +X axis (dx > 0, dy = 0)
    detector = IncidentDetector(
        fps=fps,
        dominant_direction=(1.0, 0.0),
        temporal_confirm_frames=1,
    )

    tid = 55
    # Vehicle moving leftward (-X axis: dx < 0)
    history = {tid: [(f, 500.0 - f * 10.0, 200.0, 500.0 - f * 10.0, 220.0) for f in range(15)]}
    det = sv.Detections(
        xyxy=np.array([[340, 190, 380, 230]], dtype=np.float32),
        tracker_id=np.array([tid]),
        class_id=np.array([2]),
    )

    events = detector.process_frame(det, speeds={tid: 45.0}, frame_idx=15, tracker_history=history)
    wrong_way = [e for e in events if e.event_type == "potential_wrong_way_movement"]
    assert len(wrong_way) >= 1
    assert tid in wrong_way[0].vehicle_ids


def test_stale_streak_state_pruning_on_disappearing_tracks():
    """Verify that when a track disappears, its streak state is safely pruned."""
    detector = IncidentDetector(fps=25.0, temporal_confirm_frames=5)

    tid = 88
    det = sv.Detections(
        xyxy=np.array([[100, 100, 150, 150]], dtype=np.float32),
        tracker_id=np.array([tid]),
        class_id=np.array([2]),
    )
    history = {tid: [(f, 100.0, 100.0, 100.0, 110.0) for f in range(100)]}

    # Track 88 has streak counter incremented
    detector.process_frame(det, speeds={tid: 0.0}, frame_idx=1, tracker_history=history)
    assert any(tid in key if isinstance(key, tuple) else tid == key for key in detector.anomaly_streak.keys())

    # In next frame, another vehicle appears but track 88 disappeared
    other_det = sv.Detections(
        xyxy=np.array([[300, 300, 350, 350]], dtype=np.float32),
        tracker_id=np.array([999]),
        class_id=np.array([2]),
    )
    detector.process_frame(other_det, speeds={999: 40.0}, frame_idx=2, tracker_history={999: [(2, 300.0, 300.0, 300.0, 320.0)]})

    # Streak counters containing tid 88 should be cleanly pruned
    active_streak_tids = []
    for k in detector.anomaly_streak.keys():
        if isinstance(k, tuple) and len(k) == 2:
            active_streak_tids.append(k[1])
    assert 88 not in active_streak_tids


def test_possible_collision_indicator():
    fps = 25.0
    detector = IncidentDetector(
        fps=fps,
        proximity_threshold_px=50.0,
        temporal_confirm_frames=1,
    )

    # Two vehicles in extreme close proximity (< 50px) and stopped
    det = sv.Detections(
        xyxy=np.array([
            [100, 200, 140, 250],
            [105, 205, 145, 255],
        ], dtype=np.float32),
        tracker_id=np.array([1, 2]),
        class_id=np.array([2, 2]),
    )
    speeds = {1: 0.0, 2: 0.0}
    history = {
        1: [(f, 120.0, 225.0, 120.0, 250.0) for f in range(10)],
        2: [(f, 125.0, 230.0, 125.0, 255.0) for f in range(10)],
    }

    # Frame sequence to reach confirmation
    detector.process_frame(det, speeds, frame_idx=1, tracker_history=history)
    detector.process_frame(det, speeds, frame_idx=2, tracker_history=history)
    events = detector.process_frame(det, speeds, frame_idx=3, tracker_history=history)

    collision_events = [e for e in detector.get_events() if e.event_type == "possible_collision_indicator"]
    assert len(collision_events) >= 1
    assert 1 in collision_events[0].vehicle_ids
    assert 2 in collision_events[0].vehicle_ids
    assert collision_events[0].severity == "CRITICAL"
    assert "human_verification_warning" in collision_events[0].to_dict()
