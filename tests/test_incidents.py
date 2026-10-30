"""
Unit tests for IncidentDetector in src/incidents.py.
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


def test_abnormal_trajectory_detection():
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
    assert events[0].event_type == "abnormal_trajectory"
    assert tid in events[0].vehicle_ids


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
