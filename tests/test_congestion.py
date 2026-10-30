"""
Unit tests for TrafficDensityAnalyzer in src/congestion.py.
"""
import numpy as np
import pytest
import supervision as sv

from src.congestion import TrafficDensityAnalyzer, CongestionSnapshot


def test_congestion_analyzer_empty_frame():
    analyzer = TrafficDensityAnalyzer(fps=25.0)
    empty_det = sv.Detections.empty()
    snapshot = analyzer.analyze_frame(
        detections=empty_det,
        speeds={},
        frame_idx=0,
        frame_shape=(480, 640),
        line_crossing_total=0,
    )
    assert isinstance(snapshot, CongestionSnapshot)
    assert snapshot.congestion_level == "LOW"
    assert snapshot.vehicles_in_roi == 0
    assert snapshot.active_vehicles == 0
    assert snapshot.congestion_score == 0.0


def test_congestion_analyzer_high_density_slow_speed():
    analyzer = TrafficDensityAnalyzer(fps=25.0, free_flow_speed_kmh=60.0)
    # 10 stopped/crawling vehicles
    boxes = []
    tids = []
    speeds = {}
    for i in range(10):
        boxes.append([50 + i * 40, 100, 80 + i * 40, 150])
        tids.append(i + 1)
        speeds[i + 1] = 4.0  # < 5 km/h stopped

    det = sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        tracker_id=np.array(tids),
        class_id=np.full(10, 2),
    )

    snapshot = analyzer.analyze_frame(
        detections=det,
        speeds=speeds,
        frame_idx=10,
        frame_shape=(480, 640),
        line_crossing_total=0,
    )
    assert snapshot.congestion_level in {"HIGH", "SEVERE"}
    assert snapshot.vehicles_in_roi == 10
    assert snapshot.avg_speed_kmh == 4.0
    assert snapshot.congestion_score >= 0.55


def test_congestion_analyzer_roi_filtering():
    # ROI covers x in [200, 400], y in [200, 400]
    roi = np.array([[200, 200], [400, 200], [400, 400], [200, 400]], dtype=np.int32)
    analyzer = TrafficDensityAnalyzer(fps=25.0, road_roi=roi)

    # Box 1 is inside ROI (bottom center is (300, 350))
    # Box 2 is outside ROI (bottom center is (50, 150))
    det = sv.Detections(
        xyxy=np.array([
            [280, 300, 320, 350],  # inside
            [30, 100, 70, 150],    # outside
        ], dtype=np.float32),
        tracker_id=np.array([1, 2]),
        class_id=np.array([2, 2]),
    )

    snapshot = analyzer.analyze_frame(
        detections=det,
        speeds={1: 50.0, 2: 50.0},
        frame_idx=1,
        frame_shape=(480, 640),
    )
    assert snapshot.active_vehicles == 2
    assert snapshot.vehicles_in_roi == 1


def test_congestion_summary_aggregation():
    analyzer = TrafficDensityAnalyzer(fps=25.0)
    empty_det = sv.Detections.empty()
    for f in range(20):
        analyzer.analyze_frame(empty_det, {}, frame_idx=f, frame_shape=(480, 640))

    summary = analyzer.get_summary()
    assert "average_congestion_score" in summary
    assert summary["dominant_level"] == "LOW"
    assert summary["peak_vehicles_in_roi"] == 0
    assert summary["congestion_level_distribution"]["LOW"] == 100.0


def test_congestion_missing_speeds_graceful_handling():
    """Verify analyzer gracefully handles missing speed entries without error."""
    analyzer = TrafficDensityAnalyzer(fps=30.0, free_flow_speed_kmh=50.0)
    det = sv.Detections(
        xyxy=np.array([[50, 50, 100, 100], [150, 150, 200, 200]], dtype=np.float32),
        tracker_id=np.array([1, 2]),
        class_id=np.array([2, 2]),
    )
    # Speeds dict is empty - no speed estimates yet
    snapshot = analyzer.analyze_frame(
        detections=det,
        speeds={},
        frame_idx=1,
        frame_shape=(480, 640),
    )
    assert snapshot.active_vehicles == 2
    assert snapshot.vehicles_in_roi == 2
    # Analyzer sets default nominal speed when speeds are unobserved
    assert snapshot.avg_speed_kmh > 0.0
    # Should not crash and should produce valid snapshot
    assert snapshot.congestion_level in {"LOW", "MODERATE", "HIGH", "SEVERE"}


def test_congestion_roi_area_sources_and_density_units():
    """Verify physical density is computed in veh/m^2 with explicit area provenance."""
    # Test case 1: Nominal uncalibrated default area
    analyzer_default = TrafficDensityAnalyzer(fps=30.0)
    assert analyzer_default.roi_area_source == "uncalibrated_nominal_default"
    assert analyzer_default.roi_area_sq_meters == 400.0

    # Test case 2: Calibrated ground plane area
    analyzer_calib = TrafficDensityAnalyzer(
        fps=30.0,
        roi_area_sq_meters=600.0,
        roi_area_source="calibrated_ground_plane",
    )
    assert analyzer_calib.roi_area_source == "calibrated_ground_plane"
    assert analyzer_calib.roi_area_sq_meters == 600.0

    # 6 vehicles in 600 m^2 => 0.01 veh/m^2
    det = sv.Detections(
        xyxy=np.tile(np.array([100, 100, 150, 150], dtype=np.float32), (6, 1)),
        tracker_id=np.arange(1, 7),
        class_id=np.full(6, 2),
    )
    snap = analyzer_calib.analyze_frame(
        detections=det,
        speeds={i: 40.0 for i in range(1, 7)},
        frame_idx=10,
        frame_shape=(480, 640),
    )
    assert pytest.approx(snap.density_veh_per_sq_meter, 0.0001) == 0.01
    assert snap.roi_area_source == "calibrated_ground_plane"
    assert analyzer_calib.roi_area_sq_meters == 600.0


def test_congestion_flow_rate_handles_counter_reset_and_frame_skipping():
    """Verify flow rate calculation is resilient to frame skipping and counter resets."""
    analyzer = TrafficDensityAnalyzer(fps=30.0, flow_window_sec=10.0)

    # Frame 0: 5 vehicles crossed
    empty_det = sv.Detections.empty()
    analyzer.analyze_frame(empty_det, {}, frame_idx=0, frame_shape=(480, 640), line_crossing_total=5)

    # Frame 60 (2.0 elapsed sec at 30 fps, frame_skip = 2): 15 vehicles crossed
    # Delta = 10 crossed in 2.0 sec => 5 veh/s => 300 veh/min
    snap = analyzer.analyze_frame(empty_det, {}, frame_idx=60, frame_shape=(480, 640), line_crossing_total=15)
    assert pytest.approx(snap.flow_rate_vpm, 0.1) == 300.0

    # Simulate counter reset (e.g. tracking reinitialized): count drops to 2
    # Should not produce negative flow rate
    snap_reset = analyzer.analyze_frame(empty_det, {}, frame_idx=90, frame_shape=(480, 640), line_crossing_total=2)
    assert snap_reset.flow_rate_vpm >= 0.0


def test_congestion_level_classification_tiers():
    """Verify multiple congestion score tiers are reachable."""
    analyzer = TrafficDensityAnalyzer(fps=30.0, free_flow_speed_kmh=60.0)

    # 1. LOW: 1 car moving at free-flow speed
    det_low = sv.Detections(
        xyxy=np.array([[100, 100, 150, 150]], dtype=np.float32),
        tracker_id=np.array([1]),
        class_id=np.array([2]),
    )
    snap_low = analyzer.analyze_frame(det_low, {1: 60.0}, frame_idx=1, frame_shape=(480, 640))
    assert snap_low.congestion_level == "LOW"

    # 2. HIGH/SEVERE: 15 crawling cars (2 km/h)
    det_sev = sv.Detections(
        xyxy=np.tile(np.array([100, 100, 150, 150], dtype=np.float32), (15, 1)),
        tracker_id=np.arange(1, 16),
        class_id=np.full(15, 2),
    )
    snap_sev = analyzer.analyze_frame(det_sev, {i: 2.0 for i in range(1, 16)}, frame_idx=3, frame_shape=(480, 640))
    assert snap_sev.congestion_level in ("HIGH", "SEVERE")
    assert snap_sev.congestion_score >= 0.55
