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
