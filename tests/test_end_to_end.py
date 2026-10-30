"""
End-to-end integration and edge-case tests for TrafficVision pipeline.
"""
from pathlib import Path
import numpy as np
import pytest
import cv2

from src.pipeline import TrafficPipeline, run
from src.reporting import ReportGenerator


def test_pipeline_on_sample_traffic_video(tmp_path):
    sample_video = Path("data/sample_traffic.mp4")
    assert sample_video.exists(), "Sample video data/sample_traffic.mp4 must exist"

    sample_calib = Path("data/sample_calibration.json")
    out_video = tmp_path / "test_out.mp4"

    pipeline = TrafficPipeline(
        weights="yolo11n.pt",
        conf=0.35,
        calibration_path=str(sample_calib) if sample_calib.exists() else None,
        frame_skip=2,  # test frame skip
    )

    result = pipeline.process_video(str(sample_video), output_path=str(out_video), generate_annotated_video=True)

    # 1. Output video exists and has size > 0
    assert out_video.exists()
    assert out_video.stat().st_size > 10000

    # 2. Metadata is valid
    meta = result["video_metadata"]
    assert meta["width"] == 700
    assert meta["height"] == 394
    assert meta["total_frames"] == 51

    # 3. Detections and tracking
    counts = result["vehicle_counts"]
    assert counts["total_detected_instances"] > 0
    assert counts["unique_tracked"] > 0

    # 4. Speed statistics
    spd = result["speed_statistics"]
    assert spd["is_calibrated"] is True
    assert spd["avg_speed_kmh"] > 0

    # 5. Congestion statistics
    cong = result["congestion_statistics"]
    assert cong["dominant_level"] in {"LOW", "MODERATE", "HIGH", "SEVERE"}

    # 6. Incidents recorded
    assert "incidents" in result
    assert isinstance(result["incidents"], list)


def test_edge_case_nonexistent_video():
    pipeline = TrafficPipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.process_video("data/definitely_not_existing_video.mp4")


def test_edge_case_invalid_extension(tmp_path):
    bad_file = tmp_path / "image.png"
    bad_file.write_bytes(b"dummy")
    pipeline = TrafficPipeline()
    with pytest.raises(ValueError):
        pipeline.process_video(str(bad_file))


def test_edge_case_missing_calibration(tmp_path):
    sample_video = Path("data/sample_traffic.mp4")
    pipeline = TrafficPipeline(
        weights="yolo11n.pt",
        calibration_path=None,  # Missing calibration
        frame_skip=3,
    )
    result = pipeline.process_video(str(sample_video), generate_annotated_video=False)
    spd = result["speed_statistics"]
    assert spd["is_calibrated"] is False
    assert spd["speed_warning"] is not None


def test_edge_case_empty_black_video(tmp_path):
    """Test video with zero vehicle activity (all black frames)."""
    empty_vid_path = tmp_path / "empty.mp4"
    out = cv2.VideoWriter(str(empty_vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    for _ in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    pipeline = TrafficPipeline(weights="yolo11n.pt", frame_skip=1)
    result = pipeline.process_video(str(empty_vid_path), generate_annotated_video=False)

    assert result["vehicle_counts"]["total_detected_instances"] == 0
    assert result["vehicle_counts"]["unique_tracked"] == 0
    assert result["congestion_statistics"]["dominant_level"] == "LOW"
    assert len(result["incidents"]) == 0
