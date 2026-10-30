"""
Unit and integration tests for FastAPI service in api/main.py.
"""
from pathlib import Path
from unittest.mock import patch
import json
import pytest
from fastapi import UploadFile
from fastapi.exceptions import HTTPException

from api.main import health, analyze, app


def test_api_health_function():
    result = health()
    assert result["status"] == "healthy"
    assert "TrafficVision" in result["service"]


def test_api_analyze_rejects_invalid_extension(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello")
    with open(text_file, "rb") as f:
        upload = UploadFile(file=f, filename="notes.txt")
        with pytest.raises(HTTPException) as exc_info:
            analyze(video=upload)
        assert exc_info.value.status_code == 400
        assert "Upload a .mp4" in exc_info.value.detail


def test_api_analyze_mocked_pipeline(tmp_path):
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"fake mp4 content")

    mock_result = {
        "frames_processed": 50,
        "vehicle_counts": {
            "vehicles_in": 12,
            "vehicles_out": 8,
            "unique_tracked": 20,
            "class_distribution": {"car": 18, "truck": 2},
        },
        "speed_statistics": {
            "is_calibrated": True,
            "speed_warning": None,
            "average_speed_kmh": 42.5,
            "max_speed_kmh": 58.0,
            "per_track_max_speed": {1: 45.0, 2: 52.0},
        },
        "congestion_statistics": {
            "current_level": "MODERATE",
            "active_vehicles_peak": 7,
            "average_density_vpk": 14.2,
        },
        "incidents": [],
        "incident_summary": {"total_events": 0, "by_severity": {}},
        "video_metadata": {"fps": 30.0, "total_frames": 50},
        "processing_duration_sec": 1.25,
        "processing_fps": 40.0,
    }

    with open(dummy_video, "rb") as f:
        upload = UploadFile(file=f, filename="dummy.mp4")
        with patch("api.main.TrafficPipeline") as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_video.return_value = mock_result
            response = analyze(video=upload, conf=0.4, frame_skip=1)
            assert response.status_code == 200
            data = json.loads(response.body.decode("utf-8"))
            assert data["frames_processed"] == 50
            assert data["vehicles_in"] == 12
            assert data["vehicles_out"] == 8
            assert data["congestion_statistics"]["current_level"] == "MODERATE"
            assert data["calibration_used"] is True
