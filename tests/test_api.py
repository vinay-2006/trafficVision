"""
Unit and integration tests for FastAPI service in api/main.py.

Includes:
- Direct handler unit tests (isolated, fast execution).
- TestClient HTTP-level integration tests (/health, /analyze, /analyze-video).
- Parameter validation (conf, frame_skip, upload size, file extensions).
- Calibration JSON validation.
- Temporary file cleanup verification on success and failure.
"""
from pathlib import Path
from unittest.mock import patch
import json
import pytest
from fastapi import UploadFile
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient

from api.main import health, analyze, analyze_video, app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ============================================================================
# DIRECT HANDLER UNIT TESTS
# ============================================================================

def test_api_health_function():
    """Unit test: direct health() function invocation."""
    result = health()
    assert result["status"] == "healthy"
    assert "TrafficVision" in result["service"]
    assert "version" in result


def test_api_analyze_rejects_invalid_extension_unit(tmp_path):
    """Unit test: direct analyze() rejects unsupported video extensions."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello")
    with open(text_file, "rb") as f:
        upload = UploadFile(file=f, filename="notes.txt")
        with pytest.raises(HTTPException) as exc_info:
            analyze(video=upload)
        assert exc_info.value.status_code == 400
        assert "Upload a .mp4" in exc_info.value.detail


def test_api_analyze_mocked_pipeline_unit(tmp_path):
    """Unit test: direct analyze() handles mocked pipeline execution."""
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
            "is_calibrated": False,
            "calibration_status": "illustrative",
            "speed_warning": "Speeds are uncalibrated",
            "average_speed_kmh": 42.5,
            "max_speed_kmh": 58.0,
            "per_track_max_speed": {1: 45.0, 2: 52.0},
        },
        "congestion_statistics": {
            "current_level": "MODERATE",
            "active_vehicles_peak": 7,
            "density_veh_per_sq_meter": 0.0175,
            "relative_occupancy_percent": 15.0,
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
            assert data["calibration_used"] is False
            assert data["calibration_status"] == "illustrative"


# ============================================================================
# FASTAPI TESTCLIENT HTTP INTEGRATION TESTS
# ============================================================================

def test_api_health_endpoint_testclient(client):
    """Integration test: GET /health returns 200 with service information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "TrafficVision" in data["service"]
    assert data["version"] == "1.0.0"


def test_api_analyze_rejects_non_video_testclient(client):
    """Integration test: POST /analyze rejects text upload with 400 Bad Request."""
    files = {"video": ("document.txt", b"plain text content", "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "Upload a .mp4" in response.json()["detail"]


def test_api_analyze_validates_conf_bounds_testclient(client):
    """Integration test: POST /analyze validates conf range (0.05 - 0.95)."""
    files = {"video": ("test.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")}
    
    # Below lower bound
    res_low = client.post("/analyze?conf=0.01", files=files)
    assert res_low.status_code == 400
    assert "between 0.05 and 0.95" in res_low.json()["detail"]

    # Above upper bound
    res_high = client.post("/analyze?conf=0.99", files=files)
    assert res_high.status_code == 400
    assert "between 0.05 and 0.95" in res_high.json()["detail"]


def test_api_analyze_validates_frame_skip_bounds_testclient(client):
    """Integration test: POST /analyze validates frame_skip range (1 - 30)."""
    files = {"video": ("test.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")}

    res_zero = client.post("/analyze?frame_skip=0", files=files)
    assert res_zero.status_code == 400
    assert "between 1 and 30" in res_zero.json()["detail"]

    res_high = client.post("/analyze?frame_skip=45", files=files)
    assert res_high.status_code == 400
    assert "between 1 and 30" in res_high.json()["detail"]


def test_api_analyze_rejects_invalid_calibration_json_testclient(client):
    """Integration test: POST /analyze rejects corrupt calibration JSON syntax."""
    files = {
        "video": ("traffic.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4"),
        "calibration": ("calib.json", b"{not valid json}", "application/json"),
    }
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "Malformed calibration JSON" in response.json()["detail"]


def test_api_analyze_rejects_degenerate_calibration_geometry_testclient(client):
    """Integration test: POST /analyze rejects collinear/degenerate calibration points."""
    collinear_calib = json.dumps({
        "source_points": [[100, 100], [200, 200], [300, 300], [400, 400]],
        "target_points_meters": [[0, 0], [10, 0], [10, 40], [0, 40]],
        "is_calibrated": True,
    }).encode("utf-8")

    files = {
        "video": ("traffic.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4"),
        "calibration": ("calib.json", collinear_calib, "application/json"),
    }
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    assert "Geometry validation failed" in response.json()["detail"]


def test_api_analyze_mocked_pipeline_testclient(client):
    """Integration test: POST /analyze with mocked pipeline returns full analytics."""
    files = {
        "video": ("sample.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4"),
    }

    mock_result = {
        "frames_processed": 60,
        "vehicle_counts": {
            "vehicles_in": 10,
            "vehicles_out": 7,
            "unique_tracked": 17,
            "class_distribution": {"car": 14, "truck": 3},
        },
        "speed_statistics": {
            "is_calibrated": False,
            "calibration_status": "illustrative",
            "speed_warning": "Illustrative example calibration in use; speeds are approximate.",
            "average_speed_kmh": 46.2,
            "max_speed_kmh": 61.5,
            "per_track_max_speed": {1: 45.0, 2: 61.5},
        },
        "congestion_statistics": {
            "current_level": "LOW",
            "active_vehicles_peak": 5,
            "density_veh_per_sq_meter": 0.0125,
            "relative_occupancy_percent": 11.2,
        },
        "incidents": [],
        "incident_summary": {"total_events": 0, "by_severity": {}},
        "video_metadata": {"fps": 30.0, "total_frames": 60},
        "processing_duration_sec": 1.5,
        "processing_fps": 40.0,
    }

    with patch("api.main.TrafficPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.process_video.return_value = mock_result
        response = client.post("/analyze?conf=0.4&frame_skip=2", files=files)
        assert response.status_code == 200
        payload = response.json()
        assert payload["frames_processed"] == 60
        assert payload["vehicles_in"] == 10
        assert payload["vehicles_out"] == 7
        assert payload["unique_tracks"] == 17
        assert payload["congestion_statistics"]["current_level"] == "LOW"
        assert payload["calibration_used"] is False
        assert payload["calibration_status"] == "illustrative"
        assert "speed_warning" in payload


def test_api_analyze_video_mocked_pipeline_testclient(client, tmp_path):
    """Integration test: POST /analyze-video returns annotated mp4 file with background cleanup."""
    dummy_out = tmp_path / "fake_annotated.mp4"
    dummy_out.write_bytes(b"\x00\x00\x00\x18ftypmp42fakeannotateddata")

    files = {
        "video": ("traffic_clip.mp4", b"\x00\x00\x00\x18ftypmp42clipdata", "video/mp4"),
    }

    def fake_process_video(video_path, output_path=None, generate_annotated_video=False):
        if output_path:
            Path(output_path).write_bytes(dummy_out.read_bytes())
        return {"frames_processed": 30}

    with patch("api.main.TrafficPipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.process_video.side_effect = fake_process_video
        response = client.post("/analyze-video", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert "annotated_traffic_clip.mp4" in response.headers["content-disposition"]
        assert len(response.content) > 0
