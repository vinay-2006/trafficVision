"""
Unit tests for ReportGenerator in src/reporting.py.
"""
import json
import zipfile
from pathlib import Path
import pytest

from src.reporting import ReportGenerator


@pytest.fixture
def sample_payload():
    return {
        "video_metadata": {
            "source_path": "test.mp4",
            "filename": "test.mp4",
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "total_frames": 250,
            "duration_sec": 10.0,
        },
        "processing_duration_sec": 5.2,
        "processing_fps": 48.0,
        "frames_processed": 250,
        "vehicle_counts": {
            "total_detected_instances": 120,
            "unique_tracked": 8,
            "vehicles_in": 5,
            "vehicles_out": 3,
            "total_crossings": 8,
            "class_distribution": {"car": 6, "bus": 2},
        },
        "speed_statistics": {
            "avg_speed_kmh": 45.2,
            "max_speed_kmh": 68.4,
            "is_calibrated": True,
        },
        "congestion_statistics": {
            "average_congestion_score": 0.28,
            "dominant_level": "LOW",
            "peak_vehicles_in_roi": 4,
            "average_speed_kmh": 45.2,
            "congestion_level_distribution": {"LOW": 85.0, "MODERATE": 15.0},
        },
        "congestion_timeline": [
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "active_vehicles": 3,
                "vehicles_in_roi": 2,
                "avg_speed_kmh": 50.0,
                "flow_rate_vpm": 12.0,
                "occupancy_ratio": 0.12,
                "congestion_score": 0.22,
                "congestion_level": "LOW",
                "buildup_trend": "STABLE",
            }
        ],
        "incidents": [
            {
                "event_id": "INC-001",
                "event_type": "suspicious_stationary_vehicle",
                "title": "Suspicious Stationary Vehicle",
                "vehicle_ids": [4],
                "timestamp_sec": 3.4,
                "frame_idx": 85,
                "severity": "HIGH",
                "description": "Vehicle #4 stopped in roadway",
                "evidence": "Displacement 2.1px",
                "confidence": 0.9,
                "human_verification_warning": "Automated alert",
            }
        ],
        "incident_summary": {
            "total_incidents": 1,
            "incidents_by_severity": {"HIGH": 1},
            "disclaimer": "Automated alert",
        },
    }


def test_report_json_generation(sample_payload):
    rep = ReportGenerator(sample_payload)
    json_str = rep.to_json()
    parsed = json.loads(json_str)
    assert parsed["video_metadata"]["filename"] == "test.mp4"
    assert parsed["vehicle_counts"]["unique_tracked"] == 8


def test_report_csv_generation(sample_payload):
    rep = ReportGenerator(sample_payload)
    inc_csv = rep.to_incident_csv()
    assert "INC-001" in inc_csv
    assert "suspicious_stationary_vehicle" in inc_csv
    assert "Vehicle #4 stopped" in inc_csv

    metrics_csv = rep.to_metrics_csv()
    assert "Frame Index" in metrics_csv
    assert "Congestion Score" in metrics_csv
    assert "50.0" in metrics_csv


def test_report_zip_generation(sample_payload, tmp_path):
    rep = ReportGenerator(sample_payload)
    zip_path = tmp_path / "traffic_report.zip"
    rep.export_zip(str(zip_path))

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "analytics_summary.json" in namelist
        assert "incident_log.csv" in namelist
        assert "traffic_metrics_timeline.csv" in namelist
        assert "EXECUTIVE_SUMMARY.md" in namelist

        summary_md = zf.read("EXECUTIVE_SUMMARY.md").decode("utf-8")
        assert "TrafficVision Analytics Executive Summary" in summary_md
        assert "Total Unique Vehicles" in summary_md
