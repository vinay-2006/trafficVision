"""
Report generation and export module for TrafficVision.

Handles structured result packaging:
  - JSON Analytics Summary
  - CSV Incident Log
  - CSV Traffic Metrics Timeline
  - Comprehensive ZIP Report Package
"""
import csv
import json
import zipfile
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional


class ReportGenerator:
    def __init__(self, analytics_result: Dict):
        """
        Args:
            analytics_result: Structured result dictionary from the processing pipeline.
        """
        self.data = analytics_result

    def to_json(self, indent: int = 2) -> str:
        """Serializes pipeline results to structured JSON."""
        return json.dumps(self.data, indent=indent, default=str)

    def save_json(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(self.to_json(), encoding="utf-8")

    def to_incident_csv(self) -> str:
        """Generates CSV format of all detected incident events."""
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Event ID",
            "Event Type",
            "Title",
            "Severity",
            "Timestamp (s)",
            "Frame Index",
            "Vehicle IDs",
            "Confidence",
            "Description",
            "Detection Evidence",
            "Human Verification Disclaimer",
        ])

        incidents = self.data.get("incidents", [])
        for inc in incidents:
            vids = ",".join(str(vid) for vid in inc.get("vehicle_ids", []))
            writer.writerow([
                inc.get("event_id", ""),
                inc.get("event_type", ""),
                inc.get("title", ""),
                inc.get("severity", ""),
                inc.get("timestamp_sec", ""),
                inc.get("frame_idx", ""),
                vids,
                inc.get("confidence", ""),
                inc.get("description", ""),
                inc.get("evidence", ""),
                inc.get("human_verification_warning", ""),
            ])

        return output.getvalue()

    def save_incident_csv(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(self.to_incident_csv(), encoding="utf-8")

    def to_metrics_csv(self) -> str:
        """Generates time-series CSV of per-frame / per-window congestion and counts."""
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Frame Index",
            "Timestamp (s)",
            "Active Vehicles",
            "Vehicles in ROI",
            "Average Speed (km/h)",
            "Flow Rate (veh/min)",
            "Occupancy Ratio",
            "Congestion Score",
            "Congestion Level",
            "Buildup Trend",
        ])

        snapshots = self.data.get("congestion_timeline", [])
        for snap in snapshots:
            writer.writerow([
                snap.get("frame_idx", ""),
                snap.get("timestamp_sec", ""),
                snap.get("active_vehicles", ""),
                snap.get("vehicles_in_roi", ""),
                snap.get("avg_speed_kmh", ""),
                snap.get("flow_rate_vpm", ""),
                snap.get("occupancy_ratio", ""),
                snap.get("congestion_score", ""),
                snap.get("congestion_level", ""),
                snap.get("buildup_trend", ""),
            ])

        return output.getvalue()

    def save_metrics_csv(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(self.to_metrics_csv(), encoding="utf-8")

    def build_summary_markdown(self) -> str:
        """Constructs an executive text summary of the run."""
        meta = self.data.get("video_metadata", {})
        counts = self.data.get("vehicle_counts", {})
        speed = self.data.get("speed_statistics", {})
        cong = self.data.get("congestion_statistics", {})
        inc_summary = self.data.get("incident_summary", {})

        md = f"""# TrafficVision Analytics Executive Summary

## 1. Video & Processing Metadata
- **Source File**: {meta.get("filename", "N/A")}
- **Duration**: {meta.get("duration_sec", 0):.1f} s ({meta.get("total_frames", 0)} frames @ {meta.get("fps", 0)} FPS)
- **Resolution**: {meta.get("width", 0)} x {meta.get("height", 0)} px
- **Processing Time**: {self.data.get("processing_duration_sec", 0):.2f} s ({self.data.get("processing_fps", 0):.1f} FPS)

## 2. Traffic Flow & Vehicle Counts
- **Total Unique Vehicles**: {counts.get("unique_tracked", 0)}
- **Incoming Vehicles**: {counts.get("vehicles_in", 0)}
- **Outgoing Vehicles**: {counts.get("vehicles_out", 0)}
- **Total Line Crossings**: {counts.get("total_crossings", 0)}

### Vehicle Classification Breakdown
"""
        for cls_name, cnt in counts.get("class_distribution", {}).items():
            md += f"- **{cls_name.capitalize()}**: {cnt}\n"

        calib_note = "(Calibrated Ground Truth)" if speed.get("is_calibrated") else "[Honest Caveat: Uncalibrated Placeholder Scale]"
        md += f"""
## 3. Speed & Congestion Intelligence
- **Average Traffic Speed**: {speed.get("avg_speed_kmh", 0):.1f} km/h {calib_note}
- **Maximum Observed Speed**: {speed.get("max_speed_kmh", 0):.1f} km/h
- **Dominant Congestion Level**: {cong.get("dominant_level", "LOW")}
- **Peak Vehicles in ROI**: {cong.get("peak_vehicles_in_roi", 0)}

## 4. Potential Incident Alerts
- **Total Detected Incidents**: {inc_summary.get("total_incidents", 0)}
- **Critical Severity**: {inc_summary.get("incidents_by_severity", {}).get("CRITICAL", 0)}
- **High Severity**: {inc_summary.get("incidents_by_severity", {}).get("HIGH", 0)}
- **Medium Severity**: {inc_summary.get("incidents_by_severity", {}).get("MEDIUM", 0)}

> **Human Verification Warning**:
> {inc_summary.get("disclaimer", "All logged alerts are potential rule-based indicators and require human operator confirmation.")}
"""
        return md

    def export_zip(self, output_zip_path: str, include_video_path: Optional[str] = None) -> str:
        """
        Creates a ZIP report package containing JSON summary, CSV logs, and Executive Markdown.
        """
        Path(output_zip_path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write JSON
            zf.writestr("analytics_summary.json", self.to_json())
            # Write Incident CSV
            zf.writestr("incident_log.csv", self.to_incident_csv())
            # Write Metrics CSV
            zf.writestr("traffic_metrics_timeline.csv", self.to_metrics_csv())
            # Write Executive Summary
            zf.writestr("EXECUTIVE_SUMMARY.md", self.build_summary_markdown())

            # Optionally include annotated video
            if include_video_path and Path(include_video_path).exists():
                zf.write(include_video_path, arcname=f"annotated_video{Path(include_video_path).suffix}")

        return str(output_zip_path)
