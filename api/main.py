"""
FastAPI service for TrafficVision AI Traffic Monitoring and Analytics.

Endpoints:
    POST /analyze       - Upload video (+ optional calibration.json), get comprehensive analytics JSON
    POST /analyze-video - Upload video, get back annotated MP4 video
    GET  /health        - Liveness check
"""
import os
import sys
from pathlib import Path

# Fix Matplotlib configuration directory before any sub-packages import it
_cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))

import json
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse

from src.perspective import PerspectiveTransformer
from src.pipeline import TrafficPipeline
from src.reporting import ReportGenerator

app = FastAPI(
    title="TrafficVision Intelligence API",
    version="1.0.0",
    description="AI-powered intelligent traffic monitoring, vehicle counting, speed estimation, congestion analysis, and incident detection.",
)

WEIGHTS_PATH = "yolo11n.pt"
MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 300 MB limit


def _validate_and_save_upload(video: UploadFile) -> str:
    if not video.filename or not video.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Upload a .mp4, .avi, or .mov video file")

    suffix = Path(video.filename).suffix
    total_bytes = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_vid:
        while True:
            chunk = video.file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                tmp_vid.close()
                Path(tmp_vid.name).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds maximum allowed upload size (300 MB)")
            tmp_vid.write(chunk)
        return tmp_vid.name


def _validate_and_save_calibration(calibration: Optional[UploadFile]) -> Optional[str]:
    if calibration is None or not hasattr(calibration, "file") or not getattr(calibration, "filename", None):
        return None

    try:
        calib_bytes = calibration.file.read()
        calib_data = json.loads(calib_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed calibration JSON: {str(e)}")

    valid, msg, _ = PerspectiveTransformer.validate_calibration_dict(calib_data)
    if not valid:
        # Surface the exact rejection reason (geometry, unknown provenance, forged verified, etc.)
        raise HTTPException(status_code=400, detail=msg)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_cal:
        tmp_cal.write(calib_bytes)
        return tmp_cal.name


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "TrafficVision AI Traffic Intelligence",
        "version": "1.0.0",
        "weights": WEIGHTS_PATH,
    }


@app.post("/analyze")
def analyze(
    video: UploadFile = File(...),
    calibration: Optional[UploadFile] = None,
    conf: float = 0.35,
    frame_skip: int = 1,
):
    if not (0.05 <= conf <= 0.95):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.05 and 0.95")
    if not (1 <= frame_skip <= 30):
        raise HTTPException(status_code=400, detail="Frame skip must be an integer between 1 and 30")

    video_path = _validate_and_save_upload(video)
    calib_path = None

    try:
        calib_path = _validate_and_save_calibration(calibration)
        pipeline = TrafficPipeline(
            weights=WEIGHTS_PATH,
            conf=conf,
            calibration_path=calib_path,
            frame_skip=frame_skip,
        )
        result = pipeline.process_video(video_path, generate_annotated_video=False)

        response_payload = {
            "frames_processed": result["frames_processed"],
            "vehicles_in": result["vehicle_counts"]["vehicles_in"],
            "vehicles_out": result["vehicle_counts"]["vehicles_out"],
            "unique_tracks": result["vehicle_counts"]["unique_tracked"],
            "max_speed_kmh_per_track": result["speed_statistics"]["per_track_max_speed"],
            "detection_class_counts": result["vehicle_counts"]["class_distribution"],
            "calibration_used": result["speed_statistics"]["is_calibrated"],
            "calibration_status": result["speed_statistics"]["calibration_status"],
            "speed_warning": result["speed_statistics"]["speed_warning"],
            "speed_statistics": result["speed_statistics"],
            "congestion_statistics": result["congestion_statistics"],
            "incidents": result["incidents"],
            "incident_summary": result["incident_summary"],
            "video_metadata": result["video_metadata"],
            "processing_duration_sec": result["processing_duration_sec"],
            "processing_fps": result["processing_fps"],
        }
        return JSONResponse(content=response_payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
    finally:
        Path(video_path).unlink(missing_ok=True)
        if calib_path:
            Path(calib_path).unlink(missing_ok=True)


@app.post("/analyze-video")
def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    calibration: Optional[UploadFile] = None,
    conf: float = 0.35,
    frame_skip: int = 1,
):
    if not (0.05 <= conf <= 0.95):
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.05 and 0.95")
    if not (1 <= frame_skip <= 30):
        raise HTTPException(status_code=400, detail="Frame skip must be an integer between 1 and 30")

    video_path = _validate_and_save_upload(video)
    calib_path = None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        out_annotated = tmp_out.name

    try:
        calib_path = _validate_and_save_calibration(calibration)
        pipeline = TrafficPipeline(
            weights=WEIGHTS_PATH,
            conf=conf,
            calibration_path=calib_path,
            frame_skip=frame_skip,
        )
        pipeline.process_video(video_path, output_path=out_annotated, generate_annotated_video=True)

        def cleanup():
            Path(video_path).unlink(missing_ok=True)
            if calib_path:
                Path(calib_path).unlink(missing_ok=True)
            Path(out_annotated).unlink(missing_ok=True)

        background_tasks.add_task(cleanup)
        return FileResponse(out_annotated, media_type="video/mp4", filename=f"annotated_{Path(video.filename).stem}.mp4")
    except HTTPException:
        Path(video_path).unlink(missing_ok=True)
        if calib_path:
            Path(calib_path).unlink(missing_ok=True)
        Path(out_annotated).unlink(missing_ok=True)
        raise
    except Exception as e:
        Path(video_path).unlink(missing_ok=True)
        if calib_path:
            Path(calib_path).unlink(missing_ok=True)
        Path(out_annotated).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Video annotation failed: {str(e)}")
