"""
Perspective transform: maps pixel coordinates in the camera view to
real-world ground-plane coordinates (meters), so pixel displacement
between frames can be converted into an actual speed.

Calibration integrity rules:
- Calibration is per-camera/per-video.
- Bundled sample calibrations are illustrative/examples, NEVER certified ground truth.
- Uploaded calibrations must be strictly validated for finite coordinates,
  non-degenerate quadrilateral geometry, positive dimensions, and non-singular matrix.
- A JSON field containing `is_calibrated: true` is never trusted blindly without
  provenance verification and geometric validation.
"""
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class PerspectiveTransformer:
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        is_calibrated: bool = False,
        status: str = "illustrative",
        provenance: str = "default_placeholder",
        validation_message: str = "Uncalibrated example geometry",
    ):
        """
        Args:
            source: (4, 2) array of pixel coords, road quadrilateral,
                ordered top-left, top-right, bottom-right, bottom-left.
            target: (4, 2) array of real-world coords in meters for the
                same 4 points, same ordering (a rectangle).
            is_calibrated: whether these coordinates represent certified measured ground truth.
            status: 'calibrated' | 'illustrative' | 'uncalibrated' | 'invalid'
            provenance: descriptive source of calibration data
            validation_message: diagnostic explanation of calibration state
        """
        self.source = np.asarray(source, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

        # Validate geometry on initialization
        valid, msg = self.validate_geometry(self.source, self.target)
        if not valid:
            self.is_calibrated = False
            self.status = "invalid"
            self.validation_message = msg
            self.provenance = provenance
            # Fall back to identity-like safe transform
            self.m = np.eye(3, dtype=np.float32)
        else:
            self.is_calibrated = bool(is_calibrated)
            self.status = status if status in ("calibrated", "illustrative", "uncalibrated") else "illustrative"
            self.provenance = provenance
            self.validation_message = validation_message if is_calibrated else (msg if not is_calibrated else "Valid geometry")
            self.m = cv2.getPerspectiveTransform(self.source, self.target)

        # Calculate real-world bounding dimensions
        try:
            self.target_width_m = float(np.linalg.norm(self.target[1] - self.target[0]))
            self.target_length_m = float(np.linalg.norm(self.target[3] - self.target[0]))
            self.calibrated_area_sq_m = float(cv2.contourArea(self.target))
        except Exception:
            self.target_width_m = 0.0
            self.target_length_m = 0.0
            self.calibrated_area_sq_m = 0.0

    @staticmethod
    def validate_geometry(source: np.ndarray, target: np.ndarray) -> Tuple[bool, str]:
        """
        Validates that source and target points are non-degenerate, finite,
        non-collinear, and form positive-area quadrilaterals.
        """
        if source.shape != (4, 2) or target.shape != (4, 2):
            return False, f"Points must have shape (4, 2). Source: {source.shape}, Target: {target.shape}"

        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            return False, "Points contain non-finite numbers (NaN or Inf)"

        # Check positive area in pixel quadrilateral
        src_area = float(cv2.contourArea(source.astype(np.float32)))
        if src_area < 50.0:
            return False, f"Source quadrilateral pixel area too small or degenerate: {src_area:.1f}px²"

        # Check positive area in target meter quadrilateral
        tgt_area = float(cv2.contourArea(target.astype(np.float32)))
        if tgt_area < 1.0:
            return False, f"Target real-world area too small or degenerate: {tgt_area:.1f}m²"

        # Check widths and lengths in real-world target
        width_m = float(np.linalg.norm(target[1] - target[0]))
        length_m = float(np.linalg.norm(target[3] - target[0]))
        if width_m <= 0.2 or length_m <= 0.2:
            return False, f"Target dimensions too small: width={width_m:.2f}m, length={length_m:.2f}m"

        if width_m > 300.0 or length_m > 2000.0:
            return False, f"Target dimensions exceed physical road bounds: width={width_m:.1f}m, length={length_m:.1f}m"

        # Verify perspective transform solvability
        try:
            m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
            det = float(np.linalg.det(m))
            if abs(det) < 1e-9 or not np.all(np.isfinite(m)):
                return False, f"Perspective transform matrix is singular or ill-conditioned (det={det:.2e})"
        except Exception as e:
            return False, f"Failed to compute perspective transform matrix: {str(e)}"

        return True, "Valid quadrilateral geometry"

    @classmethod
    def validate_calibration_dict(cls, data: dict) -> Tuple[bool, str, Dict]:
        """
        Inspects calibration dictionary from JSON upload.
        Guarantees that an uploaded file cannot claim genuine ground truth
        if it lacks provenance or fails geometric sanity.
        """
        if not isinstance(data, dict):
            return False, "Calibration JSON root must be an object", {}

        if "source_points" not in data or "target_points_meters" not in data:
            return False, "Missing 'source_points' or 'target_points_meters'", {}

        try:
            source = np.array(data["source_points"], dtype=np.float32)
            target = np.array(data["target_points_meters"], dtype=np.float32)
        except Exception as e:
            return False, f"Malformed point coordinate values: {str(e)}", {}

        valid_geom, geom_msg = cls.validate_geometry(source, target)
        if not valid_geom:
            return False, f"Geometry validation failed: {geom_msg}", {}

        raw_calibrated = bool(data.get("is_calibrated", False))
        provenance = str(data.get("provenance", "unknown_upload")).lower()
        status = str(data.get("status", "")).lower()

        # Integrity gate: if explicitly marked sample / illustrative / synthetic, refuse calibrated claim
        is_sample_or_illustrative = (
            "sample" in provenance
            or "illustrative" in provenance
            or "example" in provenance
            or status in ("illustrative", "uncalibrated")
            or not raw_calibrated
        )

        if is_sample_or_illustrative:
            final_is_calibrated = False
            final_status = "illustrative"
            validation_msg = "Illustrative example calibration (approximate speeds, unverified physical scale)"
        else:
            final_is_calibrated = True
            final_status = "calibrated"
            validation_msg = "Genuine measured calibration validated"

        validated_info = {
            "source": source,
            "target": target,
            "is_calibrated": final_is_calibrated,
            "status": final_status,
            "provenance": data.get("provenance", "user_measured"),
            "validation_message": validation_msg,
            "description": data.get("description", ""),
        }
        return True, validation_msg, validated_info

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """points: (N, 2) pixel coords -> (N, 2) real-world meter coords."""
        if points.size == 0:
            return points.reshape(-1, 2)
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)

    def to_json(self, path: str) -> None:
        """Save this calibration so it can be reloaded for the same
        camera/video without re-clicking points."""
        data = {
            "source_points": self.source.tolist(),
            "target_points_meters": self.target.tolist(),
            "is_calibrated": self.is_calibrated,
            "status": self.status,
            "provenance": self.provenance,
            "validation_message": self.validation_message,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: str) -> "PerspectiveTransformer":
        """Load and validate calibration previously saved by `tools/calibrate.py` or JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid, msg, validated = cls.validate_calibration_dict(data)
        if not valid:
            # Fall back to uncalibrated example with validation error explanation
            fallback = example_calibration()
            fallback.status = "invalid"
            fallback.validation_message = msg
            return fallback

        return cls(
            source=validated["source"],
            target=validated["target"],
            is_calibrated=validated["is_calibrated"],
            status=validated["status"],
            provenance=validated["provenance"],
            validation_message=validated["validation_message"],
        )


def example_calibration() -> PerspectiveTransformer:
    """
    Placeholder illustrative calibration for a 960x540 UA-DETRAC-style frame.
    Explicitly marked with is_calibrated=False and status='illustrative' so the
    system clearly informs operators that speed numbers are illustrative approximations.
    """
    source = np.array([
        [330, 250],   # top-left of road segment
        [630, 250],   # top-right
        [900, 500],   # bottom-right
        [50, 500],    # bottom-left
    ], dtype=np.float32)
    # Example: this quadrilateral represents a nominal 10m-wide, 40m-long road segment
    target = np.array([
        [0, 0],
        [10, 0],
        [10, 40],
        [0, 40],
    ], dtype=np.float32)
    return PerspectiveTransformer(
        source,
        target,
        is_calibrated=False,
        status="illustrative",
        
        provenance="illustrative_example",
        validation_message="Default illustrative perspective geometry (unverified scale; speeds are approximate)",
    )
