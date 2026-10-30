"""
Perspective transform: maps pixel coordinates in the camera view to
real-world ground-plane coordinates (meters), so pixel displacement
between frames can be converted into an actual speed.

Calibration integrity rules:
- Calibration is per-camera/per-video.
- Bundled sample calibrations are ILLUSTRATIVE/EXAMPLES, never certified ground truth.
- Uploaded calibrations must be strictly validated for finite coordinates,
  non-degenerate quadrilateral geometry, positive dimensions, and non-singular matrix.
- A JSON field containing `is_calibrated: true` is NEVER trusted blindly without
  provenance verification and geometric validation.
- Calibration status is a formal 4-state enum — not a free-text field.

Calibration States (CalibrationStatus enum):
  ILLUSTRATIVE   – example / demo / placeholder; never considered calibrated.
  UNCALIBRATED   – missing or insufficient calibration information.
  USER_ATTESTED  – user explicitly confirms measurements were obtained from the
                   actual camera scene; valid geometry required; NOT independently verified.
  VERIFIED       – reserved for a genuine independent verification workflow;
                   NOT assigned merely because JSON contains is_calibrated=true.
"""
import json
import math
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class CalibrationStatus(Enum):
    """Formal 4-state calibration provenance enum.

    ILLUSTRATIVE  – example / demo calibration; speeds are approximate only.
    UNCALIBRATED  – missing or insufficient calibration information.
    USER_ATTESTED – user has confirmed measurements from the actual scene; not independently verified.
    VERIFIED      – independently verified; only assign via an explicit verification workflow.
    """
    ILLUSTRATIVE = "illustrative"
    UNCALIBRATED = "uncalibrated"
    USER_ATTESTED = "user_attested"
    VERIFIED = "verified"


# ---------------------------------------------------------------------------
# Strict provenance allowlists
# ---------------------------------------------------------------------------
_ILLUSTRATIVE_PROVENANCES = frozenset({
    "illustrative_sample",
    "illustrative_example",
    "example",
    "demo",
    "placeholder",
    "synthetic_test",
    "unknown_upload",
    "default_placeholder",
})

_USER_ATTESTED_PROVENANCES = frozenset({
    "user_measured",
    "field_survey_measured",
    "measured_user",
    "user_attested",
})

# VERIFIED is intentionally not reachable through the JSON upload path.
# It requires an out-of-band verification token that this version does not implement.


def _resolve_calibration_status(
    raw_is_calibrated: bool,
    provenance: str,
    json_status: str,
) -> CalibrationStatus:
    """Map raw JSON fields to a formal CalibrationStatus, applying the integrity rules."""
    prov_lower = provenance.strip().lower()
    stat_lower = json_status.strip().lower()

    # Any illustrative / sample / demo provenance → always ILLUSTRATIVE regardless of flags
    if prov_lower in _ILLUSTRATIVE_PROVENANCES or stat_lower in ("illustrative", "uncalibrated"):
        return CalibrationStatus.ILLUSTRATIVE

    # "verified" status in JSON is rejected — cannot be self-asserted
    if stat_lower == "verified":
        raise ValueError(
            f"Calibration status 'verified' cannot be self-asserted via JSON upload. "
            f"Independent verification is required. Got provenance='{provenance}', status='{json_status}'."
        )

    if not raw_is_calibrated:
        return CalibrationStatus.UNCALIBRATED

    # is_calibrated=true path: provenance MUST be in the user-attested allowlist.
    # Status alone is not sufficient; provenance is the authoritative attestation field.
    if prov_lower in _USER_ATTESTED_PROVENANCES:
        return CalibrationStatus.USER_ATTESTED

    # Unknown or missing provenance with is_calibrated=true → reject
    if not prov_lower:
        raise ValueError(
            "Calibration JSON claims is_calibrated=true but is missing 'provenance'. "
            "Provide provenance='user_measured' or similar to attest your measurement."
        )

    raise ValueError(
        f"Unknown provenance '{provenance}' with is_calibrated=true. "
        f"Accepted provenances for user-attested calibration: {sorted(_USER_ATTESTED_PROVENANCES)}. "
        f"Accepted provenances for illustrative calibration: {sorted(_ILLUSTRATIVE_PROVENANCES)}."
    )


def _resolve_calibration_status_from_legacy(
    is_calibrated: bool,
    status_str: str,
    provenance: str,
) -> CalibrationStatus:
    """Map legacy (is_calibrated, status, provenance) strings to CalibrationStatus enum.
    Used by PerspectiveTransformer.__init__ for backward compatibility."""
    prov_lower = provenance.strip().lower()
    stat_lower = status_str.strip().lower()

    if prov_lower in _ILLUSTRATIVE_PROVENANCES or stat_lower in ("illustrative", "uncalibrated", "invalid"):
        return CalibrationStatus.ILLUSTRATIVE

    if not is_calibrated:
        return CalibrationStatus.UNCALIBRATED

    # For legacy fields: provenance is the authoritative field.
    # "calibrated" status alone is not accepted; provenance must be in the allowlist.
    if prov_lower in _USER_ATTESTED_PROVENANCES:
        return CalibrationStatus.USER_ATTESTED

    # Conservative fallback for unknown legacy provenances
    return CalibrationStatus.ILLUSTRATIVE


class PerspectiveTransformer:
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        is_calibrated: bool = False,
        status: str = "illustrative",
        provenance: str = "default_placeholder",
        validation_message: str = "Uncalibrated example geometry",
        calibration_status: Optional[CalibrationStatus] = None,
    ):
        """
        Args:
            source: (4, 2) array of pixel coords, road quadrilateral,
                ordered top-left, top-right, bottom-right, bottom-left.
            target: (4, 2) array of real-world coords in meters for the
                same 4 points, same ordering (a rectangle).
            is_calibrated: whether these coordinates represent certified measured ground truth.
            status: legacy string status field (kept for backward compatibility).
            provenance: descriptive source of calibration data.
            validation_message: diagnostic explanation of calibration state.
            calibration_status: formal CalibrationStatus enum. When provided, takes
                precedence over the legacy ``status`` string.
        """
        self.source = np.asarray(source, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

        # Validate geometry on initialization
        valid, msg = self.validate_geometry(self.source, self.target)
        if not valid:
            self.is_calibrated = False
            self.status = "invalid"
            self.calibration_status = CalibrationStatus.UNCALIBRATED
            self.validation_message = msg
            self.provenance = provenance
            # Fall back to identity-like safe transform
            self.m = np.eye(3, dtype=np.float32)
        else:
            # Resolve formal enum status
            if calibration_status is not None:
                self.calibration_status = calibration_status
            else:
                # Derive from legacy fields for backward compatibility
                self.calibration_status = _resolve_calibration_status_from_legacy(
                    is_calibrated=bool(is_calibrated),
                    status_str=status,
                    provenance=provenance,
                )
            self.is_calibrated = self.calibration_status in (
                CalibrationStatus.USER_ATTESTED,
                CalibrationStatus.VERIFIED,
            )
            self.status = self.calibration_status.value
            self.provenance = provenance
            self.validation_message = validation_message if self.is_calibrated else (
                msg if not self.is_calibrated else "Valid geometry"
            )
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
        Inspects calibration dictionary from JSON upload and returns a formal CalibrationStatus.

        Enforcement rules (in priority order):
        1. Geometry must be valid (finite, non-degenerate, realistic dimensions).
        2. Provenance and status fields must come from the strict allowlists.
        3. ``is_calibrated=true`` + illustrative/sample provenance → demoted to ILLUSTRATIVE.
        4. ``is_calibrated=true`` + unknown provenance → rejected with a clear error.
        5. ``is_calibrated=true`` + missing provenance → rejected with a clear error.
        6. ``status='verified'`` in JSON → rejected; cannot be self-asserted.
        7. ``is_calibrated=true`` + recognised user provenance → USER_ATTESTED (not VERIFIED).

        Returns:
            (ok: bool, message: str, validated_info: dict)
            validated_info contains ``calibration_status`` (CalibrationStatus enum).
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
        raw_provenance = str(data.get("provenance", "unknown_upload"))
        raw_status = str(data.get("status", ""))

        # Apply the strict 4-state provenance resolution
        try:
            cal_status = _resolve_calibration_status(
                raw_is_calibrated=raw_calibrated,
                provenance=raw_provenance,
                json_status=raw_status,
            )
        except ValueError as e:
            return False, str(e), {}

        final_is_calibrated = cal_status in (
            CalibrationStatus.USER_ATTESTED,
            CalibrationStatus.VERIFIED,
        )

        if cal_status == CalibrationStatus.ILLUSTRATIVE:
            validation_msg = "Illustrative example calibration (approximate speeds, unverified physical scale)"
        elif cal_status == CalibrationStatus.UNCALIBRATED:
            validation_msg = "Uncalibrated — speed estimation unavailable"
        elif cal_status == CalibrationStatus.USER_ATTESTED:
            validation_msg = "User-attested calibration: measurements supplied by operator (not independently verified)"
        else:
            validation_msg = "Verified calibration"

        validated_info = {
            "source": source,
            "target": target,
            "is_calibrated": final_is_calibrated,
            "status": cal_status.value,
            "calibration_status": cal_status,
            "provenance": raw_provenance,
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
        cs = getattr(self, "calibration_status", None)
        data = {
            "source_points": self.source.tolist(),
            "target_points_meters": self.target.tolist(),
            "is_calibrated": self.is_calibrated,
            "status": cs.value if isinstance(cs, CalibrationStatus) else self.status,
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
            fallback.calibration_status = CalibrationStatus.UNCALIBRATED
            fallback.validation_message = msg
            return fallback

        return cls(
            source=validated["source"],
            target=validated["target"],
            is_calibrated=validated["is_calibrated"],
            status=validated["status"],
            provenance=validated["provenance"],
            validation_message=validated["validation_message"],
            calibration_status=validated.get("calibration_status"),
        )


def example_calibration() -> PerspectiveTransformer:
    """
    Placeholder illustrative calibration for a 960x540 UA-DETRAC-style frame.
    Explicitly marked with is_calibrated=False and calibration_status=ILLUSTRATIVE so the
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
        calibration_status=CalibrationStatus.ILLUSTRATIVE,
    )
