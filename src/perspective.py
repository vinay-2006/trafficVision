"""
Perspective transform: maps pixel coordinates in the camera view to
real-world ground-plane coordinates (meters), so pixel displacement
between frames can be converted into an actual speed.

Calibration is per-camera/per-video — there is no way to derive real-world
scale automatically from video alone without extra assumptions (known
average vehicle size, detected lane markings, etc.), and those approaches
are approximate at best. Instead, calibration is a one-time setup step:
run `tools/calibrate.py` once per video/camera to click 4 points and enter
their real-world distance, producing a reusable JSON file. Load it here
instead of hardcoding values in source.
"""
import json
from pathlib import Path

import cv2
import numpy as np


class PerspectiveTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray, is_calibrated: bool = True):
        """
        Args:
            source: (4, 2) array of pixel coords, road quadrilateral,
                ordered top-left, top-right, bottom-right, bottom-left.
            target: (4, 2) array of real-world coords in meters for the
                same 4 points, same ordering (a rectangle).
            is_calibrated: whether these coordinates represent measured ground truth
                versus an uncalibrated default / placeholder.
        """
        self.source = np.asarray(source, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.is_calibrated = is_calibrated
        self.m = cv2.getPerspectiveTransform(self.source, self.target)

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
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: str) -> "PerspectiveTransformer":
        """Load a calibration previously saved by `tools/calibrate.py`
        or `to_json()`."""
        data = json.loads(Path(path).read_text())
        source = np.array(data["source_points"], dtype=np.float32)
        target = np.array(data["target_points_meters"], dtype=np.float32)
        is_calibrated = data.get("is_calibrated", True)
        return cls(source, target, is_calibrated=is_calibrated)


def example_calibration() -> PerspectiveTransformer:
    """
    Placeholder calibration for a 960x540 UA-DETRAC-style frame.
    Marked with is_calibrated=False so the system alerts users that
    real-world measurements must be provided for certified speed.
    """
    source = np.array([
        [330, 250],   # top-left of road segment
        [630, 250],   # top-right
        [900, 500],   # bottom-right
        [50, 500],    # bottom-left
    ])
    # Example: this quadrilateral represents a 10m-wide, 40m-long road segment
    target = np.array([
        [0, 0],
        [10, 0],
        [10, 40],
        [0, 40],
    ])
    return PerspectiveTransformer(source, target, is_calibrated=False)
