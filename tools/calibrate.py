"""
One-time interactive calibration tool.

Extracts the first frame of a video, lets you click 4 points on the road
(forming a rectangle in real life — e.g. lane edges over a measured
stretch), asks for the real-world width/length in meters, and saves the
result as a reusable JSON calibration file.

This replaces hand-editing pixel coordinates in source code. Run it once
per camera/video angle; the resulting file is passed to the pipeline via
`--calibration path/to/calibration.json`.

Usage:
    python tools/calibrate.py --video path/to/video.mp4 --output calibration.json

Controls:
    - Click 4 points on the displayed frame, in this exact order:
        1. top-left corner of your reference rectangle
        2. top-right corner
        3. bottom-right corner
        4. bottom-left corner
    - Press 'r' to reset and re-click if you make a mistake
    - Press 'q' once all 4 points look correct, to confirm and continue
    - Then enter the real-world width and length in meters when prompted
"""
import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from src.perspective import PerspectiveTransformer  # noqa: E402


class ClickCollector:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.display = frame.copy()
        self.points = []

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            self._redraw()

    def _redraw(self):
        self.display = self.frame.copy()
        labels = ["1 (top-left)", "2 (top-right)", "3 (bottom-right)", "4 (bottom-left)"]
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(self.display, labels[i], (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(self.display, [pts], isClosed=(len(self.points) == 4),
                          color=(0, 255, 0), thickness=2)

    def reset(self):
        self.points = []
        self.display = self.frame.copy()


def collect_points(frame: np.ndarray) -> np.ndarray:
    collector = ClickCollector(frame)
    window = "Calibration - click 4 points (r=reset, q=confirm)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, collector.on_click)

    while True:
        cv2.imshow(window, collector.display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            collector.reset()
        elif key == ord("q"):
            if len(collector.points) == 4:
                break
            print(f"Need exactly 4 points, you have {len(collector.points)}. Keep clicking or press 'r' to reset.")

    cv2.destroyAllWindows()
    return np.array(collector.points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Interactive perspective calibration")
    parser.add_argument("--video", required=True, help="path to a representative video")
    parser.add_argument("--output", default="calibration.json", help="where to save the calibration file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.video}")

    print("A window will open showing the first frame.")
    print("Click 4 points forming a rectangle on the road, in order:")
    print("  1) top-left  2) top-right  3) bottom-right  4) bottom-left")
    print("Press 'r' to reset, 'q' when done.\n")

    source_points = collect_points(frame)

    width_m = float(input("Real-world WIDTH of that rectangle, in meters (e.g. lane width): "))
    length_m = float(input("Real-world LENGTH of that rectangle, in meters (e.g. distance between the two marked lines): "))

    if width_m <= 0.2 or length_m <= 0.2:
        raise ValueError(f"Rectangle dimensions must be positive physical numbers (width={width_m}m, length={length_m}m)")

    target_points = np.array([
        [0, 0],
        [width_m, 0],
        [width_m, length_m],
        [0, length_m],
    ], dtype=np.float32)

    transformer = PerspectiveTransformer(
        source_points,
        target_points,
        is_calibrated=True,
        status="calibrated",
        provenance="measured_user",
        validation_message=f"Measured ground truth: width={width_m:.2f}m, length={length_m:.2f}m",
    )
    transformer.to_json(args.output)

    print(f"\nSaved calibration to {args.output}")
    print(f"Source pixel points: {source_points.tolist()}")
    print(f"Target real-world rectangle: {width_m}m x {length_m}m")
    print(f"\nUse it with: python -m src.pipeline --source your_video.mp4 --calibration {args.output}")


if __name__ == "__main__":
    main()
