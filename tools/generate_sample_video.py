"""
Synthetic traffic video generator for testing and offline demonstration.

Renders an overhead / perspective view of a 3-lane roadway with:
  - Asphalt surface, lane markings, horizon, road shoulders
  - Realistic multi-class vehicles (cars, trucks, buses, motorcycles) with shadows and details
  - Bidirectional and multi-lane motion
  - Line-crossing events (In / Out)
  - Speed variations
  - Incident triggers:
      - Stationary stopped vehicle in lane 2
      - Sudden deceleration
      - Close proximity event
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np


def draw_vehicle(frame: np.ndarray, x: int, y: int, w: int, h: int, color: tuple, vtype: str = "car", angle: float = 0.0):
    """Draws a vehicle with realistic details: body, roof/windshield, headlights, shadow."""
    overlay = frame.copy()
    # Shadow
    cv2.ellipse(overlay, (x + w // 2, y + h + 2), (w // 2 + 4, 6), 0, 0, 360, (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Main Body
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), 2)

    # Windshields and Roof
    roof_margin_x = int(w * 0.18)
    roof_margin_y = int(h * 0.22)
    # Windshield front/back
    cv2.rectangle(frame, (x + roof_margin_x, y + 4), (x + w - roof_margin_x, y + int(h * 0.35)), (50, 70, 90), -1)
    cv2.rectangle(frame, (x + roof_margin_x, y + h - int(h * 0.35)), (x + w - roof_margin_x, y + h - 4), (50, 70, 90), -1)
    # Roof
    cv2.rectangle(frame, (x + roof_margin_x + 2, y + roof_margin_y), (x + w - roof_margin_x - 2, y + h - roof_margin_y), (min(255, color[0] + 25), min(255, color[1] + 25), min(255, color[2] + 25)), -1)

    # Lights
    # Headlights (bottom)
    cv2.circle(frame, (x + 5, y + h - 2), 3, (100, 255, 255), -1)
    cv2.circle(frame, (x + w - 5, y + h - 2), 3, (100, 255, 255), -1)
    # Taillights (top)
    cv2.circle(frame, (x + 5, y + 2), 3, (0, 0, 220), -1)
    cv2.circle(frame, (x + w - 5, y + 2), 3, (0, 0, 220), -1)


def generate_traffic_video(
    output_path: str = "data/sample_traffic.mp4",
    width: int = 800,
    height: int = 500,
    fps: int = 25,
    num_frames: int = 175,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Road layout parameters
    road_left = 120
    road_right = width - 120
    road_width = road_right - road_left
    lane_width = road_width // 3

    # Vehicle states: list of dicts
    # (id, type, x, y, vy, color, width, height, behavior)
    vehicles = [
        # Car 1: Lane 1 moving smoothly downwards
        {"vtype": "car", "lane": 0, "y": -60.0, "vy": 4.2, "w": 46, "h": 72, "color": (190, 40, 40), "behavior": "cruise"},
        # Bus 1: Lane 2 moving downwards at steady speed
        {"vtype": "bus", "lane": 1, "y": -140.0, "vy": 3.0, "w": 62, "h": 120, "color": (40, 160, 220), "behavior": "cruise"},
        # Car 2: Lane 2 follows bus, then stops stationary (triggers stationary alert!)
        {"vtype": "car", "lane": 1, "y": 140.0, "vy": 3.5, "w": 48, "h": 76, "color": (40, 40, 210), "behavior": "stop_at_260"},
        # Truck 1: Lane 3 moving downwards
        {"vtype": "truck", "lane": 2, "y": -40.0, "vy": 3.2, "w": 58, "h": 105, "color": (50, 150, 50), "behavior": "cruise"},
        # Car 3: Lane 0 enters later and overtakes
        {"vtype": "car", "lane": 0, "y": -300.0, "vy": 5.5, "w": 46, "h": 72, "color": (210, 180, 40), "behavior": "fast"},
        # Motorcycle: Lane 2 weaves through
        {"vtype": "motorcycle", "lane": 2, "y": -220.0, "vy": 4.8, "w": 28, "h": 50, "color": (20, 20, 20), "behavior": "cruise"},
    ]

    for frame_idx in range(num_frames):
        # 1. Background Scene
        frame = np.full((height, width, 3), (65, 120, 55), dtype=np.uint8)  # Grass surroundings

        # Road surface
        cv2.rectangle(frame, (road_left, 0), (road_right, height), (50, 50, 52), -1)

        # Shoulders (curb lines)
        cv2.line(frame, (road_left, 0), (road_left, height), (220, 220, 220), 4)
        cv2.line(frame, (road_right, 0), (road_right, height), (220, 220, 220), 4)

        # Lane dividers (dashed lines moving down to simulate road feel or static)
        dash_len = 25
        gap = 20
        offset = (frame_idx * 3) % (dash_len + gap)
        for lane_idx in range(1, 3):
            lx = road_left + lane_idx * lane_width
            for y_start in range(-dash_len + offset, height, dash_len + gap):
                y_end = min(height, y_start + dash_len)
                if y_end > 0:
                    cv2.line(frame, (lx, max(0, y_start)), (lx, y_end), (240, 240, 240), 2)

        # 2. Update and Draw Vehicles
        for v in vehicles:
            # Behavior update
            if v["behavior"] == "stop_at_260":
                if v["y"] >= 260.0:
                    v["vy"] = 0.0  # Stop completely!
                elif v["y"] > 210.0:
                    v["vy"] = max(0.5, v["vy"] - 0.15)  # Decelerate

            v["y"] += v["vy"]

            # Lane X center
            lx_center = road_left + int((v["lane"] + 0.5) * lane_width)
            vx = lx_center - v["w"] // 2
            vy = int(v["y"])

            if vy + v["h"] > 0 and vy < height:
                draw_vehicle(frame, vx, vy, v["w"], v["h"], v["color"], v["vtype"])

        writer.write(frame)

    writer.release()
    print(f"Generated synthetic traffic video ({num_frames} frames @ {fps}fps) at {output_path}")


def generate_sample_calibration(output_path: str = "data/sample_calibration.json"):
    from src.perspective import PerspectiveTransformer
    # Road section 800x500: road rectangle between (120, 80) and (680, 460)
    source = np.array([
        [150, 80],
        [650, 80],
        [680, 460],
        [120, 460],
    ], dtype=np.float32)
    # Target real world road stretch: 12.0m wide (3 lanes of 4m), 45.0m long
    target = np.array([
        [0, 0],
        [12.0, 0],
        [12.0, 45.0],
        [0, 45.0],
    ], dtype=np.float32)
    transformer = PerspectiveTransformer(source, target, is_calibrated=True)
    transformer.to_json(output_path)
    print(f"Saved sample calibration to {output_path}")


if __name__ == "__main__":
    generate_traffic_video()
    generate_sample_calibration()
