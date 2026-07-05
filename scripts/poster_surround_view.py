"""
Creates a poster-ready surround-view visualization showing
all 6 cameras arranged in their physical layout around the vehicle.

Layout:
        [front_left]  [front]  [front_right]
        [   left   ]  [ EGO ]  [   right   ]
                      [ rear ]
"""

import cv2
import numpy as np
from pathlib import Path


def create_surround_view(data_dir: Path, frame_idx: int, output_path: Path) -> None:
    """Create a 6-camera surround view image with labels."""

    cameras = ["front", "front_left", "front_right", "left", "right", "rear"]
    images = {}

    for cam in cameras:
        img_path = data_dir / cam / f"{frame_idx:06d}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: Could not read {img_path}")
            return
        images[cam] = img

    h, w = images["front"].shape[:2]

    # Scale down each image to fit a nice grid
    scale = 0.5
    sh, sw = int(h * scale), int(w * scale)

    for cam in cameras:
        images[cam] = cv2.resize(images[cam], (sw, sh))

    # Create canvas: 3 columns x 3 rows, with center cell for "EGO" label
    canvas_w = sw * 3 + 40  # 40px for gaps
    canvas_h = sh * 3 + 40
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # dark gray background

    # Position mapping: (row, col) for each camera
    positions = {
        "front_left":  (0, 0),
        "front":       (0, 1),
        "front_right": (0, 2),
        "left":        (1, 0),
        "right":       (1, 2),
        "rear":        (2, 1),
    }

    gap = 10  # pixels between images

    for cam, (row, col) in positions.items():
        y = row * (sh + gap) + gap
        x = col * (sw + gap) + gap
        canvas[y:y + sh, x:x + sw] = images[cam]

        # Add camera label
        label = cam.upper().replace("_", " ")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x + (sw - text_size[0]) // 2
        text_y = y + 30

        # Background rectangle for readability
        cv2.rectangle(
            canvas,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(canvas, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    # Draw "EGO VEHICLE" in the center cell
    center_y = 1 * (sh + gap) + gap
    center_x = 1 * (sw + gap) + gap
    ego_text = "EGO VEHICLE"
    font_scale_ego = 1.0
    text_size = cv2.getTextSize(ego_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_ego, 2)[0]
    text_x = center_x + (sw - text_size[0]) // 2
    text_y = center_y + (sh + text_size[1]) // 2
    cv2.putText(canvas, ego_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_ego, (255, 255, 255), 2)

    # Draw a simple car icon (rectangle) in center
    car_cx = center_x + sw // 2
    car_cy = center_y + sh // 2 - 20
    cv2.rectangle(canvas, (car_cx - 20, car_cy - 35), (car_cx + 20, car_cy + 35), (0, 200, 255), 2)
    # Arrow showing forward direction
    cv2.arrowedLine(canvas, (car_cx, car_cy - 10), (car_cx, car_cy - 50), (0, 200, 255), 2, tipLength=0.4)

    # Title
    title = f"Surround-View Camera Rig | Frame {frame_idx:06d}"
    cv2.putText(canvas, title, (gap, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imwrite(str(output_path), canvas)
    print(f"Saved: {output_path} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    data_dir = Path("data/raw/town10")
    output_dir = Path("scripts/poster_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for a few frames — pick the ones that look best for your poster
    for frame_idx in [0, 10, 20, 30, 40, 50, 60, 70]:
        create_surround_view(data_dir, frame_idx, output_dir / f"surround_{frame_idx:06d}.png")
