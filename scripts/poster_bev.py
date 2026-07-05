"""
Geometric BEV (Bird's Eye View) projection from surround-view cameras.

Takes the 6 camera images and projects them onto a top-down ground plane
using the camera intrinsic and extrinsic parameters.

This is a GEOMETRIC projection (homography-based), not a learned one.
It assumes a flat ground plane (z=0), which works well for road surfaces
but distorts tall objects like buildings and vehicles.
"""

import json
import cv2
import numpy as np
from pathlib import Path


def build_projection_matrix(intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """
    Build the 3x3 homography matrix that maps ground plane points to image pixels.

    The math:
        A 3D point on the ground plane has z=0, so [X, Y, 0, 1].
        The full projection is: pixel = K @ [R|t] @ [X, Y, 0, 1]^T
        Since Z=0, we can drop the 3rd column of the rotation matrix,
        giving us a 3x3 homography: pixel = H @ [X, Y, 1]^T

    Args:
        intrinsic: 3x3 camera intrinsic matrix K.
        extrinsic: 4x4 camera-to-vehicle extrinsic matrix.

    Returns:
        3x3 homography matrix mapping ground (X,Y) to image (u,v).
    """
    # Invert extrinsic: we need vehicle-to-camera, not camera-to-vehicle
    extrinsic_inv = np.linalg.inv(extrinsic)

    # Extract rotation and translation
    R = extrinsic_inv[:3, :3]
    t = extrinsic_inv[:3, 3]

    # Drop 3rd column of R (because Z=0 on ground plane)
    # and append translation to form 3x3 matrix
    Rt = np.column_stack([R[:, 0], R[:, 1], t])

    # Homography: H = K @ Rt
    H = intrinsic @ Rt

    return H


def create_bev_image(
    images: dict,
    intrinsics: dict,
    extrinsics: dict,
    bev_size: int = 800,
    meters_per_pixel: float = 0.1,
) -> np.ndarray:
    """
    Project all camera images onto a top-down BEV plane.

    Args:
        images: Dict of camera name -> BGR image array.
        intrinsics: Dict of camera name -> 3x3 intrinsic matrix.
        extrinsics: Dict of camera name -> 4x4 extrinsic matrix.
        bev_size: Output BEV image size in pixels (square).
        meters_per_pixel: Scale — 0.1 means each pixel = 10cm.

    Returns:
        BEV image as numpy array.
    """
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    # Center of BEV image = ego vehicle position
    cx = bev_size // 2
    cy = bev_size // 2

    for cam_name, img in images.items():
        K = np.array(intrinsics[cam_name])
        E = np.array(extrinsics[cam_name])

        H = build_projection_matrix(K, E)

        # For each pixel in BEV, find corresponding pixel in camera image
        # We iterate over BEV pixels and compute where they map in the image
        h, w = img.shape[:2]

        for bev_y in range(bev_size):
            for bev_x in range(bev_size):
                # Convert BEV pixel to world coordinates (meters)
                # BEV center = vehicle position = (0, 0)
                world_x = (bev_x - cx) * meters_per_pixel
                world_y = (bev_y - cy) * meters_per_pixel

                # Project world point to image pixel using homography
                world_pt = np.array([world_x, world_y, 1.0])
                img_pt = H @ world_pt

                # Normalize homogeneous coordinates
                if abs(img_pt[2]) < 1e-6:
                    continue
                u = int(img_pt[0] / img_pt[2])
                v = int(img_pt[1] / img_pt[2])

                # Check if the projected point is within the image
                if 0 <= u < w and 0 <= v < h:
                    # Only paint if this BEV pixel is still black (first camera wins)
                    if np.sum(bev[bev_y, bev_x]) == 0:
                        bev[bev_y, bev_x] = img[v, u]

    return bev


def create_bev_fast(
    images: dict,
    intrinsics: dict,
    extrinsics: dict,
    bev_size: int = 800,
    meters_per_pixel: float = 0.1,
) -> np.ndarray:
    """
    Fast vectorized BEV projection using OpenCV warpPerspective.

    Instead of looping pixel by pixel (slow), we compute the inverse
    homography and let OpenCV do the warping in C++ (fast).
    """
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    cx = bev_size // 2
    cy = bev_size // 2

    # Matrix that converts BEV pixel coords to world coords
    # bev_pixel -> world_meters
    bev_to_world = np.array([
        [meters_per_pixel, 0, -cx * meters_per_pixel],
        [0, meters_per_pixel, -cy * meters_per_pixel],
        [0, 0, 1],
    ])

    for cam_name, img in images.items():
        K = np.array(intrinsics[cam_name])
        E = np.array(extrinsics[cam_name])

        H = build_projection_matrix(K, E)

        # Combined transform: BEV pixel -> world -> image pixel
        H_combined = H @ bev_to_world

        # Warp the camera image into BEV space
        # WARP_INVERSE_MAP means: for each output pixel, find the input pixel
        warped = cv2.warpPerspective(
            img, H_combined, (bev_size, bev_size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Blend: where BEV is still black, use the warped image
        mask = np.all(bev == 0, axis=2)
        bev[mask] = warped[mask]

    return bev


def create_poster_bev(data_dir: Path, frame_idx: int, output_path: Path) -> None:
    """Create a combined poster image: BEV in center, cameras around it."""

    # Load metadata
    meta_path = data_dir / "metadata" / f"{frame_idx:06d}.json"
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Load images
    cameras = ["front", "front_left", "front_right", "left", "right", "rear"]
    images = {}
    for cam in cameras:
        img_path = data_dir / cam / f"{frame_idx:06d}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: Could not load {img_path}")
            return
        images[cam] = img

    print(f"Generating BEV for frame {frame_idx}...")

    # Generate BEV using the fast method
    bev = create_bev_fast(
        images,
        metadata["intrinsics"],
        metadata["extrinsics"],
        bev_size=600,
        meters_per_pixel=0.1,  # each pixel = 10cm, total = 60m x 60m area
    )

    # Draw ego vehicle marker at center of BEV
    bev_cx, bev_cy = 300, 300
    cv2.rectangle(bev, (bev_cx - 8, bev_cy - 15), (bev_cx + 8, bev_cy + 15), (0, 200, 255), 2)
    cv2.arrowedLine(bev, (bev_cx, bev_cy), (bev_cx, bev_cy - 30), (0, 200, 255), 2, tipLength=0.4)

    # Add label
    cv2.putText(bev, "BEV PROJECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    ego = metadata["ego_pose"]
    info = f"Ego: ({ego['x']:.1f}, {ego['y']:.1f}) | Speed: {ego['speed']:.1f} m/s"
    cv2.putText(bev, info, (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Save just the BEV
    cv2.imwrite(str(output_path), bev)
    print(f"Saved: {output_path}")

    # Also create a combined view: front camera + BEV side by side
    front_resized = cv2.resize(images["front"], (600, 450))
    cv2.putText(front_resized, "FRONT CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    combined = np.hstack([front_resized, cv2.resize(bev, (600, 450))])
    combined_path = output_path.parent / f"combined_{frame_idx:06d}.png"
    cv2.imwrite(str(combined_path), combined)
    print(f"Saved: {combined_path}")


if __name__ == "__main__":
    data_dir = Path("data/raw/town10")
    output_dir = Path("scripts/poster_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate BEV for frames where the car was moving
    for frame_idx in [0, 25, 50, 75]:
        create_poster_bev(data_dir, frame_idx, output_dir / f"bev_{frame_idx:06d}.png")
