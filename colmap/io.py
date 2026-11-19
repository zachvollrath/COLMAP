"""COLMAP text-model writer (cameras.txt, images.txt, points3D.txt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from colmap.geometry import matrix_to_quaternion
from colmap.mapper import Reconstruction


def write_reconstruction(reconstruction: Reconstruction, output_dir: str | Path) -> None:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    _write_cameras(reconstruction, folder / "cameras.txt")
    _write_images(reconstruction, folder / "images.txt")
    _write_points(reconstruction, folder / "points3D.txt")


def _write_cameras(reconstruction: Reconstruction, path: Path) -> None:
    used = {image.camera_id for image_id, image in reconstruction.images.items() if image_id in reconstruction.poses}
    lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for camera_id in sorted(used):
        camera = reconstruction.cameras[camera_id]
        lines.append(
            f"{camera_id} SIMPLE_RADIAL {camera.width} {camera.height} "
            f"{camera.focal} {camera.cx} {camera.cy} {camera.k1}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_images(reconstruction: Reconstruction, path: Path) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    feature_point: dict[tuple[int, int], int] = {}
    for point_id, point in reconstruction.points.items():
        for obs in point.observations:
            feature_point[(obs.image_id, obs.feature_idx)] = point_id
    for image_id in reconstruction.registration_order:
        R, t = reconstruction.poses[image_id]
        image = reconstruction.images[image_id]
        qw, qx, qy, qz = matrix_to_quaternion(R)
        lines.append(
            f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {image.camera_id} {image.name}"
        )
        parts = []
        for feature_idx, xy in enumerate(image.keypoints):
            point_id = feature_point.get((image_id, feature_idx), -1)
            parts.append(f"{xy[0]} {xy[1]} {point_id}")
        lines.append(" ".join(parts))
    path.write_text("\n".join(lines) + "\n")


def _write_points(reconstruction: Reconstruction, path: Path) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for point_id, point in sorted(reconstruction.points.items()):
        r, g, b = (int(c) for c in point.color)
        track = " ".join(f"{obs.image_id} {obs.feature_idx}" for obs in point.observations)
        x, y, z = point.xyz
        lines.append(f"{point_id} {x} {y} {z} {r} {g} {b} {point.error} {track}")
    path.write_text("\n".join(lines) + "\n")


def write_ply(reconstruction: Reconstruction, path: str | Path) -> None:
    points = list(reconstruction.points.values())
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    body = []
    for point in points:
        x, y, z = point.xyz
        r, g, b = (int(c) for c in point.color)
        body.append(f"{x} {y} {z} {r} {g} {b}")
    Path(path).write_text(header + "\n".join(body) + "\n")


def camera_centers(reconstruction: Reconstruction) -> np.ndarray:
    from colmap.geometry import camera_center

    centers = []
    for image_id in reconstruction.registration_order:
        R, t = reconstruction.poses[image_id]
        centers.append(camera_center(R, t))
    return np.asarray(centers, dtype=float)
