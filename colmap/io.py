"""COLMAP text-model writer (cameras.txt, images.txt, points3D.txt)."""

from __future__ import annotations

from pathlib import Path

import torch

from colmap.geometry import DTYPE, matrix_to_quaternion
from colmap.mapper import Reconstruction


def _num(value) -> str:
    return f"{float(value):.16g}"


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
            f"{_num(camera.focal)} {_num(camera.cx)} {_num(camera.cy)} {_num(camera.k1)}"
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
            f"{image_id} {_num(qw)} {_num(qx)} {_num(qy)} {_num(qz)} "
            f"{_num(t[0])} {_num(t[1])} {_num(t[2])} {image.camera_id} {image.name}"
        )
        parts = []
        for feature_idx, xy in enumerate(image.keypoints):
            point_id = feature_point.get((image_id, feature_idx), -1)
            parts.append(f"{_num(xy[0])} {_num(xy[1])} {point_id}")
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
        lines.append(f"{point_id} {_num(x)} {_num(y)} {_num(z)} {r} {g} {b} {_num(point.error)} {track}")
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
        body.append(f"{_num(x)} {_num(y)} {_num(z)} {r} {g} {b}")
    Path(path).write_text(header + "\n".join(body) + "\n")


def camera_centers(reconstruction: Reconstruction) -> torch.Tensor:
    from colmap.geometry import camera_center

    centers = []
    for image_id in reconstruction.registration_order:
        R, t = reconstruction.poses[image_id]
        centers.append(camera_center(R, t))
    if not centers:
        return torch.zeros((0, 3), dtype=DTYPE)
    return torch.stack(centers)
