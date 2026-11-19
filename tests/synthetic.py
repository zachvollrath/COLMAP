"""Synthetic calibrated scene used by the reconstruction tests."""

from __future__ import annotations

import math

import torch

from colmap.geometry import DTYPE
from colmap.scene_graph import Camera, ImageNode


def look_at(center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = center.to(dtype=DTYPE)
    forward = -center / torch.linalg.norm(center)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    right = torch.linalg.cross(up, forward)
    right = right / torch.linalg.norm(right)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward])
    translation = -rotation @ center
    return rotation, translation


def make_scene(
    n_cameras: int = 5,
    n_points: int = 40,
    radius: float = 4.0,
    noise_px: float = 0.25,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
    focal: float = 700.0,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    world = (torch.rand((n_points, 3), generator=rng, dtype=DTYPE) - 0.5)
    cameras = {}
    images = {}
    poses = {}
    for index in range(n_cameras):
        theta = torch.tensor(2.0 * math.pi * index / n_cameras, dtype=DTYPE)
        center = torch.stack(
            [
                radius * torch.cos(theta),
                0.2 * torch.sin(2 * theta),
                radius * torch.sin(theta),
            ]
        )
        rotation, translation = look_at(center)
        poses[index + 1] = (rotation, translation, center)
        cameras[index + 1] = Camera(index + 1, width, height, focal, 0.0, focal)
        keypoints = []
        for point in world:
            xc = rotation @ point + translation
            if float(xc[2]) <= 0.1:
                keypoints.append(torch.tensor([-1.0, -1.0], dtype=DTYPE))
                continue
            u = focal * xc[0] / xc[2] + width / 2.0
            v = focal * xc[1] / xc[2] + height / 2.0
            if 20 <= float(u) < width - 20 and 20 <= float(v) < height - 20:
                noisy = torch.stack([u, v]) + torch.randn(2, generator=rng, dtype=DTYPE) * noise_px
                keypoints.append(noisy)
            else:
                keypoints.append(torch.tensor([-1.0, -1.0], dtype=DTYPE))
        images[index + 1] = ImageNode(
            index + 1,
            f"view_{index + 1:02d}.png",
            index + 1,
            torch.stack(keypoints),
        )
    return world, cameras, images, poses


def correspondences(images: dict[int, ImageNode], min_matches: int) -> dict[tuple[int, int], torch.Tensor]:
    matches = {}
    ids = sorted(images)
    for i, id1 in enumerate(ids):
        for id2 in ids[i + 1 :]:
            kp1 = images[id1].keypoints
            kp2 = images[id2].keypoints
            rows = [
                feature_idx
                for feature_idx, (xy1, xy2) in enumerate(zip(kp1, kp2))
                if float(xy1[0]) >= 0 and float(xy2[0]) >= 0
            ]
            if len(rows) >= min_matches:
                index = torch.tensor(rows, dtype=torch.long)
                matches[(id1, id2)] = torch.column_stack([index, index])
    return matches
