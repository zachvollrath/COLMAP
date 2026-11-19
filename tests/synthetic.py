"""Synthetic calibrated scene used by the reconstruction tests."""

from __future__ import annotations

import numpy as np

from colmap.scene_graph import Camera, ImageNode


def look_at(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = -center / np.linalg.norm(center)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
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
    rng = np.random.default_rng(seed)
    world = rng.uniform(-0.5, 0.5, size=(n_points, 3))
    cameras = {}
    images = {}
    poses = {}
    for index in range(n_cameras):
        theta = 2.0 * np.pi * index / n_cameras
        center = np.array([radius * np.cos(theta), 0.2 * np.sin(2 * theta), radius * np.sin(theta)])
        rotation, translation = look_at(center)
        poses[index + 1] = (rotation, translation, center)
        cameras[index + 1] = Camera(index + 1, width, height, focal, 0.0, focal)
        visible = []
        for point in world:
            xc = rotation @ point + translation
            if xc[2] <= 0.1:
                continue
            u = focal * xc[0] / xc[2] + width / 2.0
            v = focal * xc[1] / xc[2] + height / 2.0
            if 20 <= u < width - 20 and 20 <= v < height - 20:
                visible.append((u, v))
            else:
                visible.append(None)
        keypoints = []
        for projection in visible:
            if projection is None:
                keypoints.append(np.array([-1.0, -1.0]))
            else:
                noisy = np.asarray(projection, dtype=float) + rng.normal(0.0, noise_px, size=2)
                keypoints.append(noisy)
        images[index + 1] = ImageNode(index + 1, f"view_{index + 1:02d}.png", index + 1, np.asarray(keypoints, dtype=float))
    return world, cameras, images, poses


def correspondences(images: dict[int, ImageNode], min_matches: int) -> dict[tuple[int, int], np.ndarray]:
    matches = {}
    ids = sorted(images)
    for i, id1 in enumerate(ids):
        for id2 in ids[i + 1 :]:
            kp1 = images[id1].keypoints
            kp2 = images[id2].keypoints
            rows = [
                (feature_idx, feature_idx)
                for feature_idx, (xy1, xy2) in enumerate(zip(kp1, kp2))
                if xy1[0] >= 0 and xy2[0] >= 0
            ]
            if len(rows) >= min_matches:
                matches[(id1, id2)] = np.asarray(rows, dtype=int)
    return matches
