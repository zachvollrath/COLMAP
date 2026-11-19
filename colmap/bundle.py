"""Local and global bundle adjustment, Sections 4.4 and 4.5.

The cost is the Cauchy-robust reprojection error of the simple camera.
Global problems with a dense viewing graph may collapse unaffected,
highly overlapping cameras into one rigid group (redundant view mining)
so the reduced camera system is smaller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from colmap.geometry import (
    angle_axis_to_matrix,
    camera_center,
    matrix_to_angle_axis,
    project_points,
    viewing_direction,
)
from colmap.options import MapperOptions


@dataclass
class BAImage:
    image_id: int
    camera_id: int
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int


@dataclass
class BACamera:
    camera_id: int
    focal: float
    k1: float
    prior_focal: float


@dataclass
class BAObservation:
    image_id: int
    point_id: int
    xy: np.ndarray


@dataclass
class BAResult:
    rotations: dict[int, np.ndarray]
    translations: dict[int, np.ndarray]
    points: dict[int, np.ndarray]
    focals: dict[int, float]
    distortions: dict[int, float]
    success: bool


def overlap_ratio(visible_a: set[int], visible_b: set[int]) -> float:
    """Jaccard interaction V_ab of two visibility sets (Eq. 6)."""
    if not visible_a and not visible_b:
        return 0.0
    intersection = len(visible_a & visible_b)
    union = len(visible_a | visible_b)
    return intersection / union if union else 0.0


def group_redundant_views(
    images: list[BAImage],
    visibility: dict[int, set[int]],
    affected: set[int],
    options: MapperOptions,
) -> list[list[int]]:
    """Greedy grouping of unaffected cameras, Section 4.5.

    Affected cameras (just registered, or with a large fraction of high
    reprojection errors) stay in singleton groups so their poses remain free.
    """
    groups = [[image.image_id] for image in images if image.image_id in affected]
    unaffected = [image for image in images if image.image_id not in affected]
    unaffected.sort(key=lambda image: len(visibility.get(image.image_id, ())), reverse=True)
    centers = {image.image_id: camera_center(image.R, image.t) for image in unaffected}
    directions = {image.image_id: viewing_direction(image.R) for image in unaffected}
    remaining = [image.image_id for image in unaffected]
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        while len(group) < options.group_max_size:
            pool = _neighbor_pool(
                seed,
                remaining,
                centers,
                directions,
                options.group_neighbor_count,
                options.group_max_view_angle_deg,
            )
            best_id = None
            best_overlap = options.group_min_overlap
            seed_visible = visibility.get(seed, set())
            for candidate in pool:
                ratio = overlap_ratio(seed_visible, visibility.get(candidate, set()))
                if ratio > best_overlap:
                    best_overlap = ratio
                    best_id = candidate
            if best_id is None:
                break
            group.append(best_id)
            remaining.remove(best_id)
        groups.append(group)
    return groups


def _neighbor_pool(
    seed: int,
    remaining: list[int],
    centers: dict[int, np.ndarray],
    directions: dict[int, np.ndarray],
    neighbor_count: int,
    max_angle_deg: float,
) -> list[int]:
    seed_dir = directions[seed]
    cosine_limit = np.cos(np.radians(max_angle_deg))
    ranked = []
    for candidate in remaining:
        direction = directions[candidate]
        if float(np.dot(seed_dir, direction)) < cosine_limit:
            continue
        distance = float(np.linalg.norm(centers[candidate] - centers[seed]))
        ranked.append((distance, candidate))
    ranked.sort()
    return [candidate for _, candidate in ranked[:neighbor_count]]


def _compose_group_pose(R_c: np.ndarray, t_c: np.ndarray, R_g: np.ndarray, t_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P_cr = P_c G_r, with both poses stored as world-to-camera rigid transforms."""
    R = R_c @ R_g
    t = R_c @ t_g + t_c
    return R, t


def bundle_adjust(
    images: list[BAImage],
    cameras: list[BACamera],
    points: dict[int, np.ndarray],
    observations: list[BAObservation],
    options: MapperOptions,
    groups: list[list[int]] | None = None,
    refine_intrinsics: bool = False,
    max_iterations: int = 25,
    anchor_image: int | None = None,
    fixed_images: set[int] | None = None,
) -> BAResult:
    """Minimize Eq. 1, or Eq. 7 when `groups` collapses redundant cameras.

    `anchor_image` is held fixed and the baseline to the next image is pinned,
    which removes the similarity gauge. Additional `fixed_images` keep their
    current poses (local BA uses this for cameras outside the active set).
    """
    if len(images) < 2 or len(points) < 1 or len(observations) < 4:
        return _identity_result(images, cameras, points, False)

    by_id = {image.image_id: image for image in images}
    order = [image.image_id for image in images]
    anchor = anchor_image if anchor_image in by_id else order[0]
    fixed = set(fixed_images or set())
    fixed.add(anchor)
    scale_partner = next(image_id for image_id in order if image_id != anchor)
    baseline = float(
        np.linalg.norm(
            camera_center(by_id[anchor].R, by_id[anchor].t)
            - camera_center(by_id[scale_partner].R, by_id[scale_partner].t)
        )
    )
    baseline = max(baseline, 1e-6)

    if groups is None:
        groups = [[image_id] for image_id in order]
    group_of = {}
    for index, group in enumerate(groups):
        for image_id in group:
            group_of[image_id] = index
    free_groups = []
    for index, group in enumerate(groups):
        members = group
        if any(member in fixed for member in members):
            continue
        free_groups.append(index)
    # The baseline partner has to be either fixed or free. If every camera
    # except the anchor was fixed, pose refinement would be a no-op.
    if not free_groups:
        return _identity_result(images, cameras, points, False)
    group_param_index = {group_index: slot for slot, group_index in enumerate(free_groups)}

    camera_by_id = {camera.camera_id: camera for camera in cameras}
    intrinsic_ids = []
    if refine_intrinsics:
        seen = set()
        for image in images:
            if image.camera_id not in seen:
                seen.add(image.camera_id)
                intrinsic_ids.append(image.camera_id)

    point_ids = list(points.keys())
    point_index = {point_id: i for i, point_id in enumerate(point_ids)}

    n_group = 6 * len(free_groups)
    n_intr = 2 * len(intrinsic_ids)
    n_pts = 3 * len(point_ids)
    x0 = np.zeros(n_group + n_intr + n_pts)
    for group_index, slot in group_param_index.items():
        # G starts at the identity, so the current member poses are unchanged.
        # Singleton groups store the member pose directly instead of G.
        members = groups[group_index]
        if len(members) == 1:
            image = by_id[members[0]]
            x0[6 * slot : 6 * slot + 3] = matrix_to_angle_axis(image.R)
            x0[6 * slot + 3 : 6 * slot + 6] = image.t
    for slot, camera_id in enumerate(intrinsic_ids):
        camera = camera_by_id[camera_id]
        base = n_group + 2 * slot
        x0[base] = np.log(max(camera.focal, 1e-6))
        x0[base + 1] = camera.k1 if options.refine_extra_params else 0.0
    for slot, point_id in enumerate(point_ids):
        x0[n_group + n_intr + 3 * slot : n_group + n_intr + 3 * slot + 3] = points[point_id]

    grouped_obs: list[tuple[int, np.ndarray, np.ndarray]] = []
    pending: dict[int, list[tuple[int, np.ndarray]]] = {}
    for obs in observations:
        if obs.image_id not in by_id or obs.point_id not in point_index:
            continue
        pending.setdefault(obs.image_id, []).append((point_index[obs.point_id], obs.xy))
    for image_id, entries in pending.items():
        grouped_obs.append(
            (
                image_id,
                np.asarray([entry[0] for entry in entries], dtype=int),
                np.asarray([entry[1] for entry in entries], dtype=float),
            )
        )
    n_residuals = sum(len(xy) for _, _, xy in grouped_obs)
    if n_residuals < 4:
        return _identity_result(images, cameras, points, False)

    def unpack(params: np.ndarray):
        poses = {}
        for image_id, image in by_id.items():
            group_index = group_of[image_id]
            if group_index not in group_param_index:
                poses[image_id] = (image.R, image.t)
                continue
            slot = group_param_index[group_index]
            aa = params[6 * slot : 6 * slot + 3]
            vec = params[6 * slot + 3 : 6 * slot + 6]
            R_var = angle_axis_to_matrix(aa)
            if len(groups[group_index]) == 1:
                poses[image_id] = (R_var, vec)
            else:
                poses[image_id] = _compose_group_pose(image.R, image.t, R_var, vec)
        focals = {}
        distortions = {}
        for camera_id, camera in camera_by_id.items():
            focals[camera_id] = camera.focal
            distortions[camera_id] = camera.k1
        for slot, camera_id in enumerate(intrinsic_ids):
            base = n_group + 2 * slot
            focals[camera_id] = float(np.exp(params[base]))
            distortions[camera_id] = float(params[base + 1]) if options.refine_extra_params else camera_by_id[camera_id].k1
        pts = {}
        for slot, point_id in enumerate(point_ids):
            pts[point_id] = params[n_group + n_intr + 3 * slot : n_group + n_intr + 3 * slot + 3]
        return poses, focals, distortions, pts

    point_offset = n_group + n_intr

    def residuals(params: np.ndarray) -> np.ndarray:
        poses, focals, distortions, _ = unpack(params)
        point_matrix = params[point_offset:].reshape(-1, 3)
        blocks = []
        for image_id, indices, xy in grouped_obs:
            R, t = poses[image_id]
            image = by_id[image_id]
            uv, _ = project_points(
                point_matrix[indices],
                R,
                t,
                focals[image.camera_id],
                image.width,
                image.height,
                distortions[image.camera_id],
            )
            blocks.append((uv - xy).reshape(-1))
        center_a = camera_center(*poses[anchor])
        center_b = camera_center(*poses[scale_partner])
        blocks.append(np.array([50.0 * (np.linalg.norm(center_a - center_b) - baseline)]))
        return np.concatenate(blocks)

    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss="cauchy",
        f_scale=max(options.ba_loss_scale_px, 1e-6),
        max_nfev=max(max_iterations, 1) * max(int(x0.size), 1),
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    poses, focals, distortions, pts = unpack(result.x)
    return BAResult(
        rotations={image_id: pose[0] for image_id, pose in poses.items()},
        translations={image_id: pose[1] for image_id, pose in poses.items()},
        points=pts,
        focals=focals,
        distortions=distortions,
        success=bool(result.success),
    )


def _identity_result(images: list[BAImage], cameras: list[BACamera], points: dict[int, np.ndarray], success: bool) -> BAResult:
    return BAResult(
        rotations={image.image_id: image.R.copy() for image in images},
        translations={image.image_id: image.t.copy() for image in images},
        points={point_id: xyz.copy() for point_id, xyz in points.items()},
        focals={camera.camera_id: camera.focal for camera in cameras},
        distortions={camera.camera_id: camera.k1 for camera in cameras},
        success=success,
    )
