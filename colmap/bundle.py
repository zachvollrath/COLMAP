"""Local and global bundle adjustment, Sections 4.4 and 4.5.

The cost is the Cauchy-robust reprojection error of the simple camera.
Parameters are refined with Levenberg-Marquardt; the Jacobian comes from
PyTorch forward-mode autodiff. Global problems with a dense viewing graph may
collapse unaffected, highly overlapping cameras into one rigid group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from colmap.geometry import (
    DTYPE,
    angle_axis_to_matrix,
    as_tensor,
    camera_center,
    matrix_to_angle_axis,
    project_points,
    viewing_direction,
)
from colmap.optimize import cauchy, levenberg_marquardt
from colmap.options import MapperOptions


@dataclass
class BAImage:
    image_id: int
    camera_id: int
    R: torch.Tensor
    t: torch.Tensor
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
    xy: torch.Tensor


@dataclass
class BAResult:
    rotations: dict[int, torch.Tensor]
    translations: dict[int, torch.Tensor]
    points: dict[int, torch.Tensor]
    focals: dict[int, float]
    distortions: dict[int, float]
    success: bool


def overlap_ratio(visible_a: set[int], visible_b: set[int]) -> float:
    """Jaccard interaction V_ab of two visibility sets."""
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
    centers: dict[int, torch.Tensor],
    directions: dict[int, torch.Tensor],
    neighbor_count: int,
    max_angle_deg: float,
) -> list[int]:
    seed_dir = directions[seed]
    cosine_limit = float(torch.cos(torch.deg2rad(torch.tensor(max_angle_deg))))
    ranked = []
    for candidate in remaining:
        direction = directions[candidate]
        if float(torch.dot(seed_dir, direction)) < cosine_limit:
            continue
        distance = float(torch.linalg.norm(centers[candidate] - centers[seed]))
        ranked.append((distance, candidate))
    ranked.sort()
    return [candidate for _, candidate in ranked[:neighbor_count]]


def _compose_group_pose(
    R_c: torch.Tensor,
    t_c: torch.Tensor,
    R_g: torch.Tensor,
    t_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P_cr = P_c G_r, with both poses stored as world-to-camera rigid transforms."""
    R = R_c @ R_g
    t = R_c @ t_g + t_c
    return R, t


def bundle_adjust(
    images: list[BAImage],
    cameras: list[BACamera],
    points: dict[int, torch.Tensor],
    observations: list[BAObservation],
    options: MapperOptions,
    groups: list[list[int]] | None = None,
    refine_intrinsics: bool = False,
    max_iterations: int = 25,
    anchor_image: int | None = None,
    fixed_images: set[int] | None = None,
) -> BAResult:
    """Minimize the reprojection cost, or the grouped cost when cameras share a rigid motion.

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
        torch.linalg.norm(
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
        if any(member in fixed for member in group):
            continue
        free_groups.append(index)
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
    x0 = torch.zeros(n_group + n_intr + 3 * len(point_ids), dtype=DTYPE)
    fixed_pose = {}
    for image_id, image in by_id.items():
        fixed_pose[image_id] = (as_tensor(image.R).detach(), as_tensor(image.t).detach())
    for group_index, slot in group_param_index.items():
        members = groups[group_index]
        if len(members) == 1:
            image = by_id[members[0]]
            x0[6 * slot : 6 * slot + 3] = matrix_to_angle_axis(image.R)
            x0[6 * slot + 3 : 6 * slot + 6] = as_tensor(image.t)
    for slot, camera_id in enumerate(intrinsic_ids):
        camera = camera_by_id[camera_id]
        base = n_group + 2 * slot
        x0[base] = torch.log(torch.tensor(max(camera.focal, 1e-6), dtype=DTYPE))
        x0[base + 1] = camera.k1 if options.refine_extra_params else 0.0
    point_offset = n_group + n_intr
    for slot, point_id in enumerate(point_ids):
        x0[point_offset + 3 * slot : point_offset + 3 * slot + 3] = as_tensor(points[point_id])

    grouped_obs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    pending: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for obs in observations:
        if obs.image_id not in by_id or obs.point_id not in point_index:
            continue
        pending.setdefault(obs.image_id, []).append((point_index[obs.point_id], as_tensor(obs.xy)))
    for image_id, entries in pending.items():
        grouped_obs.append(
            (
                image_id,
                torch.tensor([entry[0] for entry in entries], dtype=torch.long),
                torch.stack([entry[1] for entry in entries]),
            )
        )
    if sum(len(xy) for _, _, xy in grouped_obs) < 4:
        return _identity_result(images, cameras, points, False)

    def unpack(params: torch.Tensor):
        poses = {}
        for image_id, image in by_id.items():
            group_index = group_of[image_id]
            if group_index not in group_param_index:
                poses[image_id] = fixed_pose[image_id]
                continue
            slot = group_param_index[group_index]
            aa = params[6 * slot : 6 * slot + 3]
            vec = params[6 * slot + 3 : 6 * slot + 6]
            R_var = angle_axis_to_matrix(aa)
            if len(groups[group_index]) == 1:
                poses[image_id] = (R_var, vec)
            else:
                R_c, t_c = fixed_pose[image_id]
                poses[image_id] = _compose_group_pose(R_c, t_c, R_var, vec)
        focals = {camera_id: camera.focal for camera_id, camera in camera_by_id.items()}
        distortions = {camera_id: camera.k1 for camera_id, camera in camera_by_id.items()}
        for slot, camera_id in enumerate(intrinsic_ids):
            base = n_group + 2 * slot
            focals[camera_id] = torch.exp(params[base])
            if options.refine_extra_params:
                distortions[camera_id] = params[base + 1]
        point_matrix = params[point_offset:].reshape(-1, 3)
        return poses, focals, distortions, point_matrix

    def residuals(params: torch.Tensor) -> torch.Tensor:
        poses, focals, distortions, point_matrix = unpack(params)
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
        blocks.append(50.0 * (torch.linalg.norm(center_a - center_b) - baseline).reshape(1))
        return torch.cat(blocks)

    refined_cam, refined_points, success = _adjust_partitioned(
        x0[:point_offset].detach(),
        x0[point_offset:].detach().reshape(-1, 3),
        residuals,
        unpack,
        grouped_obs,
        by_id,
        max_iterations,
        options.ba_loss_scale_px,
    )
    refined = torch.cat([refined_cam, refined_points.reshape(-1)])
    poses, focals, distortions, point_matrix = unpack(refined)
    point_matrix = point_matrix.detach()
    return BAResult(
        rotations={image_id: pose[0].detach() for image_id, pose in poses.items()},
        translations={image_id: pose[1].detach() for image_id, pose in poses.items()},
        points={point_id: point_matrix[slot].clone() for slot, point_id in enumerate(point_ids)},
        focals={camera_id: float(value) for camera_id, value in focals.items()},
        distortions={camera_id: float(value) for camera_id, value in distortions.items()},
        success=success,
    )


def _adjust_partitioned(
    cameras0: torch.Tensor,
    points0: torch.Tensor,
    residuals,
    unpack,
    grouped_obs: list[tuple[int, torch.Tensor, torch.Tensor]],
    by_id: dict[int, BAImage],
    max_iterations: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Alternate a camera step with an independent point step.

    The joint normal matrix is dense in the points, so one Levenberg-Marquardt
    step on every parameter does not scale past a few hundred tracks. Cameras
    (and intrinsics) stay a small dense block. Each point then has its own 3x3
    system, which is the point half of the Schur complement.
    """
    cameras = cameras0.clone()
    points = points0.clone()
    accepted = False
    camera_ok = False

    def total_cost(cam: torch.Tensor, xyz: torch.Tensor) -> float:
        return float(cauchy(residuals(torch.cat([cam, xyz.reshape(-1)])), scale).detach())

    cost = total_cost(cameras, points)
    for _ in range(max(max_iterations, 1)):
        def camera_residuals(cam: torch.Tensor, fixed_points: torch.Tensor = points) -> torch.Tensor:
            return residuals(torch.cat([cam, fixed_points.reshape(-1)]))

        cameras, camera_ok = levenberg_marquardt(cameras, camera_residuals, max_iterations=1, scale=scale)
        cameras = cameras.detach()
        poses, focals, distortions, _ = unpack(torch.cat([cameras, points.reshape(-1)]))
        poses = {image_id: (rotation.detach(), translation.detach()) for image_id, (rotation, translation) in poses.items()}
        focals = {camera_id: value.detach() if isinstance(value, torch.Tensor) else value for camera_id, value in focals.items()}
        distortions = {
            camera_id: value.detach() if isinstance(value, torch.Tensor) else value
            for camera_id, value in distortions.items()
        }
        points = _refine_points(points, poses, focals, distortions, grouped_obs, by_id, scale)
        updated = total_cost(cameras, points)
        if updated < cost:
            accepted = True
            if cost - updated < 1e-8:
                break
            cost = updated
            continue
        break
    return cameras, points, accepted or camera_ok


def _refine_points(
    points: torch.Tensor,
    poses: dict[int, tuple[torch.Tensor, torch.Tensor]],
    focals: dict[int, torch.Tensor | float],
    distortions: dict[int, torch.Tensor | float],
    grouped_obs: list[tuple[int, torch.Tensor, torch.Tensor]],
    by_id: dict[int, BAImage],
    scale: float,
) -> torch.Tensor:
    """One damped step per point. Points do not couple once the cameras are fixed."""
    hessian, gradient = _point_normal(points, poses, focals, distortions, grouped_obs, by_id, scale)
    damping = torch.full((len(points),), 1e-3, dtype=DTYPE)
    eye = torch.eye(3, dtype=DTYPE)
    base_cost = _point_costs(points, poses, focals, distortions, grouped_obs, by_id, scale)
    updated = points.clone()
    pending = torch.ones(len(points), dtype=torch.bool)
    for _attempt in range(8):
        diagonal = hessian.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
        system = hessian + damping[:, None, None] * torch.diag_embed(diagonal) + 1e-10 * eye
        step = torch.linalg.solve(system, -gradient)
        candidate = points + step
        cost = _point_costs(candidate, poses, focals, distortions, grouped_obs, by_id, scale)
        improved = pending & (cost < base_cost)
        updated = torch.where(improved[:, None], candidate, updated)
        pending = pending & ~improved
        if not bool(pending.any()):
            break
        damping = torch.where(pending, damping * 4.0, damping)
    return updated.detach()


def _point_costs(points, poses, focals, distortions, grouped_obs, by_id, scale: float) -> torch.Tensor:
    cost = torch.zeros(len(points), dtype=DTYPE)
    safe_scale = max(float(scale), 1e-6)
    for image_id, indices, xy in grouped_obs:
        rotation, translation = poses[image_id]
        image = by_id[image_id]
        uv, _ = project_points(
            points[indices],
            rotation,
            translation,
            focals[image.camera_id],
            image.width,
            image.height,
            distortions[image.camera_id],
        )
        residual = uv - xy
        cost.index_add_(0, indices, torch.log1p((residual / safe_scale) ** 2).sum(dim=1))
    return cost


def _point_normal(points, poses, focals, distortions, grouped_obs, by_id, scale: float):
    hessian = torch.zeros((len(points), 3, 3), dtype=DTYPE)
    gradient = torch.zeros((len(points), 3), dtype=DTYPE)
    safe_scale = max(float(scale), 1e-6)
    for image_id, indices, xy in grouped_obs:
        rotation, translation = poses[image_id]
        image = by_id[image_id]
        uv, jacobian = _reprojection_jacobian_wrt_point(
            points[indices],
            rotation,
            translation,
            focals[image.camera_id],
            image.width,
            image.height,
            distortions[image.camera_id],
        )
        residual = uv - xy
        weight = 1.0 / (1.0 + (residual / safe_scale) ** 2)
        weighted = jacobian * weight[:, :, None]
        hessian.index_add_(0, indices, torch.einsum("nri,nrj->nij", weighted, jacobian))
        gradient.index_add_(0, indices, torch.einsum("nri,nr->ni", jacobian, weight * residual))
    return hessian, gradient


def _reprojection_jacobian_wrt_point(points, rotation, translation, focal, width, height, k1):
    """Pixels and d(pixel)/d(world point) for the simple radial camera."""
    points = as_tensor(points)
    rotation = as_tensor(rotation)
    translation = as_tensor(translation).reshape(3)
    focal_t = as_tensor(focal).reshape(())
    k1_t = as_tensor(k1).reshape(())
    camera = points @ rotation.transpose(0, 1) + translation
    depth = camera[:, 2]
    z = torch.where(depth.abs() < 1e-12, torch.full_like(depth, 1e-12), depth)
    x = camera[:, 0] / z
    y = camera[:, 1] / z
    radius2 = x * x + y * y
    radial = 1.0 + k1_t * radius2
    du_dx = focal_t * (radial + 2.0 * k1_t * x * x)
    du_dy = focal_t * (2.0 * k1_t * x * y)
    dv_dy = focal_t * (radial + 2.0 * k1_t * y * y)
    inv_z = 1.0 / z
    du_d_camera = torch.stack(
        [du_dx * inv_z, du_dy * inv_z, -(du_dx * x + du_dy * y) * inv_z],
        dim=1,
    )
    dv_d_camera = torch.stack(
        [du_dy * inv_z, dv_dy * inv_z, -(du_dy * x + dv_dy * y) * inv_z],
        dim=1,
    )
    jacobian_camera = torch.stack([du_d_camera, dv_d_camera], dim=1)
    jacobian = torch.matmul(jacobian_camera, rotation)
    pixels = torch.column_stack(
        [focal_t * (x * radial) + width / 2.0, focal_t * (y * radial) + height / 2.0]
    )
    return pixels, jacobian


def _identity_result(
    images: list[BAImage],
    cameras: list[BACamera],
    points: dict[int, torch.Tensor],
    success: bool,
) -> BAResult:
    return BAResult(
        rotations={image.image_id: as_tensor(image.R).clone() for image in images},
        translations={image.image_id: as_tensor(image.t).clone() for image in images},
        points={point_id: as_tensor(xyz).clone() for point_id, xyz in points.items()},
        focals={camera.camera_id: camera.focal for camera in cameras},
        distortions={camera.camera_id: camera.k1 for camera in cameras},
        success=success,
    )
