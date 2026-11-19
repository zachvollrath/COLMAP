"""Correspondence search products and multi-model geometric verification.

Section 4.1 estimates a fundamental matrix, classifies the pair with a
homography and (when intrinsics are known) an essential matrix, and rejects
watermark / timestamp / frame links estimated by a border similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from colmap.geometry import (
    DTYPE,
    as_tensor,
    camera_center,
    decompose_essential,
    homography_transfer_error,
    pixels_to_normalized,
    project_points,
    projection_matrix,
    sampson_error,
    similarity_error,
    similarity_from_points,
    triangulate_two_views,
)
from colmap.device import generator
from colmap.options import VerificationOptions
from colmap.ransac import ransac, ransac_essential, ransac_fundamental, ransac_homography


GENERAL = "general"
PLANAR = "planar"
PANORAMIC = "panoramic"
WATERMARK = "watermark"


@dataclass
class Camera:
    camera_id: int
    width: int
    height: int
    focal: float
    k1: float = 0.0
    prior_focal: float = 0.0

    def __post_init__(self) -> None:
        if self.prior_focal <= 0:
            self.prior_focal = self.focal

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


@dataclass
class ImageNode:
    image_id: int
    name: str
    camera_id: int
    keypoints: torch.Tensor  # (N, 2) pixel locations
    descriptors: torch.Tensor | None = None


@dataclass
class PairGeometry:
    image_id1: int
    image_id2: int
    matches: torch.Tensor  # (M, 2) inlier feature indices
    model: str
    calibrated: bool
    num_inliers: int
    R: torch.Tensor | None = None
    t: torch.Tensor | None = None
    median_angle_deg: float = 0.0


@dataclass
class SceneGraph:
    cameras: dict[int, Camera]
    images: dict[int, ImageNode]
    pairs: list[PairGeometry] = field(default_factory=list)

    def neighbors(self, image_id: int) -> set[int]:
        ids = set()
        for pair in self.pairs:
            if pair.image_id1 == image_id:
                ids.add(pair.image_id2)
            elif pair.image_id2 == image_id:
                ids.add(pair.image_id1)
        return ids


def match_descriptors(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    ratio: float = 0.8,
) -> torch.Tensor:
    """Exhaustive Lowe-ratio matching. Returns (M, 2) index pairs."""
    desc_a = as_tensor(desc_a, dtype=torch.float32)
    desc_b = as_tensor(desc_b, dtype=torch.float32)
    if len(desc_a) == 0 or len(desc_b) < 2:
        return torch.zeros((0, 2), dtype=torch.long)
    block = 512
    best = torch.full((len(desc_a),), float("inf"))
    second = torch.full((len(desc_a),), float("inf"))
    index = torch.full((len(desc_a),), -1, dtype=torch.long)
    for start in range(0, len(desc_b), block):
        chunk = desc_b[start : start + block]
        dist = torch.cdist(desc_a, chunk, p=2).square()
        take = min(2, dist.shape[1])
        values, local = torch.topk(dist, k=take, largest=False, dim=1)
        d0 = values[:, 0]
        d1 = values[:, 1] if take > 1 else torch.full_like(d0, float("inf"))
        col0 = local[:, 0] + start
        better = d0 < best
        second = torch.where(better, torch.minimum(best, d1), torch.where(d0 < second, d0, second))
        best = torch.where(better, d0, best)
        index = torch.where(better, col0, index)
    keep = (index >= 0) & (best < (ratio * ratio) * second.clamp(min=1e-12))
    rows = keep.nonzero(as_tuple=False).flatten()
    return torch.column_stack([rows, index[rows]])


def _border_mask(pixels: torch.Tensor, width: int, height: int, fraction: float) -> torch.Tensor:
    bx = fraction * width
    by = fraction * height
    return (
        (pixels[:, 0] < bx)
        | (pixels[:, 0] > width - bx)
        | (pixels[:, 1] < by)
        | (pixels[:, 1] > height - by)
    )


def _recover_relative_pose(
    E: torch.Tensor,
    pts1_n: torch.Tensor,
    pts2_n: torch.Tensor,
    inliers: torch.Tensor,
    focal1: float,
    width1: int,
    height1: int,
    focal2: float,
    width2: int,
    height2: int,
    max_error_px: float,
    min_support: int,
) -> tuple[torch.Tensor, torch.Tensor, float, int] | None:
    """Choose the essential decomposition whose points reproject in front of both cameras."""
    device = pts1_n.device
    pts1 = pts1_n[inliers].detach().to(device="cpu", dtype=torch.float32)
    pts2 = pts2_n[inliers].detach().to(device="cpu", dtype=torch.float32)
    # Lift normalized coordinates back to pixels with the same intrinsics that
    # produced them, so the reprojection test is in the original image frame.
    pix1 = torch.column_stack((pts1[:, 0] * focal1 + width1 / 2.0, pts1[:, 1] * focal1 + height1 / 2.0))
    pix2 = torch.column_stack((pts2[:, 0] * focal2 + width2 / 2.0, pts2[:, 1] * focal2 + height2 / 2.0))
    identity = torch.eye(3, dtype=torch.float32, device="cpu")
    zero = torch.zeros(3, dtype=torch.float32, device="cpu")
    P1 = projection_matrix(identity, zero, focal1, width1, height1).to(dtype=torch.float32, device="cpu")
    origin = torch.zeros(3, dtype=torch.float32, device="cpu")
    best = None
    best_support = 0
    model = E.detach().to(device="cpu", dtype=torch.float32)
    for rotation, translation in decompose_essential(model):
        P2 = projection_matrix(rotation, translation, focal2, width2, height2).to(device="cpu", dtype=torch.float32)
        points = triangulate_two_views(P1, pix1, P2, pix2)
        uv1, depth1 = project_points(points, identity, zero, focal1, width1, height1, 0.0)
        uv2, depth2 = project_points(points, rotation, translation, focal2, width2, height2, 0.0)
        error1 = torch.linalg.norm(uv1 - pix1, dim=1)
        error2 = torch.linalg.norm(uv2 - pix2, dim=1)
        finite = torch.isfinite(points).all(dim=1) & torch.isfinite(error1) & torch.isfinite(error2)
        keep = finite & (depth1 > 0) & (depth2 > 0) & (error1 <= max_error_px) & (error2 <= max_error_px)
        support = int(keep.sum())
        if support <= best_support:
            continue
        center2 = camera_center(rotation, translation)
        directions_a = origin - points[keep]
        directions_b = center2 - points[keep]
        cosine = torch.sum(directions_a * directions_b, dim=1) / (
            torch.linalg.norm(directions_a, dim=1).clamp(min=1e-12)
            * torch.linalg.norm(directions_b, dim=1).clamp(min=1e-12)
        )
        median = float(torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).median())
        best_support = support
        best = (rotation.to(device=device, dtype=DTYPE), translation.to(device=device, dtype=DTYPE), median, support)
    if best is None or best_support < min_support:
        return None
    return best


def verify_pair(
    image1: ImageNode,
    image2: ImageNode,
    camera1: Camera,
    camera2: Camera,
    matches: torch.Tensor,
    options: VerificationOptions,
    rng: torch.Generator,
) -> PairGeometry | None:
    """Classify one image pair and return its verified geometry, or None."""
    if len(matches) < options.min_num_inliers:
        return None
    pts1 = image1.keypoints[matches[:, 0]]
    pts2 = image2.keypoints[matches[:, 1]]
    threshold = options.max_error_px ** 2
    F, inliers_f = ransac_fundamental(
        pts1,
        pts2,
        threshold,
        rng,
        confidence=options.ransac_confidence,
        max_iters=options.ransac_max_iters,
        min_inliers=options.min_num_inliers,
    )
    if F is None or inliers_f is None:
        return None
    n_f = int(inliers_f.sum().item())
    H, inliers_h = ransac_homography(
        pts1,
        pts2,
        threshold,
        rng,
        confidence=options.ransac_confidence,
        max_iters=options.ransac_max_iters,
        min_inliers=4,
    )
    n_h = int(inliers_h.sum().item()) if inliers_h is not None else 0
    homography_ratio = n_h / max(n_f, 1)

    n1 = pixels_to_normalized(pts1, camera1.focal, camera1.width, camera1.height, camera1.k1)
    n2 = pixels_to_normalized(pts2, camera2.focal, camera2.width, camera2.height, camera2.k1)
    # Epipolar error on the normalized plane, converted from the pixel threshold.
    focal_scale = min(camera1.focal, camera2.focal)
    essential_threshold = (options.max_error_px / focal_scale) ** 2
    E, inliers_e = ransac_essential(
        n1,
        n2,
        essential_threshold,
        rng,
        confidence=options.ransac_confidence,
        max_iters=options.ransac_max_iters,
        min_inliers=options.min_num_inliers,
    )
    n_e = int(inliers_e.sum().item()) if inliers_e is not None else 0
    calibrated = E is not None and (n_e / max(n_f, 1)) >= options.min_essential_inlier_ratio

    if _is_watermark(pts1, pts2, inliers_f, inliers_e, camera1, camera2, n_f, n_e, options, rng):
        return None

    pose = None
    if calibrated and E is not None and inliers_e is not None:
        pose = _recover_relative_pose(
            E,
            n1,
            n2,
            inliers_e,
            camera1.focal,
            camera1.width,
            camera1.height,
            camera2.focal,
            camera2.width,
            camera2.height,
            options.max_error_px,
            options.min_num_inliers,
        )
        if pose is None:
            calibrated = False

    general_scene = homography_ratio < options.max_homography_inlier_ratio
    median_angle = pose[2] if pose is not None else 0.0
    if not general_scene and median_angle < options.panoramic_max_angle_deg:
        model = PANORAMIC
    elif not general_scene:
        model = PLANAR
    else:
        model = GENERAL

    # A calibrated pair is triangulated with the essential pose, so the stored
    # correspondences have to be the essential inliers. The largest fundamental
    # support can be a many-to-one degeneracy that does not describe that pose.
    if calibrated and inliers_e is not None and E is not None:
        chosen_mask = inliers_e
        residual = sampson_error(E, n1, n2)
    elif inliers_h is not None and H is not None and n_h >= n_f:
        chosen_mask = inliers_h
        residual = homography_transfer_error(H, pts1, pts2)
    else:
        chosen_mask = inliers_f
        residual = sampson_error(F, pts1, pts2)
    inlier_matches = _one_to_one(matches[chosen_mask], residual[chosen_mask])
    return PairGeometry(
        image_id1=image1.image_id,
        image_id2=image2.image_id,
        matches=inlier_matches,
        model=model,
        calibrated=calibrated,
        num_inliers=int(len(inlier_matches)),
        R=None if pose is None else pose[0],
        t=None if pose is None else pose[1],
        median_angle_deg=median_angle,
    )


def _one_to_one(matches: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    """Keep the lowest-error match for each feature on either image."""
    best_for_first: dict[int, tuple[float, int, int]] = {}
    for index, (first, second) in enumerate(matches.tolist()):
        first = int(first)
        second = int(second)
        residual = float(error[index])
        current = best_for_first.get(first)
        if current is None or residual < current[0]:
            best_for_first[first] = (residual, second, index)
    best_for_second: dict[int, tuple[float, int]] = {}
    for residual, second, index in best_for_first.values():
        current = best_for_second.get(second)
        if current is None or residual < current[0]:
            best_for_second[second] = (residual, index)
    kept = [index for _, index in best_for_second.values()]
    if not kept:
        return matches[:0]
    return matches[torch.tensor(kept, dtype=torch.long)]


def _is_watermark(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    inliers_f: torch.Tensor,
    inliers_e: torch.Tensor | None,
    camera1: Camera,
    camera2: Camera,
    n_f: int,
    n_e: int,
    options: VerificationOptions,
    rng: torch.Generator,
) -> bool:
    border = _border_mask(pts1, camera1.width, camera1.height, options.watermark_border_fraction) & _border_mask(
        pts2, camera2.width, camera2.height, options.watermark_border_fraction
    )
    border_idx = border.nonzero(as_tuple=False).flatten()
    if len(border_idx) < options.min_border_matches:
        return False
    b1 = pts1[border_idx]
    b2 = pts2[border_idx]
    threshold = options.max_error_px ** 2

    def fit(sample: torch.Tensor):
        return similarity_from_points(b1[sample], b2[sample])

    def residual(model: object) -> torch.Tensor:
        return similarity_error(model, b1, b2)  # type: ignore[arg-type]

    model, inliers = ransac(
        len(border_idx),
        2,
        fit,
        residual,
        threshold,
        rng,
        confidence=options.ransac_confidence,
        max_iters=min(1000, options.ransac_max_iters),
        min_inliers=options.min_border_matches,
    )
    if model is None or inliers is None:
        return False
    n_s = int(inliers.sum().item())
    if n_s / max(n_f, 1) > options.max_watermark_inlier_ratio:
        return True
    if inliers_e is not None and n_e > 0 and n_s / n_e > options.max_watermark_inlier_ratio:
        return True
    return False


def build_scene_graph(
    cameras: dict[int, Camera],
    images: dict[int, ImageNode],
    matches: dict[tuple[int, int], torch.Tensor],
    options: VerificationOptions | None = None,
    seed: int = 0,
) -> SceneGraph:
    """Verify every putative pair and keep the pairs that enter the scene graph."""
    options = options or VerificationOptions()
    rng = generator(seed)
    graph = SceneGraph(cameras=cameras, images=images)
    for (id1, id2), pair_matches in matches.items():
        if id1 == id2 or len(pair_matches) == 0:
            continue
        image1 = images[id1]
        image2 = images[id2]
        geometry = verify_pair(
            image1,
            image2,
            cameras[image1.camera_id],
            cameras[image2.camera_id],
            as_tensor(pair_matches, dtype=torch.long),
            options,
            rng,
        )
        if geometry is not None:
            graph.pairs.append(geometry)
    return graph


def pairwise_matches_from_descriptors(
    images: dict[int, ImageNode],
    ratio: float = 0.8,
) -> dict[tuple[int, int], torch.Tensor]:
    """Exhaustive appearance matching. The paper's large-scale system instead
    queries a vocabulary tree; exhaustive search is the exact procedure those
    approximations replace (Section 2.1).
    """
    ids = sorted(images)
    found: dict[tuple[int, int], torch.Tensor] = {}
    for i, id1 in enumerate(ids):
        desc1 = images[id1].descriptors
        if desc1 is None:
            continue
        for id2 in ids[i + 1 :]:
            desc2 = images[id2].descriptors
            if desc2 is None:
                continue
            found[(id1, id2)] = match_descriptors(desc1, desc2, ratio=ratio)
    return found
