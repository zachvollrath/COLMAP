"""Correspondence search products and multi-model geometric verification.

Section 4.1 estimates a fundamental matrix, classifies the pair with a
homography and (when intrinsics are known) an essential matrix, and rejects
watermark / timestamp / frame links estimated by a border similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colmap.geometry import (
    camera_center,
    decompose_essential,
    essential_from_points,
    fundamental_from_points,
    homography_from_points,
    homography_transfer_error,
    pixels_to_normalized,
    projection_matrix,
    sampson_error,
    similarity_error,
    similarity_from_points,
    triangulate_dlt,
    triangulation_angle_deg,
)
from colmap.options import VerificationOptions
from colmap.ransac import ransac


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
    keypoints: np.ndarray  # (N, 2) pixel locations
    descriptors: np.ndarray | None = None


@dataclass
class PairGeometry:
    image_id1: int
    image_id2: int
    matches: np.ndarray  # (M, 2) inlier feature indices
    model: str
    calibrated: bool
    num_inliers: int
    R: np.ndarray | None = None
    t: np.ndarray | None = None
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
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio: float = 0.8,
) -> np.ndarray:
    """Exhaustive Lowe-ratio matching. Returns (M, 2) index pairs."""
    if len(desc_a) == 0 or len(desc_b) < 2:
        return np.zeros((0, 2), dtype=int)
    block = 512
    best = np.full(len(desc_a), np.inf)
    second = np.full(len(desc_a), np.inf)
    index = np.full(len(desc_a), -1, dtype=int)
    for start in range(0, len(desc_b), block):
        chunk = desc_b[start : start + block]
        dist = np.sum((desc_a[:, None, :] - chunk[None, :, :]) ** 2, axis=2)
        take = min(2, dist.shape[1])
        local = np.argpartition(dist, take - 1, axis=1)[:, :take]
        for row in range(len(desc_a)):
            cols = local[row]
            order = np.argsort(dist[row, cols])
            d0 = float(dist[row, cols[order[0]]])
            d1 = float(dist[row, cols[order[1]]]) if take > 1 else np.inf
            col0 = int(start + cols[order[0]])
            if d0 < best[row]:
                if best[row] < d1:
                    second[row] = best[row]
                else:
                    second[row] = d1
                best[row] = d0
                index[row] = col0
            elif d0 < second[row]:
                second[row] = d0
    keep = (index >= 0) & (best < (ratio * ratio) * np.maximum(second, 1e-12))
    rows = np.flatnonzero(keep)
    return np.column_stack([rows, index[rows]]).astype(int)


def _border_mask(pixels: np.ndarray, width: int, height: int, fraction: float) -> np.ndarray:
    bx = fraction * width
    by = fraction * height
    return (
        (pixels[:, 0] < bx)
        | (pixels[:, 0] > width - bx)
        | (pixels[:, 1] < by)
        | (pixels[:, 1] > height - by)
    )


def _recover_relative_pose(
    E: np.ndarray,
    pts1_n: np.ndarray,
    pts2_n: np.ndarray,
    inliers: np.ndarray,
    focal1: float,
    width1: int,
    height1: int,
    focal2: float,
    width2: int,
    height2: int,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """Choose the essential decomposition with the largest cheiral support."""
    pts1 = pts1_n[inliers]
    pts2 = pts2_n[inliers]
    # Lift normalized coordinates back to pixels of a unit-focal camera so the
    # shared DLT triangulation can consume projection matrices. The relative
    # pose itself does not depend on that arbitrary focal length.
    pix1 = np.column_stack([pts1[:, 0] * focal1 + width1 / 2.0, pts1[:, 1] * focal1 + height1 / 2.0])
    pix2 = np.column_stack([pts2[:, 0] * focal2 + width2 / 2.0, pts2[:, 1] * focal2 + height2 / 2.0])
    best = None
    best_support = 0
    I = np.eye(3)
    zero = np.zeros(3)
    P1 = projection_matrix(I, zero, focal1, width1, height1)
    for R, t in decompose_essential(E):
        P2 = projection_matrix(R, t, focal2, width2, height2)
        center2 = camera_center(R, t)
        angles = []
        support = 0
        for xy1, xy2 in zip(pix1, pix2):
            X = triangulate_dlt([(P1, xy1), (P2, xy2)])
            if X is None:
                continue
            depth1 = float(X[2])
            depth2 = float((R @ X + t)[2])
            if depth1 <= 0 or depth2 <= 0:
                continue
            support += 1
            angles.append(triangulation_angle_deg(np.zeros(3), center2, X))
        if support > best_support and angles:
            best_support = support
            best = (R, t, float(np.median(angles)), support)
    return best


def verify_pair(
    image1: ImageNode,
    image2: ImageNode,
    camera1: Camera,
    camera2: Camera,
    matches: np.ndarray,
    options: VerificationOptions,
    rng: np.random.Generator,
) -> PairGeometry | None:
    """Classify one image pair and return its verified geometry, or None."""
    if len(matches) < options.min_num_inliers:
        return None
    pts1 = image1.keypoints[matches[:, 0]]
    pts2 = image2.keypoints[matches[:, 1]]
    threshold = options.max_error_px ** 2

    def fit_f(sample: np.ndarray):
        return fundamental_from_points(pts1[sample], pts2[sample])

    def residual_f(model: np.ndarray) -> np.ndarray:
        return sampson_error(model, pts1, pts2)

    F, inliers_f = ransac(
        len(matches),
        8,
        fit_f,
        residual_f,
        threshold,
        rng,
        confidence=options.ransac_confidence,
        max_iters=options.ransac_max_iters,
        min_inliers=options.min_num_inliers,
    )
    if F is None or inliers_f is None:
        return None
    n_f = int(np.count_nonzero(inliers_f))

    def fit_h(sample: np.ndarray):
        return homography_from_points(pts1[sample], pts2[sample])

    def residual_h(model: np.ndarray) -> np.ndarray:
        return homography_transfer_error(model, pts1, pts2)

    H, inliers_h = ransac(
        len(matches),
        4,
        fit_h,
        residual_h,
        threshold,
        rng,
        confidence=options.ransac_confidence,
        max_iters=options.ransac_max_iters,
        min_inliers=4,
    )
    n_h = int(np.count_nonzero(inliers_h)) if inliers_h is not None else 0
    homography_ratio = n_h / max(n_f, 1)

    n1 = pixels_to_normalized(pts1, camera1.focal, camera1.width, camera1.height, camera1.k1)
    n2 = pixels_to_normalized(pts2, camera2.focal, camera2.width, camera2.height, camera2.k1)
    # Epipolar error on the normalized plane, converted from the pixel threshold.
    focal_scale = min(camera1.focal, camera2.focal)
    essential_threshold = (options.max_error_px / focal_scale) ** 2

    def fit_e(sample: np.ndarray):
        return essential_from_points(n1[sample], n2[sample])

    def residual_e(model: np.ndarray) -> np.ndarray:
        return sampson_error(model, n1, n2)

    E, inliers_e = ransac(
        len(matches),
        8,
        fit_e,
        residual_e,
        essential_threshold,
        rng,
        confidence=options.ransac_confidence,
        max_iters=options.ransac_max_iters,
        min_inliers=options.min_num_inliers,
    )
    n_e = int(np.count_nonzero(inliers_e)) if inliers_e is not None else 0
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

    # Keep the inliers of the model with the largest support.
    candidates = [(n_f, inliers_f)]
    if inliers_h is not None:
        candidates.append((n_h, inliers_h))
    if calibrated and inliers_e is not None:
        candidates.append((n_e, inliers_e))
    _, best_inliers = max(candidates, key=lambda item: item[0])
    inlier_matches = matches[best_inliers]
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


def _is_watermark(
    pts1: np.ndarray,
    pts2: np.ndarray,
    inliers_f: np.ndarray,
    inliers_e: np.ndarray | None,
    camera1: Camera,
    camera2: Camera,
    n_f: int,
    n_e: int,
    options: VerificationOptions,
    rng: np.random.Generator,
) -> bool:
    border = _border_mask(pts1, camera1.width, camera1.height, options.watermark_border_fraction) & _border_mask(
        pts2, camera2.width, camera2.height, options.watermark_border_fraction
    )
    border_idx = np.flatnonzero(border)
    if len(border_idx) < options.min_border_matches:
        return False
    b1 = pts1[border_idx]
    b2 = pts2[border_idx]
    threshold = options.max_error_px ** 2

    def fit(sample: np.ndarray):
        return similarity_from_points(b1[sample], b2[sample])

    def residual(model: object) -> np.ndarray:
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
    n_s = int(np.count_nonzero(inliers))
    if n_s / max(n_f, 1) > options.max_watermark_inlier_ratio:
        return True
    if inliers_e is not None and n_e > 0 and n_s / n_e > options.max_watermark_inlier_ratio:
        return True
    return False


def build_scene_graph(
    cameras: dict[int, Camera],
    images: dict[int, ImageNode],
    matches: dict[tuple[int, int], np.ndarray],
    options: VerificationOptions | None = None,
    seed: int = 0,
) -> SceneGraph:
    """Verify every putative pair and keep the pairs that enter the scene graph."""
    options = options or VerificationOptions()
    rng = np.random.default_rng(seed)
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
            np.asarray(pair_matches, dtype=int),
            options,
            rng,
        )
        if geometry is not None:
            graph.pairs.append(geometry)
    return graph


def pairwise_matches_from_descriptors(
    images: dict[int, ImageNode],
    ratio: float = 0.8,
) -> dict[tuple[int, int], np.ndarray]:
    """Exhaustive appearance matching. The paper's large-scale system instead
    queries a vocabulary tree; exhaustive search is the exact procedure those
    approximations replace (Section 2.1).
    """
    ids = sorted(images)
    found: dict[tuple[int, int], np.ndarray] = {}
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
