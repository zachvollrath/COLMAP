"""Recursive RANSAC triangulation, Section 4.3.

A feature track is treated as an outlier-contaminated set of measurements.
Unique two-view samples are triangulated with the DLT, accepted when the
triangulation angle and the cheirality constraint hold, and scored by how many
other measurements reproject inside the error threshold. The consensus is
removed and the procedure repeats so that a track merged from several scene
points can yield more than one structure point. Recursion stops when the
latest consensus is smaller than three.
"""

from __future__ import annotations

import math

import numpy as np

from colmap.geometry import (
    camera_center,
    project_points,
    projection_matrix,
    triangulate_dlt,
    triangulation_angle_deg,
)
from colmap.scene_graph import PANORAMIC, Camera


class ViewMeasurement:
    __slots__ = ("image_id", "feature_idx", "xy", "R", "t", "camera", "panoramic_with")

    def __init__(
        self,
        image_id: int,
        feature_idx: int,
        xy: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        camera: Camera,
        panoramic_with: set[int] | None = None,
    ) -> None:
        self.image_id = image_id
        self.feature_idx = feature_idx
        self.xy = np.asarray(xy, dtype=float)
        self.R = R
        self.t = t
        self.camera = camera
        self.panoramic_with = panoramic_with or set()


def _well_conditioned(
    point: np.ndarray,
    first: ViewMeasurement,
    second: ViewMeasurement,
    min_angle_deg: float,
) -> bool:
    if second.image_id in first.panoramic_with or first.image_id in second.panoramic_with:
        return False
    depth1 = float((first.R @ point + first.t)[2])
    depth2 = float((second.R @ point + second.t)[2])
    if depth1 <= 0.0 or depth2 <= 0.0:
        return False
    angle = triangulation_angle_deg(
        camera_center(first.R, first.t),
        camera_center(second.R, second.t),
        point,
    )
    return angle >= min_angle_deg


def _reprojection_error(point: np.ndarray, measurement: ViewMeasurement) -> tuple[float, float]:
    uv, depth = project_points(
        point.reshape(1, 3),
        measurement.R,
        measurement.t,
        measurement.camera.focal,
        measurement.camera.width,
        measurement.camera.height,
        measurement.camera.k1,
    )
    error = float(np.linalg.norm(uv[0] - measurement.xy))
    return error, float(depth[0])


def _refine_point(point: np.ndarray, measurements: list[ViewMeasurement]) -> np.ndarray:
    """Gauss-Newton refinement of one scene point against its consensus set."""
    current = point.astype(float).copy()
    for _ in range(10):
        jac = np.zeros((2 * len(measurements), 3))
        res = np.zeros(2 * len(measurements))
        eps = 1e-6
        base_pixels = []
        for index, measurement in enumerate(measurements):
            uv, _ = project_points(
                current.reshape(1, 3),
                measurement.R,
                measurement.t,
                measurement.camera.focal,
                measurement.camera.width,
                measurement.camera.height,
                measurement.camera.k1,
            )
            base_pixels.append(uv[0])
            res[2 * index : 2 * index + 2] = uv[0] - measurement.xy
        for axis in range(3):
            shifted = current.copy()
            shifted[axis] += eps
            for index, measurement in enumerate(measurements):
                uv, _ = project_points(
                    shifted.reshape(1, 3),
                    measurement.R,
                    measurement.t,
                    measurement.camera.focal,
                    measurement.camera.width,
                    measurement.camera.height,
                    measurement.camera.k1,
                )
                jac[2 * index : 2 * index + 2, axis] = (uv[0] - base_pixels[index]) / eps
        try:
            delta, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        current = current + delta
        if float(np.linalg.norm(delta)) < 1e-10:
            break
    return current


def triangulate_track(
    measurements: list[ViewMeasurement],
    min_angle_deg: float,
    max_error_px: float,
    confidence: float,
    min_inlier_ratio: float,
    min_consensus: int = 3,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, list[ViewMeasurement]]]:
    """Return every well-conditioned point recovered from a track."""
    rng = rng or np.random.default_rng(0)
    remaining = list(measurements)
    recovered: list[tuple[np.ndarray, list[ViewMeasurement]]] = []
    while len(remaining) >= min_consensus:
        point, consensus = _ransac_consensus(
            remaining,
            min_angle_deg,
            max_error_px,
            confidence,
            min_inlier_ratio,
            rng,
        )
        if point is None or consensus is None or len(consensus) < min_consensus:
            break
        refined = _refine_point(point, consensus)
        # Keep the consensus that still agrees with the refined point.
        kept = []
        for measurement in consensus:
            error, depth = _reprojection_error(refined, measurement)
            if depth > 0.0 and error <= max_error_px:
                kept.append(measurement)
        if len(kept) < min_consensus:
            break
        recovered.append((refined, kept))
        consumed = {(m.image_id, m.feature_idx) for m in kept}
        remaining = [m for m in remaining if (m.image_id, m.feature_idx) not in consumed]
    return recovered


def _ransac_consensus(
    measurements: list[ViewMeasurement],
    min_angle_deg: float,
    max_error_px: float,
    confidence: float,
    min_inlier_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, list[ViewMeasurement] | None]:
    n = len(measurements)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    best_point = None
    best_consensus: list[ViewMeasurement] = []
    inlier_ratio = min(max(min_inlier_ratio, 1e-6), 1.0)
    max_trials = len(pairs)
    if inlier_ratio < 1.0:
        max_trials = min(max_trials, math.ceil(math.log(1.0 - confidence) / math.log(1.0 - inlier_ratio**2)))
    trials = 0
    for i, j in pairs:
        if trials >= max_trials:
            break
        trials += 1
        first = measurements[i]
        second = measurements[j]
        if second.image_id in first.panoramic_with:
            continue
        point = triangulate_dlt(
            [
                (
                    projection_matrix(first.R, first.t, first.camera.focal, first.camera.width, first.camera.height),
                    first.xy,
                ),
                (
                    projection_matrix(
                        second.R, second.t, second.camera.focal, second.camera.width, second.camera.height
                    ),
                    second.xy,
                ),
            ]
        )
        if point is None or not _well_conditioned(point, first, second, min_angle_deg):
            continue
        consensus = []
        for measurement in measurements:
            error, depth = _reprojection_error(point, measurement)
            if depth > 0.0 and error <= max_error_px:
                consensus.append(measurement)
        if len(consensus) > len(best_consensus):
            best_consensus = consensus
            best_point = point
            inlier_ratio = max(inlier_ratio, len(consensus) / n)
            if inlier_ratio < 1.0:
                max_trials = min(
                    len(pairs),
                    math.ceil(math.log(1.0 - confidence) / math.log(1.0 - inlier_ratio**2)),
                )
            else:
                break
    if best_point is None:
        return None, None
    return best_point, best_consensus


def two_view_points(
    measurements_a: list[ViewMeasurement],
    measurements_b: list[ViewMeasurement],
    min_angle_deg: float,
    max_error_px: float,
) -> list[tuple[np.ndarray, ViewMeasurement, ViewMeasurement]]:
    """Triangulate a seed pair. Two-view points are the one exception to the
    three-observation consensus required by the recursive estimator.
    """
    points = []
    for first, second in zip(measurements_a, measurements_b):
        point = triangulate_dlt(
            [
                (
                    projection_matrix(first.R, first.t, first.camera.focal, first.camera.width, first.camera.height),
                    first.xy,
                ),
                (
                    projection_matrix(
                        second.R, second.t, second.camera.focal, second.camera.width, second.camera.height
                    ),
                    second.xy,
                ),
            ]
        )
        if point is None or not _well_conditioned(point, first, second, min_angle_deg):
            continue
        error1, depth1 = _reprojection_error(point, first)
        error2, depth2 = _reprojection_error(point, second)
        if depth1 > 0 and depth2 > 0 and error1 <= max_error_px and error2 <= max_error_px:
            points.append((point, first, second))
    return points


def panoramic_pairs(pairs: list, model_name: str = PANORAMIC) -> dict[int, set[int]]:
    linked: dict[int, set[int]] = {}
    for pair in pairs:
        if pair.model != model_name:
            continue
        linked.setdefault(pair.image_id1, set()).add(pair.image_id2)
        linked.setdefault(pair.image_id2, set()).add(pair.image_id1)
    return linked
