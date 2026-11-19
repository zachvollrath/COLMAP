"""RANSAC with the adaptive iteration bound used throughout the paper.

Minimal solvers are evaluated in batches on the CPU. One eight-point SVD is
too small to hide a GPU launch, and reading the inlier count after every
sample forces a device sync. A batch of a few hundred hypotheses is one SVD
and one reduction.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def adaptive_iterations(confidence: float, inlier_ratio: float, sample_size: int, max_iters: int, trial: int = 1) -> int:
    """RANSAC iteration bound log(1-η) / log(1-ε^s).

    A very small inlier ratio makes ε^s underflow to 0, so log(1-ε^s) is 0.
    That case cannot be certified inside the iteration budget, and the caller
    should run up to `max_iters`.
    """
    if inlier_ratio <= 0.0:
        return max_iters
    if inlier_ratio >= 1.0:
        return max(trial, 1)
    sample_probability = inlier_ratio**sample_size
    if sample_probability <= 0.0 or sample_probability >= 1.0:
        return max_iters if sample_probability <= 0.0 else max(trial, 1)
    denominator = math.log(1.0 - sample_probability)
    if denominator == 0.0 or not math.isfinite(denominator):
        return max_iters
    needed = math.ceil(math.log(1.0 - confidence) / denominator)
    return min(max_iters, max(trial, needed))


def ransac(
    n_data: int,
    sample_size: int,
    fit: Callable[[torch.Tensor], object],
    residuals: Callable[[object], torch.Tensor],
    threshold: float,
    rng: torch.Generator,
    confidence: float = 0.999,
    max_iters: int = 1000,
    min_inliers: int = 0,
) -> tuple[object | None, torch.Tensor | None]:
    """Fit a model. `residuals` returns one error per data row; inliers are error < threshold.

    `fit` may return a single model, a list of hypotheses, or None.
    """
    if n_data < sample_size:
        return None, None
    best_model = None
    best_inliers = None
    best_count = 0
    iterations = max_iters
    trial = 0
    while trial < iterations:
        sample = torch.randperm(n_data, generator=rng, device=rng.device)[:sample_size]
        fitted = fit(sample)
        trial += 1
        if fitted is None:
            continue
        hypotheses = fitted if isinstance(fitted, list) else [fitted]
        for model in hypotheses:
            if model is None:
                continue
            error = residuals(model)
            if error is None or len(error) != n_data:
                continue
            inliers = error < threshold
            count = int(inliers.sum().item())
            if count > best_count:
                best_count = count
                best_model = model
                best_inliers = inliers
                ratio = count / n_data
                iterations = adaptive_iterations(confidence, ratio, sample_size, max_iters, trial)
    if best_model is None or best_count < max(sample_size, min_inliers):
        return None, None
    return best_model, best_inliers


def _cpu_points(points: torch.Tensor) -> torch.Tensor:
    return points.detach().to(device="cpu", dtype=torch.float32)


def _batched_ransac(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    sample_size: int,
    evaluate,
    threshold: float,
    rng: torch.Generator,
    confidence: float,
    max_iters: int,
    min_inliers: int,
    batch_size: int = 256,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Draw samples without replacement, score a whole batch, keep the best model.

    `evaluate(sample_pts1, sample_pts2)` returns matrices (B, 3, 3), per-point
    errors (B, N), and a validity mask (B,). Work stays on the CPU; the winning
    matrix and inlier mask are returned on the caller's device.
    """
    device = pts1.device
    dtype = pts1.dtype
    pts1 = _cpu_points(pts1)
    pts2 = _cpu_points(pts2)
    n_data = int(pts1.shape[0])
    if n_data < sample_size:
        return None, None
    best_model = None
    best_inliers = None
    best_count = 0
    iterations = max_iters
    trial = 0
    while trial < iterations:
        batch = min(batch_size, iterations - trial)
        draw = torch.rand((batch, n_data), generator=rng)
        index = draw.topk(sample_size, dim=1).indices.to(device="cpu")
        trial += batch
        matrices, errors, valid = evaluate(pts1[index], pts2[index], pts1, pts2)
        counts = (errors < threshold).sum(dim=1)
        counts = torch.where(valid, counts, torch.zeros_like(counts))
        count = int(counts.max())
        if count <= best_count:
            continue
        winner = int(counts.argmax())
        best_count = count
        best_model = matrices[winner]
        best_inliers = errors[winner] < threshold
        iterations = adaptive_iterations(confidence, count / n_data, sample_size, max_iters, trial)
    if best_model is None or best_count < max(sample_size, min_inliers):
        return None, None
    return best_model.to(device=device, dtype=dtype), best_inliers.to(device=device)


def ransac_fundamental(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    threshold: float,
    rng: torch.Generator,
    confidence: float = 0.999,
    max_iters: int = 10000,
    min_inliers: int = 15,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Eight-point fundamental matrix. Samples explained by a homography are discarded."""
    from colmap.geometry import _fit_fundamentals, _fit_homographies, _sample_transfer_error, _sampson_batch

    def evaluate(sample1, sample2, all1, all2):
        homography, homography_ok = _fit_homographies(sample1, sample2)
        agreement = (_sample_transfer_error(homography, sample1, sample2) <= threshold).sum(dim=1)
        planar = homography_ok & (agreement >= sample1.shape[1] - 1)
        matrix, valid = _fit_fundamentals(sample1, sample2, essential=False)
        return matrix, _sampson_batch(matrix, all1, all2), valid & ~planar

    return _batched_ransac(pts1, pts2, 8, evaluate, threshold, rng, confidence, max_iters, min_inliers)


def ransac_homography(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    threshold: float,
    rng: torch.Generator,
    confidence: float = 0.999,
    max_iters: int = 10000,
    min_inliers: int = 4,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    from colmap.geometry import _fit_homographies, _transfer_batch

    def evaluate(sample1, sample2, all1, all2):
        matrix, valid = _fit_homographies(sample1, sample2)
        return matrix, _transfer_batch(matrix, all1, all2), valid

    return _batched_ransac(pts1, pts2, 4, evaluate, threshold, rng, confidence, max_iters, min_inliers)


def ransac_essential(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    threshold: float,
    rng: torch.Generator,
    confidence: float = 0.999,
    max_iters: int = 10000,
    min_inliers: int = 15,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Eight-point essential matrix on calibrated coordinates, skipping planar samples."""
    from colmap.geometry import _fit_fundamentals, _fit_homographies, _sample_transfer_error, _sampson_batch

    def evaluate(sample1, sample2, all1, all2):
        homography, homography_ok = _fit_homographies(sample1, sample2)
        agreement = (_sample_transfer_error(homography, sample1, sample2) <= threshold).sum(dim=1)
        planar = homography_ok & (agreement >= sample1.shape[1] - 1)
        matrix, valid = _fit_fundamentals(sample1, sample2, essential=True)
        return matrix, _sampson_batch(matrix, all1, all2), valid & ~planar

    return _batched_ransac(pts1, pts2, 8, evaluate, threshold, rng, confidence, max_iters, min_inliers)
