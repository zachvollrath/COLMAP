"""RANSAC with the adaptive iteration bound used throughout the paper."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def ransac(
    n_data: int,
    sample_size: int,
    fit: Callable[[np.ndarray], object],
    residuals: Callable[[object], np.ndarray],
    threshold: float,
    rng: np.random.Generator,
    confidence: float = 0.999,
    max_iters: int = 1000,
    min_inliers: int = 0,
) -> tuple[object | None, np.ndarray | None]:
    """Fit a model. `residuals` returns one error per data row; inliers are error < threshold.

    `fit` may return a single model, a sequence of hypotheses, or None.
    """
    if n_data < sample_size:
        return None, None
    best_model = None
    best_inliers = None
    best_count = 0
    iterations = max_iters
    trial = 0
    while trial < iterations:
        sample = rng.choice(n_data, size=sample_size, replace=False)
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
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_model = model
                best_inliers = inliers
                ratio = count / n_data
                if 0.0 < ratio < 1.0:
                    iterations = min(
                        max_iters,
                        max(trial, math.ceil(math.log(1.0 - confidence) / math.log(1.0 - ratio**sample_size))),
                    )
    if best_model is None or best_count < max(sample_size, min_inliers):
        return None, None
    return best_model, best_inliers
