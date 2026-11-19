"""Levenberg-Marquardt with PyTorch autograd Jacobians.

Bundle adjustment and pose refinement minimize a Cauchy-robust reprojection
cost. The Jacobian is obtained with forward-mode automatic differentiation
instead of finite differences.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch


def cauchy(residual: torch.Tensor, scale: float) -> torch.Tensor:
    scale = max(float(scale), 1e-6)
    return torch.log1p((residual / scale) ** 2).sum()


def levenberg_marquardt(
    x0: torch.Tensor,
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    max_iterations: int,
    scale: float = 1.0,
    ftol: float = 1e-10,
) -> tuple[torch.Tensor, bool]:
    """Minimize sum(log(1 + (r/scale)^2)). Returns the parameter vector and whether a step was accepted."""
    x = x0.detach().clone()
    residual = residual_fn(x)
    cost = float(cauchy(residual, scale).detach())
    damping = 1e-3
    accepted = False
    for _ in range(max(max_iterations, 1)):
        jacobian = _jacobian(residual_fn, x)
        residual = residual_fn(x).detach()
        weights = 1.0 / (1.0 + (residual / max(scale, 1e-6)) ** 2)
        normal = jacobian.T @ (jacobian * weights[:, None])
        gradient = jacobian.T @ (weights * residual)
        diagonal = normal.diagonal().clamp(min=1e-8)
        improved = False
        for _attempt in range(8):
            system = normal + damping * torch.diag(diagonal)
            try:
                step = torch.linalg.solve(system, -gradient)
            except RuntimeError:
                damping *= 10.0
                continue
            if not torch.isfinite(step).all():
                damping *= 10.0
                continue
            candidate = (x + step).detach()
            candidate_cost = float(cauchy(residual_fn(candidate), scale).detach())
            if candidate_cost < cost:
                x = candidate
                cost = candidate_cost
                damping = max(damping * 0.3, 1e-8)
                accepted = True
                improved = True
                if float(step.norm()) < 1e-12 or abs(candidate_cost) < ftol:
                    return x, True
                break
            damping *= 4.0
        if not improved:
            break
    return x, accepted or cost < 1e-8


def _jacobian(residual_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    def wrapped(point: torch.Tensor) -> torch.Tensor:
        return residual_fn(point)

    # Vectorized forward-mode is fast, but this PyTorch build emits a jit.script
    # warning on Python 3.14. The Jacobian itself is unaffected.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="`torch.jit.script` is not supported")
        try:
            jacobian = torch.autograd.functional.jacobian(wrapped, x, vectorize=True, strategy="forward-mode")
        except (RuntimeError, NotImplementedError, ValueError):
            jacobian = torch.autograd.functional.jacobian(wrapped, x, strategy="forward-mode")
    return jacobian.detach()
