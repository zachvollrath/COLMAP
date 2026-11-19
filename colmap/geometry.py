"""Projective geometry used by the incremental mapper.

Conventions:
- Extrinsics are world-to-camera: x_c = R x_w + t.
- Camera centers are C = -R^T t.
- The simple camera of Section 4.4 has one focal length, a principal point
  fixed at the image center, and one radial distortion coefficient.

Numeric work uses PyTorch in float64 so bundle adjustment stays well conditioned.
The minimal solvers inside RANSAC are the exception: they run in float32 on the
CPU, in batches. Those problems are only a few points wide, so a GPU launch
costs more than the arithmetic, and this card's float64 rate is a small
fraction of its float32 rate. Projection and the angle-axis map stay
differentiable so bundle adjustment can take analytic Jacobians.
"""

from __future__ import annotations

import torch

from colmap.device import get_device
from colmap.optimize import levenberg_marquardt

DTYPE = torch.float64


def as_tensor(value, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Tensor on the configured device, without copying when it is already there."""
    device = get_device()
    if isinstance(value, torch.Tensor):
        if value.dtype == dtype and value.device == device:
            return value
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v
    zero = v.new_zeros(())
    return torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )


def camera_center(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -R.transpose(0, 1) @ t


def calibration_matrix(focal: float, width: int, height: int) -> torch.Tensor:
    return torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )


def angle_axis_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(aa).clamp(min=1e-12)
    k = aa / theta
    K = skew(k)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def matrix_to_angle_axis(R: torch.Tensor) -> torch.Tensor:
    R = as_tensor(R)
    cos_theta = ((torch.trace(R) - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if float(theta) < 1e-12:
        return torch.zeros(3, dtype=R.dtype, device=R.device)
    axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * axis / (2.0 * torch.sin(theta))


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Return (w, x, y, z) for the COLMAP text format."""
    R = as_tensor(R)
    trace = torch.trace(R)
    if float(trace) > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif float(R[0, 0]) > float(R[1, 1]) and float(R[0, 0]) > float(R[2, 2]):
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif float(R[1, 1]) > float(R[2, 2]):
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    n = torch.linalg.norm(q)
    if float(n) == 0.0:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    return q / n


def pixels_to_normalized(
    pixels: torch.Tensor,
    focal: float,
    width: int,
    height: int,
    k1: float = 0.0,
) -> torch.Tensor:
    """Undistort pixel measurements into the z=1 plane."""
    pixels = as_tensor(pixels)
    distorted_x = (pixels[:, 0] - width / 2.0) / focal
    distorted_y = (pixels[:, 1] - height / 2.0) / focal
    x, y = distorted_x, distorted_y
    for _ in range(8):
        r2 = x * x + y * y
        factor = 1.0 + k1 * r2
        x = distorted_x / factor
        y = distorted_y / factor
    return torch.column_stack([x, y])


def project_points(
    points: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    focal: float | torch.Tensor,
    width: int,
    height: int,
    k1: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points. Returns pixels (N, 2) and camera-frame depths (N,)."""
    points = points if isinstance(points, torch.Tensor) else as_tensor(points)
    Xc = points @ R.transpose(0, 1) + t.reshape(1, 3)
    depth = Xc[:, 2]
    z = torch.where(depth.abs() < 1e-12, torch.full_like(depth, 1e-12), depth)
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2
    u = focal * (x * radial) + width / 2.0
    v = focal * (y * radial) + height / 2.0
    return torch.column_stack([u, v]), depth


def projection_matrix(R: torch.Tensor, t: torch.Tensor, focal: float, width: int, height: int) -> torch.Tensor:
    K = calibration_matrix(focal, width, height).to(device=R.device, dtype=R.dtype)
    Rt = torch.cat([R, t.reshape(3, 1)], dim=1)
    return K @ Rt


def triangulate_dlt(observations: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor | None:
    """Linear triangulation from (3x4 projection matrix, pixel) pairs."""
    if len(observations) < 2:
        return None
    rows = []
    for P, xy in observations:
        x = xy[0]
        y = xy[1]
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = torch.stack([as_tensor(row) for row in rows])
    try:
        _, _, vh = torch.linalg.svd(A, full_matrices=True)
    except RuntimeError:
        return None
    X = vh[-1]
    if abs(float(X[3])) < 1e-12:
        return None
    return X[:3] / X[3]


def triangulation_angle_deg(center_a: torch.Tensor, center_b: torch.Tensor, point: torch.Tensor) -> float:
    """Angle at the scene point between two camera centers."""
    va = center_a - point
    vb = center_b - point
    na = torch.linalg.norm(va)
    nb = torch.linalg.norm(vb)
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return 0.0
    cosine = torch.dot(va, vb) / (na * nb)
    cosine = cosine.clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def viewing_direction(R: torch.Tensor) -> torch.Tensor:
    """Unit optical axis in world coordinates."""
    direction = R[2]
    n = torch.linalg.norm(direction)
    if float(n) == 0.0:
        return direction
    return direction / n


def normalize_points(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = as_tensor(points)
    center = points.mean(dim=0)
    mean_dist = torch.linalg.norm(points - center, dim=1).mean().clamp(min=1e-12)
    scale = 2.0**0.5 / mean_dist
    T = torch.zeros((3, 3), dtype=DTYPE)
    T[0, 0] = scale
    T[1, 1] = scale
    T[0, 2] = -scale * center[0]
    T[1, 2] = -scale * center[1]
    T[2, 2] = 1.0
    hom = torch.column_stack([points, torch.ones(len(points), dtype=DTYPE)])
    normalized = (T @ hom.T).T
    return normalized[:, :2], T


def _nullspace(A: torch.Tensor) -> torch.Tensor | None:
    try:
        _, _, vh = torch.linalg.svd(A, full_matrices=True)
    except RuntimeError:
        return None
    return vh[-1]


def fundamental_from_points(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor | None:
    """Normalized eight-point fundamental matrix."""
    pts1 = as_tensor(pts1)
    pts2 = as_tensor(pts2)
    if len(pts1) < 8:
        return None
    n1, T1 = normalize_points(pts1)
    n2, T2 = normalize_points(pts2)
    ones = torch.ones(len(n1), dtype=DTYPE)
    A = torch.column_stack(
        [
            n2[:, 0] * n1[:, 0],
            n2[:, 0] * n1[:, 1],
            n2[:, 0],
            n2[:, 1] * n1[:, 0],
            n2[:, 1] * n1[:, 1],
            n2[:, 1],
            n1[:, 0],
            n1[:, 1],
            ones,
        ]
    )
    null = _nullspace(A)
    if null is None:
        return None
    F = null.reshape(3, 3)
    U, S, Vt = torch.linalg.svd(F, full_matrices=True)
    S = S.clone()
    S[2] = 0.0
    F = U @ torch.diag(S) @ Vt
    F = T2.T @ F @ T1
    scale = F[2, 2] if abs(float(F[2, 2])) > 1e-12 else torch.linalg.norm(F)
    if abs(float(scale)) < 1e-15:
        return None
    return F / scale


def sampson_error(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    pts1 = as_tensor(pts1)
    pts2 = as_tensor(pts2)
    x1 = torch.column_stack([pts1, torch.ones(len(pts1), dtype=DTYPE)])
    x2 = torch.column_stack([pts2, torch.ones(len(pts2), dtype=DTYPE)])
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numer = torch.sum(x2 * Fx1, dim=1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return numer / denom.clamp(min=1e-12)


def homography_from_points(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor | None:
    pts1 = as_tensor(pts1)
    pts2 = as_tensor(pts2)
    if len(pts1) < 4:
        return None
    n1, T1 = normalize_points(pts1)
    n2, T2 = normalize_points(pts2)
    x, y = n1[:, 0], n1[:, 1]
    u, v = n2[:, 0], n2[:, 1]
    zero = torch.zeros(len(n1), dtype=DTYPE)
    one = torch.ones(len(n1), dtype=DTYPE)
    A = torch.cat(
        [
            torch.stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u], dim=1),
            torch.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], dim=1),
        ],
        dim=0,
    )
    null = _nullspace(A)
    if null is None:
        return None
    Hn = null.reshape(3, 3)
    H = torch.linalg.inv(T2) @ Hn @ T1
    scale = H[2, 2] if abs(float(H[2, 2])) > 1e-12 else torch.linalg.norm(H)
    if abs(float(scale)) < 1e-15:
        return None
    return H / scale


def homography_transfer_error(H: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    pts1 = as_tensor(pts1)
    pts2 = as_tensor(pts2)
    hom = torch.column_stack([pts1, torch.ones(len(pts1), dtype=DTYPE)])
    mapped = (H @ hom.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return torch.sum((mapped - pts2) ** 2, dim=1)


def sample_is_planar(pts1: torch.Tensor, pts2: torch.Tensor, squared_threshold: float) -> bool:
    """A homography explains this sample, so the eight-point problem is degenerate.

    Points on a dominant plane (or a pure rotation) admit a pencil of fundamental
    matrices with zero algebraic error. Eight-point RANSAC then prefers that
    pencil over the true epipolar geometry. One mismatched point is tolerated
    so a single noisy measurement does not hide the degeneracy.
    """
    pts1 = as_tensor(pts1)
    pts2 = as_tensor(pts2)
    if len(pts1) < 4:
        return False
    homography = homography_from_points(pts1, pts2)
    if homography is None:
        return False
    error = homography_transfer_error(homography, pts1, pts2)
    agreed = int(torch.count_nonzero(error <= squared_threshold).item())
    return agreed >= max(len(pts1) - 1, 4)


def essential_from_points(pts1_n: torch.Tensor, pts2_n: torch.Tensor) -> torch.Tensor | None:
    """Eight-point essential matrix on calibrated coordinates, projected onto the essential manifold."""
    F = fundamental_from_points(pts1_n, pts2_n)
    if F is None:
        return None
    U, S, Vt = torch.linalg.svd(F, full_matrices=True)
    singular = 0.5 * (S[0] + S[1])
    E = U @ torch.diag(torch.stack([singular, singular, S.new_zeros(())])) @ Vt
    scale = torch.linalg.norm(E)
    if float(scale) < 1e-15:
        return None
    return E / scale


def _normalize_samples(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hartley normalization for a batch of samples. points is (B, S, 2)."""
    center = points.mean(dim=1)
    delta = points - center[:, None, :]
    mean_dist = torch.linalg.norm(delta, dim=2).mean(dim=1).clamp(min=1e-12)
    scale = 2.0**0.5 / mean_dist
    normalized = delta * scale[:, None, None]
    transform = torch.zeros((points.shape[0], 3, 3), dtype=points.dtype, device=points.device)
    transform[:, 0, 0] = scale
    transform[:, 1, 1] = scale
    transform[:, 0, 2] = -scale * center[:, 0]
    transform[:, 1, 2] = -scale * center[:, 1]
    transform[:, 2, 2] = 1.0
    return normalized, transform


def _fit_homographies(pts1: torch.Tensor, pts2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Homography for each sample. pts are (B, S, 2), S >= 4."""
    n1, t1 = _normalize_samples(pts1)
    n2, t2 = _normalize_samples(pts2)
    x, y = n1[..., 0], n1[..., 1]
    u, v = n2[..., 0], n2[..., 1]
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    upper = torch.stack((-x, -y, -one, zero, zero, zero, u * x, u * y, u), dim=-1)
    lower = torch.stack((zero, zero, zero, -x, -y, -one, v * x, v * y, v), dim=-1)
    design = torch.cat((upper, lower), dim=1)
    _, _, vh = torch.linalg.svd(design, full_matrices=True)
    homography = vh[:, -1, :].reshape(-1, 3, 3)
    homography = torch.linalg.solve(t2, torch.matmul(homography, t1))
    scale = homography[:, 2, 2]
    norm = torch.linalg.norm(homography.reshape(homography.shape[0], -1), dim=1)
    scale = torch.where(scale.abs() > 1e-8, scale, norm)
    valid = scale.abs() > 1e-12
    homography = homography / scale[:, None, None].clamp(min=1e-12)
    return homography, valid


def _fit_fundamentals(pts1: torch.Tensor, pts2: torch.Tensor, essential: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Fundamental, or essential, matrix for each 8-point sample. pts are (B, 8, 2)."""
    n1, t1 = _normalize_samples(pts1)
    n2, t2 = _normalize_samples(pts2)
    ones = torch.ones(n1.shape[:2], dtype=n1.dtype, device=n1.device)
    design = torch.stack(
        (
            n2[..., 0] * n1[..., 0],
            n2[..., 0] * n1[..., 1],
            n2[..., 0],
            n2[..., 1] * n1[..., 0],
            n2[..., 1] * n1[..., 1],
            n2[..., 1],
            n1[..., 0],
            n1[..., 1],
            ones,
        ),
        dim=-1,
    )
    _, _, vh = torch.linalg.svd(design, full_matrices=True)
    matrix = vh[:, -1, :].reshape(-1, 3, 3)
    left, singular, right = torch.linalg.svd(matrix, full_matrices=True)
    singular = singular.clone()
    singular[:, 2] = 0.0
    matrix = left @ torch.diag_embed(singular) @ right
    # Undo Hartley normalization in the same order as fundamental_from_points.
    # The essential constraint is applied after that, on the calibrated matrix.
    matrix = torch.matmul(t2.transpose(1, 2), torch.matmul(matrix, t1))
    if essential:
        left, singular, right = torch.linalg.svd(matrix, full_matrices=True)
        averaged = 0.5 * (singular[:, 0] + singular[:, 1])
        singular = torch.stack((averaged, averaged, torch.zeros_like(averaged)), dim=1)
        matrix = left @ torch.diag_embed(singular) @ right
        scale = torch.linalg.norm(matrix.reshape(matrix.shape[0], -1), dim=1)
    else:
        scale = matrix[:, 2, 2]
        norm = torch.linalg.norm(matrix.reshape(matrix.shape[0], -1), dim=1)
        scale = torch.where(scale.abs() > 1e-8, scale, norm)
    valid = scale.abs() > 1e-12
    matrix = matrix / scale[:, None, None].clamp(min=1e-12)
    return matrix, valid


def _sampson_batch(matrix: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """Sampson error of each matrix against every correspondence. Returns (B, N)."""
    ones = torch.ones(pts1.shape[0], dtype=pts1.dtype, device=pts1.device)
    x1 = torch.column_stack((pts1, ones))
    x2 = torch.column_stack((pts2, ones))
    fx1 = torch.einsum("bij,nj->bni", matrix, x1)
    ftx2 = torch.einsum("bji,nj->bni", matrix, x2)
    numer = torch.sum(x2 * fx1, dim=-1) ** 2
    denom = fx1[..., 0] ** 2 + fx1[..., 1] ** 2 + ftx2[..., 0] ** 2 + ftx2[..., 1] ** 2
    return numer / denom.clamp(min=1e-12)


def _transfer_batch(homography: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(pts1.shape[0], dtype=pts1.dtype, device=pts1.device)
    hom = torch.column_stack((pts1, ones))
    mapped = torch.einsum("bij,nj->bni", homography, hom)
    mapped = mapped[..., :2] / mapped[..., 2:3].clamp(min=1e-12)
    return torch.sum((mapped - pts2) ** 2, dim=-1)


def _sample_transfer_error(homography: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """Transfer error of each homography on its own sample. pts are (B, S, 2)."""
    ones = torch.ones((*pts1.shape[:2], 1), dtype=pts1.dtype, device=pts1.device)
    hom = torch.cat((pts1, ones), dim=-1)
    mapped = torch.einsum("bij,bsj->bsi", homography, hom)
    mapped = mapped[..., :2] / mapped[..., 2:3].clamp(min=1e-12)
    return torch.sum((mapped - pts2) ** 2, dim=-1)


def triangulate_two_views(
    projection_a: torch.Tensor,
    pixels_a: torch.Tensor,
    projection_b: torch.Tensor,
    pixels_b: torch.Tensor,
) -> torch.Tensor:
    """DLT for many correspondences that share two projection matrices. Returns (N, 3)."""
    pixels_a = pixels_a.reshape(-1, 2)
    pixels_b = pixels_b.reshape(-1, 2)

    def rows(projection: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                pixels[:, :1] * projection[2] - projection[0],
                pixels[:, 1:] * projection[2] - projection[1],
            ),
            dim=1,
        )

    design = torch.cat((rows(projection_a, pixels_a), rows(projection_b, pixels_b)), dim=1)
    _, _, vh = torch.linalg.svd(design, full_matrices=True)
    point = vh[:, -1, :]
    return point[:, :3] / point[:, 3:4]


def decompose_essential(E: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Four (R, t) hypotheses. Translation has unit length."""
    U, _, Vt = torch.linalg.svd(E, full_matrices=True)
    if float(torch.linalg.det(U)) < 0:
        U = -U
    if float(torch.linalg.det(Vt)) < 0:
        Vt = -Vt
    W = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=E.dtype, device=E.device)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    if float(torch.linalg.det(R1)) < 0:
        R1 = -R1
    if float(torch.linalg.det(R2)) < 0:
        R2 = -R2
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def similarity_from_points(pts1: torch.Tensor, pts2: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor] | None:
    """2D similarity mapping pts1 onto pts2. Returns scale, 2x2 rotation, translation."""
    pts1 = as_tensor(pts1)
    pts2 = as_tensor(pts2)
    if len(pts1) < 2:
        return None
    mu1 = pts1.mean(dim=0)
    mu2 = pts2.mean(dim=0)
    X = pts1 - mu1
    Y = pts2 - mu2
    cov = (Y.T @ X) / len(pts1)
    try:
        U, S, Vt = torch.linalg.svd(cov, full_matrices=True)
    except RuntimeError:
        return None
    R = U @ Vt
    if float(torch.linalg.det(R)) < 0:
        U = U.clone()
        U[:, -1] *= -1
        R = U @ Vt
        S = S.clone()
        S[-1] *= -1
    var = float(torch.sum(X ** 2) / len(pts1))
    if var < 1e-12:
        return None
    scale = float(torch.sum(S) / var)
    if scale <= 0:
        return None
    translation = mu2 - scale * (R @ mu1)
    return scale, R, translation


def similarity_error(
    model: tuple[float, torch.Tensor, torch.Tensor],
    pts1: torch.Tensor,
    pts2: torch.Tensor,
) -> torch.Tensor:
    scale, R, translation = model
    pts1 = as_tensor(pts1)
    pts2 = as_tensor(pts2)
    mapped = scale * (pts1 @ R.T) + translation
    return torch.sum((mapped - pts2) ** 2, dim=1)


def dlt_pnp(world: torch.Tensor, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Linear PnP (n >= 6) on calibrated image coordinates, followed by a polar projection onto SO(3)."""
    world = as_tensor(world)
    normalized = as_tensor(normalized)
    if len(world) < 6:
        return None
    center = world.mean(dim=0)
    scale = torch.linalg.norm(world - center, dim=1).mean().clamp(min=1e-8)
    Xn = (world - center) / scale
    rows = []
    zero = Xn.new_zeros(())
    one = Xn.new_ones(())
    for (X, Y, Z), (u, v) in zip(Xn, normalized):
        rows.append(torch.stack([X, Y, Z, one, zero, zero, zero, zero, -u * X, -u * Y, -u * Z, -u]))
        rows.append(torch.stack([zero, zero, zero, zero, X, Y, Z, one, -v * X, -v * Y, -v * Z, -v]))
    A = torch.stack(rows)
    null = _nullspace(A)
    if null is None:
        return None
    P = null.reshape(3, 4)
    M = P[:, :3]
    if float(torch.linalg.det(M)) < 0:
        P = -P
        M = -M
    U, singular, Vt = torch.linalg.svd(M, full_matrices=True)
    R = U @ Vt
    if float(torch.linalg.det(R)) < 0:
        U = U.clone()
        U[:, -1] *= -1
        R = U @ Vt
    factor = torch.sum(singular) / 3.0
    if abs(float(factor)) < 1e-12:
        return None
    t_norm = P[:, 3] / factor
    t = scale * t_norm - R @ center
    return R, t


def refine_pose(
    R: torch.Tensor,
    t: torch.Tensor,
    world: torch.Tensor,
    pixels: torch.Tensor,
    focal: float,
    width: int,
    height: int,
    k1: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nonlinear reprojection refinement of a single calibrated pose."""
    world = as_tensor(world)
    pixels = as_tensor(pixels)
    start = torch.cat([matrix_to_angle_axis(R), as_tensor(t)])

    def residual(params: torch.Tensor) -> torch.Tensor:
        Rc = angle_axis_to_matrix(params[:3])
        tc = params[3:6]
        uv, depth = project_points(world, Rc, tc, focal, width, height, k1)
        behind = torch.clamp(-depth, min=0.0)
        return torch.cat([(uv - pixels).reshape(-1), 10.0 * behind])

    refined, _ = levenberg_marquardt(start, residual, max_iterations=15, scale=1.0)
    return angle_axis_to_matrix(refined[:3]), refined[3:6]


def umeyama(src: torch.Tensor, dst: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Similarity taking src points onto dst. Returns scale, rotation, translation."""
    src = as_tensor(src)
    dst = as_tensor(dst)
    mu_s = src.mean(dim=0)
    mu_d = dst.mean(dim=0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / len(src)
    U, D, Vt = torch.linalg.svd(cov, full_matrices=True)
    sign = torch.eye(3, dtype=DTYPE)
    if float(torch.linalg.det(U) * torch.linalg.det(Vt)) < 0:
        sign[2, 2] = -1.0
    R = U @ sign @ Vt
    var = float(torch.sum(X ** 2) / len(src))
    scale = float(torch.sum(D * torch.diag(sign)) / max(var, 1e-12))
    translation = mu_d - scale * (R @ mu_s)
    return scale, R, translation
