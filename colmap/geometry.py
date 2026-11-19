"""Projective geometry used by the incremental mapper.

Conventions:
- Extrinsics are world-to-camera: x_c = R x_w + t.
- Camera centers are C = -R^T t.
- The simple camera of Section 4.4 has one focal length, a principal point
  fixed at the image center, and one radial distortion coefficient.
"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


def calibration_matrix(focal: float, width: int, height: int) -> np.ndarray:
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def angle_axis_to_matrix(aa: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrix_to_angle_axis(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    return theta * axis / (2.0 * np.sin(theta))


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return (w, x, y, z) for the COLMAP text format."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def pixels_to_normalized(pixels: np.ndarray, focal: float, width: int, height: int, k1: float = 0.0) -> np.ndarray:
    """Undistort pixel measurements into the z=1 plane."""
    x = (pixels[:, 0] - width / 2.0) / focal
    y = (pixels[:, 1] - height / 2.0) / focal
    if abs(k1) < 1e-12:
        return np.column_stack([x, y])
    xu, yu = x.copy(), y.copy()
    for _ in range(8):
        r2 = xu * xu + yu * yu
        factor = 1.0 + k1 * r2
        xu = x / factor
        yu = y / factor
    return np.column_stack([xu, yu])


def project_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    focal: float,
    width: int,
    height: int,
    k1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points. Returns pixels (N, 2) and camera-frame depths (N,)."""
    Xc = (R @ points.T).T + t.reshape(1, 3)
    depth = Xc[:, 2]
    z = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2
    u = focal * (x * radial) + width / 2.0
    v = focal * (y * radial) + height / 2.0
    return np.column_stack([u, v]), depth


def projection_matrix(R: np.ndarray, t: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    K = calibration_matrix(focal, width, height)
    Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return K @ Rt


def triangulate_dlt(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Linear triangulation from (3x4 projection matrix, pixel) pairs."""
    if len(observations) < 2:
        return None
    rows = []
    for P, xy in observations:
        x, y = float(xy[0]), float(xy[1])
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.asarray(rows, dtype=float)
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def triangulation_angle_deg(center_a: np.ndarray, center_b: np.ndarray, point: np.ndarray) -> float:
    """Angle at the scene point between two camera centers (Eq. 3)."""
    va = center_a - point
    vb = center_b - point
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def viewing_direction(R: np.ndarray) -> np.ndarray:
    """Unit optical axis in world coordinates."""
    direction = R[2]
    n = np.linalg.norm(direction)
    return direction / n if n > 0 else direction


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(points - center, axis=1))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-12)
    T = np.array(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]]
    )
    hom = np.column_stack([points, np.ones(len(points))])
    normalized = (T @ hom.T).T
    return normalized[:, :2], T


def fundamental_from_points(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray | None:
    """Normalized eight-point fundamental matrix."""
    if len(pts1) < 8:
        return None
    n1, T1 = normalize_points(pts1)
    n2, T2 = normalize_points(pts2)
    A = np.column_stack(
        [
            n2[:, 0] * n1[:, 0],
            n2[:, 0] * n1[:, 1],
            n2[:, 0],
            n2[:, 1] * n1[:, 0],
            n2[:, 1] * n1[:, 1],
            n2[:, 1],
            n1[:, 0],
            n1[:, 1],
            np.ones(len(n1)),
        ]
    )
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    F = vh[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt
    F = T2.T @ F @ T1
    scale = F[2, 2] if abs(F[2, 2]) > 1e-12 else np.linalg.norm(F)
    if abs(scale) < 1e-15:
        return None
    return F / scale


def sampson_error(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([pts1, np.ones(len(pts1))])
    x2 = np.column_stack([pts2, np.ones(len(pts2))])
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numer = np.sum(x2 * Fx1, axis=1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return numer / np.maximum(denom, 1e-12)


def homography_from_points(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray | None:
    if len(pts1) < 4:
        return None
    n1, T1 = normalize_points(pts1)
    n2, T2 = normalize_points(pts2)
    rows = []
    for (x, y), (u, v) in zip(n1, n2):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    A = np.asarray(rows, dtype=float)
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    Hn = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    scale = H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    if abs(scale) < 1e-15:
        return None
    return H / scale


def homography_transfer_error(H: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    hom = np.column_stack([pts1, np.ones(len(pts1))])
    mapped = (H @ hom.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return np.sum((mapped - pts2) ** 2, axis=1)


def essential_from_points(pts1_n: np.ndarray, pts2_n: np.ndarray) -> np.ndarray | None:
    """Eight-point essential matrix on calibrated coordinates, projected onto the essential manifold."""
    F = fundamental_from_points(pts1_n, pts2_n)
    if F is None:
        return None
    U, S, Vt = np.linalg.svd(F)
    singular = 0.5 * (S[0] + S[1])
    E = U @ np.diag([singular, singular, 0.0]) @ Vt
    scale = np.linalg.norm(E)
    if scale < 1e-15:
        return None
    return E / scale


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Four (R, t) hypotheses. Translation has unit length."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def similarity_from_points(pts1: np.ndarray, pts2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """2D similarity mapping pts1 onto pts2. Returns scale, 2x2 rotation, translation."""
    if len(pts1) < 2:
        return None
    mu1 = pts1.mean(axis=0)
    mu2 = pts2.mean(axis=0)
    X = pts1 - mu1
    Y = pts2 - mu2
    cov = (Y.T @ X) / len(pts1)
    try:
        U, S, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
        S = S.copy()
        S[-1] *= -1
    var = float(np.sum(X ** 2) / len(pts1))
    if var < 1e-12:
        return None
    scale = float(np.sum(S) / var)
    if scale <= 0:
        return None
    translation = mu2 - scale * (R @ mu1)
    return scale, R, translation


def similarity_error(model: tuple[float, np.ndarray, np.ndarray], pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    scale, R, translation = model
    mapped = scale * (pts1 @ R.T) + translation
    return np.sum((mapped - pts2) ** 2, axis=1)


def dlt_pnp(world: np.ndarray, normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Linear PnP (n >= 6) on calibrated image coordinates, followed by a polar projection onto SO(3)."""
    if len(world) < 6:
        return None
    center = world.mean(axis=0)
    scale = np.mean(np.linalg.norm(world - center, axis=1))
    scale = max(float(scale), 1e-8)
    Xn = (world - center) / scale
    rows = []
    for (X, Y, Z), (u, v) in zip(Xn, normalized):
        rows.append([X, Y, Z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * X, -u * Y, -u * Z, -u])
        rows.append([0.0, 0.0, 0.0, 0.0, X, Y, Z, 1.0, -v * X, -v * Y, -v * Z, -v])
    A = np.asarray(rows, dtype=float)
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    P = vh[-1].reshape(3, 4)
    M = P[:, :3]
    if np.linalg.det(M) < 0:
        P = -P
        M = -M
    U, singular, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    factor = float(np.sum(singular) / 3.0)
    if abs(factor) < 1e-12:
        return None
    t_norm = P[:, 3] / factor
    # Undo the world-point normalization: x_c = R_n (X - c) / s + t_n
    # = (R_n / s) X + (t_n - R_n c / s). Fold 1/s into a metric pose by
    # rescaling so R stays a rotation: t = s * t_n - R c, R = R_n.
    t = scale * t_norm - R @ center
    return R, t


def refine_pose(
    R: np.ndarray,
    t: np.ndarray,
    world: np.ndarray,
    pixels: np.ndarray,
    focal: float,
    width: int,
    height: int,
    k1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear reprojection refinement of a single calibrated pose."""
    from scipy.optimize import least_squares

    def residual(params: np.ndarray) -> np.ndarray:
        Rc = angle_axis_to_matrix(params[:3])
        tc = params[3:6]
        uv, depth = project_points(world, Rc, tc, focal, width, height, k1)
        res = (uv - pixels).reshape(-1)
        behind = np.maximum(0.0, -depth)
        return np.concatenate([res, 10.0 * behind])

    start = np.concatenate([matrix_to_angle_axis(R), t])
    result = least_squares(residual, start, method="trf", loss="linear", max_nfev=20, ftol=1e-8, xtol=1e-8)
    return angle_axis_to_matrix(result.x[:3]), result.x[3:6]


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity taking src points onto dst. Returns scale, rotation, translation."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sign[2, 2] = -1.0
    R = U @ sign @ Vt
    var = float(np.sum(X ** 2) / len(src))
    scale = float(np.sum(D * np.diag(sign)) / max(var, 1e-12))
    translation = mu_d - scale * (R @ mu_s)
    return scale, R, translation
