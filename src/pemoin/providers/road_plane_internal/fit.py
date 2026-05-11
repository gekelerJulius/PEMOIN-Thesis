"""Global plane fitting primitives."""

from __future__ import annotations

import numpy as np

from pemoin.geometry.plane import Plane


def solve_plane_weighted(
    *,
    points: np.ndarray,
    weights: np.ndarray,
    anchor_camera_center: np.ndarray,
    camera_height_m: float,
    lambda_up: float,
    lambda_temp: float,
    prev_plane: tuple[np.ndarray, float] | None,
    up_hint: np.ndarray | None = None,
    enforce_height_anchor: bool = True,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve a weighted plane fit.

    When `enforce_height_anchor=True`, the fit enforces the camera-height anchor exactly via
    `d = h - n^T c`. When `False`, it performs an unanchored weighted PCA plane fit
    (with optional temporal normal blending).
    """
    if points.size == 0:
        raise RuntimeError("No points provided to plane solver.")

    c = np.asarray(anchor_camera_center, dtype=np.float32).reshape(3)
    h = float(camera_height_m)
    pts = np.asarray(points, dtype=np.float32)
    w = np.clip(np.asarray(weights, dtype=np.float32).reshape(-1), 1e-8, None)
    if pts.shape[0] != w.shape[0]:
        raise RuntimeError("Point/weight length mismatch in plane solver.")

    if not enforce_height_anchor:
        wsum = float(np.sum(w))
        centroid = np.sum(pts * w[:, None], axis=0) / max(wsum, 1e-8)
        rel = pts - centroid[None, :]
        cov = (rel * w[:, None]).T @ rel / max(wsum, 1e-8)
        evals, evecs = np.linalg.eigh(cov.astype(np.float64))
        idx = int(np.argmin(evals))
        solved_n = np.asarray(evecs[:, idx], dtype=np.float32).reshape(3)
        n_norm = float(np.linalg.norm(solved_n))
        if n_norm < 1e-8:
            raise RuntimeError("Degenerate unanchored normal solve in road-plane fit.")
        solved_n = solved_n / n_norm
        up = np.asarray(
            up_hint if up_hint is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ).reshape(3)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1e-8:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up = up / up_norm
        if float(np.dot(solved_n, up)) < 0.0:
            solved_n = -solved_n
        if prev_plane is not None and lambda_temp > 0.0:
            prev_n, _ = prev_plane
            prev_n = np.asarray(prev_n, dtype=np.float32).reshape(3)
            mix = float(lambda_temp) / (float(lambda_temp) + max(wsum, 1e-3))
            solved_n = solved_n * (1.0 - mix) + prev_n * mix
            solved_n = solved_n / max(float(np.linalg.norm(solved_n)), 1e-8)
        solved = Plane(normal=solved_n, offset=float(-solved_n @ centroid))
        cov_diag = np.asarray(np.diag(cov), dtype=np.float32)
        return solved.normal.astype(np.float32), float(solved.offset), cov_diag

    rel = pts - c[None, :]
    # n^T (p - c) + h = 0 -> n^T (p - c) = -h
    a = rel
    b = np.full((rel.shape[0],), -h, dtype=np.float32)

    # Use sqrt(weights) for proper weighted least-squares.
    w_sqrt = np.sqrt(w.reshape(-1, 1)).astype(np.float32)
    a_w = a * w_sqrt
    b_w = b * w_sqrt[:, 0]

    if lambda_up > 0.0:
        up = np.asarray(up_hint if up_hint is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32), dtype=np.float32).reshape(3)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1e-8:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            up = up / up_norm
        w_u = np.sqrt(float(lambda_up))
        a_w = np.vstack([a_w, (np.eye(3, dtype=np.float32) * w_u)])
        b_w = np.concatenate([b_w, up * w_u])

    if prev_plane is not None:
        prev_n, _prev_d = prev_plane
        w_t = np.sqrt(max(float(lambda_temp), 0.0))
        temp_a3 = np.zeros((3, 3), dtype=np.float32)
        temp_a3[0, 0] = w_t
        temp_a3[1, 1] = w_t
        temp_a3[2, 2] = w_t
        prev_n = np.asarray(prev_n, dtype=np.float32).reshape(3)
        temp_b = prev_n * w_t
        a_w = np.vstack([a_w, temp_a3])
        b_w = np.concatenate([b_w, temp_b])

    params, *_ = np.linalg.lstsq(a_w, b_w, rcond=None)
    solved_n = np.asarray(params[:3], dtype=np.float32).reshape(3)
    solved_n_norm = float(np.linalg.norm(solved_n))
    if solved_n_norm < 1e-8:
        raise RuntimeError("Degenerate anchored normal solve in road-plane fit.")
    solved_n = solved_n / solved_n_norm
    solved = Plane(normal=solved_n, offset=float(h - solved_n @ c))

    solved = solved.enforce_normal_orientation(
        camera_center=c,
        target_height_m=h,
    )

    ata = a_w.T @ a_w
    cov = np.linalg.pinv(ata)
    cov_diag = np.diag(cov).astype(np.float32)
    return solved.normal.astype(np.float32), float(solved.offset), cov_diag


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Robust Huber weights for residual magnitudes."""
    delta = max(float(delta), 1e-6)
    abs_r = np.abs(residuals)
    weights = np.ones_like(abs_r, dtype=np.float32)
    mask = abs_r > delta
    weights[mask] = delta / abs_r[mask]
    return weights
