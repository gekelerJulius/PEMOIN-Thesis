"""RANSAC + IRLS robust plane fitting for geometry fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlaneResult:
    """Result of a robust plane fit."""

    normal: np.ndarray  # (3,) unit normal
    offset: float  # d in n·p + d = 0
    inlier_ratio: float
    residual_p90: float
    inlier_mask: np.ndarray  # (N,) bool
    weights: np.ndarray  # (N,) final weights


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Compute Huber weights from absolute residuals."""
    delta = max(float(delta), 1e-6)
    abs_r = np.abs(residuals)
    w = np.ones_like(abs_r, dtype=np.float32)
    mask = abs_r > delta
    if np.any(mask):
        w[mask] = delta / np.maximum(abs_r[mask], 1e-8)
    return w


def _weighted_plane_fit(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane using weighted PCA."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}.")
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 1e-12:
        raise RuntimeError("Degenerate weighted plane fit: non-positive weight sum.")
    pts = np.asarray(points, dtype=np.float64)
    centroid = (pts * w[:, None]).sum(axis=0) / w_sum
    centered = pts - centroid[None, :]
    cov = (centered * w[:, None]).T @ centered / w_sum
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = int(np.argmin(eigvals))
    normal = eigvecs[:, idx]
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-9:
        raise RuntimeError("Degenerate weighted plane fit: near-zero normal.")
    normal = normal / n_norm
    offset = -float(normal @ centroid)
    return normal.astype(np.float32), float(offset)


def _plane_from_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Fit a plane from three points. Returns None if degenerate."""
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return None
    n = n / norm
    d = -float(n @ a)
    return n.astype(np.float32), d


def ransac_irls_plane_fit(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    iters: int = 2000,
    inlier_thresh: float = 0.06,
    huber_delta: float = 0.08,
    irls_iters: int = 10,
    seed: int = 42,
) -> PlaneResult:
    """Robust plane fit using RANSAC initialization followed by IRLS refinement.

    Args:
        points: (N, 3) point cloud.
        weights: (N,) per-point confidence weights.
        iters: RANSAC iterations.
        inlier_thresh: RANSAC inlier distance threshold in meters.
        huber_delta: Huber delta for IRLS weighting.
        irls_iters: Number of IRLS iterations.
        seed: Random seed for reproducibility.

    Returns:
        PlaneResult with fitted plane parameters and diagnostics.
    """
    n_points = points.shape[0]
    rng = np.random.default_rng(seed=seed)

    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(iters):
        ids = rng.choice(n_points, size=3, replace=False)
        candidate = _plane_from_three_points(points[ids[0]], points[ids[1]], points[ids[2]])
        if candidate is None:
            continue
        normal, offset = candidate
        residuals = np.abs(points @ normal + float(offset))
        inliers = residuals <= inlier_thresh
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        raise RuntimeError("RANSAC plane fit failed: no valid plane found.")

    p_in = points[best_inliers]
    w_in = weights[best_inliers]
    normal, offset = _weighted_plane_fit(p_in, w_in)

    # IRLS refinement
    for _ in range(irls_iters):
        residuals = np.abs(p_in @ normal + float(offset))
        w_huber = huber_weights(residuals, huber_delta)
        normal, offset = _weighted_plane_fit(p_in, w_in * w_huber)

    # Final diagnostics on all points
    all_residuals = np.abs(points @ normal + float(offset))
    final_inliers = all_residuals <= inlier_thresh
    inlier_ratio = float(np.mean(final_inliers))
    p90 = float(np.percentile(all_residuals, 90))
    final_weights = weights * huber_weights(all_residuals, huber_delta)

    return PlaneResult(
        normal=normal,
        offset=offset,
        inlier_ratio=inlier_ratio,
        residual_p90=p90,
        inlier_mask=final_inliers,
        weights=final_weights,
    )
