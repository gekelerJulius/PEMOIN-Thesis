"""Tests for RANSAC + IRLS plane fitting."""

import numpy as np
import pytest

from pemoin.providers.geometry_fusion.utils.plane_fitting import (
    PlaneResult,
    huber_weights,
    ransac_irls_plane_fit,
)


def _make_planar_points(
    normal: np.ndarray,
    offset: float,
    n_points: int = 500,
    noise_std: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate points on a plane with optional noise.

    For the plane n·p + d = 0, generates points by finding two tangent vectors
    and sampling in the tangent plane, then adding noise along the normal.
    """
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Find a point on the plane: p0 such that n·p0 + d = 0
    # Choose p0 = -d * n (projection of origin onto plane)
    p0 = -offset * normal

    # Find two tangent vectors
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, np.array([1, 0, 0], dtype=np.float64))
    else:
        t1 = np.cross(normal, np.array([0, 1, 0], dtype=np.float64))
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 = t2 / np.linalg.norm(t2)

    # Sample in tangent plane
    coords = rng.uniform(-5, 5, size=(n_points, 2)).astype(np.float64)
    points = p0[None, :] + coords[:, 0:1] * t1[None, :] + coords[:, 1:2] * t2[None, :]

    if noise_std > 0:
        points += rng.normal(0, noise_std, points.shape)

    return points.astype(np.float32)


class TestHuberWeights:
    def test_small_residuals_weight_one(self):
        residuals = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        w = huber_weights(residuals, delta=0.1)
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0], atol=1e-5)

    def test_large_residuals_downweighted(self):
        residuals = np.array([0.01, 0.5, 1.0], dtype=np.float32)
        w = huber_weights(residuals, delta=0.1)
        assert w[0] == pytest.approx(1.0, abs=1e-5)
        assert w[1] < 1.0
        assert w[2] < w[1]


class TestRansacIrlsPlaneFit:
    def test_exact_plane_recovery(self):
        """Recover a known plane from clean points."""
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        offset = -1.6  # Camera at height 1.6
        points = _make_planar_points(normal, offset, n_points=500, noise_std=0.001)
        weights = np.ones(points.shape[0], dtype=np.float32)

        result = ransac_irls_plane_fit(
            points, weights, iters=500, inlier_thresh=0.05, huber_delta=0.05, seed=42
        )
        assert isinstance(result, PlaneResult)
        # Normal should be close to [0, 1, 0] or [0, -1, 0]
        cos_angle = abs(float(np.dot(result.normal, normal)))
        assert cos_angle > 0.99, f"Normal mismatch: got {result.normal}, expected ~{normal}"
        assert abs(abs(result.offset) - abs(offset)) < 0.05
        assert result.inlier_ratio > 0.9

    def test_noisy_plane_with_outliers(self):
        """Recover plane from noisy points with 20% outliers."""
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        offset = 2.0
        rng = np.random.default_rng(123)
        points = _make_planar_points(normal, offset, n_points=400, noise_std=0.02, seed=123)
        # Add 20% outliers
        outliers = rng.uniform(-5, 5, size=(100, 3)).astype(np.float32)
        all_points = np.vstack([points, outliers])
        weights = np.ones(all_points.shape[0], dtype=np.float32)

        result = ransac_irls_plane_fit(
            all_points, weights, iters=1000, inlier_thresh=0.1, huber_delta=0.08, seed=42
        )
        cos_angle = abs(float(np.dot(result.normal, normal)))
        assert cos_angle > 0.95
        assert result.inlier_ratio > 0.6

    def test_too_few_points_raises(self):
        """Should raise if fewer than 3 points."""
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        weights = np.ones(2, dtype=np.float32)
        with pytest.raises(Exception):
            ransac_irls_plane_fit(points, weights, iters=10, seed=42)

    def test_implied_height_from_plane(self):
        """Verify implied height calculation: h = |offset| / |normal_y|."""
        # Camera looking forward: road is at Y=-1.6 in camera coords
        normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        offset = 1.6  # n·p + d = 0 → -y + 1.6 = 0 → y = 1.6 (road below at y=1.6 in Y-down)
        h_hat = abs(offset) / abs(normal[1])
        assert h_hat == pytest.approx(1.6, abs=0.001)
