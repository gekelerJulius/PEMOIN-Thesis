"""Tests for road quality metrics."""

import numpy as np
import pytest

from pemoin.metrics.road import (
    compute_normal_stability,
    compute_plane_residuals,
    compute_smoothness,
)


class TestPlaneResiduals:
    def test_on_plane(self):
        """Points exactly on the plane should have ~0 residuals."""
        normal = np.array([0.0, 1.0, 0.0])
        offset = -2.0  # plane at y=2
        # Generate points at y=2 (on the plane: n^T x + d = y - 2 = 0)
        rng = np.random.RandomState(42)
        pts = rng.randn(100, 3)
        pts[:, 1] = 2.0  # all on the plane

        result = compute_plane_residuals(pts, normal, offset)
        assert result.rmse_m < 1e-6
        assert result.mean_m < 1e-6
        assert result.median_m < 1e-6

    def test_with_outliers(self):
        """Outliers should be captured by higher percentiles."""
        normal = np.array([0.0, 1.0, 0.0])
        offset = 0.0  # plane at y=0
        rng = np.random.RandomState(42)

        # 90 points close to plane, 10 far away
        pts_close = rng.randn(90, 3)
        pts_close[:, 1] = rng.normal(0, 0.01, 90)  # small deviation
        pts_far = rng.randn(10, 3)
        pts_far[:, 1] = 5.0  # 5m off

        pts = np.vstack([pts_close, pts_far])
        result = compute_plane_residuals(pts, normal, offset, percentiles=[50, 90, 99])

        # Median should be small (mostly inliers)
        assert result.median_m < 0.1
        # P99 should capture outliers
        assert result.percentiles[99.0] > 1.0

    def test_degenerate_normal(self):
        with pytest.raises(ValueError, match="[Dd]egenerate"):
            compute_plane_residuals(np.zeros((10, 3)), np.zeros(3), 0.0)

    def test_unnormalized_normal(self):
        """Should work with unnormalized normals."""
        normal = np.array([0.0, 2.0, 0.0])  # length 2
        offset = -4.0  # plane at y=2 after normalization
        pts = np.zeros((10, 3))
        pts[:, 1] = 2.0
        result = compute_plane_residuals(pts, normal, offset)
        assert result.rmse_m < 1e-6


class TestNormalStability:
    def test_static_normals(self):
        """Identical normals → angle ~0."""
        normals = np.tile([0.0, 1.0, 0.0], (20, 1))
        result = compute_normal_stability(normals)
        assert result.mean_angle_deg < 1e-6
        assert result.p95_angle_deg < 1e-6

    def test_rotating_normals(self):
        """Normals rotating by ~5° each frame should be detected."""
        n_frames = 20
        angles = np.linspace(0, np.radians(5) * (n_frames - 1), n_frames)
        normals = np.stack([
            np.sin(angles),
            np.cos(angles),
            np.zeros(n_frames),
        ], axis=1)
        result = compute_normal_stability(normals)
        # Each consecutive pair should differ by ~5°
        np.testing.assert_allclose(result.mean_angle_deg, 5.0, atol=0.5)

    def test_single_frame(self):
        """Single frame → empty result."""
        normals = np.array([[0.0, 1.0, 0.0]])
        result = compute_normal_stability(normals)
        assert result.mean_angle_deg == 0.0
        assert result.per_frame_angles_deg.size == 0

    def test_with_frame_indices(self):
        """Frame indices should be used for sorting."""
        # Data as stored (unsorted): indices [2, 0, 1] → normals row 0,1,2
        # After argsort by indices: order=[1,2,0] → normals [0,0,1],[0,1,0],[0,1,0]
        # Consecutive angles: 90°, 0° → mean=45°
        normals = np.array([
            [0, 1, 0],  # frame_index=2
            [0, 0, 1],  # frame_index=0
            [0, 1, 0],  # frame_index=1
        ], dtype=np.float64)
        indices = np.array([2, 0, 1])
        result = compute_normal_stability(normals, indices)
        np.testing.assert_allclose(result.mean_angle_deg, 45.0, atol=1.0)


class TestSmoothness:
    def test_flat_road(self):
        """Flat road should have low curvature."""
        rng = np.random.RandomState(42)
        # Points on flat plane y=0
        pts = rng.randn(500, 3) * 5
        pts[:, 1] = 0.0
        traj = np.column_stack([
            np.linspace(-5, 5, 20), np.zeros(20), np.zeros(20)
        ])
        result = compute_smoothness(pts, traj, window_size=5)
        assert result.mean_curvature < 0.01

    def test_bumpy_road(self):
        """Road with vertical variation should have higher curvature."""
        rng = np.random.RandomState(42)
        pts = rng.randn(500, 3) * 5
        # Add sinusoidal bumps
        pts[:, 1] = 0.5 * np.sin(pts[:, 0])
        traj = np.column_stack([
            np.linspace(-5, 5, 20), np.zeros(20), np.zeros(20)
        ])
        result = compute_smoothness(pts, traj, window_size=5)
        assert result.mean_curvature > 0.01

    def test_too_few_points(self):
        """Fewer than 4 points → zero curvature."""
        pts = np.zeros((3, 3))
        traj = np.zeros((5, 3))
        result = compute_smoothness(pts, traj)
        assert result.mean_curvature == 0.0
        assert result.per_window_curvatures.size == 0
