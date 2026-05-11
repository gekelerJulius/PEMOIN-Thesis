"""Tests for trajectory quality metrics."""

import numpy as np
import pytest

from pemoin.metrics.trajectory import (
    align_trajectories_umeyama,
    compute_ate,
    compute_rpe,
    compute_scale_drift,
)


def _make_poses(positions: np.ndarray) -> np.ndarray:
    """Create identity-rotation poses from positions (N, 3) → (N, 4, 4)."""
    n = positions.shape[0]
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses[:, :3, :3] = np.eye(3)
    poses[:, :3, 3] = positions
    poses[:, 3, 3] = 1.0
    return poses


def _make_circle_trajectory(n: int = 50, radius: float = 5.0) -> np.ndarray:
    """Generate a circular trajectory in the XZ plane."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.stack([radius * np.cos(t), np.zeros(n), radius * np.sin(t)], axis=1)
    return _make_poses(positions)


# --- Umeyama alignment tests ---

class TestUmeyamaAlignment:
    def test_identity(self):
        pts = np.random.RandomState(42).randn(20, 3)
        result = align_trajectories_umeyama(pts, pts)
        assert abs(result.scale - 1.0) < 1e-6
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(result.translation, np.zeros(3), atol=1e-6)

    def test_with_scale(self):
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 3)
        scale = 2.0
        scaled = pts * scale
        result = align_trajectories_umeyama(pts, scaled, with_scale=True)
        assert abs(result.scale - scale) < 1e-4
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-4)

    def test_with_rotation(self):
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 3)
        # 90° rotation around Z axis
        angle = np.pi / 2
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        rotated = (R @ pts.T).T
        result = align_trajectories_umeyama(pts, rotated, with_scale=False)
        np.testing.assert_allclose(result.rotation, R, atol=1e-6)
        assert abs(result.scale - 1.0) < 1e-6

    def test_with_translation(self):
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 3)
        offset = np.array([5.0, -3.0, 2.0])
        shifted = pts + offset
        result = align_trajectories_umeyama(pts, shifted, with_scale=False)
        np.testing.assert_allclose(result.translation, offset, atol=1e-6)
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-6)

    def test_too_few_points(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 3"):
            align_trajectories_umeyama(pts, pts)


# --- ATE tests ---

class TestATE:
    def test_perfect_match(self):
        poses = _make_circle_trajectory(50)
        ate = compute_ate(poses, poses)
        assert ate.rmse_m < 1e-6
        assert ate.mean_m < 1e-6
        assert ate.max_m < 1e-6

    def test_with_known_error(self):
        gt = _make_circle_trajectory(50)
        est = gt.copy()
        shift = 0.5
        est[:, 0, 3] += shift  # shift X by 0.5m
        ate = compute_ate(est, gt, align=False)
        np.testing.assert_allclose(ate.rmse_m, shift, atol=1e-6)
        np.testing.assert_allclose(ate.mean_m, shift, atol=1e-6)
        assert len(ate.per_frame_errors) == 50

    def test_with_alignment_corrects_offset(self):
        gt = _make_circle_trajectory(50)
        est = gt.copy()
        est[:, :3, 3] += np.array([10.0, 0.0, 0.0])
        ate = compute_ate(est, gt, align=True, with_scale=False)
        # Umeyama should correct pure translation
        assert ate.rmse_m < 1e-4


# --- RPE tests ---

class TestRPE:
    def test_identity(self):
        poses = _make_circle_trajectory(50)
        rpe = compute_rpe(poses, poses, delta_frames=1)
        assert rpe.trans_rmse < 1e-6
        assert rpe.rot_rmse_deg < 1e-4
        assert rpe.delta_frames == 1

    def test_with_drift(self):
        n = 50
        gt = _make_circle_trajectory(n)
        est = gt.copy()
        # Add linearly increasing drift
        for i in range(n):
            est[i, 0, 3] += i * 0.01
        rpe = compute_rpe(est, gt, delta_frames=1, align=False)
        # Each relative motion has ~0.01m extra translation
        assert rpe.trans_rmse > 0.005
        assert len(rpe.per_pair_trans_errors) == n - 1

    def test_delta_too_large(self):
        poses = _make_poses(np.zeros((5, 3)))
        with pytest.raises(ValueError, match="delta_frames"):
            compute_rpe(poses, poses, delta_frames=5)

    def test_multiple_deltas(self):
        poses = _make_circle_trajectory(50)
        for delta in [1, 5, 10]:
            rpe = compute_rpe(poses, poses, delta_frames=delta)
            assert rpe.trans_rmse < 1e-6
            assert rpe.delta_frames == delta


# --- Scale drift tests ---

class TestScaleDrift:
    def test_no_drift(self):
        poses = _make_circle_trajectory(50)
        sd = compute_scale_drift(poses, poses, window=10, stride=5)
        np.testing.assert_allclose(sd.scale_factors, 1.0, atol=1e-6)
        assert sd.drift_per_100m < 1e-4

    def test_detection(self):
        n = 60
        gt_pos = np.column_stack([
            np.linspace(0, 10, n),
            np.zeros(n),
            np.zeros(n),
        ])
        gt = _make_poses(gt_pos)
        est = gt.copy()
        # Scale estimated trajectory by increasing factor
        for i in range(n):
            factor = 1.0 + i * 0.01
            est[i, 0, 3] = gt[i, 0, 3] * factor
        sd = compute_scale_drift(est, gt, window=10, stride=5)
        # Scale factors should increase
        assert sd.scale_factors[-1] > sd.scale_factors[0]
        assert sd.drift_per_100m > 0

    def test_window_too_large(self):
        poses = _make_poses(np.zeros((5, 3)))
        with pytest.raises(ValueError, match="window"):
            compute_scale_drift(poses, poses, window=10)

    def test_window_too_small(self):
        poses = _make_poses(np.zeros((5, 3)))
        with pytest.raises(ValueError, match="window must be >= 3"):
            compute_scale_drift(poses, poses, window=2)
