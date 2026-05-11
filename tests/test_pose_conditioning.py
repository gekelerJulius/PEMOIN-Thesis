"""Tests for pose conditioning pipeline."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pemoin.utils.trajectory_cleanup import (
    PoseConditioningSettings,
    TrajectoryCleanupOptions,
    cleanup_camera_to_world,
    condition_poses,
)


def _make_identity_c2w(n: int) -> np.ndarray:
    """Create N identity 4x4 matrices."""
    return np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))


def _make_straight_line_c2w(n: int, step: float = 1.0) -> np.ndarray:
    """Create N c2w matrices moving along Z axis."""
    c2w = _make_identity_c2w(n)
    for i in range(n):
        c2w[i, 2, 3] = i * step  # Z translation
    return c2w


class TestDisabledNoop:
    def test_disabled_noop(self):
        c2w = _make_straight_line_c2w(10)
        settings = PoseConditioningSettings(enabled=False)
        out, meta = condition_poses(c2w, settings)
        np.testing.assert_array_equal(out, c2w)
        assert "pose_conditioning_applied" not in meta


class TestOutlierRemoval:
    def test_outlier_removal(self):
        c2w = _make_straight_line_c2w(20, step=1.0)
        # Inject a spike at frame 10
        c2w[10, :3, 3] += [0, 0, 100.0]
        settings = PoseConditioningSettings(
            enabled=True,
            outlier_speed_factor=5.0,
            acceleration_window=0,  # disable accel smoothing
            rotation_window=0,  # disable rotation smoothing
            driving_prior_lambda=0.0,  # disable driving prior
        )
        out, meta = condition_poses(c2w, settings)
        # The spike should be interpolated away
        assert meta.get("outliers_interpolated", 0) > 0
        # Frame 10 translation should be close to 10.0 (linear interpolation)
        assert abs(out[10, 2, 3] - 10.0) < 2.0


class TestAccelerationSmoothing:
    def test_preserves_straight_line(self):
        c2w = _make_straight_line_c2w(20, step=1.0)
        settings = PoseConditioningSettings(
            enabled=True,
            outlier_speed_factor=0.0,  # disable outlier removal
            acceleration_window=5,
            acceleration_sigma=1.0,
            rotation_window=0,
            driving_prior_lambda=0.0,
        )
        out, meta = condition_poses(c2w, settings)
        # Straight line has zero acceleration, should be preserved
        np.testing.assert_allclose(
            out[:, 2, 3], c2w[:, 2, 3], atol=1e-6
        )

    def test_reduces_noise(self):
        rng = np.random.RandomState(42)
        c2w = _make_straight_line_c2w(50, step=1.0)
        noise = rng.randn(50, 3) * 0.5
        noise[0] = 0  # keep start clean
        c2w[:, :3, 3] += noise

        settings = PoseConditioningSettings(
            enabled=True,
            outlier_speed_factor=0.0,
            acceleration_window=7,
            acceleration_sigma=2.0,
            rotation_window=0,
            driving_prior_lambda=0.0,
        )
        out, meta = condition_poses(c2w, settings)
        assert meta.get("acceleration_smoothed")

        # Smoothed trajectory should have smaller velocity variance
        vel_orig = np.diff(c2w[:, :3, 3], axis=0)
        vel_smooth = np.diff(out[:, :3, 3], axis=0)
        assert np.var(vel_smooth) < np.var(vel_orig)


class TestSlerpSmoothing:
    def test_preserves_identity(self):
        c2w = _make_identity_c2w(20)
        # Add some translation to avoid <3 frames issues
        for i in range(20):
            c2w[i, 2, 3] = float(i)
        settings = PoseConditioningSettings(
            enabled=True,
            outlier_speed_factor=0.0,
            acceleration_window=0,
            rotation_window=5,
            driving_prior_lambda=0.0,
        )
        out, meta = condition_poses(c2w, settings)
        # Identity rotations should stay identity
        for i in range(20):
            np.testing.assert_allclose(
                out[i, :3, :3], np.eye(3), atol=1e-6
            )

    def test_reduces_rotation_jitter(self):
        rng = np.random.RandomState(42)
        n = 30
        c2w = _make_identity_c2w(n)
        for i in range(n):
            c2w[i, 2, 3] = float(i)
        # Add small random rotation perturbations
        for i in range(n):
            angles = rng.randn(3) * 0.05  # small perturbation in radians
            R_pert = Rotation.from_rotvec(angles).as_matrix()
            c2w[i, :3, :3] = R_pert

        settings = PoseConditioningSettings(
            enabled=True,
            outlier_speed_factor=0.0,
            acceleration_window=0,
            rotation_window=7,
            driving_prior_lambda=0.0,
        )
        out, meta = condition_poses(c2w, settings)
        assert meta.get("rotation_smoothed")

        # Compute angular distances from identity
        orig_angles = []
        smooth_angles = []
        for i in range(n):
            orig_angles.append(
                np.linalg.norm(Rotation.from_matrix(c2w[i, :3, :3]).as_rotvec())
            )
            smooth_angles.append(
                np.linalg.norm(Rotation.from_matrix(out[i, :3, :3]).as_rotvec())
            )
        # Smoothed should have lower variance
        assert np.var(smooth_angles) < np.var(orig_angles)


class TestDrivingPrior:
    def test_preserves_yaw(self):
        n = 20
        c2w = _make_identity_c2w(n)
        for i in range(n):
            c2w[i, 2, 3] = float(i)
        # Apply yaw-only rotations
        yaw_angles = np.linspace(0, 0.5, n)
        for i in range(n):
            R = Rotation.from_euler("Y", yaw_angles[i]).as_matrix()
            c2w[i, :3, :3] = R

        settings = PoseConditioningSettings(
            enabled=True,
            outlier_speed_factor=0.0,
            acceleration_window=0,
            rotation_window=0,
            driving_prior_lambda=0.5,
            driving_prior_window=5,
        )
        out, meta = condition_poses(c2w, settings)

        # Extract yaw from output
        out_euler = Rotation.from_matrix(out[:, :3, :3]).as_euler("YXZ")
        np.testing.assert_allclose(out_euler[:, 0], yaw_angles, atol=1e-6)

    def test_shrinks_roll_deviation(self):
        n = 30
        c2w = _make_identity_c2w(n)
        for i in range(n):
            c2w[i, 2, 3] = float(i)
        # Apply large roll at frame 15
        euler = np.zeros((n, 3))  # YXZ
        euler[15, 2] = 0.5  # large roll at frame 15
        for i in range(n):
            R = Rotation.from_euler("YXZ", euler[i]).as_matrix()
            c2w[i, :3, :3] = R

        settings = PoseConditioningSettings(
            enabled=True,
            outlier_speed_factor=0.0,
            acceleration_window=0,
            rotation_window=0,
            driving_prior_lambda=0.5,
            driving_prior_window=11,
        )
        out, meta = condition_poses(c2w, settings)
        assert meta.get("driving_prior_applied")

        # Roll at frame 15 should be pulled toward 0
        out_euler = Rotation.from_matrix(out[:, :3, :3]).as_euler("YXZ")
        assert abs(out_euler[15, 2]) < abs(euler[15, 2])


class TestBackwardCompat:
    def test_legacy_api(self):
        opts = TrajectoryCleanupOptions.from_mapping({
            "trajectory_cleanup_enabled": True,
            "trajectory_cleanup_outlier_speed_factor": 5.0,
        })
        assert opts.enabled is True
        assert opts.outlier_speed_factor == 5.0

        c2w = _make_straight_line_c2w(10)
        result, meta = cleanup_camera_to_world(c2w, opts)
        # No outliers in straight line, should return unchanged
        assert result.shape == c2w.shape


class TestSettingsFromMapping:
    def test_parses_correctly(self):
        settings = PoseConditioningSettings.from_mapping({
            "enabled": True,
            "outlier_speed_factor": 3.0,
            "acceleration_window": 7,
            "acceleration_sigma": 2.0,
            "rotation_window": 9,
            "driving_prior_lambda": 0.3,
            "driving_prior_window": 15,
        })
        assert settings.enabled is True
        assert settings.outlier_speed_factor == 3.0
        assert settings.acceleration_window == 7
        assert settings.acceleration_sigma == 2.0
        assert settings.rotation_window == 9
        assert settings.driving_prior_lambda == 0.3
        assert settings.driving_prior_window == 15

    def test_defaults_on_none(self):
        settings = PoseConditioningSettings.from_mapping(None)
        assert settings.enabled is False
        assert settings.outlier_speed_factor == 5.0

    def test_defaults_on_empty(self):
        settings = PoseConditioningSettings.from_mapping({})
        assert settings.enabled is False
