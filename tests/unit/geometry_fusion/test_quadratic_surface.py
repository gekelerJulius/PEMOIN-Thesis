"""Tests for quadratic road surface fitting."""

import numpy as np
import pytest

from pemoin.data.contracts import PoseSample
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.stages.quadratic_surface import (
    QuadraticSurfaceResult,
    fit_quadratic_surfaces,
)


def _make_pose(frame_idx: int, position: np.ndarray) -> PoseSample:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = position
    return PoseSample(frame_index=frame_idx, camera_to_world=c2w)


class TestFitQuadraticSurfaces:
    def test_flat_surface(self):
        """A flat road should produce near-zero curvature coefficients."""
        rng = np.random.default_rng(42)
        n_points = 200
        # Points on z=0 plane (flat road)
        pts = np.column_stack([
            rng.uniform(-5, 5, n_points),
            rng.uniform(-3, 3, n_points),
            np.zeros(n_points) + rng.normal(0, 0.01, n_points),
        ]).astype(np.float32)

        pose = _make_pose(0, np.array([0, 0, 1.6], dtype=np.float32))
        normal = np.array([0, 0, 1], dtype=np.float32)
        settings = GeometryFusionSettings(quadratic_enabled=True, quadratic_lambda_curv=10.0)

        results = fit_quadratic_surfaces(
            [pts], [pose], [normal], settings
        )

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, QuadraticSurfaceResult)
        # Curvature coefficients (a, b, c) should be near zero
        assert abs(r.coeffs[0]) < 0.1  # a (x^2)
        assert abs(r.coeffs[1]) < 0.1  # b (xy)
        assert abs(r.coeffs[2]) < 0.1  # c (y^2)

    def test_quadratic_surface_recovery(self):
        """Recover a known quadratic surface z = 0.01*x^2."""
        rng = np.random.default_rng(42)
        n_points = 500
        x = rng.uniform(-5, 5, n_points).astype(np.float64)
        y = rng.uniform(-3, 3, n_points).astype(np.float64)
        z = 0.01 * x ** 2 + rng.normal(0, 0.005, n_points)
        pts = np.column_stack([x, y, z]).astype(np.float32)

        pose = _make_pose(0, np.array([0, 0, 1.6], dtype=np.float32))
        normal = np.array([0, 0, 1], dtype=np.float32)
        settings = GeometryFusionSettings(
            quadratic_enabled=True, quadratic_lambda_curv=0.1, quadratic_lambda_lin=0.1
        )

        results = fit_quadratic_surfaces([pts], [pose], [normal], settings)
        assert len(results) == 1
        # The coefficient for x^2 should be close to 0.01
        assert results[0].coeffs[0] == pytest.approx(0.01, abs=0.01)

    def test_disabled(self):
        """When quadratic is disabled, should return empty list."""
        settings = GeometryFusionSettings(quadratic_enabled=False)
        results = fit_quadratic_surfaces([], [], [], settings)
        assert results == []

    def test_too_few_points(self):
        """Frames with too few points should get zero coefficients."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        pose = _make_pose(0, np.array([0, 0, 1.6], dtype=np.float32))
        normal = np.array([0, 0, 1], dtype=np.float32)
        settings = GeometryFusionSettings(quadratic_enabled=True)

        results = fit_quadratic_surfaces([pts], [pose], [normal], settings)
        assert len(results) == 1
        assert results[0].confidence == 0.0
