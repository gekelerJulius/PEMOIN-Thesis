"""Tests for road-anchored depth rectification."""

import numpy as np
import pytest

from pemoin.providers.geometry_fusion.stages.road_rectification import (
    FrameRectificationResult,
    optimize_temporal_smoothness,
)
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings


class TestOptimizeTemporalSmoothness:
    def _make_results(self, scales: list[float], biases: list[float] | None = None) -> list[FrameRectificationResult]:
        biases = biases or [0.0] * len(scales)
        results = []
        for i, (s, b) in enumerate(zip(scales, biases)):
            results.append(
                FrameRectificationResult(
                    frame_index=i,
                    normal_cam=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                    offset_cam=1.6 / s,
                    implied_height_m=1.6 / s,
                    scale=s,
                    bias=b,
                    inlier_ratio=0.9,
                    residual_p90_m=0.05,
                    support_count=500,
                )
            )
        return results

    def test_smooth_scales_unchanged(self):
        """Already smooth scales should remain approximately the same."""
        scales = [1.0, 1.01, 1.02, 1.01, 1.0]
        results = self._make_results(scales)
        settings = GeometryFusionSettings(lambda_s=5.0, lbfgs_maxiter=50)
        smoothed = optimize_temporal_smoothness(results, camera_height_m=1.6, settings=settings)
        smoothed_scales = [r.scale for r in smoothed]
        for s_orig, s_smooth in zip(scales, smoothed_scales):
            assert abs(s_orig - s_smooth) < 0.1

    def test_outlier_scale_smoothed(self):
        """A single outlier scale should be pulled toward neighbors."""
        scales = [1.0, 1.0, 2.0, 1.0, 1.0]  # Frame 2 is an outlier
        results = self._make_results(scales)
        settings = GeometryFusionSettings(lambda_s=10.0, lbfgs_maxiter=100)
        smoothed = optimize_temporal_smoothness(results, camera_height_m=1.6, settings=settings)
        # The outlier should be pulled closer to 1.0
        assert smoothed[2].scale < 2.0

    def test_single_frame(self):
        """Single frame should be returned unchanged."""
        results = self._make_results([1.5])
        settings = GeometryFusionSettings()
        smoothed = optimize_temporal_smoothness(results, camera_height_m=1.6, settings=settings)
        assert len(smoothed) == 1
        assert smoothed[0].scale == pytest.approx(1.5, abs=1e-5)

    def test_affine_mode(self):
        """Affine mode should optimize both scale and bias."""
        scales = [1.0, 1.0, 1.0]
        results = self._make_results(scales)
        settings = GeometryFusionSettings(affine_mode="affine", lbfgs_maxiter=50)
        smoothed = optimize_temporal_smoothness(results, camera_height_m=1.6, settings=settings)
        # Should converge without error
        assert len(smoothed) == 3
        for r in smoothed:
            assert np.isfinite(r.scale)
            assert np.isfinite(r.bias)
