"""Tests for per-frame quality gating."""

import numpy as np

from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.stages.quality_gating import (
    assess_quality,
    check_plateau_refit_needed,
)
from pemoin.providers.geometry_fusion.stages.road_rectification import FrameRectificationResult


def _make_rect(frame_idx: int, scale: float = 1.0, inlier_ratio: float = 0.8) -> FrameRectificationResult:
    return FrameRectificationResult(
        frame_index=frame_idx,
        normal_cam=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        offset_cam=1.6,
        implied_height_m=1.6,
        scale=scale,
        bias=0.0,
        inlier_ratio=inlier_ratio,
        residual_p90_m=0.05,
        support_count=500,
    )


class TestAssessQuality:
    def test_all_good_frames(self):
        """All frames pass quality gating."""
        results = [_make_rect(i, scale=1.0, inlier_ratio=0.9) for i in range(5)]
        settings = GeometryFusionSettings(gate_min_inlier=0.5, gate_max_height_err_m=0.25)
        reports = assess_quality(results, camera_height_m=1.6, settings=settings)
        assert all(r.quality_ok for r in reports)
        assert all(r.downweight > 0.9 for r in reports)

    def test_low_inlier_downweighted(self):
        """Frames with low inlier ratio get downweighted."""
        results = [_make_rect(0, inlier_ratio=0.3)]
        settings = GeometryFusionSettings(gate_min_inlier=0.5)
        reports = assess_quality(results, camera_height_m=1.6, settings=settings)
        assert not reports[0].quality_ok
        assert reports[0].downweight < 1.0

    def test_large_height_error(self):
        """Frames with large height error get flagged."""
        results = [_make_rect(0, scale=2.0)]  # h_hat * scale = 3.2 ≠ 1.6
        settings = GeometryFusionSettings(gate_max_height_err_m=0.25)
        reports = assess_quality(results, camera_height_m=1.6, settings=settings)
        assert not reports[0].quality_ok

    def test_scale_plateau_detection(self):
        """Large scale jumps are flagged as plateaus."""
        results = [
            _make_rect(0, scale=1.0),
            _make_rect(1, scale=1.2),  # 20% jump > 7% threshold
        ]
        settings = GeometryFusionSettings(plateau_scale_jump=0.07)
        reports = assess_quality(results, camera_height_m=1.6, settings=settings)
        assert reports[1].is_plateau


class TestCheckPlateauRefitNeeded:
    def test_no_plateau(self):
        results = [_make_rect(i, scale=1.0 + 0.01 * i) for i in range(5)]
        settings = GeometryFusionSettings(plateau_scale_jump=0.07)
        assert not check_plateau_refit_needed(results, settings)

    def test_plateau_detected(self):
        results = [
            _make_rect(0, scale=1.0),
            _make_rect(1, scale=1.0),
            _make_rect(2, scale=1.5),  # Big jump
        ]
        settings = GeometryFusionSettings(plateau_scale_jump=0.07)
        assert check_plateau_refit_needed(results, settings)
