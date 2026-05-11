"""Per-frame quality assessment and gating.

Evaluates depth rectification quality per frame and assigns downweight factors.
Detects scale plateaus and flags frames for investigation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.stages.road_rectification import FrameRectificationResult
from pemoin.utils.logging import get_logger

LOG = get_logger()


@dataclass
class FrameQualityReport:
    """Quality assessment for a single frame."""

    frame_index: int
    inlier_ratio: float
    height_error_m: float
    normal_stable: bool
    is_plateau: bool
    quality_ok: bool
    downweight: float  # 1.0 = full trust, 0.0 = reject


def assess_quality(
    rect_results: list[FrameRectificationResult],
    camera_height_m: float,
    settings: GeometryFusionSettings,
) -> list[FrameQualityReport]:
    """Assess per-frame quality of depth rectification.

    Rules:
    - inlier_ratio < gate_min_inlier -> downweight road constraints
    - |h_hat - h| > gate_max_height_err_m for >20% of frames -> flag warning
    - delta_s > plateau_scale_jump (7% jump) -> flag as plateau

    Args:
        rect_results: Per-frame rectification results.
        camera_height_m: Target camera height.
        settings: Geometry fusion settings.

    Returns:
        List of FrameQualityReport, one per frame.
    """
    reports: list[FrameQualityReport] = []
    n = len(rect_results)

    for i, r in enumerate(rect_results):
        height_err = abs(r.implied_height_m * r.scale + r.bias - camera_height_m)

        # Check inlier ratio
        low_inlier = r.inlier_ratio < settings.gate_min_inlier

        # Check for scale plateau (large jump from previous frame)
        is_plateau = False
        if i > 0:
            delta_s = abs(r.scale - rect_results[i - 1].scale)
            if delta_s > settings.plateau_scale_jump:
                is_plateau = True

        # Check normal stability (compare with neighbors)
        normal_stable = True
        if i > 0:
            cos_angle = float(np.dot(r.normal_cam, rect_results[i - 1].normal_cam))
            if cos_angle < 0.95:
                normal_stable = False

        # Quality determination
        quality_ok = (
            not low_inlier
            and height_err <= settings.gate_max_height_err_m
            and normal_stable
        )

        # Downweight: smooth transition based on quality metrics
        downweight = 1.0
        if low_inlier:
            downweight *= max(0.1, r.inlier_ratio / settings.gate_min_inlier)
        if height_err > settings.gate_max_height_err_m:
            downweight *= max(0.1, settings.gate_max_height_err_m / max(height_err, 1e-6))
        if is_plateau:
            downweight *= 0.5

        reports.append(
            FrameQualityReport(
                frame_index=r.frame_index,
                inlier_ratio=r.inlier_ratio,
                height_error_m=height_err,
                normal_stable=normal_stable,
                is_plateau=is_plateau,
                quality_ok=quality_ok,
                downweight=float(np.clip(downweight, 0.0, 1.0)),
            )
        )

    # Global check: too many bad frames
    bad_ratio = sum(1 for r in reports if not r.quality_ok) / max(n, 1)
    if bad_ratio > settings.da3_trigger_height_err_pct:
        LOG.warning(
            "Geometry fusion: %.0f%% of frames failed quality gating (threshold: %.0f%%). "
            "Consider investigating depth or semantics quality.",
            bad_ratio * 100,
            settings.da3_trigger_height_err_pct * 100,
        )

    return reports


def check_plateau_refit_needed(
    rect_results: list[FrameRectificationResult],
    settings: GeometryFusionSettings,
) -> bool:
    """Check if scale plateaus require a refit with increased lambda_s.

    Returns True if any frame shows a scale jump exceeding the threshold.
    """
    for i in range(1, len(rect_results)):
        delta_s = abs(rect_results[i].scale - rect_results[i - 1].scale)
        if delta_s > settings.plateau_scale_jump:
            return True
    return False
