from __future__ import annotations

import numpy as np
import pytest

from pemoin.visualization.blender_scene.grounding_policy import (
    compute_support_relock_metrics,
    resolve_effective_hold_frames,
)


def test_resolve_effective_hold_frames_uses_time_based_budget_when_present() -> None:
    frames, seconds = resolve_effective_hold_frames(
        hold_frames=10,
        hold_seconds=5.0,
        sampling_fps=10.0,
    )

    assert frames == 50
    assert seconds == pytest.approx(5.0)


def test_resolve_effective_hold_frames_falls_back_to_frame_budget() -> None:
    frames, seconds = resolve_effective_hold_frames(
        hold_frames=10,
        hold_seconds=None,
        sampling_fps=10.0,
    )

    assert frames == 10
    assert seconds is None


def test_support_relock_metrics_compare_planes_at_common_anchor() -> None:
    current_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    previous_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    comparison_anchor = np.array([1.0, 2.0, 0.05], dtype=np.float32)
    current_anchor = np.array([1.02, 2.0, 0.05], dtype=np.float32)
    previous_anchor = np.array([1.0, 2.0, -0.77], dtype=np.float32)

    metrics = compute_support_relock_metrics(
        current_normal=current_normal,
        current_offset=-0.82,
        previous_normal=previous_normal,
        previous_offset=-0.82,
        comparison_anchor=comparison_anchor,
        current_anchor=current_anchor,
        previous_anchor=previous_anchor,
    )

    assert metrics.normal_jump_deg == pytest.approx(0.0)
    assert metrics.support_height_jump_m == pytest.approx(0.0)
    assert metrics.current_signed_distance_m == pytest.approx(-0.77)
    assert metrics.previous_signed_distance_m == pytest.approx(-0.77)
    assert metrics.anchor_shift_m == pytest.approx(0.02)
