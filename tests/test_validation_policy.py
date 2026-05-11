from __future__ import annotations

import importlib
import sys

import pytest

from pemoin.validation.policy import (
    AdaptiveValidationContext,
    ValidationPolicySettings,
    resolve_effective_sampling_fps,
)


def test_validation_policy_is_neutral_at_reference_fps():
    policy = ValidationPolicySettings(enabled=True, reference_sampling_fps=10.0)
    adaptive = AdaptiveValidationContext.from_runtime(
        policy,
        {"frame_provider_info": {"tool": "test", "settings": {"sampling_fps": 10.0}}},
    )
    assert adaptive.soft_threshold_scale == pytest.approx(1.0)
    assert adaptive.max_thresholds(7.0) == pytest.approx((7.0, 9.45))


def test_validation_policy_relaxes_thresholds_below_reference_fps():
    policy = ValidationPolicySettings(enabled=True, reference_sampling_fps=10.0)
    adaptive = AdaptiveValidationContext.from_runtime(
        policy,
        {"frame_provider_info": {"tool": "test", "settings": {"sampling_fps": 4.0}}},
    )
    assert adaptive.soft_threshold_scale == pytest.approx((10.0 / 4.0) ** 0.5)
    soft, hard = adaptive.max_thresholds(7.0)
    assert soft > 7.0
    assert hard > soft


def test_validation_policy_uses_resolved_sampling_fps_when_available():
    fps = resolve_effective_sampling_fps(
        {
            "frame_provider_info": {
                "tool": "test",
                "settings": {"sampling_fps": 10.0, "resolved_sampling_fps": 4.0},
            }
        }
    )
    assert fps == pytest.approx(4.0)


def test_importing_policy_does_not_eagerly_import_geometry_consistency_validation():
    sys.modules.pop("pemoin.validation", None)
    sys.modules.pop("pemoin.validation.policy", None)
    sys.modules.pop("pemoin.validation.depth_pose_consistency", None)

    importlib.import_module("pemoin.validation.policy")

    assert "pemoin.validation.depth_pose_consistency" not in sys.modules
