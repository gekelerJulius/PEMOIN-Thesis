from __future__ import annotations

import pytest

from pemoin.runtime.profiles.config import ModuleBinding, ProfileConfig, RuntimeBindings
from pemoin.runtime.runtime import Runtime


def _profile(*, trajectory_tool: str, comparison_mode: str = "gt") -> ProfileConfig:
    providers = {
        "trajectory": ModuleBinding(tool=trajectory_tool, settings={}),
        "geometry_fusion": ModuleBinding(tool="GeometryFusionProvider", settings={}),
    }
    return ProfileConfig(
        name="test",
        runtime=RuntimeBindings(
            state_window=1,
            degradation_policy="OfflineDegradationPolicy",
            settings={"comparison_frame": {"enabled": True, "mode": comparison_mode}},
        ),
        providers=providers,
        effects={},
        working_resolution=(640, 640),
    )


def test_runtime_gt_comparison_frame_rejects_non_gt_provider() -> None:
    with pytest.raises(ValueError, match="GT trajectory provider"):
        Runtime(_profile(trajectory_tool="DPVOTrajectoryProvider"))


def test_runtime_estimated_comparison_frame_accepts_estimated_provider() -> None:
    runtime = Runtime(_profile(trajectory_tool="DPVOTrajectoryProvider", comparison_mode="estimated"))

    assert runtime._comparison_frame_settings.mode == "estimated"


def test_runtime_gt_comparison_frame_accepts_gt_provider() -> None:
    runtime = Runtime(_profile(trajectory_tool="NuScenesTrajectoryProvider"))

    assert runtime._comparison_frame_settings.mode == "gt"
