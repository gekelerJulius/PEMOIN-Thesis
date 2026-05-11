from __future__ import annotations

import json
from pathlib import Path

import pytest

from pemoin.runtime.profiles.config import load_profiles_from_json


def test_active_profiles_use_comparison_frame_modes(tmp_path: Path) -> None:
    profiles = load_profiles_from_json(Path("config/profiles.json"))
    expected_modes = {
        "unity_gt": "gt",
        "carla_gt": "gt",
        "nuscenes_gt": "gt",
        "unity_dpvo": "estimated",
        "carla_dpvo": "estimated",
        "nuscenes_dpvo": "estimated",
    }
    assert set(profiles.keys()) == set(expected_modes.keys())
    for name, mode in expected_modes.items():
        assert profiles[name].runtime.settings["comparison_frame"]["mode"] == mode


def test_unity_dpvo_uses_unity_gt_gravity_prior() -> None:
    profiles = load_profiles_from_json(Path("config/profiles.json"))
    cfg = profiles["unity_dpvo"].runtime.settings["comparison_frame"]
    assert cfg["up_direction_source"] == "gravity_prior"
    assert cfg["gravity_prior"]["provider"] == "unity_gt"


@pytest.mark.parametrize("legacy_key", ["alignment", "grounding_to_z0", "world_frame"])
def test_legacy_runtime_settings_are_rejected(tmp_path: Path, legacy_key: str) -> None:
    payload = {
        "profiles": {
            "test_profile": {
                "working_resolution": 640,
                "runtime": {
                    "state_window": 1,
                    "degradation_policy": "OfflineDegradationPolicy",
                    "settings": {
                        "comparison_frame": {"enabled": True, "mode": "estimated"},
                        legacy_key: {},
                    },
                },
                "providers": {
                    "trajectory": {"tool": "DPVOTrajectoryProvider", "settings": {}},
                    "geometry_fusion": {"tool": "GeometryFusionProvider", "settings": {}},
                },
            }
        }
    }
    config_path = tmp_path / "profiles.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=legacy_key):
        load_profiles_from_json(config_path)
