from __future__ import annotations

import json
import math
from pathlib import Path

from pemoin.cli import _profile_snapshot
from pemoin.runtime.profiles.config import load_profiles_from_json
from pemoin.visualization.blender_scene.config import _shadow_spec_from_mapping


def _write_profile_config(config_path: Path, profile: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"profiles": {"broken": profile}}, indent=2),
        encoding="utf-8",
    )


def _base_profile() -> dict:
    return {
        "working_resolution": 640,
        "runtime": {
            "state_window": 1,
            "degradation_policy": "OfflineDegradationPolicy",
            "settings": {},
        },
        "providers": {
            "trajectory": {"tool": "CarlaTrajectoryProvider", "settings": {}}
        },
    }


def test_nuscenes_gt_uses_geometry_fusion_with_metric_trajectory_preservation():
    repo_root = Path(__file__).resolve().parents[1]
    profiles = load_profiles_from_json(repo_root / "config" / "profiles.json")

    assert profiles["nuscenes_gt"].frame_provider.settings["sampling_mode"] == "all_camera_frames"
    assert profiles["nuscenes_gt"].frame_provider.settings["sampling_fps"] == 30

    binding = profiles["nuscenes_gt"].providers["geometry_fusion"]
    assert binding.tool == "GeometryFusionProvider"
    assert profiles["nuscenes_gt"].runtime.settings["comparison_frame"]["mode"] == "gt"
    assert (
        profiles["nuscenes_gt"].runtime.settings["comparison_frame"]["enabled"]
        is True
    )
    assert (
        profiles["nuscenes_gt"].runtime.settings["blender_scene"]["local_support_temporal_hold_seconds"]
        == 5.0
    )
    gt_wrap = profiles["nuscenes_gt"].runtime.settings["blender_scene"]["lighting"]["wrap_subject_fill"]
    assert gt_wrap["global_strength_scale"] == 2.85
    assert gt_wrap["counter_wrap_role_scale"] == 0.07
    assert gt_wrap["sky_fill_role_scale"] == 0.04
    assert gt_wrap["raw_exposure_trim"] == 1.1
    assert (
        profiles["nuscenes_gt"].runtime.settings["blender_scene"]["render"][
            "raw_subject_exposure"
        ]["max_gain"]
        == 3.5
    )
    gt_lighting = profiles["nuscenes_gt"].providers["lighting"].settings
    assert gt_lighting["fill_heavy_dark_side_target_ratio"] == 0.38
    assert gt_lighting["fill_heavy_transport_gain"] == 1.35
    assert gt_lighting["wrap_geometry_min_azimuth_separation_deg"] == 55.0
    assert gt_lighting["wrap_geometry_counter_opposition_deg"] == 110.0
    assert gt_lighting["wrap_geometry_sky_min_elevation_deg"] == 55.0
    assert gt_lighting["wrap_geometry_candidate_count_per_role"] == 3

    assert profiles["nuscenes_dpvo"].frame_provider.settings["sampling_mode"] == "all_camera_frames"
    dpvo_sampling_fps = float(
        profiles["nuscenes_dpvo"].frame_provider.settings["sampling_fps"]
    )
    assert math.isfinite(dpvo_sampling_fps)
    assert dpvo_sampling_fps > 0.0
    assert dpvo_sampling_fps <= 30.0
    assert (
        profiles["nuscenes_dpvo"].runtime.settings["blender_scene"]["local_support_temporal_hold_seconds"]
        == 5.0
    )
    dpvo_wrap = profiles["nuscenes_dpvo"].runtime.settings["blender_scene"]["lighting"]["wrap_subject_fill"]
    assert dpvo_wrap["global_strength_scale"] == 2.85
    assert dpvo_wrap["counter_wrap_role_scale"] == 0.07
    assert dpvo_wrap["sky_fill_role_scale"] == 0.04
    assert dpvo_wrap["raw_exposure_trim"] == 1.1
    assert (
        profiles["nuscenes_dpvo"].runtime.settings["blender_scene"]["render"][
            "raw_subject_exposure"
        ]["max_gain"]
        == 3.5
    )
    dpvo_lighting = profiles["nuscenes_dpvo"].providers["lighting"].settings
    assert dpvo_lighting["fill_heavy_dark_side_target_ratio"] == 0.38
    assert dpvo_lighting["fill_heavy_transport_gain"] == 1.35
    assert dpvo_lighting["wrap_geometry_min_azimuth_separation_deg"] == 55.0
    assert dpvo_lighting["wrap_geometry_counter_opposition_deg"] == 110.0
    assert dpvo_lighting["wrap_geometry_sky_min_elevation_deg"] == 55.0
    assert dpvo_lighting["wrap_geometry_candidate_count_per_role"] == 3
    dpvo_consistency = profiles["nuscenes_dpvo"].runtime.settings["geometry_consistency_validation"]
    assert dpvo_consistency["exclude_dynamic_pixels"] is True
    assert dpvo_consistency["dynamic_mask_source"] == "auto"
    assert dpvo_consistency["min_static_overlap_points"] == 200
    assert dpvo_consistency["max_reprojection_rmse_px"] == 6.0
    assert dpvo_consistency["max_reprojection_p90_px"] == 4.0
    assert dpvo_consistency["max_reprojection_p95_px"] == 8.0
    assert dpvo_consistency["max_consecutive_catastrophic"] == 4
    assert dpvo_consistency["max_skipped_frames"] == 16


def test_profile_snapshot_includes_mixamo_and_unity_import_sections():
    repo_root = Path(__file__).resolve().parents[1]
    profiles = load_profiles_from_json(repo_root / "config" / "profiles.json")
    profile = profiles["nuscenes_gt"]

    snapshot = _profile_snapshot(
        profile,
        config_path=repo_root / "config" / "profiles.json",
        frame_source=Path("/tmp/nuscenes"),
        frame_provider_info={"tool": "NuScenesFrameProvider", "settings": {}},
        run_timestamp="20260304_000000",
    )

    assert "mixamo" in snapshot
    assert snapshot["mixamo"]["character_fbx_path"]
    assert snapshot["mixamo"]["animation_fbx_path"]
    assert "unity_import" in snapshot


def test_unity_import_rejects_semantics_disabled_for_unity_gt_semantics_provider(tmp_path: Path):
    config_path = tmp_path / "profiles.json"
    profile = _base_profile()
    profile["unity_import"] = {
        "enabled": True,
        "source": str(tmp_path / "unity_source"),
        "resources": {
            "frames": True,
            "depth": False,
            "semantics": False,
        },
    }
    profile["providers"]["semantics"] = {
        "tool": "UnityGTSemanticsProvider",
        "settings": {},
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "unity_import.resources.semantics=false" in str(exc)
        assert "UnityGTSemanticsProvider" in str(exc)
    else:
        raise AssertionError("Expected inconsistent unity_import semantics selection to fail.")


def test_carla_gt_profile_uses_carla_gt_lighting_provider():
    repo_root = Path(__file__).resolve().parents[1]
    profiles = load_profiles_from_json(repo_root / "config" / "profiles.json")

    binding = profiles["carla_gt"].providers["lighting"]

    assert binding.tool == "CarlaGTLightingProvider"
    assert binding.settings["require_scene_lights"] is True
    assert "include_vehicle_lights" not in binding.settings
    assert "vehicle_light_strength_scale" not in binding.settings


def test_unity_profiles_use_unity_gt_lighting_provider():
    repo_root = Path(__file__).resolve().parents[1]
    profiles = load_profiles_from_json(repo_root / "config" / "profiles.json")

    for profile_name in ("unity_gt", "unity_dpvo"):
        binding = profiles[profile_name].providers["lighting"]
        assert binding.tool == "UnityGTLightingProvider"
        assert binding.settings["require_reflection_faces"] is True
        assert binding.settings["require_frame_lighting"] is False
        assert binding.settings["force_sun_shadows"] is True


def test_shadow_spec_defaults_to_enabled_with_minimal_controls():
    spec = _shadow_spec_from_mapping(None, "runtime.settings.blender_scene.shadow")

    assert spec.enabled is True
    assert spec.receiver_patch_size_m == 4.0
    assert spec.map_resolution == "1024"
    assert spec.softness == 1.5
    assert spec.opacity == 1.0
    assert spec.tint_rgb == (0.0, 0.0, 0.0)


def test_shadow_spec_rejects_invalid_values():
    try:
        _shadow_spec_from_mapping(
            {"receiver_patch_size_m": 0.0},
            "runtime.settings.blender_scene.shadow",
        )
    except ValueError as exc:
        assert "receiver_patch_size_m" in str(exc)
    else:
        raise AssertionError("Expected invalid receiver_patch_size_m to fail.")

    try:
        _shadow_spec_from_mapping(
            {"opacity": 1.2},
            "runtime.settings.blender_scene.shadow",
        )
    except ValueError as exc:
        assert "opacity" in str(exc)
    else:
        raise AssertionError("Expected invalid opacity to fail.")


def test_profiles_reject_removed_road_height_scale_correction_runtime_setting(tmp_path):
    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
{
  "profiles": {
    "broken": {
      "working_resolution": 640,
      "runtime": {
        "state_window": 1,
        "degradation_policy": "OfflineDegradationPolicy",
        "settings": {
          "road_height_scale_correction": {
            "enabled": true
          }
        }
      },
      "providers": {
        "trajectory": {"tool": "CarlaTrajectoryProvider", "settings": {}}
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "road_height_scale_correction" in str(exc)
        assert "geometry_fusion" in str(exc)
    else:
        raise AssertionError("Expected removed road_height_scale_correction setting to fail.")


def test_profiles_reject_removed_road_height_scale_correction_provider(tmp_path):
    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
{
  "profiles": {
    "broken": {
      "working_resolution": 640,
      "runtime": {
        "state_window": 1,
        "degradation_policy": "OfflineDegradationPolicy",
        "settings": {}
      },
      "providers": {
        "geometry_fallback": {
          "tool": "RoadHeightScaleCorrectionProvider",
          "settings": {}
        }
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "RoadHeightScaleCorrectionProvider" in str(exc)
        assert "GeometryFusionProvider" in str(exc)
    else:
        raise AssertionError("Expected removed RoadHeightScaleCorrectionProvider to fail.")


def test_profiles_reject_missing_mixamo_character_path(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "assets" / "mixamo" / "animations" / "moving").mkdir(parents=True)
    animation_path = repo_root / "assets" / "mixamo" / "animations" / "moving" / "walk.fbx"
    animation_path.write_text("fbx", encoding="utf-8")
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["mixamo"] = {
        "character_fbx_path": "assets/mixamo/characters/Ch42_nonPBR.fbx",
        "animation_fbx_path": "assets/mixamo/animations/moving/walk.fbx",
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "mixamo.character_fbx_path" in str(exc)
        assert "existing file" in str(exc)
    else:
        raise AssertionError("Expected missing mixamo character path to fail.")


def test_profiles_reject_missing_mixamo_animation_path(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "assets" / "mixamo" / "characters").mkdir(parents=True)
    character_path = repo_root / "assets" / "mixamo" / "characters" / "Ch42_nonPBR.fbx"
    character_path.write_text("fbx", encoding="utf-8")
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["mixamo"] = {
        "character_fbx_path": "assets/mixamo/characters/Ch42_nonPBR.fbx",
        "animation_fbx_path": "assets/mixamo/animations/moving/walk.fbx",
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "mixamo.animation_fbx_path" in str(exc)
        assert "existing file" in str(exc)
    else:
        raise AssertionError("Expected missing mixamo animation path to fail.")


def test_profiles_reject_missing_mixamo_asset_root(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "assets" / "mixamo" / "characters").mkdir(parents=True)
    (repo_root / "assets" / "mixamo" / "animations" / "moving").mkdir(parents=True)
    (repo_root / "assets" / "mixamo" / "characters" / "Ch42_nonPBR.fbx").write_text(
        "fbx",
        encoding="utf-8",
    )
    (repo_root / "assets" / "mixamo" / "animations" / "moving" / "walk.fbx").write_text(
        "fbx",
        encoding="utf-8",
    )
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["mixamo"] = {
        "character_fbx_path": "assets/mixamo/characters/Ch42_nonPBR.fbx",
        "animation_fbx_path": "assets/mixamo/animations/moving/walk.fbx",
        "asset_root": "assets/mixamo/package",
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "mixamo.asset_root" in str(exc)
        assert "existing directory" in str(exc)
    else:
        raise AssertionError("Expected missing mixamo asset root to fail.")


def test_profiles_reject_missing_harmonizer_pretrained_path(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["runtime"]["settings"]["harmonisation"] = {
        "pretrained_path": "tools/Harmonizer/pretrained/harmonizer.pth"
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "runtime.settings.harmonisation.pretrained_path" in str(exc)
        assert "existing file" in str(exc)
    else:
        raise AssertionError("Expected missing harmonizer pretrained path to fail.")


def test_profiles_reject_missing_lighting_repo_root(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["providers"]["lighting"] = {
        "tool": "DiffusionLightTurboLightingProvider",
        "settings": {"repo_root": "tools/DiffusionLight-Turbo"},
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "providers.lighting.settings.repo_root" in str(exc)
        assert "existing directory" in str(exc)
    else:
        raise AssertionError("Expected missing lighting repo root to fail.")


def test_profiles_reject_missing_carla_label_map_path(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["providers"]["semantics"] = {
        "tool": "CarlaSemanticsProvider",
        "settings": {"label_map_path": "carla_scripts/carla_label_map_dump.json"},
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "providers.semantics.settings.label_map_path" in str(exc)
        assert "existing file" in str(exc)
    else:
        raise AssertionError("Expected missing CARLA label_map_path to fail.")


def test_profiles_reject_missing_megasam_adapter_paths(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["providers"]["depth"] = {
        "tool": "MegaSAMDepthProvider",
        "settings": {
            "adapter": {
                "checkpoint_path": "tools/mega-sam/checkpoints/megasam_final.pth",
                "config_path": "tools/mega-sam/base/environment.yaml",
            }
        },
    }
    _write_profile_config(config_path, profile)

    try:
        load_profiles_from_json(config_path)
    except ValueError as exc:
        assert "providers.depth.settings.adapter.checkpoint_path" in str(exc)
        assert "existing file" in str(exc)
    else:
        raise AssertionError("Expected missing MegaSAM adapter checkpoint to fail.")


def test_profiles_resolve_relative_paths_from_repo_root(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "assets" / "mixamo" / "characters").mkdir(parents=True)
    (repo_root / "assets" / "mixamo" / "animations" / "moving").mkdir(parents=True)
    (repo_root / "tools" / "Harmonizer" / "pretrained").mkdir(parents=True)
    (repo_root / "tools" / "DiffusionLight-Turbo").mkdir(parents=True)
    (repo_root / "carla_scripts").mkdir(parents=True)
    (repo_root / "tools" / "mega-sam" / "checkpoints").mkdir(parents=True)
    (repo_root / "tools" / "mega-sam" / "base").mkdir(parents=True)
    (repo_root / "assets" / "mixamo" / "characters" / "Ch42_nonPBR.fbx").write_text(
        "fbx",
        encoding="utf-8",
    )
    (repo_root / "assets" / "mixamo" / "animations" / "moving" / "walk.fbx").write_text(
        "fbx",
        encoding="utf-8",
    )
    (repo_root / "tools" / "Harmonizer" / "pretrained" / "harmonizer.pth").write_text(
        "weights",
        encoding="utf-8",
    )
    (repo_root / "carla_scripts" / "carla_label_map_dump.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (repo_root / "tools" / "mega-sam" / "checkpoints" / "megasam_final.pth").write_text(
        "weights",
        encoding="utf-8",
    )
    (repo_root / "tools" / "mega-sam" / "base" / "environment.yaml").write_text(
        "name: megasam",
        encoding="utf-8",
    )
    config_path = repo_root / "config" / "profiles.json"
    profile = _base_profile()
    profile["runtime"]["settings"]["harmonisation"] = {
        "pretrained_path": "tools/Harmonizer/pretrained/harmonizer.pth"
    }
    profile["providers"]["lighting"] = {
        "tool": "DiffusionLightTurboLightingProvider",
        "settings": {"repo_root": "tools/DiffusionLight-Turbo"},
    }
    profile["providers"]["semantics"] = {
        "tool": "CarlaSemanticsProvider",
        "settings": {"label_map_path": "carla_scripts/carla_label_map_dump.json"},
    }
    profile["providers"]["depth"] = {
        "tool": "MegaSAMDepthProvider",
        "settings": {
            "adapter": {
                "checkpoint_path": "tools/mega-sam/checkpoints/megasam_final.pth",
                "config_path": "tools/mega-sam/base/environment.yaml",
            }
        },
    }
    profile["mixamo"] = {
        "character_fbx_path": "assets/mixamo/characters/Ch42_nonPBR.fbx",
        "animation_fbx_path": "assets/mixamo/animations/moving/walk.fbx",
    }
    _write_profile_config(config_path, profile)

    profiles = load_profiles_from_json(config_path)

    assert "broken" in profiles
