from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pemoin.visualization.blender_scene import app as blender_app
from pemoin.visualization.blender_scene import config as blender_config
from pemoin.visualization.blender_scene.specs import SceneSpec


def test_blender_scene_config_imports_without_bpy() -> None:
    sys.modules.pop("bpy", None)
    module = importlib.import_module("pemoin.visualization.blender_scene.config")
    assert module.parse_args is not None


def test_scene_spec_from_profile_prefers_run_snapshot_sampling_fps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "standard" / "profile.json"
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {"settings": {"blender_scene": {"enabled": true}}},
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )
    snapshot_path.write_text(
        """
        {
          "profile": "demo",
          "runtime": {"settings": {"blender_scene": {"enabled": true}}},
          "mixamo": {
            "character_fbx_path": "%s",
            "animation_fbx_path": "%s"
          },
          "frame_provider": {"settings": {"resolved_sampling_fps": 12.5}}
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    spec = blender_config._scene_spec_from_profile(
        run_dir=run_dir,
        trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
        output_path=None,
        config_path=config_path,
        profile_name="demo",
    )

    assert spec.sampling_fps == pytest.approx(12.5)
    assert spec.mixamo_scene_fps == pytest.approx(12.5)
    assert spec.mixamo_export_fps == pytest.approx(30.0)


def test_scene_spec_from_profile_defaults_mixamo_asset_root_to_character_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    asset_dir = tmp_path / "mixamo_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    char_path = asset_dir / "character.fbx"
    anim_path = asset_dir / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {"settings": {"blender_scene": {"enabled": true}}},
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 12.5}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    spec = blender_config._scene_spec_from_profile(
        run_dir=run_dir,
        trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
        output_path=None,
        config_path=config_path,
        profile_name="demo",
    )

    assert spec.mixamo_asset_root == asset_dir.resolve()


def test_scene_spec_from_profile_accepts_mixamo_export_fps_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {"settings": {"blender_scene": {"enabled": true}}},
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s",
                "export_fps": 24.0
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 12.5}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    spec = blender_config._scene_spec_from_profile(
        run_dir=run_dir,
        trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
        output_path=None,
        config_path=config_path,
        profile_name="demo",
    )

    assert spec.mixamo_scene_fps == pytest.approx(12.5)
    assert spec.mixamo_export_fps == pytest.approx(24.0)


def test_scene_spec_from_profile_resolves_unity_authored_pedestrian_placement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard" / "trajectory").mkdir(parents=True, exist_ok=True)
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")
    np.savez_compressed(
        run_dir / "standard" / "trajectory" / "poses.npz",
        frame_indices=np.array([0], dtype=np.int32),
        camera_to_world=np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
        metadata={
            "comparison_frame": {
                "authoring_frame": {
                    "authoring_to_canonical_transform": [
                        [0.0, 0.0, -1.0, 10.0],
                        [1.0, 0.0, 0.0, 20.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                }
            }
        },
    )

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "unity_demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "pedestrian_placement": {
                      "mode": "unity_world_horizontal",
                      "position_x_m": 3.0,
                      "position_z_m": 4.0,
                      "heading_yaw_deg": 90.0
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"tool": "UnityFrameProvider", "settings": {"resolved_sampling_fps": 12.5}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    spec = blender_config._scene_spec_from_profile(
        run_dir=run_dir,
        trajectory_path=None,
        output_path=None,
        config_path=config_path,
        profile_name="unity_demo",
    )

    assert spec.pedestrian_placement_mode == "unity_world_horizontal"
    assert spec.pedestrian_authored_position_x_m == pytest.approx(3.0)
    assert spec.pedestrian_authored_position_z_m == pytest.approx(4.0)
    assert spec.pedestrian_authored_heading_yaw_deg == pytest.approx(90.0)
    assert spec.pedestrian_resolved_spawn_world == pytest.approx((6.0, 23.0, 0.0))
    assert spec.pedestrian_resolved_forward_world == pytest.approx((0.0, 1.0, 0.0))
    assert spec.pedestrian_resolved_heading_world_deg == pytest.approx(90.0)


def test_scene_spec_from_profile_rejects_legacy_and_unity_authored_pedestrian_placement_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard" / "trajectory").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")
    np.savez_compressed(
        run_dir / "standard" / "trajectory" / "poses.npz",
        frame_indices=np.array([0], dtype=np.int32),
        camera_to_world=np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
        metadata={
            "comparison_frame": {
                "authoring_frame": {
                    "authoring_to_canonical_transform": np.eye(4, dtype=np.float32).tolist()
                }
            }
        },
    )

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "unity_demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "pedestrian_trajectory_t": 0.25,
                    "pedestrian_placement": {
                      "mode": "unity_world_horizontal",
                      "position_x_m": 1.0,
                      "position_z_m": 2.0,
                      "heading_yaw_deg": 0.0
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"tool": "UnityFrameProvider", "settings": {"resolved_sampling_fps": 12.5}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    with pytest.raises(ValueError, match="replaces legacy pedestrian placement"):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=None,
            output_path=None,
            config_path=config_path,
            profile_name="unity_demo",
        )


def test_scene_spec_from_profile_parses_time_based_support_hold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "local_support_temporal_hold_frames": 10,
                    "local_support_temporal_hold_seconds": 5.0
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 10.0}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    spec = blender_config._scene_spec_from_profile(
        run_dir=run_dir,
        trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
        output_path=None,
        config_path=config_path,
        profile_name="demo",
    )

    assert spec.local_support_temporal_hold_frames == 10
    assert spec.local_support_temporal_hold_seconds == pytest.approx(5.0)


def test_scene_spec_from_profile_parses_contact_aware_occlusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "render": {
                      "resolution_scale": 0.65,
                      "samples": 12,
                      "material_policy": "preserve_base_alpha_normal",
                      "dynamic_light_binding": "copy_location_constraint",
                      "salience_adaptive": {
                        "enabled": true,
                        "low_salience_resolution_scale": 0.5,
                        "protect_below_visible_pixels": 12000,
                        "protect_below_bbox_short_side_px": 40,
                        "protect_when_center_distance_ratio_below": 0.28,
                        "reduce_only_when_boundary_fraction_above": 0.4,
                        "reduce_only_near_visibility_transition": true,
                        "shadow_quality_reduction_enabled": true,
                        "fill_light_reduction_enabled": true
                      },
                      "performance": {
                        "disable_bloom": true,
                        "disable_screen_space_reflections": true,
                        "disable_gtao": true,
                        "disable_motion_blur": true
                      },
                      "raw_subject_exposure": {
                        "enabled": true,
                        "target_match_strength": 0.8,
                        "max_gain": 2.2,
                        "validation_tolerance": 0.12
                      }
                    },
                    "lighting": {
                      "wrap_subject_fill": {
                        "global_strength_scale": 2.2,
                        "wrap_key_role_scale": 0.09,
                        "counter_wrap_role_scale": 0.045,
                        "sky_fill_role_scale": 0.03,
                        "raw_exposure_trim": 1.05
                      }
                    },
                    "occlusion": {
                      "contact_ground_roles": ["road", "sidewalk"],
                      "contact_ground_labels": ["crosswalk"],
                      "default_front_margin_m": 0.04,
                      "contact_plane_band_m": 0.02,
                      "contact_patch_radius_m": 0.35,
                      "contact_coplanar_tolerance_m": 0.025,
                      "write_debug": false,
                      "edge_treatment": {
                        "boundary_band_px": 5,
                        "feather_radius_px": 1.5,
                        "feather_strength": 0.4,
                        "blur_radius_px": 2.0,
                        "blur_strength": 0.3,
                        "despill_strength": 0.2,
                        "regrain_strength": 0.1,
                        "tiny_object_disable_feather": true,
                        "tiny_object_disable_blur": true,
                        "tiny_object_disable_despill": true,
                        "tiny_object_disable_regrain": true,
                        "tiny_object_max_boundary_fraction": 0.2
                      }
                    }
                    ,
                    "shadow": {
                      "map_resolution": "2048",
                      "softness": 2.25
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 10.0}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("roads",),
    )
    monkeypatch.setattr(
        blender_config,
        "_contact_ground_labels_setting",
        lambda **kwargs: ("roads", "sidewalks", "crosswalk"),
    )

    spec = blender_config._scene_spec_from_profile(
        run_dir=run_dir,
        trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
        output_path=None,
        config_path=config_path,
        profile_name="demo",
    )

    assert spec.occlusion.depth_source == "z_pass"
    assert spec.occlusion.contact_ground_labels == ("roads", "sidewalks", "crosswalk")
    assert spec.occlusion.default_front_margin_m == pytest.approx(0.04)
    assert spec.occlusion.contact_plane_band_m == pytest.approx(0.02)
    assert spec.occlusion.contact_patch_radius_m == pytest.approx(0.35)
    assert spec.occlusion.contact_coplanar_tolerance_m == pytest.approx(0.025)
    assert spec.occlusion.write_debug is False
    assert spec.occlusion.edge_treatment.enabled is True
    assert spec.occlusion.edge_treatment.boundary_band_px == 5
    assert spec.occlusion.edge_treatment.feather_radius_px == pytest.approx(1.5)
    assert spec.occlusion.edge_treatment.feather_strength == pytest.approx(0.4)
    assert spec.occlusion.edge_treatment.blur_radius_px == pytest.approx(2.0)
    assert spec.occlusion.edge_treatment.blur_strength == pytest.approx(0.3)
    assert spec.occlusion.edge_treatment.despill_strength == pytest.approx(0.2)
    assert spec.occlusion.edge_treatment.regrain_strength == pytest.approx(0.1)
    assert spec.occlusion.edge_treatment.tiny_object_disable_feather is True
    assert spec.occlusion.edge_treatment.tiny_object_disable_blur is True
    assert spec.occlusion.edge_treatment.tiny_object_disable_despill is True
    assert spec.occlusion.edge_treatment.tiny_object_disable_regrain is True
    assert spec.occlusion.edge_treatment.tiny_object_max_boundary_fraction == pytest.approx(0.2)
    assert spec.render.resolution_scale == pytest.approx(0.65)
    assert spec.render.samples == 12
    assert spec.render.material_policy == "preserve_base_alpha_normal"
    assert spec.render.dynamic_light_binding == "copy_location_constraint"
    assert spec.render.salience_adaptive.enabled is True
    assert spec.render.salience_adaptive.low_salience_resolution_scale == pytest.approx(0.5)
    assert spec.render.salience_adaptive.protect_below_visible_pixels == 12000
    assert spec.render.salience_adaptive.protect_below_bbox_short_side_px == 40
    assert spec.render.salience_adaptive.protect_when_center_distance_ratio_below == pytest.approx(0.28)
    assert spec.render.salience_adaptive.reduce_only_when_boundary_fraction_above == pytest.approx(0.4)
    assert spec.render.salience_adaptive.reduce_only_near_visibility_transition is True
    assert spec.render.salience_adaptive.shadow_quality_reduction_enabled is True
    assert spec.render.salience_adaptive.fill_light_reduction_enabled is True
    assert spec.render.performance.disable_bloom is True
    assert spec.render.performance.disable_screen_space_reflections is True
    assert spec.render.performance.disable_gtao is True
    assert spec.render.performance.disable_motion_blur is True
    assert spec.render.raw_subject_exposure.enabled is True
    assert spec.render.raw_subject_exposure.target_match_strength == pytest.approx(0.8)
    assert spec.render.raw_subject_exposure.max_gain == pytest.approx(2.2)
    assert spec.render.raw_subject_exposure.validation_tolerance == pytest.approx(0.12)
    assert spec.lighting.wrap_subject_fill.global_strength_scale == pytest.approx(2.2)
    assert spec.lighting.wrap_subject_fill.wrap_key_role_scale == pytest.approx(0.09)
    assert spec.lighting.wrap_subject_fill.counter_wrap_role_scale == pytest.approx(0.045)
    assert spec.lighting.wrap_subject_fill.sky_fill_role_scale == pytest.approx(0.03)
    assert spec.lighting.wrap_subject_fill.raw_exposure_trim == pytest.approx(1.05)
    assert spec.shadow.map_resolution == "2048"
    assert spec.shadow.softness == pytest.approx(2.25)


def test_scene_spec_from_profile_rejects_invalid_edge_treatment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "occlusion": {
                      "edge_treatment": {
                        "boundary_band_px": 0
                      }
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 10.0}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("roads",),
    )
    monkeypatch.setattr(
        blender_config,
        "_contact_ground_labels_setting",
        lambda **kwargs: ("roads", "sidewalks"),
    )

    with pytest.raises(ValueError, match="boundary_band_px"):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
            output_path=None,
            config_path=config_path,
            profile_name="demo",
        )


def test_scene_spec_from_profile_rejects_invalid_render_performance_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "render": {
                      "performance": {
                        "disable_bloom": "yes"
                      }
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 10.0}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )
    monkeypatch.setattr(
        blender_config,
        "_contact_ground_labels_setting",
        lambda **kwargs: ("road", "sidewalk"),
    )

    with pytest.raises(ValueError, match="render.performance.disable_bloom"):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
            output_path=None,
            config_path=config_path,
            profile_name="demo",
        )


def test_scene_spec_from_profile_rejects_invalid_material_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "render": {
                      "material_policy": "full_pbr"
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 10.0}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )
    monkeypatch.setattr(
        blender_config,
        "_contact_ground_labels_setting",
        lambda **kwargs: ("road", "sidewalk"),
    )

    with pytest.raises(ValueError, match="render.material_policy"):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
            output_path=None,
            config_path=config_path,
            profile_name="demo",
        )


def test_scene_spec_from_profile_rejects_invalid_dynamic_light_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "render": {
                      "dynamic_light_binding": "per_frame_keys"
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 10.0}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )
    monkeypatch.setattr(
        blender_config,
        "_contact_ground_labels_setting",
        lambda **kwargs: ("road", "sidewalk"),
    )

    with pytest.raises(ValueError, match="render.dynamic_light_binding"):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
            output_path=None,
            config_path=config_path,
            profile_name="demo",
        )


def test_scene_spec_from_profile_rejects_invalid_salience_resolution_scale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "render": {
                      "resolution_scale": 0.65,
                      "salience_adaptive": {
                        "low_salience_resolution_scale": 0.8
                      }
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              },
              "frame_provider": {"settings": {"resolved_sampling_fps": 12.5}}
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    with pytest.raises(
        ValueError,
        match="render.salience_adaptive.low_salience_resolution_scale",
    ):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
            output_path=None,
            config_path=config_path,
            profile_name="demo",
        )


def test_scene_spec_from_profile_rejects_removed_tiny_object_render_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "render": {
                      "tiny_object": {
                        "short_side_px_threshold": 14
                      }
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              }
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    with pytest.raises(ValueError, match="adaptive tiny-object rerender settings were removed"):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
            output_path=None,
            config_path=config_path,
            profile_name="demo",
        )


def test_scene_spec_from_profile_rejects_shadow_mode_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    char_path = tmp_path / "character.fbx"
    anim_path = tmp_path / "animation.fbx"
    char_path.write_text("", encoding="utf-8")
    anim_path.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        """
        {
          "profiles": {
            "demo": {
              "runtime": {
                "settings": {
                  "blender_scene": {
                    "enabled": true,
                    "shadow": {
                      "mode": "receiver_difference"
                    }
                  }
                }
              },
              "mixamo": {
                "character_fbx_path": "%s",
                "animation_fbx_path": "%s"
              }
            }
          }
        }
        """
        % (char_path.as_posix(), anim_path.as_posix()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        blender_config,
        "_road_labels_setting",
        lambda run_dir, profile=None: ("road",),
    )

    with pytest.raises(ValueError, match="Shadow mode is no longer configurable"):
        blender_config._scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=run_dir / "standard" / "trajectory" / "poses.npz",
            output_path=None,
            config_path=config_path,
            profile_name="demo",
        )


def test_wrapper_delegates_to_run_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setattr(blender_app, "run_scene", lambda argv: captured.append(list(argv)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "blender_trajectory_scene.py",
            "--",
            "--run-dir",
            "/tmp/run",
            "--output",
            "/tmp/scene.blend",
        ],
    )

    runpy.run_path(
        Path("src/pemoin/scripts/blender_trajectory_scene.py"),
        run_name="__main__",
    )

    assert captured == [["--run-dir", "/tmp/run", "--output", "/tmp/scene.blend"]]


def test_wrapper_exits_with_code_one_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setattr(
        "pemoin.visualization.blender_scene.app.run_scene",
        lambda argv: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["blender_trajectory_scene.py", "--", "--run-dir", "/tmp/run"],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(
            Path("src/pemoin/scripts/blender_trajectory_scene.py"),
            run_name="__main__",
        )

    assert exc_info.value.code == 1


def test_run_scene_from_spec_preserves_stage_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    lighting_calls: list[dict[str, object]] = []
    camera_matrix = np.eye(3, dtype=np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    frame_indices = np.asarray([0, 1], dtype=np.int32)

    def record(name: str):
        def _inner(*args, **kwargs):
            order.append(name)
            if name == "configure_scene_lighting":
                lighting_calls.append(
                    {
                        "run_dir": kwargs.get("run_dir"),
                        "anchor_world": kwargs.get("anchor_world"),
                    }
                )
            if name == "ensure_collection":
                return object()
            if name == "load_trajectory":
                return c2w, frame_indices
            if name == "load_intrinsics":
                return camera_matrix, 32, 24, {"intrinsics_resolution_source": "test"}
            if name == "create_animated_camera":
                return object(), SimpleNamespace(
                    sensor_fit="AUTO",
                    focal_residual_px=0.0,
                    principal_point_residual_px=0.0,
                )
            if name == "viz_road_planes":
                return SimpleNamespace(global_planes={})
            if name == "apply_road_support_to_inserted_pedestrian":
                return []
            if name == "bind_dynamic_subject_lights":
                return []
            if name == "_write_grounding_diagnostics":
                return (tmp_path / "grounding.json", tmp_path / "grounding.csv")
            if name == "_write_support_surface_diagnostics":
                return (tmp_path / "support.json", tmp_path / "support.csv")
            if name == "render_pedestrian":
                return tmp_path / "pedestrian_frames"
            return None

        return _inner

    for fn in (
        "clear_scene",
        "ensure_collection",
        "load_trajectory",
        "add_trajectory_cubes",
        "load_intrinsics",
        "create_animated_camera",
        "configure_render_engine",
        "_resolve_spawn",
        "configure_scene_lighting",
        "insert_mixamo_character",
        "viz_road_planes",
        "apply_road_support_to_inserted_pedestrian",
        "bind_dynamic_subject_lights",
        "_write_grounding_diagnostics",
        "_write_support_surface_diagnostics",
        "_write_road_surface_summary",
        "_raise_for_grounding_failures",
        "render_pedestrian",
        "compose_overlay_frames",
        "save_blend",
    ):
        monkeypatch.setattr(blender_app, fn, record(fn))
    monkeypatch.setattr(
        blender_app,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(
                scene=SimpleNamespace(
                    render=SimpleNamespace(fps=24, fps_base=1.0),
                )
            )
        ),
    )
    monkeypatch.setattr(
        blender_app,
        "_resolve_spawn",
        lambda spec, c2w: (
            order.append("_resolve_spawn")
            or blender_app.SpawnResolution(
                resolved_spawn_world_arr=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                trajectory_anchor_world_arr=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                motion_forward_world_arr=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                base_heading_world_deg=0.0,
                spawn_min_distance_m=0.5,
            )
        ),
    )

    spec = SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=None,
        cube_size=0.1,
        collection_name="TrajectoryDebug",
        sampling_fps=24.0,
    )
    blender_app.run_scene_from_spec(spec)

    assert order == [
        "clear_scene",
        "ensure_collection",
        "ensure_collection",
        "load_trajectory",
        "add_trajectory_cubes",
        "load_intrinsics",
        "create_animated_camera",
        "configure_render_engine",
        "_resolve_spawn",
        "configure_scene_lighting",
        "insert_mixamo_character",
        "viz_road_planes",
        "apply_road_support_to_inserted_pedestrian",
        "bind_dynamic_subject_lights",
        "_write_grounding_diagnostics",
        "_write_support_surface_diagnostics",
        "_write_road_surface_summary",
        "_raise_for_grounding_failures",
        "render_pedestrian",
        "compose_overlay_frames",
    ]
    assert lighting_calls == [
        {"run_dir": tmp_path, "anchor_world": (0.0, 0.0, 0.0)}
    ]
