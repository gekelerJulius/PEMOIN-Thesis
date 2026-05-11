"""Tests for GeometryFusionSettings."""

import pytest

from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings


class TestGeometryFusionSettings:
    def test_defaults(self):
        settings = GeometryFusionSettings()
        assert settings.road_conf_thresh == 0.6
        assert settings.affine_mode == "scale_only"
        assert settings.dpvo_scale_mode == "windowed_local"
        assert settings.dpvo_local_window_size == 15
        assert settings.factor_graph_enabled is True
        assert settings.fg_env_name == "gtsam"
        assert settings.fg_env_manager is None
        assert settings.fg_fallback_on_discontinuity is True
        assert settings.quadratic_enabled is True
        assert settings.preserve_metric_trajectory is False
        assert settings.joint_consistency_enabled is True

    def test_from_mapping_empty(self):
        settings = GeometryFusionSettings.from_mapping({})
        assert settings.road_conf_thresh == 0.6

    def test_from_mapping_overrides(self):
        settings = GeometryFusionSettings.from_mapping({
            "road_conf_thresh": 0.8,
            "affine_mode": "affine",
            "dpvo_scale_mode": "windowed_local",
            "dpvo_local_window_size": 11,
            "factor_graph_enabled": False,
            "fg_env_name": "custom-gtsam",
            "fg_env_manager": "conda",
            "preserve_metric_trajectory": True,
            "joint_consistency_gt_fail_scale_delta": 0.2,
        })
        assert settings.road_conf_thresh == 0.8
        assert settings.affine_mode == "affine"
        assert settings.dpvo_scale_mode == "windowed_local"
        assert settings.dpvo_local_window_size == 11
        assert settings.factor_graph_enabled is False
        assert settings.fg_env_name == "custom-gtsam"
        assert settings.fg_env_manager == "conda"
        assert settings.preserve_metric_trajectory is True
        assert settings.joint_consistency_gt_fail_scale_delta == 0.2

    def test_invalid_affine_mode(self):
        with pytest.raises(ValueError, match="affine_mode"):
            GeometryFusionSettings.from_mapping({"affine_mode": "invalid"})

    def test_invalid_conf_thresh(self):
        with pytest.raises(ValueError, match="road_conf_thresh"):
            GeometryFusionSettings.from_mapping({"road_conf_thresh": 0.0})

    def test_invalid_window_overlap(self):
        with pytest.raises(ValueError, match="fg_overlap"):
            GeometryFusionSettings.from_mapping({"fg_overlap": 25, "fg_window_size": 21})

    def test_invalid_local_window_overlap(self):
        with pytest.raises(ValueError, match="dpvo_local_window_overlap"):
            GeometryFusionSettings.from_mapping(
                {"dpvo_local_window_size": 9, "dpvo_local_window_overlap": 9}
            )

    def test_invalid_fg_env_name(self):
        with pytest.raises(ValueError, match="fg_env_name"):
            GeometryFusionSettings.from_mapping({"fg_env_name": "   "})

    def test_invalid_fg_max_step_jump(self):
        with pytest.raises(ValueError, match="fg_max_step_jump_m"):
            GeometryFusionSettings.from_mapping({"fg_max_step_jump_m": 0.0})

    def test_invalid_fg_max_step_inflation_ratio(self):
        with pytest.raises(ValueError, match="fg_max_step_inflation_ratio"):
            GeometryFusionSettings.from_mapping({"fg_max_step_inflation_ratio": 1.0})

    def test_invalid_fg_env_manager(self):
        with pytest.raises(ValueError, match="fg_env_manager"):
            GeometryFusionSettings.from_mapping({"fg_env_manager": "venv"})

    def test_from_mapping_none(self):
        settings = GeometryFusionSettings.from_mapping(None)
        assert settings == GeometryFusionSettings()

    def test_road_labels_config_is_rejected(self):
        with pytest.raises(ValueError, match="geometry_fusion.road_labels"):
            GeometryFusionSettings.from_mapping({"road_labels": "road, sidewalk, path"})

    def test_quadratic_bands(self):
        settings = GeometryFusionSettings.from_mapping({"quadratic_bands": [0, 10, 20]})
        assert settings.quadratic_bands == (0.0, 10.0, 20.0)

    def test_pair_mode_and_fallback_valid_pairs(self):
        settings = GeometryFusionSettings.from_mapping(
            {
                "dpvo_match_pair_mode": "directed",
                "dpvo_match_fallback_min_valid_pairs": 3,
            }
        )
        assert settings.dpvo_match_pair_mode == "directed"
        assert settings.dpvo_match_fallback_min_valid_pairs == 3

    def test_invalid_pair_mode(self):
        with pytest.raises(ValueError, match="dpvo_match_pair_mode"):
            GeometryFusionSettings.from_mapping({"dpvo_match_pair_mode": "foo"})

    def test_invalid_fallback_min_valid_pairs(self):
        with pytest.raises(ValueError, match="fallback_min_valid_pairs"):
            GeometryFusionSettings.from_mapping({"dpvo_match_fallback_min_valid_pairs": 0})

    def test_invalid_joint_consistency_fail_delta(self):
        with pytest.raises(ValueError, match="joint_consistency_gt_fail_scale_delta"):
            GeometryFusionSettings.from_mapping(
                {
                    "joint_consistency_gt_warn_scale_delta": 0.05,
                    "joint_consistency_gt_fail_scale_delta": 0.01,
                }
            )
