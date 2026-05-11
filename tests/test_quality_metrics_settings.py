"""Tests for quality metrics settings."""

import pytest

from pemoin.metrics.settings import (
    ArtifactSettings,
    QualityMetricsSettings,
    RoadMetricsSettings,
    TrajectoryMetricsSettings,
)


class TestQualityMetricsSettings:
    def test_from_mapping_defaults(self):
        """Empty dict → valid defaults."""
        settings = QualityMetricsSettings.from_mapping({})
        assert settings.enabled is True
        assert settings.trajectory.enabled is True
        assert settings.road.enabled is True
        assert settings.artifacts.enabled is True
        assert settings.trajectory.rpe_deltas == [1, 5, 10]
        assert settings.road.residual_percentiles == [50.0, 90.0, 95.0, 99.0]

    def test_from_mapping_none(self):
        """None → valid defaults."""
        settings = QualityMetricsSettings.from_mapping(None)
        assert settings.enabled is True

    def test_from_mapping_disabled(self):
        """enabled=False should propagate."""
        settings = QualityMetricsSettings.from_mapping({"enabled": False})
        assert settings.enabled is False

    def test_from_mapping_override(self):
        """Specific values should propagate to sub-settings."""
        settings = QualityMetricsSettings.from_mapping({
            "trajectory": {
                "rpe_deltas": [1, 3],
                "umeyama_align": False,
            },
            "road": {
                "residual_percentiles": [50, 95],
                "smoothness_window": 20,
            },
            "artifacts": {
                "max_frames": 8,
                "colormap": "plasma",
            },
        })
        assert settings.trajectory.rpe_deltas == [1, 3]
        assert settings.trajectory.umeyama_align is False
        assert settings.road.residual_percentiles == [50, 95]
        assert settings.road.smoothness_window == 20
        assert settings.artifacts.max_frames == 8
        assert settings.artifacts.colormap == "plasma"

    def test_unknown_keys_ignored(self):
        """Unknown keys in sub-settings should not cause errors."""
        settings = QualityMetricsSettings.from_mapping({
            "trajectory": {"unknown_key": 42},
        })
        assert settings.trajectory.enabled is True


class TestTrajectoryMetricsSettings:
    def test_defaults(self):
        s = TrajectoryMetricsSettings()
        assert s.enabled is True
        assert s.rpe_deltas == [1, 5, 10]
        assert s.scale_drift_window == 20
        assert s.scale_drift_stride == 5
        assert s.umeyama_align is True
        assert s.umeyama_with_scale is True

    def test_from_mapping(self):
        s = TrajectoryMetricsSettings.from_mapping({"enabled": False, "scale_drift_window": 30})
        assert s.enabled is False
        assert s.scale_drift_window == 30

    def test_frozen(self):
        s = TrajectoryMetricsSettings()
        with pytest.raises(AttributeError):
            s.enabled = False


class TestRoadMetricsSettings:
    def test_defaults(self):
        s = RoadMetricsSettings()
        assert s.enabled is True
        assert s.normal_stability_window == 5

    def test_from_mapping(self):
        s = RoadMetricsSettings.from_mapping({"normal_stability_window": 10})
        assert s.normal_stability_window == 10


class TestArtifactSettings:
    def test_defaults(self):
        s = ArtifactSettings()
        assert s.enabled is True
        assert s.max_frames == 16
        assert s.colormap == "viridis"
        assert s.slice_thickness_m == 1.0

    def test_from_mapping(self):
        s = ArtifactSettings.from_mapping({"max_frames": 4, "colormap": "plasma"})
        assert s.max_frames == 4
        assert s.colormap == "plasma"
