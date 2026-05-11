from __future__ import annotations

import pytest

from pemoin.runtime.profiles.config import ModuleBinding, ProfileConfig, RuntimeBindings
from pemoin.runtime.runtime import (
    Runtime,
    _semantic_role_labels_from_runtime,
)


def test_label_resolution_reads_provider_defaults() -> None:
    assert _semantic_role_labels_from_runtime("road", {}, "CarlaSemanticsProvider", {}) == (
        "crosswalk",
        "path",
        "road",
        "roads",
    )
    assert set(_semantic_role_labels_from_runtime("mobile", {}, "CarlaSemanticsProvider", {})) == {
        "human",
        "person",
        "pedestrian",
        "pedestrians",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
    }
    assert set(_semantic_role_labels_from_runtime("sidewalk", {}, "CarlaSemanticsProvider", {})) == {
        "sidewalk",
        "sidewalks",
        "pavement",
        "walkway",
    }


def test_mobile_labels_rejects_dynamic_labels_alias() -> None:
    with pytest.raises(ValueError, match="dynamic_labels"):
        _semantic_role_labels_from_runtime(
            "mobile",
            {},
            "CarlaSemanticsProvider",
            {"dynamic_labels": ["car", "person"]},
        )


def test_road_labels_reject_runtime_legacy_key() -> None:
    with pytest.raises(ValueError, match="runtime.settings.road_labels"):
        _semantic_role_labels_from_runtime(
            "road",
            {"road_labels": ["road"]},
            "CarlaSemanticsProvider",
            {},
        )


def test_mobile_labels_reject_runtime_legacy_key() -> None:
    with pytest.raises(ValueError, match="runtime.settings.mobile_labels"):
        _semantic_role_labels_from_runtime(
            "mobile",
            {"mobile_labels": ["car"]},
            "CarlaSemanticsProvider",
            {},
        )


def test_provider_semantic_label_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="providers\\.semantics\\.settings\\.road_labels"):
        Runtime(
            ProfileConfig(
                name="test",
                runtime=RuntimeBindings(
                    state_window=1,
                    degradation_policy="OfflineDegradationPolicy",
                    settings={"comparison_frame": {"enabled": False}},
                ),
                providers={
                    "semantics": ModuleBinding(
                        tool="CarlaSemanticsProvider",
                        settings={"road_labels": ["road"]},
                    ),
                },
                effects={},
                working_resolution=(640, 640),
            )
        )


def test_runtime_resolves_semantic_role_defaults_for_geometry() -> None:
    runtime = Runtime(
        ProfileConfig(
            name="test",
                runtime=RuntimeBindings(
                    state_window=1,
                    degradation_policy="OfflineDegradationPolicy",
                    settings={"comparison_frame": {"enabled": False}},
                ),
            providers={
                "semantics": ModuleBinding(
                    tool="CarlaSemanticsProvider",
                    settings={},
                ),
                "geometry_fusion": ModuleBinding(
                    tool="GeometryFusionProvider",
                    settings={},
                ),
            },
            effects={},
            working_resolution=(640, 640),
        )
    )

    class _Factory:
        def create(self, binding, context):
            return binding

    providers = runtime.build_providers(_Factory(), {})

    assert providers["geometry_fusion"].settings == {}
    assert runtime._road_labels == ("crosswalk", "path", "road", "roads")
    assert set(runtime._mobile_labels) == {
        "human",
        "person",
        "pedestrian",
        "pedestrians",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
    }
