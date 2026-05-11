from __future__ import annotations

from pemoin.providers.semantic_roles import (
    SEMANTIC_ROLES_METADATA_KEY,
    build_semantic_roles,
    merge_semantic_roles,
    resolve_role_label_ids,
    semantic_role_defaults_for_tool,
    with_semantic_roles,
)


def test_semantic_role_defaults_for_tool_unions_alias_floor_with_tool_defaults() -> None:
    roles = semantic_role_defaults_for_tool("CarlaSemanticsProvider")
    assert roles["road"] == ["crosswalk", "path", "road", "roads"]
    assert roles["sidewalk"] == ["pavement", "sidewalk", "sidewalks", "walkway"]
    assert roles["mobile"] == [
        "bicycle",
        "bus",
        "car",
        "human",
        "motorcycle",
        "pedestrian",
        "pedestrians",
        "person",
        "truck",
    ]


def test_build_semantic_roles_applies_overrides_and_filters_sky_from_mobile() -> None:
    roles = build_semantic_roles(
        {
            "road": ("road",),
            "sky": ("sky",),
            "mobile": ("car", "sky", "person"),
            "large_vehicle": ("bus", "truck"),
        },
        settings={
            "road_labels": ["road", "crosswalk"],
            "mobile_labels": ["car", "bus", "sky"],
        },
    )
    assert roles["road"] == ["crosswalk", "road", "roads"]
    assert roles["mobile"] == ["bus", "car", "human", "pedestrian", "pedestrians", "person"]
    assert roles["sky"] == ["sky"]


def test_with_semantic_roles_attaches_canonical_metadata() -> None:
    metadata = with_semantic_roles(
        {"source": "unit-test"},
        {"road": ["road"], "sky": ["sky"], "mobile": ["car"]},
    )
    assert metadata["source"] == "unit-test"
    assert metadata[SEMANTIC_ROLES_METADATA_KEY]["road"] == ["road", "roads"]
    assert metadata[SEMANTIC_ROLES_METADATA_KEY]["mobile"] == [
        "car",
        "human",
        "pedestrian",
        "pedestrians",
        "person",
    ]


def test_resolve_role_label_ids_is_case_insensitive_but_exact() -> None:
    label_map = {
        1: "Roads",
        2: "SIDEWALKS",
        3: "Pedestrian",
        4: "road_lane",
    }

    assert resolve_role_label_ids(label_map, "road", tool="CarlaSemanticsProvider") == [1]
    assert resolve_role_label_ids(label_map, "sidewalk", tool="CarlaSemanticsProvider") == [2]
    assert resolve_role_label_ids(label_map, "mobile", tool="CarlaSemanticsProvider") == [3]


def test_merge_semantic_roles_keeps_old_metadata_authoritative() -> None:
    merged = merge_semantic_roles(
        metadata={
            SEMANTIC_ROLES_METADATA_KEY: {
                "road": ["street"],
                "mobile": ["cyclist"],
            }
        },
        tool="CarlaSemanticsProvider",
    )

    assert merged["road"] == ["street"]
    assert merged["mobile"] == ["cyclist"]
