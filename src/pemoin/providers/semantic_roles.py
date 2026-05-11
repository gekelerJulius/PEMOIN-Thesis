"""Canonical semantic role metadata shared by downstream consumers."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence


SEMANTIC_ROLES_METADATA_KEY = "semantic_roles"
SEMANTIC_ROLE_NAMES = ("road", "sidewalk", "mobile", "sky", "large_vehicle")
_UNIVERSAL_ALIAS_FLOOR: dict[str, tuple[str, ...]] = {
    "road": ("road", "roads"),
    "sidewalk": ("sidewalk", "sidewalks"),
    "mobile": ("pedestrian", "pedestrians", "human", "person"),
    "sky": (),
    "large_vehicle": (),
}
_GENERIC_DEFAULTS: dict[str, tuple[str, ...]] = {
    "road": ("road", "path", "crosswalk"),
    "sidewalk": ("sidewalk", "pavement", "walkway"),
    "mobile": (
        "person",
        "human",
        "pedestrian",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
    ),
    "sky": ("sky",),
    "large_vehicle": ("bus", "truck"),
}
_TOOL_DEFAULTS: dict[str, dict[str, tuple[str, ...]]] = {
    "CAVISSemanticsProvider": {
        "road": ("road", "path", "crosswalk"),
        "sidewalk": ("sidewalk", "pavement", "walkway"),
        "sky": ("sky",),
        "mobile": ("person", "car", "bus", "truck", "bicycle", "motorcycle"),
        "large_vehicle": ("bus", "truck"),
    },
    "CarlaSemanticsProvider": {
        "road": ("road", "path", "crosswalk"),
        "sidewalk": ("sidewalk", "pavement", "walkway"),
        "sky": ("sky",),
        "mobile": (
            "person",
            "pedestrian",
            "car",
            "bus",
            "truck",
            "bicycle",
            "motorcycle",
        ),
        "large_vehicle": ("bus", "truck"),
    },
    "UnityGTSemanticsProvider": {
        "road": ("road", "path", "crosswalk"),
        "sidewalk": ("sidewalk", "pavement", "walkway"),
        "sky": ("sky",),
        "mobile": (
            "human",
            "person",
            "bicycle",
            "motorcycle",
        ),
        "large_vehicle": ("bus", "truck"),
    },
    "VirtualKitty2SemanticsProvider": {
        "road": ("road", "lane", "crosswalk"),
        "sidewalk": ("sidewalk", "pavement", "walkway"),
        "sky": ("sky",),
        "mobile": ("person", "car", "bus", "truck", "bicycle", "motorcycle"),
        "large_vehicle": ("bus", "truck"),
    },
    "TemporalFusionSemanticsProvider": {
        "road": (
            "road",
            "roads",
            "street",
            "lane",
            "highway",
            "path",
            "crosswalk",
            "ground",
            "floor",
        ),
        "sidewalk": ("sidewalk", "sidewalks", "pavement", "walkway"),
        "sky": ("sky",),
        "mobile": (
            "human",
            "person",
            "pedestrian",
            "pedestrians",
            "car",
            "bus",
            "truck",
            "bicycle",
            "motorcycle",
        ),
        "large_vehicle": ("bus", "truck"),
    },
}


def _merge_semantic_role_maps(*role_maps: Mapping[str, object]) -> dict[str, list[str]]:
    """Union semantic role tokens across role maps while keeping normalized output."""
    merged: dict[str, set[str]] = {}
    for role_map in role_maps:
        normalized = normalize_semantic_roles(role_map)
        for role, tokens in normalized.items():
            merged.setdefault(role, set()).update(tokens)
    return {role: sorted(tokens) for role, tokens in merged.items() if tokens}


def _normalize_semantic_roles_with_alias_floor(
    raw: Mapping[str, object],
) -> dict[str, list[str]]:
    """Normalize semantic roles and guarantee the built-in alias floor for matching."""
    return _merge_semantic_role_maps(_UNIVERSAL_ALIAS_FLOOR, raw)


def normalize_semantic_roles(raw: Mapping[str, object]) -> dict[str, list[str]]:
    """Normalize semantic role groups into sorted lowercase token lists."""
    normalized: dict[str, list[str]] = {}
    for role, values in raw.items():
        tokens: list[str] = []
        if isinstance(values, str):
            tokens = [
                part.strip().lower() for part in values.split(",") if part.strip()
            ]
        elif isinstance(values, (list, tuple, set)):
            tokens = [str(item).strip().lower() for item in values if str(item).strip()]
        else:
            continue
        deduped = sorted(set(tokens))
        if deduped:
            normalized[str(role).strip().lower()] = deduped
    return normalized


def semantic_role_defaults_for_tool(tool: str | None) -> dict[str, list[str]]:
    """Return provider-owned default semantic role groups for a semantics tool."""
    if tool is None:
        return _merge_semantic_role_maps(_UNIVERSAL_ALIAS_FLOOR, _GENERIC_DEFAULTS)
    return _merge_semantic_role_maps(
        _UNIVERSAL_ALIAS_FLOOR,
        _GENERIC_DEFAULTS,
        _TOOL_DEFAULTS.get(str(tool), {}),
    )


def semantic_roles_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    """Extract canonical semantic roles from persisted metadata."""
    if not isinstance(metadata, Mapping):
        return {}
    raw = metadata.get(SEMANTIC_ROLES_METADATA_KEY)
    if not isinstance(raw, Mapping):
        return {}
    return normalize_semantic_roles(raw)


def merge_semantic_roles(
    *,
    metadata: Mapping[str, Any] | None = None,
    tool: str | None = None,
    defaults: Mapping[str, object] | None = None,
) -> dict[str, list[str]]:
    """Merge provider defaults with persisted metadata, preferring metadata."""
    merged = semantic_role_defaults_for_tool(tool)
    if defaults is not None:
        merged.update(_normalize_semantic_roles_with_alias_floor(defaults))
    merged.update(semantic_roles_from_metadata(metadata))
    sky = set(merged.get("sky", ()))
    if "mobile" in merged:
        merged["mobile"] = sorted(
            token for token in set(merged["mobile"]) if token not in sky
        )
    return merged


def resolve_semantic_role_labels(
    role: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    tool: str | None = None,
    defaults: Mapping[str, object] | None = None,
    required: bool = False,
    source_name: str | None = None,
) -> tuple[str, ...]:
    """Resolve label tokens for one canonical semantic role."""
    normalized_role = str(role).strip().lower()
    labels = tuple(
        merge_semantic_roles(metadata=metadata, tool=tool, defaults=defaults).get(
            normalized_role, []
        )
    )
    if required and not labels:
        origin = f" for {source_name}" if source_name else ""
        raise ValueError(
            f"Required semantic role '{normalized_role}' could not be resolved{origin}."
        )
    return labels


def resolve_role_label_ids(
    label_map: Mapping[int, str],
    role: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    tool: str | None = None,
    defaults: Mapping[str, object] | None = None,
    required: bool = False,
    source_name: str | None = None,
) -> list[int]:
    """Resolve label ids matching a canonical semantic role."""
    labels = set(
        resolve_semantic_role_labels(
            role,
            metadata=metadata,
            tool=tool,
            defaults=defaults,
            required=required,
            source_name=source_name,
        )
    )
    matched = sorted(
        int(label_id)
        for label_id, label_name in label_map.items()
        if str(label_name).strip().lower() in labels
    )
    if required and not matched:
        origin = f" for {source_name}" if source_name else ""
        raise ValueError(
            f"Semantic role '{str(role).strip().lower()}' resolved no label ids{origin}."
        )
    return matched


def first_available_role_labels(
    roles: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    tool: str | None = None,
    defaults: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return the first non-empty role label tuple from a list of canonical roles."""
    for role in roles:
        labels = resolve_semantic_role_labels(
            role, metadata=metadata, tool=tool, defaults=defaults
        )
        if labels:
            return labels
    return ()


def build_semantic_roles(
    defaults: Mapping[str, object],
    *,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Resolve canonical semantic role groups with optional settings overrides."""
    normalized = _normalize_semantic_roles_with_alias_floor(defaults)
    if settings is not None:
        for role in tuple(normalized.keys()):
            override_key = f"{role}_labels"
            if override_key in settings and settings.get(override_key) is not None:
                normalized[role] = _normalize_semantic_roles_with_alias_floor(
                    {role: settings.get(override_key)}
                ).get(role, normalized[role])
    sky = set(normalized.get("sky", ()))
    if "mobile" in normalized:
        normalized["mobile"] = sorted(
            token for token in set(normalized["mobile"]) if token not in sky
        )
    return normalized


def with_semantic_roles(
    metadata: Mapping[str, Any] | None,
    semantic_roles: Mapping[str, object],
) -> MutableMapping[str, Any]:
    """Return metadata enriched with canonical semantic role groups."""
    result: MutableMapping[str, Any] = dict(metadata or {})
    result[SEMANTIC_ROLES_METADATA_KEY] = build_semantic_roles(semantic_roles)
    return result


def semantic_roles_metadata(
    defaults: Mapping[str, object],
    *,
    settings: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Backward-compatible wrapper for existing provider call sites."""
    return with_semantic_roles(
        metadata,
        build_semantic_roles(defaults, settings=settings),
    )
