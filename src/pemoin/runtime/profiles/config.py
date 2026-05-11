"""
Profile configuration loading utilities.

Profiles are defined in declarative JSON files to keep orchestration choices
decoupled from runtime code. JSON is broadly supported, human-readable, and
allows module settings to be expressed alongside tool selections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping


_REMOVED_ROAD_HEIGHT_SCALE_CORRECTION_MESSAGE = (
    "runtime.settings.road_height_scale_correction is no longer supported. "
    "Remove that block and use providers.geometry_fusion for the maintained "
    "metric geometry correction path."
)

_REMOVED_ROAD_HEIGHT_SCALE_CORRECTION_PROVIDER_MESSAGE = (
    "Provider tool 'RoadHeightScaleCorrectionProvider' is no longer supported. "
    "Use 'GeometryFusionProvider' for the maintained metric geometry correction path."
)

_REMOVED_ALIGNMENT_MESSAGE = (
    "runtime.settings.alignment is no longer supported. Remove that block and use "
    "runtime.settings.comparison_frame with providers.geometry_fusion."
)

_REMOVED_GROUNDING_MESSAGE = (
    "runtime.settings.grounding_to_z0 is no longer supported. Grounding is now part "
    "of the maintained runtime.settings.comparison_frame stage."
)

_REMOVED_WORLD_FRAME_MESSAGE = (
    "runtime.settings.world_frame is no longer supported. Use "
    "runtime.settings.comparison_frame.mode='gt' or 'estimated' instead."
)

_CONFIG_DIR_NAME = "config"


@dataclass(frozen=True)
class ModuleBinding:
    """Describes which tool implements a module and its configuration settings."""

    tool: str
    settings: Dict[str, Any]


@dataclass(frozen=True)
class RuntimeBindings:
    """Configuration for runtime-level parameters."""

    state_window: int
    degradation_policy: str
    settings: Dict[str, Any]


@dataclass(frozen=True)
class ProfileConfig:
    """Declarative configuration for a runtime profile."""

    name: str
    runtime: RuntimeBindings
    providers: Dict[str, ModuleBinding]
    effects: Dict[str, ModuleBinding]
    working_resolution: tuple[int, int] | None = None
    frame_provider: ModuleBinding | None = None
    megasam: Dict[str, Any] = field(default_factory=dict)
    depthanything3: Dict[str, Any] = field(default_factory=dict)
    panst3r: Dict[str, Any] = field(default_factory=dict)
    unity_import: Dict[str, Any] = field(default_factory=dict)
    mixamo: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        name: str,
        mapping: Mapping[str, object],
        *,
        config_root: Path,
    ) -> "ProfileConfig":
        """Create a ProfileConfig from a raw mapping."""
        runtime_raw = mapping.get("runtime")
        providers_raw = mapping.get("providers", {})
        effects_raw = mapping.get("effects", {})
        if "working_resolution" not in mapping:
            raise ValueError(
                f"Profile '{name}' is missing required 'working_resolution'."
            )
        working_res_raw = mapping.get("working_resolution")
        working_resolution: tuple[int, int] | None = None
        if isinstance(working_res_raw, (list, tuple)):
            if len(working_res_raw) == 1:
                size = int(working_res_raw[0])
                working_resolution = (size, size)
            elif len(working_res_raw) == 2:
                working_resolution = (int(working_res_raw[0]), int(working_res_raw[1]))
            else:
                raise ValueError(
                    f"Profile '{name}' working_resolution must be a single size or [height, width]."
                )
        elif isinstance(working_res_raw, (int, float)):
            size = int(working_res_raw)
            working_resolution = (size, size)
        else:
            raise ValueError(
                f"Profile '{name}' working_resolution must be a number or list/tuple."
            )

        if working_resolution is None or working_resolution[0] <= 0 or working_resolution[1] <= 0:
            raise ValueError(
                f"Profile '{name}' working_resolution must be positive."
            )

        if runtime_raw is None or not isinstance(runtime_raw, Mapping):
            raise ValueError(f"Profile '{name}' is missing a 'runtime' section.")

        runtime = RuntimeBindings(
            state_window=int(runtime_raw.get("state_window", 0)),
            degradation_policy=str(runtime_raw.get("degradation_policy", "")),
            settings=_ensure_mapping(runtime_raw.get("settings", {}), name, "runtime.settings"),
        )

        providers = _parse_bindings(providers_raw, name, "providers")
        effects = _parse_bindings(effects_raw, name, "effects")
        frame_provider = None
        frame_binding_raw = mapping.get("frame_provider")
        if frame_binding_raw is not None:
            frame_mapping = _parse_bindings({"frame_provider": frame_binding_raw}, name, "frame_provider")
            frame_provider = frame_mapping.get("frame_provider")

        megasam_settings = _ensure_mapping(mapping.get("megasam", {}), name, "megasam")
        depthanything3_settings = _ensure_mapping(mapping.get("depthanything3", {}), name, "depthanything3")
        panst3r_settings = _ensure_mapping(mapping.get("panst3r", {}), name, "panst3r")
        unity_import_settings = _ensure_mapping(mapping.get("unity_import", {}), name, "unity_import")
        mixamo_settings = _ensure_mapping(mapping.get("mixamo", {}), name, "mixamo")
        _validate_unity_import_selection(
            profile_name=name,
            unity_import_settings=unity_import_settings,
            providers=providers,
        )
        _validate_profile_paths(
            profile_name=name,
            config_root=config_root,
            runtime_settings=runtime.settings,
            providers=providers,
            mixamo_settings=mixamo_settings,
        )
        return cls(
            name=name,
            runtime=runtime,
            providers=providers,
            effects=effects,
            working_resolution=working_resolution,
            frame_provider=frame_provider,
            megasam=megasam_settings,
            depthanything3=depthanything3_settings,
            panst3r=panst3r_settings,
            unity_import=unity_import_settings,
            mixamo=mixamo_settings,
        )


def _ensure_mapping(value: object, profile_name: str, section_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Profile '{profile_name}' has invalid '{section_name}' entries; expected object."
        )
    normalized = {str(k): v for k, v in value.items()}
    if section_name == "runtime.settings" and "road_height_scale_correction" in normalized:
        raise ValueError(_REMOVED_ROAD_HEIGHT_SCALE_CORRECTION_MESSAGE)
    if section_name == "runtime.settings" and "alignment" in normalized:
        raise ValueError(_REMOVED_ALIGNMENT_MESSAGE)
    if section_name == "runtime.settings" and "grounding_to_z0" in normalized:
        raise ValueError(_REMOVED_GROUNDING_MESSAGE)
    if section_name == "runtime.settings" and "world_frame" in normalized:
        raise ValueError(_REMOVED_WORLD_FRAME_MESSAGE)
    return normalized


def _parse_bindings(
    section: object, profile_name: str, section_name: str
) -> Dict[str, ModuleBinding]:
    if not isinstance(section, Mapping):
        raise ValueError(
            f"Profile '{profile_name}' has invalid '{section_name}' entries; expected object."
        )

    bindings: Dict[str, ModuleBinding] = {}
    for module_name, module_config in section.items():
        if not isinstance(module_config, Mapping):
            raise ValueError(
                f"Profile '{profile_name}' module '{module_name}' must be an object."
            )

        tool = module_config.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError(
                f"Profile '{profile_name}' module '{module_name}' requires a non-empty 'tool' string."
            )
        if tool == "RoadHeightScaleCorrectionProvider":
            raise ValueError(_REMOVED_ROAD_HEIGHT_SCALE_CORRECTION_PROVIDER_MESSAGE)

        settings = _ensure_mapping(
            module_config.get("settings", {}), profile_name, f"{section_name}.{module_name}.settings"
        )
        bindings[str(module_name)] = ModuleBinding(tool=tool, settings=settings)

    return bindings


def _validate_unity_import_selection(
    *,
    profile_name: str,
    unity_import_settings: Mapping[str, Any],
    providers: Mapping[str, ModuleBinding],
) -> None:
    if not bool(unity_import_settings.get("enabled")):
        return
    resources_raw = unity_import_settings.get("resources", {})
    if resources_raw is None:
        resources: Dict[str, Any] = {}
    elif isinstance(resources_raw, Mapping):
        resources = {str(k): v for k, v in resources_raw.items()}
    else:
        raise ValueError(
            f"Profile '{profile_name}' has invalid 'unity_import.resources'; expected object."
        )

    if resources.get("frames") is False:
        raise ValueError(
            f"Profile '{profile_name}' sets unity_import.resources.frames=false, "
            "but unity_import-enabled runs require imported RGB frames."
        )

    required_resources: list[tuple[str, str]] = []
    for binding in providers.values():
        if binding.tool == "UnityGTDepthProvider":
            required_resources.append(("depth", binding.tool))
        if binding.tool == "UnityGTSemanticsProvider":
            required_resources.append(("semantics", binding.tool))

    for resource_name, tool_name in required_resources:
        if resources.get(resource_name) is False:
            raise ValueError(
                f"Profile '{profile_name}' sets unity_import.resources.{resource_name}=false, "
                f"but provider '{tool_name}' requires that imported Unity resource."
            )


def _config_root_from_path(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    if resolved.parent.name == _CONFIG_DIR_NAME:
        return resolved.parent.parent
    return resolved.parent


def _resolve_config_path(path_value: str, *, config_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (config_root / path).resolve()
    return path


def _require_existing_path(
    *,
    profile_name: str,
    key_path: str,
    path_value: object,
    config_root: Path,
    expected: str,
) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(
            f"Profile '{profile_name}' requires non-empty '{key_path}' path."
        )
    resolved = _resolve_config_path(path_value, config_root=config_root)
    exists = resolved.is_file() if expected == "file" else resolved.is_dir()
    if not exists:
        raise ValueError(
            f"Profile '{profile_name}' path '{key_path}' must resolve to an existing {expected}: "
            f"{resolved}"
        )
    return resolved


def _validate_profile_paths(
    *,
    profile_name: str,
    config_root: Path,
    runtime_settings: Mapping[str, Any],
    providers: Mapping[str, ModuleBinding],
    mixamo_settings: Mapping[str, Any],
) -> None:
    if mixamo_settings:
        _require_existing_path(
            profile_name=profile_name,
            key_path="mixamo.character_fbx_path",
            path_value=mixamo_settings.get("character_fbx_path"),
            config_root=config_root,
            expected="file",
        )
        _require_existing_path(
            profile_name=profile_name,
            key_path="mixamo.animation_fbx_path",
            path_value=mixamo_settings.get("animation_fbx_path"),
            config_root=config_root,
            expected="file",
        )
        asset_root = mixamo_settings.get("asset_root")
        if asset_root is not None:
            _require_existing_path(
                profile_name=profile_name,
                key_path="mixamo.asset_root",
                path_value=asset_root,
                config_root=config_root,
                expected="directory",
            )

    harmonisation = runtime_settings.get("harmonisation")
    if isinstance(harmonisation, Mapping) and harmonisation.get("pretrained_path"):
        _require_existing_path(
            profile_name=profile_name,
            key_path="runtime.settings.harmonisation.pretrained_path",
            path_value=harmonisation.get("pretrained_path"),
            config_root=config_root,
            expected="file",
        )

    lighting = providers.get("lighting")
    if lighting is not None and lighting.settings.get("repo_root"):
        _require_existing_path(
            profile_name=profile_name,
            key_path="providers.lighting.settings.repo_root",
            path_value=lighting.settings.get("repo_root"),
            config_root=config_root,
            expected="directory",
        )

    semantics = providers.get("semantics")
    if semantics is not None and semantics.settings.get("label_map_path"):
        _require_existing_path(
            profile_name=profile_name,
            key_path="providers.semantics.settings.label_map_path",
            path_value=semantics.settings.get("label_map_path"),
            config_root=config_root,
            expected="file",
        )

    for module_name, binding in providers.items():
        adapter_settings = binding.settings.get("adapter")
        if not isinstance(adapter_settings, Mapping):
            continue
        if adapter_settings.get("checkpoint_path"):
            _require_existing_path(
                profile_name=profile_name,
                key_path=f"providers.{module_name}.settings.adapter.checkpoint_path",
                path_value=adapter_settings.get("checkpoint_path"),
                config_root=config_root,
                expected="file",
            )
        if adapter_settings.get("config_path"):
            _require_existing_path(
                profile_name=profile_name,
                key_path=f"providers.{module_name}.settings.adapter.config_path",
                path_value=adapter_settings.get("config_path"),
                config_root=config_root,
                expected="file",
            )


def load_profiles_from_json(path: Path) -> Dict[str, ProfileConfig]:
    """
    Load profile configurations from a JSON file.

    Args:
        path: Filesystem path to the JSON configuration.

    Returns:
        Mapping from profile name to ProfileConfig instance.
    """
    with path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    config_root = _config_root_from_path(path)

    profiles_section = raw_config.get("profiles")
    if not isinstance(profiles_section, Mapping):
        raise ValueError("Profile configuration must contain a top-level 'profiles' object.")

    configs: Dict[str, ProfileConfig] = {}
    for name, mapping in profiles_section.items():
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Profile '{name}' configuration must be an object.")
        configs[str(name)] = ProfileConfig.from_mapping(
            str(name),
            mapping,
            config_root=config_root,
        )

    return configs
