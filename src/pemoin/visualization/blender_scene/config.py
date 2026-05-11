from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pemoin.providers.semantic_roles import resolve_semantic_role_labels
from pemoin.visualization.pedestrian_placement import (
    resolve_unity_world_horizontal_placement,
)

from .mixamo_assets import resolve_mixamo_asset_package
from .constants import (
    _ALLOWED_LIGHTING_PRESETS,
    _ALLOWED_RENDER_ENGINES,
    _ALLOWED_SHADOW_CUBE_SIZES,
    _LIGHTING_PRESET_NEUTRAL_HEMISPHERE,
)
from .specs import (
    EdgeTreatmentSpec,
    LightSpec,
    LightingRigSpec,
    OcclusionSpec,
    TemporalOcclusionStabilizationSpec,
    RawSubjectExposureSpec,
    RenderPerformanceSpec,
    SalienceAdaptiveRenderSpec,
    RenderSpec,
    SceneSpec,
    ShadowSpec,
    WrapSubjectFillSpec,
)

_ALLOWED_MATERIAL_POLICIES = (
    "preserve_most_maps",
    "preserve_base_alpha_normal",
    "preserve_base_alpha",
)
_ALLOWED_DYNAMIC_LIGHT_BINDINGS = (
    "copy_location_constraint",
    "sparse_keyframes",
    "spawn_only_static",
)
_LEGACY_PEDESTRIAN_PLACEMENT_KEYS = (
    "pedestrian_trajectory_t",
    "pedestrian_forward_offset_m",
    "pedestrian_left_offset_m",
    "pedestrian_up_offset_m",
)


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[4]
    return (repo_root / path).resolve()


def _coerce_metadata(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, np.ndarray) and metadata.shape == ():
        metadata = metadata.item()
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    return metadata


def _load_profile_config(config_path: Path, profile_name: str) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Profile config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        snapshot_profile_name = raw.get("profile")
        if snapshot_profile_name is not None:
            if str(snapshot_profile_name) != profile_name:
                raise ValueError(
                    f"Profile snapshot at {config_path} is for '{snapshot_profile_name}', "
                    f"not '{profile_name}'."
                )
            return raw
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(
            "Profile configuration must contain a top-level 'profiles' object or "
            "be a saved run profile snapshot."
        )
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Profile '{profile_name}' was not found in {config_path}.")
    return profile


def _raw_sampling_fps_from_profile(profile: dict) -> object | None:
    frame_provider = profile.get("frame_provider", {})
    if isinstance(frame_provider, dict):
        frame_settings = frame_provider.get("settings", {})
        if isinstance(frame_settings, dict):
            sampling_fps = frame_settings.get("resolved_sampling_fps")
            if sampling_fps is not None:
                return sampling_fps
            sampling_fps = frame_settings.get("sampling_fps")
            if sampling_fps is not None:
                return sampling_fps
    unity_import = profile.get("unity_import", {})
    if isinstance(unity_import, dict):
        sampling_fps = unity_import.get("sampling_fps")
        if sampling_fps is not None:
            return sampling_fps
    return None


def _float_setting(
    settings: dict,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = float(settings.get(key, default))
    if minimum is not None and value < minimum:
        raise ValueError(f"Invalid {key}: {value} (must be >= {minimum})")
    if maximum is not None and value > maximum:
        raise ValueError(f"Invalid {key}: {value} (must be <= {maximum})")
    return value


def _int_setting(
    settings: dict,
    key: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    value = int(settings.get(key, default))
    if minimum is not None and value < minimum:
        raise ValueError(f"Invalid {key}: {value} (must be >= {minimum})")
    return value


def _vec3_setting(
    settings: dict,
    key: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    raw = settings.get(key, default)
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid {key}: expected comma-separated x,y,z triple."
            )
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid {key}: expected numeric x,y,z triple."
            ) from exc
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"Invalid {key}: expected [x, y, z] triple.")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _phase_ranges_setting(
    settings: dict,
    key: str,
    default: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    raw = settings.get(key, default)
    if raw is None:
        return tuple(default)
    if isinstance(raw, str):
        parsed: list[tuple[float, float]] = []
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        for part in parts:
            if "-" not in part:
                raise ValueError(
                    f"Invalid {key} segment '{part}'. Expected 'start-end' in [0,1]."
                )
            start_text, end_text = part.split("-", 1)
            start = float(start_text.strip())
            end = float(end_text.strip())
            parsed.append((start, end))
        raw = parsed
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"Invalid {key}: expected list/tuple of [start, end] phase intervals."
        )
    ranges: list[tuple[float, float]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"Invalid {key}[{idx}]: expected [start, end] numeric interval."
            )
        start = float(item[0])
        end = float(item[1])
        if not (0.0 <= start <= 1.0) or not (0.0 <= end <= 1.0):
            raise ValueError(
                f"Invalid {key}[{idx}]: start/end must be in [0,1], got ({start}, {end})."
            )
        if abs(start - end) < 1e-6:
            raise ValueError(
                f"Invalid {key}[{idx}]: zero-length interval ({start}, {end})."
            )
        ranges.append((start, end))
    return tuple(ranges)


def _first_semantics_metadata(run_dir: Path) -> dict[str, Any]:
    semantics_dir = run_dir / "standard" / "semantics_2d"
    for path in sorted(semantics_dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            return _coerce_metadata(data["metadata"]) if "metadata" in data.files else {}
    return {}


def _semantics_tool_from_profile(profile: dict | None = None) -> str | None:
    if not isinstance(profile, dict):
        return None
    providers = profile.get("providers", {})
    if not isinstance(providers, dict):
        return None
    semantics = providers.get("semantics", {})
    if not isinstance(semantics, dict):
        return None
    tool = semantics.get("tool")
    return None if tool is None else str(tool)


def _road_labels_setting(run_dir: Path, profile: dict | None = None) -> tuple[str, ...]:
    semantics_tool = _semantics_tool_from_profile(profile)
    labels = resolve_semantic_role_labels(
        "road",
        metadata=_first_semantics_metadata(run_dir),
        tool=semantics_tool,
        required=True,
        source_name="Blender scene generation",
    )
    if not labels:
        raise ValueError(
            "Canonical semantic role 'road' could not be resolved for Blender scene generation."
        )
    return labels


def _string_tuple_setting(raw: Any, key: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"Invalid {key}: expected string or list of strings.")
    values: list[str] = []
    for idx, item in enumerate(raw):
        text = str(item).strip().lower()
        if not text:
            raise ValueError(f"Invalid {key}[{idx}]: expected non-empty string.")
        values.append(text)
    return tuple(values)


def _trajectory_metadata(trajectory_path: Path) -> dict[str, Any]:
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")
    with np.load(trajectory_path, allow_pickle=True) as data:
        if "metadata" not in data.files:
            return {}
        metadata = data["metadata"]
        if isinstance(metadata, np.ndarray) and metadata.shape == ():
            metadata = metadata.item()
        if not isinstance(metadata, dict):
            raise ValueError("Trajectory metadata must be a dictionary.")
        return metadata


def _profile_supports_unity_world_horizontal_placement(
    profile_name: str,
    profile: dict | None,
) -> bool:
    if str(profile_name).strip().lower().startswith("unity_"):
        return True
    if not isinstance(profile, dict):
        return False
    unity_import = profile.get("unity_import", {})
    if isinstance(unity_import, dict) and bool(unity_import.get("enabled", False)):
        return True
    frame_provider = profile.get("frame_provider", {})
    if isinstance(frame_provider, dict) and str(frame_provider.get("tool", "")).strip() == "UnityFrameProvider":
        return True
    return False


def _resolve_profile_pedestrian_placement(
    *,
    blender_settings: dict[str, Any],
    trajectory_path: Path,
    profile_name: str,
    profile: dict | None,
) -> dict[str, Any]:
    raw = blender_settings.get("pedestrian_placement")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("runtime.settings.blender_scene.pedestrian_placement must be an object.")
    mode = str(raw.get("mode", "unity_world_horizontal")).strip().lower()
    if mode != "unity_world_horizontal":
        raise ValueError(
            "runtime.settings.blender_scene.pedestrian_placement.mode must be 'unity_world_horizontal'."
        )
    if not _profile_supports_unity_world_horizontal_placement(profile_name, profile):
        raise ValueError(
            "runtime.settings.blender_scene.pedestrian_placement.mode='unity_world_horizontal' "
            f"is only supported for Unity-backed profiles, got '{profile_name}'."
        )
    legacy_keys = [key for key in _LEGACY_PEDESTRIAN_PLACEMENT_KEYS if key in blender_settings]
    if legacy_keys:
        raise ValueError(
            "runtime.settings.blender_scene.pedestrian_placement replaces legacy pedestrian placement "
            f"keys; remove {', '.join(sorted(legacy_keys))}."
        )
    position_x_m = _float_setting(raw, "position_x_m", 0.0)
    position_z_m = _float_setting(raw, "position_z_m", 0.0)
    heading_yaw_deg = _float_setting(raw, "heading_yaw_deg", 0.0)
    metadata = _trajectory_metadata(trajectory_path)
    comparison_frame = metadata.get("comparison_frame", {})
    if not isinstance(comparison_frame, dict):
        raise ValueError("Trajectory comparison-frame metadata must be an object.")
    authoring_frame = comparison_frame.get("authoring_frame", {})
    if not isinstance(authoring_frame, dict) or not authoring_frame:
        raise ValueError(
            "Unity-authored pedestrian placement requires comparison-frame authoring metadata."
        )
    transform = authoring_frame.get("authoring_to_canonical_transform")
    if transform is None:
        raise ValueError(
            "Unity-authored pedestrian placement requires authoring_to_canonical_transform metadata."
        )
    spawn_world, forward_world, heading_world_deg, diagnostics = (
        resolve_unity_world_horizontal_placement(
            transform,
            position_x_m=position_x_m,
            position_z_m=position_z_m,
            heading_yaw_deg=heading_yaw_deg,
        )
    )
    return {
        "pedestrian_placement_mode": mode,
        "pedestrian_authored_position_x_m": float(position_x_m),
        "pedestrian_authored_position_z_m": float(position_z_m),
        "pedestrian_authored_heading_yaw_deg": float(heading_yaw_deg),
        "pedestrian_authoring_to_canonical_transform": tuple(
            tuple(float(v) for v in row) for row in np.asarray(transform, dtype=np.float32).tolist()
        ),
        "pedestrian_authoring_frame_metadata": {
            **authoring_frame,
            "resolved_placement": diagnostics,
        },
        "pedestrian_resolved_spawn_world": tuple(float(v) for v in spawn_world.tolist()),
        "pedestrian_resolved_forward_world": tuple(float(v) for v in forward_world.tolist()),
        "pedestrian_resolved_heading_world_deg": float(heading_world_deg),
    }


def _contact_ground_labels_setting(
    *,
    run_dir: Path,
    profile: dict | None = None,
    roles: tuple[str, ...] = ("road", "sidewalk"),
    extra_labels: tuple[str, ...] = (),
) -> tuple[str, ...]:
    semantics_tool = _semantics_tool_from_profile(profile)
    metadata = _first_semantics_metadata(run_dir)
    labels: list[str] = []
    for role in roles:
        labels.extend(
            resolve_semantic_role_labels(
                role,
                metadata=metadata,
                tool=semantics_tool,
                required=False,
                source_name="Blender contact-aware occlusion",
            )
        )
    labels.extend(str(label).strip().lower() for label in extra_labels if str(label).strip())
    deduped = tuple(sorted(set(labels)))
    if not deduped:
        raise ValueError(
            "Blender contact-aware occlusion resolved no traversable-ground labels."
        )
    return deduped


def _bool_setting(settings: dict, key: str, default: bool) -> bool:
    lookup_key = key if key in settings else str(key).rsplit(".", 1)[-1]
    value = settings.get(lookup_key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Invalid {key}: expected boolean value.")
    return value


def _finite_float(
    value: Any,
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Invalid {key}: expected finite numeric value.")
    if minimum is not None and result < minimum:
        raise ValueError(f"Invalid {key}: {result} (must be >= {minimum})")
    if maximum is not None and result > maximum:
        raise ValueError(f"Invalid {key}: {result} (must be <= {maximum})")
    return result


def _finite_vec3(
    value: Any,
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid {key}: expected [x, y, z] triple.")
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid {key}: expected finite [x, y, z] triple.")
    if minimum is not None and float(arr.min()) < minimum:
        raise ValueError(f"Invalid {key}: values must be >= {minimum}.")
    if maximum is not None and float(arr.max()) > maximum:
        raise ValueError(f"Invalid {key}: values must be <= {maximum}.")
    return tuple(float(v) for v in arr.tolist())


def _default_lighting_rig(
    *,
    ambient_world_strength: float = 0.12,
    shadow_cube_size: str = "2048",
) -> LightingRigSpec:
    lights = (
        LightSpec("TopFrontLeft", "SUN", 2.5, (50.0, 35.0, 0.0), (1.0, 0.98, 0.96), angle_deg=55.0),
        LightSpec("TopFrontRight", "SUN", 2.5, (50.0, -35.0, 0.0), (1.0, 0.98, 0.96), angle_deg=55.0),
        LightSpec("TopBackLeft", "SUN", 2.0, (50.0, 145.0, 0.0), (0.98, 0.98, 1.0), angle_deg=55.0),
        LightSpec("TopBackRight", "SUN", 2.0, (50.0, -145.0, 0.0), (0.98, 0.98, 1.0), angle_deg=55.0),
        LightSpec("SideLeft", "SUN", 1.4, (80.0, 90.0, 0.0), (1.0, 1.0, 1.0), angle_deg=65.0),
        LightSpec("SideRight", "SUN", 1.4, (80.0, -90.0, 0.0), (1.0, 1.0, 1.0), angle_deg=65.0),
    )
    return LightingRigSpec(
        enabled=True,
        preset=_LIGHTING_PRESET_NEUTRAL_HEMISPHERE,
        ambient_world_strength=ambient_world_strength,
        shadow_cube_size=shadow_cube_size,
        wrap_subject_fill=WrapSubjectFillSpec(),
        lights=lights,
    )


def _wrap_subject_fill_spec_from_mapping(raw: Any, key: str) -> WrapSubjectFillSpec:
    if raw is None:
        return WrapSubjectFillSpec()
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid {key}: expected object.")
    return WrapSubjectFillSpec(
        global_strength_scale=_finite_float(
            raw.get("global_strength_scale", 2.0),
            f"{key}.global_strength_scale",
            minimum=0.1,
            maximum=8.0,
        ),
        wrap_key_role_scale=_finite_float(
            raw.get("wrap_key_role_scale", 0.08),
            f"{key}.wrap_key_role_scale",
            minimum=0.0,
            maximum=2.0,
        ),
        counter_wrap_role_scale=_finite_float(
            raw.get("counter_wrap_role_scale", 0.035),
            f"{key}.counter_wrap_role_scale",
            minimum=0.0,
            maximum=2.0,
        ),
        sky_fill_role_scale=_finite_float(
            raw.get("sky_fill_role_scale", 0.02),
            f"{key}.sky_fill_role_scale",
            minimum=0.0,
            maximum=2.0,
        ),
        counter_side_lift_bias=_finite_float(
            raw.get("counter_side_lift_bias", 0.6),
            f"{key}.counter_side_lift_bias",
            minimum=0.0,
            maximum=1.0,
        ),
        sky_softness_bias=_finite_float(
            raw.get("sky_softness_bias", 0.55),
            f"{key}.sky_softness_bias",
            minimum=0.0,
            maximum=1.0,
        ),
        direct_preservation_bias=_finite_float(
            raw.get("direct_preservation_bias", 0.35),
            f"{key}.direct_preservation_bias",
            minimum=0.0,
            maximum=1.0,
        ),
        raw_exposure_trim=_finite_float(
            raw.get("raw_exposure_trim", 1.0),
            f"{key}.raw_exposure_trim",
            minimum=0.75,
            maximum=1.25,
        ),
    )


def _light_spec_from_mapping(settings: dict[str, Any], key_prefix: str) -> LightSpec:
    name = str(settings.get("name", "")).strip()
    if not name:
        raise ValueError(f"Invalid {key_prefix}.name: expected non-empty string.")
    kind = str(settings.get("kind", "")).strip().upper()
    if kind not in {"SUN", "AREA", "POINT"}:
        raise ValueError(f"Invalid {key_prefix}.kind: expected 'SUN', 'AREA', or 'POINT'.")
    energy = _finite_float(settings.get("energy"), f"{key_prefix}.energy", minimum=0.0)
    if energy <= 0.0:
        raise ValueError(f"Invalid {key_prefix}.energy: must be > 0.")
    rotation = _finite_vec3(
        settings.get("rotation_euler_deg"), f"{key_prefix}.rotation_euler_deg"
    )
    color = _finite_vec3(
        settings.get("color"), f"{key_prefix}.color", minimum=0.0, maximum=1.0
    )
    angle_deg: float | None = None
    area_size: tuple[float, float] | None = None
    location: tuple[float, float, float] | None = None
    if kind == "SUN":
        angle_deg = _finite_float(
            settings.get("angle_deg"), f"{key_prefix}.angle_deg", minimum=0.0
        )
        if angle_deg <= 0.0:
            raise ValueError(f"Invalid {key_prefix}.angle_deg: must be > 0.")
    elif kind == "AREA":
        area_size = _finite_vec3(
            (
                *settings.get("area_size", (settings.get("size"), settings.get("size"))),
                0.0,
            )[:3],
            f"{key_prefix}.area_size",
            minimum=0.0,
        )[:2]
        if float(area_size[0]) <= 0.0 or float(area_size[1]) <= 0.0:
            raise ValueError(f"Invalid {key_prefix}.area_size: values must be > 0.")
        location = _finite_vec3(settings.get("location"), f"{key_prefix}.location")
    else:
        location = _finite_vec3(settings.get("location"), f"{key_prefix}.location")
    return LightSpec(
        name=name,
        kind=kind,
        energy=energy,
        rotation_euler_deg=rotation,
        color=color,
        angle_deg=angle_deg,
        area_size=area_size,
        location=location,
    )


def _lighting_rig_from_mapping(raw: Any, key: str) -> LightingRigSpec:
    if raw is None:
        return _default_lighting_rig()
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid {key}: expected object.")
    enabled = _bool_setting(raw, f"{key}.enabled", True)
    preset = str(raw.get("preset", _LIGHTING_PRESET_NEUTRAL_HEMISPHERE)).strip().lower()
    if preset not in _ALLOWED_LIGHTING_PRESETS:
        raise ValueError(
            f"Invalid {key}.preset: {preset!r} (expected one of {_ALLOWED_LIGHTING_PRESETS})."
        )
    ambient_world_strength = _finite_float(
        raw.get("ambient_world_strength", 0.12),
        f"{key}.ambient_world_strength",
        minimum=0.0,
    )
    shadow_cube_size = str(raw.get("shadow_cube_size", "2048")).strip()
    if shadow_cube_size not in _ALLOWED_SHADOW_CUBE_SIZES:
        raise ValueError(
            f"Invalid {key}.shadow_cube_size: "
            f"{shadow_cube_size!r} (expected one of {_ALLOWED_SHADOW_CUBE_SIZES})."
        )
    wrap_subject_fill = _wrap_subject_fill_spec_from_mapping(
        raw.get("wrap_subject_fill"),
        f"{key}.wrap_subject_fill",
    )
    lights_raw = raw.get("lights")
    if lights_raw is None:
        default_spec = _default_lighting_rig(
            ambient_world_strength=ambient_world_strength,
            shadow_cube_size=shadow_cube_size,
        )
        lights = default_spec.lights
    else:
        if not isinstance(lights_raw, (list, tuple)) or not lights_raw:
            raise ValueError("Invalid lighting.lights: expected non-empty list.")
        parsed_lights: list[LightSpec] = []
        for idx, light_raw in enumerate(lights_raw):
            if not isinstance(light_raw, dict):
                raise ValueError(f"Invalid lighting.lights[{idx}]: expected object.")
            parsed_lights.append(
                _light_spec_from_mapping(light_raw, f"lighting.lights[{idx}]")
            )
        lights = tuple(parsed_lights)
    return LightingRigSpec(
        enabled=enabled,
        preset=preset,
        ambient_world_strength=ambient_world_strength,
        shadow_cube_size=shadow_cube_size,
        wrap_subject_fill=wrap_subject_fill,
        lights=lights,
    )


def _lighting_rig_from_cli_args(args: Any) -> LightingRigSpec:
    ambient_world_strength = (
        _finite_float(args.ambient_world_strength, "ambient-world-strength", minimum=0.0)
        if args.ambient_world_strength is not None
        else 0.12
    )
    shadow_cube_size = str(args.shadow_cube_size).strip() if args.shadow_cube_size is not None else "2048"
    if shadow_cube_size not in _ALLOWED_SHADOW_CUBE_SIZES:
        raise ValueError(
            "shadow-cube-size must be one of: " + ", ".join(_ALLOWED_SHADOW_CUBE_SIZES)
        )
    preset = str(args.lighting_preset).strip().lower() if args.lighting_preset is not None else _LIGHTING_PRESET_NEUTRAL_HEMISPHERE
    if preset not in _ALLOWED_LIGHTING_PRESETS:
        raise ValueError(
            "lighting-preset must be one of: " + ", ".join(_ALLOWED_LIGHTING_PRESETS)
        )
    return replace(
        _default_lighting_rig(
            ambient_world_strength=ambient_world_strength,
            shadow_cube_size=shadow_cube_size,
        ),
        preset=preset,
    )


def _shadow_spec_from_mapping(raw: Any, key: str) -> ShadowSpec:
    if raw is None:
        return ShadowSpec()
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid {key}: expected object.")
    if "mode" in raw:
        mode = str(raw.get("mode")).strip().lower()
        raise ValueError(
            f"Invalid {key}.mode: {mode!r}. "
            "Shadow mode is no longer configurable; PEMOIN always uses single-pass receiver-luma shadows."
        )
    softness = raw.get("softness", raw.get("blur_radius_px", 1.5))
    map_resolution = str(raw.get("map_resolution", "1024")).strip()
    if map_resolution not in _ALLOWED_SHADOW_CUBE_SIZES:
        raise ValueError(
            f"Invalid {key}.map_resolution: "
            f"{map_resolution!r} (expected one of {_ALLOWED_SHADOW_CUBE_SIZES})."
        )
    return ShadowSpec(
        enabled=_bool_setting(raw, f"{key}.enabled", True),
        receiver_patch_size_m=_finite_float(
            raw.get("receiver_patch_size_m", 4.0),
            f"{key}.receiver_patch_size_m",
            minimum=0.05,
        ),
        map_resolution=map_resolution,  # type: ignore[arg-type]
        softness=_finite_float(
            softness,
            f"{key}.softness",
            minimum=0.0,
        ),
        opacity=_finite_float(
            raw.get("opacity", 1.0),
            f"{key}.opacity",
            minimum=0.0,
            maximum=1.0,
        ),
        tint_rgb=_finite_vec3(
            raw.get("tint_rgb", (0.0, 0.0, 0.0)),
            f"{key}.tint_rgb",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _render_spec_from_mapping(raw: Any, key: str) -> RenderSpec:
    if raw is None:
        return RenderSpec()
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid {key}: expected object.")
    engine = str(raw.get("engine", "raster")).strip().lower()
    if engine not in _ALLOWED_RENDER_ENGINES:
        raise ValueError(
            f"Invalid {key}.engine: {engine!r} (expected one of {_ALLOWED_RENDER_ENGINES})."
        )
    if "tiny_object" in raw:
        raise ValueError(
            f"Invalid {key}.tiny_object: adaptive tiny-object rerender settings were removed. "
            "Use render.resolution_scale to choose a fixed one-pass internal render scale."
        )
    material_policy = str(
        raw.get("material_policy", "preserve_base_alpha_normal")
    ).strip().lower()
    if material_policy not in _ALLOWED_MATERIAL_POLICIES:
        raise ValueError(
            f"Invalid {key}.material_policy: {material_policy!r} "
            f"(expected one of {_ALLOWED_MATERIAL_POLICIES})."
        )
    dynamic_light_binding = str(
        raw.get("dynamic_light_binding", "copy_location_constraint")
    ).strip().lower()
    if dynamic_light_binding not in _ALLOWED_DYNAMIC_LIGHT_BINDINGS:
        raise ValueError(
            f"Invalid {key}.dynamic_light_binding: {dynamic_light_binding!r} "
            f"(expected one of {_ALLOWED_DYNAMIC_LIGHT_BINDINGS})."
        )
    raw_subject_exposure_raw = raw.get("raw_subject_exposure", {})
    if not isinstance(raw_subject_exposure_raw, dict):
        raise ValueError(f"Invalid {key}.raw_subject_exposure: expected object.")
    performance_raw = raw.get("performance", {})
    if not isinstance(performance_raw, dict):
        raise ValueError(f"Invalid {key}.performance: expected object.")
    salience_adaptive_raw = raw.get("salience_adaptive", {})
    if not isinstance(salience_adaptive_raw, dict):
        raise ValueError(f"Invalid {key}.salience_adaptive: expected object.")
    resolution_scale = _finite_float(
        raw.get("resolution_scale", 1.0),
        f"{key}.resolution_scale",
        minimum=0.1,
        maximum=4.0,
    )
    low_salience_resolution_scale = _finite_float(
        salience_adaptive_raw.get("low_salience_resolution_scale", 0.85),
        f"{key}.salience_adaptive.low_salience_resolution_scale",
        minimum=0.1,
        maximum=4.0,
    )
    if low_salience_resolution_scale > resolution_scale:
        raise ValueError(
            f"Invalid {key}.salience_adaptive.low_salience_resolution_scale: "
            f"{low_salience_resolution_scale} (must be <= render.resolution_scale={resolution_scale})."
        )
    return RenderSpec(
        engine=engine,  # type: ignore[arg-type]
        resolution_scale=resolution_scale,
        samples=int(
            _finite_float(
                raw.get("samples", 16),
                f"{key}.samples",
                minimum=1.0,
            )
        ),
        material_policy=material_policy,  # type: ignore[arg-type]
        dynamic_light_binding=dynamic_light_binding,  # type: ignore[arg-type]
        performance=RenderPerformanceSpec(
            persistent_data=_bool_setting(
                performance_raw,
                f"{key}.performance.persistent_data",
                True,
            ),
            fast_png_compression=_bool_setting(
                performance_raw,
                f"{key}.performance.fast_png_compression",
                True,
            ),
            disable_raytracing=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_raytracing",
                True,
            ),
            disable_volumetric_shadows=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_volumetric_shadows",
                True,
            ),
            disable_volumetric_lighting=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_volumetric_lighting",
                True,
            ),
            disable_bloom=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_bloom",
                True,
            ),
            disable_screen_space_reflections=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_screen_space_reflections",
                True,
            ),
            disable_gtao=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_gtao",
                True,
            ),
            disable_motion_blur=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_motion_blur",
                True,
            ),
            disable_high_quality_normals=_bool_setting(
                performance_raw,
                f"{key}.performance.disable_high_quality_normals",
                True,
            ),
        ),
        salience_adaptive=SalienceAdaptiveRenderSpec(
            enabled=_bool_setting(
                salience_adaptive_raw,
                f"{key}.salience_adaptive.enabled",
                True,
            ),
            low_salience_resolution_scale=low_salience_resolution_scale,
            protect_below_visible_pixels=_int_setting(
                salience_adaptive_raw,
                "protect_below_visible_pixels",
                10000,
                minimum=1,
            ),
            protect_below_bbox_short_side_px=_int_setting(
                salience_adaptive_raw,
                "protect_below_bbox_short_side_px",
                56,
                minimum=1,
            ),
            protect_when_center_distance_ratio_below=_finite_float(
                salience_adaptive_raw.get("protect_when_center_distance_ratio_below", 0.30),
                f"{key}.salience_adaptive.protect_when_center_distance_ratio_below",
                minimum=0.0,
                maximum=1.5,
            ),
            reduce_only_when_boundary_fraction_above=_finite_float(
                salience_adaptive_raw.get("reduce_only_when_boundary_fraction_above", 0.24),
                f"{key}.salience_adaptive.reduce_only_when_boundary_fraction_above",
                minimum=0.0,
                maximum=1.0,
            ),
            reduce_only_near_visibility_transition=_bool_setting(
                salience_adaptive_raw,
                f"{key}.salience_adaptive.reduce_only_near_visibility_transition",
                True,
            ),
            shadow_quality_reduction_enabled=_bool_setting(
                salience_adaptive_raw,
                f"{key}.salience_adaptive.shadow_quality_reduction_enabled",
                True,
            ),
            fill_light_reduction_enabled=_bool_setting(
                salience_adaptive_raw,
                f"{key}.salience_adaptive.fill_light_reduction_enabled",
                True,
            ),
        ),
        raw_subject_exposure=RawSubjectExposureSpec(
            enabled=_bool_setting(
                raw_subject_exposure_raw,
                f"{key}.raw_subject_exposure.enabled",
                True,
            ),
            target_match_strength=_finite_float(
                raw_subject_exposure_raw.get("target_match_strength", 0.75),
                f"{key}.raw_subject_exposure.target_match_strength",
                minimum=0.0,
                maximum=1.0,
            ),
            max_gain=_finite_float(
                raw_subject_exposure_raw.get("max_gain", 2.5),
                f"{key}.raw_subject_exposure.max_gain",
                minimum=1.0,
            ),
            validation_tolerance=_finite_float(
                raw_subject_exposure_raw.get("validation_tolerance", 0.18),
                f"{key}.raw_subject_exposure.validation_tolerance",
                minimum=0.0,
                maximum=1.0,
            ),
            pedestrian_reference_weight=_finite_float(
                raw_subject_exposure_raw.get("pedestrian_reference_weight", 0.7),
                f"{key}.raw_subject_exposure.pedestrian_reference_weight",
                minimum=0.0,
                maximum=1.0,
            ),
            min_pedestrian_reference_pixels=_int_setting(
                raw_subject_exposure_raw,
                "min_pedestrian_reference_pixels",
                48,
                minimum=1,
            ),
        ),
    )


def _apply_cli_lighting_overrides(spec: SceneSpec, args: Any) -> SceneSpec:
    if (
        args.lighting_preset is None
        and args.ambient_world_strength is None
        and args.shadow_cube_size is None
    ):
        return spec
    lighting = spec.lighting or _default_lighting_rig()
    preset = (
        str(args.lighting_preset).strip().lower()
        if args.lighting_preset is not None
        else lighting.preset
    )
    if preset not in _ALLOWED_LIGHTING_PRESETS:
        raise ValueError(
            "lighting-preset must be one of: " + ", ".join(_ALLOWED_LIGHTING_PRESETS)
        )
    ambient_world_strength = (
        _finite_float(args.ambient_world_strength, "ambient-world-strength", minimum=0.0)
        if args.ambient_world_strength is not None
        else lighting.ambient_world_strength
    )
    shadow_cube_size = (
        str(args.shadow_cube_size).strip()
        if args.shadow_cube_size is not None
        else lighting.shadow_cube_size
    )
    if shadow_cube_size not in _ALLOWED_SHADOW_CUBE_SIZES:
        raise ValueError(
            "shadow-cube-size must be one of: " + ", ".join(_ALLOWED_SHADOW_CUBE_SIZES)
        )
    overridden = _default_lighting_rig(
        ambient_world_strength=ambient_world_strength,
        shadow_cube_size=shadow_cube_size,
    )
    if lighting.lights and lighting.preset == preset:
        overridden = replace(
            overridden,
            enabled=lighting.enabled,
            wrap_subject_fill=lighting.wrap_subject_fill,
            lights=lighting.lights,
        )
    else:
        overridden = replace(
            overridden,
            enabled=lighting.enabled,
            wrap_subject_fill=lighting.wrap_subject_fill,
        )
    return replace(spec, lighting=replace(overridden, preset=preset))


def _scene_spec_from_profile(
    *,
    run_dir: Path,
    trajectory_path: Optional[Path],
    output_path: Optional[Path],
    config_path: Path,
    profile_name: str,
) -> SceneSpec:
    profile = _load_profile_config(config_path, profile_name)
    if trajectory_path is None:
        trajectory_path = run_dir / "standard" / "trajectory" / "poses.npz"
    if _raw_sampling_fps_from_profile(profile) is None:
        profile_snapshot_path = run_dir / "standard" / "profile.json"
        if profile_snapshot_path != config_path and profile_snapshot_path.exists():
            snapshot_profile = _load_profile_config(profile_snapshot_path, profile_name)
            if _raw_sampling_fps_from_profile(snapshot_profile) is not None:
                profile = snapshot_profile
    runtime = profile.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"Profile '{profile_name}' is missing a 'runtime' section.")
    runtime_settings = runtime.get("settings")
    if not isinstance(runtime_settings, dict):
        raise ValueError(
            f"Profile '{profile_name}' has invalid runtime.settings; expected object."
        )
    blender_settings = runtime_settings.get("blender_scene")
    if not isinstance(blender_settings, dict):
        raise ValueError(
            f"Profile '{profile_name}' is missing runtime.settings.blender_scene."
        )
    if not blender_settings.get("enabled", False):
        raise ValueError(
            f"Profile '{profile_name}' has blender_scene disabled; enable it to render."
        )

    cube_size = float(blender_settings.get("cube_size", 0.1))
    collection_name = str(blender_settings.get("collection_name", "TrajectoryDebug"))
    road_plane_gap = float(blender_settings.get("road_gap", 0.05))
    if road_plane_gap < 0:
        raise ValueError(f"Invalid road-gap: {road_plane_gap} (must be non-negative)")
    global_plane_range_m = _float_setting(
        blender_settings, "global_plane_range_m", 25.0, minimum=0.1
    )
    global_plane_min_range_m = _float_setting(
        blender_settings, "global_plane_min_range_m", 3.0, minimum=0.0
    )
    if global_plane_min_range_m >= global_plane_range_m:
        raise ValueError(
            "global_plane_min_range_m must be smaller than global_plane_range_m."
        )
    global_plane_frame_window = _int_setting(
        blender_settings, "global_plane_frame_window", 3, minimum=0
    )
    global_plane_max_points_per_frame = _int_setting(
        blender_settings, "global_plane_max_points_per_frame", 4000, minimum=32
    )
    global_plane_confidence_threshold = _float_setting(
        blender_settings, "global_plane_confidence_threshold", 0.5, minimum=0.0
    )
    if global_plane_confidence_threshold > 1.0:
        raise ValueError(
            "Invalid global_plane_confidence_threshold: "
            f"{global_plane_confidence_threshold} (must be <= 1.0)"
        )
    global_plane_trim_ratio = _float_setting(
        blender_settings, "global_plane_trim_ratio", 0.2, minimum=0.0
    )
    if global_plane_trim_ratio >= 1.0:
        raise ValueError(
            f"Invalid global_plane_trim_ratio: {global_plane_trim_ratio} (must be < 1.0)"
        )
    local_support_radius_m = _float_setting(
        blender_settings, "local_support_radius_m", 2.5, minimum=0.05
    )
    local_support_frame_window = _int_setting(
        blender_settings, "local_support_frame_window", 3, minimum=0
    )
    local_support_min_points = _int_setting(
        blender_settings, "local_support_min_points", 10, minimum=3
    )
    local_support_plane_size_m = _float_setting(
        blender_settings, "local_support_plane_size_m", 0.6, minimum=0.05
    )
    local_support_confidence_threshold = _float_setting(
        blender_settings, "local_support_confidence_threshold", 0.0, minimum=0.0
    )
    if local_support_confidence_threshold > 1.0:
        raise ValueError(
            "Invalid local_support_confidence_threshold: "
            f"{local_support_confidence_threshold} (must be <= 1.0)"
        )
    local_support_max_radius_m = _float_setting(
        blender_settings, "local_support_max_radius_m", 3.0, minimum=0.05
    )
    local_support_radius_step_m = _float_setting(
        blender_settings, "local_support_radius_step_m", 0.5, minimum=0.01
    )
    if local_support_max_radius_m < local_support_radius_m:
        raise ValueError(
            "local_support_max_radius_m must be >= local_support_radius_m."
        )
    local_support_snap_to_nearest_road = bool(
        blender_settings.get("local_support_snap_to_nearest_road", True)
    )
    local_support_snap_radius_m = _float_setting(
        blender_settings, "local_support_snap_radius_m", 4.0, minimum=0.05
    )
    local_support_temporal_hold_frames = _int_setting(
        blender_settings, "local_support_temporal_hold_frames", 6, minimum=0
    )
    local_support_temporal_hold_seconds_raw = blender_settings.get(
        "local_support_temporal_hold_seconds"
    )
    local_support_temporal_hold_seconds = (
        None
        if local_support_temporal_hold_seconds_raw is None
        else float(local_support_temporal_hold_seconds_raw)
    )
    if (
        local_support_temporal_hold_seconds is not None
        and (
            not np.isfinite(local_support_temporal_hold_seconds)
            or local_support_temporal_hold_seconds < 0.0
        )
    ):
        raise ValueError(
            "local_support_temporal_hold_seconds must be a finite value >= 0."
        )
    local_support_snap_max_vertical_delta_m = _float_setting(
        blender_settings, "local_support_snap_max_vertical_delta_m", 0.2, minimum=0.0
    )
    local_support_snap_max_radius_ratio = _float_setting(
        blender_settings, "local_support_snap_max_radius_ratio", 0.5, minimum=0.0
    )
    local_support_prefilter_vertical_window_m = _float_setting(
        blender_settings, "local_support_prefilter_vertical_window_m", 0.75, minimum=0.0
    )
    trajectory_grounding_transition_frames = _int_setting(
        blender_settings,
        "trajectory_grounding_transition_frames",
        4,
        minimum=1,
    )
    trajectory_grounding_max_step_m = _float_setting(
        blender_settings,
        "trajectory_grounding_max_step_m",
        0.05,
        minimum=0.0,
    )
    trajectory_grounding_max_vertical_velocity_mps = _float_setting(
        blender_settings,
        "trajectory_grounding_max_vertical_velocity_mps",
        0.9,
        minimum=0.0,
    )
    trajectory_grounding_max_vertical_accel_mps2 = _float_setting(
        blender_settings,
        "trajectory_grounding_max_vertical_accel_mps2",
        2.5,
        minimum=0.0,
    )
    lighting = _lighting_rig_from_mapping(
        blender_settings.get("lighting"), "runtime.settings.blender_scene.lighting"
    )
    render = _render_spec_from_mapping(
        blender_settings.get("render"), "runtime.settings.blender_scene.render"
    )
    shadow = _shadow_spec_from_mapping(
        blender_settings.get("shadow"), "runtime.settings.blender_scene.shadow"
    )
    placement_kwargs = _resolve_profile_pedestrian_placement(
        blender_settings=blender_settings,
        trajectory_path=trajectory_path,
        profile_name=profile_name,
        profile=profile,
    )
    pedestrian_trajectory_t = _float_setting(
        blender_settings,
        "pedestrian_trajectory_t",
        0.0,
        minimum=0.0,
        maximum=1.0,
    )
    pedestrian_forward_offset_m = _float_setting(blender_settings, "pedestrian_forward_offset_m", 5.0)
    pedestrian_left_offset_m = _float_setting(blender_settings, "pedestrian_left_offset_m", 2.0)
    pedestrian_up_offset_m = _float_setting(blender_settings, "pedestrian_up_offset_m", 0.0)
    pedestrian_motion_policy = str(
        blender_settings.get("pedestrian_motion_policy", "auto")
    ).strip().lower()
    if pedestrian_motion_policy not in {
        "auto",
        "stationary_at_spawn",
        "animation_root_motion",
        "camera_trajectory_relative",
    }:
        raise ValueError(
            "runtime.settings.blender_scene.pedestrian_motion_policy must be one of "
            "'auto', 'stationary_at_spawn', 'animation_root_motion', or "
            "'camera_trajectory_relative'."
        )
    max_plane_center_xy_distance_m = _float_setting(
        blender_settings,
        "max_plane_center_xy_distance_m",
        8.0,
        minimum=0.01,
    )
    pedestrian_heading_deg = _float_setting(
        blender_settings, "pedestrian_heading_deg", 0.0
    )
    road_labels = _road_labels_setting(run_dir, profile)
    occlusion_settings_raw = blender_settings.get("occlusion", {})
    if not isinstance(occlusion_settings_raw, dict):
        raise ValueError("runtime.settings.blender_scene.occlusion must be an object.")
    occlusion_depth_source = str(
        occlusion_settings_raw.get("depth_source", "z_pass")
    ).strip().lower()
    if occlusion_depth_source != "z_pass":
        raise ValueError(
            "runtime.settings.blender_scene.occlusion.depth_source must be 'z_pass'."
        )
    contact_ground_roles = _string_tuple_setting(
        occlusion_settings_raw.get("contact_ground_roles", ("road", "sidewalk")),
        "runtime.settings.blender_scene.occlusion.contact_ground_roles",
    )
    contact_ground_label_overrides = _string_tuple_setting(
        occlusion_settings_raw.get("contact_ground_labels", ()),
        "runtime.settings.blender_scene.occlusion.contact_ground_labels",
    )
    contact_ground_labels = _contact_ground_labels_setting(
        run_dir=run_dir,
        profile=profile,
        roles=contact_ground_roles or ("road", "sidewalk"),
        extra_labels=contact_ground_label_overrides,
    )
    edge_treatment_raw = occlusion_settings_raw.get("edge_treatment", {})
    if not isinstance(edge_treatment_raw, dict):
        raise ValueError(
            "runtime.settings.blender_scene.occlusion.edge_treatment must be an object."
        )
    edge_treatment = EdgeTreatmentSpec(
        enabled=_bool_setting(edge_treatment_raw, "enabled", True),
        boundary_band_px=_int_setting(
            edge_treatment_raw,
            "boundary_band_px",
            4,
            minimum=1,
        ),
        feather_radius_px=_float_setting(
            edge_treatment_raw,
            "feather_radius_px",
            2.0,
            minimum=0.0,
        ),
        feather_strength=_float_setting(
            edge_treatment_raw,
            "feather_strength",
            0.35,
            minimum=0.0,
            maximum=1.0,
        ),
        blur_enabled=_bool_setting(edge_treatment_raw, "blur_enabled", True),
        blur_radius_px=_float_setting(
            edge_treatment_raw,
            "blur_radius_px",
            1.5,
            minimum=0.0,
        ),
        blur_strength=_float_setting(
            edge_treatment_raw,
            "blur_strength",
            0.25,
            minimum=0.0,
            maximum=1.0,
        ),
        despill_enabled=_bool_setting(edge_treatment_raw, "despill_enabled", True),
        despill_strength=_float_setting(
            edge_treatment_raw,
            "despill_strength",
            0.25,
            minimum=0.0,
            maximum=1.0,
        ),
        regrain_enabled=_bool_setting(edge_treatment_raw, "regrain_enabled", True),
        regrain_strength=_float_setting(
            edge_treatment_raw,
            "regrain_strength",
            0.12,
            minimum=0.0,
            maximum=1.0,
        ),
        tiny_object_disable_feather=_bool_setting(
            edge_treatment_raw,
            "tiny_object_disable_feather",
            True,
        ),
        tiny_object_disable_blur=_bool_setting(
            edge_treatment_raw,
            "tiny_object_disable_blur",
            True,
        ),
        tiny_object_disable_despill=_bool_setting(
            edge_treatment_raw,
            "tiny_object_disable_despill",
            True,
        ),
        tiny_object_disable_regrain=_bool_setting(
            edge_treatment_raw,
            "tiny_object_disable_regrain",
            True,
        ),
        tiny_object_max_boundary_fraction=_float_setting(
            edge_treatment_raw,
            "tiny_object_max_boundary_fraction",
            0.25,
            minimum=0.0,
            maximum=1.0,
        ),
        tiny_object_disable_all_below_short_side_px=_int_setting(
            edge_treatment_raw,
            "tiny_object_disable_all_below_short_side_px",
            20,
            minimum=1,
        ),
        tiny_object_disable_all_below_visible_pixels=_int_setting(
            edge_treatment_raw,
            "tiny_object_disable_all_below_visible_pixels",
            256,
            minimum=1,
        ),
        disable_when_boundary_fraction_above=_float_setting(
            edge_treatment_raw,
            "disable_when_boundary_fraction_above",
            0.6,
            minimum=0.0,
            maximum=1.0,
        ),
    )
    temporal_stabilization_raw = occlusion_settings_raw.get("temporal_stabilization", {})
    if not isinstance(temporal_stabilization_raw, dict):
        raise ValueError(
            "runtime.settings.blender_scene.occlusion.temporal_stabilization must be an object."
        )
    occlusion = OcclusionSpec(
        depth_source="z_pass",
        contact_ground_labels=contact_ground_labels,
        default_front_margin_m=_float_setting(
            occlusion_settings_raw,
            "default_front_margin_m",
            0.03,
            minimum=0.0,
        ),
        relative_margin=_float_setting(
            occlusion_settings_raw,
            "relative_margin",
            0.01,
            minimum=0.0,
        ),
        contact_plane_band_m=_float_setting(
            occlusion_settings_raw,
            "contact_plane_band_m",
            0.025,
            minimum=0.0,
        ),
        contact_patch_radius_m=_float_setting(
            occlusion_settings_raw,
            "contact_patch_radius_m",
            0.30,
            minimum=0.01,
        ),
        contact_coplanar_tolerance_m=_float_setting(
            occlusion_settings_raw,
            "contact_coplanar_tolerance_m",
            0.03,
            minimum=0.0,
        ),
        write_debug=_bool_setting(
            occlusion_settings_raw,
            "write_debug",
            True,
        ),
        edge_treatment=edge_treatment,
        temporal_stabilization=TemporalOcclusionStabilizationSpec(
            enabled=_bool_setting(temporal_stabilization_raw, "enabled", True),
            base_hysteresis_margin_m=_float_setting(
                temporal_stabilization_raw,
                "base_hysteresis_margin_m",
                0.02,
                minimum=0.0,
            ),
            state_flip_persist_frames=_int_setting(
                temporal_stabilization_raw,
                "state_flip_persist_frames",
                2,
                minimum=1,
            ),
            edge_exit_hold_frames=_int_setting(
                temporal_stabilization_raw,
                "edge_exit_hold_frames",
                2,
                minimum=0,
            ),
            max_single_frame_visible_area_drop_ratio=_float_setting(
                temporal_stabilization_raw,
                "max_single_frame_visible_area_drop_ratio",
                0.5,
                minimum=0.0,
                maximum=1.0,
            ),
        ),
    )

    mixamo_settings = profile.get("mixamo", {})
    if not isinstance(mixamo_settings, dict):
        raise ValueError(
            f"Profile '{profile_name}' has invalid mixamo settings; expected object."
        )
    character_path = mixamo_settings.get("character_fbx_path")
    animation_path = mixamo_settings.get("animation_fbx_path")
    asset_root_path = mixamo_settings.get("asset_root")
    if not character_path or not animation_path:
        raise ValueError(
            f"Profile '{profile_name}' is missing mixamo character/animation paths."
        )

    frame_provider = profile.get("frame_provider", {})
    if not isinstance(frame_provider, dict):
        raise ValueError(
            f"Profile '{profile_name}' has invalid frame_provider; expected object."
        )
    frame_settings = frame_provider.get("settings", {})
    if not isinstance(frame_settings, dict):
        raise ValueError(
            f"Profile '{profile_name}' has invalid frame_provider.settings; expected object."
        )

    unity_import = profile.get("unity_import", {})
    if unity_import is not None and not isinstance(unity_import, dict):
        raise ValueError(
            f"Profile '{profile_name}' has invalid unity_import settings; expected object."
        )

    sampling_fps = _raw_sampling_fps_from_profile(profile)
    if sampling_fps is None:
        raise ValueError(
            f"Profile '{profile_name}' is missing a resolved sampling_fps (frame_provider.settings or unity_import)."
        )
    try:
        mixamo_scene_fps = float(sampling_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "frame_provider.settings sampling_fps must resolve to a numeric value."
        ) from exc
    if mixamo_scene_fps <= 0:
        raise ValueError("frame_provider.settings sampling_fps must resolve to > 0.")

    mixamo_package = resolve_mixamo_asset_package(
        character_fbx=_resolve_repo_path(Path(character_path).expanduser()),
        animation_fbx=_resolve_repo_path(Path(animation_path).expanduser()),
        asset_root=(
            _resolve_repo_path(Path(asset_root_path).expanduser())
            if asset_root_path
            else None
        ),
    )
    pedestrian_actor_name = str(
        mixamo_settings.get("actor_name", "Pedestrian01")
    ).strip()
    if not pedestrian_actor_name:
        raise ValueError("mixamo.actor_name must be a non-empty string.")
    mixamo_source_fps = float(mixamo_settings.get("source_fps", 30.0))
    if not np.isfinite(mixamo_source_fps) or mixamo_source_fps <= 0.0:
        raise ValueError("mixamo.source_fps must be a finite value > 0.")
    mixamo_export_fps = float(mixamo_settings.get("export_fps", 30.0))
    if not np.isfinite(mixamo_export_fps) or mixamo_export_fps <= 0.0:
        raise ValueError("mixamo.export_fps must be a finite value > 0.")
    mixamo_debug = bool(mixamo_settings.get("debug", True))

    return SceneSpec(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        output_path=output_path,
        cube_size=cube_size,
        collection_name=collection_name,
        road_plane_gap=road_plane_gap,
        mixamo_character_fbx_path=mixamo_package.character_fbx,
        mixamo_animation_fbx_path=mixamo_package.animation_fbx,
        mixamo_asset_root=mixamo_package.asset_root,
        pedestrian_actor_name=pedestrian_actor_name,
        pedestrian_trajectory_t=pedestrian_trajectory_t,
        pedestrian_forward_offset_m=pedestrian_forward_offset_m,
        pedestrian_left_offset_m=pedestrian_left_offset_m,
        pedestrian_up_offset_m=pedestrian_up_offset_m,
        pedestrian_heading_deg=pedestrian_heading_deg,
        pedestrian_motion_policy=pedestrian_motion_policy,
        max_plane_center_xy_distance_m=max_plane_center_xy_distance_m,
        mixamo_scene_fps=mixamo_scene_fps,
        mixamo_export_fps=mixamo_export_fps,
        mixamo_source_fps=mixamo_source_fps,
        mixamo_debug=mixamo_debug,
        sampling_fps=mixamo_scene_fps,
        global_plane_range_m=global_plane_range_m,
        global_plane_min_range_m=global_plane_min_range_m,
        global_plane_frame_window=global_plane_frame_window,
        global_plane_max_points_per_frame=global_plane_max_points_per_frame,
        global_plane_confidence_threshold=global_plane_confidence_threshold,
        global_plane_trim_ratio=global_plane_trim_ratio,
        road_labels=road_labels,
        local_support_radius_m=local_support_radius_m,
        local_support_frame_window=local_support_frame_window,
        local_support_min_points=local_support_min_points,
        local_support_plane_size_m=local_support_plane_size_m,
        local_support_confidence_threshold=local_support_confidence_threshold,
        local_support_max_radius_m=local_support_max_radius_m,
        local_support_radius_step_m=local_support_radius_step_m,
        local_support_snap_to_nearest_road=local_support_snap_to_nearest_road,
        local_support_snap_radius_m=local_support_snap_radius_m,
        local_support_temporal_hold_frames=local_support_temporal_hold_frames,
        local_support_temporal_hold_seconds=local_support_temporal_hold_seconds,
        local_support_snap_max_vertical_delta_m=local_support_snap_max_vertical_delta_m,
        local_support_snap_max_radius_ratio=local_support_snap_max_radius_ratio,
        local_support_prefilter_vertical_window_m=local_support_prefilter_vertical_window_m,
        trajectory_grounding_transition_frames=trajectory_grounding_transition_frames,
        trajectory_grounding_max_step_m=trajectory_grounding_max_step_m,
        trajectory_grounding_max_vertical_velocity_mps=(
            trajectory_grounding_max_vertical_velocity_mps
        ),
        trajectory_grounding_max_vertical_accel_mps2=(
            trajectory_grounding_max_vertical_accel_mps2
        ),
        render=render,
        lighting=lighting,
        shadow=shadow,
        occlusion=occlusion,
        **placement_kwargs,
    )


def parse_args(argv: list[str]) -> SceneSpec:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize PEMOIN trajectory in Blender."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="outputs/<run> directory")
    parser.add_argument("--trajectory", type=Path, help="Path to trajectory/poses.npz (optional)")
    parser.add_argument("--output", type=Path, help="Output .blend file path")
    parser.add_argument("--host-python", type=Path, help="Host PEMOIN Python executable for EXR depth decode")
    parser.add_argument("--config", type=Path, help="Profile JSON path")
    parser.add_argument("--profile", type=str, help="Profile name defined in the config file")
    parser.add_argument("--cube-size", type=float, default=0.1, help="Cube size in meters")
    parser.add_argument("--collection", type=str, default="TrajectoryDebug", help="Collection name")
    parser.add_argument("--road-gap", type=float, default=0.05, help="Gap between consecutive planes in meters")
    parser.add_argument("--global-plane-range-m", type=float, default=25.0)
    parser.add_argument("--global-plane-min-range-m", type=float, default=3.0)
    parser.add_argument("--global-plane-frame-window", type=int, default=3)
    parser.add_argument("--global-plane-max-points-per-frame", type=int, default=4000)
    parser.add_argument("--global-plane-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--global-plane-trim-ratio", type=float, default=0.2)
    parser.add_argument("--local-support-radius-m", type=float, default=2.5)
    parser.add_argument("--local-support-frame-window", type=int, default=3)
    parser.add_argument("--local-support-min-points", type=int, default=10)
    parser.add_argument("--local-support-plane-size-m", type=float, default=0.6)
    parser.add_argument("--local-support-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--local-support-max-radius-m", type=float, default=3.0)
    parser.add_argument("--local-support-radius-step-m", type=float, default=0.5)
    parser.add_argument(
        "--local-support-snap-to-nearest-road",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local-support-snap-radius-m", type=float, default=4.0)
    parser.add_argument("--local-support-temporal-hold-frames", type=int, default=6)
    parser.add_argument("--local-support-temporal-hold-seconds", type=float, default=None)
    parser.add_argument("--local-support-snap-max-vertical-delta-m", type=float, default=0.2)
    parser.add_argument("--local-support-snap-max-radius-ratio", type=float, default=0.5)
    parser.add_argument("--local-support-prefilter-vertical-window-m", type=float, default=0.75)
    parser.add_argument(
        "--foot-contact-mode",
        type=str,
        default="mixamo_phase",
        choices=("nearest_plane", "mixamo_phase"),
        help="Foot contact planning mode",
    )
    parser.add_argument("--foot-contact-phase-offset", type=float, default=0.0)
    parser.add_argument("--foot-contact-gait-cycle-frames", type=float)
    parser.add_argument("--foot-contact-left-stance-phase-ranges", type=str, default="")
    parser.add_argument("--foot-contact-right-stance-phase-ranges", type=str, default="")
    parser.add_argument(
        "--foot-contact-auto-calibrate-phase-ranges",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--foot-contact-plane-mode",
        type=str,
        default="project",
        choices=("strict", "project", "off"),
    )
    parser.add_argument(
        "--foot-contact-min-plane-confidence-for-projection",
        type=float,
        default=0.35,
    )
    parser.add_argument("--foot-contact-max-plane-dist-m", type=float, default=0.08)
    parser.add_argument("--foot-contact-max-speed-mps", type=float, default=1.8)
    parser.add_argument("--foot-contact-min-stance-frames", type=int, default=2)
    parser.add_argument("--foot-contact-min-swing-frames", type=int, default=2)
    parser.add_argument("--mixamo-character-fbx", type=Path)
    parser.add_argument("--mixamo-animation-fbx", type=Path)
    parser.add_argument("--mixamo-scene-fps", type=float)
    parser.add_argument("--mixamo-source-fps", type=float, default=30.0)
    parser.add_argument("--actor-name", type=str, default="Pedestrian01")
    parser.add_argument("--pedestrian-trajectory-t", type=float, default=0.0)
    parser.add_argument("--pedestrian-forward-offset-m", type=float, default=5.0)
    parser.add_argument("--pedestrian-left-offset-m", type=float, default=2.0)
    parser.add_argument("--pedestrian-up-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--pedestrian-motion-policy",
        type=str,
        default="auto",
        choices=(
            "auto",
            "stationary_at_spawn",
            "animation_root_motion",
            "camera_trajectory_relative",
        ),
    )
    parser.add_argument("--max-plane-center-xy-distance-m", type=float, default=None)
    parser.add_argument("--pedestrian-heading-deg", type=float, default=0.0)
    parser.add_argument(
        "--mixamo-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lighting-preset", type=str, choices=_ALLOWED_LIGHTING_PRESETS)
    parser.add_argument("--ambient-world-strength", type=float)
    parser.add_argument("--shadow-cube-size", type=str, choices=_ALLOWED_SHADOW_CUBE_SIZES)
    parser.add_argument("--render-engine", type=str, choices=_ALLOWED_RENDER_ENGINES)
    parser.add_argument("--render-resolution-scale", type=float, default=1.0)
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument(
        "--shadow-catcher-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shadow-receiver-patch-size-m", type=float, default=4.0)
    parser.add_argument("--shadow-map-resolution", type=str, choices=_ALLOWED_SHADOW_CUBE_SIZES)
    parser.add_argument("--shadow-softness", type=float, default=1.5)
    parser.add_argument("--shadow-opacity", type=float, default=1.0)
    parser.add_argument(
        "--shadow-tint-rgb",
        type=str,
        default="0.0,0.0,0.0",
        help="Comma-separated shadow tint RGB in [0,1].",
    )

    args = parser.parse_args(argv)

    if args.road_gap < 0:
        raise ValueError(f"Invalid road-gap: {args.road_gap} (must be non-negative)")
    if args.global_plane_min_range_m < 0 or args.global_plane_range_m <= 0:
        raise ValueError("Global plane ranges must be positive and non-negative.")
    if args.global_plane_min_range_m >= args.global_plane_range_m:
        raise ValueError("global-plane-min-range-m must be smaller than global-plane-range-m.")
    if args.global_plane_frame_window < 0:
        raise ValueError("global-plane-frame-window must be >= 0.")
    if args.global_plane_max_points_per_frame < 32:
        raise ValueError("global-plane-max-points-per-frame must be >= 32.")
    if not (0.0 <= args.global_plane_confidence_threshold <= 1.0):
        raise ValueError("global-plane-confidence-threshold must be in [0, 1].")
    if not (0.0 <= args.global_plane_trim_ratio < 1.0):
        raise ValueError("global-plane-trim-ratio must be in [0, 1).")
    if args.local_support_radius_m <= 0 or args.local_support_plane_size_m <= 0:
        raise ValueError("local support radius/plane size must be positive.")
    if args.local_support_max_radius_m < args.local_support_radius_m:
        raise ValueError("local-support-max-radius-m must be >= local-support-radius-m.")
    if args.local_support_radius_step_m <= 0:
        raise ValueError("local-support-radius-step-m must be > 0.")
    if args.local_support_frame_window < 0:
        raise ValueError("local-support-frame-window must be >= 0.")
    if args.local_support_min_points < 3:
        raise ValueError("local-support-min-points must be >= 3.")
    if not (0.0 <= args.local_support_confidence_threshold <= 1.0):
        raise ValueError("local-support-confidence-threshold must be in [0, 1].")
    if args.local_support_snap_radius_m <= 0:
        raise ValueError("local-support-snap-radius-m must be > 0.")
    if args.local_support_temporal_hold_frames < 0:
        raise ValueError("local-support-temporal-hold-frames must be >= 0.")
    if (
        args.local_support_temporal_hold_seconds is not None
        and (
            not np.isfinite(float(args.local_support_temporal_hold_seconds))
            or float(args.local_support_temporal_hold_seconds) < 0.0
        )
    ):
        raise ValueError("local-support-temporal-hold-seconds must be >= 0.")
    if args.local_support_snap_max_vertical_delta_m < 0:
        raise ValueError("local-support-snap-max-vertical-delta-m must be >= 0.")
    if args.local_support_snap_max_radius_ratio < 0:
        raise ValueError("local-support-snap-max-radius-ratio must be >= 0.")
    if args.local_support_prefilter_vertical_window_m < 0:
        raise ValueError("local-support-prefilter-vertical-window-m must be >= 0.")
    if args.foot_contact_gait_cycle_frames is not None and args.foot_contact_gait_cycle_frames <= 1:
        raise ValueError("foot-contact-gait-cycle-frames must be > 1.")
    if not (0.0 <= args.foot_contact_min_plane_confidence_for_projection <= 1.0):
        raise ValueError("foot-contact-min-plane-confidence-for-projection must be in [0, 1].")
    if args.foot_contact_max_plane_dist_m < 0:
        raise ValueError("foot-contact-max-plane-dist-m must be >= 0.")
    if args.foot_contact_max_speed_mps < 0:
        raise ValueError("foot-contact-max-speed-mps must be >= 0.")
    if args.foot_contact_min_stance_frames < 1:
        raise ValueError("foot-contact-min-stance-frames must be >= 1.")
    if args.foot_contact_min_swing_frames < 1:
        raise ValueError("foot-contact-min-swing-frames must be >= 1.")
    if args.shadow_receiver_patch_size_m <= 0:
        raise ValueError("shadow-receiver-patch-size-m must be > 0.")
    if args.shadow_map_resolution is not None and str(args.shadow_map_resolution).strip() not in _ALLOWED_SHADOW_CUBE_SIZES:
        raise ValueError("shadow-map-resolution must be one of: " + ", ".join(_ALLOWED_SHADOW_CUBE_SIZES))
    if float(args.shadow_softness) < 0.0:
        raise ValueError("shadow-softness must be >= 0.")
    if not (0.0 <= float(args.shadow_opacity) <= 1.0):
        raise ValueError("shadow-opacity must be in [0, 1].")
    if args.render_engine is not None and str(args.render_engine).strip().lower() not in _ALLOWED_RENDER_ENGINES:
        raise ValueError("render-engine must be one of: " + ", ".join(_ALLOWED_RENDER_ENGINES))
    if not (0.1 <= float(args.render_resolution_scale) <= 4.0):
        raise ValueError("render-resolution-scale must be in [0.1, 4.0].")
    if int(args.render_samples) < 1:
        raise ValueError("render-samples must be >= 1.")
    actor_name = str(args.actor_name).strip()
    if not actor_name:
        raise ValueError("actor-name must be a non-empty string.")
    pedestrian_trajectory_t = float(args.pedestrian_trajectory_t)
    if not np.isfinite(pedestrian_trajectory_t) or not (0.0 <= pedestrian_trajectory_t <= 1.0):
        raise ValueError("pedestrian-trajectory-t must be in [0, 1].")
    pedestrian_forward_offset_m = float(args.pedestrian_forward_offset_m)
    pedestrian_left_offset_m = float(args.pedestrian_left_offset_m)
    pedestrian_up_offset_m = float(args.pedestrian_up_offset_m)
    max_plane_center_xy_distance_m = (
        8.0 if args.max_plane_center_xy_distance_m is None else float(args.max_plane_center_xy_distance_m)
    )
    if not np.isfinite(np.asarray([pedestrian_forward_offset_m, pedestrian_left_offset_m, pedestrian_up_offset_m], dtype=np.float32)).all():
        raise ValueError("Pedestrian offsets must be finite values.")
    pedestrian_motion_policy = str(args.pedestrian_motion_policy).strip().lower()
    if not np.isfinite(max_plane_center_xy_distance_m) or max_plane_center_xy_distance_m <= 0.0:
        raise ValueError("max-plane-center-xy-distance-m must be a finite value > 0.")
    pedestrian_heading_deg = float(args.pedestrian_heading_deg)

    run_dir = args.run_dir.expanduser().resolve()
    road_labels = _road_labels_setting(run_dir)
    trajectory_path = args.trajectory.expanduser().resolve() if args.trajectory else None
    output_path = args.output.expanduser().resolve() if args.output else None
    host_python = args.host_python.expanduser().resolve() if args.host_python else None

    if args.config or args.profile:
        if args.config is None or args.profile is None:
            raise ValueError("Both --config and --profile are required when using profile config.")
        config_path = args.config.expanduser().resolve()
        spec = _scene_spec_from_profile(
            run_dir=run_dir,
            trajectory_path=trajectory_path,
            output_path=output_path,
            config_path=config_path,
            profile_name=args.profile,
        )
        spec = _apply_cli_lighting_overrides(spec, args)
        if args.max_plane_center_xy_distance_m is not None:
            spec = replace(spec, max_plane_center_xy_distance_m=max_plane_center_xy_distance_m)
        if host_python is not None:
            spec = replace(spec, host_python=host_python)
        return spec

    if trajectory_path is None:
        trajectory_path = run_dir / "standard" / "trajectory" / "poses.npz"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")

    mixamo_character_fbx_path = args.mixamo_character_fbx.expanduser().resolve() if args.mixamo_character_fbx else None
    mixamo_animation_fbx_path = args.mixamo_animation_fbx.expanduser().resolve() if args.mixamo_animation_fbx else None
    mixamo_scene_fps = float(args.mixamo_scene_fps) if args.mixamo_scene_fps is not None else None
    if mixamo_character_fbx_path is None or mixamo_animation_fbx_path is None or mixamo_scene_fps is None:
        raise ValueError(
            "Mixamo settings are required: --mixamo-character-fbx, --mixamo-animation-fbx, --mixamo-scene-fps."
        )
    if mixamo_scene_fps <= 0:
        raise ValueError(f"Invalid mixamo scene FPS: {mixamo_scene_fps}")
    if not np.isfinite(float(args.mixamo_source_fps)) or float(args.mixamo_source_fps) <= 0:
        raise ValueError(f"Invalid mixamo source FPS: {args.mixamo_source_fps}")
    mixamo_export_fps = 30.0
    mixamo_package = resolve_mixamo_asset_package(
        character_fbx=mixamo_character_fbx_path,
        animation_fbx=mixamo_animation_fbx_path,
    )
    lighting = _lighting_rig_from_cli_args(args)
    render = RenderSpec(
        engine=(
            str(args.render_engine).strip().lower()
            if args.render_engine is not None
            else "raster"
        ),  # type: ignore[arg-type]
        resolution_scale=float(args.render_resolution_scale),
        samples=int(args.render_samples),
        material_policy="preserve_base_alpha_normal",
        dynamic_light_binding="copy_location_constraint",
        salience_adaptive=SalienceAdaptiveRenderSpec(),
    )
    shadow = ShadowSpec(
        enabled=bool(args.shadow_catcher_enabled),
        receiver_patch_size_m=float(args.shadow_receiver_patch_size_m),
        map_resolution=(
            str(args.shadow_map_resolution).strip()
            if args.shadow_map_resolution is not None
            else "1024"
        ),  # type: ignore[arg-type]
        softness=float(args.shadow_softness),
        opacity=float(args.shadow_opacity),
        tint_rgb=_vec3_setting(
            {"shadow_tint_rgb": args.shadow_tint_rgb},
            "shadow_tint_rgb",
            (0.0, 0.0, 0.0),
        ),
    )

    return SceneSpec(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        output_path=output_path,
        host_python=host_python,
        cube_size=args.cube_size,
        collection_name=args.collection,
        road_plane_gap=args.road_gap,
        mixamo_character_fbx_path=mixamo_package.character_fbx,
        mixamo_animation_fbx_path=mixamo_package.animation_fbx,
        mixamo_asset_root=mixamo_package.asset_root,
        pedestrian_actor_name=actor_name,
        pedestrian_trajectory_t=pedestrian_trajectory_t,
        pedestrian_forward_offset_m=pedestrian_forward_offset_m,
        pedestrian_left_offset_m=pedestrian_left_offset_m,
        pedestrian_up_offset_m=pedestrian_up_offset_m,
        pedestrian_heading_deg=pedestrian_heading_deg,
        pedestrian_motion_policy=pedestrian_motion_policy,
        max_plane_center_xy_distance_m=max_plane_center_xy_distance_m,
        mixamo_scene_fps=mixamo_scene_fps,
        mixamo_export_fps=mixamo_export_fps,
        mixamo_source_fps=float(args.mixamo_source_fps),
        mixamo_debug=bool(args.mixamo_debug),
        sampling_fps=mixamo_scene_fps,
        global_plane_range_m=float(args.global_plane_range_m),
        global_plane_min_range_m=float(args.global_plane_min_range_m),
        global_plane_frame_window=int(args.global_plane_frame_window),
        global_plane_max_points_per_frame=int(args.global_plane_max_points_per_frame),
        global_plane_confidence_threshold=float(args.global_plane_confidence_threshold),
        global_plane_trim_ratio=float(args.global_plane_trim_ratio),
        road_labels=road_labels,
        local_support_radius_m=float(args.local_support_radius_m),
        local_support_frame_window=int(args.local_support_frame_window),
        local_support_min_points=int(args.local_support_min_points),
        local_support_plane_size_m=float(args.local_support_plane_size_m),
        local_support_confidence_threshold=float(args.local_support_confidence_threshold),
        local_support_max_radius_m=float(args.local_support_max_radius_m),
        local_support_radius_step_m=float(args.local_support_radius_step_m),
        local_support_snap_to_nearest_road=bool(args.local_support_snap_to_nearest_road),
        local_support_snap_radius_m=float(args.local_support_snap_radius_m),
        local_support_temporal_hold_frames=int(args.local_support_temporal_hold_frames),
        local_support_temporal_hold_seconds=(
            None
            if args.local_support_temporal_hold_seconds is None
            else float(args.local_support_temporal_hold_seconds)
        ),
        local_support_snap_max_vertical_delta_m=float(args.local_support_snap_max_vertical_delta_m),
        local_support_snap_max_radius_ratio=float(args.local_support_snap_max_radius_ratio),
        local_support_prefilter_vertical_window_m=float(args.local_support_prefilter_vertical_window_m),
        trajectory_grounding_transition_frames=int(
            getattr(args, "trajectory_grounding_transition_frames", 4)
        ),
        trajectory_grounding_max_step_m=float(
            getattr(args, "trajectory_grounding_max_step_m", 0.05)
        ),
        trajectory_grounding_max_vertical_velocity_mps=float(
            getattr(args, "trajectory_grounding_max_vertical_velocity_mps", 0.9)
        ),
        trajectory_grounding_max_vertical_accel_mps2=float(
            getattr(args, "trajectory_grounding_max_vertical_accel_mps2", 2.5)
        ),
        render=render,
        lighting=lighting,
        shadow=shadow,
    )
