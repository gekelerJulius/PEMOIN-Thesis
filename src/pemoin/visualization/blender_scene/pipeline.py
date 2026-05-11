from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping, Optional
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Literal, Sequence
from contextlib import contextmanager, suppress

_REPO_SRC = Path(__file__).resolve().parents[2]
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import numpy as np
from mathutils import Matrix, Vector
from numpy import ndarray
from pemoin.data.contracts import LightingData, ResourceStore, lighting_from_payload
from pemoin.providers.semantic_roles import resolve_semantic_role_labels
from pemoin.geometry.camera_model import (
    backproject_uv_depth_to_camera,
    camera_to_world,
    project_world_to_image,
)
from pemoin.utils.animation_timing import (
    compute_clip_duration_seconds,
    compute_cycle_duration_seconds,
    compute_export_frame_count,
    resolve_looped_source_timing,
)
from pemoin.utils.camera_calibration import (
    BlenderCameraSolution,
    solve_blender_camera_for_intrinsics,
    validate_and_normalize_intrinsics,
)
from pemoin.utils.resolution import _resize_array
from pemoin.visualization.pedestrian_placement import (
    build_heading_aligned_root_motion_path_world,
    classify_locomotion_from_world_deltas,
    detect_mixamo_animation_motion_category,
    minimum_xy_distance_to_trajectory,
    resolve_motion_aligned_actor_yaw_deg,
    resolve_mixamo_animation_motion_category,
    resolve_mixamo_motion_policy_from_animation_path,
    resolve_pedestrian_spawn_world,
    resolve_dominant_horizontal_direction,
    sample_pedestrian_spawn_path_world,
    stationary_pedestrian_spawn_path_world,
    standard_mixamo_forward_world_xy,
    validate_pedestrian_spawn_near_trajectory,
)
from pemoin.visualization.ground_grid import (
    composite_grid_with_mask,
    render_plane_grid_layer,
)
from pemoin.visualization.overlay_compositor import (
    compose_overlay_frame_with_occlusion,
)
from pemoin.visualization.overlay_occlusion import (
    EdgeTreatmentSettings,
    OcclusionFrameDiagnostics,
    OcclusionSettings,
    TemporalOcclusionSettings,
    TemporalOcclusionState,
    write_occlusion_diagnostics,
)
from pemoin.visualization.blender_scene.grounding_policy import (
    compute_support_relock_metrics,
    resolve_effective_hold_frames,
)
from pemoin.visualization.blender_scene.mixamo_assets import (
    build_mixamo_texture_index,
    resolve_mixamo_asset_package,
)
from pemoin.visualization.blender_scene.config import parse_args as config_parse_args
from pemoin.visualization.blender_scene.constants import (
    _ALLOWED_LIGHTING_PRESETS,
    _ALLOWED_SHADOW_CUBE_SIZES,
    _LIGHTING_PRESET_NEUTRAL_HEMISPHERE,
    _LOCAL_SUPPORT_MAX_PRE_SUPPORT_DIST_M,
    _LOCAL_SUPPORT_MAX_RESIDUAL_P90_M,
    _LOCAL_SUPPORT_MIN_INLIER_RATIO,
    _OVERLAY_ALPHA_THRESHOLD,
    _OVERLAY_FAIL_MAX_FLAGGED_RATIO,
    _OVERLAY_FAIL_MEDIAN_SUPPORT_TO_CONTACT_FOOT_PX,
    _OVERLAY_FAIL_MIN_ROAD_FRACTION,
    _OVERLAY_FAIL_P90_SUPPORT_TO_CONTACT_FOOT_PX,
    _OVERLAY_FAIL_SINGLE_SUPPORT_TO_CONTACT_FOOT_PX,
    _OVERLAY_FAIL_SOFT_SUPPORT_TO_CONTACT_FOOT_PX,
    _OVERLAY_SUPPORT_LOCAL_GRID_EXTENT_M,
    _OVERLAY_SUPPORT_LOCAL_GRID_LINE_COLOR_BGR,
    _OVERLAY_SUPPORT_LOCAL_GRID_LINE_THICKNESS,
    _OVERLAY_SUPPORT_LOCAL_GRID_SPACING_M,
    _OVERLAY_SUPPORT_PATCH_RADIUS_PX,
    _OVERLAY_WARN_SUPPORT_TO_SILHOUETTE_PX,
    _PEMOIN_LIGHTING_TAG,
    _PEMOIN_LIGHT_PLACEMENT_MODE,
    _PEMOIN_LIGHT_PLACEMENT_TARGET,
    _PEMOIN_LIGHT_REALIZED_ENERGY,
    _PEMOIN_LIGHT_RELATIVE_OFFSET,
    _PEMOIN_LIGHT_SOURCE_ENERGY,
    _PEMOIN_LIGHT_TRANSPORT_MODE,
    _PERSISTED_BLEND_MAX_DISAGREEMENT_M,
    _PERSISTED_BLEND_MIN_CONFIDENCE_FOR_PROJECTION,
    _SOLE_OFFSET_M,
    _SUPPORT_MAX_ANCHOR_SHIFT_M,
    _SUPPORT_MAX_HEIGHT_JUMP_M,
    _SUPPORT_MAX_NORMAL_JUMP_DEG,
    _WRAP_SUBJECT_FILL_TRANSPORT_MODE,
    _WRAP_SUBJECT_POINT_MIN_DISTANCE_M,
)
from pemoin.visualization.blender_scene.logging import (
    LOGGER,
    log_error,
    log_info,
    log_scope,
    log_warning,
    log_warning_big,
    progress_begin,
    progress_end,
    progress_message,
    progress_step,
)
from pemoin.visualization.blender_scene.specs import (
    ActorSupportContract,
    ContactFrameState,
    ContactSegment,
    GroundingDiagnostic,
    LightSpec,
    LightingRigSpec,
    MixamoSpec,
    OverlayValidationDiagnostic,
    PersistedPlaneLocalityDecision,
    RenderVisibilityFrame,
    RoadPlaneSpec,
    RoadSurfacePipelineResult,
    SceneSpec,
    SupportAnchorHeightFilterResult,
    SupportAnchorSelection,
    SupportSurfaceResolution,
    TrajectorySpec,
    WrapSubjectFillSpec,
)
from pemoin.validation.policy import AdaptiveValidationContext, ValidationPolicySettings

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run inside Blender with bpy available."
    ) from exc

try:
    from PIL import Image
except Exception:  # pragma: no cover - Blender may not ship Pillow
    Image = None

Vec3 = tuple[float, float, float]

_SUPPORT_FAILURE_REASON_CONFIDENCE = "support_confidence_too_low_for_projection"
_SUPPORT_FAILURE_REASON_RELOCK = "support_relock_rejected"


def clear_scene() -> None:
    """Clear entire Blender scene to factory defaults (Blender 5+)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def ensure_collection(name: str) -> bpy.types.Collection:
    """Create or retrieve a collection by name."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def save_blend(filepath: Path) -> None:
    """Save Blender scene to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath))


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / path).resolve()


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
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
        ):
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


def _road_labels_setting(run_dir: Path, profile: dict | None = None) -> tuple[str, ...]:
    semantics_tool = None
    if isinstance(profile, dict):
        providers = profile.get("providers", {})
        if isinstance(providers, dict):
            semantics = providers.get("semantics", {})
            if isinstance(semantics, dict) and semantics.get("tool") is not None:
                semantics_tool = str(semantics.get("tool"))
    labels = resolve_semantic_role_labels(
        "road",
        metadata=_first_semantics_metadata(run_dir),
        tool=semantics_tool,
        required=True,
        source_name="Blender scene generation",
    )
    if not labels:
        raise ValueError("Canonical semantic role 'road' could not be resolved for Blender scene generation.")
    return labels


def _bool_setting(settings: dict, key: str, default: bool) -> bool:
    value = settings.get(key, default)
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
        LightSpec(
            name="TopFrontLeft",
            kind="SUN",
            energy=2.5,
            rotation_euler_deg=(50.0, 35.0, 0.0),
            color=(1.0, 0.98, 0.96),
            angle_deg=55.0,
        ),
        LightSpec(
            name="TopFrontRight",
            kind="SUN",
            energy=2.5,
            rotation_euler_deg=(50.0, -35.0, 0.0),
            color=(1.0, 0.98, 0.96),
            angle_deg=55.0,
        ),
        LightSpec(
            name="TopBackLeft",
            kind="SUN",
            energy=2.0,
            rotation_euler_deg=(50.0, 145.0, 0.0),
            color=(0.98, 0.98, 1.0),
            angle_deg=55.0,
        ),
        LightSpec(
            name="TopBackRight",
            kind="SUN",
            energy=2.0,
            rotation_euler_deg=(50.0, -145.0, 0.0),
            color=(0.98, 0.98, 1.0),
            angle_deg=55.0,
        ),
        LightSpec(
            name="SideLeft",
            kind="SUN",
            energy=1.4,
            rotation_euler_deg=(80.0, 90.0, 0.0),
            color=(1.0, 1.0, 1.0),
            angle_deg=65.0,
        ),
        LightSpec(
            name="SideRight",
            kind="SUN",
            energy=1.4,
            rotation_euler_deg=(80.0, -90.0, 0.0),
            color=(1.0, 1.0, 1.0),
            angle_deg=65.0,
        ),
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
    rotation = _finite_vec3(settings.get("rotation_euler_deg"), f"{key_prefix}.rotation_euler_deg")
    color = _finite_vec3(settings.get("color"), f"{key_prefix}.color", minimum=0.0, maximum=1.0)
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
            raise ValueError(f"Invalid {key_prefix}.area_size: must be > 0.")
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
                raise ValueError(
                    f"Invalid lighting.lights[{idx}]: expected object."
                )
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
        _finite_float(
            args.ambient_world_strength,
            "ambient-world-strength",
            minimum=0.0,
        )
        if args.ambient_world_strength is not None
        else 0.12
    )
    shadow_cube_size = (
        str(args.shadow_cube_size).strip() if args.shadow_cube_size is not None else "2048"
    )
    if shadow_cube_size not in _ALLOWED_SHADOW_CUBE_SIZES:
        raise ValueError(
            "shadow-cube-size must be one of: "
            + ", ".join(_ALLOWED_SHADOW_CUBE_SIZES)
        )
    preset = (
        str(args.lighting_preset).strip().lower()
        if args.lighting_preset is not None
        else _LIGHTING_PRESET_NEUTRAL_HEMISPHERE
    )
    if preset not in _ALLOWED_LIGHTING_PRESETS:
        raise ValueError(
            "lighting-preset must be one of: "
            + ", ".join(_ALLOWED_LIGHTING_PRESETS)
        )
    return replace(
        _default_lighting_rig(
            ambient_world_strength=ambient_world_strength,
            shadow_cube_size=shadow_cube_size,
        ),
        preset=preset,
    )


def _apply_cli_lighting_overrides(spec: SceneSpec, args: Any) -> SceneSpec:
    if (
        args.lighting_preset is None
        and args.ambient_world_strength is None
        and args.shadow_cube_size is None
    ):
        return spec
    lighting = spec.lighting or _default_lighting_rig()
    preset = str(args.lighting_preset).strip().lower() if args.lighting_preset is not None else lighting.preset
    if preset not in _ALLOWED_LIGHTING_PRESETS:
        raise ValueError(
            "lighting-preset must be one of: "
            + ", ".join(_ALLOWED_LIGHTING_PRESETS)
        )
    ambient_world_strength = (
        _finite_float(
            args.ambient_world_strength,
            "ambient-world-strength",
            minimum=0.0,
        )
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
            "shadow-cube-size must be one of: "
            + ", ".join(_ALLOWED_SHADOW_CUBE_SIZES)
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
            f"Invalid global_plane_confidence_threshold: {global_plane_confidence_threshold} (must be <= 1.0)"
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
    pedestrian_trajectory_t = _float_setting(
        blender_settings,
        "pedestrian_trajectory_t",
        0.0,
        minimum=0.0,
        maximum=1.0,
    )
    pedestrian_forward_offset_m = _float_setting(
        blender_settings,
        "pedestrian_forward_offset_m",
        5.0,
    )
    pedestrian_left_offset_m = _float_setting(
        blender_settings,
        "pedestrian_left_offset_m",
        2.0,
    )
    pedestrian_up_offset_m = _float_setting(
        blender_settings,
        "pedestrian_up_offset_m",
        0.0,
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
    unity_import = unity_import if isinstance(unity_import, dict) else {}

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
    pedestrian_actor_name = str(mixamo_settings.get("actor_name", "Pedestrian01")).strip()
    if not pedestrian_actor_name:
        raise ValueError("mixamo.actor_name must be a non-empty string.")
    mixamo_source_fps = float(mixamo_settings.get("source_fps", 30.0))
    if not np.isfinite(mixamo_source_fps) or mixamo_source_fps <= 0.0:
        raise ValueError("mixamo.source_fps must be a finite value > 0.")
    mixamo_export_fps = float(mixamo_settings.get("export_fps", 30.0))
    if not np.isfinite(mixamo_export_fps) or mixamo_export_fps <= 0.0:
        raise ValueError("mixamo.export_fps must be a finite value > 0.")
    mixamo_debug = bool(mixamo_settings.get("debug", True))

    if trajectory_path is None:
        trajectory_path = run_dir / "standard" / "trajectory" / "poses.npz"

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
        lighting=lighting,
    )


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load trajectory from poses.npz.

    Returns:
        (camera_to_world, frame_indices)
        camera_to_world: (N, 4, 4) float32
        frame_indices: (N,) int32
    """
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)

    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Invalid camera_to_world shape: {c2w.shape}")

    return c2w, frame_indices


def insert_mixamo_character(
    spec: SceneSpec,
    *,
    c2w_matrices: np.ndarray,
    frame_indices: np.ndarray,
    spawn_world: Vec3,
    trajectory_anchor_world: Vec3,
    intended_forward_world: Vec3,
) -> dict[str, Any] | None:
    with log_scope("Character"):
        if not (spec.mixamo_character_fbx_path and spec.mixamo_animation_fbx_path):
            log_warning(
                "Mixamo character or animation FBX path not provided; skipping import."
            )
            return None
        log_info("Importing Mixamo character and animation...")

        mixamo_spec = MixamoSpec(
            character_fbx=Path(spec.mixamo_character_fbx_path),
            animation_fbx=Path(spec.mixamo_animation_fbx_path),
            asset_root=Path(spec.mixamo_asset_root or Path(spec.mixamo_character_fbx_path).parent),
            actor_name=spec.pedestrian_actor_name,
            location=spawn_world,
            heading_deg=0.0,
            global_scale=1.0,
        )

        def _vec(v) -> str:
            return f"({v.x:+.4f},{v.y:+.4f},{v.z:+.4f})"

        scene = bpy.context.scene
        intended_scene_fps = _resolve_authoritative_sampling_fps(
            spec,
            context="Mixamo root-motion bake",
        )

        # Import character (NO animation)
        imported_objs = _import_fbx(
            mixamo_spec.character_fbx,
            global_scale=mixamo_spec.global_scale,
            use_anim=False,
        )

        # Import animation FBX (contains its own armature+action)
        imported_anim_objs = _import_fbx(
            mixamo_spec.animation_fbx,
            global_scale=mixamo_spec.global_scale,
            use_anim=True,
        )
        asset_diagnostics = _relink_and_validate_mixamo_materials(
            imported_objects=imported_objs,
            asset_root=mixamo_spec.asset_root,
            run_dir=spec.run_dir,
            material_policy=str(
                getattr(getattr(spec, "render", None), "material_policy", "preserve_base_alpha_normal")
            ),
        )
        live_scene_fps_after_import = _effective_scene_fps(scene)
        scene_fps_matches_intended = (
            abs(live_scene_fps_after_import - intended_scene_fps) <= 1e-6
        )
        if scene_fps_matches_intended:
            log_info(
                "Mixamo timing authority: intended sampling_fps matches live scene "
                f"FPS after import ({intended_scene_fps:.3f} FPS)."
            )
        else:
            log_warning(
                "Mixamo FBX import changed the live scene FPS. "
                f"intended_sampling_fps={intended_scene_fps:.3f} "
                f"live_scene_fps_after_import={live_scene_fps_after_import:.3f}. "
                "Using the intended sampling_fps as the authoritative timing "
                "source for bake calculations."
            )

        log_info(
            f"Imported {len(imported_objs)} character objects and {len(imported_anim_objs)} animation objects."
        )
        log_info("Character import objects:")
        for o in sorted(imported_objs, key=lambda x: x.name):
            log_info(
                f"  - {o.name:40s} type={o.type:10s} loc={_vec(o.location)} scale={_vec(o.scale)} parent={getattr(o.parent, 'name', None)}"
            )
        log_info("Animation import objects:")
        for o in sorted(imported_anim_objs, key=lambda x: x.name):
            log_info(
                f"  - {o.name:40s} type={o.type:10s} loc={_vec(o.location)} scale={_vec(o.scale)} parent={getattr(o.parent, 'name', None)}"
            )

        # ----------------------------
        # 1) Identify main character armature
        # ----------------------------
        char_armatures = [o for o in imported_objs if o.type == "ARMATURE"]
        if not char_armatures:
            raise RuntimeError(
                "Character FBX import did not create an ARMATURE object."
            )
        if len(char_armatures) > 1:
            log_info(
                f"WARNING: Multiple character armatures found: {[a.name for a in char_armatures]}"
            )
        char_arm = char_armatures[0]
        log_info(f"Character armature: {char_arm.name}")
        log_info(
            f"Character armature world matrix translation: {_vec(char_arm.matrix_world.translation)}"
        )

        # ----------------------------
        # 2) Create a root empty for placement/heading, parent armature under it
        # ----------------------------
        coll = scene.collection
        deps = bpy.context.evaluated_depsgraph_get()

        log_info(
            f"Scene frames: start={scene.frame_start} end={scene.frame_end} fps={scene.render.fps}/{scene.render.fps_base}"
        )
        log_info(
            f"Scene unit scale: {scene.unit_settings.scale_length} system={scene.unit_settings.system}"
        )

        root = bpy.data.objects.get(mixamo_spec.actor_name)
        if root is None:
            root = bpy.data.objects.new(mixamo_spec.actor_name, None)
            coll.objects.link(root)
            root.empty_display_type = "ARROWS"
            log_info(f"Created root empty: {root.name}")
        else:
            log_info(f"Reusing existing root object: {root.name} type={root.type}")

        log_info(f"ROOT parent={getattr(root.parent, 'name', None)}")
        log_info(f"ROOT constraints={[c.type for c in root.constraints]}")
        log_info(
            f"ROOT has anim_data={bool(root.animation_data)} action={getattr(getattr(root.animation_data, 'action', None), 'name', None)}"
        )
        # --- ensure root is truly world-space and clean ---
        root.parent = None
        root.matrix_parent_inverse.identity()

        # clear constraints
        for c in list(root.constraints):
            root.constraints.remove(c)

        # clear animation on root (very important when reusing objects)
        if root.animation_data:
            root.animation_data.action = None
            # delete existing fcurves on object transforms
            ad = root.animation_data
            if ad.action:
                bpy.data.actions.remove(ad.action)
            root.animation_data_clear()

        # also clear object-level animation on char_arm (optional but recommended)
        if char_arm.animation_data:
            char_arm.animation_data_clear()

        root.rotation_mode = "XYZ"
        root.location = mixamo_spec.location
        root.rotation_euler = (0.0, 0.0, 0.0)
        log_info(
            f"Root pose initialized: loc={_vec(root.location)} yaw_deg deferred until Mixamo-forward alignment"
        )

        # Parent keep-world for char_arm
        mw = char_arm.matrix_world.copy()
        char_arm.parent = root
        char_arm.matrix_parent_inverse = root.matrix_world.inverted() @ mw
        char_arm.matrix_world = mw
        log_info(f"Parented armature under root. arm.parent={char_arm.parent.name}")
        log_info(
            f"arm local loc={_vec(char_arm.location)}  arm world loc={_vec(char_arm.matrix_world.translation)}"
        )

        # Parent any meshes under armature
        reparented = 0
        for o in imported_objs:
            if o.type == "MESH":
                if o.parent != char_arm:
                    mw_m = o.matrix_world.copy()
                    o.parent = char_arm
                    o.matrix_parent_inverse = char_arm.matrix_world.inverted() @ mw_m
                    o.matrix_world = mw_m
                    reparented += 1
        log_info(
            f"Reparented {reparented} mesh objects under armature (if any were unparented)."
        )

        # ----------------------------
        # DEBUG: object-level parenting / constraints / animation
        # ----------------------------
        def _obj_dbg(o: bpy.types.Object) -> None:
            log_info(
                f"DBG OBJ {o.name} type={o.type} "
                f"parent={getattr(o.parent, 'name', None)} "
                f"constraints={[c.type + ':' + (getattr(c, 'target', None).name if getattr(c, 'target', None) else '') for c in o.constraints]} "
                f"anim_data={bool(o.animation_data)} "
                f"action={getattr(getattr(o.animation_data, 'action', None), 'name', None) if o.animation_data else None}"
            )

        _obj_dbg(root)
        _obj_dbg(char_arm)
        for o in imported_objs:
            if o.type == "MESH":
                _obj_dbg(o)

        # ----------------------------
        # DEBUG: bone-level constraints
        # ----------------------------
        def _bone_dbg(arm: bpy.types.Object, bone_name: str) -> None:
            pb = arm.pose.bones.get(bone_name)
            if not pb:
                log_info(f"DBG BONE {bone_name}: not found")
                return
            log_info(
                f"DBG BONE {bone_name} constraints="
                f"{[c.type + ':' + (getattr(c, 'target', None).name if getattr(c, 'target', None) else '') for c in pb.constraints]}"
            )

        # Hips (always)
        _bone_dbg(char_arm, "mixamorig7:Hips")

        # ----------------------------
        # 3) Get the animation action from the imported animation armature
        # ----------------------------
        anim_armatures = [o for o in imported_anim_objs if o.type == "ARMATURE"]
        if not anim_armatures:
            raise RuntimeError(
                "Animation FBX import did not create an ARMATURE object."
            )
        if len(anim_armatures) > 1:
            log_info(
                f"WARNING: Multiple animation armatures found: {[a.name for a in anim_armatures]}"
            )
        anim_arm = anim_armatures[0]
        log_info(f"Animation armature: {anim_arm.name}")

        anim_action = None
        if anim_arm.animation_data and anim_arm.animation_data.action:
            anim_action = anim_arm.animation_data.action
            log_info(f"Animation armature has direct action: {anim_action.name}")

        if anim_action is None:
            if anim_arm.animation_data and anim_arm.animation_data.nla_tracks:
                log_info(
                    f"Animation armature NLA tracks: {len(anim_arm.animation_data.nla_tracks)}"
                )
                for tr in anim_arm.animation_data.nla_tracks:
                    log_info(
                        f"  Track: {tr.name} strips={len(tr.strips)} mute={tr.mute}"
                    )
                    for st in tr.strips:
                        log_info(
                            f"    Strip: {st.name} action={getattr(st.action, 'name', None)} mute={st.mute}"
                        )
                        if st.action and anim_action is None:
                            anim_action = st.action
                if anim_action:
                    log_info(f"Picked action from NLA strip: {anim_action.name}")

        if anim_action is None:
            raise RuntimeError(
                "Could not find an Action on the imported animation armature."
            )

        log_info(f"Animation action: {anim_action.name}")
        fr = anim_action.frame_range
        log_info(f"Action frame_range: [{fr[0]:.2f}, {fr[1]:.2f}]")

        # Force update to ensure the imported source animation rig is active.
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()

        # ----------------------------
        # 4) Find hips bone
        # ----------------------------
        hips = None
        hips_bone_name = None
        for nm in ("mixamorig:Hips", "mixamorig7:Hips", "Hips"):
            if nm in char_arm.pose.bones:
                hips = char_arm.pose.bones[nm]
                hips_bone_name = nm
                break
        if hips is None:
            for b in char_arm.pose.bones:
                if b.name.lower().endswith("hips"):
                    hips = b
                    hips_bone_name = b.name
                    break
        if hips is None:
            raise RuntimeError("Could not find hips bone.")
        log_info(f"Hips bone: {hips_bone_name}")

        source_hips = None
        source_hips_bone_name = None
        for nm in ("mixamorig:Hips", "mixamorig7:Hips", "Hips"):
            if nm in anim_arm.pose.bones:
                source_hips = anim_arm.pose.bones[nm]
                source_hips_bone_name = nm
                break
        if source_hips is None:
            for b in anim_arm.pose.bones:
                if b.name.lower().endswith("hips"):
                    source_hips = b
                    source_hips_bone_name = b.name
                    break
        if source_hips is None or source_hips_bone_name is None:
            raise RuntimeError("Could not find hips bone on the imported animation armature.")
        log_info(f"Source animation hips bone: {source_hips_bone_name}")

        # ----------------------------
        # 5) Sample original animation and compute root motion
        # ----------------------------
        try:
            a0, a1 = anim_action.frame_range
            fs = int(round(a0))
            fe = int(round(a1))
        except Exception:
            fs = int(scene.frame_start)
            fe = int(scene.frame_end)

        if fe <= fs + 2:
            raise RuntimeError(f"Action range too small: fs={fs}, fe={fe}")

        log_info(f"Action cycle range: {fs}..{fe}")

        cycle_len = fe - fs
        if cycle_len < 2:
            raise RuntimeError(f"Cycle length too small: {cycle_len}")
        log_info(f"Source cycle length: {cycle_len} frames")

        bake_s = int(scene.frame_start)
        bake_e = int(scene.frame_end)
        log_info(f"Scene bake range: {bake_s}..{bake_e}")
        source_fps = float(spec.mixamo_source_fps)
        scene_fps = float(intended_scene_fps)
        cycle_duration_seconds = compute_cycle_duration_seconds(cycle_len, source_fps)
        cycle_len_output_frames = cycle_duration_seconds * scene_fps
        log_info(
            "Cycle timing: "
            f"source_fps={source_fps:.3f} scene_fps={scene_fps:.3f} "
            f"(authoritative) live_scene_fps_at_bake_check={live_scene_fps_after_import:.3f} "
            f"source_cycle_frames={cycle_len} cycle_duration_s={cycle_duration_seconds:.6f} "
            f"output_cycle_frames={cycle_len_output_frames:.6f}"
        )
        # Persist animation-cycle metadata for downstream contact planning.
        root["pemoin_mixamo_cycle_len_frames"] = float(cycle_len_output_frames)
        root["pemoin_mixamo_source_cycle_len_frames"] = float(cycle_len)
        root["pemoin_mixamo_source_fps"] = float(source_fps)
        root["pemoin_mixamo_scene_fps"] = float(scene_fps)
        root["pemoin_mixamo_cycle_duration_seconds"] = float(cycle_duration_seconds)
        root["pemoin_mixamo_source_start_frame"] = float(fs)
        root["pemoin_mixamo_sampling_mode"] = "continuous_phase_source_time"
        root["pemoin_mixamo_live_scene_fps_at_bake_check"] = float(
            live_scene_fps_after_import
        )
        root["pemoin_mixamo_scene_fps_matches_intended"] = bool(
            scene_fps_matches_intended
        )
        root["pemoin_mixamo_bake_start_frame"] = float(bake_s)
        root["pemoin_mixamo_bake_end_frame"] = float(bake_e)
        char_arm["pemoin_mixamo_cycle_len_frames"] = float(cycle_len_output_frames)
        char_arm["pemoin_mixamo_source_cycle_len_frames"] = float(cycle_len)
        char_arm["pemoin_mixamo_source_fps"] = float(source_fps)
        char_arm["pemoin_mixamo_scene_fps"] = float(scene_fps)
        char_arm["pemoin_mixamo_cycle_duration_seconds"] = float(cycle_duration_seconds)
        char_arm["pemoin_mixamo_source_start_frame"] = float(fs)
        char_arm["pemoin_mixamo_sampling_mode"] = "continuous_phase_source_time"
        char_arm["pemoin_mixamo_live_scene_fps_at_bake_check"] = float(
            live_scene_fps_after_import
        )
        char_arm["pemoin_mixamo_scene_fps_matches_intended"] = bool(
            scene_fps_matches_intended
        )
        char_arm["pemoin_mixamo_bake_start_frame"] = float(bake_s)
        char_arm["pemoin_mixamo_bake_end_frame"] = float(bake_e)

        dt = 1.0 / max(scene_fps, 1e-6)

        root.animation_data_create()
        spawn_arr = np.asarray(spawn_world, dtype=np.float32)
        trajectory_anchor_arr = np.asarray(trajectory_anchor_world, dtype=np.float32)
        root.location = tuple(float(v) for v in spawn_arr.tolist())
        # Sample hips WORLD positions from the imported source animation rig.
        log_info("Sampling source hips world positions for cycle...")
        hips_world = {}

        for i in range(cycle_len + 1):
            f = fs + i

            scene.frame_set(f)
            bpy.context.view_layer.update()
            deps = bpy.context.evaluated_depsgraph_get()
            anim_arm_eval = anim_arm.evaluated_get(deps)
            hips_eval = anim_arm_eval.pose.bones.get(source_hips_bone_name)
            if hips_eval:
                hw = (anim_arm_eval.matrix_world @ hips_eval.matrix).translation.copy()
            else:
                hw = (anim_arm.matrix_world @ source_hips.matrix).translation.copy()

            hw.z = 0.0
            hips_world[i] = hw

            if (i == 0) or (i == cycle_len) or (i % 10 == 0):
                hips_loc_debug = (
                    source_hips.location.copy() if source_hips else Vector((0, 0, 0))
                )
                log_info(
                    f"[source sample] i={i:3d} f={f:3d} hips_w={_vec(hw)} hips_bone_loc={_vec(hips_loc_debug)}"
                )

        log_info("Sampling source hips local positions for cycle...")
        hips_local = {}

        for i in range(cycle_len + 1):
            f = fs + i

            scene.frame_set(f)
            bpy.context.view_layer.update()
            deps = bpy.context.evaluated_depsgraph_get()
            anim_arm_eval = anim_arm.evaluated_get(deps)
            hips_eval = anim_arm_eval.pose.bones.get(source_hips_bone_name)
            if hips_eval:
                hl = hips_eval.location.copy()
            else:
                hl = source_hips.location.copy()

            hips_local[i] = hl

            if (i == 0) or (i == cycle_len) or (i % 10 == 0):
                log_info(f"[source local] i={i:3d} f={f:3d} hips_local={_vec(hl)}")

        cycle_displacement_world = hips_world[cycle_len] - hips_world[0]
        log_info(f"Cycle displacement (world): {_vec(cycle_displacement_world)}")

        cycle_offset_local = hips_local[cycle_len] - hips_local[0]
        log_info(f"Cycle offset (local): {_vec(cycle_offset_local)}")
        cycle_displacement_world_xy = (
            float(cycle_displacement_world.x),
            float(cycle_displacement_world.y),
        )
        cycle_offset_local_xy = (
            float(cycle_offset_local.x),
            float(cycle_offset_local.y),
        )
        cycle_world_delta_samples_xy = [
            (
                float(hips_world[idx + 1].x - hips_world[idx].x),
                float(hips_world[idx + 1].y - hips_world[idx].y),
            )
            for idx in range(cycle_len)
        ]
        cycle_local_delta_samples_xy = [
            (
                float(hips_local[idx + 1].x - hips_local[idx].x),
                float(hips_local[idx + 1].y - hips_local[idx].y),
            )
            for idx in range(cycle_len)
        ]
        locomotion_present, locomotion_diagnostics = classify_locomotion_from_world_deltas(
            cycle_displacement_world_xy,
            cycle_world_delta_samples_xy,
        )
        local_locomotion_present, local_locomotion_diagnostics = (
            classify_locomotion_from_world_deltas(
                cycle_offset_local_xy,
                cycle_local_delta_samples_xy,
            )
        )
        world_locomotion_axis_xy = None
        local_locomotion_axis_xy = None
        local_axis_method = "none"
        local_axis_confidence = 0.0
        motion_policy = str(getattr(spec, "pedestrian_motion_policy", "auto")).strip().lower()
        animation_motion_category = resolve_mixamo_animation_motion_category(
            mixamo_spec.animation_fbx
        )
        path_resolved_motion_policy = resolve_mixamo_motion_policy_from_animation_path(
            mixamo_spec.animation_fbx
        )
        if motion_policy == "auto":
            effective_motion_policy = path_resolved_motion_policy
        else:
            effective_motion_policy = motion_policy
        if effective_motion_policy == "animation_root_motion":
            if not locomotion_present:
                raise RuntimeError(
                    "Animation-root-motion policy requires a moving Mixamo clip with usable "
                    "horizontal locomotion, but the imported clip does not provide one."
                )
            world_locomotion_axis_xy, world_axis_method, world_axis_confidence = (
                resolve_dominant_horizontal_direction(
                    cycle_displacement_world_xy,
                    cycle_world_delta_samples_xy,
                )
            )
            if local_locomotion_present:
                try:
                    local_locomotion_axis_xy, local_axis_method, local_axis_confidence = (
                        resolve_dominant_horizontal_direction(
                            cycle_offset_local_xy,
                            cycle_local_delta_samples_xy,
                        )
                    )
                except ValueError:
                    local_locomotion_axis_xy = None
                    local_axis_method = "ambiguous_local_channel"
                    local_axis_confidence = 0.0
            else:
                log_info(
                    "Local hips locomotion is negligible in the imported clip; "
                    "preserving local pose translation and transferring only world "
                    "forward progress onto the actor root."
                )
            if local_locomotion_axis_xy is None and local_axis_method == "ambiguous_local_channel":
                log_info(
                    "Local hips locomotion samples are directionally ambiguous; "
                    "preserving local pose translation and transferring only world "
                    "forward progress onto the actor root."
                )
            log_info(
                "Resolved source locomotion axes: "
                f"world_axis=({float(world_locomotion_axis_xy[0]):+.4f},{float(world_locomotion_axis_xy[1]):+.4f}) "
                f"method={world_axis_method} confidence={world_axis_confidence:.3f} "
                f"local_axis={'none' if local_locomotion_axis_xy is None else f'({float(local_locomotion_axis_xy[0]):+.4f},{float(local_locomotion_axis_xy[1]):+.4f})'} "
                f"local_method={local_axis_method} local_confidence={local_axis_confidence:.3f} "
                f"cycle_horizontal_norm={locomotion_diagnostics['cycle_horizontal_norm']:.4f}"
            )
        if effective_motion_policy == "camera_trajectory_relative":
            root_path_world, path_forward_world, path_heading_world_deg = (
                sample_pedestrian_spawn_path_world(
                    c2w_matrices,
                    float(spec.pedestrian_trajectory_t),
                    float(spec.pedestrian_forward_offset_m),
                    float(spec.pedestrian_left_offset_m),
                    float(spec.pedestrian_up_offset_m),
                    sample_count=int(len(frame_indices)),
                )
            )
        else:
            root_path_world, path_forward_world, path_heading_world_deg = (
                stationary_pedestrian_spawn_path_world(
                    spawn_world,
                    intended_forward_world,
                    sample_count=int(len(frame_indices)),
                )
            )
        root_path_start = np.asarray(root_path_world[0], dtype=np.float32)
        spawn_start_delta = float(np.linalg.norm(root_path_start - spawn_arr))
        if spawn_start_delta > 1e-4:
            log_warning(
                "Authored root path start does not exactly match the resolved spawn. "
                f"start_delta={spawn_start_delta:.6f}m start={tuple(float(v) for v in root_path_start.tolist())} "
                f"spawn={tuple(float(v) for v in spawn_arr.tolist())}. Using the sampled path start."
            )
        root.location = tuple(float(v) for v in root_path_start.tolist())
        log_info(
            "Resolved pedestrian root-motion policy: "
            f"configured={motion_policy} effective={effective_motion_policy} "
            f"animation_category={animation_motion_category} "
            f"animation_path={mixamo_spec.animation_fbx}"
        )
        if effective_motion_policy == "camera_trajectory_relative":
            log_warning(
                "Using deprecated camera_trajectory_relative pedestrian motion policy. "
                "This keeps the actor root coupled to the camera trajectory and exists "
                "only as an explicit legacy/debug mode."
            )
        try:
            asset_forward_world_zero_xy, asset_forward_resolution_method, asset_forward_resolution_confidence = (
                _measure_armature_body_facing_world_xy(char_arm)
            )
        except Exception as exc:
            asset_forward_world_zero_xy = standard_mixamo_forward_world_xy()
            asset_forward_resolution_method = "mixamo_default_world_axis_fallback"
            asset_forward_resolution_confidence = 0.0
            log_warning(
                "Falling back to the default Mixamo facing axis because measured body-facing "
                f"resolution failed: {exc}"
            )
        resolved_root_yaw_world_deg, expected_heading_world_xy, asset_forward_world_zero_xy = (
            resolve_motion_aligned_actor_yaw_deg(
                asset_forward_world_xy=asset_forward_world_zero_xy,
                intended_forward_world=intended_forward_world,
                heading_offset_deg=float(spec.pedestrian_heading_deg),
            )
        )
        intended_forward_xy = _normalize_xy_or_none(intended_forward_world)
        if intended_forward_xy is None:
            raise RuntimeError(
                "Intended pedestrian forward direction is too small to normalize."
            )
        expected_heading_world_xy = _normalize_xy_or_none(expected_heading_world_xy)
        if expected_heading_world_xy is None:
            raise RuntimeError(
                "Resolved pedestrian heading direction is too small to normalize."
            )
        source_heading_local_axis_xy = _resolve_heading_axis_in_object_local_xy(
            anim_arm.matrix_world,
            expected_heading_world_xy,
        )
        target_heading_local_axis_xy = _resolve_heading_axis_in_object_local_xy(
            char_arm.matrix_world,
            expected_heading_world_xy,
        )
        root.rotation_euler = (0.0, 0.0, math.radians(resolved_root_yaw_world_deg))
        log_info(
            "Resolved actor motion alignment: "
            f"asset_forward_world_zero_xy=({float(asset_forward_world_zero_xy[0]):+.4f},{float(asset_forward_world_zero_xy[1]):+.4f}) "
            f"resolution={asset_forward_resolution_method} "
            f"confidence={asset_forward_resolution_confidence:.3f} "
            f"insertion_forward_xy=({float(intended_forward_xy[0]):+.4f},{float(intended_forward_xy[1]):+.4f}) "
            f"desired_locomotion_xy=({float(expected_heading_world_xy[0]):+.4f},{float(expected_heading_world_xy[1]):+.4f}) "
            f"root_yaw_world_deg={resolved_root_yaw_world_deg:.3f}"
        )
        log_info(
            "Resolved deterministic heading basis for hips correction: "
            f"source_local_axis=({float(source_heading_local_axis_xy[0]):+.4f},{float(source_heading_local_axis_xy[1]):+.4f}) "
            f"target_local_axis=({float(target_heading_local_axis_xy[0]):+.4f},{float(target_heading_local_axis_xy[1]):+.4f})"
        )
        log_info(
            "Using measured imported body-facing basis instead of animation-derived locomotion direction. "
            f"cycle_displacement_world_xy=({cycle_displacement_world_xy[0]:+.6f},{cycle_displacement_world_xy[1]:+.6f})"
        )

        # ----------------------------
        # 6) Bake sampled pose transforms at continuous authored-time phase
        # ----------------------------
        log_info("Preparing continuous-phase pose sampling...")
        sample_arm = anim_arm
        sample_hips_bone_name = source_hips_bone_name
        log_info(f"Using imported source animation armature for bone sampling: {sample_arm.name}")

        # Build source->target bone map so fallback sampling from a differently
        # named Mixamo armature (e.g. mixamorig:* vs mixamorig7:*) still bakes.
        def _bone_suffix(name: str) -> str:
            if ":" in name:
                return name.split(":", 1)[1].lower()
            return name.lower()

        target_bones = list(char_arm.pose.bones)
        target_by_suffix: dict[str, list[str]] = {}
        for target in target_bones:
            suffix = _bone_suffix(target.name)
            target_by_suffix.setdefault(suffix, []).append(target.name)

        bone_name_map: dict[str, str] = {}
        for source_name in sample_arm.pose.bones.keys():
            if source_name in char_arm.pose.bones:
                bone_name_map[source_name] = source_name
                continue
            suffix = _bone_suffix(source_name)
            candidates = target_by_suffix.get(suffix, [])
            if len(candidates) == 1:
                bone_name_map[source_name] = candidates[0]

        log_info(
            f"Bone map resolved {len(bone_name_map)}/{len(sample_arm.pose.bones)} source bones."
        )

        # ----------------------------
        # 7) Create extended action with keyframes
        # ----------------------------
        extended_action = bpy.data.actions.new(
            name=f"{mixamo_spec.actor_name}_Extended"
        )
        log_info(f"Created extended action: {extended_action.name}")

        total_frames = bake_e - bake_s + 1
        log_info(
            "Extending animation with continuous authored-time sampling: "
            f"{total_frames} total frames"
        )

        # Assign the new action first
        char_arm.animation_data_create()
        char_arm.animation_data.action = extended_action
        char_arm.animation_data.use_nla = False

        # Clear any existing NLA tracks
        if char_arm.animation_data.nla_tracks:
            for track in list(char_arm.animation_data.nla_tracks):
                char_arm.animation_data.nla_tracks.remove(track)

        log_info("Writing continuous-phase keyframes via evaluated pose sampling...")

        inserted_keys = 0
        sampled_source_frames: list[float] = []
        sampled_source_frame_by_output_frame: dict[int, float] = {}
        sampled_source_absolute_progress_frames: list[float] = []
        sampled_source_completed_cycles: list[int] = []
        sampled_root_motion_world_xy: list[np.ndarray] = []
        sampled_root_motion_forward_world_m: list[float] = []
        base_hips_world_xy = np.asarray(
            [float(hips_world[0].x), float(hips_world[0].y)],
            dtype=np.float32,
        )
        cycle_world_distance_m = (
            0.0
            if world_locomotion_axis_xy is None
            else float(
                np.dot(
                    np.asarray(cycle_displacement_world_xy, dtype=np.float32),
                    np.asarray(world_locomotion_axis_xy, dtype=np.float32),
                )
            )
        )
        base_hips_world_translation = np.asarray(
            [float(hips_world[0].x), float(hips_world[0].y), float(hips_world[0].z)],
            dtype=np.float32,
        )
        base_source_arm_world_translation = np.asarray(
            [
                float(sample_arm.matrix_world.translation.x),
                float(sample_arm.matrix_world.translation.y),
                float(sample_arm.matrix_world.translation.z),
            ],
            dtype=np.float32,
        )
        base_target_arm_world_translation = np.asarray(
            [
                float(char_arm.matrix_world.translation.x),
                float(char_arm.matrix_world.translation.y),
                float(char_arm.matrix_world.translation.z),
            ],
            dtype=np.float32,
        )
        for F in range(bake_s, bake_e + 1):
            t_scene = F - bake_s
            t_seconds = float(t_scene) * dt
            timing = resolve_looped_source_timing(
                t_seconds,
                cycle_duration_seconds,
                cycle_len,
                source_start_frame=float(fs),
            )
            source_frame_float = float(timing.wrapped_source_frame_float)
            sampled_source_frames.append(float(source_frame_float))
            sampled_source_frame_by_output_frame[int(F)] = float(source_frame_float)
            sampled_source_absolute_progress_frames.append(
                float(timing.absolute_source_progress_frames)
            )
            sampled_source_completed_cycles.append(int(timing.completed_cycles))
            source_frame_base = math.floor(float(source_frame_float))
            source_frame_sub = float(source_frame_float) - float(source_frame_base)
            cycle_phase = float(timing.cycle_phase)

            _set_scene_frame_float(scene, float(source_frame_float))
            bpy.context.view_layer.update()
            deps = bpy.context.evaluated_depsgraph_get()
            sample_arm_eval = sample_arm.evaluated_get(deps)

            for bone in sample_arm_eval.pose.bones:
                bone_name = bone.name
                target_bone_name = bone_name_map.get(bone_name)
                if target_bone_name is None:
                    continue

                target_bone = char_arm.pose.bones[target_bone_name]
                loc = bone.location.copy()
                rot = bone.rotation_quaternion.copy()
                scl = bone.scale.copy()

                if bone_name == sample_hips_bone_name:
                    hips_world_matrix = (sample_arm_eval.matrix_world @ bone.matrix).copy()
                    hips_world_now = hips_world_matrix.translation.copy()
                    completed_cycles = int(timing.completed_cycles)
                    raw_world_xy = np.asarray(
                        [float(hips_world_now.x), float(hips_world_now.y)],
                        dtype=np.float32,
                    )
                    sampled_root_motion_world_xy.append(np.asarray(raw_world_xy, dtype=np.float32))
                    in_cycle_world_forward_m = (
                        0.0
                        if world_locomotion_axis_xy is None
                        else float(
                            np.dot(
                                raw_world_xy - base_hips_world_xy,
                                np.asarray(world_locomotion_axis_xy, dtype=np.float32),
                            )
                        )
                    )
                    transferred_world_forward_m = (
                        float(completed_cycles) * cycle_world_distance_m
                        + float(in_cycle_world_forward_m)
                    )
                    sampled_root_motion_forward_world_m.append(float(transferred_world_forward_m))
                    desired_world_translation, forward_correction_world, nonforward_residual_world = (
                        _stabilize_looping_pelvis_world_translation(
                            raw_world_translation=(
                                float(hips_world_now.x),
                                float(hips_world_now.y),
                                float(hips_world_now.z),
                            ),
                            base_world_translation=tuple(
                                float(v) for v in base_hips_world_translation.tolist()
                            ),
                            source_anchor_world_translation=tuple(
                                float(v)
                                for v in base_source_arm_world_translation.tolist()
                            ),
                            target_anchor_world_translation=tuple(
                                float(v)
                                for v in base_target_arm_world_translation.tolist()
                            ),
                            locomotion_axis_xy=world_locomotion_axis_xy,
                        )
                    )
                    desired_world_matrix = hips_world_matrix.copy()
                    desired_world_matrix.translation = Vector(
                        tuple(float(v) for v in desired_world_translation.tolist())
                    )
                    desired_local_matrix = char_arm.convert_space(
                        pose_bone=target_bone,
                        matrix=desired_world_matrix,
                        from_space="WORLD",
                        to_space="LOCAL",
                    )
                    loc_corrected = desired_local_matrix.translation.copy()
                    rot = desired_local_matrix.to_quaternion()
                    scl = desired_local_matrix.to_scale()
                    forward_correction_norm_m = float(
                        np.linalg.norm(forward_correction_world[:2])
                    )
                    residual_lateral_norm_m = float(
                        np.linalg.norm(nonforward_residual_world[:2])
                    )
                    if (F == bake_s) or (F == bake_e) or ((F - bake_s) % 10 == 0):
                        log_info(
                            f"[hips correction] frame={F} src={source_frame_float:.4f} cycles={completed_cycles} "
                            f"phase={cycle_phase:.4f} abs_progress_frames={float(timing.absolute_source_progress_frames):.4f} raw={_vec(loc)} "
                            f"world_forward_m={transferred_world_forward_m:+.4f} "
                            f"in_cycle_world_forward_m={in_cycle_world_forward_m:+.4f} "
                            f"forward_correction_norm_m={forward_correction_norm_m:+.4f} "
                            f"residual_lateral_norm_m={residual_lateral_norm_m:+.4f} "
                            f"pelvis_world={tuple(round(float(v),4) for v in desired_world_translation.tolist())} "
                            f"corrected={_vec(loc_corrected)}"
                        )
                    loc = loc_corrected

                target_bone.location = loc
                target_bone.rotation_quaternion = rot
                target_bone.scale = scl

                target_bone.keyframe_insert(data_path="location", frame=F)
                target_bone.keyframe_insert(data_path="rotation_quaternion", frame=F)
                target_bone.keyframe_insert(data_path="scale", frame=F)
                inserted_keys += 3

            if (F == bake_s) or (F == bake_e) or ((F - bake_s) % 10 == 0):
                log_info(
                    f"[bake bones] frame={F} scene_idx={t_scene} t_s={t_seconds:.4f} "
                    f"source_frame={source_frame_base}+{source_frame_sub:.4f} "
                    f"cycles={int(timing.completed_cycles)} abs_progress_frames={float(timing.absolute_source_progress_frames):.4f}"
                )

        if inserted_keys == 0:
            raise RuntimeError(
                "No bone keyframes were inserted for the character armature."
            )
        log_info(
            f"Keyframed all bones for {total_frames} frames "
            f"(inserted channels={inserted_keys})."
        )
        if len(sampled_root_motion_forward_world_m) != total_frames:
            raise RuntimeError(
                "Failed to sample one source root-motion progress value per baked output frame: "
                f"expected={total_frames} got={len(sampled_root_motion_forward_world_m)}."
            )
        source_pose_candidate_bones = [
            bone_name
            for bone_name in (
                source_hips_bone_name,
                "mixamorig:Spine",
                "mixamorig7:Spine",
                "mixamorig:LeftArm",
                "mixamorig7:LeftArm",
                "mixamorig:LeftForeArm",
                "mixamorig7:LeftForeArm",
            )
            if bone_name in sample_arm.pose.bones and bone_name in bone_name_map
        ]
        pose_parity_frames = [bake_s, min(bake_s + 1, bake_e), bake_e]
        source_pose_signal = 0.0
        baked_pose_signal = 0.0
        for source_name in source_pose_candidate_bones:
            target_name = bone_name_map[source_name]
            for frame_number in pose_parity_frames:
                source_frame_float = sampled_source_frame_by_output_frame[int(frame_number)]
                _set_scene_frame_float(scene, float(source_frame_float))
                bpy.context.view_layer.update()
                deps = bpy.context.evaluated_depsgraph_get()
                source_eval = sample_arm.evaluated_get(deps)
                scene.frame_set(int(frame_number))
                bpy.context.view_layer.update()
                deps = bpy.context.evaluated_depsgraph_get()
                target_eval = char_arm.evaluated_get(deps)
                source_bone = source_eval.pose.bones.get(source_name)
                target_bone = target_eval.pose.bones.get(target_name)
                if source_bone is not None:
                    sq = source_bone.rotation_quaternion
                    source_pose_signal = max(
                        source_pose_signal,
                        abs(float(sq.w) - 1.0)
                        + abs(float(sq.x))
                        + abs(float(sq.y))
                        + abs(float(sq.z)),
                    )
                if target_bone is not None:
                    tq = target_bone.rotation_quaternion
                    baked_pose_signal = max(
                        baked_pose_signal,
                        abs(float(tq.w) - 1.0)
                        + abs(float(tq.x))
                        + abs(float(tq.y))
                        + abs(float(tq.z)),
                    )
        if source_pose_signal > 0.1 and baked_pose_signal < 0.02:
            raise RuntimeError(
                "Baked character action appears to be near bind pose even though the "
                f"source animation shows meaningful pose motion: source_signal={source_pose_signal:.4f} "
                f"baked_signal={baked_pose_signal:.4f}."
            )

        # ----------------------------
        # 8) Bake actor root motion keyframes from the resolved authored path
        # ----------------------------
        frame_numbers = [int(frame) for frame in frame_indices.tolist()]
        source_root_motion_forward_progress_m = np.asarray(
            sampled_root_motion_forward_world_m,
            dtype=np.float32,
        ).reshape(-1)
        if source_root_motion_forward_progress_m.shape[0] != total_frames:
            raise RuntimeError(
                "Animation-root-motion bake sampled an unexpected number of forward "
                f"progress values: expected={total_frames} "
                f"got={source_root_motion_forward_progress_m.shape[0]}."
            )
        if not np.isfinite(source_root_motion_forward_progress_m).all():
            raise RuntimeError(
                "Animation-root-motion bake produced non-finite forward progress samples."
            )
        source_root_motion_forward_samples_m = (
            source_root_motion_forward_progress_m
            - float(source_root_motion_forward_progress_m[0])
        ).astype(np.float32)
        raw_progress_deltas_m = np.diff(source_root_motion_forward_samples_m, axis=0)
        max_backward_progress_step_m = (
            0.0
            if raw_progress_deltas_m.size == 0
            else float(max(0.0, -float(np.min(raw_progress_deltas_m))))
        )
        max_forward_progress_step_m = (
            0.0
            if raw_progress_deltas_m.size == 0
            else float(np.max(raw_progress_deltas_m))
        )
        if effective_motion_policy == "animation_root_motion" and max_backward_progress_step_m > 1e-3:
            raise RuntimeError(
                "Animation-root-motion extraction produced backward progress across a loop seam. "
                f"max_backward_step={max_backward_progress_step_m:.6f}m."
            )
        source_root_motion_progress_path_length_m = float(
            source_root_motion_forward_progress_m[-1]
            - source_root_motion_forward_progress_m[0]
        )
        if effective_motion_policy == "animation_root_motion":
            log_info(
                "Keyframing heading-aligned animation-root-motion-authored root path..."
            )
            root_path_world, path_forward_world, path_heading_world_deg = (
                build_heading_aligned_root_motion_path_world(
                    source_root_motion_forward_progress_m,
                    spawn_world,
                    expected_heading_world_xy,
                )
            )
            resolved_root_yaws_world_deg = [
                float(resolved_root_yaw_world_deg) for _ in frame_numbers
            ]
        elif effective_motion_policy == "camera_trajectory_relative":
            log_info("Keyframing legacy camera-trajectory-relative root motion...")
            resolved_root_yaws_world_deg = []
        else:
            log_info("Keyframing stationary root motion at resolved spawn...")
            resolved_root_yaws_world_deg = []

        root_path_positions = [
            Vector(tuple(float(v) for v in np.asarray(pos, dtype=np.float32).tolist()))
            for pos in root_path_world
        ]
        root_path_forwards = [
            tuple(float(v) for v in np.asarray(vec, dtype=np.float32).tolist())
            for vec in path_forward_world
        ]
        sampled_base_heading_world_deg = [float(v) for v in path_heading_world_deg.tolist()]

        if len(frame_numbers) != len(root_path_positions):
            raise RuntimeError(
                "Frame index count does not match the authored root path length."
            )

        cumulative_xy_distances_m = [0.0]
        for idx, (frame_number, path_position, path_forward_world_sample) in enumerate(
            zip(frame_numbers, root_path_positions, root_path_forwards, strict=True)
        ):
            if effective_motion_policy == "animation_root_motion":
                resolved_frame_yaw_deg = float(resolved_root_yaw_world_deg)
            else:
                resolved_frame_yaw_deg, _expected_heading_world_xy_frame, _ = (
                    resolve_motion_aligned_actor_yaw_deg(
                        asset_forward_world_xy=asset_forward_world_zero_xy,
                        intended_forward_world=path_forward_world_sample,
                        heading_offset_deg=float(spec.pedestrian_heading_deg),
                    )
                )
                resolved_root_yaws_world_deg.append(float(resolved_frame_yaw_deg))
            root.location = path_position
            root.rotation_euler = (0.0, 0.0, math.radians(resolved_frame_yaw_deg))
            root.keyframe_insert(data_path="location", frame=frame_number)
            root.keyframe_insert(data_path="rotation_euler", frame=frame_number)
            if idx > 0:
                cumulative_xy_distances_m.append(
                    float(
                        cumulative_xy_distances_m[-1]
                        + math.hypot(
                            float(path_position.x - root_path_positions[idx - 1].x),
                            float(path_position.y - root_path_positions[idx - 1].y),
                        )
                    )
                )
            if (
                idx == 0
                or idx == len(frame_numbers) - 1
                or ((frame_number - frame_numbers[0]) % 50 == 0)
            ):
                log_info(
                    f"[root] frame={frame_number:4d} path_idx={idx:4d} "
                    f"loc={_vec(root.location)} yaw_deg={resolved_frame_yaw_deg:.3f} "
                    f"path_distance_m={cumulative_xy_distances_m[-1]:.4f}"
                )

        total_time = max(0.0, float(len(frame_numbers) - 1) * dt)
        total_dist_final = float(cumulative_xy_distances_m[-1])
        baked_root_start_world_arr = np.asarray(
            tuple(float(v) for v in root_path_positions[0]),
            dtype=np.float32,
        )
        baked_root_end_world_arr = np.asarray(
            tuple(float(v) for v in root_path_positions[-1]),
            dtype=np.float32,
        )
        frame0_root_to_spawn_delta_m = _validate_authored_root_path_starts_at_spawn(
            path_start_world=baked_root_start_world_arr,
            resolved_spawn_world=spawn_world,
        )
        trajectory_xy = np.asarray(c2w_matrices, dtype=np.float32)[:, :2, 3]
        trajectory_anchor_to_frames_m = np.linalg.norm(
            trajectory_xy - trajectory_anchor_arr[:2].reshape(1, 2),
            axis=1,
        )
        nearest_trajectory_frame_array_idx = int(np.argmin(trajectory_anchor_to_frames_m))
        nearest_trajectory_frame_index = int(frame_indices[nearest_trajectory_frame_array_idx])
        nearest_trajectory_frame_xy_distance_m = float(
            trajectory_anchor_to_frames_m[nearest_trajectory_frame_array_idx]
        )
        avg_speed = 0.0 if total_time <= 1e-6 else float(total_dist_final / total_time)
        log_info(
            "Root motion bake complete: "
            f"policy={effective_motion_policy} avg_speed={avg_speed:.3f}m/s "
            f"total_xy_dist={total_dist_final:.3f}m duration_s={total_time:.3f} "
            f"frame0_root_to_spawn_delta_m={frame0_root_to_spawn_delta_m:.6f}"
        )
        if effective_motion_policy == "stationary_at_spawn" and total_dist_final > 1e-4:
            raise RuntimeError(
                "Stationary Mixamo motion policy produced non-stationary authored root "
                f"motion: total_xy_dist={total_dist_final:.6f}m."
            )
        if effective_motion_policy == "animation_root_motion":
            if (
                animation_motion_category == "moving"
                and source_root_motion_progress_path_length_m <= 1e-3
            ):
                raise RuntimeError(
                    "Moving Mixamo clip resolved to animation_root_motion but the extracted "
                    "forward progress is effectively zero."
                )
            if abs(total_dist_final - source_root_motion_progress_path_length_m) > 1e-3:
                raise RuntimeError(
                    "Baked world root-motion path length does not match extracted Mixamo "
                    "forward progress length: "
                    f"world={total_dist_final:.6f} "
                    f"progress={source_root_motion_progress_path_length_m:.6f}."
                )
        root["pemoin_mixamo_animation_motion_category"] = animation_motion_category
        root["pemoin_configured_motion_policy"] = motion_policy
        root["pemoin_effective_motion_policy"] = effective_motion_policy
        root["pemoin_baked_root_path_length_m"] = float(total_dist_final)
        root["pemoin_baked_root_avg_speed_mps"] = float(avg_speed)
        root["pemoin_mixamo_source_sample_time_seconds_per_output_frame"] = float(dt)
        root["pemoin_mixamo_source_sample_frame_float_step"] = float(source_fps * dt)
        root["pemoin_mixamo_source_sample_frame_float_min"] = float(
            min(sampled_source_frames)
        )
        root["pemoin_mixamo_source_sample_frame_float_max"] = float(
            max(sampled_source_frames)
        )
        root["pemoin_mixamo_source_sample_absolute_progress_frame_min"] = float(
            min(sampled_source_absolute_progress_frames)
        )
        root["pemoin_mixamo_source_sample_absolute_progress_frame_max"] = float(
            max(sampled_source_absolute_progress_frames)
        )
        root["pemoin_mixamo_source_sample_completed_cycles_max"] = int(
            max(sampled_source_completed_cycles)
        )
        char_arm["pemoin_mixamo_source_sample_time_seconds_per_output_frame"] = float(dt)
        char_arm["pemoin_mixamo_source_sample_frame_float_step"] = float(source_fps * dt)
        char_arm["pemoin_mixamo_source_sample_frame_float_min"] = float(
            min(sampled_source_frames)
        )
        char_arm["pemoin_mixamo_source_sample_frame_float_max"] = float(
            max(sampled_source_frames)
        )
        char_arm["pemoin_mixamo_source_sample_absolute_progress_frame_min"] = float(
            min(sampled_source_absolute_progress_frames)
        )
        char_arm["pemoin_mixamo_source_sample_absolute_progress_frame_max"] = float(
            max(sampled_source_absolute_progress_frames)
        )
        char_arm["pemoin_mixamo_source_sample_completed_cycles_max"] = int(
            max(sampled_source_completed_cycles)
        )
        root["pemoin_mixamo_source_root_motion_forward_progress_path_length_m"] = float(
            source_root_motion_progress_path_length_m
        )
        char_arm["pemoin_mixamo_source_root_motion_forward_progress_path_length_m"] = float(
            source_root_motion_progress_path_length_m
        )
        root["pemoin_mixamo_source_root_motion_max_backward_progress_step_m"] = float(
            max_backward_progress_step_m
        )
        char_arm["pemoin_mixamo_source_root_motion_max_backward_progress_step_m"] = float(
            max_backward_progress_step_m
        )
        root["pemoin_mixamo_source_root_motion_max_forward_progress_step_m"] = float(
            max_forward_progress_step_m
        )
        char_arm["pemoin_mixamo_source_root_motion_max_forward_progress_step_m"] = float(
            max_forward_progress_step_m
        )

        baked_root_displacement_world = (
            root_path_positions[-1] - root_path_positions[0]
        ).copy()
        baked_root_displacement_world.z = 0.0
        baked_root_direction_xy = _normalize_xy_or_none(baked_root_displacement_world)
        intended_vs_baked_dot = None
        expected_heading_vs_baked_dot = None
        direction_alignment_check_passed = None
        if (
            expected_heading_world_xy is not None
            and baked_root_direction_xy is not None
        ):
            intended_vs_baked_dot = float(
                np.clip(
                    np.dot(intended_forward_xy, baked_root_direction_xy),
                    -1.0,
                    1.0,
                )
            )
            expected_heading_vs_baked_dot = float(
                np.clip(
                    np.dot(expected_heading_world_xy, baked_root_direction_xy),
                    -1.0,
                    1.0,
                )
            )
            direction_alignment_check_passed = bool(expected_heading_vs_baked_dot >= 0.5)
            log_info(
                "Motion direction parity: "
                f"insertion_basis_xy=({float(intended_forward_xy[0]):+.4f},{float(intended_forward_xy[1]):+.4f}) "
                f"resolved_facing_xy=({float(expected_heading_world_xy[0]):+.4f},{float(expected_heading_world_xy[1]):+.4f}) "
                f"baked_xy=({float(baked_root_direction_xy[0]):+.4f},{float(baked_root_direction_xy[1]):+.4f}) "
                f"insertion_dot={intended_vs_baked_dot:+.4f} "
                f"facing_dot={expected_heading_vs_baked_dot:+.4f}"
            )
        else:
            log_info(
                "Motion direction parity skipped because the desired or baked "
                "horizontal direction is too small to normalize."
            )
        if effective_motion_policy == "animation_root_motion":
            if direction_alignment_check_passed is False:
                raise RuntimeError(
                    "Heading-aligned animation-root-motion bake disagrees with the "
                    "resolved locomotion heading."
                )
            if direction_alignment_check_passed is True:
                log_info(
                    "Heading-aligned animation-root-motion bake matches the resolved "
                    "locomotion heading."
                )
        body_facing_validation_frames = sorted(
            {
                int(bake_s),
                int(min(bake_s + 1, bake_e)),
                int(bake_s + max((bake_e - bake_s) // 2, 0)),
                int(max(bake_e - 1, bake_s)),
                int(bake_e),
            }
        )
        body_facing_validation = _validate_baked_body_facing_parity(
            scene=scene,
            armature_obj=char_arm,
            expected_heading_world_xy=expected_heading_world_xy,
            sample_frames=body_facing_validation_frames,
            raise_on_failure=False,
        )
        if not bool(body_facing_validation["body_facing_check_passed"]):
            correction_deg = -float(
                body_facing_validation["median_body_facing_signed_error_deg"]
            )
            signed_error_samples = np.asarray(
                [
                    float(sample["body_facing_signed_error_deg"])
                    for sample in body_facing_validation["samples"]
                ],
                dtype=np.float32,
            )
            error_spread_deg = float(
                np.max(np.abs(signed_error_samples - np.median(signed_error_samples)))
            )
            if error_spread_deg <= 15.0:
                log_warning(
                    "Applying uniform root-yaw correction from baked torso-facing diagnostics: "
                    f"correction_deg={correction_deg:+.3f} spread_deg={error_spread_deg:.3f}"
                )
                _apply_root_yaw_correction(
                    root=root,
                    frame_numbers=frame_numbers,
                    resolved_root_yaws_world_deg=resolved_root_yaws_world_deg,
                    correction_deg=correction_deg,
                )
                resolved_root_yaw_world_deg = float(resolved_root_yaws_world_deg[0])
                body_facing_validation = _validate_baked_body_facing_parity(
                    scene=scene,
                    armature_obj=char_arm,
                    expected_heading_world_xy=expected_heading_world_xy,
                    sample_frames=body_facing_validation_frames,
                    raise_on_failure=False,
                )
                body_facing_validation["applied_root_yaw_correction_deg"] = float(
                    correction_deg
                )
                body_facing_validation["applied_root_yaw_correction_from_baked_parity"] = True
                body_facing_validation["body_facing_error_spread_deg"] = float(
                    error_spread_deg
                )
            else:
                body_facing_validation["applied_root_yaw_correction_deg"] = 0.0
                body_facing_validation["applied_root_yaw_correction_from_baked_parity"] = False
                body_facing_validation["body_facing_error_spread_deg"] = float(
                    error_spread_deg
                )
        if not bool(body_facing_validation["body_facing_check_passed"]):
            raise RuntimeError(
                "Baked Mixamo body-facing direction disagrees with the resolved heading: "
                f"median_error_deg={float(body_facing_validation['median_body_facing_error_deg']):.3f} "
                f"max_error_deg={float(body_facing_validation['max_body_facing_error_deg']):.3f} "
                f"median_dot={float(body_facing_validation['median_body_facing_vs_expected_dot']):.4f} "
                f"min_dot={float(body_facing_validation['min_body_facing_vs_expected_dot']):.4f}."
            )
        log_info(
            "Body-facing parity: "
            f"median_error_deg={float(body_facing_validation['median_body_facing_error_deg']):.3f} "
            f"max_error_deg={float(body_facing_validation['max_body_facing_error_deg']):.3f} "
            f"median_dot={float(body_facing_validation['median_body_facing_vs_expected_dot']):+.4f} "
            f"min_dot={float(body_facing_validation['min_body_facing_vs_expected_dot']):+.4f}"
        )
        alignment_bone_names = [
            bone_name
            for bone_name in (
                bone_name_map.get(source_hips_bone_name),
                bone_name_map.get("mixamorig:Spine"),
                bone_name_map.get("mixamorig7:Spine"),
                bone_name_map.get("mixamorig:LeftUpLeg"),
                bone_name_map.get("mixamorig7:LeftUpLeg"),
            )
            if bone_name is not None
        ]
        alignment_sample_frames = sorted(
            {
                int(bake_s),
                int(min(bake_s + 1, bake_e)),
                int(bake_e),
            }
        )
        alignment_report = _validate_baked_actor_hierarchy_alignment(
            scene=scene,
            root=root,
            char_arm=char_arm,
            sample_frames=alignment_sample_frames,
            key_bone_names=alignment_bone_names,
        )
        char_arm["pemoin_alignment_validation_sample_count"] = int(
            len(alignment_report["reports"])
        )
        root["pemoin_alignment_validation_sample_count"] = int(
            len(alignment_report["reports"])
        )
        log_info(
            "Baked actor hierarchy alignment validated: "
            f"frames={alignment_sample_frames}"
        )

        # DEBUG Freeze: keep root fixed, no keyframes DEBUG
        # root.location = root_loc0

        # ----------------------------
        # DEBUG: camera-relative pedestrian motion
        # ----------------------------
        cam = bpy.context.scene.camera
        if cam is None:
            log_info("REL DEBUG: scene.camera is None")
        else:

            def _rel_dbg(frame: int) -> None:
                bpy.context.scene.frame_set(frame)
                bpy.context.view_layer.update()
                rel = cam.matrix_world.inverted() @ root.matrix_world
                t = rel.translation
                log_info(f"REL frame={frame:4d} cam^-1*root t=({_vec(t)})")

            for f in (bake_s, bake_s + 10, bake_s + 30, bake_e):
                if f <= bake_e:
                    _rel_dbg(f)

        # ----------------------------
        # 9) Cleanup: delete the imported animation rig objects
        # ----------------------------
        bpy.ops.object.select_all(action="DESELECT")
        for o in imported_anim_objs:
            if o.name in bpy.data.objects:
                o.select_set(True)
        bpy.ops.object.delete()

        log_info("Mixamo character import complete!")
        return {
            "mixamo_asset_root": str(mixamo_spec.asset_root),
            "mixamo_asset_resolved_entry_count": int(asset_diagnostics["resolved_entry_count"]),
            "mixamo_asset_unresolved_entry_count": int(asset_diagnostics["unresolved_entry_count"]),
            "mixamo_asset_diagnostics_path": str(asset_diagnostics["diagnostics_path"]),
            "intended_forward_world_xy": (
                None
                if intended_forward_xy is None
                else [float(intended_forward_xy[0]), float(intended_forward_xy[1])]
            ),
            "insertion_motion_forward_world_xy": (
                None
                if intended_forward_xy is None
                else [float(intended_forward_xy[0]), float(intended_forward_xy[1])]
            ),
            "asset_forward_world_zero_xy": [
                float(asset_forward_world_zero_xy[0]),
                float(asset_forward_world_zero_xy[1]),
            ],
            "asset_forward_resolution_method": str(asset_forward_resolution_method),
            "asset_forward_resolution_confidence": float(
                asset_forward_resolution_confidence
            ),
            "motion_policy_source": "animation_path_category",
            "animation_motion_category": str(animation_motion_category),
            "animation_motion_category_detected": bool(
                detect_mixamo_animation_motion_category(mixamo_spec.animation_fbx)
                is not None
            ),
            "animation_fbx_path": str(mixamo_spec.animation_fbx),
            "cycle_distance_method": str(effective_motion_policy),
            "motion_source": (
                "mixamo_forward_progress"
                if effective_motion_policy == "animation_root_motion"
                else (
                    "camera_trajectory_relative"
                    if effective_motion_policy == "camera_trajectory_relative"
                    else "stationary_spawn"
                )
            ),
            "legacy_camera_trajectory_relative_policy": bool(
                effective_motion_policy == "camera_trajectory_relative"
            ),
            "has_locomotion": bool(effective_motion_policy != "stationary_at_spawn"),
            "configured_motion_policy": str(motion_policy),
            "effective_motion_policy": str(effective_motion_policy),
            "placement_anchor_world": [
                float(trajectory_anchor_arr[0]),
                float(trajectory_anchor_arr[1]),
                float(trajectory_anchor_arr[2]),
            ],
            "placement_spawn_world": [
                float(spawn_arr[0]),
                float(spawn_arr[1]),
                float(spawn_arr[2]),
            ],
            "path_origin_world": [
                float(root_path_start[0]),
                float(root_path_start[1]),
                float(root_path_start[2]),
            ],
            "locomotion_diagnostics": {
                "cycle_horizontal_norm": float(
                    np.linalg.norm(np.asarray(cycle_displacement_world_xy, dtype=np.float32))
                ),
                "usable_delta_count": int(
                    locomotion_diagnostics.get("usable_delta_count", 0)
                ),
                "delta_path_length": float(
                    locomotion_diagnostics.get("delta_path_length", 0.0)
                ),
                "resolved_world_locomotion_axis_xy": (
                    None
                    if world_locomotion_axis_xy is None
                    else [
                        float(world_locomotion_axis_xy[0]),
                        float(world_locomotion_axis_xy[1]),
                    ]
                ),
                "resolved_local_locomotion_axis_xy": (
                    None
                    if local_locomotion_axis_xy is None
                    else [
                        float(local_locomotion_axis_xy[0]),
                        float(local_locomotion_axis_xy[1]),
                    ]
                ),
                "source_heading_local_axis_xy": [
                    float(source_heading_local_axis_xy[0]),
                    float(source_heading_local_axis_xy[1]),
                ],
                "target_heading_local_axis_xy": [
                    float(target_heading_local_axis_xy[0]),
                    float(target_heading_local_axis_xy[1]),
                ],
                "projected_cycle_forward_distance_world_m": float(
                    cycle_world_distance_m
                ),
                "source_root_motion_forward_progress_path_length_m": float(
                    source_root_motion_progress_path_length_m
                ),
                "source_sample_absolute_progress_frame_min": float(
                    min(sampled_source_absolute_progress_frames)
                ),
                "source_sample_absolute_progress_frame_max": float(
                    max(sampled_source_absolute_progress_frames)
                ),
                "source_sample_completed_cycles_max": int(
                    max(sampled_source_completed_cycles)
                ),
                "max_backward_progress_step_m": float(max_backward_progress_step_m),
                "max_forward_progress_step_m": float(max_forward_progress_step_m),
            },
            "resolved_facing_world_xy": (
                [
                    float(expected_heading_world_xy[0]),
                    float(expected_heading_world_xy[1]),
                ]
            ),
            "desired_locomotion_world_xy": [
                float(expected_heading_world_xy[0]),
                float(expected_heading_world_xy[1]),
            ],
            "expected_heading_world_xy": [
                float(expected_heading_world_xy[0]),
                float(expected_heading_world_xy[1]),
            ],
            "resolved_root_yaw_world_deg": float(resolved_root_yaw_world_deg),
            "resolved_root_yaw_world_deg_start": float(resolved_root_yaws_world_deg[0]),
            "resolved_root_yaw_world_deg_end": float(resolved_root_yaws_world_deg[-1]),
            "baked_root_start_world": [
                float(baked_root_start_world_arr[0]),
                float(baked_root_start_world_arr[1]),
                float(baked_root_start_world_arr[2]),
            ],
            "baked_root_end_world": [
                float(baked_root_end_world_arr[0]),
                float(baked_root_end_world_arr[1]),
                float(baked_root_end_world_arr[2]),
            ],
            "frame0_root_to_spawn_delta_m": float(frame0_root_to_spawn_delta_m),
            "nearest_trajectory_frame_index_to_placement_anchor": int(
                nearest_trajectory_frame_index
            ),
            "nearest_trajectory_frame_xy_distance_to_placement_anchor_m": float(
                nearest_trajectory_frame_xy_distance_m
            ),
            "trajectory_heading_world_deg_samples": sampled_base_heading_world_deg,
            "baked_root_displacement_world_xy": [
                float(baked_root_displacement_world[0]),
                float(baked_root_displacement_world[1]),
            ],
            "baked_root_path_length_m": float(total_dist_final),
            "baked_root_avg_speed_mps": float(avg_speed),
            "baked_root_direction_world_xy": (
                None
                if baked_root_direction_xy is None
                else [float(baked_root_direction_xy[0]), float(baked_root_direction_xy[1])]
            ),
            "insertion_basis_vs_baked_dot": intended_vs_baked_dot,
            "desired_locomotion_vs_baked_dot": expected_heading_vs_baked_dot,
            "intended_vs_baked_dot": intended_vs_baked_dot,
            "expected_heading_vs_baked_dot": expected_heading_vs_baked_dot,
            "direction_alignment_dot": expected_heading_vs_baked_dot,
            "direction_alignment_check_passed": direction_alignment_check_passed,
            "facing_walk_heading_locked": bool(
                effective_motion_policy == "animation_root_motion"
            ),
            "body_facing_validation": body_facing_validation,
            "measured_body_facing_world_xy": list(
                body_facing_validation["samples"][0]["measured_body_facing_world_xy"]
            ),
            "measured_body_facing_yaw_world_deg": float(
                math.degrees(
                    math.atan2(
                        float(body_facing_validation["samples"][0]["measured_body_facing_world_xy"][1]),
                        float(body_facing_validation["samples"][0]["measured_body_facing_world_xy"][0]),
                    )
                )
            ),
            "body_facing_resolution_method": str(
                body_facing_validation["samples"][0]["body_facing_resolution_method"]
            ),
            "body_facing_resolution_confidence": float(
                body_facing_validation["samples"][0]["body_facing_resolution_confidence"]
            ),
            "body_facing_vs_expected_dot": float(
                body_facing_validation["samples"][0]["body_facing_vs_expected_dot"]
            ),
            "body_facing_error_deg": float(
                body_facing_validation["samples"][0]["body_facing_error_deg"]
            ),
            "body_facing_check_passed": bool(
                body_facing_validation["body_facing_check_passed"]
            ),
        }


def _coerce_metadata(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            try:
                return _coerce_metadata(raw.item())
            except Exception:
                return {}
        try:
            return _coerce_metadata(raw.tolist())
        except Exception:
            return {}
    if isinstance(raw, (list, tuple)):
        try:
            return dict(raw)
        except Exception:
            return {}
    if hasattr(raw, "items"):
        try:
            return dict(raw.items())
        except Exception:
            return {}
    return {}


def _resolve_intrinsics_resolution(
    metadata: dict,
    run_dir: Path,
    frame_indices: np.ndarray,
) -> tuple[int, int]:
    width = metadata.get("width")
    height = metadata.get("height")
    if width is not None and height is not None:
        log_info(f"Using intrinsics metadata width/height: {width}x{height}")
        return int(width), int(height)

    ref = (
        metadata.get("reference_resolution")
        or metadata.get("image_size")
        or metadata.get("resolution")
    )
    if isinstance(ref, (list, tuple)) and len(ref) == 2:
        height = int(ref[0])
        width = int(ref[1])
        log_info(f"Using intrinsics metadata reference_resolution: {width}x{height}")
        return width, height

    frame_dir = run_dir / "standard" / "frames"
    if frame_dir.exists():
        for frame_idx in frame_indices.tolist():
            frame_path = frame_dir / f"{int(frame_idx):06d}.png"
            if frame_path.exists():
                image = bpy.data.images.load(str(frame_path))
                width, height = int(image.size[0]), int(image.size[1])
                bpy.data.images.remove(image)
                log_warning(
                    "Intrinsics metadata missing resolution; "
                    f"using frame size from {frame_path.name} ({width}x{height})."
                )
                log_info(f"Using frame image resolution: {width}x{height}")
                return width, height

    raise ValueError(
        "Intrinsics metadata missing resolution. "
        "Expected width/height or reference_resolution in intrinsics.npz metadata."
    )


def _load_reference_frame_resolution(
    run_dir: Path,
    frame_indices: np.ndarray,
) -> tuple[int, int] | None:
    frame_dir = run_dir / "standard" / "frames"
    if not frame_dir.exists():
        return None
    for frame_idx in frame_indices.tolist():
        frame_path = frame_dir / f"{int(frame_idx):06d}.png"
        if not frame_path.exists():
            continue
        image = bpy.data.images.load(str(frame_path))
        try:
            return int(image.size[0]), int(image.size[1])
        finally:
            bpy.data.images.remove(image)
    return None


def load_intrinsics(
    run_dir: Path,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    path = run_dir / "standard" / "intrinsics" / "intrinsics.npz"
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        matrix = np.asarray(data["matrix"], dtype=np.float32)
        metadata = (
            _coerce_metadata(data["metadata"]) if "metadata" in data.files else {}
        )

    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid intrinsics matrix shape: {matrix.shape}")

    width, height = _resolve_intrinsics_resolution(metadata, run_dir, frame_indices)
    frame_size = _load_reference_frame_resolution(run_dir, frame_indices)
    if frame_size is not None and frame_size != (width, height):
        raise ValueError(
            "Intrinsics metadata resolution does not match the saved frame dimensions: "
            f"intrinsics={width}x{height}, frame={frame_size[0]}x{frame_size[1]}."
        )
    normalized_matrix, normalized_metadata, _ = validate_and_normalize_intrinsics(
        matrix,
        metadata,
        frame_shape=(height, width),
        allow_principal_point_fallback=False,
        fail_on_heuristic=True,
    )
    return normalized_matrix, width, height, normalized_metadata


def set_camera_intrinsics(
    camera: bpy.types.Camera,
    matrix: np.ndarray,
    width: int,
    height: int,
) -> BlenderCameraSolution:
    """Set camera intrinsics from a pinhole matrix with explicit parity validation."""
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    solution = solve_blender_camera_for_intrinsics(
        matrix,
        width=width,
        height=height,
    )
    scene.render.pixel_aspect_x = solution.pixel_aspect_x
    scene.render.pixel_aspect_y = solution.pixel_aspect_y
    camera.sensor_width = solution.sensor_width_mm
    camera.sensor_height = solution.sensor_height_mm
    camera.sensor_fit = solution.sensor_fit
    camera.lens = solution.lens_mm
    camera.shift_x = solution.shift_x
    camera.shift_y = solution.shift_y
    return solution


def set_linear_interpolation(obj: bpy.types.Object):
    """Set linear interpolation for all keyframes (Blender 4.0+ compatible)."""
    if not obj.animation_data:
        return

    action = obj.animation_data.action
    if not action:
        return

    # Blender 4.0+ uses action.slots and action.layers
    if hasattr(action, "fcurves"):
        # Old API (Blender < 4.0)
        fcurves = action.fcurves
    elif hasattr(action, "layers"):
        # New API (Blender 4.0+)
        fcurves = []
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbag"):
                    for channelbag in strip.channelbags:
                        fcurves.extend(channelbag.fcurves)
    else:
        return

    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def create_animated_camera(
    c2w_matrices: np.ndarray,
    frame_indices: np.ndarray,
    intrinsics_matrix: np.ndarray,
    width: int,
    height: int,
    camera_name: str = "TrajectoryCamera",
) -> tuple[bpy.types.Object, BlenderCameraSolution]:
    """Create a camera animated along the trajectory."""
    camera_data = bpy.data.cameras.new(name=camera_name)
    camera_obj = bpy.data.objects.new(camera_name, camera_data)
    bpy.context.collection.objects.link(camera_obj)

    parity_solution = set_camera_intrinsics(
        camera_data,
        intrinsics_matrix,
        width,
        height,
    )

    # Set frame range
    scene = bpy.context.scene
    scene.frame_start = int(frame_indices.min())
    scene.frame_end = int(frame_indices.max())

    # Keyframe each pose
    for c2w, frame_idx in zip(c2w_matrices, frame_indices):
        camera_obj.matrix_world = Matrix(c2w.tolist())
        camera_obj.keyframe_insert(data_path="location", frame=int(frame_idx))
        camera_obj.keyframe_insert(data_path="rotation_euler", frame=int(frame_idx))

    # Set linear interpolation
    set_linear_interpolation(camera_obj)

    scene.camera = camera_obj
    return camera_obj, parity_solution


def _iter_descendants(root: bpy.types.Object) -> Iterable[bpy.types.Object]:
    stack = list(root.children)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(list(node.children))


def validate_plane_camera_relationship(
    plane_spec: RoadPlaneSpec,
    camera_position: np.ndarray,
) -> None:
    """Validate that camera is above the road plane.

    Args:
        plane_spec: Road plane specification
        camera_position: Camera position (3,)

    Raises:
        ValueError: If camera is underground (data error)
    """
    # Compute signed distance from camera to plane
    # Distance = n·cam_pos + d
    distance = np.dot(plane_spec.normal, camera_position) + plane_spec.offset

    if distance < 0:
        raise ValueError(
            f"Camera underground at frame {plane_spec.frame_index}: "
            f"distance={distance:.3f}m (camera at {camera_position}, "
            f"plane normal={plane_spec.normal}, offset={plane_spec.offset})"
        )

    if distance < 0.5:
        log_warning(
            f"Unusually low camera at frame {plane_spec.frame_index}: {distance:.3f}m"
        )
    elif distance > 3.0:
        log_warning(
            f"Unusually high camera at frame {plane_spec.frame_index}: {distance:.3f}m"
        )


def add_trajectory_cubes(
    c2w: np.ndarray,
    frame_indices: np.ndarray,
    spec: TrajectorySpec,
    collection: bpy.types.Collection,
) -> Sequence[bpy.types.Object]:
    """Add cubes at each trajectory position.

    Args:
        c2w: Camera-to-world matrices (N, 4, 4)
        frame_indices: Frame indices (N,)
        spec: Trajectory visualization config
        collection: Target collection

    Returns:
        Created cube objects
    """
    objs: list[bpy.types.Object] = []

    for idx, frame_idx in enumerate(frame_indices.tolist()):
        location = c2w[idx, :3, 3]  # Extract translation

        bpy.ops.mesh.primitive_cube_add(
            size=spec.cube_size,
            location=(float(location[0]), float(location[1]), float(location[2])),
        )
        obj = bpy.context.active_object
        obj.name = f"traj_cube_{frame_idx:06d}"

        # Move to target collection
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        collection.objects.link(obj)
        objs.append(obj)

    return objs


def add_road_plane(
    spec: RoadPlaneSpec,
    collection: bpy.types.Collection,
    *,
    name_prefix: str = "road_plane",
) -> bpy.types.Object:
    """Add road plane mesh with proper orientation and material.

    Args:
        spec: Road plane configuration
        collection: Target collection

    Returns:
        Created plane object
    """
    # Create plane at center
    bpy.ops.mesh.primitive_plane_add(
        size=1.0,  # Will be scaled by spec.scale_u/v
        location=(float(spec.center[0]), float(spec.center[1]), float(spec.center[2])),
    )
    plane_obj = bpy.context.active_object
    plane_obj.name = f"{name_prefix}_{spec.frame_index:06d}"

    # Orient plane to match road plane normal
    plane_obj.rotation_mode = "QUATERNION"
    plane_obj.rotation_quaternion = compute_rotation_to_normal(spec.normal)

    # Scale plane
    plane_obj.scale = (float(spec.scale_u), float(spec.scale_v), 1.0)

    # Move to target collection
    for coll in list(plane_obj.users_collection):
        coll.objects.unlink(plane_obj)
    collection.objects.link(plane_obj)

    # Apply material
    material = create_plane_material(
        name=f"{name_prefix}_mat_{spec.frame_index:06d}",
        color=spec.material_color,
        alpha=spec.material_alpha,
    )
    if plane_obj.data.materials:
        plane_obj.data.materials[0] = material
    else:
        plane_obj.data.materials.append(material)

    return plane_obj


def create_plane_material(
    name: str,
    color: Vec3,
    alpha: float,
) -> bpy.types.Material:
    """Create Principled BSDF material with transparency (Blender 5+).

    Args:
        name: Material name
        color: Base color RGB (0-1 range)
        alpha: Transparency (0=transparent, 1=opaque)

    Returns:
        Material with shader nodes configured
    """
    material = bpy.data.materials.new(name=name)
    # Materials use nodes by default in modern Blender
    nodes = material.node_tree.nodes
    nodes.clear()

    # Create Principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Metallic"].default_value = 0.2
    bsdf.inputs["Roughness"].default_value = 0.8

    # Create output
    output = nodes.new(type="ShaderNodeOutputMaterial")
    material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Enable transparency (Blender 5: shadow_method removed)
    material.blend_method = "BLEND"

    return material


def compute_rotation_to_normal(normal: np.ndarray) -> tuple[float, float, float, float]:
    """Compute quaternion to rotate Z-axis to align with normal.

    Args:
        normal: Target normal vector (3,)

    Returns:
        Quaternion (w, x, y, z)
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normal = normal / np.linalg.norm(normal)

    # Handle parallel case
    if np.linalg.norm(np.cross(normal, z_axis)) < 1e-6:
        if np.dot(normal, z_axis) < 0:
            return (0.0, 1.0, 0.0, 0.0)  # 180° rotation around X
        return (1.0, 0.0, 0.0, 0.0)  # No rotation

    # Compute axis-angle rotation
    rotation_axis = np.cross(z_axis, normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(z_axis, normal))

    # Convert to quaternion
    w = np.cos(rotation_angle / 2)
    x, y, z = rotation_axis * np.sin(rotation_angle / 2)
    return (float(w), float(x), float(y), float(z))


# ---------------------------
# Import helpers
# ---------------------------


def _import_fbx(
    filepath: Path, *, global_scale: float = 1.0, use_anim: bool = True
) -> set[bpy.types.Object]:
    """Import FBX; return newly created objects."""
    before = set(bpy.data.objects)
    fp = str(filepath.resolve())

    # Blender 5.x has a newer importer operator under bpy.ops.wm
    if hasattr(bpy.ops.wm, "fbx_import"):
        bpy.ops.wm.fbx_import(filepath=fp, global_scale=global_scale, use_anim=use_anim)
    else:
        # Legacy fallback
        bpy.ops.import_scene.fbx(
            filepath=fp, global_scale=global_scale, use_anim=use_anim
        )

    after = set(bpy.data.objects)
    return after - before


def _iter_object_family(objects: Iterable[bpy.types.Object]) -> list[bpy.types.Object]:
    seen: set[int] = set()
    ordered: list[bpy.types.Object] = []
    stack = list(objects)
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        ordered.append(obj)
        for child in getattr(obj, "children", ()):
            stack.append(child)
    return ordered


def _write_mixamo_asset_diagnostics(
    *,
    run_dir: Path,
    diagnostics: Mapping[str, Any],
) -> Path:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "mixamo_asset_diagnostics.json"
    json_path.write_text(json.dumps(dict(diagnostics), indent=2), encoding="utf-8")
    return json_path


def _trim_and_shift_action(
    action: bpy.types.Action,
    *,
    source_start: int,
    source_end: int,
    target_start: int = 1,
) -> bpy.types.Action:
    trimmed = action.copy()
    frame_offset = int(target_start) - int(source_start)
    channelbags: list[Any] = []
    for layer in getattr(trimmed, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                channelbags.append(channelbag)
    if not channelbags:
        raise RuntimeError(
            f"Copied action '{trimmed.name}' has no channelbags in Blender's layered action API."
        )
    for channelbag in channelbags:
        for fcurve in list(channelbag.fcurves):
            removable_indices = [
                idx
                for idx, point in enumerate(fcurve.keyframe_points)
                if float(point.co.x) < float(source_start) - 1e-6
                or float(point.co.x) > float(source_end) + 1e-6
            ]
            for idx in reversed(removable_indices):
                fcurve.keyframe_points.remove(fcurve.keyframe_points[idx])
            for point in fcurve.keyframe_points:
                point.co.x = float(point.co.x) + float(frame_offset)
                point.handle_left.x = float(point.handle_left.x) + float(frame_offset)
                point.handle_right.x = float(point.handle_right.x) + float(frame_offset)
            if not fcurve.keyframe_points:
                channelbag.fcurves.remove(fcurve)
            else:
                fcurve.keyframe_points.sort()
                with suppress(Exception):
                    fcurve.keyframe_points.handles_recalc()
                fcurve.update()
    trimmed.frame_start = float(target_start)
    trimmed.frame_end = float(target_start + max(0, int(source_end) - int(source_start)))
    return trimmed


def _copy_object_family_for_export(
    root_object: bpy.types.Object,
    *,
    collection_name: str,
) -> tuple[bpy.types.Collection, dict[str, bpy.types.Object], list[bpy.types.Action]]:
    scene = bpy.context.scene
    export_collection = bpy.data.collections.new(collection_name)
    scene.collection.children.link(export_collection)
    family = list(reversed(_iter_object_family([root_object])))
    object_map: dict[str, bpy.types.Object] = {}
    copied_actions: list[bpy.types.Action] = []
    for obj in family:
        duplicate = obj.copy()
        if obj.data is not None:
            with suppress(Exception):
                duplicate.data = obj.data.copy()
        export_collection.objects.link(duplicate)
        object_map[obj.name] = duplicate
    for obj in family:
        duplicate = object_map[obj.name]
        if obj.parent is not None:
            duplicate.parent = object_map.get(obj.parent.name)
            duplicate.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
        if obj.animation_data is not None:
            duplicate.animation_data_create()
            if obj.animation_data.action is not None:
                duplicate.animation_data.action = obj.animation_data.action.copy()
                copied_actions.append(duplicate.animation_data.action)
            if duplicate.animation_data.nla_tracks:
                for track in list(duplicate.animation_data.nla_tracks):
                    duplicate.animation_data.nla_tracks.remove(track)
            duplicate.animation_data.use_nla = False
    for original_name, duplicate in object_map.items():
        original = bpy.data.objects.get(original_name)
        if original is None or duplicate.type != "MESH":
            continue
        for modifier in duplicate.modifiers:
            if modifier.type != "ARMATURE":
                continue
            target = getattr(modifier, "object", None)
            if target is None:
                continue
            modifier.object = object_map.get(target.name, target)
    return export_collection, object_map, copied_actions


def _remove_export_family(
    collection: bpy.types.Collection | None,
    objects: Iterable[bpy.types.Object],
    actions: Iterable[bpy.types.Action],
) -> None:
    for obj in list(objects):
        obj_data = getattr(obj, "data", None)
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        if obj_data is None:
            continue
        with suppress(Exception):
            if obj.type == "MESH" and obj_data.users == 0:
                bpy.data.meshes.remove(obj_data)
            elif obj.type == "ARMATURE" and obj_data.users == 0:
                bpy.data.armatures.remove(obj_data)
    for action in list(actions):
        with suppress(Exception):
            if action.users == 0:
                bpy.data.actions.remove(action)
    if collection is not None and collection.name in bpy.data.collections:
        with suppress(Exception):
            bpy.data.collections.remove(collection)


def _iter_material_images(
    objects: Iterable[bpy.types.Object],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in getattr(obj, "material_slots", ()):
            material = getattr(slot, "material", None)
            if material is None or not _material_uses_nodes(material):
                continue
            node_tree = getattr(material, "node_tree", None)
            if node_tree is None:
                continue
            for node in node_tree.nodes:
                if getattr(node, "type", None) != "TEX_IMAGE":
                    continue
                image = getattr(node, "image", None)
                if image is None:
                    continue
                key = (material.name_full, image.name_full)
                if key in seen:
                    continue
                seen.add(key)
                resolved_path = ""
                if str(getattr(image, "filepath", "") or "").strip():
                    with suppress(Exception):
                        resolved_path = bpy.path.abspath(
                            image.filepath,
                            library=getattr(image, "library", None),
                        )
                packed = bool(
                    getattr(image, "packed_file", None) is not None
                    or len(getattr(image, "packed_files", ())) > 0
                )
                entries.append(
                    {
                        "material_name": material.name_full,
                        "image_name": image.name_full,
                        "resolved_path": str(resolved_path),
                        "packed": packed,
                        "exists_on_disk": bool(resolved_path and Path(resolved_path).exists()),
                    }
                )
    return entries


def _validate_embeddable_export_images(objects: Iterable[bpy.types.Object]) -> dict[str, Any]:
    image_entries = _iter_material_images(objects)
    missing = [
        entry
        for entry in image_entries
        if not bool(entry["packed"]) and not bool(entry["exists_on_disk"])
    ]
    if missing:
        missing_desc = ", ".join(
            f"{entry['material_name']}:{entry['image_name']}" for entry in missing
        )
        raise RuntimeError(
            "Cannot export FBX with embedded textures because some material images are "
            f"neither packed nor resolvable on disk: {missing_desc}."
        )
    return {
        "image_entry_count": int(len(image_entries)),
        "packed_image_count": int(sum(1 for entry in image_entries if entry["packed"])),
        "file_backed_image_count": int(
            sum(1 for entry in image_entries if entry["exists_on_disk"])
        ),
        "images": image_entries,
    }


def _build_export_root_motion_action(
    root_object: bpy.types.Object,
    *,
    source_start: int,
    source_end: int,
    target_start: int = 1,
) -> bpy.types.Action:
    scene = bpy.context.scene
    sampled_world: list[Matrix] = []
    for frame in range(int(source_start), int(source_end) + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        sampled_world.append(root_object.matrix_world.copy())
    if not sampled_world:
        raise RuntimeError("FBX export normalization sampled no root transforms.")
    base_world = sampled_world[0].copy()
    normalized_action = bpy.data.actions.new(name=f"{root_object.name}_FBXRootMotion")
    root_object.animation_data_create()
    root_object.animation_data.action = normalized_action
    root_object.rotation_mode = "XYZ"
    for index, world_matrix in enumerate(sampled_world, start=int(target_start)):
        normalized = base_world.inverted() @ world_matrix
        root_object.location = normalized.to_translation()
        root_object.rotation_euler = normalized.to_euler("XYZ")
        root_object.keyframe_insert(data_path="location", frame=index)
        root_object.keyframe_insert(data_path="rotation_euler", frame=index)
    channelbags: list[Any] = []
    for layer in getattr(normalized_action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                channelbags.append(channelbag)
    if not channelbags:
        raise RuntimeError(
            f"Normalized export action '{normalized_action.name}' has no channelbags in Blender's layered action API."
        )
    for channelbag in channelbags:
        for fcurve in channelbag.fcurves:
            for point in fcurve.keyframe_points:
                point.interpolation = "LINEAR"
            with suppress(Exception):
                fcurve.keyframe_points.handles_recalc()
            fcurve.update()
    return normalized_action


def _scale_action_frame_timing(
    action: bpy.types.Action,
    *,
    scale: float,
    origin_frame: float = 1.0,
) -> None:
    if not np.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError(f"Invalid action timing scale: {scale}")
    channelbags: list[Any] = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                channelbags.append(channelbag)
    if not channelbags:
        raise RuntimeError(
            f"Action '{action.name}' has no channelbags in Blender's layered action API."
        )
    for channelbag in channelbags:
        for fcurve in channelbag.fcurves:
            for point in fcurve.keyframe_points:
                point.co.x = float(origin_frame) + (
                    float(point.co.x) - float(origin_frame)
                ) * float(scale)
                point.handle_left.x = float(origin_frame) + (
                    float(point.handle_left.x) - float(origin_frame)
                ) * float(scale)
                point.handle_right.x = float(origin_frame) + (
                    float(point.handle_right.x) - float(origin_frame)
                ) * float(scale)
            fcurve.keyframe_points.sort()
            with suppress(Exception):
                fcurve.keyframe_points.handles_recalc()
            fcurve.update()
    action.frame_start = float(origin_frame)
    action.frame_end = float(origin_frame) + (
        float(action.frame_end) - float(origin_frame)
    ) * float(scale)


def _export_fbx_embedding_manifest_path(run_dir: Path) -> tuple[Path, Path, Path]:
    export_dir = ResourceStore.blender_artifact_dir_for(
        run_dir,
        "fbx_exports",
        create=True,
    )
    artifact_path = export_dir / "character_root_motion.fbx"
    manifest_path = export_dir / "character_root_motion.export.json"
    convenience_path = Path(run_dir) / "character_root_motion.fbx"
    return artifact_path, manifest_path, convenience_path


def export_mixamo_root_motion_fbx(
    *,
    spec: SceneSpec,
    actor_name: str,
    animation_fbx_path: Path,
) -> dict[str, Any]:
    with log_scope("FBX Export"):
        actor_root = bpy.data.objects.get(actor_name)
        if actor_root is None:
            raise RuntimeError(f"Cannot export FBX because actor root '{actor_name}' was not found.")
        actor_armature = next(
            (child for child in actor_root.children if child.type == "ARMATURE"),
            None,
        )
        if actor_armature is None:
            raise RuntimeError(
                f"Cannot export FBX because actor root '{actor_name}' has no child armature."
            )
        bake_start = int(round(float(actor_root.get("pemoin_mixamo_bake_start_frame", 1.0))))
        bake_end = int(
            round(
                float(
                    actor_root.get(
                        "pemoin_mixamo_bake_end_frame",
                        bake_start,
                    )
                )
            )
        )
        cycle_frames = float(actor_root.get("pemoin_mixamo_cycle_len_frames", 0.0))
        if not np.isfinite(cycle_frames) or cycle_frames <= 0.0:
            raise RuntimeError("Cannot export FBX because Mixamo cycle metadata is missing.")
        clip_frame_count = max(2, int(round(cycle_frames)) + 1)
        clip_end = min(bake_end, bake_start + clip_frame_count - 1)
        if clip_end <= bake_start:
            raise RuntimeError(
                "Cannot export FBX because the single-clip frame window is empty."
            )
        scene = bpy.context.scene
        original_frame_start = int(scene.frame_start)
        original_frame_end = int(scene.frame_end)
        original_render_fps = int(scene.render.fps)
        original_render_fps_base = float(scene.render.fps_base)
        source_scene_fps = float(
            actor_root.get(
                "pemoin_mixamo_scene_fps",
                spec.mixamo_scene_fps
                if spec.mixamo_scene_fps is not None
                else _effective_scene_fps(scene),
            )
        )
        if not np.isfinite(source_scene_fps) or source_scene_fps <= 0.0:
            raise RuntimeError(
                "Cannot export FBX because the baked Mixamo scene FPS is invalid."
            )
        export_fps = float(getattr(spec, "mixamo_export_fps", 30.0))
        if not np.isfinite(export_fps) or export_fps <= 0.0:
            raise RuntimeError(
                f"Cannot export FBX because mixamo_export_fps is invalid: {export_fps}"
            )
        clip_duration_seconds = compute_clip_duration_seconds(
            bake_start,
            clip_end,
            source_scene_fps,
        )
        export_frame_count = compute_export_frame_count(
            clip_duration_seconds,
            export_fps,
        )
        export_collection = None
        object_map: dict[str, bpy.types.Object] = {}
        copied_actions: list[bpy.types.Action] = []
        export_artifact_path, export_manifest_path, convenience_path = _export_fbx_embedding_manifest_path(
            spec.run_dir
        )
        if export_artifact_path.exists():
            export_artifact_path.unlink()
        if export_manifest_path.exists():
            export_manifest_path.unlink()
        if convenience_path.exists():
            convenience_path.unlink()
        try:
            export_collection, object_map, copied_actions = _copy_object_family_for_export(
                actor_root,
                collection_name=f"{actor_name}_FBXExport",
            )
            export_root = object_map[actor_root.name]
            export_armature = object_map[actor_armature.name]
            export_objects = list(object_map.values())
            image_diagnostics = _validate_embeddable_export_images(export_objects)
            if export_armature.animation_data is None or export_armature.animation_data.action is None:
                raise RuntimeError(
                    "Cannot export FBX because the duplicated armature has no baked action."
                )
            trimmed_armature_action = _trim_and_shift_action(
                export_armature.animation_data.action,
                source_start=bake_start,
                source_end=clip_end,
                target_start=1,
            )
            copied_actions.append(trimmed_armature_action)
            export_armature.animation_data.action = trimmed_armature_action
            export_root_action = _build_export_root_motion_action(
                export_root,
                source_start=bake_start,
                source_end=clip_end,
                target_start=1,
            )
            copied_actions.append(export_root_action)
            source_frame_span = max(1, int(clip_end - bake_start))
            export_frame_span = max(1, int(export_frame_count - 1))
            export_time_scale = float(export_frame_span) / float(source_frame_span)
            _scale_action_frame_timing(
                trimmed_armature_action,
                scale=export_time_scale,
                origin_frame=1.0,
            )
            _scale_action_frame_timing(
                export_root_action,
                scale=export_time_scale,
                origin_frame=1.0,
            )
            scene.frame_start = 1
            scene.frame_end = int(export_frame_count)
            scene.render.fps = max(1, int(round(export_fps)))
            scene.render.fps_base = float(scene.render.fps) / float(export_fps)
            scene.frame_set(scene.frame_start)
            bpy.context.view_layer.update()
            bpy.ops.object.select_all(action="DESELECT")
            for obj in export_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = export_root
            bpy.ops.export_scene.fbx(
                filepath=str(export_artifact_path),
                use_selection=True,
                object_types={"EMPTY", "ARMATURE", "MESH"},
                global_scale=1.0,
                apply_unit_scale=True,
                use_space_transform=True,
                bake_space_transform=False,
                add_leaf_bones=False,
                use_armature_deform_only=False,
                bake_anim=True,
                bake_anim_use_all_bones=True,
                bake_anim_use_nla_strips=False,
                bake_anim_use_all_actions=False,
                bake_anim_force_startend_keying=True,
                bake_anim_step=1.0,
                bake_anim_simplify_factor=0.0,
                path_mode="COPY",
                embed_textures=True,
                axis_forward="-Z",
                axis_up="Y",
            )
            if not export_artifact_path.exists() or export_artifact_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"FBX export did not produce a usable file at {export_artifact_path}."
                )
            shutil.copy2(export_artifact_path, convenience_path)
            manifest = {
                "schema_version": 2,
                "artifact_path": str(export_artifact_path),
                "convenience_copy_path": str(convenience_path),
                "actor_name": str(actor_name),
                "animation_fbx_path": str(animation_fbx_path),
                "motion_policy": "single_source_clip_root_motion",
                "frame_start": int(scene.frame_start),
                "frame_end": int(scene.frame_end),
                "frame_count": int(scene.frame_end - scene.frame_start + 1),
                "export_fps": float(export_fps),
                "clip_duration_seconds": float(clip_duration_seconds),
                "source_bake_frame_start": int(bake_start),
                "source_bake_frame_end": int(clip_end),
                "source_cycle_frames_output": float(cycle_frames),
                "source_animation_fps": float(actor_root.get("pemoin_mixamo_source_fps", spec.mixamo_source_fps)),
                "scene_fps": float(source_scene_fps),
                "animation_motion_category": str(
                    actor_root.get("pemoin_mixamo_animation_motion_category", "unknown")
                ),
                "configured_motion_policy": str(
                    actor_root.get("pemoin_configured_motion_policy", "unknown")
                ),
                "effective_motion_policy": str(
                    actor_root.get("pemoin_effective_motion_policy", "unknown")
                ),
                "texture_embedding_mode": "copy_embed",
                "exporter": {
                    "operator": "bpy.ops.export_scene.fbx",
                    "axis_forward": "-Z",
                    "axis_up": "Y",
                    "add_leaf_bones": False,
                    "bake_space_transform": False,
                    "bake_anim_use_nla_strips": False,
                    "bake_anim_use_all_actions": False,
                    "embed_textures": True,
                    "path_mode": "COPY",
                },
                "image_diagnostics": image_diagnostics,
            }
            export_manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            log_info(
                "Exported reusable character FBX: "
                f"artifact={export_artifact_path} convenience={convenience_path}"
            )
            return manifest
        finally:
            scene.frame_start = original_frame_start
            scene.frame_end = original_frame_end
            scene.render.fps = original_render_fps
            scene.render.fps_base = original_render_fps_base
            scene.frame_set(original_frame_start)
            bpy.context.view_layer.update()
            _remove_export_family(export_collection, object_map.values(), copied_actions)


def _material_uses_nodes(material: Any) -> bool:
    try:
        return bool(material is not None and material.use_nodes)
    except Exception:
        return False


def _set_image_colorspace(image: Any, colorspace: str) -> None:
    settings = getattr(image, "colorspace_settings", None)
    if settings is None:
        return
    try:
        settings.name = str(colorspace)
    except Exception:
        return


def _first_material_link_to_input(
    links: Any,
    node: Any,
    socket_name: str,
) -> Any | None:
    for link in list(links):
        if getattr(link, "to_node", None) != node:
            continue
        to_socket = getattr(link, "to_socket", None)
        if to_socket is None or str(getattr(to_socket, "name", "")) != str(socket_name):
            continue
        return link
    return None


def _replace_node_input_link(
    links: Any,
    *,
    to_node: Any,
    to_socket_name: str,
    from_socket: Any,
) -> bool:
    if from_socket is None:
        return False
    for link in list(links):
        if getattr(link, "to_node", None) != to_node:
            continue
        to_socket = getattr(link, "to_socket", None)
        if to_socket is None or str(getattr(to_socket, "name", "")) != str(to_socket_name):
            continue
        try:
            links.remove(link)
        except Exception:
            continue
    try:
        links.new(from_socket, to_node.inputs[to_socket_name])
    except Exception:
        return False
    return True


def _clear_node_input_links(
    links: Any,
    *,
    to_node: Any,
    to_socket_name: str,
) -> int:
    removed = 0
    for link in list(links):
        if getattr(link, "to_node", None) != to_node:
            continue
        to_socket = getattr(link, "to_socket", None)
        if to_socket is None or str(getattr(to_socket, "name", "")) != str(to_socket_name):
            continue
        try:
            links.remove(link)
            removed += 1
        except Exception:
            continue
    return removed


def _node_output_socket(node: Any, *names: str) -> Any | None:
    outputs = getattr(node, "outputs", None)
    if outputs is None:
        return None
    for name in names:
        try:
            return outputs[name]
        except Exception:
            continue
    try:
        sockets = list(outputs)
    except Exception:
        sockets = []
    for socket in sockets:
        socket_name = str(getattr(socket, "name", ""))
        if socket_name in names:
            return socket
    return None


def _material_image_semantic(image: Any) -> str | None:
    candidates = [
        str(getattr(image, "filepath", "") or ""),
        str(getattr(image, "filepath_raw", "") or ""),
        str(getattr(image, "name", "") or ""),
    ]
    text = " ".join(Path(item).name.lower() for item in candidates if item)
    if not text:
        return None
    if any(token in text for token in ("basecolor", "base_color", "albedo", "diffuse")):
        return "base_color"
    if "normal" in text:
        return "normal"
    if "roughness" in text:
        return "roughness"
    if "gloss" in text:
        return "glossiness"
    if any(token in text for token in ("specular", "spec")):
        return "specular"
    if any(token in text for token in ("opacity", "alpha", "transparency")):
        return "alpha"
    return None


def _material_node_semantic(node: Any) -> str | None:
    image = getattr(node, "image", None)
    semantic = _material_image_semantic(image) if image is not None else None
    if semantic is not None:
        return semantic
    name = str(getattr(node, "name", "") or "").lower()
    if "normal" in name:
        return "normal"
    if "roughness" in name:
        return "roughness"
    if "gloss" in name:
        return "glossiness"
    if "spec" in name:
        return "specular"
    if "alpha" in name or "opacity" in name:
        return "alpha"
    return None


def _ensure_material_node(
    node_tree: Any,
    *,
    node_type: str,
    name: str,
    location: tuple[float, float] | None = None,
) -> Any:
    existing = node_tree.nodes.get(name)
    if existing is not None:
        return existing
    node = node_tree.nodes.new(type=node_type)
    node.name = name
    if location is not None:
        try:
            node.location = location
        except Exception:
            pass
    return node


def _normalize_mixamo_material_graphs(
    materials: Mapping[str, Any],
    *,
    material_policy: str = "preserve_base_alpha_normal",
) -> dict[str, int]:
    normalized_material_count = 0
    glossiness_inverted_count = 0
    alpha_link_count = 0
    roughness_link_count = 0
    specular_link_count = 0
    normal_link_count = 0
    flattened_roughness_count = 0
    flattened_specular_count = 0
    flattened_normal_count = 0
    for material_name, material in sorted(materials.items()):
        if not _material_uses_nodes(material) or material.node_tree is None:
            continue
        node_tree = material.node_tree
        principled = next(
            (node for node in node_tree.nodes if getattr(node, "type", None) == "BSDF_PRINCIPLED"),
            None,
        )
        if principled is None:
            continue
        image_nodes = [
            node for node in node_tree.nodes if getattr(node, "type", None) == "TEX_IMAGE"
        ]
        if not image_nodes:
            continue
        semantic_nodes: dict[str, Any] = {}
        for node in image_nodes:
            semantic = _material_node_semantic(node)
            if semantic is None or semantic in semantic_nodes:
                continue
            semantic_nodes[semantic] = node
        if "base_color" in semantic_nodes:
            _set_image_colorspace(semantic_nodes["base_color"].image, "sRGB")
            if _replace_node_input_link(
                node_tree.links,
                to_node=principled,
                to_socket_name="Base Color",
                from_socket=_node_output_socket(semantic_nodes["base_color"], "Color"),
            ):
                normalized_material_count += 1
        normal_node = semantic_nodes.get("normal")
        if normal_node is not None:
            _set_image_colorspace(normal_node.image, "Non-Color")
            normal_map = _ensure_material_node(
                node_tree,
                node_type="ShaderNodeNormalMap",
                name="PEMOIN Normal Map",
                location=(
                    float(getattr(normal_node.location, "x", -400.0)) + 220.0,
                    float(getattr(normal_node.location, "y", 0.0)),
                ),
            )
            _replace_node_input_link(
                node_tree.links,
                to_node=normal_map,
                to_socket_name="Color",
                from_socket=_node_output_socket(normal_node, "Color"),
            )
            if _replace_node_input_link(
                node_tree.links,
                to_node=principled,
                to_socket_name="Normal",
                from_socket=_node_output_socket(normal_map, "Normal"),
            ):
                normalized_material_count += 1
                normal_link_count += 1
        elif material_policy == "preserve_base_alpha":
            if _clear_node_input_links(
                node_tree.links,
                to_node=principled,
                to_socket_name="Normal",
            ):
                normalized_material_count += 1
            flattened_normal_count += 1
        if material_policy == "preserve_most_maps":
            roughness_node = semantic_nodes.get("roughness")
            if roughness_node is not None:
                _set_image_colorspace(roughness_node.image, "Non-Color")
                if _replace_node_input_link(
                    node_tree.links,
                    to_node=principled,
                    to_socket_name="Roughness",
                    from_socket=_node_output_socket(roughness_node, "Color"),
                ):
                    roughness_link_count += 1
                    normalized_material_count += 1
            gloss_node = semantic_nodes.get("glossiness")
            if gloss_node is not None:
                _set_image_colorspace(gloss_node.image, "Non-Color")
                invert = _ensure_material_node(
                    node_tree,
                    node_type="ShaderNodeInvert",
                    name="PEMOIN Glossiness Invert",
                    location=(
                        float(getattr(gloss_node.location, "x", -400.0)) + 220.0,
                        float(getattr(gloss_node.location, "y", 0.0)),
                    ),
                )
                _replace_node_input_link(
                    node_tree.links,
                    to_node=invert,
                    to_socket_name="Color",
                    from_socket=_node_output_socket(gloss_node, "Color"),
                )
                if _replace_node_input_link(
                    node_tree.links,
                    to_node=principled,
                    to_socket_name="Roughness",
                    from_socket=_node_output_socket(invert, "Color"),
                ):
                    glossiness_inverted_count += 1
                    roughness_link_count += 1
                    normalized_material_count += 1
            specular_node = semantic_nodes.get("specular")
            if specular_node is not None and "Specular IOR Level" in principled.inputs:
                _set_image_colorspace(specular_node.image, "Non-Color")
                if _replace_node_input_link(
                    node_tree.links,
                    to_node=principled,
                    to_socket_name="Specular IOR Level",
                    from_socket=_node_output_socket(specular_node, "Color"),
                ):
                    specular_link_count += 1
                    normalized_material_count += 1
        else:
            if "Roughness" in principled.inputs:
                if _clear_node_input_links(
                    node_tree.links,
                    to_node=principled,
                    to_socket_name="Roughness",
                ):
                    normalized_material_count += 1
                principled.inputs["Roughness"].default_value = 0.65
                flattened_roughness_count += 1
            if "Specular IOR Level" in principled.inputs:
                if _clear_node_input_links(
                    node_tree.links,
                    to_node=principled,
                    to_socket_name="Specular IOR Level",
                ):
                    normalized_material_count += 1
                principled.inputs["Specular IOR Level"].default_value = 0.35
                flattened_specular_count += 1
        alpha_node = semantic_nodes.get("alpha")
        if alpha_node is not None and "Alpha" in principled.inputs:
            _set_image_colorspace(alpha_node.image, "Non-Color")
            if _replace_node_input_link(
                node_tree.links,
                to_node=principled,
                to_socket_name="Alpha",
                from_socket=_node_output_socket(alpha_node, "Alpha", "Color"),
            ):
                alpha_link_count += 1
                normalized_material_count += 1
        elif (
            "base_color" in semantic_nodes
            and "Alpha" in principled.inputs
            and "hair" in material_name.lower()
        ):
            if _replace_node_input_link(
                node_tree.links,
                to_node=principled,
                to_socket_name="Alpha",
                from_socket=_node_output_socket(semantic_nodes["base_color"], "Alpha"),
            ):
                alpha_link_count += 1
                normalized_material_count += 1
    return {
        "material_policy": str(material_policy),
        "normalized_material_count": int(normalized_material_count),
        "roughness_link_count": int(roughness_link_count),
        "specular_link_count": int(specular_link_count),
        "normal_link_count": int(normal_link_count),
        "glossiness_inverted_count": int(glossiness_inverted_count),
        "alpha_link_count": int(alpha_link_count),
        "flattened_roughness_count": int(flattened_roughness_count),
        "flattened_specular_count": int(flattened_specular_count),
        "flattened_normal_count": int(flattened_normal_count),
    }


def _relink_and_validate_mixamo_materials(
    *,
    imported_objects: Iterable[bpy.types.Object],
    asset_root: Path,
    run_dir: Path,
    material_policy: str = "preserve_base_alpha_normal",
) -> dict[str, Any]:
    asset_root = asset_root.resolve()
    texture_index = build_mixamo_texture_index(asset_root)
    materials: dict[str, Any] = {}
    for obj in _iter_object_family(imported_objects):
        data = getattr(obj, "data", None)
        for material in getattr(data, "materials", ()) or ():
            if material is None:
                continue
            materials[str(material.name)] = material
    diagnostics: dict[str, Any] = {
        "asset_root": str(asset_root),
        "material_policy": str(material_policy),
        "texture_file_count": int(sum(len(paths) for paths in texture_index.values())),
        "material_count": int(len(materials)),
        "entries": [],
        "unresolved_entries": [],
    }
    for material_name, material in sorted(materials.items()):
        if not _material_uses_nodes(material) or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if getattr(node, "type", None) != "TEX_IMAGE":
                continue
            image = getattr(node, "image", None)
            if image is None:
                continue
            original_path = str(getattr(image, "filepath", "") or "")
            entry = {
                "material": material_name,
                "node": str(getattr(node, "name", "Image Texture")),
                "image_name": str(getattr(image, "name", "")),
                "original_path": original_path,
                "packed": bool(getattr(image, "packed_file", None) is not None),
            }
            if entry["packed"]:
                entry["status"] = "packed_embedded"
                diagnostics["entries"].append(entry)
                continue
            basename = Path(original_path or entry["image_name"]).name
            resolved_path: Path | None = None
            if basename:
                candidates = texture_index.get(basename, [])
                if len(candidates) == 1:
                    resolved_path = candidates[0]
                elif len(candidates) > 1:
                    entry["status"] = "ambiguous"
                    entry["candidates"] = [str(path) for path in candidates]
            if resolved_path is None and original_path:
                candidate = Path(bpy.path.abspath(original_path)).expanduser()
                if candidate.exists():
                    candidate = candidate.resolve()
                    if asset_root in candidate.parents:
                        resolved_path = candidate
            if resolved_path is None:
                entry["status"] = entry.get("status", "missing")
                diagnostics["entries"].append(entry)
                diagnostics["unresolved_entries"].append(entry)
                continue
            image.filepath = str(resolved_path)
            if hasattr(image, "filepath_raw"):
                image.filepath_raw = str(resolved_path)
            try:
                image.reload()
            except Exception:
                pass
            entry["status"] = "resolved"
            entry["resolved_path"] = str(resolved_path)
            diagnostics["entries"].append(entry)
    normalization = _normalize_mixamo_material_graphs(
        materials,
        material_policy=str(material_policy),
    )
    diagnostics.update(normalization)
    diagnostics["resolved_entry_count"] = int(
        sum(
            1
            for entry in diagnostics["entries"]
            if entry.get("status") in {"packed_embedded", "resolved"}
        )
    )
    diagnostics["unresolved_entry_count"] = int(len(diagnostics["unresolved_entries"]))
    diagnostics_path = _write_mixamo_asset_diagnostics(
        run_dir=run_dir,
        diagnostics=diagnostics,
    )
    diagnostics["diagnostics_path"] = str(diagnostics_path)
    if diagnostics["unresolved_entries"]:
        sample = diagnostics["unresolved_entries"][0]
        raise RuntimeError(
            "Mixamo asset package is incomplete: unresolved material texture "
            f"{sample.get('image_name')!r} for material {sample.get('material')!r}. "
            f"See {diagnostics_path}."
        )
    return diagnostics


def _effective_scene_fps(scene: bpy.types.Scene) -> float:
    return float(scene.render.fps) / max(float(scene.render.fps_base), 1e-6)


def _resolve_authoritative_sampling_fps(
    spec: SceneSpec,
    *,
    context: str,
) -> float:
    if spec.sampling_fps is None:
        raise ValueError(
            f"{context} requires a valid sampling_fps on the scene spec, but none was set."
        )
    fps = float(spec.sampling_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(
            f"{context} requires sampling_fps to be finite and > 0, got {spec.sampling_fps!r}."
        )
    return fps


def _normalize_xy_or_none(vec: Sequence[float]) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError("Expected a vector with at least 2 components.")
    xy = np.asarray(arr[:2], dtype=np.float32)
    norm = float(np.linalg.norm(xy))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return (xy / norm).astype(np.float32)


def _perpendicular_xy(axis_xy: Sequence[float]) -> np.ndarray:
    axis = _normalize_xy_or_none(axis_xy)
    if axis is None:
        raise ValueError("Expected a normalizable XY axis.")
    return np.asarray([-float(axis[1]), float(axis[0])], dtype=np.float32)


def _signed_angle_deg_between_xy(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = _normalize_xy_or_none(vec_a)
    b = _normalize_xy_or_none(vec_b)
    if a is None or b is None:
        raise ValueError("Expected normalizable XY vectors for signed angle comparison.")
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    det = float(a[0] * b[1] - a[1] * b[0])
    return float(math.degrees(math.atan2(det, dot)))


def _body_facing_landmark_world(
    armature_obj: bpy.types.Object,
    bone_name: str,
    *,
    evaluated_armature: bpy.types.Object | None = None,
) -> Vector | None:
    arm_obj = evaluated_armature if evaluated_armature is not None else armature_obj
    pose_bone = getattr(arm_obj, "pose", None)
    if pose_bone is None:
        return None
    bone = arm_obj.pose.bones.get(str(bone_name))
    if bone is None:
        return None
    return (arm_obj.matrix_world @ bone.matrix).translation.copy()


def _measure_armature_body_facing_world_xy(
    armature_obj: bpy.types.Object,
    *,
    evaluated_armature: bpy.types.Object | None = None,
) -> tuple[np.ndarray, str, float]:
    arm_obj = evaluated_armature if evaluated_armature is not None else armature_obj
    lateral_groups = (
        (
            "shoulders",
            (
                ("mixamorig:LeftShoulder", "mixamorig:RightShoulder"),
                ("mixamorig7:LeftShoulder", "mixamorig7:RightShoulder"),
            ),
        ),
        (
            "upper_arms",
            (
                ("mixamorig:LeftArm", "mixamorig:RightArm"),
                ("mixamorig7:LeftArm", "mixamorig7:RightArm"),
            ),
        ),
        (
            "upper_legs",
            (
                ("mixamorig:LeftUpLeg", "mixamorig:RightUpLeg"),
                ("mixamorig7:LeftUpLeg", "mixamorig7:RightUpLeg"),
            ),
        ),
    )
    vertical_pairs = (
        ("mixamorig:Spine2", "mixamorig:Hips", "spine2_over_hips"),
        ("mixamorig7:Spine2", "mixamorig7:Hips", "spine2_over_hips"),
        ("mixamorig:Spine", "mixamorig:Hips", "spine_over_hips"),
        ("mixamorig7:Spine", "mixamorig7:Hips", "spine_over_hips"),
        ("mixamorig:Head", "mixamorig:Hips", "head_over_hips"),
        ("mixamorig7:Head", "mixamorig7:Hips", "head_over_hips"),
    )
    best_candidate: tuple[np.ndarray, str, float] | None = None
    for lateral_label, lateral_pairs in lateral_groups:
        group_best: tuple[np.ndarray, str, float] | None = None
        for left_name, right_name in lateral_pairs:
            left_world = _body_facing_landmark_world(
                armature_obj,
                left_name,
                evaluated_armature=arm_obj,
            )
            right_world = _body_facing_landmark_world(
                armature_obj,
                right_name,
                evaluated_armature=arm_obj,
            )
            if left_world is None or right_world is None:
                continue
            lateral = left_world - right_world
            lateral_xy = _normalize_xy_or_none((float(lateral.x), float(lateral.y)))
            lateral_norm = float(
                np.linalg.norm(np.asarray([float(lateral.x), float(lateral.y)], dtype=np.float32))
            )
            if lateral_xy is None or lateral_norm <= 1e-5:
                continue
            for top_name, bottom_name, vertical_label in vertical_pairs:
                top_world = _body_facing_landmark_world(
                    armature_obj,
                    top_name,
                    evaluated_armature=arm_obj,
                )
                bottom_world = _body_facing_landmark_world(
                    armature_obj,
                    bottom_name,
                    evaluated_armature=arm_obj,
                )
                if top_world is None or bottom_world is None:
                    continue
                up = top_world - bottom_world
                if float(up.length) <= 1e-5:
                    continue
                facing = lateral.cross(up)
                facing_xy = _normalize_xy_or_none((float(facing.x), float(facing.y)))
                if facing_xy is None:
                    continue
                confidence = float(min(lateral_norm, float(up.length)))
                method = f"{lateral_label}_x_{vertical_label}"
                if group_best is None or confidence > group_best[2]:
                    group_best = (facing_xy.astype(np.float32), method, confidence)
        if group_best is not None:
            best_candidate = group_best
            break
    if best_candidate is None:
        raise RuntimeError(
            "Could not resolve a reliable body-facing basis from the imported armature landmarks."
        )
    return best_candidate


def _validate_baked_body_facing_parity(
    *,
    scene: bpy.types.Scene,
    armature_obj: bpy.types.Object,
    expected_heading_world_xy: Sequence[float],
    sample_frames: Sequence[int],
    max_error_deg: float = 20.0,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    expected_heading = _normalize_xy_or_none(expected_heading_world_xy)
    if expected_heading is None:
        raise ValueError("expected_heading_world_xy is too small to normalize.")
    sampled_reports: list[dict[str, Any]] = []
    for frame_number in sorted({int(v) for v in sample_frames}):
        scene.frame_set(int(frame_number))
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        arm_eval = armature_obj.evaluated_get(deps)
        measured_xy, method, confidence = _measure_armature_body_facing_world_xy(
            armature_obj,
            evaluated_armature=arm_eval,
        )
        dot = float(np.clip(np.dot(expected_heading, measured_xy), -1.0, 1.0))
        signed_error_deg = _signed_angle_deg_between_xy(expected_heading, measured_xy)
        sampled_reports.append(
            {
                "frame_index": int(frame_number),
                "measured_body_facing_world_xy": [
                    float(measured_xy[0]),
                    float(measured_xy[1]),
                ],
                "body_facing_resolution_method": str(method),
                "body_facing_resolution_confidence": float(confidence),
                "body_facing_vs_expected_dot": float(dot),
                "body_facing_error_deg": float(abs(signed_error_deg)),
                "body_facing_signed_error_deg": float(signed_error_deg),
            }
        )
    if not sampled_reports:
        raise RuntimeError("No sample frames were available for body-facing validation.")
    error_values = np.asarray(
        [float(item["body_facing_error_deg"]) for item in sampled_reports],
        dtype=np.float32,
    )
    signed_error_values = np.asarray(
        [float(item["body_facing_signed_error_deg"]) for item in sampled_reports],
        dtype=np.float32,
    )
    dot_values = np.asarray(
        [float(item["body_facing_vs_expected_dot"]) for item in sampled_reports],
        dtype=np.float32,
    )
    median_error_deg = float(np.median(error_values))
    median_signed_error_deg = float(np.median(signed_error_values))
    max_observed_error_deg = float(np.max(error_values))
    median_dot = float(np.median(dot_values))
    min_dot = float(np.min(dot_values))
    passed = bool(max_observed_error_deg <= float(max_error_deg))
    if (not passed) and raise_on_failure:
        raise RuntimeError(
            "Baked Mixamo body-facing direction disagrees with the resolved heading: "
            f"median_error_deg={median_error_deg:.3f} "
            f"max_error_deg={max_observed_error_deg:.3f} "
            f"median_dot={median_dot:.4f} min_dot={min_dot:.4f}."
        )
    return {
        "sample_frames": [int(v) for v in sorted({int(v) for v in sample_frames})],
        "sample_count": int(len(sampled_reports)),
        "median_body_facing_error_deg": float(median_error_deg),
        "median_body_facing_signed_error_deg": float(median_signed_error_deg),
        "max_body_facing_error_deg": float(max_observed_error_deg),
        "median_body_facing_vs_expected_dot": float(median_dot),
        "min_body_facing_vs_expected_dot": float(min_dot),
        "body_facing_check_passed": bool(passed),
        "samples": sampled_reports,
    }


def _apply_root_yaw_correction(
    *,
    root: bpy.types.Object,
    frame_numbers: Sequence[int],
    resolved_root_yaws_world_deg: list[float],
    correction_deg: float,
) -> None:
    correction = float(correction_deg)
    if abs(correction) <= 1e-6:
        return
    if len(frame_numbers) != len(resolved_root_yaws_world_deg):
        raise ValueError("frame_numbers and resolved_root_yaws_world_deg must have the same length.")
    for idx, frame_number in enumerate(frame_numbers):
        corrected_yaw_deg = float(resolved_root_yaws_world_deg[idx]) + correction
        resolved_root_yaws_world_deg[idx] = corrected_yaw_deg
        root.rotation_euler = (0.0, 0.0, math.radians(corrected_yaw_deg))
        root.keyframe_insert(data_path="rotation_euler", frame=int(frame_number))


def _resolve_heading_axis_in_object_local_xy(
    matrix_world: object,
    heading_world_xy: Sequence[float],
) -> np.ndarray:
    heading_world = _normalize_xy_or_none(heading_world_xy)
    if heading_world is None:
        raise ValueError("heading_world_xy is too small to normalize.")
    world_from_object = np.asarray(matrix_world, dtype=np.float32)
    if world_from_object.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 object matrix when resolving heading basis, got {world_from_object.shape}."
        )
    local_heading = np.linalg.pinv(world_from_object[:3, :3]) @ np.asarray(
        [float(heading_world[0]), float(heading_world[1]), 0.0],
        dtype=np.float32,
    )
    resolved = _normalize_xy_or_none(local_heading[:2])
    if resolved is None:
        raise ValueError(
            "Resolved local heading axis is too small to normalize in object space."
        )
    return resolved.astype(np.float32)


def _correct_looping_hips_translation(
    *,
    raw_local_xy: Sequence[float],
    base_local_xy: Sequence[float],
    source_forward_axis_xy: Sequence[float],
    target_forward_axis_xy: Sequence[float],
    completed_cycles: int,
    cycle_forward_distance_local_m: float,
) -> tuple[np.ndarray, float, float, float]:
    raw_xy = np.asarray(raw_local_xy, dtype=np.float32).reshape(2)
    base_xy = np.asarray(base_local_xy, dtype=np.float32).reshape(2)
    source_forward = _normalize_xy_or_none(source_forward_axis_xy)
    target_forward = _normalize_xy_or_none(target_forward_axis_xy)
    if source_forward is None or target_forward is None:
        raise ValueError("Heading basis axes must be normalizable for hips correction.")
    source_lateral = _perpendicular_xy(source_forward)
    target_lateral = _perpendicular_xy(target_forward)
    raw_delta_xy = raw_xy - base_xy
    in_cycle_forward_m = float(np.dot(raw_delta_xy, source_forward))
    in_cycle_lateral_m = float(np.dot(raw_delta_xy, source_lateral))
    transferred_forward_m = (
        float(max(0, int(completed_cycles))) * float(cycle_forward_distance_local_m)
        + in_cycle_forward_m
    )
    corrected_xy = (
        base_xy
        + np.asarray(target_lateral, dtype=np.float32) * np.float32(in_cycle_lateral_m)
    ).astype(np.float32)
    return corrected_xy, transferred_forward_m, in_cycle_forward_m, in_cycle_lateral_m


def _stabilize_looping_pelvis_world_translation(
    *,
    raw_world_translation: Sequence[float],
    base_world_translation: Sequence[float],
    source_anchor_world_translation: Sequence[float],
    target_anchor_world_translation: Sequence[float],
    locomotion_axis_xy: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_world = np.asarray(raw_world_translation, dtype=np.float32).reshape(3)
    base_world = np.asarray(base_world_translation, dtype=np.float32).reshape(3)
    source_anchor_world = np.asarray(
        source_anchor_world_translation,
        dtype=np.float32,
    ).reshape(3)
    target_anchor_world = np.asarray(
        target_anchor_world_translation,
        dtype=np.float32,
    ).reshape(3)
    transplanted_base_world = (
        target_anchor_world + (base_world - source_anchor_world)
    ).astype(np.float32)
    locomotion_axis = (
        None
        if locomotion_axis_xy is None
        else _normalize_xy_or_none(locomotion_axis_xy)
    )
    if locomotion_axis is None:
        return (
            (transplanted_base_world + (raw_world - base_world)).astype(np.float32),
            np.zeros(3, dtype=np.float32),
            (raw_world - base_world).astype(np.float32),
        )

    raw_delta_world = raw_world - base_world
    raw_delta_xy = np.asarray(raw_delta_world[:2], dtype=np.float32)
    forward_component_m = float(np.dot(raw_delta_xy, locomotion_axis))
    forward_world_xy = (
        np.asarray(locomotion_axis, dtype=np.float32) * np.float32(forward_component_m)
    )
    residual_xy = (raw_delta_xy - forward_world_xy).astype(np.float32)
    desired_world = np.asarray(
        [
            float(transplanted_base_world[0] + residual_xy[0]),
            float(transplanted_base_world[1] + residual_xy[1]),
            float(transplanted_base_world[2] + raw_delta_world[2]),
        ],
        dtype=np.float32,
    )
    forward_correction_world = np.asarray(
        [float(forward_world_xy[0]), float(forward_world_xy[1]), 0.0],
        dtype=np.float32,
    )
    nonforward_residual_world = np.asarray(
        [float(residual_xy[0]), float(residual_xy[1]), float(raw_delta_world[2])],
        dtype=np.float32,
    )
    return desired_world, forward_correction_world, nonforward_residual_world


def _validate_authored_root_path_starts_at_spawn(
    *,
    path_start_world: Sequence[float],
    resolved_spawn_world: Sequence[float],
    tolerance_m: float = 1e-4,
) -> float:
    path_start = np.asarray(path_start_world, dtype=np.float32).reshape(3)
    resolved_spawn = np.asarray(resolved_spawn_world, dtype=np.float32).reshape(3)
    delta_m = float(np.linalg.norm(path_start - resolved_spawn))
    if delta_m > float(tolerance_m):
        raise RuntimeError(
            "Authored root path does not start at the resolved spawn: "
            f"delta_m={delta_m:.6f} "
            f"path_start={tuple(float(v) for v in path_start.tolist())} "
            f"resolved_spawn={tuple(float(v) for v in resolved_spawn.tolist())}."
        )
    return delta_m


def _validate_actor_hierarchy_alignment_state(
    *,
    root_world: Sequence[float],
    armature_world: Sequence[float],
    key_bone_world_by_name: Mapping[str, Sequence[float]],
    max_root_armature_delta_m: float = 1e-3,
    max_bone_horizontal_offset_m: float = 2.0,
    min_bone_vertical_offset_m: float = -0.2,
    max_bone_vertical_offset_m: float = 3.5,
) -> dict[str, Any]:
    root = np.asarray(root_world, dtype=np.float32).reshape(3)
    armature = np.asarray(armature_world, dtype=np.float32).reshape(3)
    root_armature_delta_m = float(np.linalg.norm(root - armature))
    if root_armature_delta_m > float(max_root_armature_delta_m):
        raise RuntimeError(
            "Baked actor hierarchy is misaligned: armature object drifted away from the actor root. "
            f"root_armature_delta_m={root_armature_delta_m:.6f} "
            f"root={tuple(float(v) for v in root.tolist())} "
            f"armature={tuple(float(v) for v in armature.tolist())}."
        )

    bone_offsets: dict[str, dict[str, float]] = {}
    for bone_name, bone_world in key_bone_world_by_name.items():
        bone = np.asarray(bone_world, dtype=np.float32).reshape(3)
        delta = bone - armature
        horizontal_offset_m = float(np.linalg.norm(delta[:2]))
        vertical_offset_m = float(delta[2])
        bone_offsets[str(bone_name)] = {
            "horizontal_offset_m": horizontal_offset_m,
            "vertical_offset_m": vertical_offset_m,
        }
        if horizontal_offset_m > float(max_bone_horizontal_offset_m):
            raise RuntimeError(
                "Baked actor hierarchy is misaligned: a key pose bone is implausibly far from the actor root/armature in XY. "
                f"bone={bone_name} horizontal_offset_m={horizontal_offset_m:.6f} "
                f"bone_world={tuple(float(v) for v in bone.tolist())} "
                f"armature_world={tuple(float(v) for v in armature.tolist())}."
            )
        if not (
            float(min_bone_vertical_offset_m)
            <= vertical_offset_m
            <= float(max_bone_vertical_offset_m)
        ):
            raise RuntimeError(
                "Baked actor hierarchy is misaligned: a key pose bone has an implausible vertical offset from the actor root/armature. "
                f"bone={bone_name} vertical_offset_m={vertical_offset_m:.6f} "
                f"bone_world={tuple(float(v) for v in bone.tolist())} "
                f"armature_world={tuple(float(v) for v in armature.tolist())}."
            )

    return {
        "root_armature_delta_m": root_armature_delta_m,
        "bone_offsets": bone_offsets,
    }


def _validate_baked_actor_hierarchy_alignment(
    *,
    scene: bpy.types.Scene,
    root: bpy.types.Object,
    char_arm: bpy.types.Object,
    sample_frames: Sequence[int],
    key_bone_names: Sequence[str],
) -> dict[str, Any]:
    current_frame = int(scene.frame_current)
    current_subframe = float(getattr(scene, "frame_subframe", 0.0))
    frame_reports: list[dict[str, Any]] = []
    try:
        for frame_number in sample_frames:
            scene.frame_set(int(frame_number))
            bpy.context.view_layer.update()
            deps = bpy.context.evaluated_depsgraph_get()
            root_eval = root.evaluated_get(deps)
            arm_eval = char_arm.evaluated_get(deps)
            key_bone_world_by_name: dict[str, tuple[float, float, float]] = {}
            for bone_name in key_bone_names:
                pose_bone = arm_eval.pose.bones.get(bone_name)
                if pose_bone is None:
                    continue
                bone_world = (arm_eval.matrix_world @ pose_bone.matrix).translation
                key_bone_world_by_name[str(bone_name)] = tuple(
                    float(v) for v in bone_world
                )
            if not key_bone_world_by_name:
                raise RuntimeError(
                    "No key pose bones were available for baked actor alignment validation."
                )
            metrics = _validate_actor_hierarchy_alignment_state(
                root_world=tuple(float(v) for v in root_eval.matrix_world.translation),
                armature_world=tuple(float(v) for v in arm_eval.matrix_world.translation),
                key_bone_world_by_name=key_bone_world_by_name,
            )
            frame_reports.append(
                {
                    "frame": int(frame_number),
                    **metrics,
                }
            )
    finally:
        scene.frame_set(current_frame, subframe=current_subframe)
        bpy.context.view_layer.update()
    return {
        "sampled_frames": [int(frame) for frame in sample_frames],
        "reports": frame_reports,
    }


def _set_scene_frame_float(scene: bpy.types.Scene, frame_value: float) -> None:
    frame_f = float(frame_value)
    if not np.isfinite(frame_f):
        raise ValueError(f"Frame value must be finite, got {frame_value!r}.")
    base = math.floor(frame_f)
    subframe = frame_f - float(base)
    scene.frame_set(int(base), subframe=float(subframe))


def _grounding_failure_bucket(reason: str | None) -> str:
    if not reason:
        return "missing_support_surface"
    if reason == _SUPPORT_FAILURE_REASON_CONFIDENCE:
        return "support_confidence_rejected"
    if reason.startswith(_SUPPORT_FAILURE_REASON_RELOCK):
        return "support_relock_rejected"
    if reason in {"no_support_surface", "persisted_fallback_no_spatial_candidates"}:
        return "missing_support_surface"
    if reason.startswith("persisted_fallback_locality_rejected"):
        return "missing_support_surface"
    return "support_resolution_failed"


def _write_road_surface_summary(
    *,
    spec: SceneSpec,
    trajectory_anchor_world: tuple[float, float, float],
    motion_forward_world: tuple[float, float, float],
    resolved_spawn_world: tuple[float, float, float],
    base_heading_world_deg: float,
    resolved_heading_world_deg: float | None,
    spawn_min_distance_to_trajectory_m: float,
    global_planes: dict[int, RoadPlaneSpec],
    grounding_diagnostics: list[GroundingDiagnostic],
    motion_direction_parity: dict[str, Any] | None = None,
) -> None:
    effective_hold_frames, effective_hold_seconds = _resolve_local_support_hold_budget(spec)
    vis_dir = spec.run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = vis_dir / "road_surface_pipeline.json"
    pre_dists = [
        abs(float(d.pre_correction_signed_distance_m))
        for d in grounding_diagnostics
        if d.pre_correction_signed_distance_m is not None
    ]
    post_dists = [
        abs(float(d.post_correction_signed_distance_m))
        for d in grounding_diagnostics
        if d.post_correction_signed_distance_m is not None
    ]
    corrections = [
        float(np.linalg.norm(np.asarray(d.applied_translation_world, dtype=np.float32)))
        for d in grounding_diagnostics
        if (not d.no_plane)
    ]
    corrections_xy = [
        float(d.applied_translation_xy_m)
        for d in grounding_diagnostics
        if d.applied_translation_xy_m is not None and (not d.no_plane)
    ]
    planned_z_deltas = [
        float(d.planned_z_delta_m)
        for d in grounding_diagnostics
        if d.planned_z_delta_m is not None and (not d.no_plane)
    ]
    vertical_velocities = [
        abs(float(d.vertical_velocity_mps))
        for d in grounding_diagnostics
        if d.vertical_velocity_mps is not None and (not d.no_plane)
    ]
    vertical_accels = [
        abs(float(d.vertical_accel_mps2))
        for d in grounding_diagnostics
        if d.vertical_accel_mps2 is not None and (not d.no_plane)
    ]
    traversal_segment_ids = sorted(
        {
            int(d.traversal_segment_id)
            for d in grounding_diagnostics
            if d.traversal_segment_id is not None
        }
    )
    chosen_xy_distances = [
        float(d.chosen_plane_center_xy_distance_m)
        for d in grounding_diagnostics
        if d.chosen_plane_center_xy_distance_m is not None
    ]
    left_post_dists = [
        abs(float(d.left_post_signed_distance_m))
        for d in grounding_diagnostics
        if d.left_post_signed_distance_m is not None
    ]
    right_post_dists = [
        abs(float(d.right_post_signed_distance_m))
        for d in grounding_diagnostics
        if d.right_post_signed_distance_m is not None
    ]
    support_counts: dict[str, int] = {}
    support_mode_counts: dict[str, int] = {}
    failure_frames_by_bucket: dict[str, list[int]] = {}
    visibility_culled_frames = [
        int(d.frame_index) for d in grounding_diagnostics if d.visibility_culled
    ]
    for d in grounding_diagnostics:
        support_counts[d.selected_support_foot] = support_counts.get(d.selected_support_foot, 0) + 1
        support_mode_counts[d.support_mode] = support_mode_counts.get(d.support_mode, 0) + 1
        if d.no_plane and not d.visibility_culled:
            bucket = (
                "support_locality_rejected"
                if d.plane_selection_rejected_for_locality
                else _grounding_failure_bucket(d.support_failure_reason)
            )
            failure_frames_by_bucket.setdefault(bucket, []).append(int(d.frame_index))
    payload = {
        "persisted_plane_frame_count": len(global_planes),
        "pedestrian_anchor_world": [
            float(trajectory_anchor_world[0]),
            float(trajectory_anchor_world[1]),
            float(trajectory_anchor_world[2]),
        ],
        "pedestrian_motion_forward_world": [
            float(motion_forward_world[0]),
            float(motion_forward_world[1]),
            float(motion_forward_world[2]),
        ],
        "pedestrian_resolved_spawn_world": [
            float(resolved_spawn_world[0]),
            float(resolved_spawn_world[1]),
            float(resolved_spawn_world[2]),
        ],
        "pedestrian_base_heading_world_deg": float(base_heading_world_deg),
        "pedestrian_heading_offset_deg": float(spec.pedestrian_heading_deg),
        "pedestrian_resolved_heading_world_deg": (
            None
            if resolved_heading_world_deg is None
            else float(resolved_heading_world_deg)
        ),
        "pedestrian_min_distance_to_trajectory_m": float(
            spawn_min_distance_to_trajectory_m
        ),
        "grounding_frame_count": len(grounding_diagnostics),
        "frames_missing_left_foot": [
            int(d.frame_index) for d in grounding_diagnostics if d.missing_left_foot
        ],
        "frames_missing_right_foot": [
            int(d.frame_index) for d in grounding_diagnostics if d.missing_right_foot
        ],
        "frames_missing_support_surface": [
            int(d.frame_index)
            for d in grounding_diagnostics
            if d.no_plane
            and not d.visibility_culled
            and _grounding_failure_bucket(d.support_failure_reason) == "missing_support_surface"
        ],
        "frames_support_confidence_rejected": failure_frames_by_bucket.get(
            "support_confidence_rejected",
            [],
        ),
        "frames_support_relock_rejected": failure_frames_by_bucket.get(
            "support_relock_rejected",
            [],
        ),
        "frames_support_resolution_failed": failure_frames_by_bucket.get(
            "support_resolution_failed",
            [],
        ),
        "frames_support_locality_rejected": failure_frames_by_bucket.get(
            "support_locality_rejected",
            [],
        ),
        "frames_visibility_culled": visibility_culled_frames,
        "visibility_cull_frame_count": int(len(visibility_culled_frames)),
        "grounding_required_frame_count": int(
            sum(1 for d in grounding_diagnostics if d.frame_requires_support)
        ),
        "support_foot_counts": support_counts,
        "support_mode_counts": support_mode_counts,
        "locality_rejection_frames": [
            int(d.frame_index)
            for d in grounding_diagnostics
            if d.plane_selection_rejected_for_locality
        ],
        "median_abs_correction_m": float(np.median(corrections)) if corrections else None,
        "max_abs_correction_m": float(np.max(corrections)) if corrections else None,
        "max_applied_translation_xy_m": float(np.max(corrections_xy)) if corrections_xy else None,
        "max_abs_planned_z_delta_m": float(np.max(np.abs(planned_z_deltas))) if planned_z_deltas else None,
        "max_vertical_velocity_mps": float(np.max(vertical_velocities)) if vertical_velocities else None,
        "max_vertical_accel_mps2": float(np.max(vertical_accels)) if vertical_accels else None,
        "trajectory_traversal_segment_count": int(len(traversal_segment_ids)),
        "trajectory_traversal_segment_ids": traversal_segment_ids,
        "median_chosen_plane_center_xy_distance_m": (
            float(np.median(chosen_xy_distances)) if chosen_xy_distances else None
        ),
        "max_chosen_plane_center_xy_distance_m": (
            float(np.max(chosen_xy_distances)) if chosen_xy_distances else None
        ),
        "median_pre_residual_m": float(np.median(pre_dists)) if pre_dists else None,
        "max_pre_residual_m": float(np.max(pre_dists)) if pre_dists else None,
        "median_post_residual_m": float(np.median(post_dists)) if post_dists else None,
        "max_post_residual_m": float(np.max(post_dists)) if post_dists else None,
        "median_abs_left_post_distance_m": (
            float(np.median(left_post_dists)) if left_post_dists else None
        ),
        "median_abs_right_post_distance_m": (
            float(np.median(right_post_dists)) if right_post_dists else None
        ),
        "settings": {
            "global_plane_range_m": spec.global_plane_range_m,
            "global_plane_min_range_m": spec.global_plane_min_range_m,
            "global_plane_frame_window": spec.global_plane_frame_window,
            "pedestrian_actor_name": spec.pedestrian_actor_name,
            "pedestrian_placement_mode": str(
                getattr(spec, "pedestrian_placement_mode", "trajectory_relative")
            ),
            "pedestrian_authored_position_x_m": getattr(
                spec, "pedestrian_authored_position_x_m", None
            ),
            "pedestrian_authored_position_z_m": getattr(
                spec, "pedestrian_authored_position_z_m", None
            ),
            "pedestrian_authored_heading_yaw_deg": getattr(
                spec, "pedestrian_authored_heading_yaw_deg", None
            ),
            "pedestrian_trajectory_t": float(spec.pedestrian_trajectory_t),
            "pedestrian_forward_offset_m": float(spec.pedestrian_forward_offset_m),
            "pedestrian_left_offset_m": float(spec.pedestrian_left_offset_m),
            "pedestrian_up_offset_m": float(spec.pedestrian_up_offset_m),
            "pedestrian_heading_deg": float(spec.pedestrian_heading_deg),
            "foot_contact_max_plane_dist_m": spec.foot_contact_max_plane_dist_m,
            "max_plane_center_xy_distance_m": spec.max_plane_center_xy_distance_m,
            "road_labels": list(spec.road_labels),
        },
        "diagnostics_files": {
            "grounding_diagnostics_json": "grounding_diagnostics.json",
            "grounding_diagnostics_csv": "grounding_diagnostics.csv",
            "support_surface_diagnostics_json": "support_surface_diagnostics.json",
            "support_surface_diagnostics_csv": "support_surface_diagnostics.csv",
            "trajectory_support_segments_json": "trajectory_support_segments.json",
            "trajectory_height_profile_csv": "trajectory_height_profile.csv",
        },
        "trajectory_grounding_policy": {
            "mode": "trajectory_first_z_only",
            "sampling_fps": float(_resolve_authoritative_sampling_fps(spec, context="Road-surface summary")),
            "transition_frames": int(getattr(spec, "trajectory_grounding_transition_frames", 4)),
            "max_step_m": float(getattr(spec, "trajectory_grounding_max_step_m", 0.05)),
            "max_vertical_velocity_mps": float(
                getattr(spec, "trajectory_grounding_max_vertical_velocity_mps", 0.9)
            ),
            "max_vertical_accel_mps2": float(
                getattr(spec, "trajectory_grounding_max_vertical_accel_mps2", 2.5)
            ),
        },
        "visibility_cull_policy": {
            "mode": "per_frame_geometric_frustum",
            "culled_frames": visibility_culled_frames,
            "required_support_frames": int(
                sum(1 for d in grounding_diagnostics if d.frame_requires_support)
            ),
        },
    }
    if motion_direction_parity is not None:
        parity_payload = dict(motion_direction_parity)
        parity_payload.setdefault(
            "placement_base_heading_world_deg",
            float(base_heading_world_deg),
        )
        parity_payload.setdefault(
            "placement_resolved_heading_world_deg",
            (
                None
                if resolved_heading_world_deg is None
                else float(resolved_heading_world_deg)
            ),
        )
        payload["motion_direction_parity"] = parity_payload
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_info(f"Road-surface summary written to {summary_path}")


def viz_road_planes(
    *,
    c2w: ndarray,
    frame_indices: ndarray,
    global_plane_collection: bpy.types.Collection,
    spec: SceneSpec,
) -> RoadSurfacePipelineResult:
    global_planes = load_existing_global_road_planes(
        spec=spec,
        c2w=c2w,
        frame_indices=frame_indices,
    )
    if not global_planes:
        log_warning_big("No persisted road planes found in standard/road_plane.")
        return RoadSurfacePipelineResult(global_planes={})

    for frame_idx, plane_spec in sorted(global_planes.items()):
        traj_idx = np.where(frame_indices == frame_idx)[0]
        if len(traj_idx) > 0:
            validate_plane_camera_relationship(plane_spec, c2w[int(traj_idx[0]), :3, 3])
        add_road_plane(plane_spec, global_plane_collection, name_prefix="global_plane")

    log_info(f"Road-surface pipeline: persisted_planes={len(global_planes)}")
    return RoadSurfacePipelineResult(global_planes=global_planes)


def load_existing_global_road_planes(
    *,
    spec: SceneSpec,
    c2w: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[int, RoadPlaneSpec]:
    frame_to_idx = {int(frame): i for i, frame in enumerate(frame_indices.tolist())}
    result: dict[int, RoadPlaneSpec] = {}
    missing: list[int] = []

    for frame in frame_indices.tolist():
        frame_int = int(frame)
        road_path = spec.run_dir / "standard" / "road_plane" / f"{frame_int:06d}.npz"
        if not road_path.exists():
            missing.append(frame_int)
            continue
        with np.load(road_path, allow_pickle=True) as data:
            normal = np.asarray(data["normal"], dtype=np.float32).reshape(3)
            offset = float(data["offset"])
            metadata = _coerce_metadata(data["metadata"]) if "metadata" in data.files else {}
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            log_warning(
                f"[road-plane][global][frame {frame_int:06d}] skipped: degenerate normal in {road_path}"
            )
            missing.append(frame_int)
            continue
        normal = normal / norm
        offset = offset / norm

        traj_idx = frame_to_idx.get(frame_int)
        if traj_idx is None:
            missing.append(frame_int)
            continue
        camera_pos = c2w[traj_idx, :3, 3]
        signed = float(np.dot(normal, camera_pos) + offset)
        center = camera_pos - signed * normal

        result[frame_int] = RoadPlaneSpec(
            normal=normal.astype(np.float32),
            offset=float(offset),
            center=center.astype(np.float32),
            scale_u=spec.global_plane_range_m,
            scale_v=spec.global_plane_range_m,
            frame_index=frame_int,
            confidence=float(_derive_persisted_plane_confidence(metadata)),
            fit_point_count=(
                int(metadata.get("fit_point_count"))
                if metadata.get("fit_point_count") is not None
                else None
            ),
            metadata=dict(metadata),
            material_color=(0.1, 0.3, 0.8),
            material_alpha=0.18,
        )

    log_info(
        "[road-plane][global] loaded existing provider planes: "
        f"{len(result)}/{len(frame_indices)}"
    )
    if missing:
        log_warning_big(
            "Missing global road planes for frames: "
            f"{_format_frame_ranges(sorted(missing))}. These frames will use nearest available planes."
        )
    return result


def _format_frame_ranges(frames: Sequence[int]) -> str:
    if not frames:
        return "none"
    ordered = sorted({int(frame) for frame in frames})
    ranges: list[str] = []
    start = ordered[0]
    prev = ordered[0]
    for frame in ordered[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = frame
        prev = frame
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _bone_suffix(name: str) -> str:
    if ":" in name:
        return name.split(":", 1)[1].lower()
    return name.lower()


def _find_named_pose_bone(
    armature_obj: bpy.types.Object,
    candidate_suffixes: Sequence[str],
) -> bpy.types.PoseBone | None:
    suffixes = [suffix.lower() for suffix in candidate_suffixes]
    for pose_bone in armature_obj.pose.bones:
        bone_suffix = _bone_suffix(pose_bone.name)
        if bone_suffix in suffixes:
            return pose_bone
    for suffix in suffixes:
        for pose_bone in armature_obj.pose.bones:
            if _bone_suffix(pose_bone.name).endswith(suffix):
                return pose_bone
    return None


def _evaluate_pose_bone_world(
    armature_obj: bpy.types.Object,
    pose_bone: bpy.types.PoseBone,
) -> np.ndarray:
    deps = bpy.context.evaluated_depsgraph_get()
    armature_eval = armature_obj.evaluated_get(deps)
    bone_eval = armature_eval.pose.bones.get(pose_bone.name)
    if bone_eval is None:
        bone_eval = pose_bone
        armature_matrix = armature_obj.matrix_world
    else:
        armature_matrix = armature_eval.matrix_world
    world = (armature_matrix @ bone_eval.matrix).translation
    return np.asarray(world[:], dtype=np.float32)


def _evaluate_feet_world(
    actor_root: bpy.types.Object,
    armature_obj: bpy.types.Object,
) -> dict[str, np.ndarray | None]:
    _ = actor_root
    left_bone = _find_named_pose_bone(
        armature_obj, ("LeftToeBase", "LeftFoot", "lefttoebase", "leftfoot")
    )
    right_bone = _find_named_pose_bone(
        armature_obj, ("RightToeBase", "RightFoot", "righttoebase", "rightfoot")
    )
    return {
        "left": None if left_bone is None else _evaluate_pose_bone_world(armature_obj, left_bone),
        "right": None if right_bone is None else _evaluate_pose_bone_world(armature_obj, right_bone),
    }


def _select_support_foot(
    left_foot: np.ndarray | None,
    right_foot: np.ndarray | None,
    plane: RoadPlaneSpec,
) -> tuple[str, np.ndarray, float]:
    left_signed = (
        None
        if left_foot is None
        else float(np.dot(plane.normal, left_foot) + plane.offset)
    )
    right_signed = (
        None
        if right_foot is None
        else float(np.dot(plane.normal, right_foot) + plane.offset)
    )
    if left_signed is None and right_signed is None:
        raise ValueError("Both foot anchors are missing.")
    if left_signed is None:
        return "right", np.asarray(right_foot, dtype=np.float32), float(right_signed)
    if right_signed is None:
        return "left", np.asarray(left_foot, dtype=np.float32), float(left_signed)
    if abs(left_signed - right_signed) <= 0.02:
        if float(left_foot[2]) <= float(right_foot[2]):
            return "both", np.asarray(left_foot, dtype=np.float32), float(left_signed)
        return "both", np.asarray(right_foot, dtype=np.float32), float(right_signed)
    if abs(left_signed) <= abs(right_signed):
        return "left", np.asarray(left_foot, dtype=np.float32), float(left_signed)
    return "right", np.asarray(right_foot, dtype=np.float32), float(right_signed)


def _support_anchor_label_from_weights(left_weight: float, right_weight: float) -> str:
    if left_weight <= 0.0 and right_weight <= 0.0:
        return "none"
    if left_weight > 0.0 and right_weight <= 0.0:
        return "left"
    if right_weight > 0.0 and left_weight <= 0.0:
        return "right"
    if abs(float(left_weight) - float(right_weight)) <= 0.2:
        return "both"
    return "left" if float(left_weight) > float(right_weight) else "right"


def _phase_in_ranges(phase: float, ranges: Sequence[tuple[float, float]]) -> bool:
    wrapped_phase = float(phase) % 1.0
    for start, end in ranges:
        start_wrapped = float(start) % 1.0
        end_wrapped = float(end) % 1.0
        if start_wrapped <= end_wrapped:
            if start_wrapped <= wrapped_phase <= end_wrapped:
                return True
        else:
            if wrapped_phase >= start_wrapped or wrapped_phase <= end_wrapped:
                return True
    return False


def _resolve_contact_phase_for_frame(
    actor_root: bpy.types.Object,
    *,
    frame_idx: int,
    spec: SceneSpec | None = None,
) -> float | None:
    cycle_frames = getattr(spec, "foot_contact_gait_cycle_frames", None)
    if cycle_frames is None:
        cycle_frames = actor_root.get("pemoin_mixamo_cycle_len_frames")
    if cycle_frames is None:
        return None
    cycle_frames = float(cycle_frames)
    if not np.isfinite(cycle_frames) or cycle_frames <= 1.0:
        return None
    bake_start = float(actor_root.get("pemoin_mixamo_bake_start_frame", float(frame_idx)))
    phase_offset = 0.0 if spec is None else float(getattr(spec, "foot_contact_phase_offset", 0.0))
    phase = ((float(frame_idx) - bake_start) / cycle_frames) + phase_offset
    wrapped = math.fmod(phase, 1.0)
    if wrapped < 0.0:
        wrapped += 1.0
    return float(wrapped)


def _resolve_contact_state_for_phase(
    phase: float | None,
    *,
    spec: SceneSpec | None = None,
) -> str:
    if phase is None:
        return "swing"
    left_ranges = tuple(getattr(spec, "foot_contact_left_stance_phase_ranges", ()) or ())
    right_ranges = tuple(getattr(spec, "foot_contact_right_stance_phase_ranges", ()) or ())
    left_active = _phase_in_ranges(float(phase), left_ranges)
    right_active = _phase_in_ranges(float(phase), right_ranges)
    if left_active and right_active:
        return "dual_support"
    if left_active:
        return "left_stance"
    if right_active:
        return "right_stance"
    return "swing"


def _minimum_contact_segment_length(kind: str, spec: SceneSpec | None = None) -> int:
    if kind == "swing":
        return max(int(getattr(spec, "foot_contact_min_swing_frames", 1)), 1)
    min_stance_frames = max(int(getattr(spec, "foot_contact_min_stance_frames", 1)), 1)
    if spec is None:
        return min_stance_frames
    sampling_fps = getattr(spec, "sampling_fps", None)
    if sampling_fps is None:
        return min_stance_frames
    try:
        fps = float(sampling_fps)
    except (TypeError, ValueError):
        return min_stance_frames
    if np.isfinite(fps) and fps <= 10.0:
        return 1
    return min_stance_frames


def _iter_contact_runs(states: Sequence[str]) -> list[tuple[int, int, str]]:
    if not states:
        return []
    runs: list[tuple[int, int, str]] = []
    start = 0
    current = str(states[0])
    for idx, state in enumerate(states[1:], start=1):
        if str(state) == current:
            continue
        runs.append((start, idx, current))
        start = idx
        current = str(state)
    runs.append((start, len(states), current))
    return runs


def _choose_contact_run_replacement(
    *,
    runs: Sequence[tuple[int, int, str]],
    run_index: int,
    states: Sequence[str],
) -> str:
    _start, _end, kind = runs[run_index]
    prev_kind = runs[run_index - 1][2] if run_index > 0 else None
    next_kind = runs[run_index + 1][2] if run_index + 1 < len(runs) else None
    if prev_kind is not None and prev_kind == next_kind:
        return str(prev_kind)
    if kind == "swing":
        if prev_kind is not None and prev_kind != "swing":
            return str(prev_kind)
        if next_kind is not None and next_kind != "swing":
            return str(next_kind)
    else:
        if prev_kind is not None and prev_kind == "swing" and next_kind is not None and next_kind != "swing":
            return str(next_kind)
        if next_kind is not None and next_kind == "swing" and prev_kind is not None and prev_kind != "swing":
            return str(prev_kind)
    prev_len = 0 if prev_kind is None else int(runs[run_index - 1][1] - runs[run_index - 1][0])
    next_len = 0 if next_kind is None else int(runs[run_index + 1][1] - runs[run_index + 1][0])
    if prev_kind is None and next_kind is None:
        return str(kind)
    if next_kind is None:
        return str(prev_kind)
    if prev_kind is None:
        return str(next_kind)
    return str(prev_kind if prev_len >= next_len else next_kind)


def _clean_contact_state_sequence(
    states: Sequence[str],
    *,
    spec: SceneSpec | None = None,
) -> list[str]:
    cleaned = [str(state) for state in states]
    for _ in range(max(len(cleaned), 1) * 2):
        changed = False
        runs = _iter_contact_runs(cleaned)
        for run_index, (start, end, kind) in enumerate(runs):
            if (end - start) >= _minimum_contact_segment_length(kind, spec):
                continue
            replacement = _choose_contact_run_replacement(
                runs=runs,
                run_index=run_index,
                states=cleaned,
            )
            if replacement == kind:
                continue
            for idx in range(start, end):
                cleaned[idx] = replacement
            changed = True
        if not changed:
            break
    return cleaned


def _build_contact_schedule(
    *,
    actor_root: bpy.types.Object,
    frames: Sequence[int],
    spec: SceneSpec | None = None,
) -> tuple[dict[int, ContactFrameState], dict[int, ContactSegment]]:
    phases = [
        _resolve_contact_phase_for_frame(actor_root, frame_idx=int(frame_idx), spec=spec)
        for frame_idx in frames
    ]
    raw_states = [
        _resolve_contact_state_for_phase(phase, spec=spec)
        for phase in phases
    ]
    clean_states = _clean_contact_state_sequence(raw_states, spec=spec)
    segments: dict[int, ContactSegment] = {}
    frame_states: dict[int, ContactFrameState] = {}
    segment_runs = _iter_contact_runs(clean_states)
    segment_kinds = [kind for _, _, kind in segment_runs]
    previous_stance_by_segment: list[str | None] = []
    next_stance_by_segment: list[str | None] = []
    previous_stance: str | None = None
    for kind in segment_kinds:
        previous_stance_by_segment.append(previous_stance)
        if kind in {"left_stance", "right_stance"}:
            previous_stance = str(kind)
    next_stance: str | None = None
    for kind in reversed(segment_kinds):
        next_stance_by_segment.append(next_stance)
        if kind in {"left_stance", "right_stance"}:
            next_stance = str(kind)
    next_stance_by_segment.reverse()
    for segment_id, ((start, end, kind), previous_kind, next_kind) in enumerate(
        zip(segment_runs, previous_stance_by_segment, next_stance_by_segment),
        start=1,
    ):
        segment_frames = tuple(int(frames[idx]) for idx in range(start, end))
        segments[segment_id] = ContactSegment(
            segment_id=segment_id,
            kind=str(kind),
            frame_indices=segment_frames,
        )
        segment_length = int(end - start)
        for offset, frame_idx in enumerate(segment_frames):
            frame_states[int(frame_idx)] = ContactFrameState(
                frame_index=int(frame_idx),
                phase=phases[start + offset],
                raw_state=str(raw_states[start + offset]),
                clean_state=str(kind),
                segment_id=int(segment_id),
                segment_kind=str(kind),
                segment_frame_index=int(offset),
                segment_length=int(segment_length),
                previous_stance_kind=previous_kind,
                next_stance_kind=next_kind,
            )
    return frame_states, segments


def _foot_from_state_kind(kind: str | None) -> str | None:
    if kind == "left_stance":
        return "left"
    if kind == "right_stance":
        return "right"
    return None


def _resolve_segment_support_weights(
    frame_state: ContactFrameState,
    *,
    spec: SceneSpec | None = None,
) -> tuple[str, float, float, str, str, str | None]:
    state = str(frame_state.clean_state)
    if state == "left_stance":
        return "left", 1.0, 0.0, "contact_segment_lock", "single_support", "planted_left"
    if state == "right_stance":
        return "right", 0.0, 1.0, "contact_segment_lock", "single_support", "planted_right"
    if state == "swing":
        held = _foot_from_state_kind(frame_state.previous_stance_kind)
        transfer_frames = max(int(getattr(spec, "support_anchor_transfer_frames", 3)), 1)
        hold_previous = int(frame_state.segment_frame_index) < transfer_frames
        if held == "left":
            if hold_previous:
                return (
                    "left",
                    1.0,
                    0.0,
                    "contact_segment_swing_hold",
                    "swing_hold",
                    "held_previous_left",
                )
            return (
                "left",
                1.0,
                0.0,
                "contact_segment_swing_release",
                "swing_release",
                "released_swing_left",
            )
        if held == "right":
            if hold_previous:
                return (
                    "right",
                    0.0,
                    1.0,
                    "contact_segment_swing_hold",
                    "swing_hold",
                    "held_previous_right",
                )
            return (
                "right",
                0.0,
                1.0,
                "contact_segment_swing_release",
                "swing_release",
                "released_swing_right",
            )
        next_foot = _foot_from_state_kind(frame_state.next_stance_kind)
        if next_foot == "left":
            return (
                "left",
                1.0,
                0.0,
                "contact_segment_swing_release",
                "swing_release",
                "released_swing_lead_in_left",
            )
        if next_foot == "right":
            return (
                "right",
                0.0,
                1.0,
                "contact_segment_swing_release",
                "swing_release",
                "released_swing_lead_in_right",
            )
        return (
            "both",
            0.5,
            0.5,
            "contact_segment_swing_release",
            "swing_release",
            "released_swing_unknown",
        )
    outgoing = _foot_from_state_kind(frame_state.previous_stance_kind)
    incoming = _foot_from_state_kind(frame_state.next_stance_kind)
    if outgoing is None:
        outgoing = incoming
    if incoming is None:
        incoming = outgoing
    if outgoing is None and incoming is None:
        return "both", 0.5, 0.5, "contact_segment_dual_support", "transfer", "dual_unknown"
    if outgoing == incoming:
        if outgoing == "left":
            return "left", 1.0, 0.0, "contact_segment_dual_same", "single_support", "dual_same_left"
        return "right", 0.0, 1.0, "contact_segment_dual_same", "single_support", "dual_same_right"
    transfer_frames = max(int(getattr(spec, "support_anchor_transfer_frames", 3)), 1)
    progress = min(
        max(float(frame_state.segment_frame_index + 1) / float(transfer_frames), 0.0),
        1.0,
    )
    if outgoing == "left":
        left_weight = 1.0 - (0.5 * progress)
        right_weight = 0.5 * progress
    else:
        right_weight = 1.0 - (0.5 * progress)
        left_weight = 0.5 * progress
    total = max(float(left_weight + right_weight), 1e-6)
    left_weight /= total
    right_weight /= total
    return "both", float(left_weight), float(right_weight), "contact_segment_dual_support", "transfer", f"{outgoing}_to_{incoming}"


def _xy_lock_mode_from_policy(policy: str | None, *, clamped: bool = False) -> str | None:
    if clamped:
        return "clamped_relock"
    if policy == "contact_segment_swing_release":
        return "released_swing"
    if policy == "contact_segment_dual_support":
        return "transfer"
    if policy in {
        "contact_segment_lock",
        "contact_segment_swing_hold",
        "contact_segment_dual_same",
    }:
        return "planted"
    return None


def _same_plane_relock_xy_cap_m(spec: SceneSpec | None = None) -> float | None:
    if spec is None:
        return None
    sampling_fps = getattr(spec, "sampling_fps", None)
    foot_contact_max_speed_mps = getattr(spec, "foot_contact_max_speed_mps", None)
    if sampling_fps is None or foot_contact_max_speed_mps is None:
        return None
    try:
        fps = float(sampling_fps)
        max_speed = float(foot_contact_max_speed_mps)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fps) or fps <= 0.0 or not np.isfinite(max_speed) or max_speed <= 0.0:
        return None
    return float(max_speed / fps)


def _build_support_anchor_from_weights(
    *,
    left_foot: np.ndarray | None,
    right_foot: np.ndarray | None,
    left_weight: float,
    right_weight: float,
) -> np.ndarray:
    if left_foot is None and right_foot is None:
        raise ValueError("Both foot anchors are missing.")
    if left_foot is None:
        return np.asarray(right_foot, dtype=np.float32)
    if right_foot is None:
        return np.asarray(left_foot, dtype=np.float32)
    lw = max(float(left_weight), 0.0)
    rw = max(float(right_weight), 0.0)
    total = max(lw + rw, 1e-6)
    lw /= total
    rw /= total
    anchor = np.array(
        [
            float(lw * float(left_foot[0]) + rw * float(right_foot[0])),
            float(lw * float(left_foot[1]) + rw * float(right_foot[1])),
            float(min(float(left_foot[2]), float(right_foot[2]))),
        ],
        dtype=np.float32,
    )
    return anchor


def _support_planes_match_for_lock(
    *,
    previous_plane: SupportSurfaceResolution | None,
    current_plane: SupportSurfaceResolution | None,
    comparison_anchor: np.ndarray,
    spec: SceneSpec | None = None,
) -> tuple[bool, str | None]:
    if previous_plane is None or current_plane is None:
        return False, "missing_plane"
    previous_normal = np.asarray(previous_plane.normal, dtype=np.float32).reshape(3)
    current_normal = np.asarray(current_plane.normal, dtype=np.float32).reshape(3)
    cos_sim = float(np.clip(np.dot(previous_normal, current_normal), -1.0, 1.0))
    normal_delta_deg = float(np.degrees(np.arccos(cos_sim)))
    normal_tol_deg = float(getattr(spec, "support_anchor_same_plane_normal_tol_deg", 3.0))
    if normal_delta_deg > normal_tol_deg:
        return False, f"normal_delta_deg={normal_delta_deg:.4f}"
    anchor = np.asarray(comparison_anchor, dtype=np.float32).reshape(3)
    previous_signed = float(np.dot(previous_normal, anchor) + float(previous_plane.offset))
    current_signed = float(np.dot(current_normal, anchor) + float(current_plane.offset))
    height_delta = abs(previous_signed - current_signed)
    height_tol_m = float(getattr(spec, "support_anchor_same_plane_height_tol_m", 0.015))
    if height_delta > height_tol_m:
        return False, f"height_delta_m={height_delta:.6f}"
    return True, None


def _extract_support_point_for_foot(
    foot: str,
    *,
    left_after: np.ndarray | None,
    right_after: np.ndarray | None,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    if foot == "left" and left_after is not None:
        return _apply_sole_offset_to_support_anchor(left_after, plane_normal)
    if foot == "right" and right_after is not None:
        return _apply_sole_offset_to_support_anchor(right_after, plane_normal)
    return None


def _resolve_support_anchor_confidences(
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    *,
    plane: RoadPlaneSpec,
    spec: SceneSpec | None = None,
    previous_confidences: tuple[float, float] | None = None,
) -> tuple[float, float]:
    dual_tol = float(
        getattr(spec, "support_anchor_dual_support_height_tol_m", 0.035)
    )
    plane_scale = max(
        0.015,
        min(
            0.05,
            max(float(getattr(spec, "foot_contact_max_plane_dist_m", 0.08)) * 0.5, 0.02),
        ),
    )
    left_signed = float(np.dot(plane.normal, left_foot) + plane.offset)
    right_signed = float(np.dot(plane.normal, right_foot) + plane.offset)
    left_plane_score = math.exp(-abs(left_signed) / plane_scale)
    right_plane_score = math.exp(-abs(right_signed) / plane_scale)
    min_z = min(float(left_foot[2]), float(right_foot[2]))
    height_scale = max(dual_tol, 0.025)
    left_height_score = math.exp(-(float(left_foot[2]) - min_z) / height_scale)
    right_height_score = math.exp(-(float(right_foot[2]) - min_z) / height_scale)
    raw_left = float(left_plane_score * left_height_score)
    raw_right = float(right_plane_score * right_height_score)
    total = max(raw_left + raw_right, 1e-6)
    left_conf = raw_left / total
    right_conf = raw_right / total
    if previous_confidences is not None:
        alpha = 1.0 / max(int(getattr(spec, "support_anchor_transfer_frames", 3)), 1)
        prev_left, prev_right = previous_confidences
        left_conf = float((1.0 - alpha) * float(prev_left) + alpha * left_conf)
        right_conf = float((1.0 - alpha) * float(prev_right) + alpha * right_conf)
        total = max(left_conf + right_conf, 1e-6)
        left_conf /= total
        right_conf /= total
    return float(left_conf), float(right_conf)


def _resolve_support_anchor_selection(
    left_foot: np.ndarray | None,
    right_foot: np.ndarray | None,
    *,
    plane: RoadPlaneSpec,
    spec: SceneSpec | None = None,
    previous_label: str | None = None,
    previous_confidences: tuple[float, float] | None = None,
) -> SupportAnchorSelection:
    if left_foot is None and right_foot is None:
        raise ValueError("Both foot anchors are missing.")
    if left_foot is None:
        anchor = np.asarray(right_foot, dtype=np.float32)
        return SupportAnchorSelection(
            anchor=anchor,
            left_weight=0.0,
            right_weight=1.0,
            label="right",
            left_confidence=0.0,
            right_confidence=1.0,
            switch_decision="missing_left_foot",
            transfer_state="single_support",
        )
    if right_foot is None:
        anchor = np.asarray(left_foot, dtype=np.float32)
        return SupportAnchorSelection(
            anchor=anchor,
            left_weight=1.0,
            right_weight=0.0,
            label="left",
            left_confidence=1.0,
            right_confidence=0.0,
            switch_decision="missing_right_foot",
            transfer_state="single_support",
        )

    left = np.asarray(left_foot, dtype=np.float32)
    right = np.asarray(right_foot, dtype=np.float32)
    left_conf, right_conf = _resolve_support_anchor_confidences(
        left,
        right,
        plane=plane,
        spec=spec,
        previous_confidences=previous_confidences,
    )
    dual_tol = float(
        getattr(spec, "support_anchor_dual_support_height_tol_m", 0.035)
    )
    switch_margin = float(getattr(spec, "support_anchor_switch_margin", 0.12))
    height_gap = abs(float(left[2]) - float(right[2]))
    dominant_label = "left" if left_conf >= right_conf else "right"
    confidence_gap = abs(float(left_conf) - float(right_conf))
    label: str
    switch_decision: str
    transfer_state = "single_support"

    if height_gap <= dual_tol and (
        previous_label not in {"left", "right"} or confidence_gap <= switch_margin
    ):
        label = "both"
        switch_decision = "dual_support_close_height"
        transfer_state = "transfer"
    elif (
        previous_label in {"left", "right"}
        and previous_label != dominant_label
        and confidence_gap < switch_margin
    ):
        label = str(previous_label)
        switch_decision = "held_previous_hysteresis"
        transfer_state = "transfer"
    else:
        label = dominant_label
        switch_decision = "dominant_support"

    if label == "left":
        return SupportAnchorSelection(
            anchor=np.asarray(left, dtype=np.float32),
            left_weight=1.0,
            right_weight=0.0,
            label="left",
            left_confidence=float(left_conf),
            right_confidence=float(right_conf),
            switch_decision=switch_decision,
            transfer_state=transfer_state,
        )
    if label == "right":
        return SupportAnchorSelection(
            anchor=np.asarray(right, dtype=np.float32),
            left_weight=0.0,
            right_weight=1.0,
            label="right",
            left_confidence=float(left_conf),
            right_confidence=float(right_conf),
            switch_decision=switch_decision,
            transfer_state=transfer_state,
        )

    total = max(float(left_conf + right_conf), 1e-6)
    left_weight = float(left_conf / total)
    right_weight = float(right_conf / total)
    anchor = np.array(
        [
            float(left_weight * left[0] + right_weight * right[0]),
            float(left_weight * left[1] + right_weight * right[1]),
            float(min(left[2], right[2])),
        ],
        dtype=np.float32,
    )
    return SupportAnchorSelection(
        anchor=anchor,
        left_weight=left_weight,
        right_weight=right_weight,
        label="both",
        left_confidence=float(left_conf),
        right_confidence=float(right_conf),
        switch_decision=switch_decision,
        transfer_state=transfer_state,
    )


def _blend_support_anchor(
    left_foot: np.ndarray | None,
    right_foot: np.ndarray | None,
    *,
    plane: RoadPlaneSpec,
    previous_weights: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float, float, str]:
    selection = _resolve_support_anchor_selection(
        left_foot,
        right_foot,
        plane=plane,
        previous_confidences=previous_weights,
    )
    return (
        np.asarray(selection.anchor, dtype=np.float32),
        float(selection.left_weight),
        float(selection.right_weight),
        str(selection.label),
    )


def _filter_support_anchor_height_state(
    *,
    anchor_world: np.ndarray,
    previous_anchor_world: np.ndarray | None,
    previous_plane: SupportSurfaceResolution | None,
    current_plane: RoadPlaneSpec,
    spec: SceneSpec | None = None,
) -> SupportAnchorHeightFilterResult:
    raw_anchor = np.asarray(anchor_world, dtype=np.float32).reshape(3)
    raw_height = float(raw_anchor[2])
    if previous_anchor_world is None or previous_plane is None:
        return SupportAnchorHeightFilterResult(
            anchor=raw_anchor,
            raw_height=raw_height,
            filtered_height=raw_height,
            clamped=False,
        )
    prev_anchor = np.asarray(previous_anchor_world, dtype=np.float32).reshape(3)
    current_normal = np.asarray(current_plane.normal, dtype=np.float32).reshape(3)
    previous_normal = np.asarray(previous_plane.normal, dtype=np.float32).reshape(3)
    cos_sim = float(np.clip(np.dot(current_normal, previous_normal), -1.0, 1.0))
    normal_jump_deg = float(np.degrees(np.arccos(cos_sim)))
    current_signed = float(np.dot(current_normal, raw_anchor) + float(current_plane.offset))
    previous_signed = float(np.dot(current_normal, prev_anchor) + float(current_plane.offset))
    flat_ground_normal_z_min = float(
        getattr(spec, "support_anchor_flat_ground_normal_z_min", 0.97)
    )
    is_flat = (
        abs(float(current_normal[2])) >= flat_ground_normal_z_min
        and abs(float(previous_normal[2])) >= flat_ground_normal_z_min
    )
    plane_change_height_tol_m = float(
        getattr(spec, "support_anchor_plane_change_height_tol_m", 0.04)
    )
    plane_changed = (
        normal_jump_deg > 2.0
        or abs(current_signed - previous_signed) > plane_change_height_tol_m
        or not is_flat
    )
    if plane_changed and bool(
        getattr(spec, "support_anchor_allow_vertical_motion_on_plane_change", True)
    ):
        return SupportAnchorHeightFilterResult(
            anchor=raw_anchor,
            raw_height=raw_height,
            filtered_height=raw_height,
            clamped=False,
        )
    max_z_step = float(getattr(spec, "support_anchor_max_z_step_m", 0.01))
    filtered_anchor = raw_anchor.copy()
    filtered_anchor[2] = float(
        np.clip(raw_anchor[2], prev_anchor[2] - max_z_step, prev_anchor[2] + max_z_step)
    )
    filtered_height = float(filtered_anchor[2])
    return SupportAnchorHeightFilterResult(
        anchor=filtered_anchor.astype(np.float32),
        raw_height=raw_height,
        filtered_height=filtered_height,
        clamped=abs(filtered_height - raw_height) > 1e-6,
    )


def _filter_support_anchor_height(
    *,
    anchor_world: np.ndarray,
    previous_anchor_world: np.ndarray | None,
    previous_plane: SupportSurfaceResolution | None,
    current_plane: RoadPlaneSpec,
) -> tuple[np.ndarray, float, float]:
    result = _filter_support_anchor_height_state(
        anchor_world=anchor_world,
        previous_anchor_world=previous_anchor_world,
        previous_plane=previous_plane,
        current_plane=current_plane,
    )
    return result.anchor, result.raw_height, result.filtered_height


def _closest_persisted_plane_for_point(
    point_world: np.ndarray,
    *,
    planes: dict[int, RoadPlaneSpec],
    current_frame_index: int,
    spec: SceneSpec,
    trajectory_c2w: np.ndarray,
) -> tuple[RoadPlaneSpec | None, PersistedPlaneLocalityDecision]:
    ranked_candidates, locality = _rank_persisted_support_plane_candidates(
        spec=spec,
        frame_idx=current_frame_index,
        support_anchor_world=point_world,
        planes=planes,
        trajectory_c2w=trajectory_c2w,
    )
    if not ranked_candidates:
        return None, locality
    return ranked_candidates[0][1], locality


def _solve_plane_height_at_xy(
    *,
    normal: np.ndarray,
    offset: float,
    xy_world: np.ndarray,
) -> float:
    plane_normal = np.asarray(normal, dtype=np.float32).reshape(3)
    xy = np.asarray(xy_world, dtype=np.float32).reshape(2)
    if abs(float(plane_normal[2])) <= 1e-6:
        raise ValueError("Support plane has near-zero z normal and cannot define height.")
    z_value = -(
        float(offset)
        + float(plane_normal[0]) * float(xy[0])
        + float(plane_normal[1]) * float(xy[1])
    ) / float(plane_normal[2])
    if not np.isfinite(z_value):
        raise ValueError("Support plane height solve produced a non-finite z value.")
    return float(z_value)


def _build_actor_support_contract(
    *,
    root_positions_world: Sequence[np.ndarray],
    left_feet_world: Sequence[np.ndarray | None],
    right_feet_world: Sequence[np.ndarray | None],
) -> ActorSupportContract:
    samples: list[float] = []
    for root_world, left_foot, right_foot in zip(
        root_positions_world,
        left_feet_world,
        right_feet_world,
        strict=False,
    ):
        candidates = [
            float(np.asarray(foot, dtype=np.float32)[2])
            for foot in (left_foot, right_foot)
            if foot is not None
        ]
        if not candidates:
            continue
        root_z = float(np.asarray(root_world, dtype=np.float32).reshape(3)[2])
        samples.append(root_z - min(candidates) + float(_SOLE_OFFSET_M))
    if not samples:
        raise ValueError(
            "Unable to derive an asset-native root-to-support offset from the imported character."
        )
    offset = float(np.median(np.asarray(samples, dtype=np.float32)))
    if not np.isfinite(offset):
        raise ValueError("Computed non-finite pedestrian root support offset.")
    return ActorSupportContract(
        root_to_support_m=float(offset),
        support_samples_used=int(len(samples)),
    )


def _resolve_trajectory_support_plane(
    *,
    spec: SceneSpec,
    frame_idx: int,
    support_query_world: np.ndarray,
    road_surface: RoadSurfacePipelineResult,
    trajectory_c2w: np.ndarray,
    previous_plane: RoadPlaneSpec | None,
) -> tuple[RoadPlaneSpec | None, PersistedPlaneLocalityDecision, str]:
    ranked_candidates, locality = _rank_persisted_support_plane_candidates(
        spec=spec,
        frame_idx=int(frame_idx),
        support_anchor_world=np.asarray(support_query_world, dtype=np.float32),
        planes=road_surface.global_planes,
        trajectory_c2w=trajectory_c2w,
    )
    if not ranked_candidates:
        return None, locality, "no_support"
    best_score, best_plane, _best_xy, _best_signed = ranked_candidates[0]
    if previous_plane is not None:
        for score, candidate_plane, _xy_distance, _signed in ranked_candidates:
            if int(candidate_plane.frame_index) != int(previous_plane.frame_index):
                continue
            if float(score) <= float(best_score) + 0.15:
                return candidate_plane, locality, "persisted_path_hysteresis"
            break
    return best_plane, locality, "persisted_path"


def _smooth_grounded_root_heights(
    *,
    raw_root_heights_m: Sequence[float | None],
    segment_ids: Sequence[int | None],
    spec: SceneSpec,
) -> tuple[list[float | None], list[float | None], list[float | None], list[str | None]]:
    fps = float(_resolve_authoritative_sampling_fps(spec, context="Trajectory grounding"))
    max_step_limit = float(getattr(spec, "trajectory_grounding_max_step_m", 0.05))
    max_velocity_mps = float(
        getattr(spec, "trajectory_grounding_max_vertical_velocity_mps", 0.9)
    )
    max_accel_mps2 = float(
        getattr(spec, "trajectory_grounding_max_vertical_accel_mps2", 2.5)
    )
    transition_frames = max(
        int(getattr(spec, "trajectory_grounding_transition_frames", 4)),
        1,
    )
    max_velocity_step = max_velocity_mps / max(fps, 1e-6)
    accel_step_limit = max_accel_mps2 / max(fps * fps, 1e-6)

    smoothed: list[float | None] = [None] * len(raw_root_heights_m)
    velocities: list[float | None] = [None] * len(raw_root_heights_m)
    accelerations: list[float | None] = [None] * len(raw_root_heights_m)
    phases: list[str | None] = [None] * len(raw_root_heights_m)
    previous_velocity_step = 0.0
    previous_segment: int | None = None

    for idx, raw_height in enumerate(raw_root_heights_m):
        if raw_height is None:
            previous_velocity_step = 0.0
            previous_segment = None
            continue
        raw_value = float(raw_height)
        current_segment = segment_ids[idx]
        if idx == 0 or smoothed[idx - 1] is None:
            smoothed[idx] = raw_value
            velocities[idx] = 0.0
            accelerations[idx] = 0.0
            phases[idx] = "initial"
            previous_velocity_step = 0.0
            previous_segment = current_segment
            continue
        previous_height = float(smoothed[idx - 1])
        requested_step = raw_value - previous_height
        plane_changed = (
            current_segment is not None
            and previous_segment is not None
            and int(current_segment) != int(previous_segment)
        )
        transition_step_limit = abs(requested_step) / float(max(transition_frames, 1))
        step_limit = min(
            max_step_limit,
            max(max_velocity_step, transition_step_limit),
        )
        clamped_step = float(np.clip(requested_step, -step_limit, step_limit))
        max_accel_step = max(accel_step_limit, 1e-6)
        lower = previous_velocity_step - max_accel_step
        upper = previous_velocity_step + max_accel_step
        clamped_step = float(np.clip(clamped_step, lower, upper))
        smoothed[idx] = previous_height + clamped_step
        velocities[idx] = clamped_step * fps
        accelerations[idx] = (clamped_step - previous_velocity_step) * fps * fps
        phases[idx] = "transition" if plane_changed else "steady"
        previous_velocity_step = clamped_step
        previous_segment = current_segment
    return smoothed, velocities, accelerations, phases


def _effective_persisted_plane_locality_limit_m(
    *,
    spec: SceneSpec,
    support_anchor_world: np.ndarray,
    trajectory_c2w: np.ndarray,
) -> float:
    base_limit = float(spec.max_plane_center_xy_distance_m)
    if trajectory_c2w.size == 0:
        return base_limit
    corridor_distance = minimum_xy_distance_to_trajectory(
        trajectory_c2w,
        support_anchor_world,
    )
    bootstrap_limit = max(base_limit, float(corridor_distance) + 0.25)
    return float(min(bootstrap_limit, float(spec.global_plane_range_m)))


def _rank_persisted_support_plane_candidates(
    *,
    spec: SceneSpec,
    frame_idx: int,
    support_anchor_world: np.ndarray,
    planes: dict[int, RoadPlaneSpec],
    trajectory_c2w: np.ndarray,
) -> tuple[list[tuple[float, RoadPlaneSpec, float, float]], PersistedPlaneLocalityDecision]:
    base_limit = float(spec.max_plane_center_xy_distance_m)
    effective_limit = _effective_persisted_plane_locality_limit_m(
        spec=spec,
        support_anchor_world=np.asarray(support_anchor_world, dtype=np.float32),
        trajectory_c2w=np.asarray(trajectory_c2w, dtype=np.float32),
    )
    if not planes:
        return [], PersistedPlaneLocalityDecision(
            nearest_xy_distance_m=math.inf,
            effective_limit_m=effective_limit,
            locality_mode="no_planes",
        )

    ranked_candidates: list[tuple[float, RoadPlaneSpec, float, float]] = []
    nearest_xy_distance = math.inf
    for plane in planes.values():
        plane_conf = float(plane.confidence)
        if plane_conf < 0.15:
            continue
        xy_distance = float(
            np.linalg.norm(
                np.asarray(plane.center, dtype=np.float32)[:2]
                - np.asarray(support_anchor_world[:2], dtype=np.float32)
            )
        )
        nearest_xy_distance = min(nearest_xy_distance, xy_distance)
        if xy_distance > effective_limit:
            continue
        signed = float(np.dot(plane.normal, support_anchor_world) + plane.offset)
        temporal_penalty = _soft_temporal_penalty(
            abs(int(plane.frame_index) - int(frame_idx)),
            int(spec.global_plane_frame_window),
        )
        confidence_penalty = 0.0
        threshold = float(spec.global_plane_confidence_threshold)
        if plane_conf < threshold:
            confidence_penalty = float((threshold - plane_conf) * 2.0)
        score = float(xy_distance + abs(signed) * 2.0 + temporal_penalty + confidence_penalty)
        ranked_candidates.append((score, plane, xy_distance, signed))

    if nearest_xy_distance == math.inf:
        locality_mode: Literal["strict", "bootstrap_relaxed", "rejected", "no_planes"] = "no_planes"
    elif nearest_xy_distance <= base_limit:
        locality_mode = "strict"
    elif nearest_xy_distance <= effective_limit:
        locality_mode = "bootstrap_relaxed"
    else:
        locality_mode = "rejected"

    ranked_candidates.sort(
        key=lambda item: (item[0], item[2], abs(int(item[1].frame_index) - int(frame_idx)))
    )
    return ranked_candidates, PersistedPlaneLocalityDecision(
        nearest_xy_distance_m=float(nearest_xy_distance),
        effective_limit_m=effective_limit,
        locality_mode=locality_mode,
    )


def _derive_persisted_plane_confidence(metadata: dict[str, Any] | None) -> float:
    meta = metadata if isinstance(metadata, dict) else {}
    explicit = meta.get("measurement_quality_score")
    if explicit is not None:
        try:
            return float(np.clip(float(explicit), 0.0, 1.0))
        except (TypeError, ValueError):
            pass
    support_ok = meta.get("support_quality_ok")
    inlier_ratio = meta.get("inlier_ratio")
    residual_p90 = meta.get("residual_p90")
    confidence = 0.6
    if isinstance(support_ok, (bool, np.bool_)):
        confidence = 0.7 if bool(support_ok) else 0.2
    try:
        if inlier_ratio is not None:
            confidence *= float(np.clip(float(inlier_ratio), 0.0, 1.0))
    except (TypeError, ValueError):
        pass
    try:
        if residual_p90 is not None:
            confidence *= float(1.0 / (1.0 + max(float(residual_p90), 0.0) * 4.0))
    except (TypeError, ValueError):
        pass
    return float(np.clip(confidence, 0.0, 1.0))


def _soft_temporal_penalty(frame_distance: int, preferred_window: int) -> float:
    excess = max(0, int(frame_distance) - max(int(preferred_window), 0))
    if excess <= 0:
        return 0.0
    return float(excess) / float(max(int(preferred_window), 1) + 1)


def _score_support_source_frame(
    *,
    frame_idx: int,
    candidate_idx: int,
    support_anchor_world: np.ndarray,
    intrinsics_k: np.ndarray,
    c2w: np.ndarray,
    image_shape: tuple[int, int],
    preferred_window: int,
) -> tuple[float, np.ndarray]:
    uv, valid = project_world_to_image(
        np.asarray([support_anchor_world], dtype=np.float32),
        intrinsics_k,
        camera_to_world_matrix=c2w,
        camera_convention="blender",
    )
    if not bool(valid[0]):
        raise ValueError("support anchor projects outside candidate view")
    uv0 = np.asarray(uv[0], dtype=np.float32)
    height, width = int(image_shape[0]), int(image_shape[1])
    center = np.asarray([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float32)
    center_norm = float(np.linalg.norm((uv0 - center) / np.maximum(center, 1.0)))
    edge_margin = min(
        float(uv0[0]),
        float((width - 1) - uv0[0]),
        float(uv0[1]),
        float((height - 1) - uv0[1]),
    )
    edge_penalty = 0.0 if edge_margin >= 8.0 else (8.0 - edge_margin) / 8.0
    cam_xy = float(np.linalg.norm(np.asarray(c2w[:3, 3], dtype=np.float32)[:2] - support_anchor_world[:2]))
    temporal_penalty = _soft_temporal_penalty(
        abs(int(candidate_idx) - int(frame_idx)),
        preferred_window,
    )
    score = cam_xy + center_norm * 0.35 + edge_penalty * 2.0 + temporal_penalty
    return float(score), uv0


def _rank_local_support_source_frames(
    *,
    frame_idx: int,
    support_anchor_world: np.ndarray,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    frame_data_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
    preferred_window: int,
) -> list[int]:
    ranked: list[tuple[float, int]] = []
    for candidate, c2w in frame_to_c2w.items():
        if candidate not in frame_data_cache:
            continue
        depth, _, _, _ = frame_data_cache[candidate]
        try:
            score, _ = _score_support_source_frame(
                frame_idx=frame_idx,
                candidate_idx=int(candidate),
                support_anchor_world=support_anchor_world,
                intrinsics_k=intrinsics_k,
                c2w=c2w,
                image_shape=depth.shape,
                preferred_window=preferred_window,
            )
        except ValueError:
            continue
        ranked.append((float(score), int(candidate)))
    ranked.sort(key=lambda item: (item[0], abs(item[1] - int(frame_idx)), item[1]))
    if not ranked:
        return []
    keep = max(3, 2 * max(int(preferred_window), 0) + 1)
    return [candidate for _, candidate in ranked[:keep]]


def _load_depth_and_semantics_for_frame(
    *,
    run_dir: Path,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    depth_path = run_dir / "standard" / "depth" / f"{frame_idx:06d}.npz"
    semantics_path = run_dir / "standard" / "semantics_2d" / f"{frame_idx:06d}.npz"
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth not found for frame {frame_idx}: {depth_path}")
    if not semantics_path.exists():
        raise FileNotFoundError(f"Semantics not found for frame {frame_idx}: {semantics_path}")
    with np.load(depth_path, allow_pickle=True) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        confidence = (
            np.asarray(data["confidence"], dtype=np.float32)
            if "confidence" in data.files
            else np.ones_like(depth, dtype=np.float32)
        )
    with np.load(semantics_path, allow_pickle=True) as data:
        label_ids = np.asarray(data["label_ids"], dtype=np.int32)
        metadata = _coerce_metadata(data["metadata"]) if "metadata" in data.files else {}
    return depth, confidence, label_ids, metadata


def _resolve_road_label_ids(
    metadata: dict,
    *,
    road_labels: Sequence[str],
) -> set[int]:
    label_map = metadata.get("class_id_to_label")
    if not isinstance(label_map, dict):
        return set()
    wanted = {str(label).strip().lower() for label in road_labels}
    result: set[int] = set()
    for key, value in label_map.items():
        try:
            label_id = int(key)
        except Exception:
            continue
        if str(value).strip().lower() in wanted:
            result.add(label_id)
    return result


def _label_map_from_segments_info(segments_info: np.ndarray) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for item in np.asarray(segments_info, dtype=object).tolist():
        if not isinstance(item, dict):
            continue
        label_id = item.get("label_id")
        label = item.get("label")
        try:
            label_id_int = int(label_id)
        except Exception:
            continue
        if label is None:
            continue
        text = str(label).strip()
        if text:
            label_map[label_id_int] = text
    return label_map


def _resolve_label_ids_from_label_map(
    label_map: Mapping[int, str],
    *,
    labels: Sequence[str],
) -> set[int]:
    wanted = {str(label).strip().lower() for label in labels if str(label).strip()}
    result: set[int] = set()
    for label_id, label_name in label_map.items():
        if str(label_name).strip().lower() in wanted:
            result.add(int(label_id))
    return result


def _weighted_plane_fit(
    points_world: np.ndarray,
    weights: np.ndarray,
    *,
    trim_ratio: float,
    camera_pos: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    pts = np.asarray(points_world, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("Need at least 3 support points for plane fit.")
    if w.shape[0] != pts.shape[0]:
        raise ValueError("Plane-fit weights length mismatch.")
    w = np.clip(w, 1e-6, None)

    def _solve(p: np.ndarray, ww: np.ndarray) -> tuple[np.ndarray, float]:
        centroid = np.average(p, axis=0, weights=ww)
        centered = p - centroid
        cov = (centered * ww[:, None]).T @ centered / max(float(np.sum(ww)), 1e-6)
        evals, evecs = np.linalg.eigh(cov.astype(np.float64))
        normal = np.asarray(evecs[:, int(np.argmin(evals))], dtype=np.float32)
        n_norm = float(np.linalg.norm(normal))
        if n_norm <= 1e-6:
            raise ValueError("Degenerate plane normal.")
        normal = normal / n_norm
        offset = -float(np.dot(normal, centroid))
        signed_cam = float(np.dot(normal, camera_pos) + offset)
        if signed_cam < 0.0:
            normal = -normal
            offset = -offset
        return normal.astype(np.float32), float(offset)

    normal, offset = _solve(pts, w)
    residuals = np.abs(pts @ normal + offset).astype(np.float32)
    if 0.0 < float(trim_ratio) < 1.0 and pts.shape[0] >= 6:
        keep = max(3, int(math.ceil(pts.shape[0] * (1.0 - float(trim_ratio)))))
        order = np.argsort(residuals)
        keep_idx = order[:keep]
        pts_keep = pts[keep_idx]
        w_keep = w[keep_idx]
        normal, offset = _solve(pts_keep, w_keep)
        residuals = np.abs(pts_keep @ normal + offset).astype(np.float32)
        pts = pts_keep
    inlier_threshold = max(0.03, float(np.median(residuals)) * 1.5 if residuals.size else 0.03)
    inlier_ratio = float(np.mean(residuals <= inlier_threshold)) if residuals.size else 0.0
    return normal.astype(np.float32), float(offset), residuals.astype(np.float32), float(inlier_ratio)


def _load_support_fit_context(
    spec: SceneSpec,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray, dict]]]:
    intrinsics_path = spec.run_dir / "standard" / "intrinsics" / "intrinsics.npz"
    with np.load(intrinsics_path, allow_pickle=True) as data:
        intrinsics_k = np.asarray(data["matrix"], dtype=np.float32)
    if intrinsics_k.shape != (3, 3):
        raise ValueError(f"Invalid intrinsics matrix shape: {intrinsics_k.shape}")
    c2w, loaded_frames = load_trajectory(spec.trajectory_path)
    frame_to_c2w = {
        int(frame_idx): np.asarray(c2w[idx], dtype=np.float32)
        for idx, frame_idx in enumerate(loaded_frames.tolist())
    }
    caches: dict[int, tuple[np.ndarray, np.ndarray, dict]] = {}
    for frame_idx in frame_indices.tolist():
        frame_int = int(frame_idx)
        depth, confidence, label_ids, metadata = _load_depth_and_semantics_for_frame(
            run_dir=spec.run_dir,
            frame_idx=frame_int,
        )
        caches[frame_int] = (depth, confidence, label_ids, metadata)
    return intrinsics_k, frame_to_c2w, caches


def _find_nearest_local_road_support_point(
    *,
    spec: SceneSpec,
    support_anchor_world: np.ndarray,
    candidate_frames: Sequence[int],
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    frame_data_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
) -> np.ndarray | None:
    road_names = tuple(spec.road_labels)
    best_point: np.ndarray | None = None
    best_distance = math.inf
    for candidate in candidate_frames:
        c2w = frame_to_c2w.get(int(candidate))
        if c2w is None:
            continue
        depth, confidence, label_ids, metadata = frame_data_cache[int(candidate)]
        road_label_ids = _resolve_road_label_ids(metadata, road_labels=road_names)
        if not road_label_ids:
            continue
        road_mask = np.isin(label_ids, list(road_label_ids))
        depth_mask = np.isfinite(depth) & (depth > 0.0)
        conf_mask = confidence >= float(spec.local_support_confidence_threshold)
        base_mask = road_mask & depth_mask & conf_mask
        if not np.any(base_mask):
            continue
        ys, xs = np.where(base_mask)
        if xs.size == 0:
            continue
        max_points = max(32, int(spec.global_plane_max_points_per_frame))
        step = max(1, int(math.ceil(math.sqrt(xs.size / max_points))))
        xs = xs[::step]
        ys = ys[::step]
        depths = depth[ys, xs]
        uv_all = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        cam_points = backproject_uv_depth_to_camera(
            uv_all,
            depths.astype(np.float32),
            intrinsics_k,
            camera_convention="blender",
        )
        world_points = camera_to_world(cam_points, c2w)
        xy_dist = np.linalg.norm(
            world_points[:, :2] - np.asarray(support_anchor_world[:2], dtype=np.float32),
            axis=1,
        )
        z_delta = np.abs(world_points[:, 2] - float(support_anchor_world[2]))
        keep = (
            (xy_dist <= float(spec.local_support_snap_radius_m))
            & (z_delta <= float(spec.local_support_snap_max_vertical_delta_m))
        )
        if not np.any(keep):
            continue
        candidate_points = world_points[keep]
        candidate_xy = xy_dist[keep]
        best_idx = int(np.argmin(candidate_xy))
        candidate_distance = float(candidate_xy[best_idx])
        if candidate_distance < best_distance:
            best_distance = candidate_distance
            best_point = np.asarray(candidate_points[best_idx], dtype=np.float32)
    return best_point


def _collect_local_support_points(
    *,
    spec: SceneSpec,
    frame_idx: int,
    support_anchor_world: np.ndarray,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    frame_data_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, ...]]:
    candidates = _rank_local_support_source_frames(
        frame_idx=frame_idx,
        support_anchor_world=np.asarray(support_anchor_world, dtype=np.float32),
        intrinsics_k=intrinsics_k,
        frame_to_c2w=frame_to_c2w,
        frame_data_cache=frame_data_cache,
        preferred_window=int(spec.local_support_frame_window),
    )
    if not candidates:
        raise ValueError("local_fit_no_visible_support_frames")
    current_radius = float(
        max(float(spec.local_support_radius_m), float(spec.local_support_plane_size_m) * 0.5)
    )
    used_frames: tuple[int, ...] = tuple()
    last_points = np.zeros((0, 3), dtype=np.float32)
    last_weights = np.zeros((0,), dtype=np.float32)
    road_names = tuple(spec.road_labels)
    query_anchor = np.asarray(support_anchor_world, dtype=np.float32)
    snap_attempted = not bool(spec.local_support_snap_to_nearest_road)
    while current_radius <= float(spec.local_support_max_radius_m) + 1e-6:
        point_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        used: list[int] = []
        for candidate in candidates:
            c2w = frame_to_c2w[candidate]
            uv, valid = project_world_to_image(
                np.asarray([query_anchor], dtype=np.float32),
                intrinsics_k,
                camera_to_world_matrix=c2w,
                camera_convention="blender",
            )
            if not bool(valid[0]):
                continue
            depth, confidence, label_ids, metadata = frame_data_cache[candidate]
            road_label_ids = _resolve_road_label_ids(metadata, road_labels=road_names)
            if not road_label_ids:
                continue
            road_mask = np.isin(label_ids, list(road_label_ids))
            depth_mask = np.isfinite(depth) & (depth > 0.0)
            conf_mask = confidence >= float(spec.local_support_confidence_threshold)
            base_mask = road_mask & depth_mask & conf_mask
            if not np.any(base_mask):
                continue
            ys, xs = np.where(base_mask)
            count = xs.size
            if count == 0:
                continue
            max_points = max(32, int(spec.global_plane_max_points_per_frame))
            step = max(1, int(math.ceil(math.sqrt(count / max_points))))
            xs = xs[::step]
            ys = ys[::step]
            depths = depth[ys, xs]
            confs = np.clip(confidence[ys, xs], 1e-3, 1.0)
            uv_all = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
            cam_points = backproject_uv_depth_to_camera(
                uv_all,
                depths.astype(np.float32),
                intrinsics_k,
                camera_convention="blender",
            )
            world_points = camera_to_world(cam_points, c2w)
            xy_dist = np.linalg.norm(
                world_points[:, :2] - np.asarray(query_anchor[:2], dtype=np.float32),
                axis=1,
            )
            z_delta = np.abs(world_points[:, 2] - float(query_anchor[2]))
            keep = (
                (xy_dist <= float(current_radius))
                & (z_delta <= float(spec.local_support_prefilter_vertical_window_m))
            )
            if not np.any(keep):
                continue
            pts_keep = world_points[keep].astype(np.float32)
            w_keep = confs[keep].astype(np.float32)
            point_chunks.append(pts_keep)
            weight_chunks.append(w_keep)
            used.append(int(candidate))
        if point_chunks:
            points = np.concatenate(point_chunks, axis=0)
            weights = np.concatenate(weight_chunks, axis=0)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            weights = np.zeros((0,), dtype=np.float32)
        last_points = points
        last_weights = weights
        used_frames = tuple(sorted(set(used)))
        if points.shape[0] >= int(spec.local_support_min_points):
            return points, weights, float(current_radius), used_frames
        if (
            points.shape[0] == 0
            and not snap_attempted
        ):
            snapped_point = _find_nearest_local_road_support_point(
                spec=spec,
                support_anchor_world=query_anchor,
                candidate_frames=candidates,
                intrinsics_k=intrinsics_k,
                frame_to_c2w=frame_to_c2w,
                frame_data_cache=frame_data_cache,
            )
            snap_attempted = True
            if snapped_point is not None:
                snap_xy = float(
                    np.linalg.norm(
                        np.asarray(snapped_point[:2], dtype=np.float32)
                        - np.asarray(query_anchor[:2], dtype=np.float32)
                    )
                )
                if snap_xy <= float(spec.local_support_snap_max_radius_ratio) * float(current_radius):
                    query_anchor = np.asarray(snapped_point, dtype=np.float32)
                    current_radius = float(
                        max(
                            float(spec.local_support_radius_m),
                            float(spec.local_support_plane_size_m) * 0.5,
                        )
                    )
                    continue
        current_radius += float(spec.local_support_radius_step_m)
    if last_points.shape[0] < int(spec.local_support_min_points):
        raise ValueError(
            f"local_fit_insufficient_points: got {last_points.shape[0]}, "
            f"need >= {spec.local_support_min_points}"
        )
    return last_points, last_weights, float(min(current_radius, spec.local_support_max_radius_m)), used_frames


def _fit_local_support_plane(
    *,
    spec: SceneSpec,
    frame_idx: int,
    support_anchor_world: np.ndarray,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    frame_data_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
) -> SupportSurfaceResolution:
    points, weights, used_radius, source_frames = _collect_local_support_points(
        spec=spec,
        frame_idx=frame_idx,
        support_anchor_world=support_anchor_world,
        intrinsics_k=intrinsics_k,
        frame_to_c2w=frame_to_c2w,
        frame_data_cache=frame_data_cache,
    )
    camera_pos = np.asarray(frame_to_c2w[int(frame_idx)][:3, 3], dtype=np.float32)
    try:
        normal, offset, residuals, inlier_ratio = _weighted_plane_fit(
            points,
            weights,
            trim_ratio=float(spec.global_plane_trim_ratio),
            camera_pos=camera_pos,
        )
    except ValueError as exc:
        raise ValueError(f"local_fit_quality_failed: {exc}") from exc
    residual_p90 = float(np.percentile(residuals, 90)) if residuals.size else math.inf
    pre_support_dist = abs(float(np.dot(normal, support_anchor_world) + offset))
    if residual_p90 > _LOCAL_SUPPORT_MAX_RESIDUAL_P90_M:
        raise ValueError(f"local_fit_quality_failed: residual_p90={residual_p90:.4f}m")
    if inlier_ratio < _LOCAL_SUPPORT_MIN_INLIER_RATIO:
        raise ValueError(f"local_fit_quality_failed: inlier_ratio={inlier_ratio:.3f}")
    if pre_support_dist > _LOCAL_SUPPORT_MAX_PRE_SUPPORT_DIST_M:
        raise ValueError(
            f"local_fit_anchor_distance_too_high: {pre_support_dist:.4f}m"
        )
    confidence = float(
        np.clip(
            1.0 / (1.0 + residual_p90 * 8.0) * (0.5 + 0.5 * inlier_ratio),
            0.0,
            1.0,
        )
    )
    return SupportSurfaceResolution(
        mode="local_fit",
        normal=np.asarray(normal, dtype=np.float32),
        offset=float(offset),
        confidence=confidence,
        source_frame_indices=tuple(int(v) for v in source_frames),
        local_fit_point_count=int(points.shape[0]),
        local_fit_radius_m=float(used_radius),
        local_fit_residual_p90_m=float(residual_p90),
        local_fit_inlier_ratio=float(inlier_ratio),
        persisted_blend_candidate_count=None,
        persisted_blend_disagreement_m=None,
        held_from_previous=False,
        origin_mode="local_fit",
        failure_reason=None,
    )


def _blend_persisted_support_planes(
    *,
    spec: SceneSpec,
    frame_idx: int,
    support_anchor_world: np.ndarray,
    planes: dict[int, RoadPlaneSpec],
    trajectory_c2w: np.ndarray,
) -> SupportSurfaceResolution:
    ranked_candidates, locality = _rank_persisted_support_plane_candidates(
        spec=spec,
        frame_idx=frame_idx,
        support_anchor_world=support_anchor_world,
        planes=planes,
        trajectory_c2w=trajectory_c2w,
    )
    if not ranked_candidates:
        if locality.locality_mode == "rejected":
            raise ValueError(
                "persisted_fallback_locality_rejected: "
                f"nearest_xy_distance_m={locality.nearest_xy_distance_m:.4f} "
                f"effective_limit_m={locality.effective_limit_m:.4f}"
            )
        raise ValueError("persisted_fallback_no_spatial_candidates")
    keep = max(3, 2 * max(int(spec.global_plane_frame_window), 0) + 1)
    candidates = [(plane, xy_distance, signed) for _, plane, xy_distance, signed in ranked_candidates[:keep]]
    weights = []
    signed_values = []
    normals = []
    frames = []
    for plane, xy_distance, signed in candidates:
        plane_conf = float(plane.confidence)
        threshold = float(spec.global_plane_confidence_threshold)
        confidence_weight = plane_conf
        if plane_conf < threshold:
            confidence_weight *= max(0.2, plane_conf / max(threshold, 1e-6))
        weight = confidence_weight / max(0.05, xy_distance + abs(signed))
        weights.append(weight)
        signed_values.append(signed)
        normals.append(np.asarray(plane.normal, dtype=np.float32))
        frames.append(int(plane.frame_index))
    weights_arr = np.asarray(weights, dtype=np.float32)
    normals_arr = np.asarray(normals, dtype=np.float32)
    blended_normal = np.average(normals_arr, axis=0, weights=weights_arr)
    n_norm = float(np.linalg.norm(blended_normal))
    if n_norm <= 1e-6:
        raise ValueError("persisted_fallback_degenerate_normal")
    blended_normal = (blended_normal / n_norm).astype(np.float32)
    target_signed = float(np.average(np.asarray(signed_values, dtype=np.float32), weights=weights_arr))
    disagreement = float(
        np.max(np.abs(np.asarray(signed_values, dtype=np.float32) - target_signed))
    )
    if disagreement > _PERSISTED_BLEND_MAX_DISAGREEMENT_M:
        raise ValueError(
            f"persisted_fallback_blend_disagreement: {disagreement:.4f}m"
        )
    blended_offset = float(target_signed - np.dot(blended_normal, support_anchor_world))
    confidence = float(
        np.clip(
            float(np.average(np.asarray([p.confidence for p, _, _ in candidates], dtype=np.float32), weights=weights_arr))
            * (1.0 / (1.0 + disagreement * 5.0)),
            0.0,
            1.0,
        )
    )
    return SupportSurfaceResolution(
        mode="persisted_blend",
        normal=np.asarray(blended_normal, dtype=np.float32),
        offset=float(blended_offset),
        confidence=confidence,
        source_frame_indices=tuple(sorted(set(frames))),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=len(candidates),
        persisted_blend_disagreement_m=float(disagreement),
        held_from_previous=False,
        origin_mode="persisted_blend",
        failure_reason=None,
    )


def _apply_sole_offset_to_support_anchor(
    support_anchor_world: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(support_anchor_world, dtype=np.float32)
        - _SOLE_OFFSET_M * np.asarray(plane_normal, dtype=np.float32)
    ).astype(np.float32)


def _resolved_support_origin_mode(
    support_resolution: SupportSurfaceResolution | None,
) -> str | None:
    if support_resolution is None:
        return None
    if support_resolution.origin_mode is not None:
        return str(support_resolution.origin_mode)
    if support_resolution.mode in {"local_fit", "persisted_blend"}:
        return str(support_resolution.mode)
    return None


def _support_planes_are_continuous(
    *,
    normal_jump: float | None,
    height_jump: float | None,
    current_signed: float | None,
    previous_signed: float | None,
) -> bool:
    if (
        normal_jump is None
        or height_jump is None
        or current_signed is None
        or previous_signed is None
    ):
        return False
    signed_delta = abs(float(current_signed) - float(previous_signed))
    return (
        float(normal_jump) <= float(_SUPPORT_MAX_NORMAL_JUMP_DEG)
        and float(height_jump) <= max(float(_SUPPORT_MAX_HEIGHT_JUMP_M), 0.03)
        and signed_delta <= max(float(_SUPPORT_MAX_HEIGHT_JUMP_M), 0.03)
    )


def _support_surface_jump_metrics(
    current: SupportSurfaceResolution,
    previous: SupportSurfaceResolution | None,
    *,
    comparison_anchor: np.ndarray,
    current_anchor: np.ndarray,
    previous_anchor: np.ndarray | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if previous is None or previous_anchor is None:
        return None, None, None, None, None
    metrics = compute_support_relock_metrics(
        current_normal=np.asarray(current.normal, dtype=np.float32),
        current_offset=float(current.offset),
        previous_normal=np.asarray(previous.normal, dtype=np.float32),
        previous_offset=float(previous.offset),
        comparison_anchor=np.asarray(comparison_anchor, dtype=np.float32),
        current_anchor=np.asarray(current_anchor, dtype=np.float32),
        previous_anchor=np.asarray(previous_anchor, dtype=np.float32),
    )
    return (
        metrics.normal_jump_deg,
        metrics.support_height_jump_m,
        metrics.anchor_shift_m,
        metrics.current_signed_distance_m,
        metrics.previous_signed_distance_m,
    )


def _should_allow_support_relock(
    *,
    current: SupportSurfaceResolution,
    normal_jump: float | None,
    height_jump: float | None,
    anchor_shift: float | None,
    current_signed: float | None,
    previous_signed: float | None,
    hold_count: int,
    max_anchor_shift_m: float,
    hard_jump_exceeded: bool,
) -> bool:
    if hard_jump_exceeded and hold_count <= 0:
        return False
    if current.mode == "hold_prev":
        return False
    if float(current.confidence) < _min_support_confidence_for_projection_origin(
        _resolved_support_origin_mode(current)
    ):
        return False
    if _support_planes_are_continuous(
        normal_jump=normal_jump,
        height_jump=height_jump,
        current_signed=current_signed,
        previous_signed=previous_signed,
    ):
        return True
    if normal_jump is not None and normal_jump > (_SUPPORT_MAX_NORMAL_JUMP_DEG * 1.5):
        return False
    if height_jump is not None and height_jump > max(_SUPPORT_MAX_HEIGHT_JUMP_M * 5.0, 0.35):
        return False
    if anchor_shift is not None and anchor_shift > (float(max_anchor_shift_m) * 1.15):
        return False
    return True


def _compute_dynamic_anchor_shift_limit_m(spec: SceneSpec) -> float:
    sampling_fps = float(_resolve_authoritative_sampling_fps(spec, context="Support relock"))
    max_speed = float(spec.foot_contact_max_speed_mps)
    if not np.isfinite(max_speed) or max_speed < 0.0:
        raise ValueError(
            "Support relock requires foot_contact_max_speed_mps to be finite and >= 0, "
            f"got {spec.foot_contact_max_speed_mps!r}."
        )
    expected_step_m = max_speed / max(sampling_fps, 1e-6)
    dynamic_limit = max(
        float(_SUPPORT_MAX_ANCHOR_SHIFT_M),
        (expected_step_m * 1.25) + 0.10,
    )
    if not np.isfinite(dynamic_limit) or dynamic_limit <= 0.0:
        raise ValueError(
            "Computed invalid dynamic support relock anchor-shift limit: "
            f"{dynamic_limit!r}."
        )
    return float(dynamic_limit)


def _resolve_local_support_hold_budget(spec: SceneSpec) -> tuple[int, float | None]:
    sampling_fps = float(
        _resolve_authoritative_sampling_fps(spec, context="Support hold budget")
    )
    return resolve_effective_hold_frames(
        hold_frames=int(spec.local_support_temporal_hold_frames),
        hold_seconds=spec.local_support_temporal_hold_seconds,
        sampling_fps=sampling_fps,
    )


def _hold_previous_support_when_unresolved(
    *,
    previous: SupportSurfaceResolution | None,
    hold_count: int,
    max_hold_frames: int,
    current_anchor: np.ndarray,
    previous_anchor: np.ndarray | None,
) -> tuple[
    SupportSurfaceResolution | None,
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    if previous is None or hold_count >= int(max_hold_frames):
        return None, hold_count, None, None, None, None, None
    anchor_shift = None
    if previous_anchor is not None:
        anchor_shift = float(
            np.linalg.norm(
                np.asarray(current_anchor[:2], dtype=np.float32)
                - np.asarray(previous_anchor[:2], dtype=np.float32)
            )
        )
    held = replace(
        previous,
        mode="hold_prev",
        held_from_previous=True,
    )
    return held, hold_count + 1, None, None, anchor_shift, None, None


def _min_support_confidence_for_projection_origin(origin_mode: str | None) -> float:
    if origin_mode == "persisted_blend":
        return float(_PERSISTED_BLEND_MIN_CONFIDENCE_FOR_PROJECTION)
    return 0.25


def _min_support_confidence_for_projection(
    *,
    spec: SceneSpec,
    support_resolution: SupportSurfaceResolution,
) -> float:
    threshold = float(spec.foot_contact_min_plane_confidence_for_projection)
    if _resolved_support_origin_mode(support_resolution) == "persisted_blend":
        return min(threshold, float(_PERSISTED_BLEND_MIN_CONFIDENCE_FOR_PROJECTION))
    return threshold


def _stabilize_support_surface(
    *,
    current: SupportSurfaceResolution,
    previous: SupportSurfaceResolution | None,
    hold_count: int,
    comparison_anchor: np.ndarray,
    current_anchor: np.ndarray,
    previous_anchor: np.ndarray | None,
    max_hold_frames: int,
    max_anchor_shift_m: float,
) -> tuple[SupportSurfaceResolution, int, float | None, float | None, float | None, float | None, float | None]:
    normal_jump, height_jump, anchor_shift, current_signed, previous_signed = _support_surface_jump_metrics(
        current,
        previous,
        comparison_anchor=comparison_anchor,
        current_anchor=current_anchor,
        previous_anchor=previous_anchor,
    )
    if previous is None:
        return current, 0, normal_jump, height_jump, anchor_shift, current_signed, previous_signed
    plane_continuous = _support_planes_are_continuous(
        normal_jump=normal_jump,
        height_jump=height_jump,
        current_signed=current_signed,
        previous_signed=previous_signed,
    )
    hard_jump_exceeded = (
        (normal_jump is not None and normal_jump > _SUPPORT_MAX_NORMAL_JUMP_DEG)
        or (height_jump is not None and height_jump > _SUPPORT_MAX_HEIGHT_JUMP_M)
    )
    anchor_shift_exceeded = (
        anchor_shift is not None
        and anchor_shift > float(max_anchor_shift_m)
        and not plane_continuous
    )
    if not hard_jump_exceeded and not anchor_shift_exceeded:
        return current, 0, normal_jump, height_jump, anchor_shift, current_signed, previous_signed
    if _should_allow_support_relock(
        current=current,
        normal_jump=normal_jump,
        height_jump=height_jump,
        anchor_shift=anchor_shift,
        current_signed=current_signed,
        previous_signed=previous_signed,
        hold_count=hold_count,
        max_anchor_shift_m=max_anchor_shift_m,
        hard_jump_exceeded=hard_jump_exceeded,
    ):
        return current, 0, normal_jump, height_jump, anchor_shift, current_signed, previous_signed
    if hold_count < int(max_hold_frames):
        held = replace(
            previous,
            mode="hold_prev",
            held_from_previous=True,
        )
        return held, hold_count + 1, normal_jump, height_jump, anchor_shift, current_signed, previous_signed
    raise ValueError(
        "support_relock_rejected: "
        f"normal={normal_jump}deg height={height_jump}m "
        f"anchor_shift={anchor_shift}m dynamic_limit={float(max_anchor_shift_m):.6f}m "
        f"current_signed={current_signed}m previous_signed={previous_signed}m"
    )


def _write_grounding_diagnostics(
    *,
    run_dir: Path,
    diagnostics: list[GroundingDiagnostic],
) -> tuple[Path, Path]:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "grounding_diagnostics.json"
    csv_path = vis_dir / "grounding_diagnostics.csv"
    payload = {
        "count": len(diagnostics),
        "entries": [
            {
                "frame_index": int(d.frame_index),
                "support_mode": str(d.support_mode),
                "support_confidence": (
                    None if d.support_confidence is None else float(d.support_confidence)
                ),
                "support_source_frame_indices": [int(v) for v in d.support_source_frame_indices],
                "support_source_frame_count": int(len(d.support_source_frame_indices)),
                "support_failure_reason": d.support_failure_reason,
                "sole_offset_m": float(d.sole_offset_m),
                "chosen_plane_frame_index": (
                    None if d.chosen_plane_frame_index is None else int(d.chosen_plane_frame_index)
                ),
                "chosen_plane_normal": (
                    None
                    if d.chosen_plane_normal is None
                    else np.asarray(d.chosen_plane_normal, dtype=float).tolist()
                ),
                "chosen_plane_offset": (
                    None if d.chosen_plane_offset is None else float(d.chosen_plane_offset)
                ),
                "chosen_plane_center": (
                    None
                    if d.chosen_plane_center is None
                    else np.asarray(d.chosen_plane_center, dtype=float).tolist()
                ),
                "chosen_plane_center_xy_distance_m": (
                    None
                    if d.chosen_plane_center_xy_distance_m is None
                    else float(d.chosen_plane_center_xy_distance_m)
                ),
                "selected_support_foot": str(d.selected_support_foot),
                "selected_plane_source": d.selected_plane_source,
                "authored_root_world": (
                    None
                    if d.authored_root_world is None
                    else np.asarray(d.authored_root_world, dtype=float).tolist()
                ),
                "grounded_root_world": (
                    None
                    if d.grounded_root_world is None
                    else np.asarray(d.grounded_root_world, dtype=float).tolist()
                ),
                "root_support_offset_m": (
                    None if d.root_support_offset_m is None else float(d.root_support_offset_m)
                ),
                "plane_height_at_xy_m": (
                    None if d.plane_height_at_xy_m is None else float(d.plane_height_at_xy_m)
                ),
                "planned_z_delta_m": (
                    None if d.planned_z_delta_m is None else float(d.planned_z_delta_m)
                ),
                "vertical_velocity_mps": (
                    None if d.vertical_velocity_mps is None else float(d.vertical_velocity_mps)
                ),
                "vertical_accel_mps2": (
                    None if d.vertical_accel_mps2 is None else float(d.vertical_accel_mps2)
                ),
                "traversal_segment_id": (
                    None if d.traversal_segment_id is None else int(d.traversal_segment_id)
                ),
                "plane_transition_phase": d.plane_transition_phase,
                "left_foot_before": (
                    None
                    if d.left_foot_before is None
                    else np.asarray(d.left_foot_before, dtype=float).tolist()
                ),
                "right_foot_before": (
                    None
                    if d.right_foot_before is None
                    else np.asarray(d.right_foot_before, dtype=float).tolist()
                ),
                "left_foot_after": (
                    None
                    if d.left_foot_after is None
                    else np.asarray(d.left_foot_after, dtype=float).tolist()
                ),
                "right_foot_after": (
                    None
                    if d.right_foot_after is None
                    else np.asarray(d.right_foot_after, dtype=float).tolist()
                ),
                "support_point_before": (
                    None
                    if d.support_point_before is None
                    else np.asarray(d.support_point_before, dtype=float).tolist()
                ),
                "support_point_after": (
                    None
                    if d.support_point_after is None
                    else np.asarray(d.support_point_after, dtype=float).tolist()
                ),
                "pre_correction_signed_distance_m": (
                    None
                    if d.pre_correction_signed_distance_m is None
                    else float(d.pre_correction_signed_distance_m)
                ),
                "post_correction_signed_distance_m": (
                    None
                    if d.post_correction_signed_distance_m is None
                    else float(d.post_correction_signed_distance_m)
                ),
                "left_post_signed_distance_m": (
                    None
                    if d.left_post_signed_distance_m is None
                    else float(d.left_post_signed_distance_m)
                ),
                "right_post_signed_distance_m": (
                    None
                    if d.right_post_signed_distance_m is None
                    else float(d.right_post_signed_distance_m)
                ),
                "support_jump_from_prev_deg": (
                    None
                    if d.support_jump_from_prev_deg is None
                    else float(d.support_jump_from_prev_deg)
                ),
                "support_height_jump_from_prev_m": (
                    None
                    if d.support_height_jump_from_prev_m is None
                    else float(d.support_height_jump_from_prev_m)
                ),
                "support_anchor_shift_from_prev_m": (
                    None
                    if d.support_anchor_shift_from_prev_m is None
                    else float(d.support_anchor_shift_from_prev_m)
                ),
                "support_state": str(d.support_state),
                "visibility_contract_state": d.visibility_contract_state,
                "visibility_culled": bool(d.visibility_culled),
                "visibility_cull_reason": d.visibility_cull_reason,
                "frame_requires_support": bool(d.frame_requires_support),
                "previous_support_point_before": (
                    None
                    if d.previous_support_point_before is None
                    else np.asarray(d.previous_support_point_before, dtype=float).tolist()
                ),
                "previous_support_point_after": (
                    None
                    if d.previous_support_point_after is None
                    else np.asarray(d.previous_support_point_after, dtype=float).tolist()
                ),
                "relock_current_signed_distance_m": (
                    None
                    if d.relock_current_signed_distance_m is None
                    else float(d.relock_current_signed_distance_m)
                ),
                "relock_previous_signed_distance_m": (
                    None
                    if d.relock_previous_signed_distance_m is None
                    else float(d.relock_previous_signed_distance_m)
                ),
                "support_origin_mode": d.support_origin_mode,
                "relock_decision_reason": d.relock_decision_reason,
                "nearest_persisted_plane_center_xy_distance_m": (
                    None
                    if d.nearest_persisted_plane_center_xy_distance_m is None
                    else float(d.nearest_persisted_plane_center_xy_distance_m)
                ),
                "effective_persisted_plane_locality_limit_m": (
                    None
                    if d.effective_persisted_plane_locality_limit_m is None
                    else float(d.effective_persisted_plane_locality_limit_m)
                ),
                "persisted_plane_locality_mode": d.persisted_plane_locality_mode,
                "support_anchor_policy": d.support_anchor_policy,
                "left_support_weight": (
                    None if d.left_support_weight is None else float(d.left_support_weight)
                ),
                "right_support_weight": (
                    None if d.right_support_weight is None else float(d.right_support_weight)
                ),
                "support_anchor_blended": (
                    None
                    if d.support_anchor_blended is None
                    else np.asarray(d.support_anchor_blended, dtype=float).tolist()
                ),
                "support_height_raw_m": (
                    None if d.support_height_raw_m is None else float(d.support_height_raw_m)
                ),
                "support_height_filtered_m": (
                    None
                    if d.support_height_filtered_m is None
                    else float(d.support_height_filtered_m)
                ),
                "left_support_confidence": (
                    None
                    if d.left_support_confidence is None
                    else float(d.left_support_confidence)
                ),
                "right_support_confidence": (
                    None
                    if d.right_support_confidence is None
                    else float(d.right_support_confidence)
                ),
                "support_switch_decision": d.support_switch_decision,
                "support_transfer_state": d.support_transfer_state,
                "support_height_clamped": bool(d.support_height_clamped),
                "contact_phase": (
                    None if d.contact_phase is None else float(d.contact_phase)
                ),
                "contact_state_raw": d.contact_state_raw,
                "contact_state_clean": d.contact_state_clean,
                "contact_segment_id": (
                    None if d.contact_segment_id is None else int(d.contact_segment_id)
                ),
                "contact_segment_kind": d.contact_segment_kind,
                "plant_lock_source_frame": (
                    None
                    if d.plant_lock_source_frame is None
                    else int(d.plant_lock_source_frame)
                ),
                "plant_target_world": (
                    None
                    if d.plant_target_world is None
                    else np.asarray(d.plant_target_world, dtype=float).tolist()
                ),
                "plant_lock_error_m": (
                    None if d.plant_lock_error_m is None else float(d.plant_lock_error_m)
                ),
                "plant_lock_xy_error_m": (
                    None if d.plant_lock_xy_error_m is None else float(d.plant_lock_xy_error_m)
                ),
                "support_authority": d.support_authority,
                "same_plane_continuity": (
                    None
                    if d.same_plane_continuity is None
                    else bool(d.same_plane_continuity)
                ),
                "continuity_break_reason": d.continuity_break_reason,
                "dynamic_anchor_shift_limit_m": (
                    None
                    if d.dynamic_anchor_shift_limit_m is None
                    else float(d.dynamic_anchor_shift_limit_m)
                ),
                "applied_translation_world": np.asarray(
                    d.applied_translation_world, dtype=float
                ).tolist(),
                "applied_translation_xy_m": (
                    None
                    if d.applied_translation_xy_m is None
                    else float(d.applied_translation_xy_m)
                ),
                "xy_lock_mode": d.xy_lock_mode,
                "xy_lock_clamped": bool(d.xy_lock_clamped),
                "plane_selection_rejected_for_locality": bool(
                    d.plane_selection_rejected_for_locality
                ),
                "missing_left_foot": bool(d.missing_left_foot),
                "missing_right_foot": bool(d.missing_right_foot),
                "no_plane": bool(d.no_plane),
            }
            for d in diagnostics
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "support_mode",
                "support_confidence",
                "support_source_frame_indices",
                "support_source_frame_count",
                "support_failure_reason",
                "sole_offset_m",
                "chosen_plane_frame_index",
                "plane_normal_x",
                "plane_normal_y",
                "plane_normal_z",
                "plane_offset",
                "plane_center_x",
                "plane_center_y",
                "plane_center_z",
                "chosen_plane_center_xy_distance_m",
                "selected_support_foot",
                "selected_plane_source",
                "authored_root_x",
                "authored_root_y",
                "authored_root_z",
                "grounded_root_x",
                "grounded_root_y",
                "grounded_root_z",
                "root_support_offset_m",
                "plane_height_at_xy_m",
                "planned_z_delta_m",
                "vertical_velocity_mps",
                "vertical_accel_mps2",
                "traversal_segment_id",
                "plane_transition_phase",
                "left_before_x",
                "left_before_y",
                "left_before_z",
                "right_before_x",
                "right_before_y",
                "right_before_z",
                "left_after_x",
                "left_after_y",
                "left_after_z",
                "right_after_x",
                "right_after_y",
                "right_after_z",
                "support_before_x",
                "support_before_y",
                "support_before_z",
                "support_after_x",
                "support_after_y",
                "support_after_z",
                "pre_correction_signed_distance_m",
                "post_correction_signed_distance_m",
                "left_post_signed_distance_m",
                "right_post_signed_distance_m",
                "support_jump_from_prev_deg",
                "support_height_jump_from_prev_m",
                "support_anchor_shift_from_prev_m",
                "support_state",
                "visibility_contract_state",
                "visibility_culled",
                "visibility_cull_reason",
                "frame_requires_support",
                "prev_support_before_x",
                "prev_support_before_y",
                "prev_support_before_z",
                "prev_support_after_x",
                "prev_support_after_y",
                "prev_support_after_z",
                "relock_current_signed_distance_m",
                "relock_previous_signed_distance_m",
                "support_origin_mode",
                "relock_decision_reason",
                "nearest_persisted_plane_center_xy_distance_m",
                "effective_persisted_plane_locality_limit_m",
                "persisted_plane_locality_mode",
                "support_anchor_policy",
                "left_support_weight",
                "right_support_weight",
                "support_anchor_blended_x",
                "support_anchor_blended_y",
                "support_anchor_blended_z",
                "support_height_raw_m",
                "support_height_filtered_m",
                "left_support_confidence",
                "right_support_confidence",
                "support_switch_decision",
                "support_transfer_state",
                "support_height_clamped",
                "contact_phase",
                "contact_state_raw",
                "contact_state_clean",
                "contact_segment_id",
                "contact_segment_kind",
                "plant_lock_source_frame",
                "plant_target_x",
                "plant_target_y",
                "plant_target_z",
                "plant_lock_error_m",
                "plant_lock_xy_error_m",
                "support_authority",
                "same_plane_continuity",
                "continuity_break_reason",
                "dynamic_anchor_shift_limit_m",
                "translation_x",
                "translation_y",
                "translation_z",
                "applied_translation_xy_m",
                "xy_lock_mode",
                "xy_lock_clamped",
                "plane_selection_rejected_for_locality",
                "missing_left_foot",
                "missing_right_foot",
                "no_plane",
            ]
        )
        for d in diagnostics:
            normal = (
                np.asarray(d.chosen_plane_normal, dtype=float).reshape(3)
                if d.chosen_plane_normal is not None
                else [None, None, None]
            )
            center = (
                np.asarray(d.chosen_plane_center, dtype=float).reshape(3)
                if d.chosen_plane_center is not None
                else [None, None, None]
            )
            left_before = (
                np.asarray(d.left_foot_before, dtype=float).reshape(3)
                if d.left_foot_before is not None
                else [None, None, None]
            )
            authored_root = (
                np.asarray(d.authored_root_world, dtype=float).reshape(3)
                if d.authored_root_world is not None
                else [None, None, None]
            )
            grounded_root = (
                np.asarray(d.grounded_root_world, dtype=float).reshape(3)
                if d.grounded_root_world is not None
                else [None, None, None]
            )
            right_before = (
                np.asarray(d.right_foot_before, dtype=float).reshape(3)
                if d.right_foot_before is not None
                else [None, None, None]
            )
            left_after = (
                np.asarray(d.left_foot_after, dtype=float).reshape(3)
                if d.left_foot_after is not None
                else [None, None, None]
            )
            right_after = (
                np.asarray(d.right_foot_after, dtype=float).reshape(3)
                if d.right_foot_after is not None
                else [None, None, None]
            )
            support_before = (
                np.asarray(d.support_point_before, dtype=float).reshape(3)
                if d.support_point_before is not None
                else [None, None, None]
            )
            support_after = (
                np.asarray(d.support_point_after, dtype=float).reshape(3)
                if d.support_point_after is not None
                else [None, None, None]
            )
            previous_support_before = (
                np.asarray(d.previous_support_point_before, dtype=float).reshape(3)
                if d.previous_support_point_before is not None
                else [None, None, None]
            )
            previous_support_after = (
                np.asarray(d.previous_support_point_after, dtype=float).reshape(3)
                if d.previous_support_point_after is not None
                else [None, None, None]
            )
            support_anchor_blended = (
                np.asarray(d.support_anchor_blended, dtype=float).reshape(3)
                if d.support_anchor_blended is not None
                else [None, None, None]
            )
            plant_target_world = (
                np.asarray(d.plant_target_world, dtype=float).reshape(3)
                if d.plant_target_world is not None
                else [None, None, None]
            )
            translation = np.asarray(d.applied_translation_world, dtype=float).reshape(3)
            writer.writerow(
                [
                    int(d.frame_index),
                    str(d.support_mode),
                    "" if d.support_confidence is None else float(d.support_confidence),
                    "|".join(str(v) for v in d.support_source_frame_indices),
                    int(len(d.support_source_frame_indices)),
                    "" if d.support_failure_reason is None else str(d.support_failure_reason),
                    float(d.sole_offset_m),
                    "" if d.chosen_plane_frame_index is None else int(d.chosen_plane_frame_index),
                    normal[0],
                    normal[1],
                    normal[2],
                    "" if d.chosen_plane_offset is None else float(d.chosen_plane_offset),
                    center[0],
                    center[1],
                    center[2],
                    (
                        ""
                        if d.chosen_plane_center_xy_distance_m is None
                        else float(d.chosen_plane_center_xy_distance_m)
                    ),
                    str(d.selected_support_foot),
                    "" if d.selected_plane_source is None else str(d.selected_plane_source),
                    authored_root[0],
                    authored_root[1],
                    authored_root[2],
                    grounded_root[0],
                    grounded_root[1],
                    grounded_root[2],
                    "" if d.root_support_offset_m is None else float(d.root_support_offset_m),
                    "" if d.plane_height_at_xy_m is None else float(d.plane_height_at_xy_m),
                    "" if d.planned_z_delta_m is None else float(d.planned_z_delta_m),
                    "" if d.vertical_velocity_mps is None else float(d.vertical_velocity_mps),
                    "" if d.vertical_accel_mps2 is None else float(d.vertical_accel_mps2),
                    "" if d.traversal_segment_id is None else int(d.traversal_segment_id),
                    "" if d.plane_transition_phase is None else str(d.plane_transition_phase),
                    left_before[0],
                    left_before[1],
                    left_before[2],
                    right_before[0],
                    right_before[1],
                    right_before[2],
                    left_after[0],
                    left_after[1],
                    left_after[2],
                    right_after[0],
                    right_after[1],
                    right_after[2],
                    support_before[0],
                    support_before[1],
                    support_before[2],
                    support_after[0],
                    support_after[1],
                    support_after[2],
                    (
                        ""
                        if d.pre_correction_signed_distance_m is None
                        else float(d.pre_correction_signed_distance_m)
                    ),
                    (
                        ""
                        if d.post_correction_signed_distance_m is None
                        else float(d.post_correction_signed_distance_m)
                    ),
                    (
                        ""
                        if d.left_post_signed_distance_m is None
                        else float(d.left_post_signed_distance_m)
                    ),
                    (
                        ""
                        if d.right_post_signed_distance_m is None
                        else float(d.right_post_signed_distance_m)
                    ),
                    (
                        ""
                        if d.support_jump_from_prev_deg is None
                        else float(d.support_jump_from_prev_deg)
                    ),
                    (
                        ""
                        if d.support_height_jump_from_prev_m is None
                        else float(d.support_height_jump_from_prev_m)
                    ),
                    (
                        ""
                        if d.support_anchor_shift_from_prev_m is None
                        else float(d.support_anchor_shift_from_prev_m)
                    ),
                    str(d.support_state),
                    "" if d.visibility_contract_state is None else str(d.visibility_contract_state),
                    int(bool(d.visibility_culled)),
                    "" if d.visibility_cull_reason is None else str(d.visibility_cull_reason),
                    int(bool(d.frame_requires_support)),
                    previous_support_before[0],
                    previous_support_before[1],
                    previous_support_before[2],
                    previous_support_after[0],
                    previous_support_after[1],
                    previous_support_after[2],
                    (
                        ""
                        if d.relock_current_signed_distance_m is None
                        else float(d.relock_current_signed_distance_m)
                    ),
                    (
                        ""
                        if d.relock_previous_signed_distance_m is None
                        else float(d.relock_previous_signed_distance_m)
                    ),
                    "" if d.support_origin_mode is None else str(d.support_origin_mode),
                    "" if d.relock_decision_reason is None else str(d.relock_decision_reason),
                    (
                        ""
                        if d.nearest_persisted_plane_center_xy_distance_m is None
                        else float(d.nearest_persisted_plane_center_xy_distance_m)
                    ),
                    (
                        ""
                        if d.effective_persisted_plane_locality_limit_m is None
                        else float(d.effective_persisted_plane_locality_limit_m)
                    ),
                    (
                        ""
                        if d.persisted_plane_locality_mode is None
                        else str(d.persisted_plane_locality_mode)
                    ),
                    "" if d.support_anchor_policy is None else str(d.support_anchor_policy),
                    "" if d.left_support_weight is None else float(d.left_support_weight),
                    "" if d.right_support_weight is None else float(d.right_support_weight),
                    support_anchor_blended[0],
                    support_anchor_blended[1],
                    support_anchor_blended[2],
                    "" if d.support_height_raw_m is None else float(d.support_height_raw_m),
                    "" if d.support_height_filtered_m is None else float(d.support_height_filtered_m),
                    "" if d.left_support_confidence is None else float(d.left_support_confidence),
                    "" if d.right_support_confidence is None else float(d.right_support_confidence),
                    "" if d.support_switch_decision is None else str(d.support_switch_decision),
                    "" if d.support_transfer_state is None else str(d.support_transfer_state),
                    int(bool(d.support_height_clamped)),
                    "" if d.contact_phase is None else float(d.contact_phase),
                    "" if d.contact_state_raw is None else str(d.contact_state_raw),
                    "" if d.contact_state_clean is None else str(d.contact_state_clean),
                    "" if d.contact_segment_id is None else int(d.contact_segment_id),
                    "" if d.contact_segment_kind is None else str(d.contact_segment_kind),
                    "" if d.plant_lock_source_frame is None else int(d.plant_lock_source_frame),
                    plant_target_world[0],
                    plant_target_world[1],
                    plant_target_world[2],
                    "" if d.plant_lock_error_m is None else float(d.plant_lock_error_m),
                    "" if d.plant_lock_xy_error_m is None else float(d.plant_lock_xy_error_m),
                    "" if d.support_authority is None else str(d.support_authority),
                    (
                        ""
                        if d.same_plane_continuity is None
                        else int(bool(d.same_plane_continuity))
                    ),
                    "" if d.continuity_break_reason is None else str(d.continuity_break_reason),
                    (
                        ""
                        if d.dynamic_anchor_shift_limit_m is None
                        else float(d.dynamic_anchor_shift_limit_m)
                    ),
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                    "" if d.applied_translation_xy_m is None else float(d.applied_translation_xy_m),
                    "" if d.xy_lock_mode is None else str(d.xy_lock_mode),
                    int(bool(d.xy_lock_clamped)),
                    int(bool(d.plane_selection_rejected_for_locality)),
                    int(bool(d.missing_left_foot)),
                    int(bool(d.missing_right_foot)),
                    int(bool(d.no_plane)),
                ]
            )
    return json_path, csv_path


def _write_support_surface_diagnostics(
    *,
    run_dir: Path,
    diagnostics: Sequence[GroundingDiagnostic],
) -> tuple[Path, Path]:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "support_surface_diagnostics.json"
    csv_path = vis_dir / "support_surface_diagnostics.csv"
    payload = {
        "count": len(diagnostics),
        "entries": [
            {
                "frame_index": int(d.frame_index),
                "support_mode": str(d.support_mode),
                "support_confidence": (
                    None if d.support_confidence is None else float(d.support_confidence)
                ),
                "support_source_frame_indices": [int(v) for v in d.support_source_frame_indices],
                "support_source_frame_count": int(len(d.support_source_frame_indices)),
                "support_failure_reason": d.support_failure_reason,
                "sole_offset_m": float(d.sole_offset_m),
                "selected_support_foot": str(d.selected_support_foot),
                "support_point_before": (
                    None
                    if d.support_point_before is None
                    else np.asarray(d.support_point_before, dtype=float).tolist()
                ),
                "support_point_after": (
                    None
                    if d.support_point_after is None
                    else np.asarray(d.support_point_after, dtype=float).tolist()
                ),
                "support_surface_normal": (
                    None
                    if d.chosen_plane_normal is None
                    else np.asarray(d.chosen_plane_normal, dtype=float).tolist()
                ),
                "support_surface_offset": (
                    None if d.chosen_plane_offset is None else float(d.chosen_plane_offset)
                ),
                "support_jump_from_prev_deg": (
                    None
                    if d.support_jump_from_prev_deg is None
                    else float(d.support_jump_from_prev_deg)
                ),
                "support_height_jump_from_prev_m": (
                    None
                    if d.support_height_jump_from_prev_m is None
                    else float(d.support_height_jump_from_prev_m)
                ),
                "support_anchor_shift_from_prev_m": (
                    None
                    if d.support_anchor_shift_from_prev_m is None
                    else float(d.support_anchor_shift_from_prev_m)
                ),
                "support_state": str(d.support_state),
                "visibility_contract_state": d.visibility_contract_state,
                "visibility_culled": bool(d.visibility_culled),
                "visibility_cull_reason": d.visibility_cull_reason,
                "frame_requires_support": bool(d.frame_requires_support),
                "previous_support_point_before": (
                    None
                    if d.previous_support_point_before is None
                    else np.asarray(d.previous_support_point_before, dtype=float).tolist()
                ),
                "previous_support_point_after": (
                    None
                    if d.previous_support_point_after is None
                    else np.asarray(d.previous_support_point_after, dtype=float).tolist()
                ),
                "relock_current_signed_distance_m": (
                    None
                    if d.relock_current_signed_distance_m is None
                    else float(d.relock_current_signed_distance_m)
                ),
                "relock_previous_signed_distance_m": (
                    None
                    if d.relock_previous_signed_distance_m is None
                    else float(d.relock_previous_signed_distance_m)
                ),
                "support_origin_mode": d.support_origin_mode,
                "relock_decision_reason": d.relock_decision_reason,
                "nearest_persisted_plane_center_xy_distance_m": (
                    None
                    if d.nearest_persisted_plane_center_xy_distance_m is None
                    else float(d.nearest_persisted_plane_center_xy_distance_m)
                ),
                "effective_persisted_plane_locality_limit_m": (
                    None
                    if d.effective_persisted_plane_locality_limit_m is None
                    else float(d.effective_persisted_plane_locality_limit_m)
                ),
                "persisted_plane_locality_mode": d.persisted_plane_locality_mode,
                "support_anchor_policy": d.support_anchor_policy,
                "left_support_weight": (
                    None if d.left_support_weight is None else float(d.left_support_weight)
                ),
                "right_support_weight": (
                    None if d.right_support_weight is None else float(d.right_support_weight)
                ),
                "support_anchor_blended": (
                    None
                    if d.support_anchor_blended is None
                    else np.asarray(d.support_anchor_blended, dtype=float).tolist()
                ),
                "support_height_raw_m": (
                    None if d.support_height_raw_m is None else float(d.support_height_raw_m)
                ),
                "support_height_filtered_m": (
                    None
                    if d.support_height_filtered_m is None
                    else float(d.support_height_filtered_m)
                ),
                "left_support_confidence": (
                    None
                    if d.left_support_confidence is None
                    else float(d.left_support_confidence)
                ),
                "right_support_confidence": (
                    None
                    if d.right_support_confidence is None
                    else float(d.right_support_confidence)
                ),
                "support_switch_decision": d.support_switch_decision,
                "support_transfer_state": d.support_transfer_state,
                "support_height_clamped": bool(d.support_height_clamped),
                "contact_phase": (
                    None if d.contact_phase is None else float(d.contact_phase)
                ),
                "contact_state_raw": d.contact_state_raw,
                "contact_state_clean": d.contact_state_clean,
                "contact_segment_id": (
                    None if d.contact_segment_id is None else int(d.contact_segment_id)
                ),
                "contact_segment_kind": d.contact_segment_kind,
                "plant_lock_source_frame": (
                    None
                    if d.plant_lock_source_frame is None
                    else int(d.plant_lock_source_frame)
                ),
                "plant_target_world": (
                    None
                    if d.plant_target_world is None
                    else np.asarray(d.plant_target_world, dtype=float).tolist()
                ),
                "plant_lock_error_m": (
                    None if d.plant_lock_error_m is None else float(d.plant_lock_error_m)
                ),
                "plant_lock_xy_error_m": (
                    None if d.plant_lock_xy_error_m is None else float(d.plant_lock_xy_error_m)
                ),
                "support_authority": d.support_authority,
                "same_plane_continuity": (
                    None
                    if d.same_plane_continuity is None
                    else bool(d.same_plane_continuity)
                ),
                "continuity_break_reason": d.continuity_break_reason,
                "dynamic_anchor_shift_limit_m": (
                    None
                    if d.dynamic_anchor_shift_limit_m is None
                    else float(d.dynamic_anchor_shift_limit_m)
                ),
                "pre_correction_signed_distance_m": (
                    None
                    if d.pre_correction_signed_distance_m is None
                    else float(d.pre_correction_signed_distance_m)
                ),
                "post_correction_signed_distance_m": (
                    None
                    if d.post_correction_signed_distance_m is None
                    else float(d.post_correction_signed_distance_m)
                ),
                "applied_translation_xy_m": (
                    None
                    if d.applied_translation_xy_m is None
                    else float(d.applied_translation_xy_m)
                ),
                "xy_lock_mode": d.xy_lock_mode,
                "xy_lock_clamped": bool(d.xy_lock_clamped),
            }
            for d in diagnostics
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "support_mode",
                "support_confidence",
                "support_source_frame_indices",
                "support_source_frame_count",
                "support_failure_reason",
                "sole_offset_m",
                "selected_support_foot",
                "support_before_x",
                "support_before_y",
                "support_before_z",
                "support_after_x",
                "support_after_y",
                "support_after_z",
                "normal_x",
                "normal_y",
                "normal_z",
                "offset",
                "support_jump_from_prev_deg",
                "support_height_jump_from_prev_m",
                "support_anchor_shift_from_prev_m",
                "support_state",
                "visibility_contract_state",
                "visibility_culled",
                "visibility_cull_reason",
                "frame_requires_support",
                "prev_support_before_x",
                "prev_support_before_y",
                "prev_support_before_z",
                "prev_support_after_x",
                "prev_support_after_y",
                "prev_support_after_z",
                "relock_current_signed_distance_m",
                "relock_previous_signed_distance_m",
                "support_origin_mode",
                "relock_decision_reason",
                "nearest_persisted_plane_center_xy_distance_m",
                "effective_persisted_plane_locality_limit_m",
                "persisted_plane_locality_mode",
                "support_anchor_policy",
                "left_support_weight",
                "right_support_weight",
                "support_anchor_blended_x",
                "support_anchor_blended_y",
                "support_anchor_blended_z",
                "support_height_raw_m",
                "support_height_filtered_m",
                "left_support_confidence",
                "right_support_confidence",
                "support_switch_decision",
                "support_transfer_state",
                "support_height_clamped",
                "dynamic_anchor_shift_limit_m",
                "pre_correction_signed_distance_m",
                "post_correction_signed_distance_m",
                "plant_lock_xy_error_m",
                "applied_translation_xy_m",
                "xy_lock_mode",
                "xy_lock_clamped",
            ]
        )
        for d in diagnostics:
            support_before = (
                [None, None, None]
                if d.support_point_before is None
                else np.asarray(d.support_point_before, dtype=float).reshape(3)
            )
            support_after = (
                [None, None, None]
                if d.support_point_after is None
                else np.asarray(d.support_point_after, dtype=float).reshape(3)
            )
            normal = (
                [None, None, None]
                if d.chosen_plane_normal is None
                else np.asarray(d.chosen_plane_normal, dtype=float).reshape(3)
            )
            previous_support_before = (
                [None, None, None]
                if d.previous_support_point_before is None
                else np.asarray(d.previous_support_point_before, dtype=float).reshape(3)
            )
            previous_support_after = (
                [None, None, None]
                if d.previous_support_point_after is None
                else np.asarray(d.previous_support_point_after, dtype=float).reshape(3)
            )
            support_anchor_blended = (
                [None, None, None]
                if d.support_anchor_blended is None
                else np.asarray(d.support_anchor_blended, dtype=float).reshape(3)
            )
            writer.writerow(
                [
                    int(d.frame_index),
                    str(d.support_mode),
                    "" if d.support_confidence is None else float(d.support_confidence),
                    "|".join(str(v) for v in d.support_source_frame_indices),
                    int(len(d.support_source_frame_indices)),
                    "" if d.support_failure_reason is None else str(d.support_failure_reason),
                    float(d.sole_offset_m),
                    str(d.selected_support_foot),
                    support_before[0],
                    support_before[1],
                    support_before[2],
                    support_after[0],
                    support_after[1],
                    support_after[2],
                    normal[0],
                    normal[1],
                    normal[2],
                    "" if d.chosen_plane_offset is None else float(d.chosen_plane_offset),
                    "" if d.support_jump_from_prev_deg is None else float(d.support_jump_from_prev_deg),
                    "" if d.support_height_jump_from_prev_m is None else float(d.support_height_jump_from_prev_m),
                    "" if d.support_anchor_shift_from_prev_m is None else float(d.support_anchor_shift_from_prev_m),
                    str(d.support_state),
                    "" if d.visibility_contract_state is None else str(d.visibility_contract_state),
                    int(bool(d.visibility_culled)),
                    "" if d.visibility_cull_reason is None else str(d.visibility_cull_reason),
                    int(bool(d.frame_requires_support)),
                    previous_support_before[0],
                    previous_support_before[1],
                    previous_support_before[2],
                    previous_support_after[0],
                    previous_support_after[1],
                    previous_support_after[2],
                    "" if d.relock_current_signed_distance_m is None else float(d.relock_current_signed_distance_m),
                    "" if d.relock_previous_signed_distance_m is None else float(d.relock_previous_signed_distance_m),
                    "" if d.support_origin_mode is None else str(d.support_origin_mode),
                    "" if d.relock_decision_reason is None else str(d.relock_decision_reason),
                    (
                        ""
                        if d.nearest_persisted_plane_center_xy_distance_m is None
                        else float(d.nearest_persisted_plane_center_xy_distance_m)
                    ),
                    (
                        ""
                        if d.effective_persisted_plane_locality_limit_m is None
                        else float(d.effective_persisted_plane_locality_limit_m)
                    ),
                    (
                        ""
                        if d.persisted_plane_locality_mode is None
                        else str(d.persisted_plane_locality_mode)
                    ),
                    "" if d.support_anchor_policy is None else str(d.support_anchor_policy),
                    "" if d.left_support_weight is None else float(d.left_support_weight),
                    "" if d.right_support_weight is None else float(d.right_support_weight),
                    support_anchor_blended[0],
                    support_anchor_blended[1],
                    support_anchor_blended[2],
                    "" if d.support_height_raw_m is None else float(d.support_height_raw_m),
                    "" if d.support_height_filtered_m is None else float(d.support_height_filtered_m),
                    "" if d.left_support_confidence is None else float(d.left_support_confidence),
                    "" if d.right_support_confidence is None else float(d.right_support_confidence),
                    "" if d.support_switch_decision is None else str(d.support_switch_decision),
                    "" if d.support_transfer_state is None else str(d.support_transfer_state),
                    int(bool(d.support_height_clamped)),
                    "" if d.dynamic_anchor_shift_limit_m is None else float(d.dynamic_anchor_shift_limit_m),
                    "" if d.pre_correction_signed_distance_m is None else float(d.pre_correction_signed_distance_m),
                    "" if d.post_correction_signed_distance_m is None else float(d.post_correction_signed_distance_m),
                    "" if d.plant_lock_xy_error_m is None else float(d.plant_lock_xy_error_m),
                    "" if d.applied_translation_xy_m is None else float(d.applied_translation_xy_m),
                    "" if d.xy_lock_mode is None else str(d.xy_lock_mode),
                    int(bool(d.xy_lock_clamped)),
                ]
            )
    return json_path, csv_path


def _write_trajectory_support_segments(
    *,
    run_dir: Path,
    diagnostics: Sequence[GroundingDiagnostic],
) -> Path:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "trajectory_support_segments.json"
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for diag in diagnostics:
        if diag.traversal_segment_id is None or diag.chosen_plane_frame_index is None:
            continue
        segment_id = int(diag.traversal_segment_id)
        if current is None or int(current["segment_id"]) != segment_id:
            if current is not None:
                segments.append(current)
            current = {
                "segment_id": segment_id,
                "start_frame": int(diag.frame_index),
                "end_frame": int(diag.frame_index),
                "plane_frame_index": int(diag.chosen_plane_frame_index),
                "plane_source": diag.selected_plane_source,
                "support_mode": diag.support_mode,
            }
        else:
            current["end_frame"] = int(diag.frame_index)
    if current is not None:
        segments.append(current)
    json_path.write_text(
        json.dumps({"count": len(segments), "segments": segments}, indent=2),
        encoding="utf-8",
    )
    return json_path


def _write_trajectory_height_profile(
    *,
    run_dir: Path,
    diagnostics: Sequence[GroundingDiagnostic],
) -> Path:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = vis_dir / "trajectory_height_profile.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "plane_frame_index",
                "traversal_segment_id",
                "authored_root_z",
                "grounded_root_z",
                "plane_height_at_xy_m",
                "planned_z_delta_m",
                "vertical_velocity_mps",
                "vertical_accel_mps2",
                "plane_transition_phase",
            ]
        )
        for diag in diagnostics:
            authored_root = (
                None
                if diag.authored_root_world is None
                else np.asarray(diag.authored_root_world, dtype=float).reshape(3)
            )
            grounded_root = (
                None
                if diag.grounded_root_world is None
                else np.asarray(diag.grounded_root_world, dtype=float).reshape(3)
            )
            writer.writerow(
                [
                    int(diag.frame_index),
                    "" if diag.chosen_plane_frame_index is None else int(diag.chosen_plane_frame_index),
                    "" if diag.traversal_segment_id is None else int(diag.traversal_segment_id),
                    "" if authored_root is None else float(authored_root[2]),
                    "" if grounded_root is None else float(grounded_root[2]),
                    "" if diag.plane_height_at_xy_m is None else float(diag.plane_height_at_xy_m),
                    "" if diag.planned_z_delta_m is None else float(diag.planned_z_delta_m),
                    "" if diag.vertical_velocity_mps is None else float(diag.vertical_velocity_mps),
                    "" if diag.vertical_accel_mps2 is None else float(diag.vertical_accel_mps2),
                    "" if diag.plane_transition_phase is None else str(diag.plane_transition_phase),
                ]
            )
    return csv_path


def _write_dynamic_lighting_anchor_diagnostics(
    *,
    run_dir: Path,
    diagnostics: Sequence[Mapping[str, Any]],
) -> Path:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "lighting_anchor_diagnostics.json"
    payload = {
        "count": int(len(diagnostics)),
        "entries": [dict(item) for item in diagnostics],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def _raise_for_grounding_failures(
    *,
    diagnostics: list[GroundingDiagnostic],
    max_residual_m: float,
    max_plane_center_xy_distance_m: float,
) -> None:
    unsupported_visible = [
        int(d.frame_index)
        for d in diagnostics
        if d.no_plane and not d.visibility_culled
    ]
    locality_rejected = [
        int(d.frame_index)
        for d in diagnostics
        if d.plane_selection_rejected_for_locality and not d.visibility_culled
    ]
    invalid_grounded_root = [
        int(d.frame_index)
        for d in diagnostics
        if d.grounded_root_world is not None
        and not np.all(np.isfinite(np.asarray(d.grounded_root_world, dtype=np.float32)))
    ]
    if (
        not unsupported_visible
        and not locality_rejected
        and not invalid_grounded_root
    ):
        return
    parts: list[str] = []
    if unsupported_visible:
        parts.append(f"missing_support_surface={_format_frame_ranges(unsupported_visible)}")
    if locality_rejected:
        parts.append(
            "locality>"
            f"{max_plane_center_xy_distance_m:.3f}m={_format_frame_ranges(locality_rejected)}"
        )
    if invalid_grounded_root:
        parts.append(f"invalid_ground_root={_format_frame_ranges(invalid_grounded_root)}")
    raise ValueError("Pedestrian support-surface grounding failed: " + "; ".join(parts))


def apply_road_support_to_inserted_pedestrian(
    *,
    spec: SceneSpec,
    road_surface: RoadSurfacePipelineResult,
    frame_indices: np.ndarray,
    actor_name: str = "Pedestrian01",
) -> list[GroundingDiagnostic]:
    root = bpy.data.objects.get(actor_name)
    if root is None:
        raise ValueError(
            f"Actor root '{actor_name}' not found; cannot ground inserted pedestrian."
        )
    armature_obj = _find_actor_armature(actor_name)
    frames = [int(frame) for frame in frame_indices.tolist()]
    if not frames:
        log_warning("Skipping pedestrian grounding: no frame indices available.")
        return []

    intrinsics_k, frame_to_c2w, frame_data_cache = _load_support_fit_context(
        spec,
        frame_indices,
    )
    trajectory_c2w = np.stack(
        [np.asarray(frame_to_c2w[int(frame_idx)], dtype=np.float32) for frame_idx in frames],
        axis=0,
    )

    baseline_samples: list[dict[str, Any]] = []
    for frame_idx in frames:
        bpy.context.scene.frame_set(int(frame_idx))
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        root_eval = root.evaluated_get(deps)
        authored_root_world = np.asarray(
            root_eval.matrix_world.translation.copy(),
            dtype=np.float32,
        )
        image_shape = tuple(
            int(v) for v in np.asarray(frame_data_cache[int(frame_idx)][0]).shape[:2]
        )
        feet_before = _evaluate_feet_world(root, armature_obj)
        left_before = feet_before["left"]
        right_before = feet_before["right"]
        projected_visible, visibility_contract_state = _classify_projected_actor_visibility(
            frame_idx=int(frame_idx),
            intrinsics_k=intrinsics_k,
            frame_to_c2w=frame_to_c2w,
            actor_root=root,
            depsgraph=deps,
            image_shape=image_shape,
        )
        baseline_samples.append(
            {
                "frame_index": int(frame_idx),
                "authored_root_world": authored_root_world,
                "left_foot_before": (
                    None if left_before is None else np.asarray(left_before, dtype=np.float32)
                ),
                "right_foot_before": (
                    None if right_before is None else np.asarray(right_before, dtype=np.float32)
                ),
                "projected_visible": bool(projected_visible),
                "visibility_contract_state": visibility_contract_state,
            }
        )

    actor_support_contract = _build_actor_support_contract(
        root_positions_world=[sample["authored_root_world"] for sample in baseline_samples],
        left_feet_world=[sample["left_foot_before"] for sample in baseline_samples],
        right_feet_world=[sample["right_foot_before"] for sample in baseline_samples],
    )

    chosen_planes: list[RoadPlaneSpec | None] = []
    chosen_localities: list[PersistedPlaneLocalityDecision] = []
    plane_sources: list[str] = []
    support_queries: list[np.ndarray | None] = []
    raw_root_heights: list[float | None] = []
    plane_heights: list[float | None] = []
    previous_plane: RoadPlaneSpec | None = None

    for sample in baseline_samples:
        authored_root_world = np.asarray(sample["authored_root_world"], dtype=np.float32)
        support_query = np.asarray(
            [
                float(authored_root_world[0]),
                float(authored_root_world[1]),
                float(authored_root_world[2] - actor_support_contract.root_to_support_m),
            ],
            dtype=np.float32,
        )
        chosen_plane, locality, support_source = _resolve_trajectory_support_plane(
            spec=spec,
            frame_idx=int(sample["frame_index"]),
            support_query_world=support_query,
            road_surface=road_surface,
            trajectory_c2w=trajectory_c2w,
            previous_plane=previous_plane,
        )
        if chosen_plane is None:
            chosen_planes.append(None)
            chosen_localities.append(locality)
            plane_sources.append(str(support_source))
            support_queries.append(support_query)
            raw_root_heights.append(None)
            plane_heights.append(None)
            continue
        plane_height = _solve_plane_height_at_xy(
            normal=np.asarray(chosen_plane.normal, dtype=np.float32),
            offset=float(chosen_plane.offset),
            xy_world=np.asarray(authored_root_world[:2], dtype=np.float32),
        )
        chosen_planes.append(chosen_plane)
        chosen_localities.append(locality)
        plane_sources.append(str(support_source))
        support_queries.append(support_query)
        raw_root_heights.append(
            float(plane_height + actor_support_contract.root_to_support_m)
        )
        plane_heights.append(float(plane_height))
        previous_plane = chosen_plane

    traversal_segment_ids: list[int | None] = []
    current_segment_id = -1
    previous_plane_frame_index: int | None = None
    for chosen_plane in chosen_planes:
        if chosen_plane is None:
            traversal_segment_ids.append(None)
            previous_plane_frame_index = None
            continue
        plane_frame_index = int(chosen_plane.frame_index)
        if previous_plane_frame_index is None or plane_frame_index != previous_plane_frame_index:
            current_segment_id += 1
        traversal_segment_ids.append(int(current_segment_id))
        previous_plane_frame_index = plane_frame_index

    smoothed_root_heights, vertical_velocities, vertical_accels, transition_phases = (
        _smooth_grounded_root_heights(
            raw_root_heights_m=raw_root_heights,
            segment_ids=traversal_segment_ids,
            spec=spec,
        )
    )

    diagnostics: list[GroundingDiagnostic] = []
    previous_support_before: np.ndarray | None = None
    previous_support_after: np.ndarray | None = None
    previous_plane_normal: np.ndarray | None = None
    previous_plane_height: float | None = None
    previous_support_xy: np.ndarray | None = None

    for idx, sample in enumerate(baseline_samples):
        frame_idx = int(sample["frame_index"])
        authored_root_world = np.asarray(sample["authored_root_world"], dtype=np.float32)
        left_before = sample["left_foot_before"]
        right_before = sample["right_foot_before"]
        support_query = support_queries[idx]
        chosen_plane = chosen_planes[idx]
        locality = chosen_localities[idx]
        support_source = plane_sources[idx]
        grounded_root_height = smoothed_root_heights[idx]
        plane_height = plane_heights[idx]
        projected_visible = bool(sample["projected_visible"])
        visibility_contract_state = sample["visibility_contract_state"]

        grounded_root_world = np.array(authored_root_world, dtype=np.float32, copy=True)
        if grounded_root_height is not None:
            grounded_root_world[2] = float(grounded_root_height)
        root.location = tuple(float(v) for v in grounded_root_world.tolist())
        root.keyframe_insert(data_path="location", frame=int(frame_idx))
        bpy.context.view_layer.update()
        feet_after = _evaluate_feet_world(root, armature_obj)
        left_after = feet_after["left"]
        right_after = feet_after["right"]

        plane_support_point = (
            None
            if chosen_plane is None or plane_height is None
            else np.asarray(
                [
                    float(authored_root_world[0]),
                    float(authored_root_world[1]),
                    float(plane_height),
                ],
                dtype=np.float32,
            )
        )
        support_jump_deg = None
        support_height_jump = None
        support_anchor_shift = None
        if chosen_plane is not None and previous_plane_normal is not None:
            current_normal = np.asarray(chosen_plane.normal, dtype=np.float32)
            cos_sim = float(
                np.clip(np.dot(current_normal, previous_plane_normal), -1.0, 1.0)
            )
            support_jump_deg = float(np.degrees(np.arccos(cos_sim)))
        if plane_height is not None and previous_plane_height is not None:
            support_height_jump = float(abs(float(plane_height) - float(previous_plane_height)))
        if plane_support_point is not None and previous_support_xy is not None:
            support_anchor_shift = float(
                np.linalg.norm(
                    np.asarray(plane_support_point[:2], dtype=np.float32)
                    - np.asarray(previous_support_xy, dtype=np.float32)
                )
            )

        if chosen_plane is None or support_query is None or plane_height is None:
            diagnostics.append(
                GroundingDiagnostic(
                    frame_index=frame_idx,
                    support_mode="none",
                    support_confidence=None,
                    support_source_frame_indices=tuple(),
                    support_failure_reason=(
                        "persisted_fallback_locality_rejected"
                        if locality.locality_mode == "rejected"
                        else "no_support_surface"
                    ),
                    sole_offset_m=float(_SOLE_OFFSET_M),
                    chosen_plane_frame_index=None,
                    chosen_plane_normal=None,
                    chosen_plane_offset=None,
                    chosen_plane_center=None,
                    chosen_plane_center_xy_distance_m=(
                        None
                        if not np.isfinite(locality.nearest_xy_distance_m)
                        else float(locality.nearest_xy_distance_m)
                    ),
                    selected_support_foot="path",
                    selected_plane_source=str(support_source),
                    left_foot_before=left_before,
                    right_foot_before=right_before,
                    left_foot_after=None if left_after is None else np.asarray(left_after, dtype=np.float32),
                    right_foot_after=None if right_after is None else np.asarray(right_after, dtype=np.float32),
                    support_point_before=None if support_query is None else np.asarray(support_query, dtype=np.float32),
                    support_point_after=None,
                    pre_correction_signed_distance_m=None,
                    post_correction_signed_distance_m=None,
                    left_post_signed_distance_m=None,
                    right_post_signed_distance_m=None,
                    support_jump_from_prev_deg=None,
                    support_height_jump_from_prev_m=None,
                    support_anchor_shift_from_prev_m=None,
                    dynamic_anchor_shift_limit_m=None,
                    applied_translation_world=np.asarray(
                        grounded_root_world - authored_root_world,
                        dtype=np.float32,
                    ),
                    authored_root_world=np.asarray(authored_root_world, dtype=np.float32),
                    grounded_root_world=np.asarray(grounded_root_world, dtype=np.float32),
                    root_support_offset_m=float(actor_support_contract.root_to_support_m),
                    plane_height_at_xy_m=None,
                    planned_z_delta_m=float(grounded_root_world[2] - authored_root_world[2]),
                    vertical_velocity_mps=vertical_velocities[idx],
                    vertical_accel_mps2=vertical_accels[idx],
                    traversal_segment_id=None,
                    plane_transition_phase="unsupported",
                    plane_selection_rejected_for_locality=(
                        locality.locality_mode == "rejected"
                    ),
                    missing_left_foot=left_before is None,
                    missing_right_foot=right_before is None,
                    no_plane=True,
                    visibility_culled=not projected_visible,
                    visibility_cull_reason=(
                        None if projected_visible else str(visibility_contract_state)
                    ),
                    frame_requires_support=True,
                    previous_support_point_before=previous_support_before,
                    previous_support_point_after=previous_support_after,
                    nearest_persisted_plane_center_xy_distance_m=(
                        None
                        if not np.isfinite(locality.nearest_xy_distance_m)
                        else float(locality.nearest_xy_distance_m)
                    ),
                    effective_persisted_plane_locality_limit_m=float(locality.effective_limit_m),
                    persisted_plane_locality_mode=str(locality.locality_mode),
                    applied_translation_xy_m=0.0,
                    xy_lock_mode="trajectory_root_z_only",
                    support_state="unsupported",
                    visibility_contract_state=str(visibility_contract_state),
                )
            )
            continue

        plane_normal = np.asarray(chosen_plane.normal, dtype=np.float32)
        plane_offset = float(chosen_plane.offset)
        pre_signed = float(np.dot(plane_normal, support_query) + plane_offset)
        diagnostics.append(
            GroundingDiagnostic(
                frame_index=frame_idx,
                support_mode="trajectory_path",
                support_confidence=float(chosen_plane.confidence),
                support_source_frame_indices=(int(chosen_plane.frame_index),),
                support_failure_reason=None,
                sole_offset_m=float(_SOLE_OFFSET_M),
                chosen_plane_frame_index=int(chosen_plane.frame_index),
                chosen_plane_normal=np.asarray(plane_normal, dtype=np.float32),
                chosen_plane_offset=plane_offset,
                chosen_plane_center=np.asarray(chosen_plane.center, dtype=np.float32),
                chosen_plane_center_xy_distance_m=(
                    None
                    if not np.isfinite(locality.nearest_xy_distance_m)
                    else float(locality.nearest_xy_distance_m)
                ),
                selected_support_foot="path",
                selected_plane_source=str(support_source),
                left_foot_before=left_before,
                right_foot_before=right_before,
                left_foot_after=None if left_after is None else np.asarray(left_after, dtype=np.float32),
                right_foot_after=None if right_after is None else np.asarray(right_after, dtype=np.float32),
                support_point_before=np.asarray(support_query, dtype=np.float32),
                support_point_after=np.asarray(plane_support_point, dtype=np.float32),
                pre_correction_signed_distance_m=float(pre_signed),
                post_correction_signed_distance_m=0.0,
                left_post_signed_distance_m=None,
                right_post_signed_distance_m=None,
                support_jump_from_prev_deg=support_jump_deg,
                support_height_jump_from_prev_m=support_height_jump,
                support_anchor_shift_from_prev_m=support_anchor_shift,
                dynamic_anchor_shift_limit_m=None,
                applied_translation_world=np.asarray(
                    grounded_root_world - authored_root_world,
                    dtype=np.float32,
                ),
                authored_root_world=np.asarray(authored_root_world, dtype=np.float32),
                grounded_root_world=np.asarray(grounded_root_world, dtype=np.float32),
                root_support_offset_m=float(actor_support_contract.root_to_support_m),
                plane_height_at_xy_m=float(plane_height),
                planned_z_delta_m=float(grounded_root_world[2] - authored_root_world[2]),
                vertical_velocity_mps=vertical_velocities[idx],
                vertical_accel_mps2=vertical_accels[idx],
                traversal_segment_id=traversal_segment_ids[idx],
                plane_transition_phase=transition_phases[idx],
                plane_selection_rejected_for_locality=False,
                missing_left_foot=left_before is None,
                missing_right_foot=right_before is None,
                no_plane=False,
                visibility_culled=not projected_visible,
                visibility_cull_reason=(
                    None if projected_visible else str(visibility_contract_state)
                ),
                frame_requires_support=True,
                previous_support_point_before=previous_support_before,
                previous_support_point_after=previous_support_after,
                nearest_persisted_plane_center_xy_distance_m=(
                    None
                    if not np.isfinite(locality.nearest_xy_distance_m)
                    else float(locality.nearest_xy_distance_m)
                ),
                effective_persisted_plane_locality_limit_m=float(locality.effective_limit_m),
                persisted_plane_locality_mode=str(locality.locality_mode),
                applied_translation_xy_m=0.0,
                xy_lock_mode="trajectory_root_z_only",
                support_state="supported",
                visibility_contract_state=str(visibility_contract_state),
            )
        )
        previous_support_before = np.asarray(support_query, dtype=np.float32)
        previous_support_after = np.asarray(plane_support_point, dtype=np.float32)
        previous_plane_normal = np.asarray(plane_normal, dtype=np.float32)
        previous_plane_height = float(plane_height)
        previous_support_xy = np.asarray(plane_support_point[:2], dtype=np.float32)

    effective_motion_policy = str(root.get("pemoin_effective_motion_policy", "")).strip().lower()
    if effective_motion_policy == "stationary_at_spawn" and diagnostics:
        authored_xy = np.stack(
            [
                np.asarray(d.authored_root_world, dtype=np.float32)[:2]
                for d in diagnostics
                if d.authored_root_world is not None
            ],
            axis=0,
        )
        if authored_xy.size:
            deltas = authored_xy - authored_xy[:1]
            max_xy_drift_m = float(np.max(np.linalg.norm(deltas, axis=1)))
            root["pemoin_stationary_authored_root_max_xy_drift_m"] = max_xy_drift_m
            if max_xy_drift_m > 1e-4:
                raise RuntimeError(
                    "Stationary Mixamo motion policy violated authored-root XY lock "
                    f"during grounding: max_xy_drift_m={max_xy_drift_m:.6f}."
                )

    log_info(f"Pedestrian grounding sampled {len(diagnostics)} frames")
    return diagnostics


def setup_character_only_render(
    actor_root: bpy.types.Object,
    resolution_x: int = 1920,
    resolution_y: int = 1080,
    *,
    resolution_scale: float = 1.0,
    extra_visible_objects: Sequence[bpy.types.Object] = (),
):
    """Configure scene to render only the grounded actor hierarchy."""
    scene = bpy.context.scene

    actor_objects = get_object_hierarchy(actor_root)
    actor_objects.update(obj for obj in extra_visible_objects if obj is not None)

    # Hide all other objects from render
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.hide_render = False
            continue
        obj.hide_render = obj not in actor_objects

    # Keep camera visible
    if scene.camera:
        scene.camera.hide_render = False

    # Resolution
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = max(
        10,
        min(100, int(round(float(resolution_scale) * 100.0))),
    )


def _shadow_receiver_name(actor_name: str) -> str:
    return f"{actor_name}_ShadowReceiver"


def _ensure_shadow_receiver_object(
    actor_name: str,
) -> bpy.types.Object:
    name = _shadow_receiver_name(actor_name)
    existing = bpy.data.objects.get(name)
    if existing is not None:
        return existing
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    mesh.from_pydata(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _ensure_shadow_receiver_material() -> bpy.types.Material:
    name = "PEMOINShadowReceiverMaterial"
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _clear_animation_data(obj: bpy.types.Object) -> None:
    if obj.animation_data is not None:
        obj.animation_data_clear()


def _configure_shadow_receiver_animation(
    *,
    spec: SceneSpec,
    actor_root: bpy.types.Object,
    grounding_diagnostics: Sequence[GroundingDiagnostic],
) -> bpy.types.Object | None:
    shadow_spec = getattr(spec, "shadow", None)
    if shadow_spec is not None and not bool(getattr(shadow_spec, "enabled", True)):
        return None
    receiver = _ensure_shadow_receiver_object(actor_root.name)
    receiver.rotation_mode = "QUATERNION"
    receiver.is_shadow_catcher = True
    receiver.hide_viewport = False
    receiver.visible_shadow = False
    receiver.visible_camera = True
    receiver_material = _ensure_shadow_receiver_material()
    if receiver.data.materials:
        receiver.data.materials[0] = receiver_material
    else:
        receiver.data.materials.append(receiver_material)
    _clear_animation_data(receiver)
    patch_size_m = float(getattr(shadow_spec, "receiver_patch_size_m", 4.0))
    half_scale = max(patch_size_m / 2.0, 1e-4)
    prev_location = np.zeros(3, dtype=np.float32)
    prev_quaternion = (1.0, 0.0, 0.0, 0.0)
    for diag in grounding_diagnostics:
        frame_idx = int(diag.frame_index)
        has_support = diag.support_point_after is not None and diag.chosen_plane_normal is not None
        receiver.hide_render = not has_support
        if has_support:
            normal = np.asarray(diag.chosen_plane_normal, dtype=np.float32).reshape(3)
            anchor = np.asarray(diag.support_point_after, dtype=np.float32).reshape(3)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-6:
                raise ValueError(
                    f"Invalid support normal for shadow catcher at frame {frame_idx}."
                )
            normal = normal / norm
            anchor = anchor - normal * 1e-3
            quat = compute_rotation_to_normal(normal)
            receiver.location = tuple(float(v) for v in anchor.tolist())
            receiver.rotation_quaternion = quat
            receiver.scale = (half_scale, half_scale, 1.0)
            prev_location = anchor
            prev_quaternion = quat
        else:
            receiver.location = tuple(float(v) for v in prev_location.tolist())
            receiver.rotation_quaternion = prev_quaternion
            receiver.scale = (half_scale, half_scale, 1.0)
        receiver.keyframe_insert(data_path="location", frame=frame_idx)
        receiver.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
        receiver.keyframe_insert(data_path="scale", frame=frame_idx)
        receiver.keyframe_insert(data_path="hide_render", frame=frame_idx)
    return receiver


def get_object_hierarchy(obj: bpy.types.Object) -> set[bpy.types.Object]:
    """Get an object and all its children recursively."""
    objects = {obj}
    for child in obj.children:
        objects.update(get_object_hierarchy(child))
    return objects


def _resolve_actor_root(actor_name: str) -> bpy.types.Object:
    root = bpy.data.objects.get(actor_name)
    if root is None:
        raise ValueError(f"Actor root '{actor_name}' not found.")
    return root


def _find_actor_armature(actor_name: str) -> bpy.types.Object:
    """Resolve the single armature under the grounded actor root."""
    root = _resolve_actor_root(actor_name)
    armatures = [obj for obj in _iter_descendants(root) if obj.type == "ARMATURE"]
    if not armatures:
        raise ValueError(
            f"Actor root '{actor_name}' has no armature descendants; cannot render."
        )
    if len(armatures) > 1:
        names = ", ".join(sorted(obj.name for obj in armatures))
        raise ValueError(
            f"Actor root '{actor_name}' has multiple armature descendants: {names}"
        )
    return armatures[0]


def _write_render_parity_diagnostics(
    *,
    spec: SceneSpec,
    actor_root: bpy.types.Object,
    armature_obj: bpy.types.Object,
    target_intrinsics: np.ndarray,
    parity_solution: BlenderCameraSolution,
) -> Path:
    scene = bpy.context.scene
    camera_obj = scene.camera
    camera_data = camera_obj.data if camera_obj is not None else None
    visible_descendants = sorted(
        obj.name for obj in get_object_hierarchy(actor_root) if obj is not actor_root
    )
    render_fps = float(scene.render.fps) / max(float(scene.render.fps_base), 1e-6)
    timing_parity = None
    if "pemoin_mixamo_scene_fps" in actor_root:
        cycle_duration_seconds = float(
            actor_root.get("pemoin_mixamo_cycle_duration_seconds", 0.0)
        )
        source_cycle_frames = float(actor_root["pemoin_mixamo_source_cycle_len_frames"])
        source_start_frame = float(actor_root.get("pemoin_mixamo_source_start_frame", 0.0))
        preview_count = min(
            6,
            max(
                1,
                int(
                    round(
                        float(actor_root["pemoin_mixamo_bake_end_frame"])
                        - float(actor_root["pemoin_mixamo_bake_start_frame"])
                        + 1.0
                    )
                ),
            ),
        )
        sample_step_s = float(
            actor_root.get("pemoin_mixamo_source_sample_time_seconds_per_output_frame", 0.0)
        )
        sample_preview_timing = [
            resolve_looped_source_timing(
                idx * sample_step_s,
                cycle_duration_seconds,
                int(round(source_cycle_frames)),
                source_start_frame=source_start_frame,
            )
            for idx in range(preview_count)
        ]
        timing_parity = {
            "intended_sampling_fps": float(actor_root["pemoin_mixamo_scene_fps"]),
            "live_scene_fps_at_bake_check": float(
                actor_root.get(
                    "pemoin_mixamo_live_scene_fps_at_bake_check",
                    actor_root["pemoin_mixamo_scene_fps"],
                )
            ),
            "source_animation_fps": float(actor_root["pemoin_mixamo_source_fps"]),
            "source_cycle_frames": int(
                round(source_cycle_frames)
            ),
            "scene_cycle_frames_used": float(
                actor_root["pemoin_mixamo_cycle_len_frames"]
            ),
            "cycle_duration_seconds": cycle_duration_seconds,
            "sampling_mode": str(
                actor_root.get("pemoin_mixamo_sampling_mode", "legacy_scene_cycle")
            ),
            "source_sample_time_seconds_per_output_frame": sample_step_s,
            "source_sample_frame_float_step": float(
                actor_root.get("pemoin_mixamo_source_sample_frame_float_step", 0.0)
            ),
            "source_sample_frame_float_min": float(
                actor_root.get("pemoin_mixamo_source_sample_frame_float_min", source_start_frame)
            ),
            "source_sample_frame_float_max": float(
                actor_root.get(
                    "pemoin_mixamo_source_sample_frame_float_max",
                    source_start_frame,
                )
            ),
            "source_sample_absolute_progress_frame_min": float(
                actor_root.get(
                    "pemoin_mixamo_source_sample_absolute_progress_frame_min",
                    0.0,
                )
            ),
            "source_sample_absolute_progress_frame_max": float(
                actor_root.get(
                    "pemoin_mixamo_source_sample_absolute_progress_frame_max",
                    0.0,
                )
            ),
            "source_sample_completed_cycles_max": int(
                actor_root.get("pemoin_mixamo_source_sample_completed_cycles_max", 0)
            ),
            "source_sample_frame_float_preview": [
                float(sample.wrapped_source_frame_float) for sample in sample_preview_timing
            ],
            "source_sample_absolute_progress_frame_preview": [
                float(sample.absolute_source_progress_frames)
                for sample in sample_preview_timing
            ],
            "source_sample_completed_cycles_preview": [
                int(sample.completed_cycles) for sample in sample_preview_timing
            ],
            "scene_fps_matches_intended": bool(
                actor_root.get("pemoin_mixamo_scene_fps_matches_intended", True)
            ),
        }
    payload = {
        "actor_root_name": actor_root.name,
        "resolved_armature_name": armature_obj.name,
        "resolved_armature_parent": getattr(armature_obj.parent, "name", None),
        "descendant_object_names": visible_descendants,
        "scene_camera_name": None if camera_obj is None else camera_obj.name,
        "scene_frame_start": int(scene.frame_start),
        "scene_frame_end": int(scene.frame_end),
        "render_resolution": {
            "width": int(scene.render.resolution_x),
            "height": int(scene.render.resolution_y),
            "percentage": int(scene.render.resolution_percentage),
        },
        "render_fps": render_fps,
        "camera_lens": (
            None if camera_data is None else float(getattr(camera_data, "lens", 0.0))
        ),
        "camera_sensor_fit": (
            None if camera_data is None else str(getattr(camera_data, "sensor_fit", ""))
        ),
        "camera_shift_x": (
            None if camera_data is None else float(getattr(camera_data, "shift_x", 0.0))
        ),
        "camera_shift_y": (
            None if camera_data is None else float(getattr(camera_data, "shift_y", 0.0))
        ),
        "camera_sensor_width_mm": (
            None if camera_data is None else float(getattr(camera_data, "sensor_width", 0.0))
        ),
        "camera_sensor_height_mm": (
            None if camera_data is None else float(getattr(camera_data, "sensor_height", 0.0))
        ),
        "render_pixel_aspect_x": float(getattr(scene.render, "pixel_aspect_x", 1.0)),
        "render_pixel_aspect_y": float(getattr(scene.render, "pixel_aspect_y", 1.0)),
        "camera_parity": parity_solution.diagnostics_payload(target_intrinsics),
    }
    if timing_parity is not None:
        payload["timing_parity"] = timing_parity
    vis_dir = spec.run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / "render_parity_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_path = vis_dir / "camera_parity_summary.json"
    summary_path.write_text(
        json.dumps(payload["camera_parity"], indent=2),
        encoding="utf-8",
    )
    samples_path = vis_dir / "camera_parity_samples.csv"
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("term", "target", "effective", "delta"))
        target = np.asarray(target_intrinsics, dtype=np.float64)
        effective = np.asarray(parity_solution.effective_matrix, dtype=np.float64)
        labels = (
            ("fx", target[0, 0], effective[0, 0]),
            ("fy", target[1, 1], effective[1, 1]),
            ("cx", target[0, 2], effective[0, 2]),
            ("cy", target[1, 2], effective[1, 2]),
        )
        for name, target_value, effective_value in labels:
            writer.writerow(
                (
                    name,
                    float(target_value),
                    float(effective_value),
                    float(effective_value - target_value),
                )
    )
    return path


def _rendered_alpha_pixels(path: Path) -> int:
    image = _load_rgba_image(path)
    alpha = np.asarray(image[:, :, 3], dtype=np.float32)
    return int(np.count_nonzero(alpha > (_OVERLAY_ALPHA_THRESHOLD * 255.0)))


def _build_render_visibility_contract(
    *,
    frames_dir: Path,
    grounding_diagnostics: Sequence[GroundingDiagnostic],
) -> list[RenderVisibilityFrame]:
    frame_map = _build_frame_index_map(frames_dir)
    items: list[RenderVisibilityFrame] = []
    for diag in grounding_diagnostics:
        frame_path = frame_map.get(int(diag.frame_index))
        alpha_pixels = 0
        rendered_visible = False
        if frame_path is not None and frame_path.exists():
            alpha_pixels = _rendered_alpha_pixels(frame_path)
            rendered_visible = alpha_pixels > 0
        items.append(
            RenderVisibilityFrame(
                frame_index=int(diag.frame_index),
                rendered_visible=bool(rendered_visible),
                rendered_alpha_pixels=int(alpha_pixels),
                projected_visible=not bool(diag.visibility_culled),
                support_state=str(diag.support_state),
                visibility_contract_state=diag.visibility_contract_state,
            )
        )
    return items


def _write_render_visibility_contract(
    *,
    run_dir: Path,
    frames: Sequence[RenderVisibilityFrame],
) -> Path:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / "render_visibility_contract.json"
    payload = {
        "count": int(len(frames)),
        "rendered_visible_count": int(sum(1 for item in frames if item.rendered_visible)),
        "projected_visible_count": int(sum(1 for item in frames if item.projected_visible)),
        "entries": [
            {
                "frame_index": int(item.frame_index),
                "rendered_visible": bool(item.rendered_visible),
                "rendered_alpha_pixels": int(item.rendered_alpha_pixels),
                "projected_visible": bool(item.projected_visible),
                "support_state": str(item.support_state),
                "visibility_contract_state": item.visibility_contract_state,
            }
            for item in frames
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _enforce_render_visibility_parity(
    *,
    run_dir: Path,
    frames: Sequence[RenderVisibilityFrame],
) -> None:
    frame_list = list(frames)
    tolerated_boundary_lookup: set[int] = set()
    raw_mismatches = [
        int(item.frame_index)
        for item in frame_list
        if item.rendered_visible and not item.projected_visible
    ]
    tolerated_boundary_mismatches: list[int] = []
    mismatch_lookup = {
        int(item.frame_index): idx
        for idx, item in enumerate(frame_list)
        if item.rendered_visible and not item.projected_visible
    }
    for frame_index in raw_mismatches:
        idx = mismatch_lookup[frame_index]
        prev_item = frame_list[idx - 1] if idx > 0 else None
        next_item = frame_list[idx + 1] if idx + 1 < len(frame_list) else None
        is_boundary = bool(
            (prev_item is not None and bool(prev_item.projected_visible) != bool(frame_list[idx].projected_visible))
            or (next_item is not None and bool(next_item.projected_visible) != bool(frame_list[idx].projected_visible))
        )
        isolated = not (
            (prev_item is not None and prev_item.rendered_visible and not prev_item.projected_visible)
            or (next_item is not None and next_item.rendered_visible and not next_item.projected_visible)
        )
        if is_boundary and isolated:
            tolerated_boundary_mismatches.append(int(frame_index))
            tolerated_boundary_lookup.add(int(frame_index))
    mismatches = [
        int(frame_index)
        for frame_index in raw_mismatches
        if int(frame_index) not in tolerated_boundary_lookup
    ]
    raw_projected_but_not_rendered = [
        int(item.frame_index)
        for item in frame_list
        if item.projected_visible and not item.rendered_visible
    ]
    tolerated_empty_boundary_mismatches: list[int] = []
    projected_empty_lookup = {
        int(item.frame_index): idx
        for idx, item in enumerate(frame_list)
        if item.projected_visible and not item.rendered_visible
    }
    for frame_index in raw_projected_but_not_rendered:
        idx = projected_empty_lookup[frame_index]
        prev_item = frame_list[idx - 1] if idx > 0 else None
        next_item = frame_list[idx + 1] if idx + 1 < len(frame_list) else None
        is_boundary = bool(
            (prev_item is not None and bool(prev_item.projected_visible) != bool(frame_list[idx].projected_visible))
            or (next_item is not None and bool(next_item.projected_visible) != bool(frame_list[idx].projected_visible))
        )
        isolated = not (
            (prev_item is not None and prev_item.projected_visible and not prev_item.rendered_visible)
            or (next_item is not None and next_item.projected_visible and not next_item.rendered_visible)
        )
        if is_boundary and isolated:
            tolerated_empty_boundary_mismatches.append(int(frame_index))
            tolerated_boundary_lookup.add(int(frame_index))
    projected_but_not_rendered = [
        int(frame_index)
        for frame_index in raw_projected_but_not_rendered
        if int(frame_index) not in tolerated_boundary_lookup
    ]
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    parity_path = vis_dir / "render_parity_diagnostics.json"
    if parity_path.exists():
        payload = json.loads(parity_path.read_text(encoding="utf-8"))
        payload["render_visibility_parity"] = {
            "count": int(len(frame_list)),
            "rendered_visible_but_projected_off_camera_frames": mismatches,
            "rendered_visible_but_projected_off_camera_boundary_tolerated_frames": tolerated_boundary_mismatches,
            "projected_visible_but_rendered_empty_frames": projected_but_not_rendered,
            "projected_visible_but_rendered_empty_boundary_tolerated_frames": tolerated_empty_boundary_mismatches,
            "mismatch_count": int(len(mismatches)),
            "boundary_tolerated_mismatch_count": int(
                len(tolerated_boundary_mismatches) + len(tolerated_empty_boundary_mismatches)
            ),
            "projected_visible_but_rendered_empty_count": int(
                len(projected_but_not_rendered)
            ),
            "passed": not bool(mismatches or projected_but_not_rendered),
        }
        parity_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if mismatches:
        raise ValueError(
            "Rendered pedestrian visibility disagrees with grounding visibility: "
            f"rendered_visible_but_projected_off_camera={_format_frame_ranges(mismatches)}"
        )
    if projected_but_not_rendered:
        raise ValueError(
            "Pedestrian render produced no visible alpha for projected-visible frames: "
            f"projected_visible_but_rendered_empty={_format_frame_ranges(projected_but_not_rendered)}"
        )


def _write_render_backend_diagnostics(
    *,
    spec: Any,
    timings: Mapping[str, float],
    engine_name: str,
    backend_settings: Mapping[str, Any] | None = None,
    frame_plan: Mapping[str, Any] | None = None,
    pedestrian_rgba: Mapping[str, Any] | None = None,
    wrap_subject_fill: Mapping[str, Any] | None = None,
) -> Path:
    vis_dir = spec.run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": str(engine_name),
        "render": {
            "resolution_scale": float(getattr(getattr(spec, "render", None), "resolution_scale", 1.0)),
            "samples": int(getattr(getattr(spec, "render", None), "samples", 16)),
        },
        "shadow": {
            "enabled": bool(getattr(getattr(spec, "shadow", None), "enabled", True)),
            "map_resolution": str(getattr(getattr(spec, "shadow", None), "map_resolution", "1024")),
            "softness": float(getattr(getattr(spec, "shadow", None), "softness", 1.5)),
            "mode": "single_pass_receiver_luma",
        },
        "timings_seconds": {str(k): float(v) for k, v in timings.items()},
    }
    if backend_settings is not None:
        payload["backend_settings"] = dict(backend_settings)
    if frame_plan is not None:
        payload["frame_plan"] = dict(frame_plan)
    if pedestrian_rgba is not None:
        payload["pedestrian_rgba"] = dict(pedestrian_rgba)
    if wrap_subject_fill is not None:
        payload["wrap_subject_fill"] = dict(wrap_subject_fill)
    path = vis_dir / "render_backend_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _clear_render_output_dir(path: Path | None, *, suffixes: tuple[str, ...]) -> None:
    if path is None or not path.exists():
        return
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in suffixes:
            child.unlink()


def _clear_render_output_dir_for_frames(
    path: Path | None,
    *,
    suffixes: tuple[str, ...],
    frame_indices: Sequence[int] | None,
) -> None:
    if frame_indices is None:
        _clear_render_output_dir(path, suffixes=suffixes)
        return
    if path is None or not path.exists():
        return
    target_frames = {int(frame) for frame in frame_indices}
    if not target_frames:
        return
    for child in path.iterdir():
        if not child.is_file() or child.suffix.lower() not in suffixes:
            continue
        match = re.search(r"(\d+)$", child.stem)
        if match is not None and int(match.group(1)) in target_frames:
            child.unlink()


def _scene_custom_value(scene: Any, key: str, default: Any) -> Any:
    getter = getattr(scene, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(scene, key, default)


def _set_scene_custom_value(scene: Any, key: str, value: Any) -> None:
    try:
        scene[key] = value
        return
    except Exception:
        pass
    try:
        setattr(scene, key, value)
    except Exception:
        return


def _internal_render_shape(
    *,
    resolution_x: int,
    resolution_y: int,
    resolution_scale: float,
) -> tuple[int, int]:
    scale = max(0.1, float(resolution_scale))
    width = max(1, int(round(float(resolution_x) * scale)))
    height = max(1, int(round(float(resolution_y) * scale)))
    return height, width


def _partition_render_frame_indices(
    grounding_diagnostics: Sequence[GroundingDiagnostic],
) -> tuple[list[int], list[int]]:
    visible_frames: list[int] = []
    culled_frames: list[int] = []
    for diag in sorted(grounding_diagnostics, key=lambda item: int(item.frame_index)):
        frame_idx = int(diag.frame_index)
        if bool(diag.visibility_culled):
            culled_frames.append(frame_idx)
        else:
            visible_frames.append(frame_idx)
    return visible_frames, culled_frames


def _compute_render_salience_from_projected_bbox(
    *,
    bbox: tuple[float, float, float, float] | None,
    image_shape: tuple[int, int],
) -> dict[str, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    if bbox is None:
        return {
            "visible_pixels": 0.0,
            "bbox_short_side_px": 0.0,
            "center_distance_ratio": 1.0,
            "boundary_fraction": 1.0,
        }
    x0, y0, x1, y1 = (float(v) for v in bbox)
    bbox_w = max(0.0, x1 - x0)
    bbox_h = max(0.0, y1 - y0)
    visible_pixels = float(bbox_w * bbox_h)
    center = np.asarray([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float32)
    bbox_center = np.asarray([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float32)
    center_delta = bbox_center - center
    center_norm = np.asarray(
        [
            center_delta[0] / max(float(center[0]), 1.0),
            center_delta[1] / max(float(center[1]), 1.0),
        ],
        dtype=np.float32,
    )
    center_distance_ratio = float(np.linalg.norm(center_norm) / np.sqrt(2.0))
    boundary_hits = 0
    if x0 <= 1.0:
        boundary_hits += 1
    if y0 <= 1.0:
        boundary_hits += 1
    if x1 >= float(width - 2):
        boundary_hits += 1
    if y1 >= float(height - 2):
        boundary_hits += 1
    return {
        "visible_pixels": visible_pixels,
        "bbox_short_side_px": float(min(bbox_w, bbox_h)),
        "center_distance_ratio": center_distance_ratio,
        "boundary_fraction": float(boundary_hits) / 4.0,
    }


def _render_salience_policy(
    *,
    visible_pixels: float,
    bbox_short_side_px: float,
    center_distance_ratio: float,
    boundary_fraction: float,
    protect_below_visible_pixels: int,
    protect_below_bbox_short_side_px: int,
    protect_when_center_distance_ratio_below: float,
    reduce_only_when_boundary_fraction_above: float,
    near_visibility_transition: bool,
    reduce_only_near_visibility_transition: bool,
    bbox_missing: bool = False,
) -> tuple[str, str]:
    if bbox_missing:
        return "baseline_protected", "protected_uncertain_projection"
    if float(visible_pixels) < float(protect_below_visible_pixels):
        return "baseline_protected", "protected_tiny_visible_subject"
    if float(bbox_short_side_px) < float(protect_below_bbox_short_side_px):
        return "baseline_protected", "protected_tiny_visible_subject"
    if float(center_distance_ratio) <= float(protect_when_center_distance_ratio_below):
        return "baseline_protected", "protected_central_visible_subject"
    if reduce_only_near_visibility_transition and not bool(near_visibility_transition):
        return "baseline_protected", "protected_non_transition_visible_subject"
    if float(boundary_fraction) <= float(reduce_only_when_boundary_fraction_above):
        return "baseline_protected", "protected_not_boundary_exit_frame"
    return "reduced_allowed", "reduced_visibility_transition_boundary"


def _projected_actor_bbox_for_frame(
    *,
    frame_idx: int,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    actor_root: Any,
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    if int(frame_idx) not in frame_to_c2w:
        raise ValueError(f"Missing trajectory pose for render frame {frame_idx}.")
    scene = bpy.context.scene
    current_frame = int(scene.frame_current)
    try:
        scene.frame_set(int(frame_idx))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        points = _projected_actor_extent_points_world(
            actor_root=actor_root,
            depsgraph=depsgraph,
        )
        uv, valid = project_world_to_image(
            np.asarray(points, dtype=np.float32),
            np.asarray(intrinsics_k, dtype=np.float32),
            camera_to_world_matrix=np.asarray(frame_to_c2w[int(frame_idx)], dtype=np.float32),
            camera_convention="blender",
            image_shape=image_shape,
        )
    finally:
        scene.frame_set(current_frame)
        bpy.context.view_layer.update()
    valid_mask = np.asarray(valid, dtype=bool).reshape(-1)
    if not bool(np.any(valid_mask)):
        return None
    uv_valid = np.asarray(uv, dtype=np.float32).reshape(-1, 2)[valid_mask]
    return (
        float(np.min(uv_valid[:, 0])),
        float(np.min(uv_valid[:, 1])),
        float(np.max(uv_valid[:, 0])),
        float(np.max(uv_valid[:, 1])),
    )


def _partition_visible_render_frames_by_salience(
    *,
    visible_frame_indices: Sequence[int],
    spec: Any,
    actor_root: Any,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    image_shape: tuple[int, int],
) -> tuple[list[int], list[int], dict[str, Any]]:
    adaptive = getattr(getattr(spec, "render", None), "salience_adaptive", None)
    if adaptive is None or not bool(getattr(adaptive, "enabled", True)):
        return (
            [int(frame) for frame in visible_frame_indices],
            [],
            {
                "enabled": False,
                "baseline_frame_count": int(len(visible_frame_indices)),
                "reduced_frame_count": 0,
                "entries": [],
            },
        )
    baseline_frames: list[int] = []
    reduced_frames: list[int] = []
    entries: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    visible_frames_sorted = sorted({int(frame) for frame in visible_frame_indices})
    for idx, frame_idx in enumerate(visible_frames_sorted):
        prev_frame = None if idx == 0 else int(visible_frames_sorted[idx - 1])
        next_frame = (
            None if idx >= (len(visible_frames_sorted) - 1) else int(visible_frames_sorted[idx + 1])
        )
        near_visibility_transition = bool(
            prev_frame is None
            or next_frame is None
            or abs(int(frame_idx) - int(prev_frame)) > 1
            or abs(int(next_frame) - int(frame_idx)) > 1
        )
        bbox = _projected_actor_bbox_for_frame(
            frame_idx=frame_idx,
            intrinsics_k=intrinsics_k,
            frame_to_c2w=frame_to_c2w,
            actor_root=actor_root,
            image_shape=image_shape,
        )
        metrics = _compute_render_salience_from_projected_bbox(
            bbox=bbox,
            image_shape=image_shape,
        )
        tier, reason = _render_salience_policy(
            visible_pixels=float(metrics["visible_pixels"]),
            bbox_short_side_px=float(metrics["bbox_short_side_px"]),
            center_distance_ratio=float(metrics["center_distance_ratio"]),
            boundary_fraction=float(metrics["boundary_fraction"]),
            protect_below_visible_pixels=int(
                getattr(adaptive, "protect_below_visible_pixels", 10000)
            ),
            protect_below_bbox_short_side_px=int(
                getattr(adaptive, "protect_below_bbox_short_side_px", 56)
            ),
            protect_when_center_distance_ratio_below=float(
                getattr(adaptive, "protect_when_center_distance_ratio_below", 0.30)
            ),
            reduce_only_when_boundary_fraction_above=float(
                getattr(adaptive, "reduce_only_when_boundary_fraction_above", 0.24)
            ),
            near_visibility_transition=bool(near_visibility_transition),
            reduce_only_near_visibility_transition=bool(
                getattr(adaptive, "reduce_only_near_visibility_transition", True)
            ),
            bbox_missing=bbox is None,
        )
        if tier == "reduced_allowed":
            reduced_frames.append(frame_idx)
        else:
            baseline_frames.append(frame_idx)
        reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0)) + 1
        entries.append(
            {
                "frame_index": int(frame_idx),
                "tier": str(tier),
                "reason": str(reason),
                "bbox": None if bbox is None else [float(v) for v in bbox],
                "visible_pixels": float(metrics["visible_pixels"]),
                "bbox_short_side_px": float(metrics["bbox_short_side_px"]),
                "center_distance_ratio": float(metrics["center_distance_ratio"]),
                "boundary_fraction": float(metrics["boundary_fraction"]),
                "near_visibility_transition": bool(near_visibility_transition),
            }
        )
    return baseline_frames, reduced_frames, {
        "enabled": True,
        "low_salience_resolution_scale": float(
            getattr(adaptive, "low_salience_resolution_scale", 0.85)
        ),
        "thresholds": {
            "protect_below_visible_pixels": int(
                getattr(adaptive, "protect_below_visible_pixels", 10000)
            ),
            "protect_below_bbox_short_side_px": int(
                getattr(adaptive, "protect_below_bbox_short_side_px", 56)
            ),
            "protect_when_center_distance_ratio_below": float(
                getattr(adaptive, "protect_when_center_distance_ratio_below", 0.30)
            ),
            "reduce_only_when_boundary_fraction_above": float(
                getattr(adaptive, "reduce_only_when_boundary_fraction_above", 0.24)
            ),
            "reduce_only_near_visibility_transition": bool(
                getattr(adaptive, "reduce_only_near_visibility_transition", True)
            ),
        },
        "baseline_frame_count": int(len(baseline_frames)),
        "reduced_frame_count": int(len(reduced_frames)),
        "baseline_protected_frame_count": int(len(baseline_frames)),
        "reduced_allowed_frame_count": int(len(reduced_frames)),
        "baseline_frame_indices": [int(frame) for frame in baseline_frames],
        "reduced_frame_indices": [int(frame) for frame in reduced_frames],
        "reason_counts": {str(key): int(value) for key, value in sorted(reason_counts.items())},
        "entries": entries,
    }


def _reduced_shadow_map_resolution(map_resolution: str) -> str:
    normalized = str(map_resolution).strip()
    steps = ["512", "1024", "2048", "4096"]
    if normalized not in steps:
        return normalized
    index = steps.index(normalized)
    if index <= 1:
        return normalized
    return steps[index - 1]


def _is_subject_fill_light(light_obj: Any) -> bool:
    role = str(getattr(light_obj, "name", "")).strip().lower()
    if hasattr(light_obj, "get"):
        role = str(light_obj.get("pemoin_light_role", role)).strip().lower()
        transport_mode = str(light_obj.get(_PEMOIN_LIGHT_TRANSPORT_MODE, "")).strip().lower()
        if transport_mode == _WRAP_SUBJECT_FILL_TRANSPORT_MODE:
            return True
    return role in {"wrap_key_fill", "counter_wrap_fill", "sky_fill"}


@contextmanager
def _temporary_low_salience_render_policy(
    *,
    spec: Any,
):
    adaptive = getattr(getattr(spec, "render", None), "salience_adaptive", None)
    scene = bpy.context.scene
    eevee = getattr(scene, "eevee", None)
    if eevee is None:
        eevee = getattr(scene, "eevee_next", None)
    fill_reduction_enabled = bool(
        getattr(adaptive, "fill_light_reduction_enabled", True)
    )
    shadow_reduction_enabled = bool(
        getattr(adaptive, "shadow_quality_reduction_enabled", True)
    )
    light_restore: list[tuple[Any, float | None, bool | None]] = []
    reduced_fill_count = 0
    disabled_fill_shadow_count = 0
    for light_obj in getattr(bpy.data, "objects", ()):
        if getattr(light_obj, "type", None) != "LIGHT":
            continue
        light_data = getattr(light_obj, "data", None)
        energy = None if light_data is None else getattr(light_data, "energy", None)
        use_shadow = None if light_data is None else getattr(light_data, "use_shadow", None)
        light_restore.append((light_obj, energy, use_shadow))
        if light_data is None or not _is_subject_fill_light(light_obj):
            continue
        if fill_reduction_enabled and energy is not None:
            light_data.energy = float(energy) * 0.88
            reduced_fill_count += 1
        if shadow_reduction_enabled and hasattr(light_data, "use_shadow"):
            if bool(getattr(light_data, "use_shadow", False)):
                disabled_fill_shadow_count += 1
            light_data.use_shadow = False
    original_shadow_cube = getattr(eevee, "shadow_cube_size", None) if eevee is not None else None
    original_shadow_cascade = getattr(eevee, "shadow_cascade_size", None) if eevee is not None else None
    effective_shadow_map_resolution = str(
        getattr(getattr(spec, "shadow", None), "map_resolution", "1024")
    )
    if shadow_reduction_enabled:
        effective_shadow_map_resolution = _reduced_shadow_map_resolution(
            effective_shadow_map_resolution
        )
        if eevee is not None and hasattr(eevee, "shadow_cube_size"):
            eevee.shadow_cube_size = effective_shadow_map_resolution
        if eevee is not None and hasattr(eevee, "shadow_cascade_size"):
            eevee.shadow_cascade_size = effective_shadow_map_resolution
    try:
        yield {
            "fill_light_reduction_enabled": fill_reduction_enabled,
            "shadow_quality_reduction_enabled": shadow_reduction_enabled,
            "reduced_fill_light_count": int(reduced_fill_count),
            "disabled_fill_shadow_count": int(disabled_fill_shadow_count),
            "effective_shadow_map_resolution": str(effective_shadow_map_resolution),
        }
    finally:
        for light_obj, energy, use_shadow in light_restore:
            light_data = getattr(light_obj, "data", None)
            if light_data is None:
                continue
            if energy is not None:
                light_data.energy = energy
            if use_shadow is not None and hasattr(light_data, "use_shadow"):
                light_data.use_shadow = use_shadow
        if eevee is not None and hasattr(eevee, "shadow_cube_size") and original_shadow_cube is not None:
            eevee.shadow_cube_size = original_shadow_cube
        if eevee is not None and hasattr(eevee, "shadow_cascade_size") and original_shadow_cascade is not None:
            eevee.shadow_cascade_size = original_shadow_cascade


def _materialize_render_artifacts_to_target_shape(
    *,
    frame_indices: Sequence[int],
    target_shape: tuple[int, int],
    frames_dir: Path,
    depth_dir: Path,
    shadow_dir: Path | None,
) -> None:
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    for frame_idx in sorted({int(frame) for frame in frame_indices}):
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        if frame_path.exists():
            rgba = _load_rgba_image(frame_path)
            resized = _resize_array(
                rgba,
                (target_h, target_w),
                interpolation="bilinear",
            )
            _write_rgba_image(frame_path, np.asarray(resized, dtype=np.uint8))
        depth_path = depth_dir / f"{frame_idx:06d}.npz"
        if depth_path.exists():
            depth = _load_depth_npz_array(depth_path)
            resized_depth = _resize_array(
                depth,
                (target_h, target_w),
                interpolation="bilinear",
            )
            np.savez_compressed(
                depth_path,
                depth=np.asarray(resized_depth, dtype=np.float32),
            )
        if shadow_dir is not None:
            shadow_path = shadow_dir / f"shadow_{frame_idx:04d}.png"
            if shadow_path.exists():
                shadow_rgba = _load_rgba_image(shadow_path)
                resized_shadow = _resize_array(
                    shadow_rgba,
                    (target_h, target_w),
                    interpolation="bilinear",
                )
                _write_rgba_image(
                    shadow_path,
                    np.asarray(resized_shadow, dtype=np.uint8),
                )


def _write_visibility_culled_frame_artifacts(
    *,
    frames_dir: Path,
    depth_dir: Path,
    shadow_dir: Path | None,
    frame_indices: Sequence[int],
    image_shape: tuple[int, int],
) -> None:
    if not frame_indices:
        return
    height, width = int(image_shape[0]), int(image_shape[1])
    empty_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    empty_depth = np.zeros((height, width), dtype=np.float32)
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    if shadow_dir is not None:
        shadow_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in sorted({int(frame) for frame in frame_indices}):
        _write_rgba_image(frames_dir / f"frame_{frame_idx:04d}.png", empty_rgba)
        np.savez_compressed(
            depth_dir / f"{frame_idx:06d}.npz",
            depth=np.asarray(empty_depth, dtype=np.float32),
        )
        if shadow_dir is not None:
            _write_rgba_image(shadow_dir / f"shadow_{frame_idx:04d}.png", empty_rgba)


@contextmanager
def _render_progress_scope(
    *,
    progress_id: str,
    label: str,
    resolution_scale: float | None = None,
    rerender_index: int | None = None,
):
    scene = bpy.context.scene
    total_frames = max(0, int(scene.frame_end) - int(scene.frame_start) + 1)
    seen_frames: set[int] = set()
    current = 0

    def _on_render_write(*_args) -> None:
        nonlocal current
        frame_idx = int(getattr(scene, "frame_current", -1))
        if frame_idx in seen_frames:
            return
        seen_frames.add(frame_idx)
        current = len(seen_frames)
        progress_step(
            progress_id=progress_id,
            current=current,
            total=total_frames,
        )

    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    render_write_handlers = getattr(handlers, "render_write", None)
    progress_begin(
        progress_id=progress_id,
        label=label,
        total=total_frames,
        resolution_scale=resolution_scale,
        rerender_index=rerender_index,
    )
    if render_write_handlers is not None:
        render_write_handlers.append(_on_render_write)
    try:
        yield
    finally:
        if render_write_handlers is not None and _on_render_write in render_write_handlers:
            render_write_handlers.remove(_on_render_write)
        progress_end(
            progress_id=progress_id,
            current=current,
            total=total_frames,
        )


def render_as_image_sequence(
    output_dir: Path,
    *,
    use_compositing: bool | None = None,
    progress_id: str | None = None,
    progress_label: str | None = None,
    resolution_scale: float | None = None,
    rerender_index: int | None = None,
    frame_indices: Sequence[int] | None = None,
):
    """Render animation as PNG sequence with transparent background."""
    with log_scope("Render"):
        scene = bpy.context.scene
        output_dir.mkdir(parents=True, exist_ok=True)

        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        fast_png_compression = bool(_scene_custom_value(scene, "_pemoin_fast_png_compression", True))
        if fast_png_compression and hasattr(scene.render.image_settings, "compression"):
            # Intermediate render frames are pipeline-internal artifacts; prefer encode speed.
            scene.render.image_settings.compression = 0
        if hasattr(scene.render, "use_overwrite"):
            scene.render.use_overwrite = True
        if hasattr(scene.render, "use_file_extension"):
            scene.render.use_file_extension = True
        # Enable transparent film so background is rendered as transparent
        # (works for both EEVEE and Cycles in modern Blender)
        try:
            scene.render.film_transparent = True
        except Exception:
            # Older Blender versions may not have this attribute
            log_warning("Could not set film_transparent to True")
        if use_compositing is not None and hasattr(scene.render, "use_compositing"):
            scene.render.use_compositing = bool(use_compositing)

        # scene.render.image_settings.compression = 15
        scene.render.filepath = str(output_dir / "frame_")
        original_frame_start = int(scene.frame_start)
        original_frame_end = int(scene.frame_end)
        requested_frames = None
        if frame_indices is not None:
            requested_frames = sorted({int(frame) for frame in frame_indices})
            if not requested_frames:
                return
        frame_ranges: list[tuple[int, int]]
        if requested_frames is None:
            frame_ranges = [(original_frame_start, original_frame_end)]
        else:
            frame_ranges = []
            range_start = requested_frames[0]
            range_end = requested_frames[0]
            for frame_idx in requested_frames[1:]:
                if frame_idx == (range_end + 1):
                    range_end = frame_idx
                    continue
                frame_ranges.append((range_start, range_end))
                range_start = frame_idx
                range_end = frame_idx
            frame_ranges.append((range_start, range_end))
        try:
            for range_start, range_end in frame_ranges:
                scene.frame_start = int(range_start)
                scene.frame_end = int(range_end)
                log_info(
                    f"Rendering animation frames {scene.frame_start}-{scene.frame_end} -> {output_dir}"
                )
                if progress_id is not None and progress_label is not None:
                    progress_message(
                        progress_id=progress_id,
                        message=f"frames {scene.frame_start}-{scene.frame_end}",
                    )
                    with _render_progress_scope(
                        progress_id=progress_id,
                        label=progress_label,
                        resolution_scale=resolution_scale,
                        rerender_index=rerender_index,
                    ):
                        bpy.ops.render.render(animation=True)
                else:
                    bpy.ops.render.render(animation=True)
        finally:
            scene.frame_start = original_frame_start
            scene.frame_end = original_frame_end
        log_info(f"Render complete. Frames saved to {output_dir}")


def _blender_version_string() -> str:
    version = getattr(bpy.app, "version_string", None)
    if version:
        return str(version)
    version_tuple = getattr(bpy.app, "version", None)
    if isinstance(version_tuple, tuple):
        return ".".join(str(part) for part in version_tuple)
    return "unknown"


def _raise_render_output_compatibility_error(
    export_api: str,
    message: str,
) -> None:
    raise RuntimeError(
        "Blender render output configuration is incompatible with this Blender build "
        f"(version={_blender_version_string()} export_api={export_api}): {message}"
    )


def _configure_output_format(
    image_format: Any,
    *,
    file_format: str,
    color_mode: str,
    color_depth: str | None = None,
    exr_codec: str | None = None,
) -> None:
    image_format.file_format = file_format
    image_format.color_mode = color_mode
    if color_depth is not None and hasattr(image_format, "color_depth"):
        image_format.color_depth = color_depth
    if exr_codec is not None and hasattr(image_format, "exr_codec"):
        image_format.exr_codec = exr_codec


def _find_node_socket(
    sockets: Any,
    *,
    preferred_names: Sequence[str],
    export_api: str,
    node: Any,
    socket_kind: str,
    fallback_index: int | None = None,
):
    if sockets is None:
        _raise_render_output_compatibility_error(
            export_api,
            f"{type(node).__name__} exposes no {socket_kind} sockets.",
        )
    resolved_sockets = []
    try:
        resolved_sockets = list(sockets)
    except Exception:
        resolved_sockets = []
    for preferred_name in preferred_names:
        for socket in resolved_sockets:
            if getattr(socket, "name", None) == preferred_name:
                return socket
        try:
            return sockets[preferred_name]
        except Exception:
            continue
    if fallback_index is not None:
        try:
            return sockets[fallback_index]
        except Exception:
            pass
    socket_names = []
    try:
        socket_names = [getattr(socket, "name", "<unnamed>") for socket in sockets]
    except Exception:
        socket_names = []
    _raise_render_output_compatibility_error(
        export_api,
        f"Expected {socket_kind} socket in {list(preferred_names)!r} on {type(node).__name__}; "
        f"available {socket_kind}s={socket_names}.",
    )


def _find_node_input_socket(
    node: Any,
    *,
    preferred_names: Sequence[str],
    export_api: str,
    fallback_index: int | None = None,
):
    return _find_node_socket(
        getattr(node, "inputs", None),
        preferred_names=preferred_names,
        export_api=export_api,
        node=node,
        socket_kind="input",
        fallback_index=fallback_index,
    )


def _find_node_output_socket(
    node: Any,
    *,
    preferred_names: Sequence[str],
    export_api: str,
    fallback_index: int | None = None,
):
    return _find_node_socket(
        getattr(node, "outputs", None),
        preferred_names=preferred_names,
        export_api=export_api,
        node=node,
        socket_kind="output",
        fallback_index=fallback_index,
    )


def _configure_legacy_output_slot(node: Any, *, path_prefix: str) -> None:
    slots = getattr(node, "file_slots", None)
    if slots is None or len(slots) == 0:
        _raise_render_output_compatibility_error(
            "legacy_node_tree",
            f"{type(node).__name__} exposes no file_slots collection.",
        )
    while len(slots) > 1:
        slots.remove(slots[-1])
    slots[0].path = path_prefix


def _configure_group_output_item(
    node: Any,
    *,
    socket_type: str,
    name: str,
    file_format: str,
    color_mode: str,
    color_depth: str | None = None,
    exr_codec: str | None = None,
):
    items = getattr(node, "file_output_items", None)
    if items is None:
        _raise_render_output_compatibility_error(
            "compositing_node_group",
            f"{type(node).__name__} exposes no file_output_items collection.",
        )
    try:
        item = items.new(socket_type, name)
    except Exception as exc:
        _raise_render_output_compatibility_error(
            "compositing_node_group",
            f"Could not create output item name={name!r} socket_type={socket_type!r}: {exc}",
        )
    if hasattr(item, "override_node_format"):
        item.override_node_format = True
    elif hasattr(item, "use_node_format"):
        item.use_node_format = False
    else:
        _raise_render_output_compatibility_error(
            "compositing_node_group",
            f"{type(item).__name__} exposes neither override_node_format nor use_node_format.",
        )
    item_format = getattr(item, "format", None)
    if item_format is None:
        _raise_render_output_compatibility_error(
            "compositing_node_group",
            f"{type(item).__name__} exposes no format configuration.",
        )
    _configure_output_format(
        item_format,
        file_format=file_format,
        color_mode=color_mode,
        color_depth=color_depth,
        exr_codec=exr_codec,
    )
    return item


def _configure_legacy_render_output_nodes(
    depth_output_dir: Path,
    shadow_output_dir: Path | None,
) -> str | None:
    scene = bpy.context.scene
    tree = getattr(scene, "node_tree", None)
    if tree is None:
        return None
    tree.nodes.clear()
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    depth_out = tree.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = str(depth_output_dir)
    _configure_output_format(
        depth_out.format,
        file_format="OPEN_EXR",
        color_mode="BW",
        color_depth="32",
        exr_codec="ZIP",
    )
    _configure_legacy_output_slot(depth_out, path_prefix="depth_")
    tree.links.new(
        render_layers.outputs["Depth"],
        _find_node_input_socket(
            depth_out,
            preferred_names=("Image",),
            fallback_index=0,
            export_api="legacy_node_tree",
        ),
    )
    if shadow_output_dir is not None:
        rgb_node = tree.nodes.new("CompositorNodeRGB")
        rgb_node.outputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
        shadow_out = tree.nodes.new("CompositorNodeOutputFile")
        shadow_out.base_path = str(shadow_output_dir)
        _configure_output_format(
            shadow_out.format,
            file_format="PNG",
            color_mode="RGBA",
        )
        _configure_legacy_output_slot(shadow_out, path_prefix="shadow_")
        tree.links.new(rgb_node.outputs["Image"], set_alpha.inputs["Image"])
        tree.links.new(render_layers.outputs["Shadow"], set_alpha.inputs["Alpha"])
        tree.links.new(
            set_alpha.outputs["Image"],
            _find_node_input_socket(
                shadow_out,
                preferred_names=("Image",),
                fallback_index=0,
                export_api="legacy_node_tree",
            ),
        )
    return "legacy_node_tree"


def _configure_compositing_group_render_outputs(
    depth_output_dir: Path,
    shadow_output_dir: Path | None,
) -> str | None:
    scene = bpy.context.scene
    if not hasattr(scene, "compositing_node_group"):
        return None
    tree = bpy.data.node_groups.new("PEMOINDepthOutput", "CompositorNodeTree")
    scene.compositing_node_group = tree
    if hasattr(tree, "interface"):
        tree.interface.new_socket(
            name="Image",
            in_out="OUTPUT",
            socket_type="NodeSocketColor",
        )
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    depth_out = tree.nodes.new("CompositorNodeOutputFile")
    depth_out.directory = str(depth_output_dir)
    depth_out.file_name = "depth_"
    _configure_group_output_item(
        depth_out,
        socket_type="FLOAT",
        name="Depth",
        file_format="OPEN_EXR",
        color_mode="BW",
        color_depth="32",
        exr_codec="ZIP",
    )
    group_output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(render_layers.outputs["Image"], group_output.inputs[0])
    tree.links.new(
        render_layers.outputs["Depth"],
        _find_node_input_socket(
            depth_out,
            preferred_names=("Depth",),
            export_api="compositing_node_group",
        ),
    )
    if shadow_output_dir is not None:
        rgb_node = tree.nodes.new("CompositorNodeRGB")
        rgb_node.outputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
        set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
        shadow_out = tree.nodes.new("CompositorNodeOutputFile")
        shadow_out.directory = str(shadow_output_dir)
        shadow_out.file_name = "shadow_"
        _configure_group_output_item(
            shadow_out,
            socket_type="RGBA",
            name="Shadow",
            file_format="PNG",
            color_mode="RGBA",
        )
        tree.links.new(
            _find_node_output_socket(
                rgb_node,
                preferred_names=("Image", "Color"),
                fallback_index=0,
                export_api="compositing_node_group",
            ),
            set_alpha.inputs["Image"],
        )
        tree.links.new(
            _find_node_output_socket(
                render_layers,
                preferred_names=("Shadow",),
                export_api="compositing_node_group",
            ),
            set_alpha.inputs["Alpha"],
        )
        tree.links.new(
            set_alpha.outputs["Image"],
            _find_node_input_socket(
                shadow_out,
                preferred_names=("Shadow",),
                export_api="compositing_node_group",
            ),
        )
    return "compositing_node_group"


def _configure_render_output_nodes(
    depth_output_dir: Path,
    shadow_output_dir: Path | None,
) -> str | None:
    scene = bpy.context.scene
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    if shadow_output_dir is not None:
        shadow_output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(scene, "use_nodes"):
        scene.use_nodes = True
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True
    view_layer = bpy.context.view_layer
    if hasattr(view_layer, "use_pass_z"):
        view_layer.use_pass_z = True
    if shadow_output_dir is not None and hasattr(view_layer, "use_pass_shadow"):
        view_layer.use_pass_shadow = True
    export_api = _configure_compositing_group_render_outputs(
        depth_output_dir,
        shadow_output_dir,
    )
    if export_api is not None:
        return export_api
    return _configure_legacy_render_output_nodes(depth_output_dir, shadow_output_dir)


def _materialize_depth_npz_from_host_python(
    *,
    depth_exr_dir: Path,
    depth_output_dir: Path,
    host_python: Path | None,
    export_api: str,
) -> None:
    python_exe = None if host_python is None else Path(host_python).expanduser().resolve()
    if python_exe is None:
        raise RuntimeError(
            "Blender depth EXR materialization requires --host-python so PEMOIN can "
            "decode EXRs outside Blender's embedded Python."
        )
    decoder_script = Path(__file__).resolve().parent / "depth_decode.py"
    cmd = [
        str(python_exe),
        str(decoder_script),
        "--depth-exr-dir",
        str(depth_exr_dir),
        "--depth-output-dir",
        str(depth_output_dir),
        "--blender-version",
        _blender_version_string(),
        "--export-api",
        str(export_api),
    ]
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Host Python depth EXR materialization failed "
            f"(exit code {result.returncode})."
        )


def _materialize_shadow_png_sequence_from_host_python(
    *,
    shadow_render_dir: Path,
    shadow_output_dir: Path,
    baseline_render_dir: Path | None,
    host_python: Path | None,
    export_api: str,
) -> None:
    python_exe = None if host_python is None else Path(host_python).expanduser().resolve()
    if python_exe is None:
        raise RuntimeError(
            "Blender shadow PNG materialization requires --host-python so PEMOIN can "
            "synthesize shadow alpha outside Blender's embedded Python."
        )
    script = Path(__file__).resolve().parent / "shadow_extract.py"
    cmd = [
        str(python_exe),
        str(script),
        "--shadow-render-dir",
        str(shadow_render_dir),
        "--shadow-output-dir",
        str(shadow_output_dir),
        "--blender-version",
        _blender_version_string(),
        "--export-api",
        str(export_api),
    ]
    if baseline_render_dir is not None:
        cmd.extend(
            [
                "--baseline-render-dir",
                str(baseline_render_dir),
            ]
        )
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Host Python shadow PNG materialization failed "
            f"(exit code {result.returncode})."
        )


def _materialize_single_pass_shadow_png_sequence_from_host_python(
    *,
    shadow_render_dir: Path,
    shadow_output_dir: Path,
    host_python: Path | None,
    export_api: str,
) -> None:
    _materialize_shadow_png_sequence_from_host_python(
        shadow_render_dir=shadow_render_dir,
        shadow_output_dir=shadow_output_dir,
        baseline_render_dir=None,
        host_python=host_python,
        export_api=export_api,
    )


def render_as_image_sequence_with_depth(
    output_dir: Path,
    *,
    depth_output_dir: Path,
    shadow_output_dir: Path | None,
    spec: SceneSpec,
    actor_root: bpy.types.Object,
    receiver_obj: bpy.types.Object | None,
    render_pass_name: str = "Pedestrian render",
    render_pass_id: str = "pedestrian_render",
    resolution_scale: float | None = None,
    rerender_index: int | None = None,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Render animation as PNG sequence plus EXR-backed depth NPZs."""
    with log_scope("Render"):
        depth_exr_dir = output_dir.parent / "_pedestrian_depth_exr"
        shadow_render_dir = (
            None
            if shadow_output_dir is None
            else shadow_output_dir.parent / "_shadow_catcher_render"
        )
        receiver_state = None
        timings: dict[str, float] = {}
        target_frame_set = None if frame_indices is None else {int(frame) for frame in frame_indices}
        if depth_exr_dir.exists():
            for path in depth_exr_dir.glob("*"):
                if not path.is_file():
                    continue
                if target_frame_set is None:
                    path.unlink()
                    continue
                match = re.search(r"(\d+)$", path.stem)
                if match is not None and int(match.group(1)) in target_frame_set:
                    path.unlink()
        if shadow_output_dir is not None and shadow_output_dir.exists():
            for path in shadow_output_dir.glob("*"):
                if not path.is_file():
                    continue
                if path.name == "metadata.json":
                    path.unlink()
                    continue
                if target_frame_set is None:
                    path.unlink()
                    continue
                match = re.search(r"(\d+)$", path.stem)
                if match is not None and int(match.group(1)) in target_frame_set:
                    path.unlink()
        if shadow_render_dir is not None and shadow_render_dir.exists():
            for path in shadow_render_dir.glob("*"):
                if not path.is_file():
                    continue
                if target_frame_set is None:
                    path.unlink()
                    continue
                match = re.search(r"(\d+)$", path.stem)
                if match is not None and int(match.group(1)) in target_frame_set:
                    path.unlink()
        export_api = _configure_render_output_nodes(depth_exr_dir, shadow_render_dir)
        if export_api is None:
            raise RuntimeError(
                "Blender depth export could not be configured. "
                "This Blender build exposes neither scene.compositing_node_group "
                "nor scene.node_tree."
            )
        if receiver_obj is not None:
            receiver_state = (
                bool(getattr(receiver_obj, "hide_render", False)),
                bool(getattr(receiver_obj, "visible_camera", True)),
            )
            receiver_obj.visible_camera = False
        try:
            render_started = time.perf_counter()
            render_as_image_sequence(
                output_dir,
                use_compositing=True,
                progress_id=render_pass_id,
                progress_label=render_pass_name,
                resolution_scale=resolution_scale,
                rerender_index=rerender_index,
                frame_indices=frame_indices,
            )
            timings["pedestrian_render"] = float(time.perf_counter() - render_started)
        finally:
            if receiver_obj is not None and receiver_state is not None:
                hide_render, visible_camera = receiver_state
                receiver_obj.hide_render = hide_render
                receiver_obj.visible_camera = visible_camera
        depth_started = time.perf_counter()
        _materialize_depth_npz_from_host_python(
            depth_exr_dir=depth_exr_dir,
            depth_output_dir=depth_output_dir,
            host_python=spec.host_python,
            export_api=export_api,
        )
        timings["depth_materialize"] = float(time.perf_counter() - depth_started)
        log_info(
            "Pedestrian depth NPZ sequence written to "
            f"{depth_output_dir} (mode=z_pass_exr export_api={export_api})"
        )
        if shadow_output_dir is not None and shadow_render_dir is not None:
            shadow_started = time.perf_counter()
            _materialize_single_pass_shadow_png_sequence_from_host_python(
                shadow_render_dir=shadow_render_dir,
                shadow_output_dir=shadow_output_dir,
                host_python=spec.host_python,
                export_api=export_api,
            )
            timings["shadow_render_and_materialize"] = float(time.perf_counter() - shadow_started)
            log_info(f"Shadow catcher PNG sequence written to {shadow_output_dir}")
        return timings


def render_pedestrian(
    spec: SceneSpec,
    render_width: int,
    render_height: int,
    *,
    target_intrinsics: np.ndarray,
    parity_solution: BlenderCameraSolution,
    grounding_diagnostics: Sequence[GroundingDiagnostic],
) -> Path:
    with log_scope("Render"):
        actor_root = _resolve_actor_root(spec.pedestrian_actor_name)
        armature_obj = _find_actor_armature(spec.pedestrian_actor_name)
        log_info(
            f"Resolved actor for render: root={actor_root.name} armature={armature_obj.name}"
        )

        scene = bpy.context.scene
        if spec.sampling_fps is not None:
            fps = float(spec.sampling_fps)
            if fps <= 0:
                raise ValueError(f"Invalid sampling_fps: {fps}")
            scene.render.fps = max(1, int(round(fps)))
            scene.render.fps_base = scene.render.fps / fps

        receiver_obj = _configure_shadow_receiver_animation(
            spec=spec,
            actor_root=actor_root,
            grounding_diagnostics=grounding_diagnostics,
        )
        diag_path = _write_render_parity_diagnostics(
            spec=spec,
            actor_root=actor_root,
            armature_obj=armature_obj,
            target_intrinsics=target_intrinsics,
            parity_solution=parity_solution,
        )
        log_info(f"Render parity diagnostics written: {diag_path}")

        frames_dir = ResourceStore.blender_artifact_dir_for(
            spec.run_dir,
            "pedestrian_frames",
        )
        depth_dir = ResourceStore.blender_artifact_dir_for(
            spec.run_dir,
            "pedestrian_depth_frames",
        )
        shadow_dir = (
            ResourceStore.blender_artifact_dir_for(spec.run_dir, "shadow_frames")
            if bool(getattr(getattr(spec, "shadow", None), "enabled", True))
            else None
        )
        render_spec = getattr(spec, "render", None)
        current_resolution_scale = float(getattr(render_spec, "resolution_scale", 1.0))
        current_timings: dict[str, float] = {}
        pedestrian_rgba_diagnostics: dict[str, Any] | None = None
        visible_frame_indices, visibility_culled_frame_indices = _partition_render_frame_indices(
            grounding_diagnostics
        )
        c2w, loaded_frame_indices = load_trajectory(spec.trajectory_path)
        frame_to_c2w = {
            int(frame_idx): np.asarray(c2w[idx], dtype=np.float32)
            for idx, frame_idx in enumerate(np.asarray(loaded_frame_indices, dtype=np.int32).tolist())
        }
        render_frame_shape = _internal_render_shape(
            resolution_x=render_width,
            resolution_y=render_height,
            resolution_scale=current_resolution_scale,
        )
        baseline_frame_indices, reduced_frame_indices, salience_adaptive_diagnostics = (
            _partition_visible_render_frames_by_salience(
                visible_frame_indices=visible_frame_indices,
                spec=spec,
                actor_root=actor_root,
                intrinsics_k=np.asarray(target_intrinsics, dtype=np.float32),
                frame_to_c2w=frame_to_c2w,
                image_shape=(int(render_height), int(render_width)),
            )
        )

        def _render_once(
            resolution_scale: float,
            *,
            frame_indices: Sequence[int] | None,
        ) -> dict[str, float]:
            setup_character_only_render(
                actor_root=actor_root,
                resolution_x=render_width,
                resolution_y=render_height,
                resolution_scale=float(resolution_scale),
                extra_visible_objects=(() if receiver_obj is None else (receiver_obj,)),
            )
            _clear_render_output_dir_for_frames(
                frames_dir,
                suffixes=(".png",),
                frame_indices=frame_indices,
            )
            _clear_render_output_dir_for_frames(
                depth_dir,
                suffixes=(".npz", ".json"),
                frame_indices=frame_indices,
            )
            _clear_render_output_dir_for_frames(
                shadow_dir,
                suffixes=(".png", ".json"),
                frame_indices=frame_indices,
            )
            return render_as_image_sequence_with_depth(
                frames_dir,
                depth_output_dir=depth_dir,
                shadow_output_dir=shadow_dir,
                spec=spec,
                actor_root=actor_root,
                receiver_obj=receiver_obj,
                render_pass_name="Pedestrian render",
                render_pass_id="pedestrian_render",
                resolution_scale=float(resolution_scale),
                rerender_index=0,
                frame_indices=frame_indices,
            )

        if baseline_frame_indices:
            current_timings = _render_once(
                current_resolution_scale,
                frame_indices=baseline_frame_indices,
            )
        if reduced_frame_indices:
            reduced_resolution_scale = float(
                getattr(
                    getattr(render_spec, "salience_adaptive", None),
                    "low_salience_resolution_scale",
                    min(current_resolution_scale, 0.85),
                )
            )
            salience_adaptive_diagnostics["reduced_target_shape"] = [
                int(
                    _internal_render_shape(
                        resolution_x=render_width,
                        resolution_y=render_height,
                        resolution_scale=reduced_resolution_scale,
                    )[0]
                ),
                int(
                    _internal_render_shape(
                        resolution_x=render_width,
                        resolution_y=render_height,
                        resolution_scale=reduced_resolution_scale,
                    )[1]
                ),
            ]
            with _temporary_low_salience_render_policy(spec=spec) as reduced_policy_diag:
                reduced_timings = _render_once(
                    reduced_resolution_scale,
                    frame_indices=reduced_frame_indices,
                )
            for key, value in reduced_timings.items():
                current_timings[f"{key}_low_salience"] = float(value)
            upsample_started = time.perf_counter()
            _materialize_render_artifacts_to_target_shape(
                frame_indices=reduced_frame_indices,
                target_shape=render_frame_shape,
                frames_dir=frames_dir,
                depth_dir=depth_dir,
                shadow_dir=shadow_dir,
            )
            current_timings["low_salience_render_upsample"] = float(
                time.perf_counter() - upsample_started
            )
            salience_adaptive_diagnostics["reduced_policy"] = dict(reduced_policy_diag)
        if not baseline_frame_indices and not reduced_frame_indices:
            current_timings = {
                "pedestrian_render": 0.0,
                "depth_materialize": 0.0,
                "shadow_render_and_materialize": 0.0,
            }
        _write_visibility_culled_frame_artifacts(
            frames_dir=frames_dir,
            depth_dir=depth_dir,
            shadow_dir=shadow_dir,
            frame_indices=visibility_culled_frame_indices,
            image_shape=render_frame_shape,
        )
        rgba_started = time.perf_counter()
        pedestrian_rgba_diagnostics = _normalize_pedestrian_rgba_sequence_to_straight_alpha(
            run_dir=spec.run_dir,
            pedestrian_frames_dir=frames_dir,
        )
        current_timings["pedestrian_rgba_normalization"] = float(
            time.perf_counter() - rgba_started
        )
        log_info(
            "Pedestrian RGBA normalization: "
            f"normalized_frames={int(pedestrian_rgba_diagnostics.get('normalized_frame_count', 0))} "
            f"detected_premultiplied_frames={int(pedestrian_rgba_diagnostics.get('detected_premultiplied_frame_count', 0))} "
            f"diagnostics={pedestrian_rgba_diagnostics.get('diagnostics_path')}"
        )
        render_outlier_sanitization = _sanitize_corrupted_subject_rgba_frames(
            run_dir=spec.run_dir,
            pedestrian_frames_dir=frames_dir,
            grounding_diagnostics=grounding_diagnostics,
        )
        log_info(
            "Pedestrian render outlier sanitization: "
            f"sanitized_frames={int(render_outlier_sanitization.get('sanitized_frame_count', 0))} "
            f"diagnostics={render_outlier_sanitization.get('diagnostics_path')}"
        )
        raw_subject_exposure = getattr(render_spec, "raw_subject_exposure", None)
        wrap_subject_fill = getattr(
            getattr(spec, "lighting", None),
            "wrap_subject_fill",
            WrapSubjectFillSpec(),
        )
        if raw_subject_exposure is not None:
            exposure_started = time.perf_counter()
            exposure_diagnostics = _calibrate_raw_subject_exposure(
                run_dir=spec.run_dir,
                original_frames_dir=spec.run_dir / "standard" / "frames",
                pedestrian_frames_dir=frames_dir,
                settings=raw_subject_exposure,
                trim=float(getattr(wrap_subject_fill, "raw_exposure_trim", 1.0)),
            )
            current_timings["raw_subject_exposure_calibration"] = float(
                time.perf_counter() - exposure_started
            )
            log_info(
                "Raw subject exposure calibration: "
                f"gain={float(exposure_diagnostics.get('applied_gain', 1.0)):.3f} "
                f"eligible_frames={int(exposure_diagnostics.get('eligible_frame_count', 0))} "
                f"validation_passed={bool(exposure_diagnostics.get('validation_passed', True))} "
                f"diagnostics={exposure_diagnostics.get('diagnostics_path')}"
            )
        eevee_settings = getattr(bpy.context.scene, "eevee", None)
        if eevee_settings is None:
            eevee_settings = getattr(bpy.context.scene, "eevee_next", None)
        scene_lights = [
            obj for obj in getattr(bpy.data, "objects", ())
            if getattr(obj, "type", None) == "LIGHT"
        ]
        diagnostics_path = _write_render_backend_diagnostics(
            spec=spec,
            timings=current_timings,
            engine_name=str(getattr(bpy.context.scene.render, "engine", "unknown")),
            backend_settings={
                "material_policy": str(
                    getattr(getattr(spec, "render", None), "material_policy", "preserve_base_alpha_normal")
                ),
                "dynamic_light_binding": str(
                    getattr(getattr(spec, "render", None), "dynamic_light_binding", "copy_location_constraint")
                ),
                "salience_adaptive": salience_adaptive_diagnostics,
                "persistent_data": bool(
                    getattr(getattr(bpy.context.scene, "render", None), "use_persistent_data", False)
                ),
                "fast_png_compression": bool(
                    _scene_custom_value(bpy.context.scene, "_pemoin_fast_png_compression", True)
                ),
                "image_compression": getattr(
                    getattr(getattr(bpy.context.scene, "render", None), "image_settings", None),
                    "compression",
                    None,
                ),
                "eevee_raytracing_enabled": (
                    None
                    if getattr(eevee_settings, "use_raytracing", None) is None
                    else bool(getattr(eevee_settings, "use_raytracing"))
                ),
                "eevee_volumetric_shadows_enabled": (
                    None
                    if getattr(eevee_settings, "use_volumetric_shadows", None) is None
                    else bool(getattr(eevee_settings, "use_volumetric_shadows"))
                ),
                "eevee_soft_shadows_enabled": (
                    None
                    if getattr(eevee_settings, "use_soft_shadows", None) is None
                    else bool(getattr(eevee_settings, "use_soft_shadows"))
                ),
                "eevee_bloom_enabled": (
                    None
                    if getattr(eevee_settings, "use_bloom", None) is None
                    else bool(getattr(eevee_settings, "use_bloom"))
                ),
                "eevee_ssr_enabled": (
                    None
                    if getattr(eevee_settings, "use_ssr", None) is None
                    else bool(getattr(eevee_settings, "use_ssr"))
                ),
                "eevee_gtao_enabled": (
                    None
                    if getattr(eevee_settings, "use_gtao", None) is None
                    else bool(getattr(eevee_settings, "use_gtao"))
                ),
                "eevee_volumetric_lights_enabled": (
                    None
                    if getattr(eevee_settings, "use_volumetric_lights", None) is None
                    else bool(getattr(eevee_settings, "use_volumetric_lights"))
                ),
                "eevee_high_quality_normals_enabled": (
                    None
                    if getattr(eevee_settings, "use_high_quality_normals", None) is None
                    else bool(getattr(eevee_settings, "use_high_quality_normals"))
                ),
                "motion_blur_enabled": (
                    None
                    if getattr(getattr(bpy.context.scene, "render", None), "use_motion_blur", None) is None
                    else bool(getattr(getattr(bpy.context.scene, "render", None), "use_motion_blur"))
                ),
                "total_light_count": int(len(scene_lights)),
                "dynamic_subject_light_count": int(len(_dynamic_subject_light_objects())),
                "shadow_casting_light_count": int(
                    sum(
                        1
                        for obj in scene_lights
                        if bool(getattr(getattr(obj, "data", None), "use_shadow", False))
                    )
                ),
            },
            frame_plan={
                "total_frames": int(len(grounding_diagnostics)),
                "rendered_frames": int(len(visible_frame_indices)),
                "baseline_frame_count": int(len(baseline_frame_indices)),
                "low_salience_frame_count": int(len(reduced_frame_indices)),
                "visibility_culled_frames": int(len(visibility_culled_frame_indices)),
                "rendered_frame_indices": [int(idx) for idx in visible_frame_indices],
                "baseline_frame_indices": [int(idx) for idx in baseline_frame_indices],
                "low_salience_frame_indices": [int(idx) for idx in reduced_frame_indices],
                "visibility_culled_frame_indices": [
                    int(idx) for idx in visibility_culled_frame_indices
                ],
            },
            pedestrian_rgba=pedestrian_rgba_diagnostics,
            wrap_subject_fill={
                "global_strength_scale": float(
                    getattr(wrap_subject_fill, "global_strength_scale", 2.0)
                ),
                "wrap_key_role_scale": float(
                    getattr(wrap_subject_fill, "wrap_key_role_scale", 0.08)
                ),
                "counter_wrap_role_scale": float(
                    getattr(wrap_subject_fill, "counter_wrap_role_scale", 0.035)
                ),
                "sky_fill_role_scale": float(
                    getattr(wrap_subject_fill, "sky_fill_role_scale", 0.02)
                ),
                "counter_side_lift_bias": float(
                    getattr(wrap_subject_fill, "counter_side_lift_bias", 0.6)
                ),
                "sky_softness_bias": float(
                    getattr(wrap_subject_fill, "sky_softness_bias", 0.55)
                ),
                "direct_preservation_bias": float(
                    getattr(wrap_subject_fill, "direct_preservation_bias", 0.35)
                ),
                "raw_exposure_trim": float(
                    getattr(wrap_subject_fill, "raw_exposure_trim", 1.0)
                ),
                "render_outlier_sanitization": render_outlier_sanitization,
            },
        )
        log_info(f"Render backend diagnostics written: {diagnostics_path}")
        visibility_contract = _build_render_visibility_contract(
            frames_dir=frames_dir,
            grounding_diagnostics=grounding_diagnostics,
        )
        visibility_path = _write_render_visibility_contract(
            run_dir=spec.run_dir,
            frames=visibility_contract,
        )
        log_info(f"Render visibility contract written: {visibility_path}")
        _enforce_render_visibility_parity(
            run_dir=spec.run_dir,
            frames=visibility_contract,
        )
        return frames_dir


def _build_frame_index_map(frame_dir: Path) -> dict[int, Path]:
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    frame_map: dict[int, Path] = {}
    for frame_path in sorted(frame_dir.glob("*.png")):
        match = re.search(r"(\d+)$", frame_path.stem)
        if not match:
            continue
        frame_idx = int(match.group(1))
        frame_map[frame_idx] = frame_path

    if not frame_map:
        raise ValueError(f"No frame PNGs found in {frame_dir}")

    return frame_map


def _build_depth_index_map(frame_dir: Path) -> dict[int, Path]:
    if not frame_dir.exists():
        raise FileNotFoundError(f"Depth frame directory not found: {frame_dir}")
    frame_map: dict[int, Path] = {}
    for frame_path in sorted(frame_dir.glob("*.npz")):
        match = re.search(r"(\d+)$", frame_path.stem)
        if not match:
            continue
        frame_map[int(match.group(1))] = frame_path
    if not frame_map:
        raise ValueError(f"No depth NPZ frames found in {frame_dir}")
    return frame_map


def _read_depth_sequence_mode(depth_dir: Path) -> str:
    metadata_path = depth_dir / "metadata.json"
    if not metadata_path.exists():
        return "per_pixel"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return "per_pixel"
    mode = payload.get("mode")
    return "per_pixel" if mode is None else str(mode)


def _load_depth_npz_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "depth" not in data.files:
            raise ValueError(f"Depth file missing `depth` key: {path}")
        return np.asarray(data["depth"], dtype=np.float32)


def _load_rgba_image(path: Path) -> np.ndarray:
    """Load a persisted PNG into PEMOIN's top-origin RGBA array convention."""
    if not path.exists():
        raise FileNotFoundError(f"RGBA image not found: {path}")
    if Image is not None:
        return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    image = bpy.data.images.load(str(path))
    try:
        width, height = int(image.size[0]), int(image.size[1])
        rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape((height, width, 4))
    finally:
        bpy.data.images.remove(image)
    rgba = np.flipud(rgba)
    rgba = np.clip(np.rint(rgba * 255.0), 0.0, 255.0).astype(np.uint8)
    return rgba


def _unpremultiply_rgba_uint8(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected RGBA uint8 image, got {arr.shape}.")
    rgb = np.asarray(arr[:, :, :3], dtype=np.float32)
    alpha = np.asarray(arr[:, :, 3], dtype=np.float32) / 255.0
    straight = rgb.copy()
    valid = alpha > (1.0 / 255.0)
    straight[valid] = rgb[valid] / alpha[valid, None]
    out = arr.copy()
    out[:, :, :3] = np.clip(np.rint(straight), 0.0, 255.0).astype(np.uint8)
    return out


def _detect_premultiplied_rgba(rgba: np.ndarray) -> dict[str, float | int | bool]:
    arr = np.asarray(rgba, dtype=np.uint8)
    alpha = np.asarray(arr[:, :, 3], dtype=np.float32) / 255.0
    semi = (alpha > _OVERLAY_ALPHA_THRESHOLD) & (alpha < 0.95)
    semi_count = int(np.count_nonzero(semi))
    if semi_count == 0:
        return {
            "detected": False,
            "semi_transparent_pixel_count": 0,
            "premultiplied_consistency_ratio": 0.0,
            "median_unpremultiply_gain": 1.0,
        }
    rgb = np.asarray(arr[:, :, :3], dtype=np.float32)
    max_rgb = np.max(rgb, axis=2)
    consistent = max_rgb <= (alpha * 255.0 + 2.0)
    premult_ratio = float(np.mean(consistent[semi]))
    premul_luma = _rgb_luminance(rgb)
    straight_luma = _rgb_luminance(_unpremultiply_rgba_uint8(arr)[:, :, :3].astype(np.float32))
    denom = np.maximum(premul_luma[semi], 1.0)
    median_gain = float(np.median(straight_luma[semi] / denom))
    detected = bool(semi_count >= 32 and premult_ratio >= 0.95 and median_gain >= 1.15)
    return {
        "detected": detected,
        "semi_transparent_pixel_count": semi_count,
        "premultiplied_consistency_ratio": premult_ratio,
        "median_unpremultiply_gain": median_gain,
    }


def _write_rgba_image(path: Path, rgba: np.ndarray) -> None:
    """Persist a PEMOIN top-origin RGBA array to PNG."""
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected RGBA uint8 image, got {arr.shape}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(arr, mode="RGBA").save(path)
        return
    image = bpy.data.images.new(
        name=f"pemoin_write_{path.stem}",
        width=int(arr.shape[1]),
        height=int(arr.shape[0]),
        alpha=True,
    )
    try:
        flipped = np.flipud(arr).astype(np.float32) / 255.0
        image.pixels[:] = flipped.reshape(-1)
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def _write_pedestrian_rgba_diagnostics(
    *,
    run_dir: Path,
    diagnostics: Mapping[str, Any],
) -> Path:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    path = vis_dir / "pedestrian_rgba_diagnostics.json"
    path.write_text(json.dumps(dict(diagnostics), indent=2), encoding="utf-8")
    return path


def _normalize_pedestrian_rgba_sequence_to_straight_alpha(
    *,
    run_dir: Path,
    pedestrian_frames_dir: Path,
) -> dict[str, Any]:
    frame_map = _build_frame_index_map(pedestrian_frames_dir)
    diagnostics: dict[str, Any] = {
        "frame_count": int(len(frame_map)),
        "normalized_frame_count": 0,
        "detected_premultiplied_frame_count": 0,
        "detection_samples": [],
    }
    for frame_idx, path in sorted(frame_map.items()):
        rgba = _load_rgba_image(path)
        detection = _detect_premultiplied_rgba(rgba)
        diagnostics["detection_samples"].append(
            {
                "frame_index": int(frame_idx),
                **detection,
            }
        )
        if not bool(detection["detected"]):
            continue
        diagnostics["detected_premultiplied_frame_count"] = int(
            diagnostics["detected_premultiplied_frame_count"]
        ) + 1
        _write_rgba_image(path, _unpremultiply_rgba_uint8(rgba))
        diagnostics["normalized_frame_count"] = int(diagnostics["normalized_frame_count"]) + 1
    diagnostics["normalization_applied"] = bool(diagnostics["normalized_frame_count"] > 0)
    diagnostics_path = _write_pedestrian_rgba_diagnostics(
        run_dir=run_dir,
        diagnostics=diagnostics,
    )
    diagnostics["diagnostics_path"] = str(diagnostics_path)
    return diagnostics


def _sanitize_corrupted_subject_rgba_frames(
    *,
    run_dir: Path,
    pedestrian_frames_dir: Path,
    grounding_diagnostics: Sequence[GroundingDiagnostic],
) -> dict[str, Any]:
    frame_map = _build_frame_index_map(pedestrian_frames_dir)
    diag_by_frame = {
        int(item.frame_index): item for item in grounding_diagnostics
    }
    sanitized_frames: list[int] = []
    alpha_pixels_by_frame: dict[int, int] = {}
    opaque_coverages_by_frame: dict[int, float] = {}
    total_frames = 0
    for frame_idx, path in sorted(frame_map.items()):
        rgba = _load_rgba_image(path)
        total_frames += 1
        alpha = np.asarray(rgba[:, :, 3], dtype=np.uint8)
        alpha_pixels = int(np.count_nonzero(alpha > 0))
        alpha_pixels_by_frame[int(frame_idx)] = alpha_pixels
        opaque_coverages_by_frame[int(frame_idx)] = float(alpha_pixels / float(alpha.size))

    sorted_frames = sorted(frame_map.keys())
    for idx, frame_idx in enumerate(sorted_frames):
        diag = diag_by_frame.get(int(frame_idx))
        if diag is None or not bool(diag.visibility_culled):
            continue
        coverage = float(opaque_coverages_by_frame.get(int(frame_idx), 0.0))
        if coverage < 0.98:
            continue
        prev_frame = sorted_frames[idx - 1] if idx > 0 else None
        next_frame = sorted_frames[idx + 1] if idx + 1 < len(sorted_frames) else None
        prev_alpha_pixels = (
            None
            if prev_frame is None
            else alpha_pixels_by_frame.get(int(prev_frame))
        )
        next_alpha_pixels = (
            None
            if next_frame is None
            else alpha_pixels_by_frame.get(int(next_frame))
        )
        if not any(
            neighbor_alpha is not None and int(neighbor_alpha) == 0
            for neighbor_alpha in (prev_alpha_pixels, next_alpha_pixels)
        ):
            continue
        rgba = np.asarray(_load_rgba_image(frame_map[int(frame_idx)]), dtype=np.uint8).copy()
        rgba[:, :, 3] = 0
        _write_rgba_image(frame_map[int(frame_idx)], rgba)
        sanitized_frames.append(int(frame_idx))

    diagnostics = {
        "frame_count": int(total_frames),
        "sanitized_frame_count": int(len(sanitized_frames)),
        "sanitized_frames": [int(frame) for frame in sanitized_frames],
        "alpha_coverage_threshold": 0.98,
        "applied": bool(sanitized_frames),
    }
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = vis_dir / "pedestrian_render_outlier_sanitization.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    diagnostics["diagnostics_path"] = str(diagnostics_path)
    return diagnostics


def _rgb_luminance(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    return (
        0.2126 * arr[:, :, 0]
        + 0.7152 * arr[:, :, 1]
        + 0.0722 * arr[:, :, 2]
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _ring_mask_from_bbox(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    inner_px: int,
    outer_px: int,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    left, top, right, bottom = bbox
    outer = np.zeros((height, width), dtype=bool)
    inner = np.zeros((height, width), dtype=bool)
    outer[
        max(0, top - outer_px) : min(height, bottom + outer_px),
        max(0, left - outer_px) : min(width, right + outer_px),
    ] = True
    inner[
        max(0, top - inner_px) : min(height, bottom + inner_px),
        max(0, left - inner_px) : min(width, right + inner_px),
    ] = True
    return outer & ~inner


def _pedestrian_label_tokens(metadata: Mapping[str, Any] | None = None) -> set[str]:
    tokens = {"person", "pedestrian", "human"}
    try:
        resolved = resolve_semantic_role_labels(
            "mobile",
            metadata=dict(metadata or {}),
            required=False,
            source_name="raw subject exposure",
        )
    except Exception:
        resolved = ()
    for token in resolved:
        text = str(token).strip().lower()
        if any(marker in text for marker in ("person", "pedestrian", "human")):
            tokens.add(text)
    return tokens


def _scene_pedestrian_mask_for_frame(
    run_dir: Path,
    frame_idx: int,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    semantics_path = run_dir / "standard" / "semantics_2d" / f"{frame_idx:06d}.npz"
    if not semantics_path.exists():
        return None
    with np.load(semantics_path, allow_pickle=True) as data:
        label_ids = np.asarray(data["label_ids"], dtype=np.int32)
        metadata = _coerce_metadata(data["metadata"]) if "metadata" in data.files else {}
        segments_info = (
            np.asarray(data["segments_info"], dtype=object)
            if "segments_info" in data.files
            else np.asarray([], dtype=object)
        )
    if tuple(label_ids.shape) != tuple(target_shape):
        return None
    label_map_raw = metadata.get("class_id_to_label")
    if isinstance(label_map_raw, dict):
        label_map = {
            int(key): str(value).strip().lower()
            for key, value in label_map_raw.items()
            if str(value).strip()
        }
    else:
        label_map = _label_map_from_segments_info(segments_info)
    pedestrian_ids = {
        idx
        for idx, label in label_map.items()
        if label in _pedestrian_label_tokens(metadata)
        or any(marker in label for marker in ("person", "pedestrian", "human"))
    }
    if not pedestrian_ids:
        return None
    return np.isin(label_ids, list(pedestrian_ids))


def _robust_luminance_anchor(values: np.ndarray, *, percentile: float = 80.0) -> float | None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, float(percentile)))


def _gain_dispersion_ratio(gains: Sequence[float]) -> float | None:
    if not gains:
        return None
    arr = np.asarray(gains, dtype=np.float32)
    median = float(np.median(arr))
    if median <= 1e-6:
        return None
    mad = float(np.median(np.abs(arr - median)))
    return float(mad / median)


def _write_raw_subject_exposure_diagnostics(
    *,
    run_dir: Path,
    diagnostics: Mapping[str, Any],
) -> Path:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "raw_subject_exposure_diagnostics.json"
    json_path.write_text(json.dumps(dict(diagnostics), indent=2), encoding="utf-8")
    return json_path


def _align_rgba_to_shape(rgba: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(rgba)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected RGBA image with shape (H, W, 4), got {image.shape}.")
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    return _resize_array(
        image,
        (target_h, target_w),
        interpolation="nearest",
    )


def _calibrate_raw_subject_exposure(
    *,
    run_dir: Path,
    original_frames_dir: Path,
    pedestrian_frames_dir: Path,
    settings: Any,
    trim: float = 1.0,
) -> dict[str, Any]:
    enabled = bool(getattr(settings, "enabled", True))
    diagnostics: dict[str, Any] = {
        "enabled": enabled,
        "applied_gain": 1.0,
        "computed_gain": 1.0,
        "raw_exposure_trim": float(trim),
        "target_match_strength": float(getattr(settings, "target_match_strength", 0.75)),
        "max_gain": float(getattr(settings, "max_gain", 2.5)),
        "validation_tolerance": float(getattr(settings, "validation_tolerance", 0.18)),
        "frame_diagnostics": [],
        "eligible_frame_count": 0,
        "pedestrian_reference_frame_count": 0,
        "validation_passed": True,
        "reason": "disabled" if not enabled else "ok",
    }
    if not enabled:
        diagnostics_path = _write_raw_subject_exposure_diagnostics(
            run_dir=run_dir,
            diagnostics=diagnostics,
        )
        diagnostics["diagnostics_path"] = str(diagnostics_path)
        return diagnostics
    if not pedestrian_frames_dir.exists() or not original_frames_dir.exists():
        diagnostics["reason"] = "missing_input_frames"
        diagnostics["validation_passed"] = False
        diagnostics_path = _write_raw_subject_exposure_diagnostics(
            run_dir=run_dir,
            diagnostics=diagnostics,
        )
        diagnostics["diagnostics_path"] = str(diagnostics_path)
        return diagnostics

    frame_map = _build_frame_index_map(pedestrian_frames_dir)
    original_map = _build_frame_index_map(original_frames_dir)
    per_frame_gains: list[float] = []
    inner_px = 10
    outer_px = 40
    min_foreground_pixels = 64
    min_ring_pixels = 256
    match_strength = float(getattr(settings, "target_match_strength", 0.75))
    max_gain = float(getattr(settings, "max_gain", 2.5))
    validation_tolerance = float(getattr(settings, "validation_tolerance", 0.18))
    if abs(match_strength) <= 1e-6 and abs(float(trim) - 1.0) <= 1e-6:
        diagnostics["reason"] = "no_op_target_match"
        diagnostics["validation_passed"] = True
        diagnostics_path = _write_raw_subject_exposure_diagnostics(
            run_dir=run_dir,
            diagnostics=diagnostics,
        )
        diagnostics["diagnostics_path"] = str(diagnostics_path)
        return diagnostics
    anchor_percentile = 80.0
    conservative_brighten_cap = min(max_gain, 1.15)
    conservative_darken_cap = max(1.0 / max_gain, 0.85)
    max_gain_dispersion_ratio = 0.12
    pedestrian_reference_weight = float(
        getattr(settings, "pedestrian_reference_weight", 0.7)
    )
    min_pedestrian_reference_pixels = int(
        getattr(settings, "min_pedestrian_reference_pixels", 48)
    )
    original_rgba_cache: dict[int, np.ndarray] = {}
    pedestrian_reference_mask_cache: dict[int, np.ndarray | None] = {}

    def _original_frame_rgba(frame_idx: int, target_shape: tuple[int, int]) -> np.ndarray:
        cached = original_rgba_cache.get(int(frame_idx))
        if cached is None:
            cached = _load_rgba_image(original_map[frame_idx])
            original_rgba_cache[int(frame_idx)] = cached
        return _align_rgba_to_shape(cached, target_shape)

    def _pedestrian_reference_mask(frame_idx: int, target_shape: tuple[int, int]) -> np.ndarray | None:
        if int(frame_idx) not in pedestrian_reference_mask_cache:
            pedestrian_reference_mask_cache[int(frame_idx)] = _scene_pedestrian_mask_for_frame(
                run_dir=run_dir,
                frame_idx=frame_idx,
                target_shape=target_shape,
            )
        return pedestrian_reference_mask_cache[int(frame_idx)]

    for frame_idx in sorted(frame_map):
        if frame_idx not in original_map:
            continue
        ped_rgba = _load_rgba_image(frame_map[frame_idx])
        bg_rgba = _original_frame_rgba(frame_idx, tuple(ped_rgba.shape[:2]))
        alpha = np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0
        fg_mask = alpha > _OVERLAY_ALPHA_THRESHOLD
        if int(np.count_nonzero(fg_mask)) < min_foreground_pixels:
            continue
        bbox = _bbox_from_mask(fg_mask)
        if bbox is None:
            continue
        ring_mask = _ring_mask_from_bbox(
            fg_mask.shape,
            bbox,
            inner_px=inner_px,
            outer_px=outer_px,
        ) & ~fg_mask
        if int(np.count_nonzero(ring_mask)) < min_ring_pixels:
            continue
        fg_luma = _rgb_luminance(ped_rgba[:, :, :3])[fg_mask]
        ring_luma = _rgb_luminance(bg_rgba[:, :, :3])[ring_mask]
        fg_anchor = _robust_luminance_anchor(fg_luma, percentile=anchor_percentile)
        ring_anchor = _robust_luminance_anchor(ring_luma, percentile=anchor_percentile)
        fg_mean = float(np.mean(fg_luma))
        ring_mean = float(np.mean(ring_luma))
        if (
            fg_anchor is None
            or ring_anchor is None
            or not np.isfinite(fg_anchor)
            or not np.isfinite(ring_anchor)
            or fg_anchor <= 1e-3
        ):
            continue
        pedestrian_reference_mean = None
        pedestrian_reference_anchor = None
        pedestrian_reference_pixels = 0
        pedestrian_mask = _pedestrian_reference_mask(frame_idx, tuple(fg_mask.shape))
        if pedestrian_mask is not None:
            pedestrian_mask = np.asarray(pedestrian_mask, dtype=bool) & (~fg_mask)
            pedestrian_mask = pedestrian_mask & _ring_mask_from_bbox(
                fg_mask.shape,
                bbox,
                inner_px=0,
                outer_px=max(outer_px * 2, 48),
            )
            pedestrian_reference_pixels = int(np.count_nonzero(pedestrian_mask))
            if pedestrian_reference_pixels >= min_pedestrian_reference_pixels:
                pedestrian_reference_luma = _rgb_luminance(bg_rgba[:, :, :3])[pedestrian_mask]
                pedestrian_reference_mean = float(np.mean(pedestrian_reference_luma))
                pedestrian_reference_anchor = _robust_luminance_anchor(
                    pedestrian_reference_luma,
                    percentile=anchor_percentile,
                )
                diagnostics["pedestrian_reference_frame_count"] = int(
                    diagnostics["pedestrian_reference_frame_count"]
                ) + 1
        target_mean = float(ring_mean)
        target_anchor = float(ring_anchor)
        if pedestrian_reference_anchor is not None and np.isfinite(pedestrian_reference_anchor):
            target_mean = float(
                (1.0 - pedestrian_reference_weight) * ring_mean
                + pedestrian_reference_weight * float(pedestrian_reference_mean)
            )
            target_anchor = float(
                (1.0 - pedestrian_reference_weight) * ring_anchor
                + pedestrian_reference_weight * float(pedestrian_reference_anchor)
            )
        raw_gain = target_anchor / max(fg_anchor, 1e-3)
        blended_gain = 1.0 + (raw_gain - 1.0) * match_strength
        clamped_gain = float(
            np.clip(blended_gain, conservative_darken_cap, conservative_brighten_cap)
        )
        per_frame_gains.append(clamped_gain)
        diagnostics["frame_diagnostics"].append(
            {
                "frame_index": int(frame_idx),
                "foreground_pixel_count": int(np.count_nonzero(fg_mask)),
                "ring_pixel_count": int(np.count_nonzero(ring_mask)),
                "foreground_luminance_mean_before": fg_mean,
                "foreground_luminance_anchor_before": float(fg_anchor),
                "ring_luminance_mean": ring_mean,
                "ring_luminance_anchor": float(ring_anchor),
                "pedestrian_reference_luminance_mean": pedestrian_reference_mean,
                "pedestrian_reference_luminance_anchor": pedestrian_reference_anchor,
                "pedestrian_reference_pixels": int(pedestrian_reference_pixels),
                "target_luminance_mean": target_mean,
                "target_luminance_anchor": float(target_anchor),
                "suggested_gain": float(clamped_gain),
            }
        )

    if not per_frame_gains:
        diagnostics["reason"] = "no_eligible_frames"
        diagnostics["validation_passed"] = False
        diagnostics_path = _write_raw_subject_exposure_diagnostics(
            run_dir=run_dir,
            diagnostics=diagnostics,
        )
        diagnostics["diagnostics_path"] = str(diagnostics_path)
        return diagnostics

    clip_gain = float(np.median(np.asarray(per_frame_gains, dtype=np.float32)))
    clip_gain = float(np.clip(clip_gain, conservative_darken_cap, conservative_brighten_cap))
    diagnostics["eligible_frame_count"] = int(len(per_frame_gains))
    diagnostics["computed_gain"] = clip_gain
    diagnostics["median_gain_dispersion_ratio"] = _gain_dispersion_ratio(per_frame_gains)
    if (
        diagnostics["median_gain_dispersion_ratio"] is not None
        and float(diagnostics["median_gain_dispersion_ratio"]) > max_gain_dispersion_ratio
    ):
        diagnostics["reason"] = "gain_dispersion_above_tolerance"
        diagnostics["validation_passed"] = False
        diagnostics_path = _write_raw_subject_exposure_diagnostics(
            run_dir=run_dir,
            diagnostics=diagnostics,
        )
        diagnostics["diagnostics_path"] = str(diagnostics_path)
        return diagnostics
    clip_gain = float(
        np.clip(clip_gain * float(trim), conservative_darken_cap, conservative_brighten_cap)
    )
    diagnostics["applied_gain"] = clip_gain
    predicted_mean_residuals = [
        float(
            abs(float(entry["foreground_luminance_mean_before"]) * clip_gain - float(entry["target_luminance_mean"]))
            / max(float(entry["target_luminance_mean"]), 1e-3)
        )
        for entry in diagnostics["frame_diagnostics"]
        if float(entry.get("target_luminance_mean", 0.0)) > 1e-3
    ]
    predicted_anchor_residuals = [
        float(
            abs(
                float(entry["foreground_luminance_anchor_before"]) * clip_gain
                - float(entry["target_luminance_anchor"])
            )
            / max(float(entry["target_luminance_anchor"]), 1e-3)
        )
        for entry in diagnostics["frame_diagnostics"]
        if float(entry.get("target_luminance_anchor", 0.0)) > 1e-3
    ]
    predicted_mean_residual_ratio = (
        None
        if not predicted_mean_residuals
        else float(np.median(np.asarray(predicted_mean_residuals, dtype=np.float32)))
    )
    predicted_anchor_residual_ratio = (
        None
        if not predicted_anchor_residuals
        else float(np.median(np.asarray(predicted_anchor_residuals, dtype=np.float32)))
    )
    diagnostics["predicted_residual_luminance_gap_ratio"] = predicted_mean_residual_ratio
    diagnostics["predicted_residual_luminance_anchor_gap_ratio"] = predicted_anchor_residual_ratio
    if (
        (predicted_mean_residual_ratio is not None and predicted_mean_residual_ratio > validation_tolerance)
        or (
            predicted_anchor_residual_ratio is not None
            and predicted_anchor_residual_ratio > validation_tolerance
        )
    ):
        diagnostics["reason"] = "predicted_residual_gap_above_tolerance"
        diagnostics["validation_passed"] = False
        diagnostics_path = _write_raw_subject_exposure_diagnostics(
            run_dir=run_dir,
            diagnostics=diagnostics,
        )
        diagnostics["diagnostics_path"] = str(diagnostics_path)
        return diagnostics

    residuals: list[float] = []
    anchor_residuals: list[float] = []
    should_apply_gain = abs(float(clip_gain) - 1.0) > 1e-3
    for frame_idx in sorted(frame_map):
        path = frame_map[frame_idx]
        ped_rgba = _load_rgba_image(path)
        alpha = np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0
        rgb = np.asarray(ped_rgba[:, :, :3], dtype=np.float32)
        if should_apply_gain and np.any(alpha > 0.0):
            scaled = np.clip(rgb * clip_gain, 0.0, 255.0)
            rgb = np.where(alpha[:, :, None] > 0.0, scaled, rgb)
            ped_rgba = np.concatenate(
                [np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8), ped_rgba[:, :, 3:4]],
                axis=2,
            )
            _write_rgba_image(path, ped_rgba)
        if frame_idx not in original_map:
            continue
        fg_mask = alpha > _OVERLAY_ALPHA_THRESHOLD
        if int(np.count_nonzero(fg_mask)) < min_foreground_pixels:
            continue
        bbox = _bbox_from_mask(fg_mask)
        if bbox is None:
            continue
        ring_mask = _ring_mask_from_bbox(
            fg_mask.shape,
            bbox,
            inner_px=inner_px,
            outer_px=outer_px,
        ) & ~fg_mask
        if int(np.count_nonzero(ring_mask)) < min_ring_pixels:
            continue
        bg_rgba = _original_frame_rgba(frame_idx, tuple(ped_rgba.shape[:2]))
        fg_luma_after = _rgb_luminance(ped_rgba[:, :, :3])[fg_mask]
        ring_luma = _rgb_luminance(bg_rgba[:, :, :3])[ring_mask]
        fg_anchor_after = _robust_luminance_anchor(fg_luma_after, percentile=anchor_percentile)
        ring_anchor = _robust_luminance_anchor(ring_luma, percentile=anchor_percentile)
        fg_mean_after = float(np.mean(fg_luma_after))
        ring_mean = float(np.mean(ring_luma))
        target_mean = ring_mean
        target_anchor = None if ring_anchor is None else float(ring_anchor)
        pedestrian_reference_mean = None
        pedestrian_mask = _pedestrian_reference_mask(frame_idx, tuple(fg_mask.shape))
        if pedestrian_mask is not None:
            pedestrian_mask = np.asarray(pedestrian_mask, dtype=bool) & (~fg_mask)
            pedestrian_mask = pedestrian_mask & _ring_mask_from_bbox(
                fg_mask.shape,
                bbox,
                inner_px=0,
                outer_px=max(outer_px * 2, 48),
            )
            if int(np.count_nonzero(pedestrian_mask)) >= min_pedestrian_reference_pixels:
                pedestrian_reference_luma = _rgb_luminance(bg_rgba[:, :, :3])[pedestrian_mask]
                pedestrian_reference_mean = float(np.mean(pedestrian_reference_luma))
                pedestrian_reference_anchor = _robust_luminance_anchor(
                    pedestrian_reference_luma,
                    percentile=anchor_percentile,
                )
            else:
                pedestrian_reference_anchor = None
        else:
            pedestrian_reference_anchor = None
        if pedestrian_reference_mean is not None and np.isfinite(pedestrian_reference_mean):
            target_mean = float(
                (1.0 - pedestrian_reference_weight) * ring_mean
                + pedestrian_reference_weight * pedestrian_reference_mean
            )
        if pedestrian_reference_anchor is not None and target_anchor is not None:
            target_anchor = float(
                (1.0 - pedestrian_reference_weight) * target_anchor
                + pedestrian_reference_weight * pedestrian_reference_anchor
            )
        if target_mean > 1e-3:
            residuals.append(float(abs(fg_mean_after - target_mean) / target_mean))
        if fg_anchor_after is not None and target_anchor is not None and target_anchor > 1e-3:
            anchor_residuals.append(float(abs(fg_anchor_after - target_anchor) / target_anchor))
        for entry in diagnostics["frame_diagnostics"]:
            if int(entry["frame_index"]) != int(frame_idx):
                continue
            entry["foreground_luminance_mean_after"] = fg_mean_after
            entry["foreground_luminance_anchor_after"] = fg_anchor_after
            entry["residual_luminance_gap_ratio"] = (
                None if target_mean <= 1e-3 else float(abs(fg_mean_after - target_mean) / target_mean)
            )
            entry["residual_luminance_anchor_gap_ratio"] = (
                None
                if fg_anchor_after is None or target_anchor is None or target_anchor <= 1e-3
                else float(abs(fg_anchor_after - target_anchor) / target_anchor)
            )
            break

    residual_ratio = (
        None if not residuals else float(np.median(np.asarray(residuals, dtype=np.float32)))
    )
    anchor_residual_ratio = (
        None if not anchor_residuals else float(np.median(np.asarray(anchor_residuals, dtype=np.float32)))
    )
    diagnostics["median_residual_luminance_gap_ratio"] = residual_ratio
    diagnostics["median_residual_luminance_anchor_gap_ratio"] = anchor_residual_ratio
    diagnostics["validation_passed"] = bool(
        (residual_ratio is None or residual_ratio <= validation_tolerance)
        and (anchor_residual_ratio is None or anchor_residual_ratio <= validation_tolerance)
    )
    if not diagnostics["validation_passed"]:
        diagnostics["reason"] = "residual_gap_above_tolerance"
    diagnostics_path = _write_raw_subject_exposure_diagnostics(
        run_dir=run_dir,
        diagnostics=diagnostics,
    )
    diagnostics["diagnostics_path"] = str(diagnostics_path)
    return diagnostics


def _load_overlay_validation_context(
    run_dir: Path,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, tuple[np.ndarray, float]]]:
    trajectory_path = run_dir / "standard" / "trajectory" / "poses.npz"
    c2w, frame_indices = load_trajectory(trajectory_path)
    frame_to_c2w = {
        int(frame_idx): np.asarray(c2w[idx], dtype=np.float32)
        for idx, frame_idx in enumerate(frame_indices.tolist())
    }
    intrinsics_path = run_dir / "standard" / "intrinsics" / "intrinsics.npz"
    with np.load(intrinsics_path, allow_pickle=True) as data:
        intrinsics_k = np.asarray(data["matrix"], dtype=np.float32)
    if intrinsics_k.shape != (3, 3):
        raise ValueError(f"Invalid intrinsics matrix shape: {intrinsics_k.shape}")

    road_planes: dict[int, tuple[np.ndarray, float]] = {}
    road_dir = run_dir / "standard" / "road_plane"
    if road_dir.exists():
        for path in sorted(road_dir.glob("*.npz")):
            match = re.search(r"(\d+)$", path.stem)
            if not match:
                continue
            frame_idx = int(match.group(1))
            with np.load(path, allow_pickle=True) as data:
                normal = np.asarray(data["normal"], dtype=np.float32).reshape(3)
                offset = float(data["offset"])
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-6:
                continue
            road_planes[frame_idx] = (normal / norm, float(offset / norm))
    return intrinsics_k, frame_to_c2w, road_planes


def _estimate_actor_depth_map_from_rays(
    *,
    frame_idx: int,
    ped_rgba_top_origin: np.ndarray,
    actor_root: bpy.types.Object,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    max_distance_m: float = 500.0,
    max_hits: int = 8,
) -> np.ndarray:
    if frame_idx not in frame_to_c2w:
        raise ValueError(f"Missing camera pose for overlay frame {frame_idx}.")
    if ped_rgba_top_origin.ndim != 3 or ped_rgba_top_origin.shape[2] < 4:
        raise ValueError(f"Expected RGBA pedestrian frame, got {ped_rgba_top_origin.shape}.")
    height = int(ped_rgba_top_origin.shape[0])
    width = int(ped_rgba_top_origin.shape[1])
    alpha = np.asarray(ped_rgba_top_origin[:, :, 3], dtype=np.float32) / 255.0
    mask = alpha > _OVERLAY_ALPHA_THRESHOLD
    depth = np.zeros((height, width), dtype=np.float32)
    if not np.any(mask):
        return depth

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    actor_objects = get_object_hierarchy(actor_root)
    actor_object_names = {str(obj.name) for obj in actor_objects}
    actor_object_ptrs = {
        int(obj.as_pointer())
        for obj in actor_objects
        if hasattr(obj, "as_pointer")
    }
    actor_eval_ptrs = {
        int(obj.evaluated_get(depsgraph).as_pointer())
        for obj in actor_objects
        if hasattr(obj, "evaluated_get") and hasattr(obj.evaluated_get(depsgraph), "as_pointer")
    }
    c2w = np.asarray(frame_to_c2w[int(frame_idx)], dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    camera_origin = np.asarray(c2w[:3, 3], dtype=np.float32)
    rotation = np.asarray(c2w[:3, :3], dtype=np.float32)
    ys, xs = np.where(mask)

    for v, u in zip(ys.tolist(), xs.tolist()):
        dir_cam = backproject_uv_depth_to_camera(
            np.asarray([[float(u), float(v)]], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            intrinsics_k,
            camera_convention="blender",
        )[0]
        dir_cam_norm = float(np.linalg.norm(dir_cam))
        if dir_cam_norm <= 1e-8:
            continue
        dir_cam /= dir_cam_norm
        dir_world = rotation @ dir_cam
        dir_world_norm = float(np.linalg.norm(dir_world))
        if dir_world_norm <= 1e-8:
            continue
        dir_world /= dir_world_norm
        origin = np.asarray(camera_origin, dtype=np.float32)
        remaining = float(max_distance_m)
        for _ in range(max_hits):
            hit, loc, _normal, _face, obj, _matrix = scene.ray_cast(
                depsgraph,
                Vector(origin.tolist()),
                Vector(dir_world.tolist()),
                distance=float(remaining),
            )
            if not hit:
                break
            loc_np = np.asarray(loc, dtype=np.float32)
            traveled = float(np.linalg.norm(loc_np - origin))
            obj_name = None if obj is None else str(getattr(obj, "name", ""))
            obj_ptr = (
                None
                if obj is None or not hasattr(obj, "as_pointer")
                else int(obj.as_pointer())
            )
            if (
                obj in actor_objects
                or (obj_name in actor_object_names)
                or (obj_ptr in actor_object_ptrs)
                or (obj_ptr in actor_eval_ptrs)
            ):
                hom = np.asarray([loc_np[0], loc_np[1], loc_np[2], 1.0], dtype=np.float32)
                cam = w2c @ hom
                z_depth = float(abs(cam[2]))
                if np.isfinite(z_depth) and z_depth > 0.0:
                    depth[int(v), int(u)] = z_depth
                break
            eps = 1e-4
            origin = loc_np + dir_world * eps
            remaining -= max(traveled + eps, eps)
            if remaining <= 0.0:
                break
    return depth


def _materialize_depth_npz_from_raycast_sequence(
    *,
    pedestrian_frames_dir: Path,
    depth_output_dir: Path,
    actor_root: bpy.types.Object,
    intrinsics_k: np.ndarray,
    run_dir: Path,
) -> None:
    intrinsics_runtime, frame_to_c2w, _ = _load_overlay_validation_context(run_dir)
    if intrinsics_runtime.shape == (3, 3):
        intrinsics_k = np.asarray(intrinsics_runtime, dtype=np.float32)
    frame_map = _build_frame_index_map(pedestrian_frames_dir)
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame_path in sorted(frame_map.items()):
        ped_rgba_top_origin = _load_rgba_image(frame_path)
        depth = _estimate_actor_depth_map_from_rays(
            frame_idx=int(frame_idx),
            ped_rgba_top_origin=ped_rgba_top_origin,
            actor_root=actor_root,
            intrinsics_k=np.asarray(intrinsics_k, dtype=np.float32),
            frame_to_c2w=frame_to_c2w,
        )
        np.savez_compressed(
            depth_output_dir / f"{int(frame_idx):06d}.npz",
            depth=np.asarray(depth, dtype=np.float32),
        )


def _find_lowest_visible_pixel(
    alpha: np.ndarray, *, threshold: float = _OVERLAY_ALPHA_THRESHOLD
) -> tuple[int, int, int] | None:
    if alpha.ndim != 2:
        raise ValueError(f"Expected 2D alpha image, got {alpha.shape}.")
    mask = alpha > float(threshold)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    v_max = int(ys.max())
    row_xs = xs[ys == v_max]
    if row_xs.size == 0:
        return None
    u_med = int(np.rint(np.median(row_xs)))
    return u_med, v_max, int(row_xs.size)


def _select_overlay_plane_candidates(
    frame_idx: int,
    road_planes: dict[int, tuple[np.ndarray, float]],
) -> tuple[tuple[int, np.ndarray, float] | None, list[tuple[int, np.ndarray, float]]]:
    if not road_planes:
        return None, []
    if frame_idx in road_planes:
        normal, offset = road_planes[frame_idx]
        return (frame_idx, normal, offset), [(frame_idx, normal, offset)]
    nearest_idx = min(road_planes.keys(), key=lambda idx: abs(int(idx) - int(frame_idx)))
    normal, offset = road_planes[nearest_idx]
    return (nearest_idx, normal, offset), [(nearest_idx, normal, offset)]


def _project_overlay_feet(
    *,
    frame_idx: int,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    left_foot: np.ndarray | None,
    right_foot: np.ndarray | None,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray | None, bool, np.ndarray | None, bool]:
    if frame_idx not in frame_to_c2w:
        raise ValueError(f"Missing trajectory pose for overlay validation frame {frame_idx}.")
    c2w = frame_to_c2w[frame_idx]
    points: list[np.ndarray] = []
    if left_foot is not None:
        points.append(np.asarray(left_foot, dtype=np.float32))
    if right_foot is not None:
        points.append(np.asarray(right_foot, dtype=np.float32))
    if not points:
        return None, False, None, False
    uv, valid = project_world_to_image(
        np.asarray(points, dtype=np.float32),
        intrinsics_k,
        camera_to_world_matrix=c2w,
        camera_convention="blender",
        image_shape=image_shape,
    )
    left_uv = None
    right_uv = None
    left_valid = False
    right_valid = False
    cursor = 0
    if left_foot is not None:
        left_uv = np.asarray(uv[cursor], dtype=np.float32)
        left_valid = bool(valid[cursor])
        cursor += 1
    if right_foot is not None:
        right_uv = np.asarray(uv[cursor], dtype=np.float32)
        right_valid = bool(valid[cursor])
    return left_uv, left_valid, right_uv, right_valid


def _project_overlay_support_point(
    *,
    frame_idx: int,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    support_point: np.ndarray | None,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray | None, bool, float | None]:
    if support_point is None:
        return None, False, None
    if frame_idx not in frame_to_c2w:
        raise ValueError(f"Missing trajectory pose for overlay validation frame {frame_idx}.")
    c2w = frame_to_c2w[frame_idx]
    uv, valid = project_world_to_image(
        np.asarray([support_point], dtype=np.float32),
        intrinsics_k,
        camera_to_world_matrix=c2w,
        camera_convention="blender",
        image_shape=image_shape,
    )
    support_world = np.asarray(support_point, dtype=np.float32).reshape(3)
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    cam_h = w2c @ np.asarray([support_world[0], support_world[1], support_world[2], 1.0], dtype=np.float32)
    support_depth = float(abs(cam_h[2]))
    return np.asarray(uv[0], dtype=np.float32), bool(valid[0]), support_depth


def _classify_projected_actor_visibility(
    *,
    frame_idx: int,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    actor_root: bpy.types.Object,
    depsgraph: Any,
    image_shape: tuple[int, int],
) -> tuple[bool, str | None]:
    if frame_idx not in frame_to_c2w:
        raise ValueError(f"Missing trajectory pose for grounding frame {frame_idx}.")
    points = _projected_actor_extent_points_world(actor_root=actor_root, depsgraph=depsgraph)
    _, valid = project_world_to_image(
        np.asarray(points, dtype=np.float32),
        intrinsics_k,
        camera_to_world_matrix=frame_to_c2w[frame_idx],
        camera_convention="blender",
        image_shape=image_shape,
    )
    if bool(np.any(np.asarray(valid, dtype=bool))):
        return True, "projected_visible"
    return False, "actor_off_camera"


def _matrix4x4_to_numpy(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got shape {arr.shape}.")
    return arr


def _transform_local_points_to_world(
    points_local: np.ndarray,
    matrix_world: Any,
) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float32).reshape(-1, 3)
    world_from_local = _matrix4x4_to_numpy(matrix_world)
    hom = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = hom @ world_from_local.T
    return np.asarray(transformed[:, :3], dtype=np.float32)


def _projected_actor_extent_points_world(
    *,
    actor_root: bpy.types.Object,
    depsgraph: Any,
) -> np.ndarray:
    renderable_meshes: list[Any] = []
    for obj in _iter_descendants(actor_root):
        if getattr(obj, "type", None) != "MESH":
            continue
        if bool(getattr(obj, "hide_render", False)):
            continue
        renderable_meshes.append(obj)
    if not renderable_meshes:
        raise ValueError(
            f"Actor root '{actor_root.name}' has no renderable mesh descendants; cannot classify visibility."
        )

    points_world: list[np.ndarray] = []
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for mesh_obj in renderable_meshes:
        mesh_eval = mesh_obj.evaluated_get(depsgraph)
        bound_box = getattr(mesh_eval, "bound_box", None)
        if bound_box is None:
            continue
        local_corners = np.asarray(bound_box, dtype=np.float32).reshape(-1, 3)
        if local_corners.shape[0] == 0:
            continue
        world_corners = _transform_local_points_to_world(
            local_corners,
            mesh_eval.matrix_world,
        )
        points_world.extend(np.asarray(world_corners, dtype=np.float32))
        mins.append(np.min(world_corners, axis=0))
        maxs.append(np.max(world_corners, axis=0))
    if not points_world:
        raise ValueError(
            f"Actor root '{actor_root.name}' has renderable meshes but no usable evaluated bounds."
        )

    scene_min = np.min(np.stack(mins, axis=0), axis=0)
    scene_max = np.max(np.stack(maxs, axis=0), axis=0)
    scene_center = 0.5 * (scene_min + scene_max)
    extent_corners = np.asarray(
        [
            [scene_min[0], scene_min[1], scene_min[2]],
            [scene_min[0], scene_min[1], scene_max[2]],
            [scene_min[0], scene_max[1], scene_min[2]],
            [scene_min[0], scene_max[1], scene_max[2]],
            [scene_max[0], scene_min[1], scene_min[2]],
            [scene_max[0], scene_min[1], scene_max[2]],
            [scene_max[0], scene_max[1], scene_min[2]],
            [scene_max[0], scene_max[1], scene_max[2]],
            [scene_center[0], scene_center[1], scene_center[2]],
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [np.asarray(points_world, dtype=np.float32), extent_corners],
        axis=0,
    )


def _build_support_point_lookup(
    diagnostics: Sequence[GroundingDiagnostic],
) -> dict[int, np.ndarray | None]:
    support_point_by_frame: dict[int, np.ndarray | None] = {}
    for diag in diagnostics:
        frame_idx = int(diag.frame_index)
        if frame_idx in support_point_by_frame:
            raise ValueError(f"Duplicate grounding diagnostic for frame {frame_idx}.")
        support_point_by_frame[frame_idx] = (
            None
            if diag.support_point_after is None
            else np.asarray(diag.support_point_after, dtype=np.float32).reshape(3)
        )
    return support_point_by_frame


def _build_grounding_diagnostic_lookup(
    diagnostics: Sequence[GroundingDiagnostic],
) -> dict[int, GroundingDiagnostic]:
    lookup: dict[int, GroundingDiagnostic] = {}
    for diag in diagnostics:
        frame_idx = int(diag.frame_index)
        if frame_idx in lookup:
            raise ValueError(f"Duplicate grounding diagnostic for frame {frame_idx}.")
        lookup[frame_idx] = diag
    return lookup


def _draw_support_marker_rgba(
    image_rgba: np.ndarray,
    u_top: int,
    v_top: int,
    *,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> None:
    """Draw a marker at a top-origin overlay pixel coordinate."""
    if image_rgba.ndim != 3 or image_rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA image buffer, got {image_rgba.shape}.")
    height, width, _ = image_rgba.shape
    u_center = int(u_top)
    v_center = int(v_top)
    for dv in (-1, 0, 1):
        yy = v_center + dv
        if yy < 0 or yy >= height:
            continue
        for du in (-1, 0, 1):
            xx = u_center + du
            if xx < 0 or xx >= width:
                continue
            image_rgba[yy, xx, 0] = float(color[0])
            image_rgba[yy, xx, 1] = float(color[1])
            image_rgba[yy, xx, 2] = float(color[2])
            image_rgba[yy, xx, 3] = 1.0


def _draw_support_patch_outline_rgba(
    image_rgba: np.ndarray,
    u_top: int,
    v_top: int,
    *,
    radius_px: int,
) -> None:
    if radius_px <= 0:
        return
    for uu in range(int(u_top) - radius_px, int(u_top) + radius_px + 1):
        for vv in (int(v_top) - radius_px, int(v_top) + radius_px):
            _draw_support_marker_rgba(image_rgba, uu, vv, color=(0.0, 1.0, 0.0))
    for vv in range(int(v_top) - radius_px, int(v_top) + radius_px + 1):
        for uu in (int(u_top) - radius_px, int(u_top) + radius_px):
            _draw_support_marker_rgba(image_rgba, uu, vv, color=(0.0, 1.0, 0.0))


def _compute_support_road_context(
    *,
    ped_rgba: np.ndarray,
    run_dir: Path,
    frame_idx: int,
    support_point_uv: np.ndarray | None,
    support_point_visible: bool,
    support_point_depth_m: float | None,
    road_labels: Sequence[str],
) -> tuple[float | None, float | None, int, bool, bool, float | None, str]:
    if support_point_uv is None or not support_point_visible:
        return None, None, 0, False, False, None, "no_support_point"
    semantics_path = run_dir / "standard" / "semantics_2d" / f"{frame_idx:06d}.npz"
    depth_path = run_dir / "standard" / "depth" / f"{frame_idx:06d}.npz"
    if not semantics_path.exists():
        return None, None, 0, False, False, None, "no_semantics"
    with np.load(semantics_path, allow_pickle=True) as data:
        label_ids = np.asarray(data["label_ids"], dtype=np.int32)
        metadata = _coerce_metadata(data["metadata"]) if "metadata" in data.files else {}
        segments_info = (
            np.asarray(data["segments_info"], dtype=object)
            if "segments_info" in data.files
            else np.asarray([], dtype=object)
        )
    depth = None
    if depth_path.exists():
        with np.load(depth_path, allow_pickle=True) as data:
            depth = np.asarray(data["depth"], dtype=np.float32)
    label_map_raw = metadata.get("class_id_to_label")
    label_map: dict[int, str]
    if isinstance(label_map_raw, dict):
        label_map = {
            int(key): str(value)
            for key, value in label_map_raw.items()
            if str(value).strip()
        }
    else:
        label_map = _label_map_from_segments_info(segments_info)
    road_ids = _resolve_label_ids_from_label_map(label_map, labels=road_labels)
    if not road_ids:
        return None, None, 0, False, False, None, "no_road_labels"
    u = int(np.rint(float(support_point_uv[0])))
    v = int(np.rint(float(support_point_uv[1])))
    h, w = label_ids.shape
    x0 = max(0, u - _OVERLAY_SUPPORT_PATCH_RADIUS_PX)
    x1 = min(w, u + _OVERLAY_SUPPORT_PATCH_RADIUS_PX + 1)
    y0 = max(0, v - _OVERLAY_SUPPORT_PATCH_RADIUS_PX)
    y1 = min(h, v + _OVERLAY_SUPPORT_PATCH_RADIUS_PX + 1)
    scene_depth = None
    occluded = False
    if depth is not None and 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
        sampled_depth = float(depth[v, u])
        if np.isfinite(sampled_depth) and sampled_depth > 0.0:
            scene_depth = sampled_depth
            if support_point_depth_m is not None and sampled_depth + 0.5 < float(support_point_depth_m):
                occluded = True
    alpha = np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0
    alpha_mask = alpha > _OVERLAY_ALPHA_THRESHOLD
    search_mode = "direct_patch"
    patch = label_ids[y0:y1, x0:x1]
    if occluded:
        search_mode = "occluded_skip"
    elif patch.size:
        patch_alpha = alpha_mask[y0:y1, x0:x1]
        if patch_alpha.size == patch.size:
            patch = patch[~patch_alpha]
        if patch.size == 0:
            # Search slightly below the support point for visible road context.
            y0_down = max(0, v)
            y1_down = min(h, v + (2 * _OVERLAY_SUPPORT_PATCH_RADIUS_PX) + 1)
            downward = label_ids[y0_down:y1_down, x0:x1]
            downward_alpha = alpha_mask[y0_down:y1_down, x0:x1]
            if downward.size and downward_alpha.size == downward.size:
                downward = downward[~downward_alpha]
            patch = downward
            search_mode = "downward_search"
    if patch.size == 0 or occluded:
        return None, None, int(patch.size), True, occluded, scene_depth, search_mode
    road_fraction = float(np.mean(np.isin(patch, list(road_ids))))
    return (
        road_fraction,
        float(1.0 - road_fraction),
        int(patch.size),
        True,
        occluded,
        scene_depth,
        search_mode,
    )


def _load_overlay_ground_mask(
    *,
    run_dir: Path,
    frame_idx: int,
    image_shape: tuple[int, int],
    ground_labels: Sequence[str],
    required: bool = False,
) -> np.ndarray | None:
    semantics_path = run_dir / "standard" / "semantics_2d" / f"{frame_idx:06d}.npz"
    if not semantics_path.exists():
        if required:
            raise ValueError(
                f"Missing semantics for overlay frame {frame_idx}: {semantics_path}."
            )
        return None
    with np.load(semantics_path, allow_pickle=True) as data:
        if "label_ids" not in data.files:
            if required:
                raise ValueError(
                    f"Semantics file for overlay frame {frame_idx} is missing label_ids: {semantics_path}."
                )
            return None
        label_ids = np.asarray(data["label_ids"], dtype=np.int32)
        metadata = _coerce_metadata(data["metadata"]) if "metadata" in data.files else {}
        segments_info = (
            np.asarray(data["segments_info"], dtype=object)
            if "segments_info" in data.files
            else np.asarray([], dtype=object)
        )
    if label_ids.ndim != 2:
        raise ValueError(
            f"Semantics label_ids for overlay frame {frame_idx} must be 2D, got {label_ids.shape}."
        )
    label_map_raw = metadata.get("class_id_to_label")
    label_map: dict[int, str]
    if isinstance(label_map_raw, dict):
        label_map = {
            int(key): str(value)
            for key, value in label_map_raw.items()
            if str(value).strip()
        }
    else:
        label_map = _label_map_from_segments_info(segments_info)
    ground_ids = _resolve_label_ids_from_label_map(label_map, labels=ground_labels)
    if not ground_ids:
        if required:
            raise ValueError(
                "Overlay frame "
                f"{frame_idx} resolved no traversable-ground labels from semantics metadata."
            )
        return None
    ground_mask = np.isin(label_ids, np.asarray(list(ground_ids), dtype=np.int32))
    target_shape = tuple(int(v) for v in image_shape)
    if ground_mask.shape == target_shape:
        return ground_mask
    resized = _resize_array(
        ground_mask.astype(np.uint8),
        target_shape,
        interpolation="nearest",
    )
    log_info(
        "Resized traversable-ground overlay mask for frame "
        f"{frame_idx}: semantics_shape={ground_mask.shape} overlay_shape={target_shape}."
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def _render_overlay_support_local_grid(
    *,
    run_dir: Path,
    frame_idx: int,
    out_pixels: np.ndarray,
    grounding_diag: GroundingDiagnostic,
    intrinsics_k: np.ndarray,
    frame_to_c2w: dict[int, np.ndarray],
    road_labels: Sequence[str],
    pedestrian_visible_mask_top: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render the local support grid on a top-origin overlay RGBA image."""
    output = np.asarray(out_pixels, dtype=np.float32).copy()
    if (
        grounding_diag.support_point_after is None
        or grounding_diag.chosen_plane_normal is None
        or grounding_diag.chosen_plane_offset is None
    ):
        return output
    if int(frame_idx) not in frame_to_c2w:
        raise ValueError(f"Missing camera pose for overlay local-grid frame {frame_idx}.")
    height = int(output.shape[0])
    width = int(output.shape[1])
    top_origin_rgb = np.clip(output[:, :, :3], 0.0, 1.0)
    base_bgr = np.ascontiguousarray(
        np.rint(top_origin_rgb[:, :, ::-1] * 255.0).astype(np.uint8)
    )
    grid_layer = render_plane_grid_layer(
        base_bgr.shape,
        intrinsics_k,
        normal=np.asarray(grounding_diag.chosen_plane_normal, dtype=np.float32),
        offset=float(grounding_diag.chosen_plane_offset),
        camera_to_world=np.asarray(frame_to_c2w[int(frame_idx)], dtype=np.float32),
        anchor_point_world=np.asarray(grounding_diag.support_point_after, dtype=np.float32),
        grid_spacing_m=_OVERLAY_SUPPORT_LOCAL_GRID_SPACING_M,
        extent_m=_OVERLAY_SUPPORT_LOCAL_GRID_EXTENT_M,
        line_color_bgr=_OVERLAY_SUPPORT_LOCAL_GRID_LINE_COLOR_BGR,
        line_thickness=_OVERLAY_SUPPORT_LOCAL_GRID_LINE_THICKNESS,
    )
    effective_mask: np.ndarray | None
    if road_mask is None:
        effective_mask = None
    else:
        effective_mask = np.asarray(road_mask, dtype=bool)
    if pedestrian_visible_mask_top is not None:
        ped_mask = np.asarray(pedestrian_visible_mask_top, dtype=bool)
        if ped_mask.shape != (height, width):
            raise ValueError(
                "pedestrian_visible_mask_top shape mismatch: "
                f"got {ped_mask.shape}, expected {(height, width)}."
            )
        if effective_mask is None:
            effective_mask = ~ped_mask
        else:
            effective_mask = effective_mask & (~ped_mask)
    composited_bgr = composite_grid_with_mask(base_bgr, grid_layer, effective_mask)
    composited_rgb = np.asarray(composited_bgr[:, :, ::-1], dtype=np.float32) / 255.0
    output[:, :, :3] = composited_rgb
    output[:, :, 3] = 1.0
    return output


def _make_overlay_validation_diagnostic(
    *,
    run_dir: Path,
    frame_idx: int,
    ped_rgba: np.ndarray,
    overlay_shape: tuple[int, int] | None = None,
    left_uv: np.ndarray | None,
    left_valid: bool,
    right_uv: np.ndarray | None,
    right_valid: bool,
    selected_support_foot: str,
    support_mode: str,
    support_point_uv: np.ndarray | None,
    support_point_visible: bool,
    support_point_depth_m: float | None,
    road_labels: Sequence[str],
) -> OverlayValidationDiagnostic:
    height = int(ped_rgba.shape[0])
    width = int(ped_rgba.shape[1])
    alpha = np.asarray(ped_rgba[:, :, 3], dtype=np.float32) / 255.0
    alpha_mask = alpha > _OVERLAY_ALPHA_THRESHOLD
    lowest = _find_lowest_visible_pixel(alpha)
    touches_left = bool(width > 0 and np.any(alpha_mask[:, 0]))
    touches_right = bool(width > 0 and np.any(alpha_mask[:, width - 1]))
    left_distance = (
        None
        if left_uv is None or not left_valid or support_point_uv is None or not support_point_visible
        else float(np.linalg.norm(np.asarray(support_point_uv, dtype=np.float32) - np.asarray(left_uv, dtype=np.float32)))
    )
    right_distance = (
        None
        if right_uv is None or not right_valid or support_point_uv is None or not support_point_visible
        else float(np.linalg.norm(np.asarray(support_point_uv, dtype=np.float32) - np.asarray(right_uv, dtype=np.float32)))
    )
    contact_mode = "none"
    contact_distance = None
    trajectory_support_mode = (
        str(selected_support_foot).strip().lower() in {"path", "none"}
        or str(support_mode).strip().lower() == "trajectory_path"
    )
    if trajectory_support_mode:
        contact_mode = "trajectory_path"
    elif selected_support_foot == "left" and left_distance is not None:
        contact_mode = "left"
        contact_distance = left_distance
    elif selected_support_foot == "right" and right_distance is not None:
        contact_mode = "right"
        contact_distance = right_distance
    elif selected_support_foot == "both":
        candidates = [v for v in [left_distance, right_distance] if v is not None]
        if candidates:
            contact_mode = "both_min"
            contact_distance = float(min(candidates))
    if (not trajectory_support_mode) and contact_distance is None:
        fallback_candidates = [v for v in [left_distance, right_distance] if v is not None]
        if fallback_candidates:
            contact_mode = "fallback_other_visible"
            contact_distance = float(min(fallback_candidates))
    selected_uv = None
    selected_visible = False
    if trajectory_support_mode:
        selected_uv = None
        selected_visible = False
    elif selected_support_foot == "left":
        selected_uv = left_uv
        selected_visible = bool(left_valid)
    elif selected_support_foot == "right":
        selected_uv = right_uv
        selected_visible = bool(right_valid)
    elif selected_support_foot == "both":
        if left_distance is not None and right_distance is not None:
            selected_uv = left_uv if float(left_distance) <= float(right_distance) else right_uv
            selected_visible = True
        elif left_valid and left_uv is not None:
            selected_uv = left_uv
            selected_visible = True
        elif right_valid and right_uv is not None:
            selected_uv = right_uv
            selected_visible = True
    if selected_uv is None:
        if left_distance is not None and (right_distance is None or float(left_distance) <= float(right_distance)):
            selected_uv = left_uv
            selected_visible = bool(left_valid and left_uv is not None)
        elif right_distance is not None:
            selected_uv = right_uv
            selected_visible = bool(right_valid and right_uv is not None)
    selected_expected_visible = False
    if trajectory_support_mode:
        selected_expected_visible = False
    elif selected_support_foot == "left":
        selected_expected_visible = bool(left_valid)
    elif selected_support_foot == "right":
        selected_expected_visible = bool(right_valid)
    elif selected_support_foot == "both":
        selected_expected_visible = bool(left_valid or right_valid)
    (
        road_fraction,
        nonroad_fraction,
        patch_size,
        road_available,
        support_occluded,
        scene_depth,
        road_context_mode,
    ) = _compute_support_road_context(
        ped_rgba=ped_rgba,
        run_dir=run_dir,
        frame_idx=frame_idx,
        support_point_uv=support_point_uv,
        support_point_visible=support_point_visible,
        support_point_depth_m=support_point_depth_m,
        road_labels=road_labels,
    )
    if lowest is None:
        return OverlayValidationDiagnostic(
            frame_index=int(frame_idx),
            has_visible_pedestrian=False,
            internal_render_shape=tuple(int(v) for v in ped_rgba.shape[:2]),
            overlay_shape=(
                None if overlay_shape is None else tuple(int(v) for v in overlay_shape)
            ),
            lowest_alpha_u=None,
            lowest_alpha_v=None,
            lowest_alpha_row_coverage_px=0,
            touches_image_bottom=False,
            left_foot_projected_uv=None,
            right_foot_projected_uv=None,
            left_foot_visible_expected=False,
            right_foot_visible_expected=False,
            support_point_projected_uv=(
                None
                if support_point_uv is None
                else np.asarray(support_point_uv, dtype=np.float32)
            ),
            support_point_projected_visible=bool(support_point_visible),
            support_point_depth_m=(
                None if support_point_depth_m is None else float(support_point_depth_m)
            ),
            scene_depth_at_support_px_m=(
                None if scene_depth is None else float(scene_depth)
            ),
            support_point_occluded_by_scene=bool(support_occluded),
            selected_foot_projected_uv=(
                None if selected_uv is None else np.asarray(selected_uv, dtype=np.float32)
            ),
            selected_foot_projected_visible=bool(selected_visible),
            support_to_left_foot_px=left_distance,
            support_to_right_foot_px=right_distance,
            support_to_contact_foot_px=contact_distance,
            contact_foot_comparison_mode=str(contact_mode),
            support_to_silhouette_bottom_px=None,
            support_to_selected_foot_px=None,
            support_patch_road_fraction=road_fraction,
            support_patch_nonroad_fraction=nonroad_fraction,
            support_patch_size_px=int(patch_size),
            road_region_validation_available=bool(road_available),
            road_context_search_mode=str(road_context_mode),
            validation_passed=True,
            failure_reason=None,
            support_mode=str(support_mode),
            selected_support_foot=str(selected_support_foot),
            warning_flags=tuple(),
            touches_image_left=bool(touches_left),
            touches_image_right=bool(touches_right),
            selected_support_foot_expected_visible=bool(selected_expected_visible),
            contact_validation_state="unverifiable",
            contact_validation_trusted=False,
            abort_relevant=False,
        )

    u_foot, v_foot, row_coverage = lowest
    warning_flags: list[str] = []
    if v_foot >= height - 1:
        warning_flags.append("foot_clipped_by_frame_bottom")
    if touches_left:
        warning_flags.append("pedestrian_touches_image_left")
    if touches_right:
        warning_flags.append("pedestrian_touches_image_right")
    bottom = np.asarray([float(u_foot), float(v_foot)], dtype=np.float32)
    support_to_bottom = (
        None
        if support_point_uv is None or not support_point_visible
        else float(np.linalg.norm(np.asarray(support_point_uv, dtype=np.float32) - bottom))
    )
    support_to_selected = (
        None
        if selected_uv is None or not selected_visible or support_point_uv is None or not support_point_visible
        else float(
            np.linalg.norm(
                np.asarray(support_point_uv, dtype=np.float32)
                - np.asarray(selected_uv, dtype=np.float32)
            )
        )
    )
    validation_passed = True
    failure_reason = None
    contact_validation_state: Literal["verified", "degraded", "unverifiable", "hard_failure"] = "verified"
    contact_validation_trusted = bool(
        support_point_visible
        and contact_distance is not None
        and contact_mode != "fallback_other_visible"
        and selected_expected_visible
    )
    if trajectory_support_mode:
        contact_validation_trusted = False
    abort_relevant = bool(contact_validation_trusted or not support_point_visible)
    if support_to_bottom is not None:
        if support_to_bottom > _OVERLAY_WARN_SUPPORT_TO_SILHOUETTE_PX:
            warning_flags.append("support_far_from_silhouette_bottom")
    else:
        warning_flags.append("support_point_not_visible")
    if road_available and (not support_occluded) and road_fraction is not None and road_fraction < _OVERLAY_FAIL_MIN_ROAD_FRACTION:
        warning_flags.append("support_patch_low_road_fraction")
    if not support_point_visible:
        validation_passed = False
        failure_reason = "support_point_not_visible"
        contact_validation_trusted = False
        if trajectory_support_mode:
            contact_validation_state = "unverifiable"
            abort_relevant = False
        else:
            contact_validation_state = "hard_failure"
            abort_relevant = True
    elif trajectory_support_mode:
        if road_available and (not support_occluded) and road_fraction is not None and road_fraction < _OVERLAY_FAIL_MIN_ROAD_FRACTION:
            validation_passed = False
            failure_reason = "trajectory_support_off_road"
            contact_validation_state = "hard_failure"
        elif warning_flags:
            contact_validation_state = "unverifiable"
    elif not selected_expected_visible:
        warning_flags.append("selected_support_foot_not_visible")
        if contact_mode == "fallback_other_visible":
            warning_flags.append("contact_validation_fallback_other_visible")
        contact_validation_state = "unverifiable"
        contact_validation_trusted = False
        abort_relevant = False
    elif contact_distance is not None and contact_distance > _OVERLAY_FAIL_SINGLE_SUPPORT_TO_CONTACT_FOOT_PX:
        validation_passed = False
        failure_reason = "contact_foot_mismatch"
        contact_validation_state = "hard_failure"
    elif (
        contact_distance is not None
        and contact_distance > _OVERLAY_FAIL_SOFT_SUPPORT_TO_CONTACT_FOOT_PX
        and road_available
        and (not support_occluded)
        and road_fraction is not None
        and road_fraction < _OVERLAY_FAIL_MIN_ROAD_FRACTION
    ):
        validation_passed = False
        failure_reason = "contact_mismatch_with_bad_road_context"
        contact_validation_state = "hard_failure"
    elif warning_flags:
        contact_validation_state = "degraded" if contact_validation_trusted else "unverifiable"

    return OverlayValidationDiagnostic(
        frame_index=int(frame_idx),
        has_visible_pedestrian=True,
        internal_render_shape=tuple(int(v) for v in ped_rgba.shape[:2]),
        overlay_shape=(
            None if overlay_shape is None else tuple(int(v) for v in overlay_shape)
        ),
        lowest_alpha_u=int(u_foot),
        lowest_alpha_v=int(v_foot),
        lowest_alpha_row_coverage_px=int(row_coverage),
        touches_image_bottom=bool(v_foot >= height - 1),
        left_foot_projected_uv=None if left_uv is None else np.asarray(left_uv, dtype=np.float32),
        right_foot_projected_uv=None if right_uv is None else np.asarray(right_uv, dtype=np.float32),
        left_foot_visible_expected=bool(left_valid),
        right_foot_visible_expected=bool(right_valid),
        support_point_projected_uv=(
            None if support_point_uv is None else np.asarray(support_point_uv, dtype=np.float32)
        ),
        support_point_projected_visible=bool(support_point_visible),
        support_point_depth_m=(
            None if support_point_depth_m is None else float(support_point_depth_m)
        ),
        scene_depth_at_support_px_m=(
            None if scene_depth is None else float(scene_depth)
        ),
        support_point_occluded_by_scene=bool(support_occluded),
        selected_foot_projected_uv=(
            None if selected_uv is None else np.asarray(selected_uv, dtype=np.float32)
        ),
        selected_foot_projected_visible=bool(selected_visible),
        support_to_left_foot_px=left_distance,
        support_to_right_foot_px=right_distance,
        support_to_contact_foot_px=contact_distance,
        contact_foot_comparison_mode=str(contact_mode),
        support_to_silhouette_bottom_px=support_to_bottom,
        support_to_selected_foot_px=support_to_selected,
        support_patch_road_fraction=road_fraction,
        support_patch_nonroad_fraction=nonroad_fraction,
        support_patch_size_px=int(patch_size),
        road_region_validation_available=bool(road_available),
        road_context_search_mode=str(road_context_mode),
        validation_passed=bool(validation_passed),
        failure_reason=failure_reason,
        support_mode=str(support_mode),
        selected_support_foot=str(selected_support_foot),
        warning_flags=tuple(sorted(set(warning_flags))),
        touches_image_left=bool(touches_left),
        touches_image_right=bool(touches_right),
        selected_support_foot_expected_visible=bool(selected_expected_visible),
        contact_validation_state=contact_validation_state,
        contact_validation_trusted=bool(contact_validation_trusted),
        abort_relevant=bool(abort_relevant),
    )


def _make_overlay_visibility_culled_diagnostic(
    *,
    frame_idx: int,
    internal_render_shape: tuple[int, int] | None,
    overlay_shape: tuple[int, int] | None,
    support_mode: str,
    selected_support_foot: str,
    left_uv: np.ndarray | None,
    left_valid: bool,
    right_uv: np.ndarray | None,
    right_valid: bool,
) -> OverlayValidationDiagnostic:
    return OverlayValidationDiagnostic(
        frame_index=int(frame_idx),
        has_visible_pedestrian=False,
        internal_render_shape=internal_render_shape,
        overlay_shape=overlay_shape,
        lowest_alpha_u=None,
        lowest_alpha_v=None,
        lowest_alpha_row_coverage_px=0,
        touches_image_bottom=False,
        left_foot_projected_uv=(
            None if left_uv is None else np.asarray(left_uv, dtype=np.float32)
        ),
        right_foot_projected_uv=(
            None if right_uv is None else np.asarray(right_uv, dtype=np.float32)
        ),
        left_foot_visible_expected=bool(left_valid),
        right_foot_visible_expected=bool(right_valid),
        support_point_projected_uv=None,
        support_point_projected_visible=False,
        support_point_depth_m=None,
        scene_depth_at_support_px_m=None,
        support_point_occluded_by_scene=False,
        selected_foot_projected_uv=None,
        selected_foot_projected_visible=False,
        support_to_left_foot_px=None,
        support_to_right_foot_px=None,
        support_to_contact_foot_px=None,
        contact_foot_comparison_mode="culled",
        support_to_silhouette_bottom_px=None,
        support_to_selected_foot_px=None,
        support_patch_road_fraction=None,
        support_patch_nonroad_fraction=None,
        support_patch_size_px=int(_OVERLAY_SUPPORT_PATCH_RADIUS_PX),
        road_region_validation_available=False,
        road_context_search_mode="culled",
        validation_passed=True,
        failure_reason=None,
        support_mode=str(support_mode),
        selected_support_foot=str(selected_support_foot),
        warning_flags=tuple(),
        selected_support_foot_expected_visible=False,
        contact_validation_state="unverifiable",
        contact_validation_trusted=False,
        abort_relevant=False,
    )


def _evaluate_overlay_validation_summary(
    diagnostics: Sequence[OverlayValidationDiagnostic],
    *,
    adaptive: AdaptiveValidationContext | None = None,
) -> dict[str, object]:
    adaptive = adaptive or AdaptiveValidationContext.from_runtime(
        ValidationPolicySettings(), None
    )
    visible = [d for d in diagnostics if d.has_visible_pedestrian]
    trusted_visible = [d for d in visible if d.contact_validation_trusted]
    support_contact = [
        float(d.support_to_contact_foot_px)
        for d in trusted_visible
        if d.support_to_contact_foot_px is not None
    ]
    median_contact = (
        None
        if not support_contact
        else float(np.median(np.asarray(support_contact, dtype=np.float32)))
    )
    p90_contact = (
        None
        if not support_contact
        else float(np.percentile(np.asarray(support_contact, dtype=np.float32), 90))
    )
    hard_fail_visible = [
        d for d in visible if str(d.contact_validation_state) == "hard_failure"
    ]
    degraded_visible = [d for d in visible if str(d.contact_validation_state) == "degraded"]
    unverifiable_visible = [
        d for d in visible if str(d.contact_validation_state) == "unverifiable"
    ]
    abort_relevant_visible = [d for d in visible if d.abort_relevant]
    hard_failure_ratio = (
        0.0
        if not abort_relevant_visible
        else float(len(hard_fail_visible) / len(abort_relevant_visible))
    )
    median_contact_soft, median_contact_hard = adaptive.max_thresholds(
        _OVERLAY_FAIL_MEDIAN_SUPPORT_TO_CONTACT_FOOT_PX
    )
    p90_contact_soft, p90_contact_hard = adaptive.max_thresholds(
        _OVERLAY_FAIL_P90_SUPPORT_TO_CONTACT_FOOT_PX
    )
    hard_failure_ratio_soft, hard_failure_ratio_hard = adaptive.max_thresholds(
        _OVERLAY_FAIL_MAX_FLAGGED_RATIO
    )
    hard_fail = bool(
        (
            median_contact is not None
            and median_contact > median_contact_hard
        )
        or (
            p90_contact is not None
            and p90_contact > p90_contact_hard
        )
        or hard_failure_ratio > hard_failure_ratio_hard
    )
    failure_reason = (
        next((d.failure_reason for d in hard_fail_visible if d.failure_reason), None)
        if hard_fail
        else None
    )
    degraded_by_threshold = bool(
        (median_contact is not None and median_contact > median_contact_soft)
        or (p90_contact is not None and p90_contact > p90_contact_soft)
        or hard_failure_ratio > hard_failure_ratio_soft
    )
    return {
        "visible": visible,
        "trusted_visible": trusted_visible,
        "hard_fail_visible": hard_fail_visible,
        "degraded_visible": degraded_visible,
        "unverifiable_visible": unverifiable_visible,
        "median_contact": median_contact,
        "p90_contact": p90_contact,
        "hard_failure_ratio": hard_failure_ratio,
        "hard_fail": hard_fail,
        "failure_reason": failure_reason,
        "validation_degraded": bool(
            degraded_visible or unverifiable_visible or hard_fail_visible or degraded_by_threshold
        ),
        "effective_thresholds": {
            "median_support_to_contact_foot_px": {
                "base": float(_OVERLAY_FAIL_MEDIAN_SUPPORT_TO_CONTACT_FOOT_PX),
                "soft": float(median_contact_soft),
                "hard": float(median_contact_hard),
            },
            "p90_support_to_contact_foot_px": {
                "base": float(_OVERLAY_FAIL_P90_SUPPORT_TO_CONTACT_FOOT_PX),
                "soft": float(p90_contact_soft),
                "hard": float(p90_contact_hard),
            },
            "max_flagged_ratio": {
                "base": float(_OVERLAY_FAIL_MAX_FLAGGED_RATIO),
                "soft": float(hard_failure_ratio_soft),
                "hard": float(hard_failure_ratio_hard),
            },
        },
        "validation_policy": adaptive.diagnostic_summary(),
    }


def _write_overlay_validation_diagnostics(
    *,
    run_dir: Path,
    diagnostics: Sequence[OverlayValidationDiagnostic],
    adaptive: AdaptiveValidationContext | None = None,
) -> tuple[Path, Path]:
    vis_dir = run_dir / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = vis_dir / "overlay_validation_diagnostics.json"
    csv_path = vis_dir / "overlay_validation_diagnostics.csv"

    summary = _evaluate_overlay_validation_summary(diagnostics, adaptive=adaptive)
    visible = summary["visible"]
    trusted_visible = summary["trusted_visible"]
    support_to_contact = [
        float(d.support_to_contact_foot_px)
        for d in trusted_visible
        if d.support_to_contact_foot_px is not None
    ]
    support_to_silhouette = [
        float(d.support_to_silhouette_bottom_px)
        for d in visible
        if d.support_to_silhouette_bottom_px is not None
    ]
    support_to_selected = [
        float(d.support_to_selected_foot_px)
        for d in visible
        if d.support_to_selected_foot_px is not None
    ]
    road_fraction = [
        float(d.support_patch_road_fraction)
        for d in visible
        if d.support_patch_road_fraction is not None
    ]
    warning_counts: dict[str, int] = {}
    longest_empty_run = 0
    current_empty_run = 0
    for diag in diagnostics:
        if diag.has_visible_pedestrian:
            longest_empty_run = max(longest_empty_run, current_empty_run)
            current_empty_run = 0
        else:
            current_empty_run += 1
        for flag in diag.warning_flags:
            warning_counts[flag] = warning_counts.get(flag, 0) + 1
    longest_empty_run = max(longest_empty_run, current_empty_run)

    def _summary_stats(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "p90": None}
        arr = np.asarray(values, dtype=np.float32)
        return {
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        }

    payload = {
        "count": len(diagnostics),
        "visible_frame_count": len(visible),
        "trusted_visible_frame_count": len(trusted_visible),
        "empty_frame_count": len(diagnostics) - len(visible),
        "longest_empty_run": int(longest_empty_run),
        "validation_passed": bool(not summary["hard_fail"]),
        "validation_degraded": bool(summary["validation_degraded"]),
        "failure_reason": summary["failure_reason"],
        "hard_failure_visible_frame_count": int(len(summary["hard_fail_visible"])),
        "degraded_visible_frame_count": int(len(summary["degraded_visible"])),
        "unverifiable_visible_frame_count": int(len(summary["unverifiable_visible"])),
        "support_to_contact_foot_distance_stats_px": _summary_stats(support_to_contact),
        "support_to_silhouette_distance_stats_px": _summary_stats(support_to_silhouette),
        "support_to_selected_foot_distance_stats_px": _summary_stats(support_to_selected),
        "road_patch_fraction_stats": _summary_stats(road_fraction),
        "warning_counts": warning_counts,
        "aggregate_abort_metrics": {
            "trusted_median_support_to_contact_foot_px": summary["median_contact"],
            "trusted_p90_support_to_contact_foot_px": summary["p90_contact"],
            "hard_failure_ratio": summary["hard_failure_ratio"],
            "hard_fail": bool(summary["hard_fail"]),
        },
        "validation_policy": summary["validation_policy"],
        "effective_thresholds": summary["effective_thresholds"],
        "fail_thresholds": {
            "median_support_to_contact_foot_px": _OVERLAY_FAIL_MEDIAN_SUPPORT_TO_CONTACT_FOOT_PX,
            "p90_support_to_contact_foot_px": _OVERLAY_FAIL_P90_SUPPORT_TO_CONTACT_FOOT_PX,
            "single_support_to_contact_foot_px": _OVERLAY_FAIL_SINGLE_SUPPORT_TO_CONTACT_FOOT_PX,
            "soft_support_to_contact_foot_px": _OVERLAY_FAIL_SOFT_SUPPORT_TO_CONTACT_FOOT_PX,
            "max_flagged_ratio": _OVERLAY_FAIL_MAX_FLAGGED_RATIO,
            "min_road_fraction": _OVERLAY_FAIL_MIN_ROAD_FRACTION,
        },
        "entries": [
            {
                "frame_index": int(d.frame_index),
                "has_visible_pedestrian": bool(d.has_visible_pedestrian),
                "internal_render_shape": (
                    None
                    if d.internal_render_shape is None
                    else [int(v) for v in d.internal_render_shape]
                ),
                "overlay_shape": (
                    None if d.overlay_shape is None else [int(v) for v in d.overlay_shape]
                ),
                "lowest_alpha_u": d.lowest_alpha_u,
                "lowest_alpha_v": d.lowest_alpha_v,
                "lowest_alpha_row_coverage_px": int(d.lowest_alpha_row_coverage_px),
                "touches_image_bottom": bool(d.touches_image_bottom),
                "touches_image_left": bool(d.touches_image_left),
                "touches_image_right": bool(d.touches_image_right),
                "left_foot_projected_uv": (
                    None
                    if d.left_foot_projected_uv is None
                    else np.asarray(d.left_foot_projected_uv, dtype=float).tolist()
                ),
                "right_foot_projected_uv": (
                    None
                    if d.right_foot_projected_uv is None
                    else np.asarray(d.right_foot_projected_uv, dtype=float).tolist()
                ),
                "left_foot_visible_expected": bool(d.left_foot_visible_expected),
                "right_foot_visible_expected": bool(d.right_foot_visible_expected),
                "support_point_projected_uv": (
                    None
                    if d.support_point_projected_uv is None
                    else np.asarray(d.support_point_projected_uv, dtype=float).tolist()
                ),
                "support_point_projected_visible": bool(d.support_point_projected_visible),
                "support_point_depth_m": (
                    None
                    if d.support_point_depth_m is None
                    else float(d.support_point_depth_m)
                ),
                "scene_depth_at_support_px_m": (
                    None
                    if d.scene_depth_at_support_px_m is None
                    else float(d.scene_depth_at_support_px_m)
                ),
                "support_point_occluded_by_scene": bool(d.support_point_occluded_by_scene),
                "selected_foot_projected_uv": (
                    None
                    if d.selected_foot_projected_uv is None
                    else np.asarray(d.selected_foot_projected_uv, dtype=float).tolist()
                ),
                "selected_foot_projected_visible": bool(d.selected_foot_projected_visible),
                "support_to_left_foot_px": (
                    None
                    if d.support_to_left_foot_px is None
                    else float(d.support_to_left_foot_px)
                ),
                "support_to_right_foot_px": (
                    None
                    if d.support_to_right_foot_px is None
                    else float(d.support_to_right_foot_px)
                ),
                "support_to_contact_foot_px": (
                    None
                    if d.support_to_contact_foot_px is None
                    else float(d.support_to_contact_foot_px)
                ),
                "contact_foot_comparison_mode": str(d.contact_foot_comparison_mode),
                "support_to_silhouette_bottom_px": (
                    None
                    if d.support_to_silhouette_bottom_px is None
                    else float(d.support_to_silhouette_bottom_px)
                ),
                "support_to_selected_foot_px": (
                    None
                    if d.support_to_selected_foot_px is None
                    else float(d.support_to_selected_foot_px)
                ),
                "support_patch_road_fraction": (
                    None
                    if d.support_patch_road_fraction is None
                    else float(d.support_patch_road_fraction)
                ),
                "support_patch_nonroad_fraction": (
                    None
                    if d.support_patch_nonroad_fraction is None
                    else float(d.support_patch_nonroad_fraction)
                ),
                "support_patch_size_px": int(d.support_patch_size_px),
                "road_region_validation_available": bool(d.road_region_validation_available),
                "road_context_search_mode": str(d.road_context_search_mode),
                "validation_passed": bool(d.validation_passed),
                "failure_reason": d.failure_reason,
                "support_mode": str(d.support_mode),
                "selected_support_foot": str(d.selected_support_foot),
                "selected_support_foot_expected_visible": bool(
                    d.selected_support_foot_expected_visible
                ),
                "contact_validation_state": str(d.contact_validation_state),
                "contact_validation_trusted": bool(d.contact_validation_trusted),
                "abort_relevant": bool(d.abort_relevant),
                "warning_flags": list(d.warning_flags),
            }
            for d in diagnostics
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "has_visible_pedestrian",
                "internal_render_height",
                "internal_render_width",
                "overlay_height",
                "overlay_width",
                "lowest_alpha_u",
                "lowest_alpha_v",
                "lowest_alpha_row_coverage_px",
                "touches_image_bottom",
                "touches_image_left",
                "touches_image_right",
                "left_foot_u",
                "left_foot_v",
                "right_foot_u",
                "right_foot_v",
                "left_foot_visible_expected",
                "right_foot_visible_expected",
                "support_point_u",
                "support_point_v",
                "support_point_projected_visible",
                "support_point_depth_m",
                "scene_depth_at_support_px_m",
                "support_point_occluded_by_scene",
                "selected_foot_u",
                "selected_foot_v",
                "selected_foot_projected_visible",
                "support_to_left_foot_px",
                "support_to_right_foot_px",
                "support_to_contact_foot_px",
                "contact_foot_comparison_mode",
                "support_to_silhouette_bottom_px",
                "support_to_selected_foot_px",
                "support_patch_road_fraction",
                "support_patch_nonroad_fraction",
                "support_patch_size_px",
                "road_region_validation_available",
                "road_context_search_mode",
                "validation_passed",
                "failure_reason",
                "support_mode",
                "selected_support_foot",
                "selected_support_foot_expected_visible",
                "contact_validation_state",
                "contact_validation_trusted",
                "abort_relevant",
                "warning_flags",
            ]
        )
        for d in diagnostics:
            left_uv = (
                [None, None]
                if d.left_foot_projected_uv is None
                else np.asarray(d.left_foot_projected_uv, dtype=float).reshape(2)
            )
            right_uv = (
                [None, None]
                if d.right_foot_projected_uv is None
                else np.asarray(d.right_foot_projected_uv, dtype=float).reshape(2)
            )
            support_uv = (
                [None, None]
                if d.support_point_projected_uv is None
                else np.asarray(d.support_point_projected_uv, dtype=float).reshape(2)
            )
            selected_uv = (
                [None, None]
                if d.selected_foot_projected_uv is None
                else np.asarray(d.selected_foot_projected_uv, dtype=float).reshape(2)
            )
            writer.writerow(
                [
                    int(d.frame_index),
                    int(bool(d.has_visible_pedestrian)),
                    (
                        ""
                        if d.internal_render_shape is None
                        else int(d.internal_render_shape[0])
                    ),
                    (
                        ""
                        if d.internal_render_shape is None
                        else int(d.internal_render_shape[1])
                    ),
                    "" if d.overlay_shape is None else int(d.overlay_shape[0]),
                    "" if d.overlay_shape is None else int(d.overlay_shape[1]),
                    "" if d.lowest_alpha_u is None else int(d.lowest_alpha_u),
                    "" if d.lowest_alpha_v is None else int(d.lowest_alpha_v),
                    int(d.lowest_alpha_row_coverage_px),
                    int(bool(d.touches_image_bottom)),
                    int(bool(d.touches_image_left)),
                    int(bool(d.touches_image_right)),
                    left_uv[0],
                    left_uv[1],
                    right_uv[0],
                    right_uv[1],
                    int(bool(d.left_foot_visible_expected)),
                    int(bool(d.right_foot_visible_expected)),
                    support_uv[0],
                    support_uv[1],
                    int(bool(d.support_point_projected_visible)),
                    (
                        ""
                        if d.support_point_depth_m is None
                        else float(d.support_point_depth_m)
                    ),
                    (
                        ""
                        if d.scene_depth_at_support_px_m is None
                        else float(d.scene_depth_at_support_px_m)
                    ),
                    int(bool(d.support_point_occluded_by_scene)),
                    selected_uv[0],
                    selected_uv[1],
                    int(bool(d.selected_foot_projected_visible)),
                    (
                        ""
                        if d.support_to_left_foot_px is None
                        else float(d.support_to_left_foot_px)
                    ),
                    (
                        ""
                        if d.support_to_right_foot_px is None
                        else float(d.support_to_right_foot_px)
                    ),
                    (
                        ""
                        if d.support_to_contact_foot_px is None
                        else float(d.support_to_contact_foot_px)
                    ),
                    str(d.contact_foot_comparison_mode),
                    (
                        ""
                        if d.support_to_silhouette_bottom_px is None
                        else float(d.support_to_silhouette_bottom_px)
                    ),
                    (
                        ""
                        if d.support_to_selected_foot_px is None
                        else float(d.support_to_selected_foot_px)
                    ),
                    (
                        ""
                        if d.support_patch_road_fraction is None
                        else float(d.support_patch_road_fraction)
                    ),
                    (
                        ""
                        if d.support_patch_nonroad_fraction is None
                        else float(d.support_patch_nonroad_fraction)
                    ),
                    int(d.support_patch_size_px),
                    int(bool(d.road_region_validation_available)),
                    str(d.road_context_search_mode),
                    int(bool(d.validation_passed)),
                    "" if d.failure_reason is None else str(d.failure_reason),
                    str(d.support_mode),
                    str(d.selected_support_foot),
                    int(bool(d.selected_support_foot_expected_visible)),
                    str(d.contact_validation_state),
                    int(bool(d.contact_validation_trusted)),
                    int(bool(d.abort_relevant)),
                    "|".join(d.warning_flags),
                ]
            )
    return json_path, csv_path


def _overlay_validation_policy_for_run(run_dir: Path) -> AdaptiveValidationContext:
    profile_snapshot_path = run_dir / "standard" / "profile.json"
    policy = ValidationPolicySettings()
    fps = None
    if profile_snapshot_path.exists():
        try:
            with profile_snapshot_path.open("r", encoding="utf-8") as handle:
                profile = json.load(handle)
            runtime = profile.get("runtime", {})
            runtime_settings = runtime.get("settings", {}) if isinstance(runtime, dict) else {}
            if isinstance(runtime_settings, dict):
                policy = ValidationPolicySettings.from_mapping(runtime_settings.get("validation_policy"))
            raw_fps = _raw_sampling_fps_from_profile(profile) if isinstance(profile, dict) else None
            if raw_fps is not None:
                fps = float(raw_fps)
        except Exception:
            policy = ValidationPolicySettings()
            fps = None
    context = None
    if fps is not None:
        context = {
            "frame_provider_info": {
                "tool": "profile_snapshot",
                "settings": {"sampling_fps": float(fps)},
            }
        }
    return AdaptiveValidationContext.from_runtime(policy, context)


def compose_overlay_frames(
    *,
    run_dir: Path,
    actor_name: str,
    road_labels: Sequence[str],
    contact_ground_labels: Sequence[str] | None,
    occlusion_spec: Any | None,
    shadow_spec: Any | None,
    grounding_diagnostics: list[GroundingDiagnostic],
    original_frames_dir: Path,
    pedestrian_frames_dir: Path,
    pedestrian_depth_frames_dir: Path,
    shadow_frames_dir: Path | None,
    output_dir: Path,
    debug_output_dir: Path | None = None,
    support_debug_output_dir: Path | None = None,
    support_local_grid_output_dir: Path | None = None,
    occlusion_mask_output_dir: Path | None = None,
    occlusion_debug_output_dir: Path | None = None,
) -> None:
    """Overlay pedestrian RGBA frames on top of original frames."""
    with log_scope("Overlay"):
        original_frames = _build_frame_index_map(original_frames_dir)
        pedestrian_frames = _build_frame_index_map(pedestrian_frames_dir)
        shadow_enabled = bool(getattr(shadow_spec, "enabled", True))
        shadow_frames = (
            _build_frame_index_map(shadow_frames_dir)
            if shadow_frames_dir is not None and shadow_frames_dir.exists()
            else {}
        )
        if shadow_enabled and shadow_frames_dir is not None and not shadow_frames:
            raise ValueError(f"Shadow frames were expected but not found in {shadow_frames_dir}.")
        intrinsics_k, frame_to_c2w, _road_planes = _load_overlay_validation_context(run_dir)
        overlay_validation_policy = _overlay_validation_policy_for_run(run_dir)
        ped_depth_mode = _read_depth_sequence_mode(pedestrian_depth_frames_dir)
        validation_diagnostics: list[OverlayValidationDiagnostic] = []
        occlusion_diagnostics: list[OcclusionFrameDiagnostics] = []
        actor_root = _resolve_actor_root(actor_name)
        armature_obj = _find_actor_armature(actor_name)
        support_point_by_frame = _build_support_point_lookup(grounding_diagnostics)
        grounding_by_frame = _build_grounding_diagnostic_lookup(grounding_diagnostics)
        missing_original = [
            idx
            for idx in sorted(pedestrian_frames.keys())
            if idx not in original_frames
        ]
        if missing_original:
            missing_preview = ", ".join(str(idx) for idx in missing_original[:10])
            raise ValueError(
                "Missing original frames for pedestrian indices: "
                f"{missing_preview}{'...' if len(missing_original) > 10 else ''}"
            )
        pedestrian_depth_frames_dir.mkdir(parents=True, exist_ok=True)

        if occlusion_mask_output_dir is None:
            occlusion_mask_output_dir = ResourceStore.blender_artifact_dir_for(
                run_dir,
                "occlusion_masks",
            )
        if occlusion_debug_output_dir is None:
            occlusion_debug_output_dir = ResourceStore.blender_artifact_dir_for(
                run_dir,
                "occlusion_debug",
            )
        occlusion_mask_output_dir.mkdir(parents=True, exist_ok=True)
        occlusion_debug_output_dir.mkdir(parents=True, exist_ok=True)
        occlusion_settings = OcclusionSettings(
            default_front_margin_m=float(
                getattr(occlusion_spec, "default_front_margin_m", 0.03)
            ),
            relative_margin=float(getattr(occlusion_spec, "relative_margin", 0.01)),
            alpha_presence_threshold=_OVERLAY_ALPHA_THRESHOLD,
            alpha_visible_threshold=_OVERLAY_ALPHA_THRESHOLD,
            contact_plane_band_m=float(
                getattr(occlusion_spec, "contact_plane_band_m", 0.025)
            ),
            contact_patch_radius_m=float(
                getattr(occlusion_spec, "contact_patch_radius_m", 0.30)
            ),
            contact_coplanar_tolerance_m=float(
                getattr(occlusion_spec, "contact_coplanar_tolerance_m", 0.03)
            ),
            edge_treatment=EdgeTreatmentSettings(
                enabled=bool(
                    getattr(getattr(occlusion_spec, "edge_treatment", None), "enabled", True)
                ),
                boundary_band_px=int(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "boundary_band_px",
                        4,
                    )
                ),
                feather_radius_px=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "feather_radius_px",
                        2.0,
                    )
                ),
                feather_strength=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "feather_strength",
                        0.35,
                    )
                ),
                blur_enabled=bool(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "blur_enabled",
                        True,
                    )
                ),
                blur_radius_px=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "blur_radius_px",
                        1.5,
                    )
                ),
                blur_strength=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "blur_strength",
                        0.25,
                    )
                ),
                despill_enabled=bool(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "despill_enabled",
                        True,
                    )
                ),
                despill_strength=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "despill_strength",
                        0.25,
                    )
                ),
                regrain_enabled=bool(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "regrain_enabled",
                        True,
                    )
                ),
                regrain_strength=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "regrain_strength",
                        0.12,
                    )
                ),
                tiny_object_disable_feather=bool(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "tiny_object_disable_feather",
                        True,
                    )
                ),
                tiny_object_disable_blur=bool(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "tiny_object_disable_blur",
                        True,
                    )
                ),
                tiny_object_disable_despill=bool(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "tiny_object_disable_despill",
                        True,
                    )
                ),
                tiny_object_disable_regrain=bool(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "tiny_object_disable_regrain",
                        True,
                    )
                ),
                tiny_object_max_boundary_fraction=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "tiny_object_max_boundary_fraction",
                        0.25,
                    )
                ),
                tiny_object_disable_all_below_short_side_px=int(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "tiny_object_disable_all_below_short_side_px",
                        20,
                    )
                ),
                tiny_object_disable_all_below_visible_pixels=int(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "tiny_object_disable_all_below_visible_pixels",
                        256,
                    )
                ),
                disable_when_boundary_fraction_above=float(
                    getattr(
                        getattr(occlusion_spec, "edge_treatment", None),
                        "disable_when_boundary_fraction_above",
                        0.6,
                    )
                ),
            ),
            temporal_stabilization=TemporalOcclusionSettings(
                enabled=bool(
                    getattr(
                        getattr(occlusion_spec, "temporal_stabilization", None),
                        "enabled",
                        True,
                    )
                ),
                base_hysteresis_margin_m=float(
                    getattr(
                        getattr(occlusion_spec, "temporal_stabilization", None),
                        "base_hysteresis_margin_m",
                        0.02,
                    )
                ),
                state_flip_persist_frames=int(
                    getattr(
                        getattr(occlusion_spec, "temporal_stabilization", None),
                        "state_flip_persist_frames",
                        2,
                    )
                ),
                edge_exit_hold_frames=int(
                    getattr(
                        getattr(occlusion_spec, "temporal_stabilization", None),
                        "edge_exit_hold_frames",
                        2,
                    )
                ),
                max_single_frame_visible_area_drop_ratio=float(
                    getattr(
                        getattr(occlusion_spec, "temporal_stabilization", None),
                        "max_single_frame_visible_area_drop_ratio",
                        0.5,
                    )
                ),
            ),
        )
        temporal_occlusion_state = TemporalOcclusionState()

        output_dir.mkdir(parents=True, exist_ok=True)
        if debug_output_dir is not None:
            debug_output_dir.mkdir(parents=True, exist_ok=True)
        if support_debug_output_dir is not None:
            support_debug_output_dir.mkdir(parents=True, exist_ok=True)
        if support_local_grid_output_dir is not None:
            support_local_grid_output_dir.mkdir(parents=True, exist_ok=True)
        log_info(
            f"Compositing {len(pedestrian_frames)} overlay frames -> {output_dir}"
            + (
                f" (support-point frames -> {debug_output_dir})"
                if debug_output_dir is not None
                else ""
            )
            + (
                f" (support-debug frames -> {support_debug_output_dir})"
                if support_debug_output_dir is not None
                else ""
            )
            + (
                f" (support-local-grid frames -> {support_local_grid_output_dir})"
                if support_local_grid_output_dir is not None
                else ""
            )
            + f" (occlusion masks -> {occlusion_mask_output_dir})"
        )

        for frame_idx in sorted(pedestrian_frames.keys()):
            background_path = original_frames[frame_idx]
            pedestrian_path = pedestrian_frames[frame_idx]
            pedestrian_depth_path = pedestrian_depth_frames_dir / f"{frame_idx:06d}.npz"
            shadow_path = shadow_frames.get(frame_idx)
            if shadow_enabled and shadow_frames_dir is not None and shadow_path is None:
                raise ValueError(f"Missing shadow frame for overlay frame {frame_idx}.")
            scene_depth_path = run_dir / "standard" / "depth" / f"{frame_idx:06d}.npz"
            if frame_idx not in support_point_by_frame:
                raise ValueError(f"Missing grounding diagnostic for overlay frame {frame_idx}.")
            if frame_idx not in grounding_by_frame:
                raise ValueError(f"Missing grounding metadata for overlay frame {frame_idx}.")
            support_point = support_point_by_frame[frame_idx]
            grounding_diag = grounding_by_frame[frame_idx]
            ped_rgba_top_origin = _load_rgba_image(pedestrian_path)
            ped_alpha_mask_top = (
                np.asarray(ped_rgba_top_origin[:, :, 3], dtype=np.float32) / 255.0
            ) > _OVERLAY_ALPHA_THRESHOLD
            ped_pixel_count = int(np.count_nonzero(ped_alpha_mask_top))
            background_rgba_top_origin = _load_rgba_image(background_path)
            overlay_image_shape = tuple(
                int(v) for v in background_rgba_top_origin.shape[:2]
            )
            bpy.context.scene.frame_set(int(frame_idx))
            bpy.context.view_layer.update()
            height_rgba = int(ped_rgba_top_origin.shape[0])
            width_rgba = int(ped_rgba_top_origin.shape[1])
            support_uv, support_visible, support_depth = _project_overlay_support_point(
                frame_idx=frame_idx,
                intrinsics_k=intrinsics_k,
                frame_to_c2w=frame_to_c2w,
                support_point=support_point,
                image_shape=overlay_image_shape,
            )
            support_depth_invalid = ped_pixel_count > 0 and (
                (not support_visible)
                or support_depth is None
                or (not np.isfinite(float(support_depth)))
                or float(support_depth) <= 0.0
            )
            ped_depth_missing = not pedestrian_depth_path.exists()
            synthesized_support_depth = False
            ped_depth_empty = False
            if (not ped_depth_missing) and ped_pixel_count > 0:
                ped_depth = _load_depth_npz_array(pedestrian_depth_path)
                if ped_depth.shape != (height_rgba, width_rgba):
                    raise ValueError(
                        "Pedestrian depth shape mismatch before composition: "
                        f"got {ped_depth.shape}, expected {(height_rgba, width_rgba)}."
                    )
                ped_depth_empty = not bool(
                    np.any(
                        np.isfinite(ped_depth)
                        & (ped_depth > 0.0)
                        & ped_alpha_mask_top
                    )
                )
                if ped_depth_empty and (not support_depth_invalid):
                    ped_depth = ped_depth.copy()
                    ped_depth[ped_alpha_mask_top] = float(support_depth)
                    np.savez_compressed(
                        pedestrian_depth_path,
                        depth=np.asarray(ped_depth, dtype=np.float32),
                    )
                    synthesized_support_depth = True
                    ped_depth_empty = False
            ground_mask = _load_overlay_ground_mask(
                run_dir=run_dir,
                frame_idx=int(frame_idx),
                image_shape=overlay_image_shape,
                ground_labels=(
                    tuple(contact_ground_labels)
                    if contact_ground_labels is not None
                    else tuple(road_labels)
                ),
                required=True,
            )

            try:
                if ped_depth_missing:
                    log_info(
                        "Falling back to alpha-only overlay for frame "
                        f"(pedestrian_depth_path missing: {pedestrian_depth_path})."
                    )
                elif ped_depth_empty and support_depth_invalid:
                    log_info(
                        "Falling back to alpha-only overlay for frame "
                        f"{frame_idx}: empty pedestrian depth and invalid support depth."
                    )
                elif synthesized_support_depth:
                    log_info(
                        "Replacing empty pedestrian depth with support-constant depth for frame "
                        f"{frame_idx}."
                    )
                out_rgb_top, ped_visible_mask_top, occlusion_diag = (
                    compose_overlay_frame_with_occlusion(
                        frame_idx=int(frame_idx),
                        original_frame_path=background_path,
                        pedestrian_rgba_path=pedestrian_path,
                        scene_depth_path=scene_depth_path,
                        pedestrian_depth_path=pedestrian_depth_path,
                        settings=occlusion_settings,
                        mask_output_path=occlusion_mask_output_dir / f"{frame_idx:06d}.png",
                        debug_output_path=occlusion_debug_output_dir / f"{frame_idx:06d}.png",
                        force_alpha_only=bool(
                            ped_depth_missing or (ped_depth_empty and support_depth_invalid)
                        ),
                        intrinsics_k=intrinsics_k,
                        camera_to_world_matrix=frame_to_c2w[int(frame_idx)],
                        traversable_ground_mask=ground_mask,
                        support_anchor_world=(
                            None
                            if support_depth_invalid or support_point is None
                            else np.asarray(support_point, dtype=np.float32)
                        ),
                        support_plane_normal=(
                            None
                            if grounding_diag.chosen_plane_normal is None
                            else np.asarray(grounding_diag.chosen_plane_normal, dtype=np.float32)
                        ),
                        support_plane_offset=(
                            None
                            if grounding_diag.chosen_plane_offset is None
                            else float(grounding_diag.chosen_plane_offset)
                        ),
                        temporal_state=temporal_occlusion_state,
                        shadow_rgba_path=shadow_path,
                        shadow_opacity=float(getattr(shadow_spec, "opacity", 1.0)),
                        shadow_blur_radius_px=float(
                            getattr(shadow_spec, "softness", 1.5)
                        ),
                        shadow_tint_rgb=tuple(
                            float(v)
                            for v in getattr(
                                shadow_spec,
                                "tint_rgb",
                                (0.0, 0.0, 0.0),
                            )
                        ),
                    )
                )
                occlusion_diagnostics.append(
                    replace(
                        occlusion_diag,
                        ped_depth_mode=(
                            (
                                "support_constant_fallback"
                                if synthesized_support_depth
                                else ped_depth_mode
                            )
                            if not (ped_depth_missing or (ped_depth_empty and support_depth_invalid))
                            else "alpha_only_missing_ped_depth"
                        ),
                        support_depth_m=(
                            None
                            if (support_depth is None or support_depth_invalid)
                            else float(support_depth)
                        ),
                    )
                )
                height = int(out_rgb_top.shape[0])
                width = int(out_rgb_top.shape[1])
                feet = _evaluate_feet_world(actor_root, armature_obj)
                left_foot = feet["left"]
                right_foot = feet["right"]
                left_uv, left_valid, right_uv, right_valid = _project_overlay_feet(
                    frame_idx=frame_idx,
                    intrinsics_k=intrinsics_k,
                    frame_to_c2w=frame_to_c2w,
                    left_foot=left_foot,
                    right_foot=right_foot,
                    image_shape=(height, width),
                )
                alpha_mask_top = (
                    np.asarray(ped_rgba_top_origin[:, :, 3], dtype=np.float32)
                    > (_OVERLAY_ALPHA_THRESHOLD * 255.0)
                )
                has_rendered_subject = bool(np.any(alpha_mask_top))
                if support_depth_invalid and not has_rendered_subject:
                    validation_diag = _make_overlay_visibility_culled_diagnostic(
                        frame_idx=frame_idx,
                        internal_render_shape=(height_rgba, width_rgba),
                        overlay_shape=overlay_image_shape,
                        support_mode=str(grounding_diag.support_mode),
                        selected_support_foot=str(grounding_diag.selected_support_foot),
                        left_uv=left_uv,
                        left_valid=left_valid,
                        right_uv=right_uv,
                        right_valid=right_valid,
                    )
                else:
                    validation_diag = _make_overlay_validation_diagnostic(
                        run_dir=run_dir,
                        frame_idx=frame_idx,
                        ped_rgba=ped_rgba_top_origin,
                        overlay_shape=overlay_image_shape,
                        left_uv=left_uv,
                        left_valid=left_valid,
                        right_uv=right_uv,
                        right_valid=right_valid,
                        selected_support_foot=str(grounding_diag.selected_support_foot),
                        support_mode=str(grounding_diag.support_mode),
                        support_point_uv=support_uv,
                        support_point_visible=support_visible,
                        support_point_depth_m=support_depth,
                        road_labels=road_labels,
                    )
                validation_diagnostics.append(validation_diag)

                # Optional overlay diagnostics (kept for debugging, gated by logger settings).
                if LOGGER.show_overlay_logs:

                    def _alpha_bbox_and_centroid(
                        alpha: np.ndarray, min_alpha: float = 0.05
                    ):
                        mask = alpha > min_alpha
                        if not np.any(mask):
                            return None
                        ys, xs = np.where(mask)
                        x0, x1 = int(xs.min()), int(xs.max())
                        y0, y1 = int(ys.min()), int(ys.max())
                        center_x = float(xs.mean())
                        center_y = float(ys.mean())
                        return (x0, y0, x1, y1, center_x, center_y)

                    info = _alpha_bbox_and_centroid(
                        np.asarray(ped_rgba_top_origin[:, :, 3], dtype=np.float32) / 255.0,
                        min_alpha=0.05,
                    )
                    if info is None:
                        log_info(f"[compose] frame={frame_idx} no-alpha")
                    else:
                        x0, y0, x1, y1, center_x, center_y = info
                        log_info(
                            f"[compose] frame={frame_idx} ped_bbox=({x0},{y0})-({x1},{y1}) "
                            f"ped_ctr=({center_x:.1f},{center_y:.1f})"
                        )

                # `compose_overlay_frame_with_occlusion` already returns a top-origin image.
                out_rgb = np.asarray(out_rgb_top, dtype=np.float32) / 255.0
                out_alpha = np.ones((height, width, 1), dtype=np.float32)
                out_pixels = np.concatenate([out_rgb, out_alpha], axis=2)
                debug_out_pixels = out_pixels.copy()
                support_debug_pixels = out_pixels.copy()
                support_local_grid_pixels = out_pixels.copy()
                if debug_output_dir is not None and support_uv is not None and support_visible:
                    _draw_support_marker_rgba(
                        debug_out_pixels,
                        int(np.rint(float(support_uv[0]))),
                        int(np.rint(float(support_uv[1]))),
                    )
                elif debug_output_dir is not None and LOGGER.show_overlay_logs:
                    log_info(f"[compose] frame={frame_idx} support marker omitted")
                if support_debug_output_dir is not None:
                    if support_uv is not None and support_visible:
                        _draw_support_marker_rgba(
                            support_debug_pixels,
                            int(np.rint(float(support_uv[0]))),
                            int(np.rint(float(support_uv[1]))),
                            color=(1.0, 0.0, 0.0),
                        )
                        _draw_support_patch_outline_rgba(
                            support_debug_pixels,
                            int(np.rint(float(support_uv[0]))),
                            int(np.rint(float(support_uv[1]))),
                            radius_px=_OVERLAY_SUPPORT_PATCH_RADIUS_PX,
                        )
                    if (
                        validation_diag.selected_foot_projected_uv is not None
                        and validation_diag.selected_foot_projected_visible
                    ):
                        _draw_support_marker_rgba(
                            support_debug_pixels,
                            int(np.rint(float(validation_diag.selected_foot_projected_uv[0]))),
                            int(np.rint(float(validation_diag.selected_foot_projected_uv[1]))),
                            color=(1.0, 1.0, 0.0),
                        )
                if support_local_grid_output_dir is not None:
                    support_local_grid_pixels = _render_overlay_support_local_grid(
                        run_dir=run_dir,
                        frame_idx=int(frame_idx),
                        out_pixels=out_pixels,
                        grounding_diag=grounding_diag,
                        intrinsics_k=intrinsics_k,
                        frame_to_c2w=frame_to_c2w,
                        road_labels=road_labels,
                        pedestrian_visible_mask_top=(
                            None if support_depth_invalid else ped_visible_mask_top
                        ),
                        road_mask=ground_mask,
                    )
                    if support_uv is not None and support_visible:
                        _draw_support_marker_rgba(
                            support_local_grid_pixels,
                            int(np.rint(float(support_uv[0]))),
                            int(np.rint(float(support_uv[1]))),
                            color=(1.0, 0.0, 0.0),
                        )

                _write_rgba_image(
                    output_dir / f"{frame_idx:06d}.png",
                    np.clip(np.rint(out_pixels * 255.0), 0.0, 255.0).astype(np.uint8),
                )
                if debug_output_dir is not None:
                    _write_rgba_image(
                        debug_output_dir / f"{frame_idx:06d}.png",
                        np.clip(np.rint(debug_out_pixels * 255.0), 0.0, 255.0).astype(np.uint8),
                    )
                if support_debug_output_dir is not None:
                    _write_rgba_image(
                        support_debug_output_dir / f"{frame_idx:06d}.png",
                        np.clip(np.rint(support_debug_pixels * 255.0), 0.0, 255.0).astype(np.uint8),
                    )
                if support_local_grid_output_dir is not None:
                    _write_rgba_image(
                        support_local_grid_output_dir / f"{frame_idx:06d}.png",
                        np.clip(
                            np.rint(support_local_grid_pixels * 255.0),
                            0.0,
                            255.0,
                        ).astype(np.uint8),
                    )
            finally:
                pass

        occ_json_path, occ_csv_path = write_occlusion_diagnostics(
            run_dir=run_dir,
            diagnostics=occlusion_diagnostics,
        )
        log_info(
            "Occlusion diagnostics written: "
            f"json={occ_json_path} csv={occ_csv_path} entries={len(occlusion_diagnostics)}"
        )
        json_path, csv_path = _write_overlay_validation_diagnostics(
            run_dir=run_dir,
            diagnostics=validation_diagnostics,
            adaptive=overlay_validation_policy,
        )
        log_info(
            "Overlay validation diagnostics written: "
            f"json={json_path} csv={csv_path} entries={len(validation_diagnostics)}"
        )
        summary = _evaluate_overlay_validation_summary(
            validation_diagnostics,
            adaptive=overlay_validation_policy,
        )
        visible = summary["visible"]
        hard_failed = summary["hard_fail_visible"]
        if visible and bool(summary["hard_fail"]):
            median_contact = summary["median_contact"]
            p90_contact = summary["p90_contact"]
            ratio = float(summary["hard_failure_ratio"])
            median_text = "n/a" if median_contact is None else f"{float(median_contact):.1f}px"
            p90_text = "n/a" if p90_contact is None else f"{float(p90_contact):.1f}px"
            raise ValueError(
                "Overlay support validation failed after diagnostics were written: "
                f"median_support_to_contact_foot={median_text} "
                f"p90_support_to_contact_foot={p90_text} "
                f"hard_failed_visible_frames={len(hard_failed)}/{len(visible)} "
                f"hard_failure_ratio={ratio:.2f} "
                f"(see {json_path.name})."
            )


def configure_world_ambient(strength: float) -> None:
    """Configure subtle world-space ambient fill for neutral coverage."""
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    node_tree = world.node_tree
    if node_tree is None:
        raise ValueError("World node tree is unavailable.")
    background = None
    output = None
    for node in node_tree.nodes:
        if node.type == "BACKGROUND" and background is None:
            background = node
        if node.type == "OUTPUT_WORLD" and output is None:
            output = node
    if background is None:
        background = node_tree.nodes.new(type="ShaderNodeBackground")
    if output is None:
        output = node_tree.nodes.new(type="ShaderNodeOutputWorld")
    if not any(
        link.from_node == background and link.to_node == output
        for link in node_tree.links
    ):
        node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = float(strength)


def _standardized_lighting_payload(run_dir: Path) -> LightingData | None:
    json_path = run_dir / "standard" / "lighting" / "lighting.json"
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    validation = payload.get("validation", {})
    if not isinstance(validation, Mapping) or not bool(validation.get("passed", False)):
        raise ValueError(
            f"Lighting contract at {json_path} is not validated for rendering."
        )
    envmap_path = run_dir / str(payload.get("envmap_path", "standard/lighting/envmap.exr"))
    if not envmap_path.exists():
        raise FileNotFoundError(
            f"Lighting envmap declared by {json_path} was not found at {envmap_path}."
        )
    lighting = lighting_from_payload(payload, envmap_path=str(envmap_path), key=str(json_path))
    lighting.metadata["_json_path"] = str(json_path)
    return lighting


def configure_world_envmap(envmap_path: Path, rotation_world: Sequence[float], strength: float) -> None:
    """Configure a world HDR envmap with explicit mapping rotation."""
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    node_tree = world.node_tree
    if node_tree is None:
        raise ValueError("World node tree is unavailable.")
    node_tree.nodes.clear()
    tex_coord = node_tree.nodes.new(type="ShaderNodeTexCoord")
    mapping = node_tree.nodes.new(type="ShaderNodeMapping")
    env_tex = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    background = node_tree.nodes.new(type="ShaderNodeBackground")
    output = node_tree.nodes.new(type="ShaderNodeOutputWorld")
    env_tex.image = bpy.data.images.load(str(envmap_path), check_existing=True)
    mapping.inputs["Rotation"].default_value = tuple(float(v) for v in rotation_world)
    background.inputs["Strength"].default_value = float(strength)
    node_tree.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    node_tree.links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    node_tree.links.new(env_tex.outputs["Color"], background.inputs["Color"])
    node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])


def _remove_pemoin_lights() -> None:
    """Remove previously created PEMOIN lights before rebuilding the rig."""
    for obj in list(bpy.data.objects):
        if not obj.get(_PEMOIN_LIGHTING_TAG, False):
            continue
        light_data = obj.data if obj.type == "LIGHT" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if light_data is not None and light_data.users == 0:
            bpy.data.lights.remove(light_data)


def _wrap_subject_fill_role_scale(
    role: str,
    settings: WrapSubjectFillSpec | None,
) -> float:
    cfg = settings or WrapSubjectFillSpec()
    role_key = str(role)
    if role_key == "wrap_key_fill":
        return float(
            cfg.wrap_key_role_scale * (1.0 - 0.35 * float(cfg.direct_preservation_bias))
        )
    if role_key == "counter_wrap_fill":
        return float(
            cfg.counter_wrap_role_scale * (1.0 + 0.75 * float(cfg.counter_side_lift_bias))
        )
    if role_key == "sky_fill":
        return float(
            cfg.sky_fill_role_scale * (1.0 + 0.75 * float(cfg.sky_softness_bias))
        )
    return 0.05


def create_light(
    spec: LightSpec,
    *,
    wrap_subject_fill: WrapSubjectFillSpec | None = None,
) -> bpy.types.Object:
    """Create a Blender light from the normalized light spec."""
    realized_energy = float(spec.energy)
    if (
        spec.kind == "POINT"
        and str(spec.transport_mode or "") == _WRAP_SUBJECT_FILL_TRANSPORT_MODE
    ):
        offset = spec.relative_location if spec.relative_location is not None else spec.location
        if offset is not None:
            distance_m = max(
                _WRAP_SUBJECT_POINT_MIN_DISTANCE_M,
                float(np.linalg.norm(np.asarray(offset, dtype=np.float32).reshape(3))),
            )
            role_scale = _wrap_subject_fill_role_scale(str(spec.role), wrap_subject_fill)
            global_scale = float(
                getattr(
                    wrap_subject_fill or WrapSubjectFillSpec(),
                    "global_strength_scale",
                    WrapSubjectFillSpec().global_strength_scale,
                )
            )
            realized_energy = float(
                spec.energy
                * (distance_m ** 2)
                * role_scale
                * global_scale
            )
    light_data = bpy.data.lights.new(name=spec.name, type=spec.kind)
    light_data.energy = realized_energy
    light_data.color = spec.color
    if spec.kind == "SUN":
        light_data.angle = math.radians(float(spec.angle_deg))
    elif spec.kind == "AREA" and spec.area_size is not None:
        light_data.shape = "RECTANGLE"
        light_data.size = float(spec.area_size[0])
        light_data.size_y = float(spec.area_size[1])
    light_data.use_shadow = bool(spec.casts_shadow)
    light_obj = bpy.data.objects.new(spec.name, light_data)
    light_obj.rotation_euler = tuple(
        math.radians(float(v)) for v in spec.rotation_euler_deg
    )
    if spec.location is not None:
        light_obj.location = spec.location
    light_obj[_PEMOIN_LIGHTING_TAG] = True
    light_obj["pemoin_light_role"] = str(spec.role)
    light_obj[_PEMOIN_LIGHT_PLACEMENT_MODE] = str(spec.placement_mode)
    light_obj[_PEMOIN_LIGHT_PLACEMENT_TARGET] = str(spec.placement_target)
    light_obj[_PEMOIN_LIGHT_TRANSPORT_MODE] = (
        "" if spec.transport_mode is None else str(spec.transport_mode)
    )
    light_obj[_PEMOIN_LIGHT_SOURCE_ENERGY] = float(spec.energy)
    light_obj[_PEMOIN_LIGHT_REALIZED_ENERGY] = float(realized_energy)
    if spec.relative_location is not None:
        light_obj[_PEMOIN_LIGHT_RELATIVE_OFFSET] = [float(v) for v in spec.relative_location]
    lighting_collection = ensure_collection("SceneLighting")
    lighting_collection.objects.link(light_obj)
    return light_obj


def _rotation_euler_deg_from_direction(direction_world: Sequence[float]) -> tuple[float, float, float]:
    direction = Vector(direction_world).normalized()
    quat = direction.to_track_quat("-Z", "Y")
    euler = quat.to_euler()
    return tuple(math.degrees(float(v)) for v in euler)


def _light_spec_from_standardized_light(
    light: Any,
    *,
    anchor_world: Sequence[float] | None = None,
) -> LightSpec:
    placement_mode = str(getattr(light, "placement_mode", "world_absolute"))
    placement_target = str(
        getattr(
            light,
            "placement_target",
            "world" if placement_mode == "world_absolute" else "subject_root_dynamic",
        )
    )
    direction = (
        tuple(float(v) for v in np.asarray(light.direction_world, dtype=np.float32).reshape(3))
        if light.direction_world is not None
        else (0.0, 0.0, 1.0)
    )
    rotation = (
        _rotation_euler_deg_from_direction(direction)
        if light.direction_world is not None
        else (
            tuple(float(v) for v in np.asarray(light.rotation_world, dtype=np.float32).reshape(3))
            if light.rotation_world is not None
            else (0.0, 0.0, 0.0)
        )
    )
    diagnostics = getattr(light, "diagnostics", None)
    transport_mode = None
    if isinstance(diagnostics, dict):
        raw_transport_mode = diagnostics.get("transport_mode")
        if raw_transport_mode is not None:
            transport_mode = str(raw_transport_mode)
    area_size = None
    if str(light.kind) == "AREA" and light.area_size is not None:
        area_size = tuple(float(v) for v in np.asarray(light.area_size, dtype=np.float32).reshape(2))
    location = None
    relative_location = None
    if light.location_world is not None:
        base_location = np.asarray(light.location_world, dtype=np.float32).reshape(3)
        if placement_mode == "subject_anchor_relative":
            relative_location = tuple(float(v) for v in base_location)
            if anchor_world is None:
                raise ValueError(
                    f"Light '{light.name}' requires subject anchor placement but no anchor_world was provided."
                )
            base_location = np.asarray(anchor_world, dtype=np.float32).reshape(3) + base_location
        location = tuple(float(v) for v in base_location)
    return LightSpec(
        name=str(light.name),
        kind=str(light.kind),
        energy=float(light.strength),
        rotation_euler_deg=rotation,
        color=tuple(float(v) for v in np.asarray(light.color, dtype=np.float32).reshape(3)),
        role=str(light.role),
        casts_shadow=bool(light.casts_shadow),
        angle_deg=float(light.angular_size_deg or 2.0) if str(light.kind) == "SUN" else None,
        area_size=area_size,
        location=location,
        placement_mode=placement_mode,
        placement_target=placement_target,
        relative_location=relative_location,
        transport_mode=transport_mode,
    )


def _dynamic_subject_light_objects() -> list[bpy.types.Object]:
    dynamic_lights: list[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if not hasattr(obj, "get"):
            continue
        if not obj.get(_PEMOIN_LIGHTING_TAG, False):
            continue
        if str(obj.get(_PEMOIN_LIGHT_PLACEMENT_MODE, "world_absolute")) != "subject_anchor_relative":
            continue
        if str(obj.get(_PEMOIN_LIGHT_PLACEMENT_TARGET, "world")) != "subject_root_dynamic":
            continue
        dynamic_lights.append(obj)
    return dynamic_lights


def _clear_dynamic_light_motion(light_obj: Any) -> None:
    if getattr(light_obj, "animation_data", None) is not None:
        try:
            light_obj.animation_data_clear()
        except Exception:
            pass
    constraints = getattr(light_obj, "constraints", None)
    if constraints is None:
        return
    for constraint in list(constraints):
        if not str(getattr(constraint, "name", "")).startswith("PEMOIN Dynamic Light"):
            continue
        try:
            constraints.remove(constraint)
        except Exception:
            continue


def _sample_actor_anchor_world(
    *,
    actor_root: Any,
    frame_idx: int,
) -> np.ndarray:
    scene = bpy.context.scene
    scene.frame_set(int(frame_idx))
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    actor_eval = actor_root.evaluated_get(deps)
    return np.asarray(actor_eval.matrix_world.translation, dtype=np.float32).reshape(3)


def _bind_dynamic_light_copy_location(
    *,
    light_obj: Any,
    actor_root: Any,
    offset: np.ndarray,
) -> str:
    constraints = getattr(light_obj, "constraints", None)
    if constraints is None or not hasattr(constraints, "new"):
        if hasattr(light_obj, "parent"):
            light_obj.parent = actor_root
            light_obj.location = tuple(float(v) for v in offset.tolist())
            return "parent_fallback"
        light_obj.location = tuple(float(v) for v in offset.tolist())
        return "static_fallback"
    constraint = constraints.new(type="COPY_LOCATION")
    constraint.name = "PEMOIN Dynamic Light Copy Location"
    constraint.target = actor_root
    for attr, value in (
        ("use_offset", True),
        ("use_x", True),
        ("use_y", True),
        ("use_z", True),
    ):
        if hasattr(constraint, attr):
            setattr(constraint, attr, value)
    for attr, value in (
        ("target_space", "WORLD"),
        ("owner_space", "WORLD"),
    ):
        if hasattr(constraint, attr):
            with suppress(Exception):
                setattr(constraint, attr, value)
    light_obj.location = tuple(float(v) for v in offset.tolist())
    return "copy_location_constraint"


def bind_dynamic_subject_lights(
    *,
    actor_name: str,
    frame_indices: Sequence[int],
    binding_mode: str = "copy_location_constraint",
) -> list[dict[str, Any]]:
    dynamic_lights = _dynamic_subject_light_objects()
    if not dynamic_lights:
        return []
    actor_root = bpy.data.objects.get(actor_name)
    if actor_root is None:
        raise ValueError(
            f"Actor root '{actor_name}' not found; cannot bind dynamic subject lights."
        )
    scene = bpy.context.scene
    current_frame = int(scene.frame_current)
    unique_frames = [int(frame) for frame in sorted({int(frame) for frame in frame_indices})]
    if not unique_frames:
        unique_frames = [current_frame]
    for light_obj in dynamic_lights:
        _clear_dynamic_light_motion(light_obj)
    diagnostics: list[dict[str, Any]] = []
    first_frame = unique_frames[0]
    last_frame = unique_frames[-1]
    first_anchor_world = _sample_actor_anchor_world(actor_root=actor_root, frame_idx=first_frame)
    last_anchor_world = (
        np.asarray(first_anchor_world, dtype=np.float32)
        if last_frame == first_frame
        else _sample_actor_anchor_world(actor_root=actor_root, frame_idx=last_frame)
    )
    for light_obj in dynamic_lights:
        offset = np.asarray(
            light_obj.get(_PEMOIN_LIGHT_RELATIVE_OFFSET, (0.0, 0.0, 0.0)),
            dtype=np.float32,
        ).reshape(3)
        entry = {
            "light_name": str(light_obj.name),
            "placement_mode": str(light_obj.get(_PEMOIN_LIGHT_PLACEMENT_MODE, "")),
            "placement_target": str(light_obj.get(_PEMOIN_LIGHT_PLACEMENT_TARGET, "")),
            "relative_offset_world": [float(v) for v in offset.tolist()],
            "frame_start": first_frame,
            "frame_end": last_frame,
            "binding_mode_requested": str(binding_mode),
            "actor_anchor_world_start": [float(v) for v in first_anchor_world.tolist()],
            "actor_anchor_world_end": [float(v) for v in last_anchor_world.tolist()],
            "light_world_start": [float(v) for v in (first_anchor_world + offset).tolist()],
            "light_world_end": [float(v) for v in (last_anchor_world + offset).tolist()],
            "keyframed_frame_count": 0,
        }
        if binding_mode == "spawn_only_static":
            light_obj.location = tuple(float(v) for v in (first_anchor_world + offset).tolist())
            entry["binding_mode"] = "spawn_only_static"
        elif binding_mode == "sparse_keyframes":
            sample_frames = unique_frames if len(unique_frames) <= 3 else [
                unique_frames[0],
                unique_frames[len(unique_frames) // 2],
                unique_frames[-1],
            ]
            for frame_idx in sample_frames:
                anchor_world = _sample_actor_anchor_world(actor_root=actor_root, frame_idx=frame_idx)
                world_location = anchor_world + offset
                light_obj.location = tuple(float(v) for v in world_location.tolist())
                light_obj.keyframe_insert(data_path="location", frame=frame_idx)
            entry["binding_mode"] = "sparse_keyframes"
            entry["keyframed_frame_count"] = int(len(sample_frames))
        else:
            entry["binding_mode"] = _bind_dynamic_light_copy_location(
                light_obj=light_obj,
                actor_root=actor_root,
                offset=offset,
            )
        diagnostics.append(entry)
    scene.frame_set(current_frame)
    bpy.context.view_layer.update()
    return diagnostics


def create_standardized_sun_light(
    *,
    direction_world: Sequence[float],
    strength: float,
    color: Sequence[float],
    angle_deg: float,
) -> bpy.types.Object:
    """Compatibility wrapper that creates one standardized Sun light."""
    return create_light(
        LightSpec(
            name="PEMOINSun",
            kind="SUN",
            energy=float(strength),
            rotation_euler_deg=_rotation_euler_deg_from_direction(direction_world),
            color=tuple(float(v) for v in color[:3]),
            role="direct_key",
            casts_shadow=True,
            angle_deg=float(angle_deg),
        )
    )


def configure_scene_lighting(
    lighting: LightingRigSpec | None,
    *,
    run_dir: Path | None = None,
    anchor_world: Sequence[float] | None = None,
) -> dict[str, bpy.types.Object]:
    """Build the scene-wide lighting rig with balanced multi-direction coverage."""
    _remove_pemoin_lights()
    resolved_lighting = lighting or _default_lighting_rig()
    if run_dir is not None:
        lighting_payload = _standardized_lighting_payload(run_dir)
        if lighting_payload is not None:
            configure_world_envmap(
                Path(lighting_payload.envmap_path),
                lighting_payload.envmap_rotation_world,
                float(lighting_payload.ambient_strength),
            )
            lights = {
                light_spec.name: create_light(
                    light_spec,
                    wrap_subject_fill=resolved_lighting.wrap_subject_fill,
                )
                for light_spec in (
                    _light_spec_from_standardized_light(light, anchor_world=anchor_world)
                    for light in lighting_payload.light_rig
                )
            }
            log_info(
                "Lighting resource: using standardized lighting package "
                f"({lighting_payload.metadata.get('_json_path')}) rig_mode={lighting_payload.rig_mode} "
                f"analytic_lights={len(lighting_payload.light_rig)}."
            )
            return lights
        log_info("Lighting resource: standardized lighting package not found; using fallback rig.")
    lighting = resolved_lighting
    if not lighting.enabled:
        configure_world_ambient(0.0)
        return {}
    configure_world_ambient(lighting.ambient_world_strength)
    lights: dict[str, bpy.types.Object] = {}
    for light_spec in lighting.lights:
        lights[light_spec.name] = create_light(
            light_spec,
            wrap_subject_fill=lighting.wrap_subject_fill,
        )
    return lights


def _preferred_raster_engine() -> str:
    for engine_name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            bpy.context.scene.render.engine = engine_name
            return engine_name
        except Exception:
            continue
    raise RuntimeError(
        "No supported Blender raster render engine was available. "
        "Expected BLENDER_EEVEE_NEXT or BLENDER_EEVEE."
    )


def configure_render_engine(spec: Any) -> None:
    """Configure the single fast raster render backend used for all scene outputs."""
    scene = bpy.context.scene
    render_spec = getattr(spec, "render", None)
    shadow_spec = getattr(spec, "shadow", None)
    performance_spec = getattr(render_spec, "performance", None)
    scene.render.engine = _preferred_raster_engine()
    eevee = getattr(scene, "eevee", None)
    if eevee is None:
        eevee = getattr(scene, "eevee_next", None)
    if eevee is None:
        return
    samples = int(getattr(render_spec, "samples", 16))
    shadow_map_resolution = str(getattr(shadow_spec, "map_resolution", "1024"))
    fast_png_compression = True
    persistent_data = True
    disable_raytracing = True
    disable_volumetric_shadows = True
    disable_volumetric_lighting = True
    disable_bloom = True
    disable_screen_space_reflections = True
    disable_gtao = True
    disable_motion_blur = True
    disable_high_quality_normals = True
    if performance_spec is not None:
        fast_png_compression = bool(getattr(performance_spec, "fast_png_compression", True))
        persistent_data = bool(getattr(performance_spec, "persistent_data", True))
        disable_raytracing = bool(getattr(performance_spec, "disable_raytracing", True))
        disable_volumetric_shadows = bool(
            getattr(performance_spec, "disable_volumetric_shadows", True)
        )
        disable_volumetric_lighting = bool(
            getattr(performance_spec, "disable_volumetric_lighting", True)
        )
        disable_bloom = bool(getattr(performance_spec, "disable_bloom", True))
        disable_screen_space_reflections = bool(
            getattr(performance_spec, "disable_screen_space_reflections", True)
        )
        disable_gtao = bool(getattr(performance_spec, "disable_gtao", True))
        disable_motion_blur = bool(getattr(performance_spec, "disable_motion_blur", True))
        disable_high_quality_normals = bool(
            getattr(performance_spec, "disable_high_quality_normals", True)
        )
    _set_scene_custom_value(scene, "_pemoin_fast_png_compression", bool(fast_png_compression))
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = samples
    if hasattr(eevee, "taa_samples"):
        eevee.taa_samples = samples
    if hasattr(eevee, "use_shadows"):
        eevee.use_shadows = True
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = bool(persistent_data)
    if disable_motion_blur and hasattr(scene.render, "use_motion_blur"):
        with suppress(Exception):
            scene.render.use_motion_blur = False
    if disable_raytracing and hasattr(eevee, "use_raytracing"):
        with suppress(Exception):
            eevee.use_raytracing = False
    if disable_volumetric_shadows and hasattr(eevee, "use_volumetric_shadows"):
        with suppress(Exception):
            eevee.use_volumetric_shadows = False
    if disable_volumetric_lighting and hasattr(eevee, "use_volumetric_lights"):
        with suppress(Exception):
            eevee.use_volumetric_lights = False
    if disable_bloom and hasattr(eevee, "use_bloom"):
        with suppress(Exception):
            eevee.use_bloom = False
    if disable_screen_space_reflections and hasattr(eevee, "use_ssr"):
        with suppress(Exception):
            eevee.use_ssr = False
    if disable_gtao and hasattr(eevee, "use_gtao"):
        with suppress(Exception):
            eevee.use_gtao = False
    if disable_high_quality_normals and hasattr(eevee, "use_high_quality_normals"):
        with suppress(Exception):
            eevee.use_high_quality_normals = False
    if hasattr(eevee, "use_soft_shadows"):
        with suppress(Exception):
            eevee.use_soft_shadows = True
    if hasattr(eevee, "shadow_cube_size"):
        eevee.shadow_cube_size = shadow_map_resolution
    if hasattr(eevee, "shadow_cascade_size"):
        eevee.shadow_cascade_size = shadow_map_resolution


def parse_args(argv: list[str]) -> SceneSpec:
    """Parse CLI arguments into SceneSpec.

    Expected usage:
        blender --background --python script.py -- \
            --run-dir /path/to/run \
            --output /path/to/scene.blend \
            --config /path/to/profiles.json \
            --profile unity_gt_offline
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize PEMOIN trajectory in Blender."
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="outputs/<run> directory"
    )
    parser.add_argument(
        "--trajectory", type=Path, help="Path to trajectory/poses.npz (optional)"
    )
    parser.add_argument("--output", type=Path, help="Output .blend file path")
    parser.add_argument("--config", type=Path, help="Profile JSON path")
    parser.add_argument(
        "--profile", type=str, help="Profile name defined in the config file"
    )
    parser.add_argument(
        "--cube-size", type=float, default=0.1, help="Cube size in meters"
    )
    parser.add_argument(
        "--collection", type=str, default="TrajectoryDebug", help="Collection name"
    )
    parser.add_argument(
        "--road-gap",
        type=float,
        default=0.05,
        help="Gap between consecutive planes in meters",
    )
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
    parser.add_argument(
        "--local-support-snap-max-vertical-delta-m", type=float, default=0.2
    )
    parser.add_argument(
        "--local-support-snap-max-radius-ratio", type=float, default=0.5
    )
    parser.add_argument(
        "--local-support-prefilter-vertical-window-m", type=float, default=0.75
    )
    parser.add_argument(
        "--foot-contact-mode",
        type=str,
        default="mixamo_phase",
        choices=("nearest_plane", "mixamo_phase"),
        help="Foot contact planning mode",
    )
    parser.add_argument("--foot-contact-phase-offset", type=float, default=0.0)
    parser.add_argument("--foot-contact-gait-cycle-frames", type=float)
    parser.add_argument(
        "--foot-contact-left-stance-phase-ranges",
        type=str,
        default="",
        help="Comma-separated phase windows, e.g. '0.05-0.42,0.90-0.10'",
    )
    parser.add_argument(
        "--foot-contact-right-stance-phase-ranges",
        type=str,
        default="",
        help="Comma-separated phase windows, e.g. '0.55-0.92'",
    )
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
    parser.add_argument(
        "--mixamo-character-fbx", type=Path, help="Path to Mixamo character FBX"
    )
    parser.add_argument(
        "--mixamo-animation-fbx", type=Path, help="Path to Mixamo walk animation FBX"
    )
    parser.add_argument(
        "--mixamo-scene-fps", type=float, help="Scene FPS for Mixamo animation"
    )
    parser.add_argument(
        "--mixamo-source-fps",
        type=float,
        default=30.0,
        help="Source FPS authored into the Mixamo animation.",
    )
    parser.add_argument(
        "--actor-name",
        type=str,
        default="Pedestrian01",
        help="Actor root object name used for insertion and grounding",
    )
    parser.add_argument(
        "--pedestrian-trajectory-t",
        type=float,
        default=0.0,
        help="Normalized arc-length position along the camera trajectory (0=first frame, 1=last frame).",
    )
    parser.add_argument(
        "--pedestrian-forward-offset-m",
        type=float,
        default=5.0,
        help="Meters forward along the local camera-motion direction at the chosen trajectory point.",
    )
    parser.add_argument(
        "--pedestrian-left-offset-m",
        type=float,
        default=2.0,
        help="Meters left of the local camera-motion direction at the chosen trajectory point.",
    )
    parser.add_argument(
        "--pedestrian-up-offset-m",
        type=float,
        default=0.0,
        help="Meters upward in world +Z from the grounded trajectory anchor.",
    )
    parser.add_argument(
        "--max-plane-center-xy-distance-m",
        type=float,
        default=None,
        help="Maximum allowed XY distance from the actor to the chosen persisted plane center.",
    )
    parser.add_argument(
        "--pedestrian-heading-deg",
        type=float,
        default=0.0,
        help="Yaw offset in degrees relative to the local camera-motion direction.",
    )
    parser.add_argument(
        "--mixamo-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Mixamo debug artifacts",
    )
    parser.add_argument(
        "--lighting-preset",
        type=str,
        choices=_ALLOWED_LIGHTING_PRESETS,
        help="Lighting rig preset for scene-wide illumination.",
    )
    parser.add_argument(
        "--ambient-world-strength",
        type=float,
        help="Ambient world fill strength used by the lighting rig.",
    )
    parser.add_argument(
        "--shadow-cube-size",
        type=str,
        choices=_ALLOWED_SHADOW_CUBE_SIZES,
        help="EEVEE shadow map resolution for scene lights.",
    )

    args = parser.parse_args(argv)

    # Validation
    if args.road_gap < 0:
        raise ValueError(f"Invalid road-gap: {args.road_gap} (must be non-negative)")
    if args.global_plane_min_range_m < 0 or args.global_plane_range_m <= 0:
        raise ValueError("Global plane ranges must be positive and non-negative.")
    if args.global_plane_min_range_m >= args.global_plane_range_m:
        raise ValueError(
            "global-plane-min-range-m must be smaller than global-plane-range-m."
        )
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
        raise ValueError(
            "local-support-max-radius-m must be >= local-support-radius-m."
        )
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
    if (
        args.foot_contact_gait_cycle_frames is not None
        and args.foot_contact_gait_cycle_frames <= 1
    ):
        raise ValueError("foot-contact-gait-cycle-frames must be > 1.")
    if not (0.0 <= args.foot_contact_min_plane_confidence_for_projection <= 1.0):
        raise ValueError(
            "foot-contact-min-plane-confidence-for-projection must be in [0, 1]."
        )
    if args.foot_contact_max_plane_dist_m < 0:
        raise ValueError("foot-contact-max-plane-dist-m must be >= 0.")
    if args.foot_contact_max_speed_mps < 0:
        raise ValueError("foot-contact-max-speed-mps must be >= 0.")
    if args.foot_contact_min_stance_frames < 1:
        raise ValueError("foot-contact-min-stance-frames must be >= 1.")
    if args.foot_contact_min_swing_frames < 1:
        raise ValueError("foot-contact-min-swing-frames must be >= 1.")
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
        8.0
        if args.max_plane_center_xy_distance_m is None
        else float(args.max_plane_center_xy_distance_m)
    )
    if not np.isfinite(
        np.asarray(
            [pedestrian_forward_offset_m, pedestrian_left_offset_m, pedestrian_up_offset_m],
            dtype=np.float32,
        )
    ).all():
        raise ValueError("Pedestrian offsets must be finite values.")
    if (
        not np.isfinite(max_plane_center_xy_distance_m)
        or max_plane_center_xy_distance_m <= 0.0
    ):
        raise ValueError("max-plane-center-xy-distance-m must be a finite value > 0.")
    pedestrian_heading_deg = float(args.pedestrian_heading_deg)

    run_dir = args.run_dir.expanduser().resolve()
    road_labels = _road_labels_setting(run_dir)
    trajectory_path = (
        args.trajectory.expanduser().resolve() if args.trajectory else None
    )
    output_path = args.output.expanduser().resolve() if args.output else None

    if args.config or args.profile:
        if args.config is None or args.profile is None:
            raise ValueError(
                "Both --config and --profile are required when using profile config."
            )
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
            spec = replace(
                spec,
                max_plane_center_xy_distance_m=max_plane_center_xy_distance_m,
            )
        return spec

    # Resolve trajectory path
    if trajectory_path is None:
        trajectory_path = run_dir / "standard" / "trajectory" / "poses.npz"

    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")

    mixamo_character_fbx_path = (
        args.mixamo_character_fbx.expanduser().resolve()
        if args.mixamo_character_fbx
        else None
    )
    mixamo_animation_fbx_path = (
        args.mixamo_animation_fbx.expanduser().resolve()
        if args.mixamo_animation_fbx
        else None
    )
    mixamo_scene_fps = (
        float(args.mixamo_scene_fps) if args.mixamo_scene_fps is not None else None
    )

    if (
        mixamo_character_fbx_path is None
        or mixamo_animation_fbx_path is None
        or mixamo_scene_fps is None
    ):
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

    return SceneSpec(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        output_path=output_path,
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
        local_support_confidence_threshold=float(
            args.local_support_confidence_threshold
        ),
        local_support_max_radius_m=float(args.local_support_max_radius_m),
        local_support_radius_step_m=float(args.local_support_radius_step_m),
        local_support_snap_to_nearest_road=bool(
            args.local_support_snap_to_nearest_road
        ),
        local_support_snap_radius_m=float(args.local_support_snap_radius_m),
        local_support_temporal_hold_frames=int(
            args.local_support_temporal_hold_frames
        ),
        local_support_temporal_hold_seconds=(
            None
            if args.local_support_temporal_hold_seconds is None
            else float(args.local_support_temporal_hold_seconds)
        ),
        local_support_snap_max_vertical_delta_m=float(
            args.local_support_snap_max_vertical_delta_m
        ),
        local_support_snap_max_radius_ratio=float(
            args.local_support_snap_max_radius_ratio
        ),
        local_support_prefilter_vertical_window_m=float(
            args.local_support_prefilter_vertical_window_m
        ),
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
        lighting=lighting,
    )


def main(argv: list[str]) -> None:
    """Main entry point for Blender trajectory visualization."""
    # Parse arguments
    spec = config_parse_args(argv)

    # Clear scene to factory defaults
    clear_scene()

    # Create collections
    traj_collection = ensure_collection(spec.collection_name)
    global_plane_collection = ensure_collection("RoadPlanesGlobal")
    # Load and visualize trajectory
    c2w, frame_indices = load_trajectory(spec.trajectory_path)
    traj_spec = TrajectorySpec(cube_size=spec.cube_size)
    add_trajectory_cubes(c2w, frame_indices, traj_spec, traj_collection)

    # Load camera intrinsics and create animated camera
    intrinsics_matrix, width, height, intrinsics_metadata = load_intrinsics(
        spec.run_dir,
        frame_indices,
    )
    camera, parity_solution = create_animated_camera(
        c2w_matrices=c2w,
        frame_indices=frame_indices,
        intrinsics_matrix=intrinsics_matrix,
        width=width,
        height=height,
    )
    fx = float(intrinsics_matrix[0, 0])
    fy = float(intrinsics_matrix[1, 1])
    cx = float(intrinsics_matrix[0, 2])
    cy = float(intrinsics_matrix[1, 2])
    log_info(
        f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, width={width}, height={height}"
    )
    log_info(
        "Blender camera parity: "
        f"fit={parity_solution.sensor_fit} "
        f"focal_residual={parity_solution.focal_residual_px:.6f}px "
        f"principal_point_residual={parity_solution.principal_point_residual_px:.6f}px "
        f"resolution_source={intrinsics_metadata.get('intrinsics_resolution_source')}"
    )

    if spec.sampling_fps is not None:
        scene = bpy.context.scene
        fps = float(spec.sampling_fps)
        if fps <= 0:
            raise ValueError(f"Invalid sampling_fps: {fps}")
        scene.render.fps = max(1, int(round(fps)))
        log_info(f"Setting scene FPS to {scene.render.fps} based on sampling_fps={fps}")
        scene.render.fps_base = scene.render.fps / fps

    configure_render_engine(spec)

    (
        resolved_spawn_world_arr,
        trajectory_anchor_world_arr,
        motion_forward_world_arr,
        base_heading_world_deg,
    ) = resolve_pedestrian_spawn_world(
        c2w,
        spec.pedestrian_trajectory_t,
        spec.pedestrian_forward_offset_m,
        spec.pedestrian_left_offset_m,
        spec.pedestrian_up_offset_m,
    )
    spawn_threshold_m = max(10.0, 2.0 * float(spec.global_plane_range_m))
    spawn_min_distance_m = validate_pedestrian_spawn_near_trajectory(
        c2w,
        resolved_spawn_world_arr,
        max_distance_m=spawn_threshold_m,
    )
    resolved_spawn_world = tuple(float(v) for v in resolved_spawn_world_arr.tolist())
    trajectory_anchor_world = tuple(float(v) for v in trajectory_anchor_world_arr.tolist())
    motion_forward_world = tuple(float(v) for v in motion_forward_world_arr.tolist())
    trajectory_heading_world_deg = float(base_heading_world_deg)
    log_info(
        "Resolved pedestrian spawn: "
        f"trajectory_t={spec.pedestrian_trajectory_t:.3f} "
        f"anchor={trajectory_anchor_world} forward={motion_forward_world} "
        f"offsets=(fwd={spec.pedestrian_forward_offset_m:.3f}, "
        f"left={spec.pedestrian_left_offset_m:.3f}, up={spec.pedestrian_up_offset_m:.3f}) "
        f"world={resolved_spawn_world} trajectory_heading_world_deg={trajectory_heading_world_deg:.3f} "
        f"pedestrian_heading_offset_deg={float(spec.pedestrian_heading_deg):.3f} "
        f"min_xy_to_trajectory={spawn_min_distance_m:.3f}m threshold={spawn_threshold_m:.3f}m"
    )
    configure_scene_lighting(
        spec.lighting,
        run_dir=spec.run_dir,
        anchor_world=resolved_spawn_world,
    )

    # Stage 1: insert pedestrian before grounding against persisted planes.
    motion_direction_parity = insert_mixamo_character(
        spec,
        c2w_matrices=c2w,
        frame_indices=frame_indices,
        spawn_world=resolved_spawn_world,
        trajectory_anchor_world=trajectory_anchor_world,
        intended_forward_world=motion_forward_world,
    )

    # Stage 2: load and visualize the already persisted road planes.
    road_surface = viz_road_planes(
        c2w=c2w,
        frame_indices=frame_indices,
        global_plane_collection=global_plane_collection,
        spec=spec,
    )

    # Stage 3: keep the authored XY path and solve only the grounded vertical profile.
    grounding_diagnostics = apply_road_support_to_inserted_pedestrian(
        spec=spec,
        road_surface=road_surface,
        frame_indices=frame_indices,
        actor_name=spec.pedestrian_actor_name,
    )
    lighting_anchor_diagnostics = bind_dynamic_subject_lights(
        actor_name=spec.pedestrian_actor_name,
        frame_indices=frame_indices,
        binding_mode=str(
            getattr(getattr(spec, "render", None), "dynamic_light_binding", "copy_location_constraint")
        ),
    )
    if lighting_anchor_diagnostics:
        lighting_json = _write_dynamic_lighting_anchor_diagnostics(
            run_dir=spec.run_dir,
            diagnostics=lighting_anchor_diagnostics,
        )
        log_info(
            "Lighting anchor diagnostics written: "
            f"json={lighting_json} entries={len(lighting_anchor_diagnostics)}"
        )
    diag_json, diag_csv = _write_grounding_diagnostics(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    log_info(
        "Grounding diagnostics written: "
        f"json={diag_json} csv={diag_csv} entries={len(grounding_diagnostics)}"
    )
    support_json, support_csv = _write_support_surface_diagnostics(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    log_info(
        "Support-surface diagnostics written: "
        f"json={support_json} csv={support_csv} entries={len(grounding_diagnostics)}"
    )
    trajectory_segments_json = _write_trajectory_support_segments(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    trajectory_height_csv = _write_trajectory_height_profile(
        run_dir=spec.run_dir,
        diagnostics=grounding_diagnostics,
    )
    log_info(
        "Trajectory grounding debug written: "
        f"segments={trajectory_segments_json} height_profile={trajectory_height_csv}"
    )
    _write_road_surface_summary(
        spec=spec,
        trajectory_anchor_world=trajectory_anchor_world,
        motion_forward_world=motion_forward_world,
        resolved_spawn_world=resolved_spawn_world,
        base_heading_world_deg=float(trajectory_heading_world_deg),
        resolved_heading_world_deg=(
            None
            if motion_direction_parity is None
            else motion_direction_parity.get("resolved_root_yaw_world_deg")
        ),
        spawn_min_distance_to_trajectory_m=spawn_min_distance_m,
        global_planes=road_surface.global_planes,
        grounding_diagnostics=grounding_diagnostics,
        motion_direction_parity=motion_direction_parity,
    )
    _raise_for_grounding_failures(
        diagnostics=grounding_diagnostics,
        max_residual_m=float(spec.foot_contact_max_plane_dist_m),
        max_plane_center_xy_distance_m=float(spec.max_plane_center_xy_distance_m),
    )

    # Render frames after persisted-plane grounding.
    pedestrian_frames_dir = render_pedestrian(
        spec,
        render_width=width,
        render_height=height,
        target_intrinsics=intrinsics_matrix,
        parity_solution=parity_solution,
        grounding_diagnostics=grounding_diagnostics,
    )
    pedestrian_depth_frames_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "pedestrian_depth_frames",
    )
    shadow_frames_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "shadow_frames",
    )
    overlay_frames_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "overlayed_frames",
    )
    overlay_support_local_grid_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "overlayed_frames_support_local_grid",
    )
    occlusion_masks_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "occlusion_masks",
    )
    occlusion_debug_dir = ResourceStore.blender_artifact_dir_for(
        spec.run_dir,
        "occlusion_debug",
    )
    compose_overlay_frames(
        run_dir=spec.run_dir,
        actor_name=spec.pedestrian_actor_name,
        road_labels=spec.road_labels,
        contact_ground_labels=getattr(getattr(spec, "occlusion", None), "contact_ground_labels", None),
        occlusion_spec=getattr(spec, "occlusion", None),
        shadow_spec=getattr(spec, "shadow", None),
        grounding_diagnostics=grounding_diagnostics,
        original_frames_dir=spec.run_dir / "standard" / "frames",
        pedestrian_frames_dir=pedestrian_frames_dir,
        pedestrian_depth_frames_dir=pedestrian_depth_frames_dir,
        shadow_frames_dir=shadow_frames_dir,
        output_dir=overlay_frames_dir,
        debug_output_dir=None,
        support_debug_output_dir=None,
        support_local_grid_output_dir=overlay_support_local_grid_dir,
        occlusion_mask_output_dir=occlusion_masks_dir,
        occlusion_debug_output_dir=occlusion_debug_dir,
    )

    # Save scene
    if spec.output_path:
        save_blend(spec.output_path)
        log_info(f"Scene saved to {spec.output_path}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    try:
        main(argv)
    except Exception as exc:
        log_error(str(exc))
        traceback.print_exc()
        raise SystemExit(1) from exc
