from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_DEFAULT_ROLE_DEFAULTS = {
    "road": ("road", "path", "crosswalk"),
    "sky": ("sky",),
    "mobile": ("person", "human", "car", "bus", "truck", "bicycle", "motorcycle"),
    "large_vehicle": ("bus", "truck"),
}
_PRIMARY_KEYFRAMES = 5
_RECOVERY_KEYFRAMES = 3
_MIN_RECOVERY_INPUT_SIZE = 640
_MAX_SUN_CANDIDATES = 2
_CAMERA_CLUSTER_SUPPORT_DEG = 35.0
_CAMERA_CLUSTER_MEAN_DEG = 25.0
_WORLD_CLUSTER_MEAN_DEG = 30.0
_WORLD_CLUSTER_MAX_DEG = 40.0
_MIN_CLUSTER_FRAMES = 2
_DEFAULT_FILL_LIGHT_COUNT = 3
_FILL_LIGHT_MIN_SEPARATION_DEG = 55.0
_FILL_LIGHT_DISTANCE_M = 7.5
_FILL_LIGHT_AREA_SIZE_M = np.asarray((12.0, 12.0), dtype=np.float32)
_WRAP_FILL_FRONT_OFFSET_M = np.asarray((5.2, 1.5, 2.8), dtype=np.float32)
_WRAP_FILL_COUNTER_OFFSET_M = np.asarray((2.8, -4.0, 2.4), dtype=np.float32)
_WRAP_FILL_SKY_OFFSET_M = np.asarray((1.2, 0.0, 4.8), dtype=np.float32)
_VIEW_FILL_PRIMARY_BLEND = 0.72
_VIEW_FILL_SECONDARY_VIEW_BLEND = 0.55
_MIN_FILL_HEAVY_DARK_TO_BRIGHT_RATIO = 0.38
_MAX_DIFFUSE_WORLD_STRENGTH = 0.35
_WRAP_GEOMETRY_MIN_AZIMUTH_SEPARATION_DEG = 55.0
_WRAP_GEOMETRY_COUNTER_OPPOSITION_DEG = 110.0
_WRAP_GEOMETRY_SKY_MIN_ELEVATION_DEG = 55.0
_WRAP_GEOMETRY_CANDIDATE_COUNT_PER_ROLE = 3
_DIFFUSE_DEMOTE_MODES = frozenset({"off", "moderate", "aggressive"})
_DLT_INFERENCE_CACHE_PROVIDER_ID = "lighting_dlt_inference"
_MIN_FILL_HEAVY_BRIGHTNESS_PRESERVATION = 0.85

_VALIDATION_LIMITS = {
    "min_mean_luminance": 1e-4,
    "min_p95_luminance": 5e-4,
    "min_max_luminance": 5e-3,
    "min_dynamic_range_ratio": 4.0,
    "min_relative_p95_ratio": 0.25,
    "min_sun_strength": 0.05,
    "max_sun_strength": 50.0,
    "min_ambient_strength": 0.01,
    "max_ambient_strength": 10.0,
}
_RECOVERABLE_ENV_FALLBACK_LIMITS = {
    "min_dynamic_range_ratio": 3.0,
    "min_quality_envmap": 0.65,
}
_RECOVERABLE_LIGHTING_FAILURES = frozenset(
    {
        "dynamic_range_too_low",
        "ineffective_subject_fill_transport",
    }
)
_CARLA_GT_SUN_ANGULAR_SIZE_DEG = 0.53
_UNITY_GT_SUN_ANGULAR_SIZE_DEG = 0.53
_UNITY_REFLECTION_FACE_ORDER = (
    "PositiveX",
    "NegativeX",
    "PositiveY",
    "NegativeY",
    "PositiveZ",
    "NegativeZ",
)


def _rounded_input_size(size: int) -> int:
    rounded = int(round(float(size) / 64.0) * 64)
    return max(256, rounded)


def _normalize_diffuse_demote_aggressiveness(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _DIFFUSE_DEMOTE_MODES:
        raise ValueError(
            "DiffusionLightTurboLightingProvider.diffuse_demote_aggressiveness must be "
            f"one of {sorted(_DIFFUSE_DEMOTE_MODES)}; got {value!r}."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class DiffusionLightTurboSettings:
    repo_root: Path
    conda_env: str
    env_manager: str | None
    hf_home: str | None
    allow_online_model_fetch: bool
    input_size: int
    recovery_input_size: int
    primary_keyframes: int
    recovery_keyframes: int
    algorithm: str
    offload: bool
    no_controlnet: bool
    sun_sigma_deg: float
    max_fill_lights: int
    diffuse_demote_enabled: bool
    diffuse_demote_aggressiveness: str
    max_direct_to_fill_ratio_for_diffuse: float
    fill_heavy_min_fill_count: int
    fill_heavy_direct_scale: float
    diffuse_softness_bias: float
    fill_heavy_dark_side_target_ratio: float
    fill_heavy_transport_gain: float
    wrap_geometry_min_azimuth_separation_deg: float
    wrap_geometry_counter_opposition_deg: float
    wrap_geometry_sky_min_elevation_deg: float
    wrap_geometry_candidate_count_per_role: int
    sdxl_model: str
    sdxl_vae_model: str
    sdxl_controlnet_model: str
    depth_estimator_model: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DiffusionLightTurboSettings":
        repo_root = Path(str(mapping.get("repo_root", "tools/DiffusionLight-Turbo"))).expanduser()
        if not repo_root.is_absolute():
            repo_root = (Path.cwd() / repo_root).resolve()
        configured_keyframes = int(mapping.get("num_keyframes", _PRIMARY_KEYFRAMES))
        primary_keyframes = max(_PRIMARY_KEYFRAMES, configured_keyframes)
        recovery_keyframes = int(mapping.get("recovery_keyframes", _RECOVERY_KEYFRAMES))
        input_size = int(mapping.get("input_size", 1024))
        recovery_input_size = int(
            mapping.get(
                "recovery_input_size",
                _rounded_input_size(max(_MIN_RECOVERY_INPUT_SIZE, int(round(input_size * 0.75)))),
            )
        )
        return cls(
            repo_root=repo_root,
            conda_env=str(mapping.get("conda_env", "diffusionlight-turbo")),
            env_manager=str(mapping["env_manager"]) if mapping.get("env_manager") else None,
            hf_home=str(mapping["hf_home"]) if mapping.get("hf_home") else None,
            allow_online_model_fetch=bool(mapping.get("allow_online_model_fetch", False)),
            input_size=input_size,
            recovery_input_size=min(input_size, _rounded_input_size(recovery_input_size)),
            primary_keyframes=max(3, primary_keyframes),
            recovery_keyframes=max(1, min(recovery_keyframes, primary_keyframes)),
            algorithm=str(mapping.get("algorithm", "normal")),
            offload=bool(mapping.get("offload", True)),
            no_controlnet=bool(mapping.get("no_controlnet", True)),
            sun_sigma_deg=float(mapping.get("sun_sigma_deg", 6.0)),
            max_fill_lights=max(0, int(mapping.get("max_fill_lights", _DEFAULT_FILL_LIGHT_COUNT))),
            diffuse_demote_enabled=bool(mapping.get("diffuse_demote_enabled", True)),
            diffuse_demote_aggressiveness=_normalize_diffuse_demote_aggressiveness(
                mapping.get("diffuse_demote_aggressiveness", "aggressive")
            ),
            max_direct_to_fill_ratio_for_diffuse=max(
                0.25,
                float(mapping.get("max_direct_to_fill_ratio_for_diffuse", 2.0)),
            ),
            fill_heavy_min_fill_count=max(1, int(mapping.get("fill_heavy_min_fill_count", 2))),
            fill_heavy_direct_scale=float(
                np.clip(float(mapping.get("fill_heavy_direct_scale", 0.2)), 0.0, 1.0)
            ),
            diffuse_softness_bias=float(
                np.clip(float(mapping.get("diffuse_softness_bias", 0.6)), 0.0, 1.0)
            ),
            fill_heavy_dark_side_target_ratio=float(
                np.clip(
                    float(
                        mapping.get(
                            "fill_heavy_dark_side_target_ratio",
                            _MIN_FILL_HEAVY_DARK_TO_BRIGHT_RATIO,
                        )
                    ),
                    0.1,
                    0.95,
                )
            ),
            fill_heavy_transport_gain=float(
                np.clip(float(mapping.get("fill_heavy_transport_gain", 1.35)), 1.0, 3.0)
            ),
            wrap_geometry_min_azimuth_separation_deg=float(
                np.clip(
                    float(
                        mapping.get(
                            "wrap_geometry_min_azimuth_separation_deg",
                            _WRAP_GEOMETRY_MIN_AZIMUTH_SEPARATION_DEG,
                        )
                    ),
                    10.0,
                    180.0,
                )
            ),
            wrap_geometry_counter_opposition_deg=float(
                np.clip(
                    float(
                        mapping.get(
                            "wrap_geometry_counter_opposition_deg",
                            _WRAP_GEOMETRY_COUNTER_OPPOSITION_DEG,
                        )
                    ),
                    30.0,
                    180.0,
                )
            ),
            wrap_geometry_sky_min_elevation_deg=float(
                np.clip(
                    float(
                        mapping.get(
                            "wrap_geometry_sky_min_elevation_deg",
                            _WRAP_GEOMETRY_SKY_MIN_ELEVATION_DEG,
                        )
                    ),
                    5.0,
                    89.0,
                )
            ),
            wrap_geometry_candidate_count_per_role=max(
                1,
                min(
                    int(
                        mapping.get(
                            "wrap_geometry_candidate_count_per_role",
                            _WRAP_GEOMETRY_CANDIDATE_COUNT_PER_ROLE,
                        )
                    ),
                    6,
                ),
            ),
            sdxl_model=str(mapping.get("sdxl_model", "stabilityai/stable-diffusion-xl-base-1.0")),
            sdxl_vae_model=str(mapping.get("sdxl_vae_model", "madebyollin/sdxl-vae-fp16-fix")),
            sdxl_controlnet_model=str(
                mapping.get("sdxl_controlnet_model", "diffusers/controlnet-depth-sdxl-1.0")
            ),
            depth_estimator_model=str(
                mapping.get("depth_estimator_model", "Intel/dpt-hybrid-midas")
            ),
        )


@dataclass(frozen=True, slots=True)
class CarlaGTLightingSettings:
    require_scene_lights: bool
    envmap_height: int
    envmap_width: int
    sun_strength_scale: float
    ambient_strength_scale: float
    scene_light_strength_scale: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CarlaGTLightingSettings":
        resolution = mapping.get("synthetic_envmap_resolution", (128, 256))
        if not isinstance(resolution, Sequence) or len(resolution) != 2:
            raise ValueError(
                "CarlaGTLightingProvider.synthetic_envmap_resolution must be a two-element sequence."
            )
        return cls(
            require_scene_lights=bool(mapping.get("require_scene_lights", True)),
            envmap_height=max(16, int(resolution[0])),
            envmap_width=max(32, int(resolution[1])),
            sun_strength_scale=float(mapping.get("sun_strength_scale", 1.0)),
            ambient_strength_scale=float(mapping.get("ambient_strength_scale", 1.0)),
            scene_light_strength_scale=float(mapping.get("scene_light_strength_scale", 0.0035)),
        )


@dataclass(frozen=True, slots=True)
class UnityGTLightingSettings:
    require_reflection_faces: bool
    require_frame_lighting: bool
    force_sun_shadows: bool
    sun_strength_scale: float
    ambient_strength_scale: float
    envmap_height: int
    envmap_width: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "UnityGTLightingSettings":
        resolution = mapping.get("envmap_resolution", (256, 512))
        if not isinstance(resolution, Sequence) or len(resolution) != 2:
            raise ValueError(
                "UnityGTLightingProvider.envmap_resolution must be a two-element sequence."
            )
        return cls(
            require_reflection_faces=bool(mapping.get("require_reflection_faces", True)),
            require_frame_lighting=bool(mapping.get("require_frame_lighting", False)),
            force_sun_shadows=bool(mapping.get("force_sun_shadows", True)),
            sun_strength_scale=float(mapping.get("sun_strength_scale", 2.5e-5)),
            ambient_strength_scale=float(mapping.get("ambient_strength_scale", 0.0025)),
            envmap_height=max(32, int(resolution[0])),
            envmap_width=max(64, int(resolution[1])),
        )
