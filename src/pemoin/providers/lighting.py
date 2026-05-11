"""Clip-level lighting provider backed by DiffusionLight-Turbo."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from pemoin.data.carla import resolve_carla_dataset
from pemoin.data.contracts import LightingData, LightingLightData, ResourceKind, ResourceStore
from pemoin.data.unity import resolve_unity_lighting_dataset
from pemoin.providers.base import Provider, ProviderExecutionMode
from pemoin.providers.factory import ProviderFactory
from pemoin.providers.lighting_settings import (
    _CAMERA_CLUSTER_MEAN_DEG,
    _CAMERA_CLUSTER_SUPPORT_DEG,
    _CARLA_GT_SUN_ANGULAR_SIZE_DEG,
    _DEFAULT_ROLE_DEFAULTS,
    _DLT_INFERENCE_CACHE_PROVIDER_ID,
    _FILL_LIGHT_AREA_SIZE_M,
    _FILL_LIGHT_DISTANCE_M,
    _FILL_LIGHT_MIN_SEPARATION_DEG,
    _MAX_DIFFUSE_WORLD_STRENGTH,
    _MAX_SUN_CANDIDATES,
    _MIN_CLUSTER_FRAMES,
    _MIN_FILL_HEAVY_BRIGHTNESS_PRESERVATION,
    _RECOVERABLE_ENV_FALLBACK_LIMITS,
    _RECOVERABLE_LIGHTING_FAILURES,
    _UNITY_GT_SUN_ANGULAR_SIZE_DEG,
    _UNITY_REFLECTION_FACE_ORDER,
    _VALIDATION_LIMITS,
    _WORLD_CLUSTER_MAX_DEG,
    _WORLD_CLUSTER_MEAN_DEG,
    _WRAP_FILL_COUNTER_OFFSET_M,
    _WRAP_FILL_FRONT_OFFSET_M,
    _WRAP_FILL_SKY_OFFSET_M,
    CarlaGTLightingSettings,
    DiffusionLightTurboSettings,
    UnityGTLightingSettings,
)
from pemoin.providers.semantic_roles import (
    SEMANTIC_ROLES_METADATA_KEY,
    build_semantic_roles,
)
from pemoin.runtime.cache import CrossRunCacheManager
from pemoin.utils.env_launcher import resolve_env_launcher
from pemoin.utils.model_cache import (
    configure_hf_subprocess_env,
    has_cached_repo,
    hub_cache_dir,
    transformers_cache_dir,
)

LOG = logging.getLogger(__name__)


class CarlaGTLightingProvider(Provider):
    """Batch provider that converts exported CARLA lighting GT into the standard lighting package."""

    batch_oriented = True
    execution_mode = ProviderExecutionMode.BATCH
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.LIGHTING})

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = CarlaGTLightingSettings.from_mapping(settings)

    def setup(self, context: MutableMapping[str, Any]) -> None:
        return None

    def teardown(self) -> None:
        return None

    def run(
        self, resources: ResourceStore, context: MutableMapping[str, object] | None = None
    ) -> None:
        self.validate_requirements(resources)
        if context is None:
            raise RuntimeError("CarlaGTLightingProvider requires runtime context.")
        dataset = resolve_carla_dataset({}, context)
        if not dataset.has_lighting_gt():
            raise FileNotFoundError(
                "CarlaGTLightingProvider requires lighting_gt metadata in the CARLA export."
            )
        run_lighting = dataset.run_lighting()
        weather = run_lighting.get("weather")
        if not isinstance(weather, Mapping):
            raise ValueError("CARLA run_lighting.json is missing weather.")
        scene_lights = dataset.scene_lights()
        lights_raw = scene_lights.get("lights")
        if self.settings.require_scene_lights and not isinstance(lights_raw, list):
            raise ValueError("CARLA scene_lights.json is missing lights[].")
        sun_altitude_angle = float(weather.get("sun_altitude_angle", -90.0))
        sun_direction = _carla_sun_direction(weather)
        sun_strength = _carla_sun_strength(weather, self.settings.sun_strength_scale)
        sun_color = _carla_sun_color(weather)
        ambient_strength = _carla_ambient_strength(weather, self.settings.ambient_strength_scale)
        analytic_lights: list[LightingLightData] = []
        if sun_strength > 0.0:
            analytic_lights.append(
                LightingLightData(
                    name="CARLADirectSun",
                    kind="SUN",
                    role="direct_key",
                    strength=float(sun_strength),
                    color=np.asarray(sun_color, dtype=np.float32).reshape(3),
                    casts_shadow=True,
                    direction_world=np.asarray(sun_direction, dtype=np.float32).reshape(3),
                    angular_size_deg=float(_CARLA_GT_SUN_ANGULAR_SIZE_DEG),
                    confidence=1.0,
                    diagnostics={
                        "source": "carla_weather",
                        "weather_driven": True,
                        "sun_altitude_angle": sun_altitude_angle,
                    },
                )
            )
        analytic_lights.extend(
            _carla_scene_light_rig(
            scene_lights=scene_lights,
            scene_light_strength_scale=self.settings.scene_light_strength_scale,
            sun_altitude_angle=sun_altitude_angle,
            )
        )
        provider_dir = resources.provider_dir("lighting")
        envmap_dir = provider_dir / "carla_gt_envmap"
        if envmap_dir.exists():
            shutil.rmtree(envmap_dir)
        envmap_dir.mkdir(parents=True, exist_ok=True)
        envmap_path = envmap_dir / "synthetic.exr"
        hdr = _carla_synthetic_envmap(
            weather=weather,
            ambient_strength=ambient_strength,
            height=self.settings.envmap_height,
            width=self.settings.envmap_width,
        )
        _write_hdr_envmap(envmap_path, hdr)
        preview = _tonemap_preview(hdr)
        imageio.imwrite(envmap_dir / "synthetic_preview.png", preview)
        frame_indices = dataset.frame_indices()
        resources.save_lighting(
            LightingData(
                sun_direction_world=sun_direction,
                sun_strength=float(sun_strength),
                sun_color=sun_color,
                mode="full_sun" if sun_strength > 0.0 else "ambient_only",
                envmap_path=str(envmap_path),
                envmap_rotation_world=np.zeros((3,), dtype=np.float32),
                ambient_strength=float(ambient_strength),
                schema_version=2,
                rig_mode=(
                    "sun_plus_fill"
                    if sun_strength > 0.0
                    else ("analytic_rig" if analytic_lights else "envmap_only")
                ),
                light_rig=analytic_lights,
                decomposition={
                    "method": "carla_gt_v1",
                    "analytic_light_count": int(len(analytic_lights)),
                    "scene_light_count": int(
                        sum(1 for light in analytic_lights if str(light.role) != "direct_key")
                    ),
                    "daylight_source": "carla_weather",
                    "direct_sun_mode": "analytic_sun_light" if sun_strength > 0.0 else "disabled_below_horizon",
                    "envmap_mode": "sky_ambient_only",
                },
                quality={"sun": 1.0, "envmap": 1.0, "scene_lights": float(bool(analytic_lights))},
                sun_diagnostics={
                    "source": "carla_weather",
                    "sun_altitude_angle": sun_altitude_angle,
                    "sun_azimuth_angle": float(weather.get("sun_azimuth_angle", 0.0)),
                    "analytic_sun_emitted": bool(sun_strength > 0.0),
                },
                validation={
                    "passed": True,
                    "ambient_passed": True,
                    "sun_passed": bool(sun_strength > 0.0),
                    "ambient_failures": [],
                    "sun_failures": [],
                },
                recovery={"used": False, "reason": None},
                selected_frame_indices=[int(idx) for idx in frame_indices],
                per_keyframe_diagnostics=[
                    {
                        "frame_index": int(idx),
                        "weather": dataset.frame_lighting(int(idx)).get("weather", {}),
                    }
                    for idx in frame_indices
                ],
                metadata={
                    "provider": "CarlaGTLightingProvider",
                    "source": "carla_gt",
                    "town": str(run_lighting.get("town", "")),
                },
            )
        )


class UnityGTLightingProvider(Provider):
    """Batch provider that converts exported Unity lighting GT into the standard lighting package."""

    batch_oriented = True
    execution_mode = ProviderExecutionMode.BATCH
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.LIGHTING})

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = UnityGTLightingSettings.from_mapping(settings)
        self._source_settings = dict(settings)

    def setup(self, context: MutableMapping[str, Any]) -> None:
        return None

    def teardown(self) -> None:
        return None

    def run(
        self, resources: ResourceStore, context: MutableMapping[str, object] | None = None
    ) -> None:
        self.validate_requirements(resources)
        if context is None:
            raise RuntimeError("UnityGTLightingProvider requires runtime context.")
        dataset = resolve_unity_lighting_dataset(self._source_settings, context)
        if not dataset.has_lighting_gt():
            raise FileNotFoundError(
                "UnityGTLightingProvider requires lighting_gt metadata and reflection probe faces in the Unity export."
            )
        if self.settings.require_frame_lighting and not dataset.has_frame_lighting():
            raise FileNotFoundError(
                "UnityGTLightingProvider requires frame_lighting.jsonl when require_frame_lighting=true."
            )
        reflection_faces = dataset.reflection_faces()
        if self.settings.require_reflection_faces:
            missing_faces = [name for name in _UNITY_REFLECTION_FACE_ORDER if name not in reflection_faces]
            if missing_faces:
                raise FileNotFoundError(
                    "UnityGTLightingProvider reflection_probe_faces is incomplete. "
                    f"Missing faces: {', '.join(missing_faces)}."
                )

        run_lighting = dataset.run_lighting()
        scene_lights = dataset.scene_lights()
        provider_dir = resources.provider_dir("lighting")
        envmap_dir = provider_dir / "unity_gt_envmap"
        if envmap_dir.exists():
            shutil.rmtree(envmap_dir)
        envmap_dir.mkdir(parents=True, exist_ok=True)

        envmap_hdr = _unity_reflection_faces_to_latlong(
            reflection_faces=reflection_faces,
            height=self.settings.envmap_height,
            width=self.settings.envmap_width,
        )
        envmap_validation = _validate_unity_envmap(envmap_hdr)
        if not envmap_validation["passed"]:
            raise RuntimeError(
                "UnityGTLightingProvider reflection envmap failed validation: "
                f"{', '.join(envmap_validation['failures'])}."
            )
        envmap_path = envmap_dir / "reflection_probe_latlong.exr"
        _write_hdr_envmap(envmap_path, envmap_hdr)
        imageio.imwrite(envmap_dir / "reflection_probe_preview.png", _tonemap_preview(envmap_hdr))

        main_light = scene_lights.get("mainDirectionalLight", {})
        if main_light is not None and not isinstance(main_light, Mapping):
            raise ValueError("Unity scene_lights.json mainDirectionalLight must be an object.")
        main_light = main_light if isinstance(main_light, Mapping) else {}

        analytic_lights: list[LightingLightData] = []
        sun_direction = _normalize(
            np.asarray(main_light.get("directionWorld", (0.0, 0.0, 1.0)), dtype=np.float32).reshape(3)
        )
        sun_color = _normalize_color(
            np.asarray(main_light.get("colorLinear", (1.0, 1.0, 1.0)), dtype=np.float32).reshape(3)
        )
        sun_strength = _unity_sun_strength(main_light, self.settings.sun_strength_scale)
        sun_enabled = bool(main_light.get("enabled", False)) and sun_strength > 0.0
        if sun_enabled:
            analytic_lights.append(
                LightingLightData(
                    name=str(main_light.get("name", "UnityDirectionalSun")),
                    kind="SUN",
                    role="direct_key",
                    strength=float(sun_strength),
                    color=sun_color,
                    casts_shadow=bool(main_light.get("castsShadows", False) or self.settings.force_sun_shadows),
                    direction_world=sun_direction,
                    angular_size_deg=float(_UNITY_GT_SUN_ANGULAR_SIZE_DEG),
                    confidence=1.0,
                    diagnostics={
                        "source": "unity_directional_light",
                        "unity_intensity": float(main_light.get("intensity", 0.0) or 0.0),
                        "unity_indirect_multiplier": float(main_light.get("indirectMultiplier", 1.0) or 1.0),
                        "unity_color_temperature": float(main_light.get("colorTemperature", 6500.0) or 6500.0),
                        "force_sun_shadows": bool(self.settings.force_sun_shadows),
                    },
                )
            )
        else:
            sun_strength = 0.0

        ambient_source, ambient_coefficients, ambient_sample = _unity_resolve_ambient_probe(dataset)
        ambient_strength = _unity_ambient_strength(
            ambient_coefficients,
            scale=self.settings.ambient_strength_scale,
        )
        frame_indices = dataset.frame_indices()
        selected_indices = frame_indices if frame_indices else list(resources.frame_indices(ResourceKind.FRAMES))

        resources.save_lighting(
            LightingData(
                sun_direction_world=sun_direction,
                sun_strength=float(sun_strength),
                sun_color=sun_color,
                mode="full_sun" if sun_enabled else "ambient_only",
                envmap_path=str(envmap_path),
                envmap_rotation_world=np.zeros((3,), dtype=np.float32),
                ambient_strength=float(ambient_strength),
                schema_version=2,
                rig_mode="sun_plus_fill" if sun_enabled else "envmap_only",
                light_rig=analytic_lights,
                decomposition={
                    "method": "unity_gt_v1",
                    "analytic_light_count": int(len(analytic_lights)),
                    "direct_sun_mode": "analytic_sun_light" if sun_enabled else "disabled_or_missing",
                    "envmap_mode": "reflection_probe_latlong",
                    "diffuse_mode": "ambient_probe_scalar_only",
                    "ambient_source": ambient_source,
                },
                quality={
                    "sun": 1.0 if sun_enabled else 0.0,
                    "envmap": float(envmap_validation["quality_envmap"]),
                    "ambient_probe": 1.0 if ambient_coefficients.size == 27 else 0.0,
                },
                sun_diagnostics={
                    "source": "unity_directional_light",
                    "analytic_sun_emitted": bool(sun_enabled),
                    "unity_reflection_source": str(run_lighting.get("reflectionSource", "")),
                    "reflection_probe_name": str(run_lighting.get("reflectionProbeName", "")),
                },
                validation={
                    "passed": True,
                    "ambient_passed": True,
                    "sun_passed": bool(sun_enabled),
                    "ambient_failures": [],
                    "sun_failures": [],
                    "envmap_failures": [],
                },
                recovery={"used": False, "reason": None},
                selected_frame_indices=[int(idx) for idx in selected_indices],
                per_keyframe_diagnostics=[
                    {
                        "frame_index": int(idx),
                        "reflection_probe_name": str(dataset.frame_lighting(int(idx)).get("reflectionProbeName", "")),
                        "ambient_probe_source": ambient_source,
                    }
                    for idx in frame_indices
                ],
                metadata={
                    "provider": "UnityGTLightingProvider",
                    "source": "unity_gt",
                    "unity_pipeline": str(run_lighting.get("pipeline", "")),
                    "unity_version": str(run_lighting.get("unityVersion", "")),
                    "scene_name": str(run_lighting.get("sceneName", "")),
                    "reflection_probe_name": str(run_lighting.get("reflectionProbeName", "")),
                    "ambient_probe_sample_label": None if ambient_sample is None else str(ambient_sample.get("label", "")),
                },
            )
        )


class DiffusionLightTurboLightingProvider(Provider):
    """Batch provider that estimates one validated lighting package per clip."""

    batch_oriented = True
    execution_mode = ProviderExecutionMode.BATCH
    required_resources = frozenset(
        {
            ResourceKind.FRAMES,
            ResourceKind.DEPTH,
            ResourceKind.TRAJECTORY,
            ResourceKind.SEMANTICS_2D,
        }
    )
    produced_resources = frozenset({ResourceKind.LIGHTING})

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = DiffusionLightTurboSettings.from_mapping(settings)
        self._cache_manager: CrossRunCacheManager | None = None
        self._profile_name: str | None = None
        self._cache_signature: str | None = None
        self._cache_payload: dict[str, Any] | None = None
        self._inference_cache_statuses: list[dict[str, Any]] = []
        self._cache_enabled = False
        self._cache_status: dict[str, Any] = {
            "cross_run_cache_enabled": False,
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled",
            "dlt_inference_cache_enabled": False,
            "dlt_inference_cache_attempts": [],
        }

    def setup(self, context: MutableMapping[str, Any]) -> None:
        cache_manager = context.get("cross_run_cache")
        stage_settings = context.get("cross_run_cache_stage_settings")
        stage_enabled = True
        if isinstance(stage_settings, Mapping):
            lighting_settings = stage_settings.get("lighting")
            if isinstance(lighting_settings, Mapping) and "enabled" in lighting_settings:
                stage_enabled = bool(lighting_settings.get("enabled"))
        self._cache_manager = (
            cache_manager if isinstance(cache_manager, CrossRunCacheManager) else None
        )
        self._cache_enabled = bool(
            self._cache_manager is not None and self._cache_manager.enabled and stage_enabled
        )
        self._profile_name = (
            str(context.get("profile_name"))
            if context.get("profile_name") is not None
            else None
        )
        self._cache_status = {
            "cross_run_cache_enabled": self._cache_enabled,
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled" if not self._cache_enabled else "not-checked",
            "dlt_inference_cache_enabled": self._cache_enabled,
            "dlt_inference_cache_attempts": [],
        }
        self._inference_cache_statuses = []

    def teardown(self) -> None:
        return None

    def run(
        self, resources: ResourceStore, context: MutableMapping[str, object] | None = None
    ) -> None:
        self.validate_requirements(resources)
        if not self.settings.repo_root.exists():
            raise FileNotFoundError(
                f"DiffusionLight-Turbo repository not found at '{self.settings.repo_root}'."
            )
        provider_dir = resources.provider_dir("lighting")
        self._cache_payload = self._cross_run_payload(resources)
        if self._cache_payload is not None and self._cache_manager is not None:
            self._cache_signature = self._cache_manager.signature("lighting", self._cache_payload)
            lookup = self._cache_manager.lookup(
                "lighting",
                self._cache_signature,
                required_relpaths=[
                    "standard/lighting/lighting.json",
                    "standard/lighting/envmap.exr",
                ],
            )
            self._cache_status.update(
                {
                    "cross_run_cache_signature": self._cache_signature,
                    "cross_run_cache_hit": lookup.hit,
                    "cross_run_cache_entry": str(lookup.entry_dir),
                    "cross_run_cache_validation": lookup.reason,
                }
            )
            if lookup.hit:
                materialized = self._cache_manager.materialize(
                    "lighting",
                    self._cache_signature,
                    run_root=resources.root,
                )
                self._cache_status["cross_run_cache_materialized"] = materialized
                LOG.info("Reused cross-run lighting cache at '%s'.", lookup.entry_dir)
                return
            self._cache_status["cross_run_cache_reason"] = lookup.reason
        scored = self._score_frames(resources)
        primary_selected = self._select_keyframes(scored, count=self.settings.primary_keyframes)
        attempts: list[dict[str, Any]] = []

        primary = self._run_attempt(
            resources=resources,
            selected=primary_selected,
            provider_dir=provider_dir,
            attempt_name="primary",
            input_size=self.settings.input_size,
            no_controlnet=self.settings.no_controlnet,
        )
        attempts.append(primary)

        if primary["rig_mode"] == "envmap_only":
            recovery_selected = self._select_recovery_keyframes(
                primary["per_keyframe_results"],
                count=self.settings.recovery_keyframes,
            )
            recovery = self._run_attempt(
                resources=resources,
                selected=recovery_selected,
                provider_dir=provider_dir,
                attempt_name="recovery",
                input_size=self.settings.recovery_input_size,
                no_controlnet=self.settings.no_controlnet,
            )
            attempts.append(recovery)

        final_attempt = self._choose_final_attempt(attempts)
        self._write_attempt_diagnostics(provider_dir, scored, attempts)
        final_attempt = self._recover_degraded_envmap_only_attempt(final_attempt)
        if not final_attempt["validation"]["passed"]:
            raise RuntimeError(self._validation_failure_message(attempts))

        envmap_path = self._write_fused_envmap(final_attempt["envmap_hdr"], provider_dir)
        recovery = {
            "used": len(attempts) > 1,
            "reason": (
                "primary_rig_degraded" if len(attempts) > 1 else None
            ),
            "primary_variant": str(primary["variant"]),
            "final_variant": str(final_attempt["variant"]),
        }
        if final_attempt["rig_mode"] == "envmap_only":
            diagnostics = final_attempt["sun_diagnostics"]
            LOG.warning(
                "Lighting provider published envmap-only lighting because analytic decomposition degraded. "
                "reason=%s camera_mean_spread_deg=%.2f world_mean_spread_deg=%.2f variant=%s",
                diagnostics.get("degraded_reason"),
                float(diagnostics.get("camera_mean_spread_deg", float("nan"))),
                float(diagnostics.get("world_mean_spread_deg", float("nan"))),
                final_attempt["variant"],
            )

        resources.save_lighting(
            LightingData(
                sun_direction_world=np.asarray(
                    final_attempt["sun_direction_world"], dtype=np.float32
                ),
                sun_strength=float(final_attempt["sun_strength"]),
                sun_color=np.asarray(final_attempt["sun_color"], dtype=np.float32),
                mode=str(final_attempt["mode"]),
                envmap_path=str(envmap_path),
                envmap_rotation_world=np.zeros((3,), dtype=np.float32),
                ambient_strength=float(final_attempt["ambient_strength"]),
                schema_version=2,
                rig_mode=str(final_attempt["rig_mode"]),
                light_rig=list(final_attempt["light_rig"]),
                decomposition=dict(final_attempt["decomposition"]),
                quality={
                    str(key): float(value)
                    for key, value in final_attempt["quality"].items()
                },
                sun_diagnostics=dict(final_attempt["sun_diagnostics"]),
                validation=dict(final_attempt["validation"]),
                recovery=recovery,
                selected_frame_indices=[
                    int(item["frame_index"]) for item in final_attempt["selected"]
                ],
                per_keyframe_diagnostics=list(final_attempt["per_keyframe_diagnostics"]),
                metadata={
                    "provider": "DiffusionLightTurboLightingProvider",
                    "tool_repo_root": str(self.settings.repo_root),
                    "tool_env": self.settings.conda_env,
                },
            )
        )

    def _cross_run_payload(self, resources: ResourceStore) -> dict[str, Any] | None:
        if self._cache_manager is None or not self._cache_enabled:
            return None
        payload: dict[str, Any] = {
            "settings": self._cache_manager.normalize(asdict(self.settings)),
            "frames_dir": self._cache_manager.directory_signature(resources.base_dir(ResourceKind.FRAMES)),
            "depth_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.DEPTH),
                canonicalize_npz=True,
            ),
            "trajectory": self._cache_manager.resource_file_key_signature(
                resources.path_for(ResourceKind.TRAJECTORY),
                logical_name="standard/trajectory/poses.npz",
            ),
            "semantics_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.SEMANTICS_2D),
                canonicalize_npz=True,
            ),
            "provider_script": self._cache_manager.script_key_signature(
                Path(__file__),
                repo_root=Path(__file__).resolve().parents[3],
            ),
        }
        for key, script_name in (
            ("dlt_inpaint_script", "inpaint.py"),
            ("dlt_ball2envmap_script", "ball2envmap.py"),
            ("dlt_exposure2hdr_script", "exposure2hdr.py"),
        ):
            script_path = self.settings.repo_root / script_name
            if script_path.exists():
                payload[key] = self._cache_manager.script_key_signature(
                    script_path,
                    repo_root=self.settings.repo_root,
                )
        return payload

    def _dlt_inference_cache_payload(
        self,
        resources: ResourceStore,
        *,
        selected: Sequence[Mapping[str, Any]],
        input_size: int,
        no_controlnet: bool,
    ) -> dict[str, Any] | None:
        if self._cache_manager is None or not self._cache_enabled:
            return None
        return {
            "inference_settings": self._cache_manager.normalize(
                {
                    "repo_root": self.settings.repo_root,
                    "conda_env": self.settings.conda_env,
                    "env_manager": self.settings.env_manager,
                    "hf_home": self.settings.hf_home,
                    "allow_online_model_fetch": self.settings.allow_online_model_fetch,
                    "input_size": int(input_size),
                    "algorithm": self.settings.algorithm,
                    "offload": self.settings.offload,
                    "no_controlnet": bool(no_controlnet),
                    "sdxl_model": self.settings.sdxl_model,
                    "sdxl_vae_model": self.settings.sdxl_vae_model,
                    "sdxl_controlnet_model": self.settings.sdxl_controlnet_model,
                    "depth_estimator_model": self.settings.depth_estimator_model,
                    "primary_keyframes": self.settings.primary_keyframes,
                    "recovery_keyframes": self.settings.recovery_keyframes,
                }
            ),
            "selected_frame_indices": [int(item["frame_index"]) for item in selected],
            "selected_frame_scores": [
                float(item.get("score", item.get("frame_score", 0.0))) for item in selected
            ],
            "frames_dir": self._cache_manager.directory_signature(resources.base_dir(ResourceKind.FRAMES)),
            "depth_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.DEPTH),
                canonicalize_npz=True,
            ),
            "trajectory": self._cache_manager.resource_file_key_signature(
                resources.path_for(ResourceKind.TRAJECTORY),
                logical_name="standard/trajectory/poses.npz",
            ),
            "semantics_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.SEMANTICS_2D),
                canonicalize_npz=True,
            ),
            "provider_script": self._cache_manager.script_key_signature(
                Path(__file__),
                repo_root=Path(__file__).resolve().parents[3],
            ),
            "dlt_inpaint_script": self._cache_manager.script_key_signature(
                self.settings.repo_root / "inpaint.py",
                repo_root=self.settings.repo_root,
            ) if (self.settings.repo_root / "inpaint.py").exists() else None,
            "dlt_ball2envmap_script": self._cache_manager.script_key_signature(
                self.settings.repo_root / "ball2envmap.py",
                repo_root=self.settings.repo_root,
            ) if (self.settings.repo_root / "ball2envmap.py").exists() else None,
            "dlt_exposure2hdr_script": self._cache_manager.script_key_signature(
                self.settings.repo_root / "exposure2hdr.py",
                repo_root=self.settings.repo_root,
            ) if (self.settings.repo_root / "exposure2hdr.py").exists() else None,
        }

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._cache_status)

    def get_cross_run_cache_spec(self, resources: ResourceStore | None) -> dict[str, Any] | None:
        if (
            resources is None
            or self._cache_manager is None
            or not self._cache_enabled
            or self._cache_signature is None
            or self._cache_payload is None
        ):
            return None
        raw_dir = resources.provider_dir("lighting")
        lighting_json = resources.lighting_json_path()
        envmap_path = resources.lighting_envmap_path()
        artifacts = self._cache_manager.collect_tree(raw_dir, rel_prefix="raw/lighting")
        artifacts.update(
            self._cache_manager.collect_file(
                lighting_json,
                relpath="standard/lighting/lighting.json",
            )
        )
        artifacts.update(
            self._cache_manager.collect_file(
                envmap_path,
                relpath="standard/lighting/envmap.exr",
            )
        )
        ready = True
        not_ready_reason: str | None = None
        if not (raw_dir.exists() and any(path.is_file() for path in raw_dir.rglob("*"))):
            ready = False
            not_ready_reason = "raw-lighting-missing"
        elif not lighting_json.exists():
            ready = False
            not_ready_reason = "standard-lighting-json-missing"
        elif not envmap_path.exists():
            ready = False
            not_ready_reason = "standard-lighting-envmap-missing"
        spec: dict[str, Any] = {
            "provider_id": "lighting",
            "signature": self._cache_signature,
            "payload": self._cache_payload,
            "artifacts": artifacts,
            "ready": ready,
            "source_summary": {
                "profile": self._profile_name,
                "run_root": str(resources.root),
            },
            "provenance": {
                "repo_root": str(self.settings.repo_root),
                "frames_dir": str(resources.base_dir(ResourceKind.FRAMES)),
            },
        }
        if not_ready_reason is not None:
            spec["not_ready_reason"] = not_ready_reason
        return spec

    def _build_tool_env(self) -> dict[str, str]:
        return configure_hf_subprocess_env(
            os.environ.copy(),
            hf_home=self.settings.hf_home,
            allow_online_model_fetch=self.settings.allow_online_model_fetch,
        )

    def _required_model_sources(self, *, no_controlnet: bool) -> list[tuple[str, str]]:
        sources = [
            ("sdxl_model", self.settings.sdxl_model),
            ("sdxl_vae_model", self.settings.sdxl_vae_model),
        ]
        if not no_controlnet:
            sources.extend(
                [
                    ("sdxl_controlnet_model", self.settings.sdxl_controlnet_model),
                    ("depth_estimator_model", self.settings.depth_estimator_model),
                ]
            )
        return sources

    def _validate_model_sources(self, env: Mapping[str, str], *, no_controlnet: bool) -> None:
        if self.settings.allow_online_model_fetch:
            return
        missing: list[str] = []
        for setting_name, source in self._required_model_sources(no_controlnet=no_controlnet):
            candidate = Path(source).expanduser()
            if candidate.is_dir():
                continue
            if has_cached_repo(source, env):
                continue
            missing.append(f"{setting_name}={source!r}")
        if not missing:
            return
        raise RuntimeError(
            "DiffusionLight-Turbo models are not available offline. "
            f"Missing sources: {', '.join(missing)}. "
            f"HF_HOME={env.get('HF_HOME')!r} HF_HUB_CACHE={str(hub_cache_dir(env))!r} "
            f"TRANSFORMERS_CACHE={str(transformers_cache_dir(env))!r}. "
            "Populate the cache once with internet access, pass local model directories, "
            "or set allow_online_model_fetch=true."
        )

    def _run_attempt(
        self,
        *,
        resources: ResourceStore,
        selected: Sequence[Mapping[str, Any]],
        provider_dir: Path,
        attempt_name: str,
        input_size: int,
        no_controlnet: bool,
    ) -> dict[str, Any]:
        tool_info = self._run_diffusionlight_cached(
            resources,
            selected,
            provider_dir / attempt_name,
            input_size=input_size,
            no_controlnet=no_controlnet,
            attempt_name=attempt_name,
        )
        estimates = self._build_estimates(resources, selected, tool_info)
        fused = self._fuse_estimates(estimates)
        validation = self._validate_fused_result(fused, estimates)
        return {
            **fused,
            "validation": validation,
            "selected": list(selected),
            "variant": (
                f"{attempt_name}:{'no_controlnet' if no_controlnet else 'controlnet'}:{input_size}px"
            ),
            "per_keyframe_results": list(estimates),
        }

    def _run_diffusionlight_cached(
        self,
        resources: ResourceStore,
        selected: Sequence[Mapping[str, Any]],
        provider_dir: Path,
        *,
        input_size: int,
        no_controlnet: bool,
        attempt_name: str,
    ) -> dict[int, dict[str, Any]]:
        cache_payload = self._dlt_inference_cache_payload(
            resources,
            selected=selected,
            input_size=input_size,
            no_controlnet=no_controlnet,
        )
        if cache_payload is None or self._cache_manager is None or not self._cache_enabled:
            return self._run_diffusionlight(
                resources,
                selected,
                provider_dir,
                input_size=input_size,
                no_controlnet=no_controlnet,
            )
        signature = self._cache_manager.signature(_DLT_INFERENCE_CACHE_PROVIDER_ID, cache_payload)
        lookup = self._cache_manager.lookup(
            _DLT_INFERENCE_CACHE_PROVIDER_ID,
            signature,
            required_relpaths=[f"raw/lighting_inference/{attempt_name}/manifest.json"],
        )
        status = {
            "attempt": attempt_name,
            "signature": signature,
            "hit": lookup.hit,
            "validation": lookup.reason,
            "entry": str(lookup.entry_dir),
        }
        self._inference_cache_statuses.append(status)
        self._cache_status["dlt_inference_cache_attempts"] = list(self._inference_cache_statuses)
        if lookup.hit:
            self._cache_manager.materialize(_DLT_INFERENCE_CACHE_PROVIDER_ID, signature, run_root=resources.root)
            return self._load_cached_dlt_inference(
                resources.root / "raw" / "lighting_inference" / attempt_name,
                selected,
            )
        tool_info = self._run_diffusionlight(
            resources,
            selected,
            provider_dir,
            input_size=input_size,
            no_controlnet=no_controlnet,
        )
        self._publish_dlt_inference_cache(
            signature=signature,
            payload=cache_payload,
            attempt_name=attempt_name,
            provider_dir=provider_dir,
        )
        return tool_info

    def _load_cached_dlt_inference(
        self,
        provider_dir: Path,
        selected: Sequence[Mapping[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        manifest_path = provider_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cached DLT inference manifest not found at '{manifest_path}'.")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        results: dict[int, dict[str, Any]] = {}
        for item in selected:
            frame_index = int(item["frame_index"])
            hdr_rel = payload.get("frames", {}).get(str(frame_index))
            if not hdr_rel:
                raise FileNotFoundError(
                    f"Cached DLT inference for frame {frame_index} missing from {manifest_path}."
                )
            hdr_path = provider_dir / str(hdr_rel)
            results[frame_index] = {
                "hdr_path": hdr_path,
                "hdr": _load_hdr_envmap(hdr_path),
            }
        return results

    def _publish_dlt_inference_cache(
        self,
        *,
        signature: str,
        payload: Mapping[str, Any],
        attempt_name: str,
        provider_dir: Path,
    ) -> None:
        if self._cache_manager is None or not self._cache_enabled:
            return
        hdr_dir = provider_dir / "diffusionlight_turbo" / "output" / "hdr"
        frames: dict[str, str] = {}
        artifacts = self._cache_manager.collect_tree(
            hdr_dir,
            rel_prefix=f"raw/lighting_inference/{attempt_name}",
        )
        for path in sorted(hdr_dir.glob("*.exr")):
            frames[path.stem] = path.name
        manifest_path = provider_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"attempt": attempt_name, "frames": frames}, indent=2),
            encoding="utf-8",
        )
        artifacts.update(
            self._cache_manager.collect_file(
                manifest_path,
                relpath=f"raw/lighting_inference/{attempt_name}/manifest.json",
            )
        )
        self._cache_manager.publish(
            _DLT_INFERENCE_CACHE_PROVIDER_ID,
            signature,
            payload=payload,
            artifacts=artifacts,
            source_summary={"profile": self._profile_name, "attempt": attempt_name},
            provenance={"repo_root": str(self.settings.repo_root)},
        )

    def _score_frames(self, resources: ResourceStore) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for frame_index in resources.frame_indices(ResourceKind.FRAMES):
            frame = resources.load_frame(frame_index)
            depth = resources.load_depth(frame_index)
            semantics = resources.load_semantics2d(frame_index)
            metrics = self._frame_metrics(frame.image, depth.depth, semantics)
            score = (
                2.5 * metrics["sky_fraction"]
                + 1.4 * metrics["road_fraction"]
                + 0.6 * metrics["sharpness_score"]
                - 1.6 * metrics["dynamic_fraction"]
                - 1.1 * metrics["large_vehicle_fraction"]
                - 0.9 * metrics["large_vehicle_near_fraction"]
                - 0.8 * metrics["overexposure_fraction"]
                - 0.3 * metrics["saturation_fraction"]
            )
            scored.append(
                {
                    "frame_index": int(frame_index),
                    "score": float(score),
                    "metrics": metrics,
                }
            )
        return scored

    def _select_keyframes(
        self, scored: Sequence[Mapping[str, Any]], *, count: int
    ) -> list[dict[str, Any]]:
        if not scored:
            raise ValueError("Cannot select lighting keyframes from an empty score list.")
        ordered = sorted(
            (
                {
                    "frame_index": int(item["frame_index"]),
                    "score": float(item["score"]),
                    "metrics": dict(item.get("metrics", {})),
                }
                for item in scored
            ),
            key=lambda item: item["frame_index"],
        )
        if count >= len(ordered):
            return list(ordered)
        segments = np.array_split(np.arange(len(ordered)), count)
        selected: list[dict[str, Any]] = []
        for segment in segments:
            candidates = [ordered[int(idx)] for idx in segment.tolist()]
            selected.append(max(candidates, key=lambda item: item["score"]))
        return selected

    def _select_recovery_keyframes(
        self,
        estimates: Sequence[Mapping[str, Any]],
        *,
        count: int,
    ) -> list[dict[str, Any]]:
        cluster = _best_camera_cluster(estimates)
        prioritized: list[int] = []
        for member in cluster.get("members", []):
            frame_index = int(member["frame_index"])
            if frame_index not in prioritized:
                prioritized.append(frame_index)
        remaining = sorted(
            estimates,
            key=lambda item: (
                float(item.get("consensus_score", 0.0)),
                float(item.get("estimate_quality", 0.0)),
                -int(item["frame_index"]),
            ),
            reverse=True,
        )
        for item in remaining:
            frame_index = int(item["frame_index"])
            if frame_index not in prioritized:
                prioritized.append(frame_index)
        selected_frames = sorted(prioritized[:count])
        selected_by_frame = {
            int(item["frame_index"]): {
                "frame_index": int(item["frame_index"]),
                "score": float(item["frame_score"]),
                "metrics": dict(item.get("frame_metrics", {})),
            }
            for item in estimates
        }
        return [selected_by_frame[idx] for idx in selected_frames]

    def _write_attempt_diagnostics(
        self,
        provider_dir: Path,
        scored: Sequence[Mapping[str, Any]],
        attempts: Sequence[Mapping[str, Any]],
    ) -> None:
        path = provider_dir / "frame_selection.json"
        payload = {
            "frames": list(scored),
            "attempts": [
                {
                    "variant": str(attempt.get("variant")),
                    "mode": str(attempt.get("mode", "unknown")),
                    "rig_mode": str(attempt.get("rig_mode", "unknown")),
                    "selected_frame_indices": [
                        int(item["frame_index"]) for item in attempt.get("selected", [])
                    ],
                    "sun_diagnostics": dict(attempt.get("sun_diagnostics", {})),
                    "decomposition": dict(attempt.get("decomposition", {})),
                    "validation": dict(attempt.get("validation", {})),
                }
                for attempt in attempts
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run_diffusionlight(
        self,
        resources: ResourceStore,
        selected: Sequence[Mapping[str, Any]],
        provider_dir: Path,
        *,
        input_size: int,
        no_controlnet: bool,
    ) -> dict[int, dict[str, Any]]:
        tool_root = provider_dir / "diffusionlight_turbo"
        inputs_dir = tool_root / "inputs"
        output_dir = tool_root / "output"
        if tool_root.exists():
            shutil.rmtree(tool_root)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            frame_index = int(item["frame_index"])
            frame = resources.load_frame(frame_index)
            prepared = _prepare_model_image(frame.image, input_size)
            imageio.imwrite(inputs_dir / f"{frame_index:06d}.png", prepared)
        launcher = resolve_env_launcher(self.settings.conda_env, self.settings.env_manager)
        env = self._build_tool_env()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.settings.repo_root), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        self._validate_model_sources(env, no_controlnet=no_controlnet)
        inpaint_cmd = [
            *launcher,
            "python",
            "inpaint.py",
            "--dataset",
            str(inputs_dir),
            "--output_dir",
            str(output_dir),
            "--algorithm",
            self.settings.algorithm,
            "--sdxl-model",
            self.settings.sdxl_model,
            "--sdxl-vae-model",
            self.settings.sdxl_vae_model,
            "--sdxl-controlnet-model",
            self.settings.sdxl_controlnet_model,
            "--depth-estimator-model",
            self.settings.depth_estimator_model,
            "--allow-online-model-fetch",
            "true" if self.settings.allow_online_model_fetch else "false",
        ]
        if self.settings.offload:
            inpaint_cmd.append("--offload")
        if no_controlnet:
            inpaint_cmd.append("--no_controlnet")
        subprocess.run(inpaint_cmd, check=True, cwd=self.settings.repo_root, env=env)
        subprocess.run(
            [
                *launcher,
                "python",
                "ball2envmap.py",
                "--ball_dir",
                str(output_dir / "square"),
                "--envmap_dir",
                str(output_dir / "envmap"),
            ],
            check=True,
            cwd=self.settings.repo_root,
            env=env,
        )
        subprocess.run(
            [
                *launcher,
                "python",
                "exposure2hdr.py",
                "--input_dir",
                str(output_dir / "envmap"),
                "--output_dir",
                str(output_dir / "hdr"),
            ],
            check=True,
            cwd=self.settings.repo_root,
            env=env,
        )
        results: dict[int, dict[str, Any]] = {}
        for item in selected:
            frame_index = int(item["frame_index"])
            stem = f"{frame_index:06d}"
            hdr_path = output_dir / "hdr" / f"{stem}.exr"
            results[frame_index] = {
                "hdr_path": hdr_path,
                "hdr": _load_hdr_envmap(hdr_path),
            }
        return results

    def _build_estimates(
        self,
        resources: ResourceStore,
        selected: Sequence[Mapping[str, Any]],
        tool_info: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        estimates: list[dict[str, Any]] = []
        frame_scores = np.asarray([float(item["score"]) for item in selected], dtype=np.float32)
        score_min = float(np.min(frame_scores)) if frame_scores.size else 0.0
        score_span = float(np.max(frame_scores) - score_min) if frame_scores.size else 1.0
        for item in selected:
            frame_index = int(item["frame_index"])
            pose = resources.load_pose(frame_index)
            rotation = np.asarray(pose.camera_to_world[:3, :3], dtype=np.float32)
            hdr = np.asarray(tool_info[frame_index]["hdr"], dtype=np.float32)
            sun_candidates = _extract_sun_candidates(hdr)
            best_candidate = (
                sun_candidates[0]
                if sun_candidates
                else {
                    "direction_camera": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                    "strength": float(np.max(_luminance(hdr))),
                    "color": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "confidence": 0.0,
                    "peak_contrast": 0.0,
                    "peak_sharpness": 0.0,
                    "high_energy_fraction": 1.0,
                }
            )
            world_direction = _normalize(rotation @ best_candidate["direction_camera"])
            aligned_hdr = _rotate_envmap(hdr, rotation)
            ambient_hdr = _suppress_sun(aligned_hdr, world_direction)
            score_norm = float(
                np.clip((float(item["score"]) - score_min) / max(score_span, 1e-6), 0.0, 1.0)
            )
            estimate_quality = float(
                np.clip(
                    0.30 * score_norm
                    + 0.35 * float(best_candidate["confidence"])
                    + 0.20 * min(float(best_candidate["peak_contrast"]) / 8.0, 1.0)
                    + 0.15
                    * max(
                        0.0,
                        1.0 - min(float(best_candidate["high_energy_fraction"]) / 0.15, 1.0),
                    ),
                    0.05,
                    1.0,
                )
            )
            estimates.append(
                {
                    "frame_index": frame_index,
                    "frame_score": float(item["score"]),
                    "frame_metrics": dict(item.get("metrics", {})),
                    "rotation_c2w": rotation,
                    "sun_candidates_camera": sun_candidates,
                    "best_candidate_world": world_direction,
                    "sun_strength": float(best_candidate["strength"]),
                    "sun_color": np.asarray(best_candidate["color"], dtype=np.float32),
                    "ambient_hdr": ambient_hdr,
                    "aligned_hdr": aligned_hdr,
                    "weight": estimate_quality,
                    "estimate_quality": estimate_quality,
                    "hdr_p95": float(np.percentile(_luminance(aligned_hdr), 95)),
                    "consensus_score": float(best_candidate["confidence"]),
                    "diagnostics": {
                        "peak_contrast": float(best_candidate["peak_contrast"]),
                        "peak_sharpness": float(best_candidate["peak_sharpness"]),
                        "high_energy_fraction": float(best_candidate["high_energy_fraction"]),
                        "candidate_confidence": float(best_candidate["confidence"]),
                        "num_candidates": int(len(sun_candidates)),
                        "estimate_quality": estimate_quality,
                        "frame_metrics": dict(item.get("metrics", {})),
                    },
                }
            )
        return estimates

    def _fuse_estimates(self, estimates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not estimates:
            raise ValueError("Cannot fuse an empty lighting estimate list.")
        weights = np.asarray(
            [float(item.get("weight", 1.0)) for item in estimates], dtype=np.float32
        )
        if not np.any(weights > 0):
            weights[:] = 1.0
        weights /= np.sum(weights)
        ambient_hdr = np.tensordot(
            weights,
            np.stack(
                [np.asarray(item["ambient_hdr"], dtype=np.float32) for item in estimates],
                axis=0,
            ),
            axes=(0, 0),
        )
        ambient_lum = _luminance(ambient_hdr)
        ambient_strength = float(
            np.clip(
                np.percentile(ambient_lum, 90),
                _VALIDATION_LIMITS["min_ambient_strength"],
                _VALIDATION_LIMITS["max_ambient_strength"],
            )
        )
        analytic_energy_before = 0.0
        keyframe_p95s = np.asarray(
            [float(item["hdr_p95"]) for item in estimates], dtype=np.float32
        )
        camera_cluster = _best_camera_cluster(estimates)
        world_cluster = _world_cluster_from_camera_cluster(camera_cluster, estimates)
        mode = "ambient_only"
        degraded_reason = str(camera_cluster.get("failure_reason", "no_camera_cluster"))
        sun_direction_world = _best_effort_world_direction(estimates, world_cluster)
        sun_color = _best_effort_sun_color(estimates, world_cluster)
        sun_strength = 0.0
        envmap_hdr = ambient_hdr
        quality_sun = 0.0
        if world_cluster["passed"]:
            mode = "full_sun"
            degraded_reason = None
            sun_direction_world = np.asarray(world_cluster["mean_direction"], dtype=np.float32)
            sun_color = _normalize_color(np.asarray(world_cluster["mean_color"], dtype=np.float32))
            sun_strength = float(
                np.clip(
                    float(world_cluster["mean_strength"]),
                    _VALIDATION_LIMITS["min_sun_strength"],
                    _VALIDATION_LIMITS["max_sun_strength"],
                )
            )
            dirs = _latlong_directions(ambient_hdr.shape[0], ambient_hdr.shape[1])
            envmap_hdr = np.clip(
                ambient_hdr
                + _sun_blob(
                    dirs,
                    sun_direction_world,
                    sun_color,
                    sun_strength,
                    self.settings.sun_sigma_deg,
                ),
                0.0,
                None,
            )
            quality_sun = float(
                np.clip(
                    1.0 - float(world_cluster["mean_spread_deg"]) / _WORLD_CLUSTER_MEAN_DEG,
                    0.0,
                    1.0,
                )
            )
        else:
            degraded_reason = str(world_cluster.get("failure_reason", degraded_reason))
        fused_p95 = float(np.percentile(_luminance(envmap_hdr), 95))
        quality_envmap = float(
            np.clip(
                fused_p95 / max(float(np.median(keyframe_p95s)), 1e-6),
                0.0,
                1.0,
            )
        )
        quality_total = float(
            np.clip(
                (0.5 * quality_sun + 0.3 * quality_envmap + 0.2 * float(np.mean(weights)))
                if mode == "full_sun"
                else (0.7 * quality_envmap + 0.3 * float(np.mean(weights))),
                0.0,
                1.0,
            )
        )
        member_frames = {
            int(member["frame_index"]): member
            for member in world_cluster.get("members", camera_cluster.get("members", []))
        }
        fill_lights = _extract_fill_lights(
            ambient_hdr,
            max_lights=self.settings.max_fill_lights,
            min_separation_deg=_FILL_LIGHT_MIN_SEPARATION_DEG,
            min_strength=max(ambient_strength * 0.18, 0.05),
        )
        fill_energy_before = float(
            sum(max(float(item.get("strength", 0.0)), 0.0) for item in fill_lights)
        )
        analytic_energy_before = float(sun_strength + fill_energy_before)
        planner = self._plan_analytic_rig(
            mode=mode,
            sun_strength=sun_strength,
            ambient_strength=ambient_strength,
            quality_sun=quality_sun,
            fill_lights=fill_lights,
            estimates=estimates,
            envmap_hdr=envmap_hdr,
            analytic_energy_before=analytic_energy_before,
        )
        fill_lights = planner["fill_lights"]
        sun_strength = float(planner["sun_strength"])
        ambient_strength = float(planner["ambient_strength"])
        if sun_strength <= 0.0:
            mode = "ambient_only"
        rig_mode = str(planner["rig_mode"])
        light_rig = _build_light_rig(
            rig_mode=rig_mode,
            sun_direction_world=sun_direction_world,
            sun_strength=sun_strength,
            sun_color=sun_color,
            fill_lights=fill_lights,
        )
        if not light_rig and rig_mode != "envmap_only":
            rig_mode = "envmap_only"
        fill_energy_after = float(
            sum(max(float(item.get("strength", 0.0)), 0.0) for item in fill_lights)
        )
        sun_diagnostics = {
            "camera_cluster_count": int(camera_cluster.get("frame_count", 0)),
            "camera_mean_spread_deg": float(camera_cluster.get("mean_spread_deg", 180.0)),
            "world_mean_spread_deg": float(world_cluster.get("mean_spread_deg", 180.0)),
            "winning_frame_indices": [int(item["frame_index"]) for item in member_frames.values()],
            "degraded_reason": degraded_reason,
            "candidate_count": int(camera_cluster.get("candidate_count", 0)),
            "fill_light_count": int(len(fill_lights)),
            "planner_mode": str(planner["planner_mode"]),
        }
        return {
            "mode": mode,
            "rig_mode": rig_mode,
            "sun_direction_world": sun_direction_world,
            "sun_strength": sun_strength,
            "sun_color": sun_color,
            "ambient_strength": ambient_strength,
            "envmap_hdr": envmap_hdr,
            "light_rig": light_rig,
            "decomposition": {
                "method": "dlt_envmap_analytic_lobes_v1",
                "direct_lobe_count": int(1 if mode == "full_sun" and rig_mode != "envmap_only" else 0),
                "fill_light_count": int(len(fill_lights)),
                "analytic_light_count": int(len(light_rig)),
                "confidence": quality_total,
                "fallback_reason": degraded_reason if rig_mode == "envmap_only" else None,
                "planner": dict(planner["diagnostics"]),
                "fill_energy_before": fill_energy_before,
                "fill_energy_after": fill_energy_after,
                "analytic_energy_before": analytic_energy_before,
                "analytic_energy_after": float(sun_strength + fill_energy_after),
            },
            "quality": {
                "total": quality_total,
                "sun": quality_sun,
                "envmap": quality_envmap,
            },
            "sun_diagnostics": sun_diagnostics,
            "per_keyframe_diagnostics": [
                {
                    "frame_index": int(item["frame_index"]),
                    "weight": float(weights[idx]),
                    "in_winning_cluster": int(item["frame_index"]) in member_frames,
                    "agreement_deg": float(
                        member_frames[int(item["frame_index"])]["residual_deg"]
                    )
                    if int(item["frame_index"]) in member_frames
                    else None,
                    **dict(item.get("diagnostics", {})),
                }
                for idx, item in enumerate(estimates)
            ],
        }

    def _validate_fused_result(
        self,
        fused: Mapping[str, Any],
        estimates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        envmap_hdr = np.asarray(fused["envmap_hdr"], dtype=np.float32)
        lum = _luminance(envmap_hdr)
        finite_ratio = float(np.isfinite(envmap_hdr).mean())
        mean_lum = float(np.mean(lum))
        p50_lum = float(np.percentile(lum, 50))
        p95_lum = float(np.percentile(lum, 95))
        p99_lum = float(np.percentile(lum, 99))
        max_lum = float(np.max(lum))
        dynamic_ratio = float(p99_lum / max(p50_lum, 1e-6))
        keyframe_p95s = np.asarray(
            [float(item["hdr_p95"]) for item in estimates], dtype=np.float32
        )
        relative_p95_ratio = float(
            p95_lum / max(float(np.median(keyframe_p95s)), 1e-6)
        )
        direction_norm = float(
            np.linalg.norm(np.asarray(fused["sun_direction_world"], dtype=np.float32))
        )
        quality = fused["quality"]
        mode = str(fused.get("mode", "ambient_only"))
        rig_mode = str(fused.get("rig_mode", "envmap_only"))
        sun_diagnostics = dict(fused.get("sun_diagnostics", {}))
        planner = dict(fused.get("decomposition", {}).get("planner", {}))
        camera_mean_agreement_deg = float(
            sun_diagnostics.get("camera_mean_spread_deg", 180.0)
        )
        world_mean_agreement_deg = float(
            sun_diagnostics.get("world_mean_spread_deg", 180.0)
        )
        checks = {
            "mode": mode,
            "rig_mode": rig_mode,
            "finite_ratio": finite_ratio,
            "mean_luminance": mean_lum,
            "p50_luminance": p50_lum,
            "p95_luminance": p95_lum,
            "p99_luminance": p99_lum,
            "max_luminance": max_lum,
            "dynamic_range_ratio": dynamic_ratio,
            "relative_p95_ratio": relative_p95_ratio,
            "sun_direction_norm": direction_norm,
            "sun_strength": float(fused["sun_strength"]),
            "ambient_strength": float(fused["ambient_strength"]),
            "quality_total": float(quality["total"]),
            "quality_sun": float(quality["sun"]),
            "quality_envmap": float(quality["envmap"]),
            "camera_mean_agreement_deg": camera_mean_agreement_deg,
            "world_mean_agreement_deg": world_mean_agreement_deg,
            "camera_cluster_count": int(sun_diagnostics.get("camera_cluster_count", 0)),
            "analytic_light_count": int(len(fused.get("light_rig", ()))),
            "planner_mode": str(planner.get("mode", "unknown")),
            "direct_scene_score": float(planner.get("direct_scene_score", 0.0)),
            "diffuse_scene_score": float(planner.get("diffuse_scene_score", 0.0)),
            "direct_to_fill_ratio": float(planner.get("direct_to_fill_ratio", float("inf"))),
        }
        ambient_failures: list[str] = []
        if finite_ratio < 1.0:
            ambient_failures.append("non_finite_envmap")
        if mean_lum < _VALIDATION_LIMITS["min_mean_luminance"]:
            ambient_failures.append("mean_luminance_too_low")
        if p95_lum < _VALIDATION_LIMITS["min_p95_luminance"]:
            ambient_failures.append("p95_luminance_too_low")
        if max_lum < _VALIDATION_LIMITS["min_max_luminance"]:
            ambient_failures.append("max_luminance_too_low")
        if dynamic_ratio < _VALIDATION_LIMITS["min_dynamic_range_ratio"]:
            ambient_failures.append("dynamic_range_too_low")
        if relative_p95_ratio < _VALIDATION_LIMITS["min_relative_p95_ratio"]:
            ambient_failures.append("relative_p95_ratio_too_low")
        if not (0.98 <= direction_norm <= 1.02):
            ambient_failures.append("invalid_sun_direction_norm")
        if not (
            _VALIDATION_LIMITS["min_ambient_strength"]
            <= float(fused["ambient_strength"])
            <= _VALIDATION_LIMITS["max_ambient_strength"]
        ):
            ambient_failures.append("ambient_strength_out_of_range")
        light_rig = fused.get("light_rig", ())
        if rig_mode not in {"analytic_rig", "sun_plus_fill", "envmap_only"}:
            ambient_failures.append("invalid_rig_mode")
        if rig_mode == "envmap_only" and light_rig:
            ambient_failures.append("envmap_only_should_not_emit_analytic_lights")
        planner_mode = str(planner.get("mode", "unknown"))
        if planner_mode == "fill_heavy":
            direct_to_fill_ratio = float(planner.get("direct_to_fill_ratio", float("inf")))
            if direct_to_fill_ratio > self.settings.max_direct_to_fill_ratio_for_diffuse + 1e-6:
                ambient_failures.append("diffuse_direct_to_fill_ratio_too_high")
            brightness_preservation_ratio = float(planner.get("brightness_preservation_ratio", 0.0))
            if (
                brightness_preservation_ratio + 1e-6
                < _MIN_FILL_HEAVY_BRIGHTNESS_PRESERVATION
            ):
                ambient_failures.append("fill_heavy_brightness_preservation_too_low")
        if planner.get("fill_transport_mode") not in {None, "none"} and not bool(
            planner.get("transport_validation_passed", False)
        ):
            ambient_failures.append("ineffective_subject_fill_transport")
        sun_failures: list[str] = []
        if mode == "full_sun":
            if not (
                _VALIDATION_LIMITS["min_sun_strength"]
                <= float(fused["sun_strength"])
                <= _VALIDATION_LIMITS["max_sun_strength"]
            ):
                sun_failures.append("sun_strength_out_of_range")
            if world_mean_agreement_deg > _WORLD_CLUSTER_MEAN_DEG:
                sun_failures.append("sun_agreement_too_low")
        blocking_failures = list(ambient_failures if ambient_failures else sun_failures)
        return {
            "passed": not blocking_failures,
            "attempts_used": 1,
            "mode": mode,
            "rig_mode": rig_mode,
            "ambient_passed": not ambient_failures,
            "sun_passed": mode == "full_sun" and not sun_failures,
            "checks": checks,
            "failures": blocking_failures,
            "sun_failures": sun_failures,
            "degraded_reason": sun_diagnostics.get("degraded_reason"),
        }

    def _validation_failure_message(self, attempts: Sequence[Mapping[str, Any]]) -> str:
        summary = [
            {
                "variant": str(item.get("variant")),
                "mode": str(item.get("mode", "unknown")),
                "rig_mode": str(item.get("rig_mode", "unknown")),
                "sun_diagnostics": dict(item.get("sun_diagnostics", {})),
                "decomposition": dict(item.get("decomposition", {})),
                "validation": dict(item.get("validation", {})),
            }
            for item in attempts
        ]
        return "Lighting provider failed to produce plausible lighting.\n" + json.dumps(
            summary, indent=2
        )

    def _recover_degraded_envmap_only_attempt(
        self,
        attempt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        validation = attempt.get("validation", {})
        if not isinstance(validation, Mapping) or bool(validation.get("passed", False)):
            return attempt
        failures = validation.get("failures", ())
        if not isinstance(failures, Sequence):
            return attempt
        failure_set = {str(item) for item in failures}
        if not failure_set or not failure_set.issubset(_RECOVERABLE_LIGHTING_FAILURES):
            return attempt

        checks = validation.get("checks", {})
        quality = attempt.get("quality", {})
        if not isinstance(checks, Mapping) or not isinstance(quality, Mapping):
            return attempt
        envmap_hdr = attempt.get("envmap_hdr")
        if envmap_hdr is None:
            return attempt

        finite_ratio = float(checks.get("finite_ratio", 0.0))
        mean_lum = float(checks.get("mean_luminance", 0.0))
        p95_lum = float(checks.get("p95_luminance", 0.0))
        max_lum = float(checks.get("max_luminance", 0.0))
        relative_p95_ratio = float(checks.get("relative_p95_ratio", 0.0))
        dynamic_ratio = float(checks.get("dynamic_range_ratio", 0.0))
        ambient_strength = float(attempt.get("ambient_strength", 0.0))
        quality_envmap = float(quality.get("envmap", 0.0))
        if finite_ratio < 1.0:
            return attempt
        if mean_lum < _VALIDATION_LIMITS["min_mean_luminance"]:
            return attempt
        if p95_lum < _VALIDATION_LIMITS["min_p95_luminance"]:
            return attempt
        if max_lum < _VALIDATION_LIMITS["min_max_luminance"]:
            return attempt
        if relative_p95_ratio < _VALIDATION_LIMITS["min_relative_p95_ratio"]:
            return attempt
        if dynamic_ratio < _RECOVERABLE_ENV_FALLBACK_LIMITS["min_dynamic_range_ratio"]:
            return attempt
        if quality_envmap < _RECOVERABLE_ENV_FALLBACK_LIMITS["min_quality_envmap"]:
            return attempt
        if not (
            _VALIDATION_LIMITS["min_ambient_strength"]
            <= ambient_strength
            <= _VALIDATION_LIMITS["max_ambient_strength"]
        ):
            return attempt

        recovered = dict(attempt)
        recovered["mode"] = "ambient_only"
        recovered["rig_mode"] = "envmap_only"
        recovered["sun_strength"] = 0.0
        recovered["light_rig"] = []
        decomposition = dict(attempt.get("decomposition", {}))
        decomposition["fallback_reason"] = "recoverable_envmap_only_validation_fallback"
        decomposition["analytic_light_count"] = 0
        recovered["decomposition"] = decomposition
        recovered_validation = dict(validation)
        recovered_validation.update(
            {
                "passed": True,
                "mode": "ambient_only",
                "rig_mode": "envmap_only",
                "ambient_passed": True,
                "sun_passed": False,
                "degraded_fallback_used": True,
                "degraded_fallback_reason": "recoverable_envmap_only_validation_fallback",
                "original_failures": list(failures),
                "failures": [],
            }
        )
        recovered["validation"] = recovered_validation
        return recovered

    def _choose_final_attempt(self, attempts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        valid = [attempt for attempt in attempts if bool(attempt["validation"]["passed"])]
        if not valid:
            return attempts[-1]
        ranked = sorted(
            valid,
            key=lambda item: (
                _rig_mode_rank(str(item.get("rig_mode", "envmap_only"))),
                str(item.get("mode", "ambient_only")) == "full_sun",
                float(item.get("quality", {}).get("total", 0.0)),
            ),
            reverse=True,
        )
        return ranked[0]

    def _write_fused_envmap(self, hdr: np.ndarray, provider_dir: Path) -> Path:
        fused_dir = provider_dir / "fused_envmap"
        if fused_dir.exists():
            shutil.rmtree(fused_dir)
        fused_dir.mkdir(parents=True, exist_ok=True)
        hdr_path = fused_dir / "fused.exr"
        _write_hdr_envmap(hdr_path, hdr)
        preview = _tonemap_preview(hdr)
        imageio.imwrite(fused_dir / "fused_preview.png", preview)
        return hdr_path

    def _frame_metrics(self, image: np.ndarray, depth: np.ndarray, semantics: Any) -> dict[str, float]:
        rgb = np.asarray(image, dtype=np.float32)[..., :3]
        if rgb.max() > 1.0:
            rgb /= 255.0
        label_map = _semantic_label_map(semantics)
        labels = semantics.label_ids if semantics.label_ids is not None else semantics.segment_ids
        if labels is None:
            raise ValueError(
                f"Semantics frame {semantics.frame_index} is missing label_ids and segment_ids."
            )
        labels = np.asarray(labels, dtype=np.int32)
        role_defaults = build_semantic_roles(
            semantics.metadata.get(SEMANTIC_ROLES_METADATA_KEY, _DEFAULT_ROLE_DEFAULTS)
            if isinstance(semantics.metadata, Mapping)
            else _DEFAULT_ROLE_DEFAULTS
        )
        sky_mask = np.isin(labels, _label_ids_for_tokens(label_map, role_defaults.get("sky", ())))
        road_mask = np.isin(labels, _label_ids_for_tokens(label_map, role_defaults.get("road", ())))
        mobile_mask = np.isin(labels, _label_ids_for_tokens(label_map, role_defaults.get("mobile", ())))
        large_vehicle_mask = np.isin(
            labels, _label_ids_for_tokens(label_map, role_defaults.get("large_vehicle", ()))
        )
        near_large_vehicle = large_vehicle_mask & np.isfinite(depth) & (depth > 0.0) & (depth < 12.0)
        gray = rgb.mean(axis=-1)
        grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
        grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
        sharpness = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))
        return {
            "sky_fraction": float(np.mean(sky_mask)),
            "road_fraction": float(np.mean(road_mask)),
            "dynamic_fraction": float(np.mean(mobile_mask)),
            "large_vehicle_fraction": float(np.mean(large_vehicle_mask)),
            "large_vehicle_near_fraction": float(np.mean(near_large_vehicle)),
            "overexposure_fraction": float(np.mean(gray > 0.96)),
            "saturation_fraction": float(np.mean((rgb.max(axis=-1) - rgb.min(axis=-1)) > 0.85)),
            "sharpness_score": float(min(sharpness / 0.12, 1.0)),
        }

    def _plan_analytic_rig(
        self,
        *,
        mode: str,
        sun_strength: float,
        ambient_strength: float,
        quality_sun: float,
        fill_lights: Sequence[Mapping[str, Any]],
        estimates: Sequence[Mapping[str, Any]],
        envmap_hdr: np.ndarray,
        analytic_energy_before: float,
    ) -> dict[str, Any]:
        fill_lights_out = [dict(item) for item in fill_lights]
        ambient_strength_out = float(ambient_strength)
        fill_energy = float(sum(max(float(item.get("strength", 0.0)), 0.0) for item in fill_lights_out))
        envmap_lum = _luminance(np.asarray(envmap_hdr, dtype=np.float32))
        p95 = float(np.percentile(envmap_lum, 95))
        p99 = float(np.percentile(envmap_lum, 99))
        max_lum = float(np.max(envmap_lum))
        envmap_peak_ratio = p99 / max(p95, 1e-6)
        frame_metrics = [dict(item.get("frame_metrics", {})) for item in estimates if isinstance(item.get("frame_metrics"), Mapping)]
        mean_sky_fraction = _mean_metric(frame_metrics, "sky_fraction")
        mean_sharpness = _mean_metric(frame_metrics, "sharpness_score")
        mean_saturation = _mean_metric(frame_metrics, "saturation_fraction")
        mean_overexposure = _mean_metric(frame_metrics, "overexposure_fraction")
        mean_candidate_confidence = _mean_metric(
            [dict(item.get("diagnostics", {})) for item in estimates if isinstance(item.get("diagnostics"), Mapping)],
            "candidate_confidence",
        )
        diffuse_scene_score = float(
            np.clip(
                0.42 * (1.0 - np.clip(mean_sharpness, 0.0, 1.0))
                + 0.28 * (1.0 - np.clip(mean_saturation * 8.0, 0.0, 1.0))
                + 0.18 * np.clip(mean_sky_fraction / 0.22, 0.0, 1.0)
                + 0.12 * np.clip(fill_energy / max(sun_strength + fill_energy, 1e-6), 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        direct_scene_score = float(
            np.clip(
                0.46 * np.clip(quality_sun, 0.0, 1.0)
                + 0.26 * np.clip(mean_candidate_confidence, 0.0, 1.0)
                + 0.18 * np.clip((envmap_peak_ratio - 1.0) / 8.0, 0.0, 1.0)
                + 0.10 * np.clip(mean_overexposure / 0.01, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        direct_to_fill_ratio = float(sun_strength / max(fill_energy, 1e-6)) if sun_strength > 0.0 else 0.0
        planner_mode = "envmap_only"
        rig_mode = "envmap_only"
        demoted_direct_sun = False
        reasons: list[str] = []
        sun_strength_out = float(sun_strength)
        view_direction_world = _representative_view_direction(estimates)
        subject_metrics_before = _subject_probe_metrics(
            sun_strength=sun_strength_out,
            fill_lights=fill_lights_out,
            ambient_strength=ambient_strength_out,
            view_direction_world=view_direction_world,
        )
        if mode == "full_sun":
            if fill_lights_out and _should_demote_to_fill_heavy(
                diffuse_demote_enabled=self.settings.diffuse_demote_enabled,
                aggressiveness=self.settings.diffuse_demote_aggressiveness,
                diffuse_scene_score=diffuse_scene_score,
                direct_scene_score=direct_scene_score,
                direct_to_fill_ratio=direct_to_fill_ratio,
                max_direct_to_fill_ratio_for_diffuse=self.settings.max_direct_to_fill_ratio_for_diffuse,
                fill_count=len(fill_lights_out),
                min_fill_count=self.settings.fill_heavy_min_fill_count,
            ):
                planner_mode = "fill_heavy"
                demoted_direct_sun = True
                reasons.append("diffuse_scene_demoted_direct_sun")
                sun_strength_out, fill_lights_out = _rebalance_fill_heavy_rig(
                    sun_strength=sun_strength_out,
                    fill_lights=fill_lights_out,
                    direct_scale=self.settings.fill_heavy_direct_scale,
                    max_direct_to_fill_ratio=self.settings.max_direct_to_fill_ratio_for_diffuse,
                    ambient_strength=ambient_strength,
                    analytic_energy_before=analytic_energy_before,
                )
                if sun_strength_out <= 0.0:
                    reasons.append("direct_sun_suppressed_after_fill_heavy_rebalance")
            else:
                planner_mode = "direct_dominant"
        elif fill_lights_out:
            planner_mode = "fill_heavy"
            reasons.append("no_valid_direct_sun_using_diffuse_fills")
        if fill_lights_out:
            ambient_strength_out, fill_lights_out, transport_diagnostics = _apply_subject_fill_targets(
                sun_strength=sun_strength_out,
                fill_lights=fill_lights_out,
                ambient_strength=ambient_strength_out,
                analytic_energy_before=analytic_energy_before,
                planner_mode=planner_mode,
                view_direction_world=view_direction_world,
                fill_heavy_transport_gain=self.settings.fill_heavy_transport_gain,
                fill_heavy_dark_side_target_ratio=self.settings.fill_heavy_dark_side_target_ratio,
                diffuse_softness_bias=self.settings.diffuse_softness_bias,
                wrap_geometry_min_azimuth_separation_deg=self.settings.wrap_geometry_min_azimuth_separation_deg,
                wrap_geometry_counter_opposition_deg=self.settings.wrap_geometry_counter_opposition_deg,
                wrap_geometry_sky_min_elevation_deg=self.settings.wrap_geometry_sky_min_elevation_deg,
                wrap_geometry_candidate_count_per_role=self.settings.wrap_geometry_candidate_count_per_role,
            )
            reasons.extend(str(item) for item in transport_diagnostics.get("reasons", ()))
        else:
            transport_diagnostics = {
                "fill_transport_mode": "none",
                "view_direction_world": view_direction_world.tolist(),
                "subject_total_irradiance_before": float(subject_metrics_before["total"]),
                "subject_total_irradiance_after": float(subject_metrics_before["total"]),
                "subject_dark_side_irradiance_before": float(subject_metrics_before["dark_side"]),
                "subject_dark_side_irradiance_after": float(subject_metrics_before["dark_side"]),
                "dark_to_bright_ratio_before": float(subject_metrics_before["dark_to_bright_ratio"]),
                "dark_to_bright_ratio_after": float(subject_metrics_before["dark_to_bright_ratio"]),
                "view_facing_irradiance_before": float(subject_metrics_before["view_facing"]),
                "view_facing_irradiance_after": float(subject_metrics_before["view_facing"]),
                "direct_to_diffuse_subject_ratio_before": float(subject_metrics_before["direct_to_diffuse_ratio"]),
                "direct_to_diffuse_subject_ratio_after": float(subject_metrics_before["direct_to_diffuse_ratio"]),
                "camera_side_fill_balance_before": float(subject_metrics_before["camera_side_fill_balance"]),
                "camera_side_fill_balance_after": float(subject_metrics_before["camera_side_fill_balance"]),
                "fill_view_alignment_score_before": float(subject_metrics_before["fill_view_alignment_score"]),
                "fill_view_alignment_score_after": float(subject_metrics_before["fill_view_alignment_score"]),
                "world_strength_before": float(ambient_strength),
                "world_strength_after": float(ambient_strength_out),
                "fill_target_total_irradiance": float(subject_metrics_before["total"]),
                "fill_target_dark_side_irradiance": float(subject_metrics_before["dark_side"]),
                "fill_target_view_facing_irradiance": float(subject_metrics_before["view_facing"]),
                "fill_heavy_transport_gain": float(self.settings.fill_heavy_transport_gain),
                "fill_heavy_dark_side_target_ratio": float(
                    self.settings.fill_heavy_dark_side_target_ratio
                ),
                "diffuse_softness_bias": float(self.settings.diffuse_softness_bias),
                "effective_fill_heavy_transport_gain": float(self.settings.fill_heavy_transport_gain),
                "effective_fill_heavy_dark_side_target_ratio": float(
                    self.settings.fill_heavy_dark_side_target_ratio
                ),
                "target_direct_to_diffuse_subject_ratio": float(0.16),
                "wrap_geometry_min_azimuth_separation_deg": float(
                    self.settings.wrap_geometry_min_azimuth_separation_deg
                ),
                "wrap_geometry_counter_opposition_deg": float(
                    self.settings.wrap_geometry_counter_opposition_deg
                ),
                "wrap_geometry_sky_min_elevation_deg": float(
                    self.settings.wrap_geometry_sky_min_elevation_deg
                ),
                "wrap_geometry_candidate_count_per_role": int(
                    self.settings.wrap_geometry_candidate_count_per_role
                ),
                "dark_to_bright_ratio_before_geometry": float(
                    subject_metrics_before["dark_to_bright_ratio"]
                ),
                "dark_to_bright_ratio_after_geometry": float(
                    subject_metrics_before["dark_to_bright_ratio"]
                ),
                "camera_side_fill_balance_before_geometry": float(
                    subject_metrics_before["camera_side_fill_balance"]
                ),
                "camera_side_fill_balance_after_geometry": float(
                    subject_metrics_before["camera_side_fill_balance"]
                ),
                "geometry_candidate_count": 0,
                "geometry_winning_candidate_index": None,
                "geometry_winning_candidate_score": 0.0,
                "geometry_best_dark_to_bright_ratio": float(
                    subject_metrics_before["dark_to_bright_ratio"]
                ),
                "geometry_best_camera_side_fill_balance": float(
                    subject_metrics_before["camera_side_fill_balance"]
                ),
                "role_geometry": {},
                "transport_validation_passed": True,
                "reasons": [],
            }
        if planner_mode == "direct_dominant":
            rig_mode = "analytic_rig" if len(fill_lights_out) >= self.settings.fill_heavy_min_fill_count else "sun_plus_fill"
        elif planner_mode == "fill_heavy":
            rig_mode = "analytic_rig" if len(fill_lights_out) >= self.settings.fill_heavy_min_fill_count else "sun_plus_fill"
            if not fill_lights_out:
                rig_mode = "envmap_only"
                planner_mode = "envmap_only"
                sun_strength_out = 0.0
                reasons.append("fill_heavy_requested_without_fill_lights")
        if planner_mode == "envmap_only":
            sun_strength_out = 0.0
            rig_mode = "envmap_only"
        direct_to_fill_ratio_out = float(
            sun_strength_out / max(sum(max(float(item.get("strength", 0.0)), 0.0) for item in fill_lights_out), 1e-6)
        ) if sun_strength_out > 0.0 else 0.0
        fill_energy_after = float(
            sum(max(float(item.get("strength", 0.0)), 0.0) for item in fill_lights_out)
        )
        analytic_energy_after = float(sun_strength_out + fill_energy_after)
        brightness_preservation_ratio = (
            float(analytic_energy_after / max(analytic_energy_before, 1e-6))
            if analytic_energy_before > 0.0
            else 1.0
        )
        diagnostics = {
            "mode": planner_mode,
            "demoted_direct_sun": demoted_direct_sun,
            "reasons": reasons,
            "direct_scene_score": direct_scene_score,
            "diffuse_scene_score": diffuse_scene_score,
            "direct_to_fill_ratio": direct_to_fill_ratio_out,
            "direct_to_fill_ratio_before_rebalance": direct_to_fill_ratio,
            "envmap_peak_ratio": envmap_peak_ratio,
            "direct_energy_before": float(sun_strength),
            "direct_energy_after": float(sun_strength_out),
            "fill_energy_before": fill_energy,
            "fill_energy_after": fill_energy_after,
            "analytic_energy_before": float(analytic_energy_before),
            "analytic_energy_after": analytic_energy_after,
            "brightness_preservation_ratio": brightness_preservation_ratio,
            "demotion_energy_redistributed": float(max(fill_energy_after - fill_energy, 0.0)),
            "fill_transport_mode": str(transport_diagnostics["fill_transport_mode"]),
            "view_direction_world": list(transport_diagnostics["view_direction_world"]),
            "subject_total_irradiance_before": float(transport_diagnostics["subject_total_irradiance_before"]),
            "subject_total_irradiance_after": float(transport_diagnostics["subject_total_irradiance_after"]),
            "subject_dark_side_irradiance_before": float(transport_diagnostics["subject_dark_side_irradiance_before"]),
            "subject_dark_side_irradiance_after": float(transport_diagnostics["subject_dark_side_irradiance_after"]),
            "dark_to_bright_ratio_before": float(transport_diagnostics["dark_to_bright_ratio_before"]),
            "dark_to_bright_ratio_after": float(transport_diagnostics["dark_to_bright_ratio_after"]),
            "view_facing_irradiance_before": float(transport_diagnostics["view_facing_irradiance_before"]),
            "view_facing_irradiance_after": float(transport_diagnostics["view_facing_irradiance_after"]),
            "direct_to_diffuse_subject_ratio_before": float(transport_diagnostics["direct_to_diffuse_subject_ratio_before"]),
            "direct_to_diffuse_subject_ratio_after": float(transport_diagnostics["direct_to_diffuse_subject_ratio_after"]),
            "camera_side_fill_balance_before": float(transport_diagnostics["camera_side_fill_balance_before"]),
            "camera_side_fill_balance_after": float(transport_diagnostics["camera_side_fill_balance_after"]),
            "fill_view_alignment_score_before": float(transport_diagnostics["fill_view_alignment_score_before"]),
            "fill_view_alignment_score_after": float(transport_diagnostics["fill_view_alignment_score_after"]),
            "world_strength_before": float(transport_diagnostics["world_strength_before"]),
            "world_strength_after": float(transport_diagnostics["world_strength_after"]),
            "fill_target_total_irradiance": float(transport_diagnostics["fill_target_total_irradiance"]),
            "fill_target_dark_side_irradiance": float(transport_diagnostics["fill_target_dark_side_irradiance"]),
            "fill_target_view_facing_irradiance": float(transport_diagnostics["fill_target_view_facing_irradiance"]),
            "fill_heavy_transport_gain": float(transport_diagnostics["fill_heavy_transport_gain"]),
            "fill_heavy_dark_side_target_ratio": float(
                transport_diagnostics["fill_heavy_dark_side_target_ratio"]
            ),
            "diffuse_softness_bias": float(transport_diagnostics["diffuse_softness_bias"]),
            "effective_fill_heavy_transport_gain": float(
                transport_diagnostics["effective_fill_heavy_transport_gain"]
            ),
            "effective_fill_heavy_dark_side_target_ratio": float(
                transport_diagnostics["effective_fill_heavy_dark_side_target_ratio"]
            ),
            "target_direct_to_diffuse_subject_ratio": float(
                transport_diagnostics["target_direct_to_diffuse_subject_ratio"]
            ),
            "wrap_geometry_min_azimuth_separation_deg": float(
                transport_diagnostics["wrap_geometry_min_azimuth_separation_deg"]
            ),
            "wrap_geometry_counter_opposition_deg": float(
                transport_diagnostics["wrap_geometry_counter_opposition_deg"]
            ),
            "wrap_geometry_sky_min_elevation_deg": float(
                transport_diagnostics["wrap_geometry_sky_min_elevation_deg"]
            ),
            "wrap_geometry_candidate_count_per_role": int(
                transport_diagnostics["wrap_geometry_candidate_count_per_role"]
            ),
            "dark_to_bright_ratio_before_geometry": float(
                transport_diagnostics["dark_to_bright_ratio_before_geometry"]
            ),
            "dark_to_bright_ratio_after_geometry": float(
                transport_diagnostics["dark_to_bright_ratio_after_geometry"]
            ),
            "camera_side_fill_balance_before_geometry": float(
                transport_diagnostics["camera_side_fill_balance_before_geometry"]
            ),
            "camera_side_fill_balance_after_geometry": float(
                transport_diagnostics["camera_side_fill_balance_after_geometry"]
            ),
            "geometry_candidate_count": int(transport_diagnostics["geometry_candidate_count"]),
            "geometry_winning_candidate_index": transport_diagnostics["geometry_winning_candidate_index"],
            "geometry_winning_candidate_score": float(
                transport_diagnostics["geometry_winning_candidate_score"]
            ),
            "geometry_best_dark_to_bright_ratio": float(
                transport_diagnostics["geometry_best_dark_to_bright_ratio"]
            ),
            "geometry_best_camera_side_fill_balance": float(
                transport_diagnostics["geometry_best_camera_side_fill_balance"]
            ),
            "role_geometry": dict(transport_diagnostics["role_geometry"]),
            "transport_validation_passed": bool(transport_diagnostics["transport_validation_passed"]),
            "frame_mean_sky_fraction": mean_sky_fraction,
            "frame_mean_sharpness_score": mean_sharpness,
            "frame_mean_saturation_fraction": mean_saturation,
            "frame_mean_overexposure_fraction": mean_overexposure,
            "mean_candidate_confidence": mean_candidate_confidence,
            "max_envmap_luminance": max_lum,
        }
        return {
            "planner_mode": planner_mode,
            "rig_mode": rig_mode,
            "sun_strength": sun_strength_out,
            "ambient_strength": ambient_strength_out,
            "fill_lights": fill_lights_out,
            "diagnostics": diagnostics,
        }


def _carla_sun_direction(weather: Mapping[str, Any]) -> np.ndarray:
    azimuth_deg = float(weather.get("sun_azimuth_angle", 0.0))
    altitude_deg = float(weather.get("sun_altitude_angle", 0.0))
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    return _normalize(
        np.asarray(
            (
                math.cos(alt) * math.cos(az),
                math.cos(alt) * math.sin(az),
                math.sin(alt),
            ),
            dtype=np.float32,
        )
    )


def _carla_sun_strength(weather: Mapping[str, Any], scale: float) -> float:
    altitude = max(float(weather.get("sun_altitude_angle", -90.0)), -90.0)
    cloudiness = np.clip(float(weather.get("cloudiness", 0.0)) / 100.0, 0.0, 1.0)
    if altitude <= 0.0:
        return 0.0
    altitude_factor = max(math.sin(math.radians(altitude)), 0.0)
    return float(np.clip((0.4 + 2.2 * altitude_factor) * (1.0 - 0.75 * cloudiness) * scale, 0.0, 5.0))


def _carla_sun_color(weather: Mapping[str, Any]) -> np.ndarray:
    altitude = float(weather.get("sun_altitude_angle", 0.0))
    if altitude < 0.0:
        return np.asarray((0.65, 0.72, 0.95), dtype=np.float32)
    sunset_mix = float(np.clip(1.0 - max(altitude, 0.0) / 25.0, 0.0, 1.0))
    day = np.asarray((1.0, 0.98, 0.95), dtype=np.float32)
    sunset = np.asarray((1.0, 0.72, 0.5), dtype=np.float32)
    return _normalize_color((1.0 - sunset_mix) * day + sunset_mix * sunset)


def _carla_ambient_strength(weather: Mapping[str, Any], scale: float) -> float:
    cloudiness = np.clip(float(weather.get("cloudiness", 0.0)) / 100.0, 0.0, 1.0)
    fog = np.clip(float(weather.get("fog_density", 0.0)) / 100.0, 0.0, 1.0)
    wetness = np.clip(float(weather.get("wetness", 0.0)) / 100.0, 0.0, 1.0)
    scattering = np.clip(float(weather.get("scattering_intensity", 0.0)), 0.0, 5.0) / 5.0
    mie = np.clip(float(weather.get("mie_scattering_scale", 0.0)), 0.0, 5.0) / 5.0
    rayleigh = np.clip(float(weather.get("rayleigh_scattering_scale", 0.0)), 0.0, 5.0) / 5.0
    base = 0.16 + 0.30 * cloudiness + 0.10 * fog + 0.06 * wetness + 0.10 * scattering + 0.06 * mie + 0.08 * rayleigh
    return float(np.clip(base * scale, 0.05, 1.25))


def _carla_scene_light_rig(
    *,
    scene_lights: Mapping[str, Any],
    scene_light_strength_scale: float,
    sun_altitude_angle: float,
) -> list[LightingLightData]:
    lights: list[LightingLightData] = []
    daylight_attenuation = _carla_scene_light_daylight_attenuation(sun_altitude_angle)
    for idx, item in enumerate(scene_lights.get("lights", []) or []):
        if not isinstance(item, Mapping) or not bool(item.get("is_on")):
            continue
        location = item.get("location", {})
        color = item.get("color", {})
        group = str(item.get("light_group", "Other")).strip().lower()
        role = {"street": "street_fill", "building": "building_fill"}.get(group, "other_fill")
        intensity = max(float(item.get("intensity", 0.0)), 0.0)
        lights.append(
            LightingLightData(
                name=f"CARLASceneLight{idx:04d}",
                kind="POINT",
                role=role,
                strength=float(
                    np.clip(
                        intensity * scene_light_strength_scale * daylight_attenuation,
                        0.0,
                        50.0,
                    )
                ),
                color=np.asarray(
                    (
                        float(color.get("r", 255)) / 255.0,
                        float(color.get("g", 255)) / 255.0,
                        float(color.get("b", 255)) / 255.0,
                    ),
                    dtype=np.float32,
                ),
                casts_shadow=False,
                location_world=np.asarray(
                    (
                        float(location.get("x", 0.0)),
                        float(location.get("y", 0.0)),
                        float(location.get("z", 0.0)),
                    ),
                    dtype=np.float32,
                ),
                confidence=1.0,
                diagnostics={
                    "source": "carla_scene_light",
                    "carla_light_group": str(item.get("light_group", "")),
                    "carla_light_id": int(item.get("id", idx)),
                    "daylight_attenuation": float(daylight_attenuation),
                },
            )
        )
    return lights


def _carla_scene_light_daylight_attenuation(sun_altitude_angle: float) -> float:
    if sun_altitude_angle <= 0.0:
        return 1.0
    altitude_ratio = float(np.clip(sun_altitude_angle / 45.0, 0.0, 1.0))
    return float(1.0 - 0.85 * altitude_ratio)


def _carla_synthetic_envmap(
    *,
    weather: Mapping[str, Any],
    ambient_strength: float,
    height: int,
    width: int,
) -> np.ndarray:
    directions = _latlong_directions(height, width)
    vertical = np.clip(directions[..., 2], -1.0, 1.0)
    cloudiness = np.clip(float(weather.get("cloudiness", 0.0)) / 100.0, 0.0, 1.0)
    fog = np.clip(float(weather.get("fog_density", 0.0)) / 100.0, 0.0, 1.0)
    scattering = np.clip(float(weather.get("scattering_intensity", 0.0)), 0.0, 5.0) / 5.0
    mie = np.clip(float(weather.get("mie_scattering_scale", 0.0)), 0.0, 5.0) / 5.0
    rayleigh = np.clip(float(weather.get("rayleigh_scattering_scale", 0.0)), 0.0, 5.0) / 5.0
    horizon = np.asarray((0.74, 0.77, 0.81), dtype=np.float32) + np.asarray((0.03, 0.02, 0.0), dtype=np.float32) * mie
    zenith = np.asarray((0.18, 0.32, 0.62), dtype=np.float32) * (1.0 - 0.45 * cloudiness) + np.asarray((0.02, 0.05, 0.10), dtype=np.float32) * rayleigh
    ground = np.asarray((0.08, 0.09, 0.11), dtype=np.float32) * (1.0 + 0.35 * fog)
    sky_mix = np.clip((vertical + 1.0) * 0.5, 0.0, 1.0)[..., None]
    ambient = ((1.0 - sky_mix) * horizon + sky_mix * zenith) * ambient_strength * (1.0 + 0.15 * scattering)
    ambient[vertical < 0.0] = ground * max(ambient_strength * 0.6, 0.03)
    return np.asarray(np.clip(ambient, 0.0, None), dtype=np.float32)


def _unity_sun_strength(main_light: Mapping[str, Any], scale: float) -> float:
    intensity = max(float(main_light.get("intensity", 0.0) or 0.0), 0.0)
    indirect_multiplier = max(float(main_light.get("indirectMultiplier", 1.0) or 1.0), 0.0)
    return float(
        np.clip(
            intensity * indirect_multiplier * scale,
            0.0,
            _VALIDATION_LIMITS["max_sun_strength"],
        )
    )


def _unity_resolve_ambient_probe(
    dataset: Any,
) -> tuple[str, np.ndarray, Mapping[str, Any] | None]:
    if dataset.has_frame_lighting():
        for frame_index in dataset.frame_indices():
            payload = dataset.frame_lighting(frame_index)
            probe_samples = payload.get("probeSamples", [])
            if not isinstance(probe_samples, Sequence):
                continue
            preferred = _unity_preferred_probe_sample(probe_samples)
            if preferred is not None:
                coeffs = preferred.get("sh", {}).get("coefficientsRGB27", [])
                arr = np.asarray(coeffs, dtype=np.float32).reshape(-1)
                if arr.size == 27 and np.isfinite(arr).all():
                    return "frame_probe_sample", arr, preferred
    ambient_probe = dataset.scene_lights().get("ambientProbe", {})
    if isinstance(ambient_probe, Mapping):
        coeffs = ambient_probe.get("coefficientsRGB27", [])
        arr = np.asarray(coeffs, dtype=np.float32).reshape(-1)
        if arr.size == 27 and np.isfinite(arr).all():
            return "scene_ambient_probe", arr, None
    raise ValueError("Unity lighting GT is missing valid ambient probe SH coefficients.")


def _unity_preferred_probe_sample(
    probe_samples: Sequence[Any],
) -> Mapping[str, Any] | None:
    normalized = [item for item in probe_samples if isinstance(item, Mapping)]
    if not normalized:
        return None
    for preferred_label in ("subject_anchor", "camera"):
        for item in normalized:
            if str(item.get("label", "")).strip().lower() == preferred_label:
                return item
    return normalized[0]


def _unity_ambient_strength(coefficients: np.ndarray, *, scale: float) -> float:
    arr = np.asarray(coefficients, dtype=np.float32).reshape(-1)
    if arr.size != 27 or not np.isfinite(arr).all():
        raise ValueError("Unity ambient probe coefficients must contain 27 finite values.")
    l0 = np.asarray((arr[0], arr[9], arr[18]), dtype=np.float32)
    luminance = float(np.dot(np.clip(l0, 0.0, None), np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)))
    return float(
        np.clip(
            luminance * scale,
            _VALIDATION_LIMITS["min_ambient_strength"],
            _VALIDATION_LIMITS["max_ambient_strength"],
        )
    )


def _validate_unity_envmap(hdr: np.ndarray) -> dict[str, Any]:
    luminance = _luminance(np.asarray(hdr, dtype=np.float32))
    mean = float(np.mean(luminance))
    p95 = float(np.quantile(luminance, 0.95))
    max_value = float(np.max(luminance))
    failures: list[str] = []
    if mean < _VALIDATION_LIMITS["min_mean_luminance"]:
        failures.append("mean_luminance_too_low")
    if p95 < _VALIDATION_LIMITS["min_p95_luminance"]:
        failures.append("p95_luminance_too_low")
    if max_value < _VALIDATION_LIMITS["min_max_luminance"]:
        failures.append("max_luminance_too_low")
    quality = float(np.clip(max_value / max(mean, 1e-6), 0.0, 10.0) / 10.0)
    return {
        "passed": not failures,
        "failures": failures,
        "quality_envmap": quality,
        "mean_luminance": mean,
        "p95_luminance": p95,
        "max_luminance": max_value,
    }


def _unity_reflection_faces_to_latlong(
    *,
    reflection_faces: Mapping[str, Path],
    height: int,
    width: int,
) -> np.ndarray:
    faces = {
        name: _load_rgb_exr_face(reflection_faces[name]) for name in _UNITY_REFLECTION_FACE_ORDER
    }
    latlong = np.zeros((height, width, 3), dtype=np.float32)
    theta = (np.arange(width, dtype=np.float32) + 0.5) / float(width) * (2.0 * math.pi) - math.pi
    phi = (np.arange(height, dtype=np.float32) + 0.5) / float(height) * math.pi
    sin_phi = np.sin(phi)[:, None]
    cos_phi = np.cos(phi)[:, None]
    sin_theta = np.sin(theta)[None, :]
    cos_theta = np.cos(theta)[None, :]
    directions = np.stack(
        (
            sin_phi * cos_theta,
            cos_phi * np.ones_like(theta, dtype=np.float32)[None, :],
            sin_phi * sin_theta,
        ),
        axis=-1,
    )
    for y in range(height):
        for x in range(width):
            latlong[y, x] = _sample_unity_cubemap(faces, directions[y, x])
    return np.asarray(np.clip(latlong, 0.0, None), dtype=np.float32)


def _sample_unity_cubemap(faces: Mapping[str, np.ndarray], direction: np.ndarray) -> np.ndarray:
    x, y, z = (float(v) for v in _normalize(direction))
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        if x >= 0.0:
            face = "PositiveX"
            sc, tc, ma = -z, -y, ax
        else:
            face = "NegativeX"
            sc, tc, ma = z, -y, ax
    elif ay >= ax and ay >= az:
        if y >= 0.0:
            face = "PositiveY"
            sc, tc, ma = x, z, ay
        else:
            face = "NegativeY"
            sc, tc, ma = x, -z, ay
    else:
        if z >= 0.0:
            face = "PositiveZ"
            sc, tc, ma = x, -y, az
        else:
            face = "NegativeZ"
            sc, tc, ma = -x, -y, az
    u = 0.5 * (sc / max(ma, 1e-8) + 1.0)
    v = 0.5 * (tc / max(ma, 1e-8) + 1.0)
    image = faces[face]
    height, width = image.shape[:2]
    px = int(np.clip(round(u * (width - 1)), 0, width - 1))
    py = int(np.clip(round((1.0 - v) * (height - 1)), 0, height - 1))
    return np.asarray(image[py, px], dtype=np.float32).reshape(3)


def _load_rgb_exr_face(path: Path) -> np.ndarray:
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore
    except Exception:
        OpenEXR = None  # type: ignore[assignment]
        Imath = None  # type: ignore[assignment]
    if OpenEXR is not None and Imath is not None:
        exr = OpenEXR.InputFile(str(path))
        try:
            header = exr.header()
            window = header["dataWindow"]
            width = int(window.max.x - window.min.x + 1)
            height = int(window.max.y - window.min.y + 1)
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = []
            for name in ("R", "G", "B"):
                raw = exr.channel(name, pixel_type)
                channels.append(np.frombuffer(raw, dtype=np.float32).copy().reshape(height, width))
            return np.stack(channels, axis=-1)
        finally:
            close = getattr(exr, "close", None)
            if callable(close):
                close()
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unity reflection face EXR is unreadable: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] >= 3:
        image = image[..., :3][:, :, ::-1]
    return np.asarray(image, dtype=np.float32)


def register_lighting_provider_builders(factory: ProviderFactory) -> None:
    factory.register(
        "UnityGTLightingProvider",
        lambda binding, context: UnityGTLightingProvider(binding.settings),
    )
    factory.register(
        "CarlaGTLightingProvider",
        lambda binding, context: CarlaGTLightingProvider(binding.settings),
    )
    factory.register(
        "DiffusionLightTurboLightingProvider",
        lambda binding, context: DiffusionLightTurboLightingProvider(binding.settings),
    )


def _prepare_model_image(image: np.ndarray, size: int) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)[..., :3]
    pil = Image.fromarray(rgb)
    scale = min(size / pil.size[0], size / pil.size[1])
    resized = pil.resize(
        (
            max(1, int(round(pil.size[0] * scale))),
            max(1, int(round(pil.size[1] * scale))),
        ),
        Image.Resampling.BICUBIC,
    )
    resized_arr = np.asarray(resized, dtype=np.uint8)
    pad_h = size - resized_arr.shape[0]
    pad_w = size - resized_arr.shape[1]
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(
        resized_arr,
        ((top, bottom), (left, right), (0, 0)),
        mode="edge",
    )

def _mean_metric(items: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(item.get(key, 0.0)) for item in items if np.isfinite(float(item.get(key, 0.0)))]
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _should_demote_to_fill_heavy(
    *,
    diffuse_demote_enabled: bool,
    aggressiveness: str,
    diffuse_scene_score: float,
    direct_scene_score: float,
    direct_to_fill_ratio: float,
    max_direct_to_fill_ratio_for_diffuse: float,
    fill_count: int,
    min_fill_count: int,
) -> bool:
    if not diffuse_demote_enabled or fill_count <= 0:
        return False
    score_margin = diffuse_scene_score - direct_scene_score
    ratio_violation = direct_to_fill_ratio > max_direct_to_fill_ratio_for_diffuse
    if aggressiveness == "off":
        return False
    if aggressiveness == "aggressive":
        return bool(
            diffuse_scene_score >= 0.60
            and (score_margin >= 0.05 or ratio_violation)
            and fill_count >= max(1, min_fill_count - 1)
        )
    return bool(
        diffuse_scene_score >= 0.70
        and (score_margin >= 0.10 or (ratio_violation and fill_count >= min_fill_count))
    )


def _rebalance_fill_heavy_rig(
    *,
    sun_strength: float,
    fill_lights: Sequence[Mapping[str, Any]],
    direct_scale: float,
    max_direct_to_fill_ratio: float,
    ambient_strength: float,
    analytic_energy_before: float,
) -> tuple[float, list[dict[str, Any]]]:
    fills = [dict(item) for item in fill_lights]
    if not fills:
        return 0.0, []
    demoted_sun = float(max(sun_strength * direct_scale, 0.0))
    current_fill_total = float(sum(max(float(item.get("strength", 0.0)), 0.0) for item in fills))
    target_analytic_energy = max(
        analytic_energy_before * _MIN_FILL_HEAVY_BRIGHTNESS_PRESERVATION,
        ambient_strength * 10.0,
    )
    target_fill_total = max(
        current_fill_total,
        demoted_sun / max(max_direct_to_fill_ratio, 1e-6),
        target_analytic_energy - demoted_sun,
    )
    scale = target_fill_total / max(current_fill_total, 1e-6)
    for item in fills:
        item["strength"] = float(max(float(item.get("strength", 0.0)) * scale, 0.0))
    return demoted_sun, fills


def _representative_view_direction(
    estimates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    weighted = np.zeros((3,), dtype=np.float32)
    total_weight = 0.0
    for item in estimates:
        rotation = np.asarray(item.get("rotation_c2w"), dtype=np.float32)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            continue
        view_direction = _normalize(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        weight = float(item.get("weight", item.get("estimate_quality", 1.0)))
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        weighted += view_direction * weight
        total_weight += weight
    if total_weight <= 1e-6:
        return np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    return _normalize(weighted / total_weight)


def _view_horizontal_axes(
    view_direction_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    view_horizontal = np.asarray(view_direction_world[:2], dtype=np.float32)
    view_horizontal_norm = float(np.linalg.norm(view_horizontal))
    if view_horizontal_norm <= 1e-6:
        view_horizontal = np.asarray([0.0, -1.0], dtype=np.float32)
        view_horizontal_norm = 1.0
    view_horizontal = view_horizontal / view_horizontal_norm
    side_axis = np.asarray([-view_horizontal[1], view_horizontal[0]], dtype=np.float32)
    return view_horizontal, side_axis


def _normalize_xy(vector_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector_xy, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-6:
        return np.asarray([0.0, -1.0], dtype=np.float32)
    return arr / norm


def _blend_horizontal_direction(
    primary_xy: np.ndarray,
    secondary_xy: np.ndarray | None,
    secondary_weight: float,
) -> np.ndarray:
    primary = _normalize_xy(primary_xy)
    if secondary_xy is None:
        return primary
    secondary = _normalize_xy(secondary_xy)
    return _normalize_xy(
        primary * float(max(1.0 - secondary_weight, 0.0))
        + secondary * float(max(secondary_weight, 0.0))
    )


def _horizontal_to_world(
    horizontal_xy: np.ndarray,
    *,
    elevation_deg: float,
) -> np.ndarray:
    horizontal = _normalize_xy(horizontal_xy)
    elevation_rad = math.radians(float(elevation_deg))
    horizontal_scale = float(max(math.cos(elevation_rad), 1e-6))
    return _normalize(
        np.asarray(
            [
                horizontal[0] * horizontal_scale,
                horizontal[1] * horizontal_scale,
                math.sin(elevation_rad),
            ],
            dtype=np.float32,
        )
    )


def _point_to_light_to_offset(
    point_to_light: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    return _normalize(np.asarray(point_to_light, dtype=np.float32)) * float(distance_m)


def _view_space_angles(
    point_to_light: np.ndarray,
    *,
    view_horizontal: np.ndarray,
    side_axis: np.ndarray,
) -> tuple[float, float]:
    direction = _normalize(np.asarray(point_to_light, dtype=np.float32))
    horizontal = _normalize_xy(direction[:2])
    azimuth_deg = float(
        math.degrees(
            math.atan2(
                float(np.dot(horizontal, side_axis)),
                float(np.dot(horizontal, view_horizontal)),
            )
        )
    )
    elevation_deg = float(
        math.degrees(
            math.atan2(
                float(direction[2]),
                max(float(np.linalg.norm(direction[:2])), 1e-6),
            )
        )
    )
    return azimuth_deg, elevation_deg


def _horizontal_angle_deg(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    aa = _normalize_xy(a_xy)
    bb = _normalize_xy(b_xy)
    return float(
        math.degrees(
            math.acos(float(np.clip(np.dot(aa, bb), -1.0, 1.0)))
        )
    )


def _wrap_role_strength_scales(
    fill_lights: Sequence[Mapping[str, Any]],
    *,
    side_axis: np.ndarray,
    softness_bias: float,
) -> dict[str, float]:
    weighted_bias = 0.0
    total_strength = 0.0
    for item in fill_lights:
        strength = max(float(item.get("strength", 0.0)), 0.0)
        if strength <= 0.0:
            continue
        point_to_light = _normalize(
            -np.asarray(item.get("direction_world", (0.0, 0.0, -1.0)), dtype=np.float32)
        )
        weighted_bias += strength * abs(float(np.dot(point_to_light[:2], side_axis)))
        total_strength += strength
    lateral_bias = weighted_bias / max(total_strength, 1e-6)
    if lateral_bias >= 0.55:
        return {
            "wrap_key_fill": float(np.clip(0.24 - 0.04 * softness_bias, 0.12, 0.32)),
            "counter_wrap_fill": float(np.clip(0.48 + 0.05 * softness_bias, 0.32, 0.62)),
            "sky_fill": float(np.clip(0.28 - 0.01 * softness_bias, 0.20, 0.36)),
        }
    return {
        "wrap_key_fill": float(np.clip(0.28 - 0.05 * softness_bias, 0.14, 0.34)),
        "counter_wrap_fill": float(np.clip(0.44 + 0.06 * softness_bias, 0.32, 0.62)),
        "sky_fill": float(np.clip(0.28 - 0.01 * softness_bias, 0.20, 0.36)),
    }


def _fixed_wrap_geometry_projection(
    fill_lights: Sequence[Mapping[str, Any]],
    *,
    view_direction_world: np.ndarray,
    softness_bias: float = 0.0,
) -> list[dict[str, Any]]:
    if not fill_lights:
        return []
    projected: list[dict[str, Any]] = []
    view_horizontal, side_axis = _view_horizontal_axes(view_direction_world)
    front_offset = np.asarray(
        [
            view_horizontal[0] * _WRAP_FILL_FRONT_OFFSET_M[0] + side_axis[0] * _WRAP_FILL_FRONT_OFFSET_M[1],
            view_horizontal[1] * _WRAP_FILL_FRONT_OFFSET_M[0] + side_axis[1] * _WRAP_FILL_FRONT_OFFSET_M[1],
            _WRAP_FILL_FRONT_OFFSET_M[2],
        ],
        dtype=np.float32,
    )
    counter_offset = np.asarray(
        [
            view_horizontal[0] * _WRAP_FILL_COUNTER_OFFSET_M[0] + side_axis[0] * _WRAP_FILL_COUNTER_OFFSET_M[1],
            view_horizontal[1] * _WRAP_FILL_COUNTER_OFFSET_M[0] + side_axis[1] * _WRAP_FILL_COUNTER_OFFSET_M[1],
            _WRAP_FILL_COUNTER_OFFSET_M[2],
        ],
        dtype=np.float32,
    )
    sky_offset = np.asarray(
        [
            view_horizontal[0] * _WRAP_FILL_SKY_OFFSET_M[0] + side_axis[0] * _WRAP_FILL_SKY_OFFSET_M[1],
            view_horizontal[1] * _WRAP_FILL_SKY_OFFSET_M[0] + side_axis[1] * _WRAP_FILL_SKY_OFFSET_M[1],
            _WRAP_FILL_SKY_OFFSET_M[2],
        ],
        dtype=np.float32,
    )
    base_lights = sorted(fill_lights, key=lambda item: float(item.get("strength", 0.0)), reverse=True)
    primary = dict(base_lights[0])
    secondary = dict(base_lights[1] if len(base_lights) > 1 else base_lights[0])
    tertiary = dict(base_lights[2] if len(base_lights) > 2 else base_lights[-1])
    wrap_specs = (
        ("wrap_key_fill", primary, front_offset),
        ("counter_wrap_fill", secondary, counter_offset),
        ("sky_fill", tertiary, sky_offset),
    )
    total_strength = float(sum(max(float(item.get("strength", 0.0)), 0.0) for item in base_lights))
    strength_scales = _wrap_role_strength_scales(
        base_lights,
        side_axis=side_axis,
        softness_bias=softness_bias,
    )
    for role, fill, offset in wrap_specs:
        subject_to_light = _normalize(np.asarray(offset, dtype=np.float32))
        item = dict(fill)
        item["kind"] = "POINT"
        item["role"] = role
        item["direction_world"] = -subject_to_light
        item["placement_mode"] = "subject_anchor_relative"
        item["placement_target"] = "subject_root_dynamic"
        item["location_world"] = np.asarray(offset, dtype=np.float32)
        item.pop("area_size", None)
        scaled_strength = total_strength * strength_scales[role]
        item["strength"] = float(max(scaled_strength, float(fill.get("strength", 0.0)) * 0.65))
        diagnostics = dict(item.get("diagnostics", {}))
        diagnostics["transport_mode"] = "wrap_subject_fill"
        diagnostics["raw_direction_world"] = np.asarray(fill.get("direction_world", (0.0, 0.0, -1.0)), dtype=np.float32).reshape(3).tolist()
        diagnostics["view_alignment"] = float(np.dot(np.asarray(subject_to_light[:2], dtype=np.float32), view_horizontal))
        diagnostics["geometry_strategy"] = "fixed_template"
        item["diagnostics"] = diagnostics
        projected.append(item)
    return projected


def _plan_subject_wrap_geometry(
    fill_lights: Sequence[Mapping[str, Any]],
    *,
    view_direction_world: np.ndarray,
    min_azimuth_separation_deg: float,
    counter_opposition_deg: float,
    sky_min_elevation_deg: float,
    candidate_count_per_role: int,
    sun_strength: float,
    ambient_strength: float,
    softness_bias: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not fill_lights:
        return [], {
            "winning_candidate_index": None,
            "winning_candidate_score": 0.0,
            "candidate_count": 0,
            "best_dark_to_bright_ratio": 0.0,
            "best_camera_side_fill_balance": 0.0,
            "dark_to_bright_ratio_before_geometry": 0.0,
            "dark_to_bright_ratio_after_geometry": 0.0,
            "camera_side_fill_balance_before_geometry": 0.0,
            "camera_side_fill_balance_after_geometry": 0.0,
            "role_geometry": {},
            "strategy": "none",
        }
    view_horizontal, side_axis = _view_horizontal_axes(view_direction_world)
    baseline = _fixed_wrap_geometry_projection(
        fill_lights,
        view_direction_world=view_direction_world,
        softness_bias=softness_bias,
    )
    baseline_metrics = _subject_probe_metrics(
        sun_strength=sun_strength,
        fill_lights=baseline,
        ambient_strength=ambient_strength,
        view_direction_world=view_direction_world,
    )
    base_lights = sorted(fill_lights, key=lambda item: float(item.get("strength", 0.0)), reverse=True)
    primary = dict(base_lights[0])
    secondary = dict(base_lights[1] if len(base_lights) > 1 else base_lights[0])
    tertiary = dict(base_lights[2] if len(base_lights) > 2 else base_lights[-1])
    lobe_directions = [
        _normalize(-np.asarray(item.get("direction_world", (0.0, 0.0, -1.0)), dtype=np.float32))
        for item in base_lights
    ]
    strongest_lobe_xy = _normalize_xy(lobe_directions[0][:2])
    side_sign = 1.0 if float(np.dot(strongest_lobe_xy, side_axis)) >= 0.0 else -1.0
    preferred_counter_xy = _normalize_xy(
        view_horizontal * 0.12 - side_axis * side_sign * 1.0
    )
    strength_scales = _wrap_role_strength_scales(
        base_lights,
        side_axis=side_axis,
        softness_bias=softness_bias,
    )
    total_strength = float(sum(max(float(item.get("strength", 0.0)), 0.0) for item in base_lights))
    key_distance = float(np.linalg.norm(_WRAP_FILL_FRONT_OFFSET_M))
    counter_distance = float(np.linalg.norm(_WRAP_FILL_COUNTER_OFFSET_M))
    sky_distance = float(np.linalg.norm(_WRAP_FILL_SKY_OFFSET_M))

    candidate_count = max(int(candidate_count_per_role), 1)
    key_laterals = np.linspace(0.35, 0.95, candidate_count, dtype=np.float32)
    counter_forwards = np.linspace(-0.05, 0.25, candidate_count, dtype=np.float32)
    counter_laterals = np.linspace(0.9, 1.35, candidate_count, dtype=np.float32)
    sky_laterals = np.linspace(0.0, 0.3, candidate_count, dtype=np.float32)
    sky_elevations = np.linspace(
        float(sky_min_elevation_deg),
        min(float(sky_min_elevation_deg) + 15.0, 82.0),
        candidate_count,
        dtype=np.float32,
    )

    best_projected = baseline
    best_metrics = baseline_metrics
    best_score = float("-inf")
    best_index: int | None = None
    best_role_geometry: dict[str, Any] = {}
    candidate_index = 0

    for key_lat in key_laterals:
        key_xy = _blend_horizontal_direction(
            view_horizontal + side_axis * side_sign * float(key_lat),
            strongest_lobe_xy,
            0.28,
        )
        key_world = _horizontal_to_world(key_xy, elevation_deg=26.0)
        key_offset = _point_to_light_to_offset(key_world, key_distance)
        for counter_lat in counter_laterals:
            for counter_fwd in counter_forwards:
                counter_xy = _blend_horizontal_direction(
                    view_horizontal * float(counter_fwd) - side_axis * side_sign * float(counter_lat),
                    preferred_counter_xy,
                    0.32,
                )
                azimuth_separation = _horizontal_angle_deg(key_xy, counter_xy)
                if azimuth_separation + 1e-6 < float(min_azimuth_separation_deg):
                    continue
                if azimuth_separation + 1e-6 < float(counter_opposition_deg):
                    continue
                counter_world = _horizontal_to_world(counter_xy, elevation_deg=34.0)
                counter_offset = _point_to_light_to_offset(counter_world, counter_distance)
                for sky_lat, sky_elevation in zip(sky_laterals, sky_elevations):
                    sky_xy = _blend_horizontal_direction(
                        view_horizontal * 0.08 - side_axis * side_sign * float(sky_lat),
                        preferred_counter_xy,
                        0.15,
                    )
                    sky_world = _horizontal_to_world(
                        sky_xy,
                        elevation_deg=float(sky_elevation),
                    )
                    role_specs = (
                        ("wrap_key_fill", primary, key_offset, key_world),
                        ("counter_wrap_fill", secondary, counter_offset, counter_world),
                        ("sky_fill", tertiary, _point_to_light_to_offset(sky_world, sky_distance), sky_world),
                    )
                    projected: list[dict[str, Any]] = []
                    for role, fill, offset, point_to_light in role_specs:
                        item = dict(fill)
                        item["kind"] = "POINT"
                        item["role"] = role
                        item["direction_world"] = -_normalize(point_to_light)
                        item["placement_mode"] = "subject_anchor_relative"
                        item["placement_target"] = "subject_root_dynamic"
                        item["location_world"] = np.asarray(offset, dtype=np.float32)
                        item.pop("area_size", None)
                        item["strength"] = float(
                            max(
                                total_strength * strength_scales[role],
                                float(fill.get("strength", 0.0)) * 0.65,
                            )
                        )
                        diagnostics = dict(item.get("diagnostics", {}))
                        diagnostics["transport_mode"] = "wrap_subject_fill"
                        diagnostics["raw_direction_world"] = (
                            np.asarray(
                                fill.get("direction_world", (0.0, 0.0, -1.0)),
                                dtype=np.float32,
                            )
                            .reshape(3)
                            .tolist()
                        )
                        diagnostics["view_alignment"] = float(
                            np.dot(np.asarray(point_to_light[:2], dtype=np.float32), view_horizontal)
                        )
                        diagnostics["geometry_strategy"] = "hybrid_adaptive"
                        item["diagnostics"] = diagnostics
                        projected.append(item)
                    metrics = _subject_probe_metrics(
                        sun_strength=sun_strength,
                        fill_lights=projected,
                        ambient_strength=ambient_strength,
                        view_direction_world=view_direction_world,
                    )
                    score = (
                        metrics["dark_to_bright_ratio"] * (8.0 + 2.0 * softness_bias)
                        + metrics["camera_side_fill_balance"] * (4.0 + 1.5 * softness_bias)
                        + min(metrics["view_facing"], baseline_metrics["view_facing"]) * 0.08
                        + metrics["dark_side"] * (0.35 + 0.10 * softness_bias)
                        - max(0.0, metrics["direct_to_diffuse_ratio"] - (0.14 - 0.04 * softness_bias)) * 6.0
                        - max(0.0, float(min_azimuth_separation_deg) - azimuth_separation) * 0.1
                    )
                    if score <= best_score:
                        candidate_index += 1
                        continue
                    best_score = float(score)
                    best_projected = projected
                    best_metrics = metrics
                    best_index = int(candidate_index)
                    best_role_geometry = {}
                    for role, _, offset, point_to_light in role_specs:
                        azimuth_deg, elevation_deg = _view_space_angles(
                            point_to_light,
                            view_horizontal=view_horizontal,
                            side_axis=side_axis,
                        )
                        best_role_geometry[role] = {
                            "location_world": np.asarray(offset, dtype=np.float32).tolist(),
                            "point_to_light_world": _normalize(point_to_light).tolist(),
                            "azimuth_deg": float(azimuth_deg),
                            "elevation_deg": float(elevation_deg),
                        }
                    candidate_index += 1

    diagnostics = {
        "winning_candidate_index": best_index,
        "winning_candidate_score": float(0.0 if not np.isfinite(best_score) else best_score),
        "candidate_count": int(candidate_index),
        "best_dark_to_bright_ratio": float(best_metrics["dark_to_bright_ratio"]),
        "best_camera_side_fill_balance": float(best_metrics["camera_side_fill_balance"]),
        "dark_to_bright_ratio_before_geometry": float(baseline_metrics["dark_to_bright_ratio"]),
        "dark_to_bright_ratio_after_geometry": float(best_metrics["dark_to_bright_ratio"]),
        "camera_side_fill_balance_before_geometry": float(
            baseline_metrics["camera_side_fill_balance"]
        ),
        "camera_side_fill_balance_after_geometry": float(
            best_metrics["camera_side_fill_balance"]
        ),
        "role_geometry": best_role_geometry,
        "strategy": "hybrid_adaptive",
    }
    return best_projected, diagnostics


def _subject_probe_metrics(
    *,
    sun_strength: float,
    fill_lights: Sequence[Mapping[str, Any]],
    ambient_strength: float,
    view_direction_world: np.ndarray,
) -> dict[str, float]:
    view_horizontal = np.asarray(view_direction_world[:2], dtype=np.float32)
    view_horizontal_norm = float(np.linalg.norm(view_horizontal))
    if view_horizontal_norm <= 1e-6:
        view_horizontal = np.asarray([0.0, -1.0], dtype=np.float32)
        view_horizontal_norm = 1.0
    view_horizontal /= view_horizontal_norm
    side_axis = np.asarray([-view_horizontal[1], view_horizontal[0]], dtype=np.float32)
    probes = {
        "view_facing": np.asarray([view_horizontal[0], view_horizontal[1], 0.0], dtype=np.float32),
        "camera_left": np.asarray([side_axis[0], side_axis[1], 0.0], dtype=np.float32),
        "camera_right": np.asarray([-side_axis[0], -side_axis[1], 0.0], dtype=np.float32),
        "back": np.asarray([-view_horizontal[0], -view_horizontal[1], 0.0], dtype=np.float32),
        "up": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }
    visible_side_values: list[float] = []
    total_values: list[float] = []
    total_direct = 0.0
    total_diffuse = 0.0
    fill_alignments: list[float] = []
    left_value = 0.0
    right_value = 0.0
    view_facing_value = 0.0
    for name, normal in probes.items():
        direct_component = float(sun_strength * (0.18 if name == "up" else 0.12))
        diffuse_component = float(ambient_strength)
        if name == "up":
            diffuse_component += float(ambient_strength * 0.35)
        for light in fill_lights:
            direction = _normalize(np.asarray(light.get("direction_world", (0.0, 0.0, -1.0)), dtype=np.float32))
            point_to_light = -direction
            diffuse_component += float(max(np.dot(normal, point_to_light), 0.0) * float(light.get("strength", 0.0)))
            fill_alignments.append(float(np.dot(np.asarray(point_to_light[:2], dtype=np.float32), view_horizontal)))
        total = direct_component + diffuse_component
        if name in {"view_facing", "camera_left", "camera_right"}:
            visible_side_values.append(total)
        if name == "view_facing":
            view_facing_value = total
        elif name == "camera_left":
            left_value = total
        elif name == "camera_right":
            right_value = total
        total_values.append(total)
        total_direct += direct_component
        total_diffuse += diffuse_component
    return {
        "total": float(np.mean(np.asarray(total_values, dtype=np.float32))) if total_values else 0.0,
        "dark_side": float(min(visible_side_values)) if visible_side_values else 0.0,
        "bright_side": float(max(visible_side_values)) if visible_side_values else 0.0,
        "dark_to_bright_ratio": (
            float(min(visible_side_values) / max(max(visible_side_values), 1e-6))
            if visible_side_values
            else 0.0
        ),
        "view_facing": float(view_facing_value),
        "camera_side_fill_balance": float(min(left_value, right_value) / max(max(left_value, right_value), 1e-6)),
        "fill_view_alignment_score": float(max(fill_alignments)) if fill_alignments else 0.0,
        "direct_to_diffuse_ratio": float(total_direct / max(total_diffuse, 1e-6)),
    }


def _apply_subject_fill_targets(
    *,
    sun_strength: float,
    fill_lights: Sequence[Mapping[str, Any]],
    ambient_strength: float,
    analytic_energy_before: float,
    planner_mode: str,
    view_direction_world: np.ndarray,
    fill_heavy_transport_gain: float,
    fill_heavy_dark_side_target_ratio: float,
    diffuse_softness_bias: float,
    wrap_geometry_min_azimuth_separation_deg: float,
    wrap_geometry_counter_opposition_deg: float,
    wrap_geometry_sky_min_elevation_deg: float,
    wrap_geometry_candidate_count_per_role: int,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    fills, geometry_diagnostics = _plan_subject_wrap_geometry(
        fill_lights,
        view_direction_world=view_direction_world,
        min_azimuth_separation_deg=wrap_geometry_min_azimuth_separation_deg,
        counter_opposition_deg=wrap_geometry_counter_opposition_deg,
        sky_min_elevation_deg=wrap_geometry_sky_min_elevation_deg,
        candidate_count_per_role=wrap_geometry_candidate_count_per_role,
        sun_strength=sun_strength,
        ambient_strength=ambient_strength,
        softness_bias=diffuse_softness_bias,
    )
    before = _subject_probe_metrics(
        sun_strength=sun_strength,
        fill_lights=fills,
        ambient_strength=ambient_strength,
        view_direction_world=view_direction_world,
    )
    softness_gain = 1.0 + 0.20 * float(diffuse_softness_bias)
    transport_gain = (
        float(fill_heavy_transport_gain) * softness_gain if planner_mode == "fill_heavy" else 1.0
    )
    effective_dark_ratio_target = float(
        np.clip(
            float(fill_heavy_dark_side_target_ratio) + 0.08 * float(diffuse_softness_bias),
            0.1,
            0.95,
        )
    )
    target_direct_to_diffuse_ratio = float(
        np.clip(0.16 - 0.06 * float(diffuse_softness_bias), 0.05, 0.25)
    )
    base_target_total = max(
        before["total"],
        analytic_energy_before * (0.34 if planner_mode == "fill_heavy" else 0.22),
        ambient_strength * 10.0,
    )
    target_total = base_target_total * transport_gain
    base_target_dark = max(
        before["dark_side"],
        analytic_energy_before * (0.045 if planner_mode == "fill_heavy" else 0.03),
        base_target_total * (0.13 if planner_mode == "fill_heavy" else 0.10),
        ambient_strength * 2.5,
    )
    target_dark = base_target_dark * transport_gain
    base_target_view_facing = max(
        before["view_facing"],
        base_target_total * (0.26 if planner_mode == "fill_heavy" else 0.18),
        base_target_dark * 1.35,
    )
    target_view_facing = base_target_view_facing * transport_gain
    ambient_out = float(ambient_strength)
    reasons: list[str] = []
    if before["dark_side"] < target_dark:
        boosted = min(
            _MAX_DIFFUSE_WORLD_STRENGTH,
            ambient_out + max(target_dark - before["dark_side"], 0.0) * 0.35,
        )
        if boosted > ambient_out + 1e-6:
            ambient_out = boosted
            reasons.append("raised_world_env_strength_for_subject_fill")
    after_world = _subject_probe_metrics(
        sun_strength=sun_strength,
        fill_lights=fills,
        ambient_strength=ambient_out,
        view_direction_world=view_direction_world,
    )
    fill_scale = max(
        1.0,
        target_total / max(after_world["total"], 1e-6),
        target_dark / max(after_world["dark_side"], 1e-6),
        target_view_facing / max(after_world["view_facing"], 1e-6),
    )
    if fill_scale > 1.0 + 1e-6:
        fill_scale = min(fill_scale, 2.5)
        for item in fills:
            item["strength"] = float(max(float(item.get("strength", 0.0)) * fill_scale, 0.0))
        reasons.append("scaled_subject_relative_fills_for_transport_targets")
    after = _subject_probe_metrics(
        sun_strength=sun_strength,
        fill_lights=fills,
        ambient_strength=ambient_out,
        view_direction_world=view_direction_world,
    )
    diagnostics = {
        "fill_transport_mode": "wrap_subject_fill",
        "view_direction_world": view_direction_world.tolist(),
        "subject_total_irradiance_before": float(before["total"]),
        "subject_total_irradiance_after": float(after["total"]),
        "subject_dark_side_irradiance_before": float(before["dark_side"]),
        "subject_dark_side_irradiance_after": float(after["dark_side"]),
        "dark_to_bright_ratio_before": float(before["dark_to_bright_ratio"]),
        "dark_to_bright_ratio_after": float(after["dark_to_bright_ratio"]),
        "view_facing_irradiance_before": float(before["view_facing"]),
        "view_facing_irradiance_after": float(after["view_facing"]),
        "direct_to_diffuse_subject_ratio_before": float(before["direct_to_diffuse_ratio"]),
        "direct_to_diffuse_subject_ratio_after": float(after["direct_to_diffuse_ratio"]),
        "camera_side_fill_balance_before": float(before["camera_side_fill_balance"]),
        "camera_side_fill_balance_after": float(after["camera_side_fill_balance"]),
        "fill_view_alignment_score_before": float(before["fill_view_alignment_score"]),
        "fill_view_alignment_score_after": float(after["fill_view_alignment_score"]),
        "world_strength_before": float(ambient_strength),
        "world_strength_after": float(ambient_out),
        "fill_target_total_irradiance": float(target_total),
        "fill_target_dark_side_irradiance": float(target_dark),
        "fill_target_view_facing_irradiance": float(target_view_facing),
        "fill_heavy_transport_gain": float(fill_heavy_transport_gain),
        "fill_heavy_dark_side_target_ratio": float(fill_heavy_dark_side_target_ratio),
        "diffuse_softness_bias": float(diffuse_softness_bias),
        "effective_fill_heavy_transport_gain": float(transport_gain),
        "effective_fill_heavy_dark_side_target_ratio": float(effective_dark_ratio_target),
        "target_direct_to_diffuse_subject_ratio": float(target_direct_to_diffuse_ratio),
        "wrap_geometry_min_azimuth_separation_deg": float(
            wrap_geometry_min_azimuth_separation_deg
        ),
        "wrap_geometry_counter_opposition_deg": float(
            wrap_geometry_counter_opposition_deg
        ),
        "wrap_geometry_sky_min_elevation_deg": float(
            wrap_geometry_sky_min_elevation_deg
        ),
        "wrap_geometry_candidate_count_per_role": int(
            wrap_geometry_candidate_count_per_role
        ),
        "dark_to_bright_ratio_before_geometry": float(
            geometry_diagnostics["dark_to_bright_ratio_before_geometry"]
        ),
        "dark_to_bright_ratio_after_geometry": float(
            geometry_diagnostics["dark_to_bright_ratio_after_geometry"]
        ),
        "camera_side_fill_balance_before_geometry": float(
            geometry_diagnostics["camera_side_fill_balance_before_geometry"]
        ),
        "camera_side_fill_balance_after_geometry": float(
            geometry_diagnostics["camera_side_fill_balance_after_geometry"]
        ),
        "geometry_candidate_count": int(geometry_diagnostics["candidate_count"]),
        "geometry_winning_candidate_index": geometry_diagnostics["winning_candidate_index"],
        "geometry_winning_candidate_score": float(
            geometry_diagnostics["winning_candidate_score"]
        ),
        "geometry_best_dark_to_bright_ratio": float(
            geometry_diagnostics["best_dark_to_bright_ratio"]
        ),
        "geometry_best_camera_side_fill_balance": float(
            geometry_diagnostics["best_camera_side_fill_balance"]
        ),
        "role_geometry": dict(geometry_diagnostics["role_geometry"]),
        "transport_validation_passed": bool(
            (after["total"] + 1e-6 >= target_total * 0.85)
            and (after["dark_side"] + 1e-6 >= target_dark * 0.75)
            and (after["view_facing"] + 1e-6 >= target_view_facing * 0.80)
            and (
                after["dark_to_bright_ratio"] + 1e-6
                >= float(effective_dark_ratio_target)
            )
            and (after["direct_to_diffuse_ratio"] <= float(target_direct_to_diffuse_ratio) + 1e-6)
            and (after["total"] > before["total"])
            and (after["dark_side"] > before["dark_side"])
            and (after["view_facing"] > before["view_facing"])
        ),
        "reasons": reasons,
    }
    return ambient_out, fills, diagnostics

def _load_hdr_envmap(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read HDR envmap at {path}.")
    arr = cv2.cvtColor(np.asarray(image, dtype=np.float32)[..., :3], cv2.COLOR_BGR2RGB)
    return np.clip(arr, 0.0, None)


def _write_hdr_envmap(path: Path, hdr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(hdr, dtype=np.float32), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write HDR envmap to {path}.")


def _tonemap_preview(hdr: np.ndarray) -> np.ndarray:
    arr = np.asarray(hdr, dtype=np.float32)
    scale = max(float(np.percentile(arr, 99)), 1e-6)
    mapped = np.clip(arr / scale, 0.0, 1.0)
    gamma = np.power(mapped, 1.0 / 2.2)
    return np.clip(np.rint(gamma * 255.0), 0.0, 255.0).astype(np.uint8)


def _semantic_label_map(semantics: Any) -> dict[int, str]:
    raw = semantics.metadata.get("class_id_to_label", {})
    if isinstance(raw, Mapping) and raw:
        return {int(key): str(value).strip().lower() for key, value in raw.items()}
    mapping: dict[int, str] = {}
    for segment in semantics.segments:
        label = str(segment.label).strip().lower()
        if segment.label_id is not None:
            mapping[int(segment.label_id)] = label
        mapping[int(segment.segment_id)] = label
    return mapping


def _label_ids_for_tokens(label_map: Mapping[int, str], tokens: Sequence[str]) -> np.ndarray:
    wanted = {str(token).strip().lower() for token in tokens if str(token).strip()}
    return np.asarray(
        [label_id for label_id, label in label_map.items() if label in wanted],
        dtype=np.int32,
    )


def _extract_sun_candidates(envmap: np.ndarray) -> list[dict[str, Any]]:
    luminance = _luminance(envmap)
    height, width = luminance.shape
    directions = _latlong_directions(height, width)
    upper_mask = directions[..., 2] >= 0.05
    global_p95 = float(np.percentile(luminance, 95))
    global_p99 = float(np.percentile(luminance, 99))
    blurred = cv2.GaussianBlur(luminance, (0, 0), sigmaX=1.2, sigmaY=1.2)
    response = np.where(upper_mask, blurred, -np.inf)
    candidates: list[dict[str, Any]] = []
    suppression_radius = max(6, width // 24)
    for _ in range(_MAX_SUN_CANDIDATES):
        flat_index = int(np.argmax(response))
        y, x = np.unravel_index(flat_index, response.shape)
        if not np.isfinite(response[y, x]):
            break
        peak = float(luminance[y, x])
        if peak <= max(global_p95, 1e-6):
            break
        local = luminance[max(0, y - 6) : min(height, y + 7), max(0, x - 6) : min(width, x + 7)]
        local_mean = float(np.mean(local))
        local_p95 = float(np.percentile(local, 95))
        high_energy_fraction = float(np.mean(local >= max(peak * 0.6, 1e-6)))
        prominence = peak / max(local_p95, 1e-6)
        sharpness = max(peak - local_mean, 0.0)
        compactness = max(0.0, 1.0 - min(high_energy_fraction / 0.5, 1.0))
        strength_ratio = min(peak / max(global_p99, 1e-6), 1.5)
        confidence = float(
            np.clip(
                0.45 * min(prominence / 4.0, 1.0)
                + 0.30 * compactness
                + 0.25 * min(strength_ratio, 1.0),
                0.0,
                1.0,
            )
        )
        candidates.append(
            {
                "direction_camera": _direction_from_uv(y, x, height, width),
                "strength": peak,
                "color": _normalize_color(envmap[y, x]),
                "confidence": confidence,
                "peak_contrast": prominence,
                "peak_sharpness": sharpness,
                "high_energy_fraction": high_energy_fraction,
            }
        )
        yy, xx = np.ogrid[:height, :width]
        wrap_dx = np.minimum(np.abs(xx - x), width - np.abs(xx - x))
        mask = (yy - y) ** 2 + wrap_dx ** 2 <= suppression_radius ** 2
        response = np.where(mask, -np.inf, response)
    return sorted(candidates, key=lambda item: float(item["confidence"]), reverse=True)


def _best_camera_cluster(estimates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for estimate in estimates:
        frame_weight = float(estimate.get("estimate_quality", 0.0))
        for index, candidate in enumerate(estimate.get("sun_candidates_camera", [])):
            candidates.append(
                {
                    "frame_index": int(estimate["frame_index"]),
                    "candidate_index": index,
                    "direction": _normalize(np.asarray(candidate["direction_camera"], dtype=np.float32)),
                    "weight": max(frame_weight * float(candidate.get("confidence", 0.0)), 1e-6),
                    "strength": float(candidate.get("strength", 0.0)),
                    "color": np.asarray(candidate.get("color", (1.0, 1.0, 1.0)), dtype=np.float32),
                }
            )
    best: dict[str, Any] | None = None
    for seed in candidates:
        members: list[dict[str, Any]] = []
        for estimate in estimates:
            frame_candidates = [
                candidate for candidate in candidates if candidate["frame_index"] == int(estimate["frame_index"])
            ]
            best_match: dict[str, Any] | None = None
            best_angle = float("inf")
            for candidate in frame_candidates:
                angle = _angle_deg(candidate["direction"], seed["direction"])
                if angle <= _CAMERA_CLUSTER_SUPPORT_DEG and angle < best_angle:
                    best_match = candidate
                    best_angle = angle
            if best_match is not None:
                members.append({**best_match, "seed_angle_deg": best_angle})
        if len(members) < _MIN_CLUSTER_FRAMES:
            continue
        weights = np.asarray([float(member["weight"]) for member in members], dtype=np.float32)
        weights /= np.sum(weights)
        mean_direction = _normalize(
            np.sum(np.stack([member["direction"] for member in members], axis=0) * weights[:, None], axis=0)
        )
        residuals = np.asarray(
            [_angle_deg(member["direction"], mean_direction) for member in members],
            dtype=np.float32,
        )
        mean_spread = float(np.average(residuals, weights=weights))
        max_residual = float(np.max(residuals))
        score = float(np.sum(weights) + 0.05 * len(members) - 0.002 * mean_spread)
        cluster = {
            "passed": (
                len(members) >= _MIN_CLUSTER_FRAMES
                and mean_spread <= _CAMERA_CLUSTER_MEAN_DEG
                and max_residual <= _CAMERA_CLUSTER_SUPPORT_DEG
            ),
            "frame_count": len(members),
            "candidate_count": len(candidates),
            "mean_direction": mean_direction,
            "mean_spread_deg": mean_spread,
            "max_residual_deg": max_residual,
            "score": score,
            "members": [
                {**member, "residual_deg": float(residuals[idx])}
                for idx, member in enumerate(members)
            ],
            "failure_reason": (
                None
                if len(members) >= _MIN_CLUSTER_FRAMES
                and mean_spread <= _CAMERA_CLUSTER_MEAN_DEG
                and max_residual <= _CAMERA_CLUSTER_SUPPORT_DEG
                else (
                    "camera_cluster_too_small"
                    if len(members) < _MIN_CLUSTER_FRAMES
                    else "camera_sun_incoherent"
                )
            ),
        }
        if best is None or cluster["score"] > best["score"]:
            best = cluster
    return best or {
        "passed": False,
        "frame_count": 0,
        "candidate_count": len(candidates),
        "mean_direction": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "mean_spread_deg": 180.0,
        "max_residual_deg": 180.0,
        "score": 0.0,
        "members": [],
        "failure_reason": "no_camera_cluster",
    }


def _world_cluster_from_camera_cluster(
    camera_cluster: Mapping[str, Any],
    estimates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not bool(camera_cluster.get("members")):
        return {
            "passed": False,
            "members": [],
            "mean_direction": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "mean_color": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "mean_strength": 0.0,
            "mean_spread_deg": 180.0,
            "failure_reason": str(camera_cluster.get("failure_reason", "no_camera_cluster")),
        }
    estimate_by_frame = {int(item["frame_index"]): item for item in estimates}
    members: list[dict[str, Any]] = []
    for member in camera_cluster.get("members", []):
        estimate = estimate_by_frame.get(int(member["frame_index"]))
        if estimate is None:
            continue
        rotation = np.asarray(estimate["rotation_c2w"], dtype=np.float32)
        world_direction = _normalize(rotation @ np.asarray(member["direction"], dtype=np.float32))
        members.append(
            {
                **member,
                "direction_world": world_direction,
            }
        )
    if len(members) < _MIN_CLUSTER_FRAMES:
        return {
            "passed": False,
            "members": members,
            "mean_direction": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "mean_color": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "mean_strength": 0.0,
            "mean_spread_deg": 180.0,
            "failure_reason": "world_cluster_too_small",
        }
    weights = np.asarray([float(member["weight"]) for member in members], dtype=np.float32)
    weights /= np.sum(weights)
    mean_direction = _normalize(
        np.sum(np.stack([member["direction_world"] for member in members], axis=0) * weights[:, None], axis=0)
    )
    residuals = np.asarray(
        [_angle_deg(member["direction_world"], mean_direction) for member in members],
        dtype=np.float32,
    )
    mean_spread = float(np.average(residuals, weights=weights))
    max_residual = float(np.max(residuals))
    mean_color = _normalize_color(
        np.sum(np.stack([member["color"] for member in members], axis=0) * weights[:, None], axis=0)
    )
    mean_strength = float(np.sum(np.asarray([member["strength"] for member in members]) * weights))
    passed = mean_spread <= _WORLD_CLUSTER_MEAN_DEG and max_residual <= _WORLD_CLUSTER_MAX_DEG
    return {
        "passed": passed,
        "members": [
            {**member, "residual_deg": float(residuals[idx])}
            for idx, member in enumerate(members)
        ],
        "mean_direction": mean_direction,
        "mean_color": mean_color,
        "mean_strength": mean_strength,
        "mean_spread_deg": mean_spread,
        "max_residual_deg": max_residual,
        "failure_reason": None if passed else "world_sun_incoherent",
    }


def _best_effort_world_direction(
    estimates: Sequence[Mapping[str, Any]],
    world_cluster: Mapping[str, Any],
) -> np.ndarray:
    if world_cluster.get("members"):
        return _normalize(np.asarray(world_cluster["mean_direction"], dtype=np.float32))
    if estimates:
        return _normalize(np.asarray(estimates[0]["best_candidate_world"], dtype=np.float32))
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _best_effort_sun_color(
    estimates: Sequence[Mapping[str, Any]],
    world_cluster: Mapping[str, Any],
) -> np.ndarray:
    if world_cluster.get("members"):
        return _normalize_color(np.asarray(world_cluster["mean_color"], dtype=np.float32))
    if estimates:
        return _normalize_color(np.asarray(estimates[0]["sun_color"], dtype=np.float32))
    return np.asarray([1.0, 1.0, 1.0], dtype=np.float32)


def _latlong_directions(height: int, width: int) -> np.ndarray:
    u = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False, dtype=np.float32)
    v = np.linspace(0.0, np.pi, height, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    return np.stack(
        [np.sin(vv) * np.cos(uu), np.sin(vv) * np.sin(uu), np.cos(vv)],
        axis=-1,
    )


def _direction_from_uv(y: int, x: int, height: int, width: int) -> np.ndarray:
    phi = (float(x) + 0.5) / float(width) * 2.0 * np.pi
    theta = (float(y) + 0.5) / float(height) * np.pi
    return _normalize(
        np.asarray(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ],
            dtype=np.float32,
        )
    )


def _direction_to_uv(direction: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    vec = _normalize(np.asarray(direction, dtype=np.float32))
    phi = np.mod(np.arctan2(vec[..., 1], vec[..., 0]), 2.0 * np.pi)
    theta = np.arccos(np.clip(vec[..., 2], -1.0, 1.0))
    x = phi / (2.0 * np.pi) * width - 0.5
    y = theta / np.pi * height - 0.5
    return y, x


def _rotate_envmap(envmap: np.ndarray, rotation_c2w: np.ndarray) -> np.ndarray:
    height, width = envmap.shape[:2]
    world_dirs = _latlong_directions(height, width)
    camera_dirs = world_dirs @ np.asarray(rotation_c2w, dtype=np.float32)
    sample_y, sample_x = _direction_to_uv(camera_dirs, height, width)
    return _bilinear_sample(envmap, sample_y, sample_x)


def _suppress_sun(envmap: np.ndarray, sun_direction_world: np.ndarray) -> np.ndarray:
    dirs = _latlong_directions(envmap.shape[0], envmap.shape[1])
    dot = np.clip(
        np.sum(dirs * _normalize(sun_direction_world)[None, None, :], axis=-1),
        -1.0,
        1.0,
    )
    mask = np.arccos(dot) <= math.radians(14.0)
    return np.where(mask[..., None], 0.0, envmap)


def _sun_blob(
    directions: np.ndarray,
    sun_direction_world: np.ndarray,
    color: np.ndarray,
    strength: float,
    sigma_deg: float,
) -> np.ndarray:
    sigma_rad = math.radians(sigma_deg)
    dot = np.clip(
        np.sum(directions * _normalize(sun_direction_world)[None, None, :], axis=-1),
        -1.0,
        1.0,
    )
    angles = np.arccos(dot)
    blob = np.exp(-0.5 * (angles / max(sigma_rad, 1e-6)) ** 2)[..., None]
    return np.clip(
        strength * blob * np.asarray(color, dtype=np.float32)[None, None, :],
        0.0,
        None,
    )


def _extract_fill_lights(
    envmap: np.ndarray,
    *,
    max_lights: int,
    min_separation_deg: float,
    min_strength: float,
) -> list[dict[str, Any]]:
    if max_lights <= 0:
        return []
    luminance = _luminance(np.asarray(envmap, dtype=np.float32))
    smoothed = luminance.copy()
    for _ in range(3):
        smoothed = _box_blur(smoothed)
    baseline = float(np.percentile(smoothed, 50))
    contrast = float(np.percentile(smoothed, 95) - baseline)
    if contrast < max(min_strength * 0.25, 1e-3):
        return []
    working = smoothed.copy()
    height, width = working.shape
    selected: list[dict[str, Any]] = []
    directions: list[np.ndarray] = []
    for _ in range(max_lights):
        flat_index = int(np.argmax(working))
        peak = float(working.reshape(-1)[flat_index])
        if peak < min_strength or (peak - baseline) < max(contrast * 0.2, 5e-4):
            break
        y, x = np.unravel_index(flat_index, working.shape)
        direction = _direction_from_uv(int(y), int(x), height, width)
        if any(
            math.degrees(math.acos(float(np.clip(np.dot(direction, other), -1.0, 1.0))))
            < min_separation_deg
            for other in directions
        ):
            working[y, x] = 0.0
            continue
        color = _normalize_color(np.asarray(envmap[y, x], dtype=np.float32))
        strength = float(np.percentile(envmap[y, x], 90))
        selected.append(
            {
                "direction_world": direction,
                "color": color,
                "strength": max(strength, min_strength),
                "confidence": float(np.clip(peak / max(float(np.max(smoothed)), 1e-6), 0.0, 1.0)),
            }
        )
        directions.append(direction)
        dot = np.clip(np.sum(_latlong_directions(height, width) * direction[None, None, :], axis=-1), -1.0, 1.0)
        working[np.arccos(dot) <= math.radians(min_separation_deg)] = 0.0
    return selected


def _build_light_rig(
    *,
    rig_mode: str,
    sun_direction_world: np.ndarray,
    sun_strength: float,
    sun_color: np.ndarray,
    fill_lights: Sequence[Mapping[str, Any]],
) -> list[LightingLightData]:
    lights: list[LightingLightData] = []
    if rig_mode in {"analytic_rig", "sun_plus_fill"} and sun_strength > 0.0:
        lights.append(
            LightingLightData(
                name="PEMOINDirectSun",
                kind="SUN",
                role="direct_key",
                strength=float(sun_strength),
                color=np.asarray(sun_color, dtype=np.float32).reshape(3),
                casts_shadow=True,
                direction_world=_normalize(np.asarray(sun_direction_world, dtype=np.float32)),
                angular_size_deg=2.0,
                confidence=1.0,
                diagnostics={"source": "world_cluster"},
            )
        )
    for idx, fill in enumerate(fill_lights):
        direction = _normalize(np.asarray(fill["direction_world"], dtype=np.float32))
        light_kind = str(fill.get("kind", "AREA")).upper()
        lights.append(
            LightingLightData(
                name=f"PEMOINFill{idx + 1}",
                kind=light_kind,
                role=str(fill.get("role", "diffuse_fill")),
                strength=float(fill["strength"]),
                color=np.asarray(fill["color"], dtype=np.float32).reshape(3),
                casts_shadow=False,
                placement_mode=str(fill.get("placement_mode", "world_absolute")),
                placement_target=str(fill.get("placement_target", "subject_root_dynamic")),
                direction_world=direction,
                location_world=np.asarray(
                    fill.get("location_world", -direction * _FILL_LIGHT_DISTANCE_M),
                    dtype=np.float32,
                ).reshape(3),
                area_size=(
                    np.asarray(
                        fill.get("area_size", _FILL_LIGHT_AREA_SIZE_M),
                        dtype=np.float32,
                    ).reshape(2)
                    if light_kind == "AREA"
                    else None
                ),
                confidence=float(fill.get("confidence", 1.0)),
                diagnostics=dict(fill.get("diagnostics", {"source": "ambient_envmap"})),
            )
        )
    if rig_mode == "envmap_only":
        return []
    return lights


def _rig_mode_rank(mode: str) -> int:
    if mode == "analytic_rig":
        return 2
    if mode == "sun_plus_fill":
        return 1
    return 0


def _bilinear_sample(image: np.ndarray, sample_y: np.ndarray, sample_x: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.mod(x0, width)
    y0 = np.clip(y0, 0, height - 1)
    wx = sample_x - np.floor(sample_x)
    wy = sample_y - np.floor(sample_y)
    top = (1.0 - wx)[..., None] * image[y0, x0] + wx[..., None] * image[y0, x1]
    bottom = (1.0 - wx)[..., None] * image[y1, x0] + wx[..., None] * image[y1, x1]
    return (1.0 - wy)[..., None] * top + wy[..., None] * bottom


def _luminance(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.212671 + image[..., 1] * 0.715160 + image[..., 2] * 0.072169


def _box_blur(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="wrap")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return arr / norm


def _normalize_color(color: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(color, dtype=np.float32), 0.0, None)
    peak = float(np.max(arr))
    if peak <= 1e-8:
        return np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    return arr / peak


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(aa, bb), -1.0, 1.0))))
