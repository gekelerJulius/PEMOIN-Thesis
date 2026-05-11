from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pemoin.data.contracts import (
    DepthData,
    FrameData,
    LightingData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    SemanticsData,
)
from pemoin.providers.lighting import (
    CarlaGTLightingProvider,
    DiffusionLightTurboLightingProvider,
    DiffusionLightTurboSettings,
    UnityGTLightingProvider,
    _prepare_model_image,
)
from pemoin.runtime.cache import CrossRunCacheManager


class _DummyResources:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.saved: LightingData | None = None

    def provider_dir(self, name: str) -> Path:
        path = self._root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_lighting(self, lighting: LightingData) -> None:
        self.saved = lighting

    def has(self, kind) -> bool:
        return kind == ResourceKind.FRAMES

    def frame_indices(self, kind) -> list[int]:
        return [0] if kind == ResourceKind.FRAMES else []


def _build_lighting_cache_store(root: Path, name: str) -> ResourceStore:
    store = ResourceStore(name, root=root)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    store.save_frame(FrameData(frame_id="000000", index=0, image=frame))
    store.save_depth(
        DepthData(
            frame_index=0,
            depth=np.ones((2, 2), dtype=np.float32),
            metadata={"source": "unit-test"},
        )
    )
    c2w = np.eye(4, dtype=np.float32)
    store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=0,
                    camera_to_world=c2w,
                    world_to_camera=np.linalg.inv(c2w),
                    metadata={"source": "unit-test"},
                )
            ],
            metadata={"source": "unit-test"},
        )
    )
    store.save_semantics2d(
        SemanticsData(
            frame_index=0,
            segments=[],
            segment_ids=np.zeros((2, 2), dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )
    return store


def _build_lighting_cache_store_with_transform_ids(
    root: Path,
    name: str,
    *,
    alignment_transform_id: str,
    grounding_transform_id: str,
) -> ResourceStore:
    store = ResourceStore(name, root=root)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    store.save_frame(FrameData(frame_id="000000", index=0, image=frame))
    store.save_depth(
        DepthData(
            frame_index=0,
            depth=np.ones((2, 2), dtype=np.float32),
            metadata={
                "source": "unit-test",
                "alignment_transform_id": alignment_transform_id,
            },
        )
    )
    c2w = np.eye(4, dtype=np.float32)
    store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=0,
                    camera_to_world=c2w,
                    world_to_camera=np.linalg.inv(c2w),
                    confidence=float("nan"),
                    metadata={
                        "source": "unit-test",
                        "alignment_transform_id": alignment_transform_id,
                        "grounding_transform_id": grounding_transform_id,
                        "scale_source": "stable-geometry",
                    },
                )
            ],
            metadata={
                "source": "unit-test",
                "alignment_transform_id": alignment_transform_id,
                "grounding_transform_id": grounding_transform_id,
                "scale_source": "stable-geometry",
            },
        )
    )
    store.save_semantics2d(
        SemanticsData(
            frame_index=0,
            segments=[],
            segment_ids=np.zeros((2, 2), dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )
    return store


def test_lighting_provider_selects_temporally_spread_keyframes() -> None:
    provider = DiffusionLightTurboLightingProvider({})
    scored = [
        {"frame_index": idx, "score": float(score)}
        for idx, score in enumerate([0.1, 0.9, 0.4, 0.3, 0.8, 0.2, 0.7, 0.6, 0.5, 0.95])
    ]
    selected = provider._select_keyframes(scored, count=5)
    assert [item["frame_index"] for item in selected] == [1, 2, 4, 6, 9]


class _FakeCarlaDataset:
    def __init__(self, *, weather: dict, scene_lights: list[dict] | None = None) -> None:
        self._weather = dict(weather)
        self._scene_lights = list(scene_lights or [])

    def has_lighting_gt(self) -> bool:
        return True

    def run_lighting(self) -> dict:
        return {
            "town": "Town01",
            "weather": dict(self._weather),
        }

    def scene_lights(self) -> dict:
        return {"lights": list(self._scene_lights)}

    def frame_indices(self) -> list[int]:
        return [0]

    def frame_lighting(self, frame_index: int) -> dict:
        return {"frame": int(frame_index), "weather": dict(self._weather)}


def test_carla_gt_lighting_provider_emits_explicit_sun_for_daylight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = CarlaGTLightingProvider({})
    resources = _DummyResources(tmp_path)
    dataset = _FakeCarlaDataset(
        weather={
            "sun_azimuth_angle": 45.0,
            "sun_altitude_angle": 35.0,
            "cloudiness": 10.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "scattering_intensity": 1.0,
            "mie_scattering_scale": 0.03,
            "rayleigh_scattering_scale": 0.0331,
        },
        scene_lights=[
            {
                "id": 1,
                "is_on": True,
                "intensity": 100.0,
                "light_group": "Street",
                "location": {"x": 1.0, "y": 2.0, "z": 3.0},
                "color": {"r": 255, "g": 240, "b": 220},
            }
        ],
    )
    monkeypatch.setattr("pemoin.providers.lighting.resolve_carla_dataset", lambda *_args, **_kwargs: dataset)

    provider.run(resources, context={"frame_source": str(tmp_path)})

    assert resources.saved is not None
    assert resources.saved.rig_mode == "sun_plus_fill"
    assert resources.saved.light_rig[0].kind == "SUN"
    assert resources.saved.light_rig[0].role == "direct_key"
    assert resources.saved.light_rig[0].casts_shadow is True
    assert resources.saved.decomposition["envmap_mode"] == "sky_ambient_only"
    assert resources.saved.sun_diagnostics["analytic_sun_emitted"] is True
    assert resources.saved.light_rig[1].strength < 0.2


def test_carla_gt_lighting_provider_drops_sun_below_horizon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = CarlaGTLightingProvider({})
    resources = _DummyResources(tmp_path)
    dataset = _FakeCarlaDataset(
        weather={
            "sun_azimuth_angle": 45.0,
            "sun_altitude_angle": -5.0,
            "cloudiness": 60.0,
            "fog_density": 5.0,
            "wetness": 0.0,
            "scattering_intensity": 1.0,
            "mie_scattering_scale": 0.03,
            "rayleigh_scattering_scale": 0.0331,
        },
        scene_lights=[],
    )
    monkeypatch.setattr("pemoin.providers.lighting.resolve_carla_dataset", lambda *_args, **_kwargs: dataset)

    provider.run(resources, context={"frame_source": str(tmp_path)})

    assert resources.saved is not None
    assert all(light.kind != "SUN" for light in resources.saved.light_rig)
    assert resources.saved.rig_mode == "envmap_only"
    assert resources.saved.sun_strength == 0.0
    assert resources.saved.sun_diagnostics["analytic_sun_emitted"] is False


def _write_test_exr(path: Path, value: float) -> None:
    import cv2

    image = np.full((8, 8, 3), value, dtype=np.float32)
    cv2.imwrite(str(path), image[:, :, ::-1])


def _write_unity_lighting_gt_export(
    root: Path,
    *,
    sun_enabled: bool = True,
    include_frame_lighting: bool = True,
) -> None:
    lighting_root = root / "lighting_gt"
    face_root = lighting_root / "reflection_probe_faces"
    face_root.mkdir(parents=True, exist_ok=True)
    for idx, face in enumerate(
        ("PositiveX", "NegativeX", "PositiveY", "NegativeY", "PositiveZ", "NegativeZ")
    ):
        _write_test_exr(face_root / f"fallback_capture_{face}.exr", 0.1 + 0.05 * idx)
    (lighting_root / "run_lighting.json").write_text(
        """
        {
          "pipeline": "HDRenderPipelineAsset",
          "unityVersion": "6000.3.5f1",
          "sceneName": "City_Main",
          "reflectionSource": "ReflectionProbe",
          "reflectionProbeName": "Reflection Probe"
        }
        """.strip(),
        encoding="utf-8",
    )
    (lighting_root / "scene_lights.json").write_text(
        (
            """
        {
          "mainDirectionalLight": {
            "name": "Directional Light",
            "enabled": %s,
            "castsShadows": false,
            "intensity": 110000.0,
            "indirectMultiplier": 1.0,
            "colorTemperature": 6155.0,
            "colorLinear": [1.0, 0.98, 0.95],
            "directionWorld": [-0.54, 0.07, -0.83]
          },
          "ambientProbe": {
            "coefficientsRGB27": [
              743.37, 219.58, -362.30, -244.91, -62.03, -98.50, 113.55, 417.14, 155.40,
              648.27, 306.78, -149.48, -108.24, -30.45, -46.70, 71.83, 222.33, 119.42,
              598.07, 399.06, -38.73, -34.03, -13.75, -19.62, 34.63, 93.38, 64.40
            ]
          }
        }
        """
            % ("true" if sun_enabled else "false")
        ).strip(),
        encoding="utf-8",
    )
    if include_frame_lighting:
        (lighting_root / "frame_lighting.jsonl").write_text(
            """
            {"frameIndex":0,"timestampSec":0.0,"reflectionProbeName":"Reflection Probe","probeSamples":[{"label":"subject_anchor","sh":{"coefficientsRGB27":[743.37,219.58,-362.30,-244.91,-62.03,-98.50,113.55,417.14,155.40,648.27,306.78,-149.48,-108.24,-30.45,-46.70,71.83,222.33,119.42,598.07,399.06,-38.73,-34.03,-13.75,-19.62,34.63,93.38,64.40]}}]}
            """.strip()
            + "\n",
            encoding="utf-8",
        )


def test_unity_gt_lighting_provider_emits_sun_plus_envmap(tmp_path: Path) -> None:
    export_root = tmp_path / "unity_export"
    _write_unity_lighting_gt_export(export_root)
    provider = UnityGTLightingProvider({"path": str(export_root)})
    resources = _DummyResources(tmp_path)

    provider.run(resources, context={"frame_source": str(export_root)})

    assert resources.saved is not None
    assert resources.saved.rig_mode == "sun_plus_fill"
    assert len(resources.saved.light_rig) == 1
    assert resources.saved.light_rig[0].kind == "SUN"
    assert resources.saved.light_rig[0].casts_shadow is True
    assert resources.saved.decomposition["envmap_mode"] == "reflection_probe_latlong"
    assert resources.saved.decomposition["diffuse_mode"] == "ambient_probe_scalar_only"
    assert resources.saved.ambient_strength > 0.01
    assert Path(resources.saved.envmap_path).exists()


def test_unity_gt_lighting_provider_degrades_to_envmap_only_without_sun(tmp_path: Path) -> None:
    export_root = tmp_path / "unity_export"
    _write_unity_lighting_gt_export(export_root, sun_enabled=False)
    provider = UnityGTLightingProvider({"path": str(export_root)})
    resources = _DummyResources(tmp_path)

    provider.run(resources, context={"frame_source": str(export_root)})

    assert resources.saved is not None
    assert resources.saved.rig_mode == "envmap_only"
    assert resources.saved.sun_strength == 0.0
    assert resources.saved.light_rig == []


def test_unity_gt_lighting_provider_requires_complete_reflection_faces(tmp_path: Path) -> None:
    export_root = tmp_path / "unity_export"
    _write_unity_lighting_gt_export(export_root)
    (export_root / "lighting_gt" / "reflection_probe_faces" / "fallback_capture_NegativeZ.exr").unlink()
    provider = UnityGTLightingProvider({"path": str(export_root)})
    resources = _DummyResources(tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing faces: NegativeZ"):
        provider.run(resources, context={"frame_source": str(export_root)})


def test_prepare_model_image_avoids_black_letterbox_padding() -> None:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[..., 0] = 32
    image[..., 1] = 64
    image[..., 2] = 128
    prepared = _prepare_model_image(image, 128)
    assert prepared.shape == (128, 128, 3)
    assert not np.any(np.all(prepared == 0, axis=-1))


def test_lighting_provider_uses_smaller_recovery_input_size_by_default() -> None:
    provider = DiffusionLightTurboLightingProvider({"input_size": 1024})
    assert provider.settings.input_size == 1024
    assert provider.settings.recovery_input_size == 768
    assert provider.settings.allow_online_model_fetch is False
    assert provider.settings.sdxl_model == "stabilityai/stable-diffusion-xl-base-1.0"


def test_lighting_provider_offline_preflight_reports_missing_models(tmp_path: Path) -> None:
    provider = DiffusionLightTurboLightingProvider(
        {
            "hf_home": str(tmp_path / "hf"),
            "sdxl_model": "missing/base",
            "sdxl_vae_model": "missing/vae",
        }
    )
    env = provider._build_tool_env()
    with pytest.raises(RuntimeError, match="DiffusionLight-Turbo models are not available offline"):
        provider._validate_model_sources(env, no_controlnet=True)


def test_lighting_provider_builds_offline_inpaint_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = DiffusionLightTurboLightingProvider(
        {
            "repo_root": str(tmp_path / "repo"),
            "conda_env": "diffusionlight-turbo",
            "sdxl_model": str(tmp_path / "models" / "sdxl"),
            "sdxl_vae_model": str(tmp_path / "models" / "vae"),
        }
    )
    (provider.settings.repo_root).mkdir(parents=True)
    for model_dir in (
        Path(provider.settings.sdxl_model),
        Path(provider.settings.sdxl_vae_model),
    ):
        model_dir.mkdir(parents=True)

    class _Resources:
        def load_frame(self, frame_index: int):
            class _Frame:
                image = np.zeros((32, 64, 3), dtype=np.uint8)

            return _Frame()

    captured: list[tuple[list[str], dict[str, str], Path]] = []

    def _fake_run(cmd, check, cwd, env):
        captured.append((list(cmd), dict(env), cwd))

    monkeypatch.setattr("pemoin.providers.lighting.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "pemoin.providers.lighting.resolve_env_launcher",
        lambda env_name, env_manager: ("mamba", "run", "-n", env_name),
    )
    monkeypatch.setattr(
        "pemoin.providers.lighting._load_hdr_envmap",
        lambda path: np.ones((4, 8, 3), dtype=np.float32),
    )

    results = provider._run_diffusionlight(
        _Resources(),
        [{"frame_index": 3, "score": 0.7}],
        tmp_path / "provider",
        input_size=256,
        no_controlnet=True,
    )

    assert 3 in results
    assert len(captured) == 3
    inpaint_cmd, env, cwd = captured[0]
    assert cwd == provider.settings.repo_root
    assert "--sdxl-model" in inpaint_cmd
    assert "--sdxl-vae-model" in inpaint_cmd
    assert "--allow-online-model-fetch" in inpaint_cmd
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["TRANSFORMERS_CACHE"] == env["HF_HUB_CACHE"]


def test_lighting_provider_selects_recovery_keyframes_by_consensus_first() -> None:
    provider = DiffusionLightTurboLightingProvider({})
    estimates = [
        {
            "frame_index": 4,
            "frame_score": 0.9,
            "estimate_quality": 0.7,
            "consensus_score": 0.7,
            "sun_candidates_camera": [
                {"direction_camera": np.array([0.0, 0.0, 1.0], dtype=np.float32), "confidence": 0.7}
            ],
        },
        {
            "frame_index": 9,
            "frame_score": 0.95,
            "estimate_quality": 0.95,
            "consensus_score": 0.95,
            "sun_candidates_camera": [
                {"direction_camera": np.array([1.0, 0.0, 0.0], dtype=np.float32), "confidence": 0.95}
            ],
        },
        {
            "frame_index": 16,
            "frame_score": 0.85,
            "estimate_quality": 0.75,
            "consensus_score": 0.75,
            "sun_candidates_camera": [
                {"direction_camera": np.array([0.0, 0.05, 0.998], dtype=np.float32), "confidence": 0.75}
            ],
        },
    ]
    selected = provider._select_recovery_keyframes(estimates, count=2)
    assert [item["frame_index"] for item in selected] == [4, 16]


def test_lighting_provider_fuses_estimates_into_full_sun_when_world_consensus_holds() -> None:
    provider = DiffusionLightTurboLightingProvider({})
    envmap = np.full((8, 16, 3), 0.4, dtype=np.float32)
    estimates = [
        {
            "frame_index": 1,
            "frame_score": 0.9,
            "rotation_c2w": np.eye(3, dtype=np.float32),
            "sun_candidates_camera": [
                {
                    "direction_camera": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    "strength": 3.0,
                    "color": np.array([1.0, 0.9, 0.8], dtype=np.float32),
                    "confidence": 0.8,
                }
            ],
            "best_candidate_world": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "sun_strength": 3.0,
            "sun_color": np.array([1.0, 0.9, 0.8], dtype=np.float32),
            "ambient_hdr": envmap,
            "weight": 0.8,
            "estimate_quality": 0.8,
            "hdr_p95": 0.4,
            "diagnostics": {"estimate_quality": 0.8},
        },
        {
            "frame_index": 5,
            "frame_score": 0.8,
            "rotation_c2w": np.eye(3, dtype=np.float32),
            "sun_candidates_camera": [
                {
                    "direction_camera": np.array([0.0, 0.08, 0.997], dtype=np.float32),
                    "strength": 2.8,
                    "color": np.array([0.95, 0.9, 0.85], dtype=np.float32),
                    "confidence": 0.7,
                }
            ],
            "best_candidate_world": np.array([0.0, 0.08, 0.997], dtype=np.float32),
            "sun_strength": 2.8,
            "sun_color": np.array([0.95, 0.9, 0.85], dtype=np.float32),
            "ambient_hdr": envmap * 1.1,
            "weight": 0.7,
            "estimate_quality": 0.7,
            "hdr_p95": 0.44,
            "diagnostics": {"estimate_quality": 0.7},
        },
    ]
    fused = provider._fuse_estimates(estimates)
    assert fused["mode"] == "full_sun"
    assert fused["rig_mode"] == "sun_plus_fill"
    assert fused["quality"]["sun"] > 0.0
    assert fused["envmap_hdr"].shape == (8, 16, 3)
    assert fused["sun_diagnostics"]["degraded_reason"] is None
    assert 2.0 <= fused["sun_strength"] <= 4.0
    assert len(fused["light_rig"]) == 1
    assert fused["light_rig"][0].kind == "SUN"


def test_lighting_provider_drops_to_ambient_only_when_world_consensus_breaks() -> None:
    provider = DiffusionLightTurboLightingProvider({})
    envmap = np.full((8, 16, 3), 0.4, dtype=np.float32)
    rot_y_90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    estimates = [
        {
            "frame_index": 1,
            "frame_score": 0.9,
            "rotation_c2w": np.eye(3, dtype=np.float32),
            "sun_candidates_camera": [
                {
                    "direction_camera": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    "strength": 3.0,
                    "color": np.array([1.0, 0.9, 0.8], dtype=np.float32),
                    "confidence": 0.8,
                }
            ],
            "best_candidate_world": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "sun_strength": 3.0,
            "sun_color": np.array([1.0, 0.9, 0.8], dtype=np.float32),
            "ambient_hdr": envmap,
            "weight": 0.8,
            "estimate_quality": 0.8,
            "hdr_p95": 0.4,
            "diagnostics": {"estimate_quality": 0.8},
        },
        {
            "frame_index": 5,
            "frame_score": 0.8,
            "rotation_c2w": rot_y_90,
            "sun_candidates_camera": [
                {
                    "direction_camera": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    "strength": 2.8,
                    "color": np.array([0.95, 0.9, 0.85], dtype=np.float32),
                    "confidence": 0.7,
                }
            ],
            "best_candidate_world": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "sun_strength": 2.8,
            "sun_color": np.array([0.95, 0.9, 0.85], dtype=np.float32),
            "ambient_hdr": envmap * 1.1,
            "weight": 0.7,
            "estimate_quality": 0.7,
            "hdr_p95": 0.44,
            "diagnostics": {"estimate_quality": 0.7},
        },
    ]
    fused = provider._fuse_estimates(estimates)
    assert fused["mode"] == "ambient_only"
    assert fused["rig_mode"] == "envmap_only"
    assert fused["sun_strength"] == 0.0
    assert fused["sun_diagnostics"]["degraded_reason"] == "world_sun_incoherent"


def test_lighting_provider_demotes_coherent_sun_for_diffuse_scene_cues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DiffusionLightTurboLightingProvider(
        {
            "fill_heavy_dark_side_target_ratio": 0.18,
            "fill_heavy_transport_gain": 1.0,
        }
    )
    envmap = np.full((8, 16, 3), 0.08, dtype=np.float32)
    view_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    estimates = [
        {
            "frame_index": 16,
            "frame_score": 0.9,
            "rotation_c2w": view_rotation,
            "sun_candidates_camera": [
                {
                    "direction_camera": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    "strength": 16.0,
                    "color": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    "confidence": 0.7,
                }
            ],
            "best_candidate_world": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "sun_strength": 16.0,
            "sun_color": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "ambient_hdr": envmap,
            "weight": 0.55,
            "estimate_quality": 0.55,
            "hdr_p95": 0.08,
            "frame_metrics": {
                "sky_fraction": 0.22,
                "road_fraction": 0.38,
                "dynamic_fraction": 0.04,
                "large_vehicle_fraction": 0.0,
                "large_vehicle_near_fraction": 0.0,
                "overexposure_fraction": 0.0,
                "saturation_fraction": 0.0,
                "sharpness_score": 0.21,
            },
            "diagnostics": {"estimate_quality": 0.55, "candidate_confidence": 0.7},
        },
        {
            "frame_index": 25,
            "frame_score": 0.85,
            "rotation_c2w": view_rotation,
            "sun_candidates_camera": [
                {
                    "direction_camera": np.array([0.0, 0.05, 0.998], dtype=np.float32),
                    "strength": 15.5,
                    "color": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    "confidence": 0.68,
                }
            ],
            "best_candidate_world": np.array([0.0, 0.05, 0.998], dtype=np.float32),
            "sun_strength": 15.5,
            "sun_color": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "ambient_hdr": envmap,
            "weight": 0.45,
            "estimate_quality": 0.45,
            "hdr_p95": 0.08,
            "frame_metrics": {
                "sky_fraction": 0.20,
                "road_fraction": 0.37,
                "dynamic_fraction": 0.05,
                "large_vehicle_fraction": 0.0,
                "large_vehicle_near_fraction": 0.0,
                "overexposure_fraction": 0.0,
                "saturation_fraction": 0.0,
                "sharpness_score": 0.24,
            },
            "diagnostics": {"estimate_quality": 0.45, "candidate_confidence": 0.68},
        },
    ]

    monkeypatch.setattr(
        "pemoin.providers.lighting._extract_fill_lights",
        lambda *args, **kwargs: [
                {
                    "direction_world": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                    "color": np.array([0.95, 0.98, 1.0], dtype=np.float32),
                    "strength": 1.8,
                    "confidence": 0.9,
                },
                {
                    "direction_world": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "color": np.array([0.98, 1.0, 0.96], dtype=np.float32),
                    "strength": 1.6,
                    "confidence": 0.8,
                },
        ],
    )
    monkeypatch.setattr(
        "pemoin.providers.lighting._sun_blob",
        lambda directions, sun_direction_world, color, strength, sigma_deg: np.zeros(
            directions.shape, dtype=np.float32
        ),
    )

    fused = provider._fuse_estimates(estimates)
    planner = fused["decomposition"]["planner"]
    assert fused["mode"] == "full_sun"
    assert fused["rig_mode"] == "analytic_rig"
    assert planner["mode"] == "fill_heavy"
    assert planner["demoted_direct_sun"] is True
    assert planner["diffuse_scene_score"] > planner["direct_scene_score"]
    assert planner["direct_to_fill_ratio"] <= provider.settings.max_direct_to_fill_ratio_for_diffuse + 1e-6
    assert planner["brightness_preservation_ratio"] >= 0.85 - 1e-6
    assert planner["fill_transport_mode"] == "wrap_subject_fill"
    assert planner["subject_total_irradiance_after"] > planner["subject_total_irradiance_before"]
    assert planner["subject_dark_side_irradiance_after"] > planner["subject_dark_side_irradiance_before"]
    assert planner["view_facing_irradiance_after"] > planner["view_facing_irradiance_before"]
    assert planner["world_strength_after"] >= planner["world_strength_before"]
    assert planner["fill_heavy_transport_gain"] == pytest.approx(
        provider.settings.fill_heavy_transport_gain
    )
    assert planner["fill_heavy_dark_side_target_ratio"] == pytest.approx(
        provider.settings.fill_heavy_dark_side_target_ratio
    )
    assert planner["dark_to_bright_ratio_after"] > 0.0
    assert planner["fill_view_alignment_score_after"] >= planner["fill_view_alignment_score_before"]
    direct_light = fused["light_rig"][0]
    fill_lights = fused["light_rig"][1:]
    fill_energy = sum(light.strength for light in fill_lights)
    assert direct_light.kind == "SUN"
    assert {light.role for light in fill_lights} == {"wrap_key_fill", "counter_wrap_fill", "sky_fill"}
    assert all(light.kind == "POINT" for light in fill_lights)
    assert fill_energy > 0.0
    assert direct_light.strength / fill_energy <= provider.settings.max_direct_to_fill_ratio_for_diffuse + 1e-6
    assert planner["analytic_energy_after"] >= planner["analytic_energy_before"] * 0.85 - 1e-6
    assert all(light.placement_mode == "subject_anchor_relative" for light in fill_lights)
    assert all(light.placement_target == "subject_root_dynamic" for light in fill_lights)
    assert all(float(light.location_world[2]) > 0.0 for light in fill_lights)
    view_horizontal = np.asarray(planner["view_direction_world"][:2], dtype=np.float32)
    view_horizontal /= max(float(np.linalg.norm(view_horizontal)), 1e-6)
    assert (
        max(
            float(
                np.dot(
                    -np.asarray(light.direction_world[:2], dtype=np.float32),
                    view_horizontal,
                )
            )
            for light in fill_lights
        )
        > 0.4
    )
    wrap_alignment = {
        light.role: float(
            np.dot(
                -np.asarray(light.direction_world[:2], dtype=np.float32),
                view_horizontal,
            )
        )
        for light in fill_lights
        if light.role != "sky_fill"
    }
    assert wrap_alignment["counter_wrap_fill"] < wrap_alignment["wrap_key_fill"]
    key_horizontal = -np.asarray(
        next(light.direction_world for light in fill_lights if light.role == "wrap_key_fill")[:2],
        dtype=np.float32,
    )
    key_horizontal /= max(float(np.linalg.norm(key_horizontal)), 1e-6)
    counter_horizontal = -np.asarray(
        next(light.direction_world for light in fill_lights if light.role == "counter_wrap_fill")[:2],
        dtype=np.float32,
    )
    counter_horizontal /= max(float(np.linalg.norm(counter_horizontal)), 1e-6)
    separation_deg = float(
        np.degrees(
            np.arccos(float(np.clip(np.dot(key_horizontal, counter_horizontal), -1.0, 1.0)))
        )
    )
    assert separation_deg >= provider.settings.wrap_geometry_counter_opposition_deg - 1e-6
    assert planner["geometry_candidate_count"] > 0
    assert planner["geometry_winning_candidate_index"] is not None
    assert planner["geometry_best_dark_to_bright_ratio"] >= planner["dark_to_bright_ratio_after_geometry"]
    assert set(planner["role_geometry"].keys()) == {
        "wrap_key_fill",
        "counter_wrap_fill",
        "sky_fill",
    }


def test_lighting_provider_validation_rejects_black_envmap() -> None:
    provider = DiffusionLightTurboLightingProvider({})
    fused = {
        "mode": "ambient_only",
        "rig_mode": "envmap_only",
        "envmap_hdr": np.zeros((8, 16, 3), dtype=np.float32),
        "sun_direction_world": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "sun_strength": 0.0,
        "ambient_strength": 0.5,
        "light_rig": [],
        "quality": {"total": 0.8, "sun": 0.8, "envmap": 0.8},
        "sun_diagnostics": {
            "camera_mean_spread_deg": 180.0,
            "world_mean_spread_deg": 180.0,
            "camera_cluster_count": 0,
            "degraded_reason": "world_sun_incoherent",
        },
        "per_keyframe_diagnostics": [{"agreement_deg": None}],
    }
    estimates = [{"hdr_p95": 0.5}]
    validation = provider._validate_fused_result(fused, estimates)
    assert not validation["passed"]
    assert "mean_luminance_too_low" in validation["failures"]


def test_lighting_provider_run_fails_before_publish_when_recovery_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = DiffusionLightTurboLightingProvider({})
    resources = _DummyResources(tmp_path)

    attempt_results = [
        {
            "mode": "ambient_only",
            "rig_mode": "envmap_only",
            "light_rig": [],
            "decomposition": {"method": "test", "analytic_light_count": 0},
            "sun_diagnostics": {"degraded_reason": "world_sun_incoherent"},
            "validation": {"passed": False, "failures": ["primary_failed"]},
            "per_keyframe_results": [
                {"frame_index": 1, "frame_score": 0.9, "estimate_quality": 0.8}
            ],
            "selected": [{"frame_index": 1}],
            "variant": "primary:no_controlnet:1024px",
        },
        {
            "mode": "ambient_only",
            "rig_mode": "envmap_only",
            "light_rig": [],
            "decomposition": {"method": "test", "analytic_light_count": 0},
            "sun_diagnostics": {"degraded_reason": "world_sun_incoherent"},
            "validation": {"passed": False, "failures": ["recovery_failed"]},
            "per_keyframe_results": [
                {"frame_index": 1, "frame_score": 0.9, "estimate_quality": 0.8}
            ],
            "selected": [{"frame_index": 1}],
            "variant": "recovery:no_controlnet:768px",
        },
    ]

    monkeypatch.setattr(provider, "validate_requirements", lambda resources: None)
    monkeypatch.setattr(
        provider,
        "_score_frames",
        lambda resources: [{"frame_index": 1, "score": 0.9}],
    )
    monkeypatch.setattr(
        provider,
        "_select_keyframes",
        lambda scored, count: [{"frame_index": 1, "score": 0.9}],
    )
    monkeypatch.setattr(
        provider,
        "_select_recovery_keyframes",
        lambda estimates, count: [{"frame_index": 1, "score": 0.9}],
    )
    monkeypatch.setattr(provider, "_write_attempt_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(provider, "_run_attempt", lambda **kwargs: attempt_results.pop(0))

    with pytest.raises(RuntimeError, match="failed to produce plausible lighting"):
        provider.run(resources)
    assert resources.saved is None


def test_lighting_provider_run_publishes_validated_recovery_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = DiffusionLightTurboLightingProvider({})
    resources = _DummyResources(tmp_path)
    envmap_path = tmp_path / "lighting.exr"
    envmap_path.write_bytes(b"hdr")

    attempt_results = [
        {
            "mode": "ambient_only",
            "rig_mode": "envmap_only",
            "light_rig": [],
            "decomposition": {"method": "test", "analytic_light_count": 0},
            "sun_diagnostics": {"degraded_reason": "world_sun_incoherent"},
            "validation": {"passed": True, "failures": [], "mode": "ambient_only"},
            "per_keyframe_results": [
                {"frame_index": 3, "frame_score": 0.9, "estimate_quality": 0.8}
            ],
            "selected": [{"frame_index": 3}],
            "variant": "primary:no_controlnet:1024px",
            "envmap_hdr": np.ones((4, 8, 3), dtype=np.float32) * 0.5,
            "sun_direction_world": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "sun_strength": 0.0,
            "sun_color": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "ambient_strength": 0.8,
            "quality": {"total": 0.7, "sun": 0.0, "envmap": 0.9},
            "per_keyframe_diagnostics": [{"frame_index": 3, "agreement_deg": None}],
        },
        {
            "mode": "full_sun",
            "rig_mode": "sun_plus_fill",
            "light_rig": [],
            "decomposition": {"method": "test", "analytic_light_count": 1},
            "sun_diagnostics": {"degraded_reason": None},
            "validation": {"passed": True, "checks": {"mean_luminance": 0.5}, "failures": []},
            "per_keyframe_results": [
                {"frame_index": 3, "frame_score": 0.9, "estimate_quality": 0.8}
            ],
            "selected": [{"frame_index": 3}],
            "variant": "recovery:no_controlnet:768px",
            "envmap_hdr": np.ones((4, 8, 3), dtype=np.float32),
            "sun_direction_world": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "sun_strength": 3.5,
            "sun_color": np.array([1.0, 0.95, 0.9], dtype=np.float32),
            "ambient_strength": 0.8,
            "quality": {"total": 0.7, "sun": 0.8, "envmap": 0.6},
            "per_keyframe_diagnostics": [{"frame_index": 3, "agreement_deg": 4.0}],
        },
    ]

    monkeypatch.setattr(provider, "validate_requirements", lambda resources: None)
    monkeypatch.setattr(
        provider,
        "_score_frames",
        lambda resources: [{"frame_index": 3, "score": 0.9}],
    )
    monkeypatch.setattr(
        provider,
        "_select_keyframes",
        lambda scored, count: [{"frame_index": 3, "score": 0.9}],
    )
    monkeypatch.setattr(
        provider,
        "_select_recovery_keyframes",
        lambda estimates, count: [{"frame_index": 3, "score": 0.9}],
    )
    monkeypatch.setattr(provider, "_write_attempt_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(provider, "_run_attempt", lambda **kwargs: attempt_results.pop(0))
    monkeypatch.setattr(provider, "_write_fused_envmap", lambda hdr, root: envmap_path)

    provider.run(resources)
    assert resources.saved is not None
    assert resources.saved.validation["passed"] is True
    assert resources.saved.mode == "full_sun"
    assert resources.saved.rig_mode == "sun_plus_fill"
    assert resources.saved.sun_diagnostics["degraded_reason"] is None
    assert resources.saved.recovery["used"] is True
    assert resources.saved.metadata["provider"] == "DiffusionLightTurboLightingProvider"


def test_lighting_provider_run_recovers_to_validated_envmap_only_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = DiffusionLightTurboLightingProvider({})
    resources = _DummyResources(tmp_path)
    envmap_path = tmp_path / "lighting.exr"
    envmap_path.write_bytes(b"hdr")

    attempt_result = {
        "mode": "ambient_only",
        "rig_mode": "analytic_rig",
        "light_rig": [{"name": "wrap", "kind": "POINT", "role": "wrap_key_fill", "strength": 1.0}],
        "decomposition": {"method": "test", "analytic_light_count": 1},
        "sun_diagnostics": {"degraded_reason": "world_sun_incoherent"},
        "validation": {
            "passed": False,
            "mode": "ambient_only",
            "rig_mode": "analytic_rig",
            "failures": ["dynamic_range_too_low", "ineffective_subject_fill_transport"],
            "checks": {
                "finite_ratio": 1.0,
                "mean_luminance": 0.29,
                "p95_luminance": 0.68,
                "max_luminance": 1.17,
                "relative_p95_ratio": 0.90,
                "dynamic_range_ratio": 3.81,
            },
        },
        "per_keyframe_results": [
            {"frame_index": 3, "frame_score": 0.9, "estimate_quality": 0.8}
        ],
        "selected": [{"frame_index": 3}],
        "variant": "primary:no_controlnet:1024px",
        "envmap_hdr": np.ones((4, 8, 3), dtype=np.float32) * 0.5,
        "sun_direction_world": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "sun_strength": 0.0,
        "sun_color": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "ambient_strength": 0.8,
        "quality": {"total": 0.69, "sun": 0.0, "envmap": 0.90},
        "per_keyframe_diagnostics": [{"frame_index": 3, "agreement_deg": None}],
    }

    monkeypatch.setattr(provider, "validate_requirements", lambda resources: None)
    monkeypatch.setattr(
        provider,
        "_score_frames",
        lambda resources: [{"frame_index": 3, "score": 0.9}],
    )
    monkeypatch.setattr(
        provider,
        "_select_keyframes",
        lambda scored, count: [{"frame_index": 3, "score": 0.9}],
    )
    monkeypatch.setattr(provider, "_write_attempt_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(provider, "_run_attempt", lambda **kwargs: attempt_result)
    monkeypatch.setattr(provider, "_write_fused_envmap", lambda hdr, root: envmap_path)

    provider.run(resources)

    assert resources.saved is not None
    assert resources.saved.validation["passed"] is True
    assert resources.saved.validation["degraded_fallback_used"] is True
    assert resources.saved.validation["original_failures"] == [
        "dynamic_range_too_low",
        "ineffective_subject_fill_transport",
    ]
    assert resources.saved.mode == "ambient_only"
    assert resources.saved.rig_mode == "envmap_only"
    assert resources.saved.light_rig == []


def test_lighting_provider_cross_run_cache_spec_requires_raw_and_standard_outputs(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "DiffusionLight-Turbo"
    repo_root.mkdir(parents=True)
    for script_name in ("inpaint.py", "ball2envmap.py", "exposure2hdr.py"):
        (repo_root / script_name).write_text("print('ok')\n", encoding="utf-8")

    cache = CrossRunCacheManager(tmp_path / "cache")
    store = ResourceStore("lighting_cache_spec", root=tmp_path)
    frames_dir = store.base_dir(ResourceKind.FRAMES)
    depth_dir = store.base_dir(ResourceKind.DEPTH)
    trajectory_path = store.path_for(ResourceKind.TRAJECTORY)
    semantics_dir = store.base_dir(ResourceKind.SEMANTICS_2D)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    semantics_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "000000.png").write_bytes(b"frame")
    np.savez_compressed(depth_dir / "000000.npz", depth=np.ones((2, 2), dtype=np.float32))
    np.savez_compressed(
        trajectory_path,
        frame_indices=np.array([0], dtype=np.int32),
        camera_to_world=np.eye(4, dtype=np.float32)[None],
    )
    np.savez_compressed(
        semantics_dir / "000000.npz",
        segment_ids=np.zeros((2, 2), dtype=np.int32),
    )

    provider = DiffusionLightTurboLightingProvider({"repo_root": str(repo_root)})
    provider.setup(
        {
            "cross_run_cache": cache,
            "cross_run_cache_stage_settings": {"lighting": {"enabled": True}},
            "profile_name": "test",
        }
    )
    payload = provider._cross_run_payload(store)
    assert payload is not None
    provider._cache_payload = payload
    provider._cache_signature = cache.signature("lighting", payload)

    not_ready = provider.get_cross_run_cache_spec(store)
    assert not_ready is not None
    assert not_ready["ready"] is False
    assert not_ready["not_ready_reason"] == "raw-lighting-missing"

    raw_dir = store.provider_dir("lighting")
    (raw_dir / "attempt.json").write_text("{}", encoding="utf-8")
    envmap_path = raw_dir / "envmap.exr"
    envmap_path.write_bytes(b"exr")
    store.save_lighting(
        LightingData(
            sun_direction_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            sun_strength=1.0,
            sun_color=np.ones((3,), dtype=np.float32),
            mode="full_sun",
            envmap_path=str(envmap_path),
            envmap_rotation_world=np.zeros((3,), dtype=np.float32),
            ambient_strength=0.2,
            rig_mode="sun_plus_fill",
            decomposition={"method": "unit-test", "analytic_light_count": 0},
            validation={"passed": True},
            metadata={"provider": "DiffusionLightTurboLightingProvider"},
        )
    )

    ready = provider.get_cross_run_cache_spec(store)
    assert ready is not None
    assert ready["ready"] is True
    assert "standard/lighting/lighting.json" in ready["artifacts"]
    assert "standard/lighting/envmap.exr" in ready["artifacts"]


def test_lighting_provider_cross_run_payload_is_stable_for_equivalent_rewritten_npz_inputs(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    for script_name in ("inpaint.py", "ball2envmap.py", "exposure2hdr.py"):
        (repo_root / script_name).write_text("print('ok')\n", encoding="utf-8")

    cache = CrossRunCacheManager(tmp_path / "cache")
    provider = DiffusionLightTurboLightingProvider({"repo_root": str(repo_root)})
    provider.setup(
        {
            "cross_run_cache": cache,
            "cross_run_cache_stage_settings": {"lighting": {"enabled": True}},
            "profile_name": "test",
        }
    )

    signatures: list[str] = []
    for run_name in ("run_a", "run_b"):
        store = _build_lighting_cache_store(tmp_path, run_name)
        payload = provider._cross_run_payload(store)
        assert payload is not None
        signatures.append(cache.signature("lighting", payload))

    assert signatures[0] == signatures[1]


def test_lighting_inference_cache_payload_ignores_planner_only_setting_changes(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    for script_name in ("inpaint.py", "ball2envmap.py", "exposure2hdr.py"):
        (repo_root / script_name).write_text("print('ok')\n", encoding="utf-8")

    cache = CrossRunCacheManager(tmp_path / "cache")
    provider_a = DiffusionLightTurboLightingProvider({"repo_root": str(repo_root)})
    provider_b = DiffusionLightTurboLightingProvider(
        {
            "repo_root": str(repo_root),
            "diffuse_demote_enabled": True,
            "diffuse_demote_aggressiveness": "moderate",
            "max_direct_to_fill_ratio_for_diffuse": 1.5,
            "fill_heavy_min_fill_count": 3,
            "fill_heavy_direct_scale": 0.35,
        }
    )
    for provider in (provider_a, provider_b):
        provider.setup(
            {
                "cross_run_cache": cache,
                "cross_run_cache_stage_settings": {"lighting": {"enabled": True}},
                "profile_name": "test",
            }
        )
    store = _build_lighting_cache_store(tmp_path, "run")
    selected = [{"frame_index": 0, "score": 0.9, "metrics": {}}]

    inference_sig_a = cache.signature(
        "lighting_dlt_inference",
        provider_a._dlt_inference_cache_payload(
            store,
            selected=selected,
            input_size=provider_a.settings.input_size,
            no_controlnet=provider_a.settings.no_controlnet,
        ),
    )
    inference_sig_b = cache.signature(
        "lighting_dlt_inference",
        provider_b._dlt_inference_cache_payload(
            store,
            selected=selected,
            input_size=provider_b.settings.input_size,
            no_controlnet=provider_b.settings.no_controlnet,
        ),
    )
    final_sig_a = cache.signature("lighting", provider_a._cross_run_payload(store))
    final_sig_b = cache.signature("lighting", provider_b._cross_run_payload(store))

    assert inference_sig_a == inference_sig_b
    assert final_sig_a != final_sig_b


def test_diffusion_light_turbo_settings_parse_fill_heavy_transport_controls() -> None:
    settings = DiffusionLightTurboSettings.from_mapping(
        {
            "repo_root": "tools/DiffusionLight-Turbo",
            "diffuse_softness_bias": 0.7,
            "fill_heavy_dark_side_target_ratio": 0.42,
            "fill_heavy_transport_gain": 1.6,
            "wrap_geometry_min_azimuth_separation_deg": 60.0,
            "wrap_geometry_counter_opposition_deg": 120.0,
            "wrap_geometry_sky_min_elevation_deg": 58.0,
            "wrap_geometry_candidate_count_per_role": 4,
        }
    )

    assert settings.diffuse_softness_bias == pytest.approx(0.7)
    assert settings.fill_heavy_dark_side_target_ratio == pytest.approx(0.42)
    assert settings.fill_heavy_transport_gain == pytest.approx(1.6)
    assert settings.wrap_geometry_min_azimuth_separation_deg == pytest.approx(60.0)
    assert settings.wrap_geometry_counter_opposition_deg == pytest.approx(120.0)
    assert settings.wrap_geometry_sky_min_elevation_deg == pytest.approx(58.0)
    assert settings.wrap_geometry_candidate_count_per_role == 4


def test_lighting_provider_cross_run_payload_ignores_volatile_transform_ids(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    for script_name in ("inpaint.py", "ball2envmap.py", "exposure2hdr.py"):
        (repo_root / script_name).write_text("print('ok')\n", encoding="utf-8")

    cache = CrossRunCacheManager(tmp_path / "cache")
    provider = DiffusionLightTurboLightingProvider({"repo_root": str(repo_root)})
    provider.setup(
        {
            "cross_run_cache": cache,
            "cross_run_cache_stage_settings": {"lighting": {"enabled": True}},
            "profile_name": "test",
        }
    )

    signatures: list[str] = []
    for run_name, alignment_id, grounding_id in (
        ("run_a", "align-a", "ground-a"),
        ("run_b", "align-b", "ground-b"),
    ):
        store = _build_lighting_cache_store_with_transform_ids(
            tmp_path,
            run_name,
            alignment_transform_id=alignment_id,
            grounding_transform_id=grounding_id,
        )
        payload = provider._cross_run_payload(store)
        assert payload is not None
        signatures.append(cache.signature("lighting", payload))

    assert signatures[0] == signatures[1]
