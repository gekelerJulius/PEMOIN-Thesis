from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from pemoin.data.contracts import _STANDARD_LAYOUTS
from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    FrameData,
    IntrinsicsData,
    LightingLightData,
    LightingData,
    ResourceKind,
    ResourceStore,
    RoadPlaneSupportData,
    SemanticsAuxData,
    TrajectoryMatchGraphData,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "data-contract.md"


def test_standard_data_contract_doc_covers_all_registered_layouts() -> None:
    assert DOC_PATH.exists(), "Canonical standardized data contract doc is missing."
    content = DOC_PATH.read_text(encoding="utf-8")
    for layout in _STANDARD_LAYOUTS.values():
        assert layout.subdir in content, f"Missing documented layout: {layout.subdir}"
    assert "must not consume" in content.lower()
    assert "`raw/`" in content or "raw/" in content


def test_downstream_modules_do_not_depend_on_raw_metadata_paths() -> None:
    forbidden_tokens = (
        "class_probabilities_path",
        "segformer_probabilities_path",
        "semantics_confidence_path",
        "road_confidence_path",
        "raw_root",
    )
    guarded_modules = [
        REPO_ROOT / "src/pemoin/runtime/runtime.py",
        REPO_ROOT / "src/pemoin/providers/geometry_fusion/stages/scale_alignment.py",
        REPO_ROOT / "src/pemoin/providers/geometry_fusion/utils/road_pixel_selection.py",
        REPO_ROOT / "src/pemoin/providers/point_cloud_3d/provider.py",
        REPO_ROOT / "src/pemoin/utils/geometry_validation.py",
        REPO_ROOT / "src/pemoin/visualization/semantics_debug.py",
    ]
    for path in guarded_modules:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"Forbidden downstream raw dependency {token!r} in {path}"


def test_strict_downstream_modules_do_not_open_raw_provider_dirs() -> None:
    guarded_modules = [
        REPO_ROOT / "src/pemoin/coordinate_systems/alignment.py",
        REPO_ROOT / "src/pemoin/utils/geometry_validation.py",
        REPO_ROOT / "src/pemoin/visualization/semantics_debug.py",
    ]
    for path in guarded_modules:
        text = path.read_text(encoding="utf-8")
        assert "provider_dir(" not in text, f"Raw provider_dir access is forbidden in {path}"
        assert "raw_root" not in text, f"Raw root access is forbidden in {path}"


def test_resource_store_roundtrip_for_standardized_intermediate_resources(tmp_path) -> None:
    store = ResourceStore("contract_roundtrip", root=tmp_path)

    store.save_semantics_aux(
        SemanticsAuxData(
            frame_index=3,
            class_probabilities=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            class_ids=np.array([4, 7], dtype=np.int32),
            confidence=np.full((3, 4), 0.9, dtype=np.float32),
            road_confidence=np.full((3, 4), 0.8, dtype=np.float32),
            validity_mask=np.ones((3, 4), dtype=bool),
            debug_maps={"road_agreement": np.full((3, 4), 0.7, dtype=np.float32)},
            model_outputs={
                "segformer": {
                    "label_ids": np.ones((3, 4), dtype=np.int32),
                    "confidence": np.full((3, 4), 0.6, dtype=np.float32),
                }
            },
            road_prior_outputs={"twinlite": np.full((3, 4), 0.5, dtype=np.float32)},
            metadata={"source": "unit-test"},
        )
    )
    aux = store.load_semantics_aux(3)
    assert aux.class_probabilities is not None
    assert aux.class_probabilities.shape == (2, 3, 4)
    assert aux.class_ids.tolist() == [4, 7]
    assert "road_agreement" in aux.debug_maps
    assert "segformer" in aux.model_outputs
    assert "twinlite" in aux.road_prior_outputs

    store.save_road_plane_support(
        RoadPlaneSupportData(
            frame_index=3,
            points_world=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            weights=np.array([0.5], dtype=np.float32),
            source_frame_index=1,
            diagnostics={"alignment_transform_id": "abc"},
            metadata={"source": "unit-test"},
        )
    )
    support = store.load_road_plane_support(3)
    assert support.source_frame_index == 1
    assert support.points_world.shape == (1, 3)
    assert support.diagnostics["alignment_transform_id"] == "abc"

    store.save_trajectory_match_graph(
        TrajectoryMatchGraphData(
            payload={
                "schema_version": np.int32(2),
                "coord_space": np.array("full_res_pixels"),
                "res_factor": np.int32(4),
                "edge_src_frame_id": np.array([0], dtype=np.int32),
                "edge_tgt_frame_id": np.array([1], dtype=np.int32),
            },
            metadata={"source": "unit-test"},
        )
    )
    match_graph = store.load_trajectory_match_graph()
    assert int(np.asarray(match_graph.payload["schema_version"]).reshape(())) == 2
    assert str(np.asarray(match_graph.payload["coord_space"]).reshape(())) == "full_res_pixels"
    assert match_graph.metadata["source"] == "unit-test"

    assert store.has(ResourceKind.SEMANTICS_AUX)
    assert store.has(ResourceKind.ROAD_PLANE_SUPPORT)
    assert store.has(ResourceKind.TRAJECTORY_MATCH_GRAPH)

    envmap_path = tmp_path / "input_envmap.exr"
    envmap = np.full((8, 16, 3), 0.25, dtype=np.float32)
    assert cv2.imwrite(str(envmap_path), cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR))
    store.save_lighting(
        LightingData(
            sun_direction_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            sun_strength=2.5,
            sun_color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
            mode="full_sun",
            envmap_path=str(envmap_path),
            envmap_rotation_world=np.zeros((3,), dtype=np.float32),
            ambient_strength=0.4,
            schema_version=2,
            rig_mode="analytic_rig",
            light_rig=[
                LightingLightData(
                    name="DirectSun",
                    kind="SUN",
                    role="direct_key",
                    strength=2.5,
                    color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
                    casts_shadow=True,
                    direction_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    angular_size_deg=2.0,
                ),
                LightingLightData(
                    name="DiffuseFill",
                    kind="POINT",
                    role="wrap_key_fill",
                    strength=1.5,
                    color=np.array([0.9, 0.95, 1.0], dtype=np.float32),
                    casts_shadow=False,
                    placement_mode="subject_anchor_relative",
                    placement_target="subject_root_dynamic",
                    direction_world=np.array([0.4, 0.2, -0.89], dtype=np.float32),
                    location_world=np.array([-3.0, -2.0, 4.5], dtype=np.float32),
                    diagnostics={"transport_mode": "wrap_subject_fill"},
                )
            ],
            decomposition={"method": "unit-test", "analytic_light_count": 1},
            quality={"total": 0.8, "sun": 0.9, "envmap": 0.7},
            sun_diagnostics={
                "camera_cluster_count": 2,
                "camera_mean_spread_deg": 12.0,
                "world_mean_spread_deg": 10.0,
                "winning_frame_indices": [1, 4],
                "degraded_reason": None,
                "candidate_count": 4,
            },
            validation={"passed": True, "checks": {"mean_luminance": 0.3}},
            recovery={"used": False, "reason": None},
            selected_frame_indices=[1, 4, 8],
            per_keyframe_diagnostics=[{"frame_index": 1}],
            metadata={
                "source": "unit-test",
                "provider": "DiffusionLightTurboLightingProvider",
            },
        )
    )
    lighting = store.load_lighting()
    assert lighting.mode == "full_sun"
    assert lighting.schema_version == 2
    assert lighting.rig_mode == "analytic_rig"
    assert lighting.selected_frame_indices == [1, 4, 8]
    assert lighting.quality["total"] == 0.8
    assert len(lighting.light_rig) == 2
    assert lighting.light_rig[0].casts_shadow is True
    assert lighting.light_rig[1].placement_mode == "subject_anchor_relative"
    assert lighting.light_rig[1].placement_target == "subject_root_dynamic"
    assert lighting.light_rig[1].kind == "POINT"
    assert lighting.light_rig[1].role == "wrap_key_fill"
    assert lighting.light_rig[1].diagnostics["transport_mode"] == "wrap_subject_fill"
    assert lighting.sun_diagnostics["camera_cluster_count"] == 2
    assert lighting.validation["passed"] is True
    assert lighting.recovery["used"] is False
    assert Path(lighting.envmap_path).exists()
    assert store.has(ResourceKind.LIGHTING)


def test_load_lighting_rejects_legacy_or_incomplete_payload(tmp_path: Path) -> None:
    store = ResourceStore("lighting_contract_strict", root=tmp_path)
    lighting_dir = store.base_dir(ResourceKind.LIGHTING)
    lighting_dir.mkdir(parents=True, exist_ok=True)
    envmap_path = lighting_dir / "envmap.exr"
    envmap = np.full((8, 16, 3), 0.25, dtype=np.float32)
    assert cv2.imwrite(str(envmap_path), cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR))
    (lighting_dir / "lighting.json").write_text(
        '{"provider":"legacy","mode":"full_sun","validation":{"passed":true},"recovery":{},"sun_diagnostics":{}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields: schema_version, rig_mode, light_rig, quality"):
        store.load_lighting()


def test_load_lighting_rejects_invalid_dynamic_placement_target(tmp_path: Path) -> None:
    store = ResourceStore("lighting_contract_dynamic_target", root=tmp_path)
    lighting_dir = store.base_dir(ResourceKind.LIGHTING)
    lighting_dir.mkdir(parents=True, exist_ok=True)
    envmap_path = lighting_dir / "envmap.exr"
    envmap = np.full((8, 16, 3), 0.25, dtype=np.float32)
    assert cv2.imwrite(str(envmap_path), cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR))
    (lighting_dir / "lighting.json").write_text(
        """
        {
          "provider": "test",
          "schema_version": 2,
          "rig_mode": "analytic_rig",
          "mode": "full_sun",
          "sun_direction_world": [0.0, 0.0, 1.0],
          "sun_strength": 1.0,
          "sun_color": [1.0, 1.0, 1.0],
          "envmap_path": "standard/lighting/envmap.exr",
          "envmap_rotation_world": [0.0, 0.0, 0.0],
          "ambient_strength": 0.2,
          "light_rig": [
            {
              "name": "BadFill",
              "kind": "AREA",
              "role": "diffuse_fill",
              "strength": 1.0,
              "color": [1.0, 1.0, 1.0],
              "casts_shadow": false,
              "placement_mode": "subject_anchor_relative",
              "placement_target": "world",
              "direction_world": [0.0, 1.0, -1.0],
              "location_world": [0.0, 0.0, 1.0],
              "area_size": [2.0, 2.0]
            }
          ],
          "quality": {},
          "sun_diagnostics": {},
          "validation": {"passed": true},
          "recovery": {},
          "decomposition": {}
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="placement_target must not be 'world'"):
        store.load_lighting()


def test_resource_store_save_operations_do_not_write_visualization_side_effects(
    tmp_path: Path,
) -> None:
    store = ResourceStore("no_side_effects", root=tmp_path)
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 8.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"width": 16, "height": 16},
        )
    )
    store.save_depth(
        DepthData(
            frame_index=0,
            depth=np.ones((16, 16), dtype=np.float32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=0,
            height_m=1.7,
            metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
        )
    )

    assert not (store.standard_root / "visualizations").exists()


def test_contract_models_import_without_imageio(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "pemoin.data.contracts", raising=False)
    monkeypatch.delitem(sys.modules, "pemoin.data.store", raising=False)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "imageio" or name.startswith("imageio."):
            raise ModuleNotFoundError("No module named 'imageio'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    contracts = importlib.import_module("pemoin.data.contracts")

    assert contracts.FrameData.__name__ == "FrameData"
    assert contracts.SemanticsData.__name__ == "SemanticsData"


def test_resource_store_defers_imageio_requirement_until_png_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pemoin.data.store as store_module

    monkeypatch.setattr(store_module, "imageio", None)

    store = store_module.ResourceStore("no_imageio", root=tmp_path)
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 8.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"width": 16, "height": 16},
        )
    )
    assert store.load_intrinsics().metadata["width"] == 16

    with pytest.raises(RuntimeError, match="require imageio"):
        store.save_frame(
            FrameData(
                frame_id="000000",
                index=0,
                image=np.zeros((4, 4, 3), dtype=np.uint8),
            )
        )
