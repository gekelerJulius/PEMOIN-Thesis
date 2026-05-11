from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "src" / "pemoin" / "scripts" / "blender_trajectory_scene.py"


def _install_blender_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    bpy_module = types.ModuleType("bpy")
    bpy_module.context = types.SimpleNamespace(
        scene=types.SimpleNamespace(render=types.SimpleNamespace())
    )
    bpy_module.types = types.SimpleNamespace(
        Object=object,
        Scene=object,
        Collection=object,
        Camera=object,
        Material=object,
    )
    mathutils_module = types.ModuleType("mathutils")
    mathutils_module.Matrix = type("Matrix", (), {})
    mathutils_module.Vector = type("Vector", (), {})
    monkeypatch.setitem(sys.modules, "bpy", bpy_module)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils_module)


def test_blender_scene_pure_modules_import_without_bpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "bpy", raising=False)
    monkeypatch.delitem(sys.modules, "mathutils", raising=False)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.config", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.specs", raising=False
    )

    specs = importlib.import_module("pemoin.visualization.blender_scene.specs")
    config = importlib.import_module("pemoin.visualization.blender_scene.config")

    assert specs.SceneSpec.__name__ == "SceneSpec"
    assert callable(config.parse_args)


def test_blender_scene_app_imports_without_imageio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    original_import = builtins.__import__

    for name in (
        "pemoin.visualization.blender_scene.app",
        "pemoin.visualization.blender_scene.grounding",
        "pemoin.visualization.blender_scene.pipeline",
        "pemoin.visualization.overlay_compositor",
        "pemoin.utils.resolution",
        "pemoin.data.contracts",
        "pemoin.data.store",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "imageio" or name.startswith("imageio."):
            raise ModuleNotFoundError("No module named 'imageio'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module("pemoin.visualization.blender_scene.app")

    assert callable(module.run_scene)


def test_load_overlay_ground_mask_resizes_standardized_semantics_to_overlay_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")
    importlib.import_module("pemoin.visualization.blender_scene.specs")

    semantics_dir = tmp_path / "standard" / "semantics_2d"
    semantics_dir.mkdir(parents=True, exist_ok=True)
    label_ids = np.full((281, 500), 3, dtype=np.int32)
    label_ids[140:, :] = 7
    np.savez_compressed(
        semantics_dir / "000000.npz",
        label_ids=label_ids,
        metadata={"class_id_to_label": {"3": "road", "7": "building"}},
        segments_info=np.asarray([], dtype=object),
    )

    mask = pipeline._load_overlay_ground_mask(
        run_dir=tmp_path,
        frame_idx=0,
        image_shape=(140, 250),
        ground_labels=("road",),
        required=True,
    )

    assert mask.shape == (140, 250)
    assert mask.dtype == np.bool_
    assert bool(mask[20, 20]) is True
    assert bool(mask[-1, -1]) is False


def test_overlay_validation_marks_untrusted_trajectory_path_frames_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    monkeypatch.setattr(
        pipeline,
        "_compute_support_road_context",
        lambda **_: (None, None, 0, False, False, None, "no_support_point"),
    )

    ped_rgba = np.zeros((202, 360, 4), dtype=np.uint8)
    ped_rgba[40:120, 120:180, 3] = 255

    diagnostic = pipeline._make_overlay_validation_diagnostic(
        run_dir=tmp_path,
        frame_idx=49,
        ped_rgba=ped_rgba,
        overlay_shape=(405, 720),
        left_uv=np.asarray([260.0, 300.0], dtype=np.float32),
        left_valid=True,
        right_uv=np.asarray([280.0, 300.0], dtype=np.float32),
        right_valid=True,
        selected_support_foot="path",
        support_mode="trajectory_path",
        support_point_uv=np.asarray([286.0, 314.0], dtype=np.float32),
        support_point_visible=False,
        support_point_depth_m=9.4,
        road_labels=("road",),
    )

    assert diagnostic.contact_validation_state == "unverifiable"
    assert diagnostic.abort_relevant is False
    assert diagnostic.failure_reason == "support_point_not_visible"
    assert diagnostic.internal_render_shape == (202, 360)
    assert diagnostic.overlay_shape == (405, 720)


def test_draw_support_marker_rgba_uses_top_origin_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    image = np.zeros((8, 9, 4), dtype=np.float32)
    pipeline._draw_support_marker_rgba(image, 4, 2, color=(1.0, 0.0, 0.0))

    expected_patch = image[1:4, 3:6, :]
    mirrored_patch = image[4:7, 3:6, :]

    assert np.all(expected_patch[:, :, 0] == 1.0)
    assert np.all(expected_patch[:, :, 1] == 0.0)
    assert np.all(expected_patch[:, :, 2] == 0.0)
    assert np.all(expected_patch[:, :, 3] == 1.0)
    assert np.all(mirrored_patch == 0.0)


def test_partition_render_frame_indices_splits_visible_and_culled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    diagnostics = [
        pipeline.GroundingDiagnostic(
            frame_index=5,
            support_mode="supported",
            support_confidence=1.0,
            support_source_frame_indices=(),
            support_failure_reason=None,
            sole_offset_m=0.0,
            chosen_plane_frame_index=None,
            chosen_plane_normal=None,
            chosen_plane_offset=None,
            chosen_plane_center=None,
            chosen_plane_center_xy_distance_m=None,
            selected_support_foot="left",
            left_foot_before=None,
            right_foot_before=None,
            left_foot_after=None,
            right_foot_after=None,
            support_point_before=None,
            support_point_after=None,
            pre_correction_signed_distance_m=None,
            post_correction_signed_distance_m=None,
            left_post_signed_distance_m=None,
            right_post_signed_distance_m=None,
            support_jump_from_prev_deg=None,
            support_height_jump_from_prev_m=None,
            support_anchor_shift_from_prev_m=None,
            dynamic_anchor_shift_limit_m=None,
            applied_translation_world=np.zeros(3, dtype=np.float32),
            plane_selection_rejected_for_locality=False,
            missing_left_foot=False,
            missing_right_foot=False,
            no_plane=False,
            visibility_culled=True,
            visibility_cull_reason="actor_off_camera",
        ),
        pipeline.GroundingDiagnostic(
            frame_index=2,
            support_mode="supported",
            support_confidence=1.0,
            support_source_frame_indices=(),
            support_failure_reason=None,
            sole_offset_m=0.0,
            chosen_plane_frame_index=None,
            chosen_plane_normal=None,
            chosen_plane_offset=None,
            chosen_plane_center=None,
            chosen_plane_center_xy_distance_m=None,
            selected_support_foot="left",
            left_foot_before=None,
            right_foot_before=None,
            left_foot_after=None,
            right_foot_after=None,
            support_point_before=None,
            support_point_after=None,
            pre_correction_signed_distance_m=None,
            post_correction_signed_distance_m=None,
            left_post_signed_distance_m=None,
            right_post_signed_distance_m=None,
            support_jump_from_prev_deg=None,
            support_height_jump_from_prev_m=None,
            support_anchor_shift_from_prev_m=None,
            dynamic_anchor_shift_limit_m=None,
            applied_translation_world=np.zeros(3, dtype=np.float32),
            plane_selection_rejected_for_locality=False,
            missing_left_foot=False,
            missing_right_foot=False,
            no_plane=False,
            visibility_culled=False,
        ),
    ]

    visible, culled = pipeline._partition_render_frame_indices(diagnostics)

    assert visible == [2]
    assert culled == [5]


def test_compute_render_salience_from_projected_bbox_protects_tiny_visible_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    metrics = pipeline._compute_render_salience_from_projected_bbox(
        bbox=(0.0, 10.0, 40.0, 70.0),
        image_shape=(180, 320),
    )
    tier, reason = pipeline._render_salience_policy(
        visible_pixels=metrics["visible_pixels"],
        bbox_short_side_px=metrics["bbox_short_side_px"],
        center_distance_ratio=metrics["center_distance_ratio"],
        boundary_fraction=metrics["boundary_fraction"],
        protect_below_visible_pixels=10000,
        protect_below_bbox_short_side_px=56,
        protect_when_center_distance_ratio_below=0.30,
        reduce_only_when_boundary_fraction_above=0.24,
        near_visibility_transition=True,
        reduce_only_near_visibility_transition=True,
    )

    assert metrics["visible_pixels"] == pytest.approx(2400.0)
    assert metrics["bbox_short_side_px"] == pytest.approx(40.0)
    assert metrics["boundary_fraction"] == pytest.approx(0.25)
    assert tier == "baseline_protected"
    assert reason == "protected_tiny_visible_subject"


def test_render_salience_policy_only_reduces_boundary_transition_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    tier, reason = pipeline._render_salience_policy(
        visible_pixels=18000.0,
        bbox_short_side_px=72.0,
        center_distance_ratio=0.72,
        boundary_fraction=0.25,
        protect_below_visible_pixels=10000,
        protect_below_bbox_short_side_px=56,
        protect_when_center_distance_ratio_below=0.30,
        reduce_only_when_boundary_fraction_above=0.24,
        near_visibility_transition=True,
        reduce_only_near_visibility_transition=True,
    )

    assert tier == "reduced_allowed"
    assert reason == "reduced_visibility_transition_boundary"


def test_temporary_low_salience_render_policy_reduces_fill_lights_and_shadow_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    class FakeLightData:
        def __init__(self, energy: float, use_shadow: bool) -> None:
            self.energy = energy
            self.use_shadow = use_shadow

    class FakeLight:
        def __init__(self, name: str, energy: float, use_shadow: bool, props: dict[str, object]) -> None:
            self.type = "LIGHT"
            self.name = name
            self.data = FakeLightData(energy, use_shadow)
            self._props = dict(props)

        def get(self, key: str, default=None):
            return self._props.get(key, default)

    eevee = types.SimpleNamespace(shadow_cube_size="1024", shadow_cascade_size="1024")
    scene = types.SimpleNamespace(eevee=eevee, eevee_next=None)
    bpy_stub = types.SimpleNamespace(
        context=types.SimpleNamespace(scene=scene),
        data=types.SimpleNamespace(
            objects=[
                FakeLight(
                    "WrapFill",
                    10.0,
                    True,
                    {
                        "pemoin_light_role": "wrap_key_fill",
                        pipeline._PEMOIN_LIGHT_TRANSPORT_MODE: pipeline._WRAP_SUBJECT_FILL_TRANSPORT_MODE,
                    },
                ),
                FakeLight("DirectSun", 5.0, True, {"pemoin_light_role": "direct_key"}),
            ]
        ),
    )
    monkeypatch.setattr(pipeline, "bpy", bpy_stub)
    spec = types.SimpleNamespace(
        render=types.SimpleNamespace(
            salience_adaptive=types.SimpleNamespace(
                fill_light_reduction_enabled=True,
                shadow_quality_reduction_enabled=True,
            )
        ),
        shadow=types.SimpleNamespace(map_resolution="1024"),
    )

    fill_light = bpy_stub.data.objects[0]
    direct_light = bpy_stub.data.objects[1]
    with pipeline._temporary_low_salience_render_policy(spec=spec) as diag:
        assert fill_light.data.energy == pytest.approx(8.8)
        assert fill_light.data.use_shadow is False
        assert direct_light.data.energy == pytest.approx(5.0)
        assert direct_light.data.use_shadow is True
        assert eevee.shadow_cube_size == "1024"
        assert diag["reduced_fill_light_count"] == 1
        assert diag["effective_shadow_map_resolution"] == "1024"

    assert fill_light.data.energy == pytest.approx(10.0)
    assert fill_light.data.use_shadow is True
    assert eevee.shadow_cube_size == "1024"


def test_partition_visible_render_frames_by_salience_protects_small_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    bbox_by_frame = {
        7: (480.0, 149.0, 500.0, 181.0),
        8: (448.0, 146.0, 466.0, 178.0),
        9: (419.0, 144.0, 446.0, 177.0),
        30: (0.0, 214.0, 92.0, 421.0),
    }
    monkeypatch.setattr(
        pipeline,
        "_projected_actor_bbox_for_frame",
        lambda **kwargs: bbox_by_frame[int(kwargs["frame_idx"])],
    )
    spec = types.SimpleNamespace(
        render=types.SimpleNamespace(
            salience_adaptive=types.SimpleNamespace(
                enabled=True,
                low_salience_resolution_scale=0.85,
                protect_below_visible_pixels=10000,
                protect_below_bbox_short_side_px=56,
                protect_when_center_distance_ratio_below=0.30,
                reduce_only_when_boundary_fraction_above=0.24,
                reduce_only_near_visibility_transition=True,
            )
        )
    )

    baseline, reduced, diagnostics = pipeline._partition_visible_render_frames_by_salience(
        visible_frame_indices=[7, 8, 9, 30],
        spec=spec,
        actor_root=object(),
        intrinsics_k=np.eye(3, dtype=np.float32),
        frame_to_c2w={7: np.eye(4), 8: np.eye(4), 9: np.eye(4), 30: np.eye(4)},
        image_shape=(422, 750),
    )

    assert baseline == [7, 8, 9]
    assert reduced == [30]
    assert diagnostics["baseline_protected_frame_count"] == 3
    assert diagnostics["reduced_allowed_frame_count"] == 1


def test_write_visibility_culled_frame_artifacts_materializes_empty_sequences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    frames_dir = tmp_path / "pedestrian_frames"
    depth_dir = tmp_path / "pedestrian_depth_frames"
    shadow_dir = tmp_path / "shadow_frames"

    pipeline._write_visibility_culled_frame_artifacts(
        frames_dir=frames_dir,
        depth_dir=depth_dir,
        shadow_dir=shadow_dir,
        frame_indices=[3, 1],
        image_shape=(4, 6),
    )

    assert (frames_dir / "frame_0001.png").exists()
    assert (frames_dir / "frame_0003.png").exists()
    assert (shadow_dir / "shadow_0001.png").exists()
    assert (shadow_dir / "shadow_0003.png").exists()
    depth = pipeline._load_depth_npz_array(depth_dir / "000001.npz")
    assert depth.shape == (4, 6)
    assert np.count_nonzero(depth) == 0


def test_normalize_mixamo_material_graphs_flattens_secondary_maps_in_fast_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    class _FakeSocket:
        def __init__(self, node, name: str) -> None:
            self.node = node
            self.name = name
            self.default_value = None

    class _FakeSocketMap(dict):
        def __iter__(self):
            return iter(self.values())

    class _FakeLink:
        def __init__(self, from_socket, to_socket) -> None:
            self.from_socket = from_socket
            self.to_socket = to_socket
            self.from_node = from_socket.node
            self.to_node = to_socket.node

    class _FakeLinks(list):
        def new(self, from_socket, to_socket) -> None:
            self.append(_FakeLink(from_socket, to_socket))

        def remove(self, link) -> None:
            super().remove(link)

    class _FakeImage:
        def __init__(self, name: str) -> None:
            self.name = name
            self.filepath = name
            self.filepath_raw = name
            self.colorspace_settings = types.SimpleNamespace(name=None)

    class _FakeNode:
        def __init__(self, node_type: str, name: str) -> None:
            self.type = node_type
            self.name = name
            self.location = types.SimpleNamespace(x=0.0, y=0.0)
            self.image = None
            self.inputs = _FakeSocketMap()
            self.outputs = _FakeSocketMap()
            if node_type == "BSDF_PRINCIPLED":
                for socket_name in (
                    "Base Color",
                    "Normal",
                    "Roughness",
                    "Specular IOR Level",
                    "Alpha",
                ):
                    self.inputs[socket_name] = _FakeSocket(self, socket_name)
            elif node_type == "TEX_IMAGE":
                self.outputs["Color"] = _FakeSocket(self, "Color")
                self.outputs["Alpha"] = _FakeSocket(self, "Alpha")
            elif node_type == "NORMAL_MAP":
                self.inputs["Color"] = _FakeSocket(self, "Color")
                self.outputs["Normal"] = _FakeSocket(self, "Normal")
            elif node_type == "INVERT":
                self.inputs["Color"] = _FakeSocket(self, "Color")
                self.outputs["Color"] = _FakeSocket(self, "Color")

    class _FakeNodes(list):
        def get(self, name: str):
            for node in self:
                if node.name == name:
                    return node
            return None

        def new(self, *, type: str):
            type_name = {
                "ShaderNodeNormalMap": "NORMAL_MAP",
                "ShaderNodeInvert": "INVERT",
            }.get(type, type)
            node = _FakeNode(type_name, f"{type_name}_{len(self)}")
            self.append(node)
            return node

    node_tree = types.SimpleNamespace(nodes=_FakeNodes(), links=_FakeLinks())
    principled = _FakeNode("BSDF_PRINCIPLED", "Principled BSDF")
    base = _FakeNode("TEX_IMAGE", "Base Color")
    base.image = _FakeImage("character_basecolor.png")
    normal = _FakeNode("TEX_IMAGE", "Normal")
    normal.image = _FakeImage("character_normal.png")
    roughness = _FakeNode("TEX_IMAGE", "Roughness")
    roughness.image = _FakeImage("character_roughness.png")
    specular = _FakeNode("TEX_IMAGE", "Specular")
    specular.image = _FakeImage("character_specular.png")
    alpha = _FakeNode("TEX_IMAGE", "Alpha")
    alpha.image = _FakeImage("character_alpha.png")
    node_tree.nodes.extend([principled, base, normal, roughness, specular, alpha])
    node_tree.links.new(roughness.outputs["Color"], principled.inputs["Roughness"])
    node_tree.links.new(specular.outputs["Color"], principled.inputs["Specular IOR Level"])

    material = types.SimpleNamespace(use_nodes=True, node_tree=node_tree)
    diagnostics = pipeline._normalize_mixamo_material_graphs(
        {"Body": material},
        material_policy="preserve_base_alpha_normal",
    )

    assert diagnostics["material_policy"] == "preserve_base_alpha_normal"
    assert diagnostics["normal_link_count"] == 1
    assert diagnostics["flattened_roughness_count"] == 1
    assert diagnostics["flattened_specular_count"] == 1
    assert diagnostics["flattened_normal_count"] == 0
    assert principled.inputs["Roughness"].default_value == pytest.approx(0.65)
    assert principled.inputs["Specular IOR Level"].default_value == pytest.approx(0.35)
    roughness_links = [
        link for link in node_tree.links
        if link.to_node == principled and link.to_socket.name == "Roughness"
    ]
    specular_links = [
        link for link in node_tree.links
        if link.to_node == principled and link.to_socket.name == "Specular IOR Level"
    ]
    normal_links = [
        link for link in node_tree.links
        if link.to_node == principled and link.to_socket.name == "Normal"
    ]
    assert not roughness_links
    assert not specular_links
    assert len(normal_links) == 1


def test_configure_render_engine_applies_fast_backend_toggles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")
    specs = importlib.import_module("pemoin.visualization.blender_scene.specs")

    image_settings = types.SimpleNamespace(file_format=None, color_mode=None, compression=15)
    render = types.SimpleNamespace(
        engine=None,
        image_settings=image_settings,
        use_persistent_data=False,
        use_motion_blur=True,
    )
    eevee = types.SimpleNamespace(
        taa_render_samples=0,
        taa_samples=0,
        use_shadows=False,
        use_raytracing=True,
        use_volumetric_shadows=True,
        use_volumetric_lights=True,
        use_bloom=True,
        use_ssr=True,
        use_gtao=True,
        use_high_quality_normals=True,
        use_soft_shadows=False,
        shadow_cube_size="512",
        shadow_cascade_size="512",
    )
    scene = types.SimpleNamespace(render=render, eevee=eevee)
    monkeypatch.setattr(
        pipeline,
        "bpy",
        types.SimpleNamespace(context=types.SimpleNamespace(scene=scene)),
    )
    monkeypatch.setattr(pipeline, "_preferred_raster_engine", lambda: "BLENDER_EEVEE")

    spec = specs.SceneSpec(
        run_dir=Path("/tmp/run"),
        trajectory_path=Path("/tmp/run/standard/trajectory/poses.npz"),
        output_path=None,
        cube_size=1.0,
        collection_name="Trajectory",
        render=specs.RenderSpec(
            samples=12,
            performance=specs.RenderPerformanceSpec(),
        ),
        shadow=specs.ShadowSpec(map_resolution="2048"),
    )

    pipeline.configure_render_engine(spec)

    assert render.engine == "BLENDER_EEVEE"
    assert render.use_persistent_data is True
    assert render.use_motion_blur is False
    assert getattr(scene, "_pemoin_fast_png_compression") is True
    assert eevee.taa_render_samples == 12
    assert eevee.taa_samples == 12
    assert eevee.use_shadows is True
    assert eevee.use_raytracing is False
    assert eevee.use_volumetric_shadows is False
    assert eevee.use_volumetric_lights is False
    assert eevee.use_bloom is False
    assert eevee.use_ssr is False
    assert eevee.use_gtao is False
    assert eevee.use_high_quality_normals is False
    assert eevee.use_soft_shadows is True
    assert eevee.shadow_cube_size == "2048"
    assert eevee.shadow_cascade_size == "2048"


def test_compose_overlay_frames_projects_support_in_overlay_space(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    background_path = tmp_path / "background.png"
    pedestrian_path = tmp_path / "pedestrian.png"
    original_dir = tmp_path / "original"
    pedestrian_dir = tmp_path / "pedestrian"
    output_dir = tmp_path / "output"
    original_dir.mkdir()
    pedestrian_dir.mkdir()
    output_dir.mkdir()
    background_path.write_bytes(b"")
    pedestrian_path.write_bytes(b"")

    captured: dict[str, object] = {}

    class _FakeSavedImage:
        def __init__(self, name: str, width: int, height: int, alpha: bool) -> None:
            self.name = name
            self.width = width
            self.height = height
            self.alpha = alpha
            self.pixels = None
            self.filepath_raw = None
            self.file_format = None

        def save(self) -> None:
            return None

    fake_images = types.SimpleNamespace(
        new=lambda name, width, height, alpha: _FakeSavedImage(name, width, height, alpha),
        remove=lambda _image: None,
    )
    fake_scene = types.SimpleNamespace(frame_set=lambda _frame: None)
    fake_view_layer = types.SimpleNamespace(update=lambda: None)
    monkeypatch.setattr(
        pipeline,
        "bpy",
        types.SimpleNamespace(
            context=types.SimpleNamespace(scene=fake_scene, view_layer=fake_view_layer),
            data=types.SimpleNamespace(images=fake_images),
        ),
    )

    monkeypatch.setattr(
        pipeline,
        "_build_frame_index_map",
        lambda path: {0: background_path} if path == original_dir else {0: pedestrian_path},
    )
    monkeypatch.setattr(
        pipeline,
        "_load_overlay_validation_context",
        lambda _run_dir: (
            np.eye(3, dtype=np.float32),
            {0: np.eye(4, dtype=np.float32)},
            None,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_overlay_validation_policy_for_run",
        lambda _run_dir: object(),
    )
    monkeypatch.setattr(pipeline, "_resolve_actor_root", lambda _name: object())
    monkeypatch.setattr(pipeline, "_find_actor_armature", lambda _name: object())
    monkeypatch.setattr(pipeline, "_build_support_point_lookup", lambda _diags: {0: np.zeros(3, dtype=np.float32)})
    grounding_diag = types.SimpleNamespace(
        support_mode="trajectory_path",
        selected_support_foot="path",
        chosen_plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        chosen_plane_offset=0.0,
    )
    monkeypatch.setattr(
        pipeline,
        "_build_grounding_diagnostic_lookup",
        lambda _diags: {0: grounding_diag},
    )
    monkeypatch.setattr(
        pipeline,
        "_load_rgba_image",
        lambda path: (
            np.zeros((405, 720, 4), dtype=np.uint8)
            if path == background_path
            else np.pad(
                np.full((202, 360, 4), 255, dtype=np.uint8),
                ((0, 0), (0, 0), (0, 0)),
            )
        ),
    )

    def _capture_support_projection(**kwargs):
        captured["image_shape"] = kwargs["image_shape"]
        return np.asarray([286.0, 314.0], dtype=np.float32), True, 9.4

    monkeypatch.setattr(pipeline, "_project_overlay_support_point", _capture_support_projection)
    monkeypatch.setattr(pipeline, "_load_overlay_ground_mask", lambda **_: np.ones((405, 720), dtype=bool))
    monkeypatch.setattr(
        pipeline,
        "compose_overlay_frame_with_occlusion",
        lambda **_: (
            np.zeros((405, 720, 3), dtype=np.uint8),
            np.ones((405, 720), dtype=bool),
            pipeline.OcclusionFrameDiagnostics(
                frame_index=0,
                pedestrian_pixels=100,
                visible_pixels=100,
                occluded_pixels=0,
                visible_ratio=1.0,
                min_scene_depth_m=1.0,
                max_scene_depth_m=1.0,
                min_ped_depth_m=1.0,
                max_ped_depth_m=1.0,
                median_depth_margin_m=0.1,
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_evaluate_feet_world",
        lambda *_: {"left": np.zeros(3, dtype=np.float32), "right": np.zeros(3, dtype=np.float32)},
    )
    monkeypatch.setattr(
        pipeline,
        "_project_overlay_feet",
        lambda **_: (
            np.asarray([260.0, 300.0], dtype=np.float32),
            True,
            np.asarray([280.0, 300.0], dtype=np.float32),
            True,
        ),
    )

    def _capture_validation(**kwargs):
        captured["overlay_shape"] = kwargs["overlay_shape"]
        return pipeline.OverlayValidationDiagnostic(
            frame_index=0,
            has_visible_pedestrian=True,
            internal_render_shape=(202, 360),
            overlay_shape=kwargs["overlay_shape"],
            lowest_alpha_u=100,
            lowest_alpha_v=100,
            lowest_alpha_row_coverage_px=10,
            touches_image_bottom=False,
            left_foot_projected_uv=None,
            right_foot_projected_uv=None,
            left_foot_visible_expected=True,
            right_foot_visible_expected=True,
            support_point_projected_uv=np.asarray([286.0, 314.0], dtype=np.float32),
            support_point_projected_visible=True,
            support_point_depth_m=9.4,
            scene_depth_at_support_px_m=9.0,
            support_point_occluded_by_scene=False,
            selected_foot_projected_uv=None,
            selected_foot_projected_visible=False,
            support_to_left_foot_px=None,
            support_to_right_foot_px=None,
            support_to_contact_foot_px=None,
            contact_foot_comparison_mode="trajectory_path",
            support_to_silhouette_bottom_px=None,
            support_to_selected_foot_px=None,
            support_patch_road_fraction=1.0,
            support_patch_nonroad_fraction=0.0,
            support_patch_size_px=1,
            road_region_validation_available=True,
            road_context_search_mode="direct_patch",
            validation_passed=True,
            failure_reason=None,
            support_mode="trajectory_path",
            selected_support_foot="path",
            warning_flags=tuple(),
            contact_validation_state="unverifiable",
            contact_validation_trusted=False,
            abort_relevant=False,
        )

    monkeypatch.setattr(pipeline, "_make_overlay_validation_diagnostic", _capture_validation)
    monkeypatch.setattr(
        pipeline,
        "write_occlusion_diagnostics",
        lambda **_: (tmp_path / "occ.json", tmp_path / "occ.csv"),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_overlay_validation_diagnostics",
        lambda **_: (tmp_path / "overlay.json", tmp_path / "overlay.csv"),
    )
    monkeypatch.setattr(
        pipeline,
        "_evaluate_overlay_validation_summary",
        lambda *_args, **_kwargs: {
            "visible": [object()],
            "hard_fail_visible": [],
            "hard_fail": False,
            "median_contact": None,
            "p90_contact": None,
            "hard_failure_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        pipeline.ResourceStore,
        "blender_artifact_dir_for",
        staticmethod(lambda _run_dir, name: tmp_path / name),
    )

    pipeline.compose_overlay_frames(
        run_dir=tmp_path,
        actor_name="Pedestrian01",
        road_labels=("road",),
        contact_ground_labels=None,
        occlusion_spec=None,
        shadow_spec=types.SimpleNamespace(enabled=False),
        grounding_diagnostics=[object()],
        original_frames_dir=original_dir,
        pedestrian_frames_dir=pedestrian_dir,
        pedestrian_depth_frames_dir=tmp_path / "ped_depth",
        shadow_frames_dir=None,
        output_dir=output_dir,
    )

    assert captured["image_shape"] == (405, 720)
    assert captured["overlay_shape"] == (405, 720)


def test_compose_overlay_frames_preserves_top_origin_when_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    background_path = tmp_path / "background.png"
    pedestrian_path = tmp_path / "pedestrian.png"
    original_dir = tmp_path / "original"
    pedestrian_dir = tmp_path / "pedestrian"
    output_dir = tmp_path / "output"
    original_dir.mkdir()
    pedestrian_dir.mkdir()
    output_dir.mkdir()
    background_path.write_bytes(b"")
    pedestrian_path.write_bytes(b"")

    fake_scene = types.SimpleNamespace(frame_set=lambda _frame: None)
    fake_view_layer = types.SimpleNamespace(update=lambda: None)
    monkeypatch.setattr(
        pipeline,
        "bpy",
        types.SimpleNamespace(
            context=types.SimpleNamespace(scene=fake_scene, view_layer=fake_view_layer),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_frame_index_map",
        lambda path: {0: background_path} if path == original_dir else {0: pedestrian_path},
    )
    monkeypatch.setattr(
        pipeline,
        "_load_overlay_validation_context",
        lambda _run_dir: (
            np.eye(3, dtype=np.float32),
            {0: np.eye(4, dtype=np.float32)},
            None,
        ),
    )
    monkeypatch.setattr(pipeline, "_overlay_validation_policy_for_run", lambda _run_dir: object())
    monkeypatch.setattr(pipeline, "_resolve_actor_root", lambda _name: object())
    monkeypatch.setattr(pipeline, "_find_actor_armature", lambda _name: object())
    monkeypatch.setattr(
        pipeline,
        "_build_support_point_lookup",
        lambda _diags: {0: np.zeros(3, dtype=np.float32)},
    )
    grounding_diag = types.SimpleNamespace(
        support_mode="trajectory_path",
        selected_support_foot="path",
        chosen_plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        chosen_plane_offset=0.0,
    )
    monkeypatch.setattr(
        pipeline,
        "_build_grounding_diagnostic_lookup",
        lambda _diags: {0: grounding_diag},
    )
    monkeypatch.setattr(
        pipeline,
        "_load_rgba_image",
        lambda _path: np.zeros((6, 4, 4), dtype=np.uint8),
    )
    monkeypatch.setattr(
        pipeline,
        "_project_overlay_support_point",
        lambda **_: (np.asarray([1.0, 1.0], dtype=np.float32), True, 1.0),
    )
    monkeypatch.setattr(pipeline, "_load_overlay_ground_mask", lambda **_: np.ones((6, 4), dtype=bool))
    overlay_top = np.zeros((6, 4, 3), dtype=np.uint8)
    overlay_top[0, :, :] = np.array([255, 32, 16], dtype=np.uint8)
    overlay_top[-1, :, :] = np.array([8, 64, 128], dtype=np.uint8)
    monkeypatch.setattr(
        pipeline,
        "compose_overlay_frame_with_occlusion",
        lambda **_: (
            overlay_top.copy(),
            np.ones((6, 4), dtype=bool),
            pipeline.OcclusionFrameDiagnostics(
                frame_index=0,
                pedestrian_pixels=24,
                visible_pixels=24,
                occluded_pixels=0,
                visible_ratio=1.0,
                min_scene_depth_m=1.0,
                max_scene_depth_m=1.0,
                min_ped_depth_m=1.0,
                max_ped_depth_m=1.0,
                median_depth_margin_m=0.1,
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_evaluate_feet_world",
        lambda *_: {"left": np.zeros(3, dtype=np.float32), "right": np.zeros(3, dtype=np.float32)},
    )
    monkeypatch.setattr(
        pipeline,
        "_project_overlay_feet",
        lambda **_: (
            np.asarray([1.0, 1.0], dtype=np.float32),
            True,
            np.asarray([2.0, 1.0], dtype=np.float32),
            True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_make_overlay_validation_diagnostic",
        lambda **kwargs: pipeline.OverlayValidationDiagnostic(
            frame_index=0,
            has_visible_pedestrian=True,
            internal_render_shape=(6, 4),
            overlay_shape=kwargs["overlay_shape"],
            lowest_alpha_u=0,
            lowest_alpha_v=0,
            lowest_alpha_row_coverage_px=4,
            touches_image_bottom=False,
            left_foot_projected_uv=None,
            right_foot_projected_uv=None,
            left_foot_visible_expected=True,
            right_foot_visible_expected=True,
            support_point_projected_uv=np.asarray([1.0, 1.0], dtype=np.float32),
            support_point_projected_visible=True,
            support_point_depth_m=1.0,
            scene_depth_at_support_px_m=1.0,
            support_point_occluded_by_scene=False,
            selected_foot_projected_uv=None,
            selected_foot_projected_visible=False,
            support_to_left_foot_px=None,
            support_to_right_foot_px=None,
            support_to_contact_foot_px=None,
            contact_foot_comparison_mode="trajectory_path",
            support_to_silhouette_bottom_px=None,
            support_to_selected_foot_px=None,
            support_patch_road_fraction=1.0,
            support_patch_nonroad_fraction=0.0,
            support_patch_size_px=1,
            road_region_validation_available=True,
            road_context_search_mode="direct_patch",
            validation_passed=True,
            failure_reason=None,
            support_mode="trajectory_path",
            selected_support_foot="path",
            warning_flags=tuple(),
            contact_validation_state="unverifiable",
            contact_validation_trusted=False,
            abort_relevant=False,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_occlusion_diagnostics",
        lambda **_: (tmp_path / "occ.json", tmp_path / "occ.csv"),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_overlay_validation_diagnostics",
        lambda **_: (tmp_path / "overlay.json", tmp_path / "overlay.csv"),
    )
    monkeypatch.setattr(
        pipeline,
        "_evaluate_overlay_validation_summary",
        lambda *_args, **_kwargs: {
            "visible": [object()],
            "hard_fail_visible": [],
            "hard_fail": False,
            "median_contact": None,
            "p90_contact": None,
            "hard_failure_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        pipeline.ResourceStore,
        "blender_artifact_dir_for",
        staticmethod(lambda _run_dir, name: tmp_path / name),
    )

    written: dict[str, np.ndarray] = {}

    def _fake_write(path: Path, rgba: np.ndarray) -> None:
        written[str(path)] = np.asarray(rgba, dtype=np.uint8).copy()

    monkeypatch.setattr(pipeline, "_write_rgba_image", _fake_write)

    pipeline.compose_overlay_frames(
        run_dir=tmp_path,
        actor_name="Pedestrian01",
        road_labels=("road",),
        contact_ground_labels=None,
        occlusion_spec=None,
        shadow_spec=types.SimpleNamespace(enabled=False),
        grounding_diagnostics=[object()],
        original_frames_dir=original_dir,
        pedestrian_frames_dir=pedestrian_dir,
        pedestrian_depth_frames_dir=tmp_path / "ped_depth",
        shadow_frames_dir=None,
        output_dir=output_dir,
    )

    saved = written[str(output_dir / "000000.png")]
    np.testing.assert_array_equal(saved[0, :, :3], overlay_top[0, :, :])
    np.testing.assert_array_equal(saved[-1, :, :3], overlay_top[-1, :, :])


def test_compose_overlay_frames_support_local_grid_marker_uses_top_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    background_path = tmp_path / "background.png"
    pedestrian_path = tmp_path / "pedestrian.png"
    original_dir = tmp_path / "original"
    pedestrian_dir = tmp_path / "pedestrian"
    output_dir = tmp_path / "output"
    support_local_grid_dir = tmp_path / "support_local_grid"
    original_dir.mkdir()
    pedestrian_dir.mkdir()
    output_dir.mkdir()
    support_local_grid_dir.mkdir()
    background_path.write_bytes(b"")
    pedestrian_path.write_bytes(b"")

    fake_scene = types.SimpleNamespace(frame_set=lambda _frame: None)
    fake_view_layer = types.SimpleNamespace(update=lambda: None)
    monkeypatch.setattr(
        pipeline,
        "bpy",
        types.SimpleNamespace(
            context=types.SimpleNamespace(scene=fake_scene, view_layer=fake_view_layer),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_frame_index_map",
        lambda path: {0: background_path} if path == original_dir else {0: pedestrian_path},
    )
    monkeypatch.setattr(
        pipeline,
        "_load_overlay_validation_context",
        lambda _run_dir: (
            np.eye(3, dtype=np.float32),
            {0: np.eye(4, dtype=np.float32)},
            None,
        ),
    )
    monkeypatch.setattr(pipeline, "_overlay_validation_policy_for_run", lambda _run_dir: object())
    monkeypatch.setattr(pipeline, "_resolve_actor_root", lambda _name: object())
    monkeypatch.setattr(pipeline, "_find_actor_armature", lambda _name: object())
    monkeypatch.setattr(
        pipeline,
        "_build_support_point_lookup",
        lambda _diags: {0: np.zeros(3, dtype=np.float32)},
    )
    grounding_diag = types.SimpleNamespace(
        support_mode="trajectory_path",
        selected_support_foot="path",
        chosen_plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        chosen_plane_offset=0.0,
        support_point_after=np.zeros(3, dtype=np.float32),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_grounding_diagnostic_lookup",
        lambda _diags: {0: grounding_diag},
    )
    monkeypatch.setattr(
        pipeline,
        "_load_rgba_image",
        lambda _path: np.zeros((6, 4, 4), dtype=np.uint8),
    )
    monkeypatch.setattr(
        pipeline,
        "_project_overlay_support_point",
        lambda **_: (np.asarray([1.0, 1.0], dtype=np.float32), True, 1.0),
    )
    monkeypatch.setattr(pipeline, "_load_overlay_ground_mask", lambda **_: np.ones((6, 4), dtype=bool))
    monkeypatch.setattr(
        pipeline,
        "compose_overlay_frame_with_occlusion",
        lambda **_: (
            np.zeros((6, 4, 3), dtype=np.uint8),
            np.ones((6, 4), dtype=bool),
            pipeline.OcclusionFrameDiagnostics(
                frame_index=0,
                pedestrian_pixels=24,
                visible_pixels=24,
                occluded_pixels=0,
                visible_ratio=1.0,
                min_scene_depth_m=1.0,
                max_scene_depth_m=1.0,
                min_ped_depth_m=1.0,
                max_ped_depth_m=1.0,
                median_depth_margin_m=0.1,
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_evaluate_feet_world",
        lambda *_: {"left": np.zeros(3, dtype=np.float32), "right": np.zeros(3, dtype=np.float32)},
    )
    monkeypatch.setattr(
        pipeline,
        "_project_overlay_feet",
        lambda **_: (
            np.asarray([1.0, 1.0], dtype=np.float32),
            True,
            np.asarray([2.0, 1.0], dtype=np.float32),
            True,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_render_overlay_support_local_grid",
        lambda **kwargs: np.asarray(kwargs["out_pixels"], dtype=np.float32).copy(),
    )
    monkeypatch.setattr(
        pipeline,
        "_make_overlay_validation_diagnostic",
        lambda **kwargs: pipeline.OverlayValidationDiagnostic(
            frame_index=0,
            has_visible_pedestrian=True,
            internal_render_shape=(6, 4),
            overlay_shape=kwargs["overlay_shape"],
            lowest_alpha_u=0,
            lowest_alpha_v=0,
            lowest_alpha_row_coverage_px=4,
            touches_image_bottom=False,
            left_foot_projected_uv=None,
            right_foot_projected_uv=None,
            left_foot_visible_expected=True,
            right_foot_visible_expected=True,
            support_point_projected_uv=np.asarray([1.0, 1.0], dtype=np.float32),
            support_point_projected_visible=True,
            support_point_depth_m=1.0,
            scene_depth_at_support_px_m=1.0,
            support_point_occluded_by_scene=False,
            selected_foot_projected_uv=None,
            selected_foot_projected_visible=False,
            support_to_left_foot_px=None,
            support_to_right_foot_px=None,
            support_to_contact_foot_px=None,
            contact_foot_comparison_mode="trajectory_path",
            support_to_silhouette_bottom_px=None,
            support_to_selected_foot_px=None,
            support_patch_road_fraction=1.0,
            support_patch_nonroad_fraction=0.0,
            support_patch_size_px=1,
            road_region_validation_available=True,
            road_context_search_mode="direct_patch",
            validation_passed=True,
            failure_reason=None,
            support_mode="trajectory_path",
            selected_support_foot="path",
            warning_flags=tuple(),
            contact_validation_state="unverifiable",
            contact_validation_trusted=False,
            abort_relevant=False,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "write_occlusion_diagnostics",
        lambda **_: (tmp_path / "occ.json", tmp_path / "occ.csv"),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_overlay_validation_diagnostics",
        lambda **_: (tmp_path / "overlay.json", tmp_path / "overlay.csv"),
    )
    monkeypatch.setattr(
        pipeline,
        "_evaluate_overlay_validation_summary",
        lambda *_args, **_kwargs: {
            "visible": [object()],
            "hard_fail_visible": [],
            "hard_fail": False,
            "median_contact": None,
            "p90_contact": None,
            "hard_failure_ratio": 0.0,
        },
    )
    monkeypatch.setattr(
        pipeline.ResourceStore,
        "blender_artifact_dir_for",
        staticmethod(lambda _run_dir, name: tmp_path / name),
    )

    written: dict[str, np.ndarray] = {}

    def _fake_write(path: Path, rgba: np.ndarray) -> None:
        written[str(path)] = np.asarray(rgba, dtype=np.uint8).copy()

    monkeypatch.setattr(pipeline, "_write_rgba_image", _fake_write)

    pipeline.compose_overlay_frames(
        run_dir=tmp_path,
        actor_name="Pedestrian01",
        road_labels=("road",),
        contact_ground_labels=None,
        occlusion_spec=None,
        shadow_spec=types.SimpleNamespace(enabled=False),
        grounding_diagnostics=[object()],
        original_frames_dir=original_dir,
        pedestrian_frames_dir=pedestrian_dir,
        pedestrian_depth_frames_dir=tmp_path / "ped_depth",
        shadow_frames_dir=None,
        output_dir=output_dir,
        support_local_grid_output_dir=support_local_grid_dir,
    )

    saved = written[str(support_local_grid_dir / "000000.png")]
    np.testing.assert_array_equal(saved[0:3, 0:3, 0], np.full((3, 3), 255, dtype=np.uint8))
    np.testing.assert_array_equal(saved[3:6, 0:3, 0], np.zeros((3, 3), dtype=np.uint8))


def test_render_overlay_support_local_grid_preserves_top_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    out_pixels = np.zeros((6, 4, 4), dtype=np.float32)
    out_pixels[0, :, :3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out_pixels[-1, :, :3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    out_pixels[:, :, 3] = 1.0
    road_mask = np.ones((6, 4), dtype=bool)
    grounding_diag = types.SimpleNamespace(
        support_point_after=np.zeros(3, dtype=np.float32),
        chosen_plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        chosen_plane_offset=0.0,
    )

    monkeypatch.setattr(
        pipeline,
        "render_plane_grid_layer",
        lambda *args, **kwargs: np.zeros((6, 4, 3), dtype=np.uint8),
    )

    def _fake_composite(base_bgr, _grid_layer, _mask):
        out = np.asarray(base_bgr, dtype=np.uint8).copy()
        out[0, :, :] = np.array([0, 255, 0], dtype=np.uint8)
        return out

    monkeypatch.setattr(pipeline, "composite_grid_with_mask", _fake_composite)

    rendered = pipeline._render_overlay_support_local_grid(
        run_dir=Path("."),
        frame_idx=0,
        out_pixels=out_pixels,
        grounding_diag=grounding_diag,
        intrinsics_k=np.eye(3, dtype=np.float32),
        frame_to_c2w={0: np.eye(4, dtype=np.float32)},
        road_labels=("road",),
        pedestrian_visible_mask_top=np.zeros((6, 4), dtype=bool),
        road_mask=road_mask,
    )

    top_rgb = np.rint(rendered[0, :, :3] * 255.0).astype(np.uint8)
    bottom_rgb = np.rint(rendered[-1, :, :3] * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(top_rgb, np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (4, 1)))
    np.testing.assert_array_equal(bottom_rgb, np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (4, 1)))


def test_resolve_heading_axis_in_object_local_xy_projects_world_heading_into_object_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    axis_xy = pipeline._resolve_heading_axis_in_object_local_xy(
        np.eye(4, dtype=np.float32),
        (0.0, 1.0),
    )

    np.testing.assert_allclose(axis_xy, np.array([0.0, 1.0], dtype=np.float32))


def test_correct_looping_hips_translation_preserves_lateral_sway_without_cycle_snap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    corrected_a, transferred_a, in_cycle_forward_a, lateral_a = (
        pipeline._correct_looping_hips_translation(
            raw_local_xy=(1.8, 0.2),
            base_local_xy=(0.0, 0.0),
            source_forward_axis_xy=(1.0, 0.0),
            target_forward_axis_xy=(1.0, 0.0),
            completed_cycles=0,
            cycle_forward_distance_local_m=2.0,
        )
    )
    corrected_b, transferred_b, in_cycle_forward_b, lateral_b = (
        pipeline._correct_looping_hips_translation(
            raw_local_xy=(0.1, 0.25),
            base_local_xy=(0.0, 0.0),
            source_forward_axis_xy=(1.0, 0.0),
            target_forward_axis_xy=(1.0, 0.0),
            completed_cycles=1,
            cycle_forward_distance_local_m=2.0,
        )
    )

    np.testing.assert_allclose(corrected_a, np.array([0.0, 0.2], dtype=np.float32))
    np.testing.assert_allclose(corrected_b, np.array([0.0, 0.25], dtype=np.float32))
    assert transferred_a == pytest.approx(1.8)
    assert transferred_b == pytest.approx(2.1)
    assert in_cycle_forward_a == pytest.approx(1.8)
    assert in_cycle_forward_b == pytest.approx(0.1)
    assert lateral_a == pytest.approx(0.2)
    assert lateral_b == pytest.approx(0.25)


def test_stabilize_looping_pelvis_world_translation_strips_only_forward_world_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    desired_world, forward_correction_world, residual_world = (
        pipeline._stabilize_looping_pelvis_world_translation(
            raw_world_translation=(1.75, -0.3, 1.2),
            base_world_translation=(0.0, 0.0, 1.0),
            source_anchor_world_translation=(0.0, 0.0, 0.0),
            target_anchor_world_translation=(10.0, 20.0, 0.0),
            locomotion_axis_xy=(1.0, 0.0),
        )
    )

    np.testing.assert_allclose(
        desired_world,
        np.array([10.0, 19.7, 1.2], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forward_correction_world,
        np.array([1.75, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        residual_world,
        np.array([0.0, -0.3, 0.2], dtype=np.float32),
        atol=1e-6,
    )


def test_stabilize_looping_pelvis_world_translation_without_axis_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    desired_world, forward_correction_world, residual_world = (
        pipeline._stabilize_looping_pelvis_world_translation(
            raw_world_translation=(0.2, -0.4, 1.15),
            base_world_translation=(0.0, 0.0, 1.0),
            source_anchor_world_translation=(0.0, 0.0, 0.0),
            target_anchor_world_translation=(10.0, 20.0, 0.0),
            locomotion_axis_xy=None,
        )
    )

    np.testing.assert_allclose(
        desired_world,
        np.array([10.2, 19.6, 1.15], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forward_correction_world,
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_allclose(
        residual_world,
        np.array([0.2, -0.4, 0.15], dtype=np.float32),
        atol=1e-6,
    )


def test_validate_authored_root_path_starts_at_spawn_accepts_exact_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    delta_m = pipeline._validate_authored_root_path_starts_at_spawn(
        path_start_world=(1.0, 2.0, 3.0),
        resolved_spawn_world=(1.0, 2.0, 3.0),
    )

    assert delta_m == pytest.approx(0.0)


def test_validate_authored_root_path_starts_at_spawn_rejects_rebased_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    with pytest.raises(RuntimeError, match="does not start at the resolved spawn"):
        pipeline._validate_authored_root_path_starts_at_spawn(
            path_start_world=(0.0, 0.0, 0.0),
            resolved_spawn_world=(5.0, 6.0, 0.0),
            tolerance_m=1e-4,
        )


def test_validate_actor_hierarchy_alignment_state_accepts_aligned_root_armature_and_bones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    report = pipeline._validate_actor_hierarchy_alignment_state(
        root_world=(-26.4, 17.3, -0.7),
        armature_world=(-26.4, 17.3, -0.7),
        key_bone_world_by_name={
            "mixamorig:Hips": (-26.38, 17.29, 0.45),
            "mixamorig:Spine": (-26.39, 17.30, 0.60),
        },
    )

    assert report["root_armature_delta_m"] == pytest.approx(0.0)
    assert report["bone_offsets"]["mixamorig:Hips"]["horizontal_offset_m"] < 0.1
    assert report["bone_offsets"]["mixamorig:Hips"]["vertical_offset_m"] > 1.0


def test_validate_actor_hierarchy_alignment_state_rejects_bone_rebased_far_from_armature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    with pytest.raises(RuntimeError, match="implausibly far"):
        pipeline._validate_actor_hierarchy_alignment_state(
            root_world=(-26.4, 17.3, -0.7),
            armature_world=(-26.4, 17.3, -0.7),
            key_bone_world_by_name={
                "mixamorig:Hips": (0.0, 0.0, 1.2),
            },
        )


def test_sanitize_corrupted_subject_rgba_frames_clears_off_camera_full_frame_outlier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    opaque = np.zeros((4, 6, 4), dtype=np.uint8)
    opaque[:, :, :3] = 50
    opaque[:, :, 3] = 255
    empty = np.zeros((4, 6, 4), dtype=np.uint8)
    pipeline._write_rgba_image(frames_dir / "frame_0000.png", opaque)
    pipeline._write_rgba_image(frames_dir / "frame_0001.png", empty)
    pipeline._write_rgba_image(frames_dir / "frame_0002.png", empty)

    diagnostics = [
        types.SimpleNamespace(frame_index=0, visibility_culled=True),
        types.SimpleNamespace(frame_index=1, visibility_culled=True),
        types.SimpleNamespace(frame_index=2, visibility_culled=True),
    ]

    result = pipeline._sanitize_corrupted_subject_rgba_frames(
        run_dir=tmp_path,
        pedestrian_frames_dir=frames_dir,
        grounding_diagnostics=diagnostics,
    )

    assert result["applied"] is True
    assert result["sanitized_frames"] == [0]
    repaired = pipeline._load_rgba_image(frames_dir / "frame_0000.png")
    assert int(np.count_nonzero(repaired[:, :, 3])) == 0


def test_scene_spec_from_profile_uses_saved_profile_snapshot_sampling_fps(
    tmp_path: Path,
) -> None:
    config = importlib.import_module("pemoin.visualization.blender_scene.config")

    run_dir = tmp_path / "run"
    (run_dir / "standard").mkdir(parents=True, exist_ok=True)
    trajectory_dir = run_dir / "standard" / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trajectory_dir / "poses.npz",
        camera_to_world=np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0),
        frame_indices=np.array([0, 1], dtype=np.int32),
    )

    character_fbx = tmp_path / "character.fbx"
    animation_fbx = tmp_path / "animation.fbx"
    character_fbx.write_text("", encoding="utf-8")
    animation_fbx.write_text("", encoding="utf-8")

    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "demo": {
                        "runtime": {"settings": {"blender_scene": {"enabled": True}}},
                        "mixamo": {
                            "character_fbx_path": str(character_fbx),
                            "animation_fbx_path": str(animation_fbx),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "standard" / "profile.json").write_text(
        json.dumps(
            {
                "profile": "demo",
                "frame_provider": {"settings": {"resolved_sampling_fps": 12.5}},
                "runtime": {"settings": {"blender_scene": {"enabled": True}}},
                "mixamo": {
                    "character_fbx_path": str(character_fbx),
                    "animation_fbx_path": str(animation_fbx),
                },
            }
        ),
        encoding="utf-8",
    )

    spec = config._scene_spec_from_profile(
        run_dir=run_dir,
        trajectory_path=None,
        output_path=None,
        config_path=config_path,
        profile_name="demo",
    )

    assert spec.sampling_fps == pytest.approx(12.5)
    assert spec.mixamo_scene_fps == pytest.approx(12.5)
    assert spec.mixamo_export_fps == pytest.approx(30.0)
    assert spec.pedestrian_motion_policy == "auto"
    assert spec.trajectory_path == trajectory_dir / "poses.npz"


def test_blender_scene_script_wrapper_delegates_to_run_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    fake_app = types.ModuleType("pemoin.visualization.blender_scene.app")
    fake_app.run_scene = lambda argv: calls.append(list(argv))
    fake_config = types.ModuleType("pemoin.visualization.blender_scene.config")
    fake_config.parse_args = lambda argv: argv
    fake_logging = types.ModuleType("pemoin.visualization.blender_scene.logging")
    fake_logging.log_error = lambda message: None

    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    monkeypatch.setitem(sys.modules, fake_app.__name__, fake_app)
    monkeypatch.setitem(sys.modules, fake_config.__name__, fake_config)
    monkeypatch.setitem(sys.modules, fake_logging.__name__, fake_logging)

    spec = importlib.util.spec_from_file_location("test_blender_scene_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main(["--run-dir", "/tmp/run"])

    assert calls == [["--run-dir", "/tmp/run"]]


def test_blender_scene_script_wrapper_exits_with_code_one_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    fake_app = types.ModuleType("pemoin.visualization.blender_scene.app")

    def _raise(_: list[str]) -> None:
        raise RuntimeError("boom")

    fake_app.run_scene = _raise
    fake_config = types.ModuleType("pemoin.visualization.blender_scene.config")
    fake_config.parse_args = lambda argv: argv
    fake_logging = types.ModuleType("pemoin.visualization.blender_scene.logging")
    fake_logging.log_error = errors.append

    monkeypatch.setitem(sys.modules, "bpy", types.ModuleType("bpy"))
    monkeypatch.setitem(sys.modules, fake_app.__name__, fake_app)
    monkeypatch.setitem(sys.modules, fake_config.__name__, fake_config)
    monkeypatch.setitem(sys.modules, fake_logging.__name__, fake_logging)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--", "--run-dir", "/tmp/run"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")

    assert exc_info.value.code == 1
    assert errors == ["boom"]


def test_run_scene_from_spec_preserves_stage_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.app", raising=False
    )
    app = importlib.import_module("pemoin.visualization.blender_scene.app")
    specs = importlib.import_module("pemoin.visualization.blender_scene.specs")

    order: list[str] = []
    lighting_calls: list[dict[str, object]] = []

    monkeypatch.setattr(app, "clear_scene", lambda: order.append("clear_scene"))
    monkeypatch.setattr(
        app,
        "ensure_collection",
        lambda name: order.append(f"ensure_collection:{name}") or object(),
    )
    monkeypatch.setattr(
        app,
        "_setup_camera_and_trajectory",
        lambda spec, traj_collection: (
            order.append("_setup_camera_and_trajectory") or np.zeros((1, 4, 4), dtype=np.float32),
            np.array([0], dtype=np.int32),
            app.CameraSetupResult(
                intrinsics_matrix=np.eye(3, dtype=np.float32),
                width=16,
                height=16,
                intrinsics_metadata={},
                parity_solution=types.SimpleNamespace(
                    sensor_fit="AUTO",
                    focal_residual_px=0.0,
                    principal_point_residual_px=0.0,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        app, "_configure_scene_timing", lambda spec: order.append("_configure_scene_timing")
    )
    monkeypatch.setattr(
        app,
        "configure_render_engine",
        lambda lighting: order.append("configure_render_engine"),
    )
    monkeypatch.setattr(
        app,
        "configure_scene_lighting",
        lambda lighting, run_dir=None, anchor_world=None: (
            lighting_calls.append({"run_dir": run_dir, "anchor_world": anchor_world}),
            order.append("configure_scene_lighting"),
        )[-1],
    )
    monkeypatch.setattr(
        app,
        "_resolve_spawn",
        lambda spec, c2w: order.append("_resolve_spawn")
        or app.SpawnResolution(
            resolved_spawn_world_arr=np.zeros(3, dtype=np.float32),
            trajectory_anchor_world_arr=np.zeros(3, dtype=np.float32),
            motion_forward_world_arr=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            base_heading_world_deg=0.0,
            spawn_min_distance_m=1.0,
        ),
    )
    monkeypatch.setattr(
        app,
        "insert_mixamo_character",
        lambda spec, c2w_matrices, frame_indices, spawn_world, trajectory_anchor_world, intended_forward_world: order.append(
            "insert_mixamo_character"
        )
        or {"resolved_root_yaw_world_deg": 0.0},
    )
    monkeypatch.setattr(
        app,
        "viz_road_planes",
        lambda **kwargs: order.append("viz_road_planes")
        or types.SimpleNamespace(global_planes={}),
    )
    monkeypatch.setattr(
        app,
        "apply_road_support_to_inserted_pedestrian",
        lambda **kwargs: order.append("apply_road_support_to_inserted_pedestrian") or [],
    )
    monkeypatch.setattr(
        app,
        "bind_dynamic_subject_lights",
        lambda **kwargs: order.append("bind_dynamic_subject_lights") or [],
    )
    monkeypatch.setattr(
        app,
        "_write_dynamic_lighting_anchor_diagnostics",
        lambda **kwargs: order.append("_write_dynamic_lighting_anchor_diagnostics")
        or (tmp_path / "lighting_anchor.json"),
    )
    monkeypatch.setattr(
        app,
        "_write_grounding_diagnostics",
        lambda **kwargs: order.append("_write_grounding_diagnostics")
        or (tmp_path / "grounding.json", tmp_path / "grounding.csv"),
    )
    monkeypatch.setattr(
        app,
        "_write_support_surface_diagnostics",
        lambda **kwargs: order.append("_write_support_surface_diagnostics")
        or (tmp_path / "support.json", tmp_path / "support.csv"),
    )
    monkeypatch.setattr(
        app,
        "_write_trajectory_support_segments",
        lambda **kwargs: order.append("_write_trajectory_support_segments")
        or (tmp_path / "trajectory_support_segments.json"),
    )
    monkeypatch.setattr(
        app,
        "_write_trajectory_height_profile",
        lambda **kwargs: order.append("_write_trajectory_height_profile")
        or (tmp_path / "trajectory_height_profile.csv"),
    )
    monkeypatch.setattr(
        app,
        "_write_road_surface_summary",
        lambda **kwargs: order.append("_write_road_surface_summary"),
    )
    monkeypatch.setattr(
        app,
        "_raise_for_grounding_failures",
        lambda **kwargs: order.append("_raise_for_grounding_failures"),
    )
    monkeypatch.setattr(
        app,
        "_render_outputs",
        lambda spec, camera_setup: order.append("_render_outputs")
        or app.RenderOutputPaths(
            pedestrian_frames_dir=tmp_path / "pedestrian_frames",
            pedestrian_depth_frames_dir=tmp_path / "pedestrian_depth_frames",
            overlay_frames_dir=tmp_path / "overlayed_frames",
            overlay_support_local_grid_dir=tmp_path / "overlayed_frames_support_local_grid",
            occlusion_masks_dir=tmp_path / "occlusion_masks",
            occlusion_debug_dir=tmp_path / "occlusion_debug",
        ),
    )
    monkeypatch.setattr(
        app,
        "compose_overlay_frames",
        lambda **kwargs: order.append("compose_overlay_frames"),
    )
    monkeypatch.setattr(
        app,
        "save_blend",
        lambda path: order.append("save_blend"),
    )

    spec = specs.SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=tmp_path / "scene.blend",
        cube_size=0.1,
        collection_name="TrajectoryDebug",
    )

    app.run_scene_from_spec(spec)

    assert order == [
        "clear_scene",
        "ensure_collection:TrajectoryDebug",
        "ensure_collection:RoadPlanesGlobal",
        "_setup_camera_and_trajectory",
        "_configure_scene_timing",
        "configure_render_engine",
        "_resolve_spawn",
        "configure_scene_lighting",
        "insert_mixamo_character",
        "viz_road_planes",
        "apply_road_support_to_inserted_pedestrian",
        "bind_dynamic_subject_lights",
        "_write_grounding_diagnostics",
        "_write_support_surface_diagnostics",
        "_write_trajectory_support_segments",
        "_write_trajectory_height_profile",
        "_write_road_surface_summary",
        "_raise_for_grounding_failures",
        "_render_outputs",
        "compose_overlay_frames",
        "save_blend",
    ]
    assert lighting_calls == [
        {"run_dir": tmp_path, "anchor_world": (0.0, 0.0, 0.0)}
    ]


def test_bind_dynamic_subject_lights_follows_grounded_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    class _FakeMatrix:
        def __init__(self, translation: tuple[float, float, float]) -> None:
            self.translation = np.asarray(translation, dtype=np.float32)

    class _FakeRoot:
        def __init__(self, name: str, positions: dict[int, tuple[float, float, float]]) -> None:
            self.name = name
            self._positions = positions
            self._frame = 0

        def evaluated_get(self, _deps):
            return types.SimpleNamespace(
                matrix_world=_FakeMatrix(self._positions[self._frame])
            )

    class _FakeConstraint(types.SimpleNamespace):
        pass

    class _FakeConstraints(list):
        def new(self, *, type: str):
            constraint = _FakeConstraint(type=type, name="", target=None)
            self.append(constraint)
            return constraint

        def remove(self, constraint) -> None:
            super().remove(constraint)

    class _FakeLight(dict):
        def __init__(self, name: str, offset: tuple[float, float, float]) -> None:
            super().__init__()
            self.name = name
            self.animation_data = None
            self.location = (0.0, 0.0, 0.0)
            self.constraints = _FakeConstraints()
            self[pipeline._PEMOIN_LIGHTING_TAG] = True
            self[pipeline._PEMOIN_LIGHT_PLACEMENT_MODE] = "subject_anchor_relative"
            self[pipeline._PEMOIN_LIGHT_PLACEMENT_TARGET] = "subject_root_dynamic"
            self[pipeline._PEMOIN_LIGHT_RELATIVE_OFFSET] = list(offset)
            self.keyframes: list[tuple[str, int, tuple[float, float, float]]] = []

        def keyframe_insert(self, *, data_path: str, frame: int) -> None:
            self.keyframes.append((data_path, frame, tuple(float(v) for v in self.location)))

    class _FakeObjects(list):
        def get(self, name: str):
            for item in self:
                if getattr(item, "name", None) == name:
                    return item
            return None

    root_positions = {
        0: (-7.84, 64.76, 0.0),
        10: (-5.58, 64.70, 0.0),
        20: (-3.32, 64.64, 0.0),
    }
    root = _FakeRoot("Pedestrian01", root_positions)
    fill = _FakeLight("PEMOINFill1", (2.0, 6.0, 3.375))
    objects = _FakeObjects([root, fill])

    class _FakeScene:
        def __init__(self) -> None:
            self.frame_current = 20

        def frame_set(self, frame: int) -> None:
            self.frame_current = int(frame)
            root._frame = int(frame)

    scene = _FakeScene()
    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(objects=objects),
        context=types.SimpleNamespace(
            scene=scene,
            view_layer=types.SimpleNamespace(update=lambda: None),
            evaluated_depsgraph_get=lambda: object(),
        ),
    )
    monkeypatch.setattr(pipeline, "bpy", fake_bpy)

    diagnostics = pipeline.bind_dynamic_subject_lights(
        actor_name="Pedestrian01",
        frame_indices=[0, 10, 20],
        binding_mode="copy_location_constraint",
    )

    assert len(diagnostics) == 1
    assert diagnostics[0]["placement_target"] == "subject_root_dynamic"
    assert diagnostics[0]["binding_mode"] == "copy_location_constraint"
    assert diagnostics[0]["actor_anchor_world_start"] == pytest.approx((-7.84, 64.76, 0.0))
    assert diagnostics[0]["actor_anchor_world_end"] == pytest.approx((-3.32, 64.64, 0.0))
    assert fill.keyframes == []
    assert len(fill.constraints) == 1
    assert fill.constraints[0].type == "COPY_LOCATION"
    assert fill.constraints[0].target is root
    assert fill.location == pytest.approx((2.0, 6.0, 3.375))
    assert scene.frame_current == 20


def test_light_spec_from_standardized_point_wrap_fill_uses_anchor_relative_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    class _FakeVector:
        def __init__(self, values) -> None:
            self._values = np.asarray(values, dtype=np.float32)

        def normalized(self):
            norm = float(np.linalg.norm(self._values))
            if norm <= 1e-8:
                values = self._values
            else:
                values = self._values / norm
            return _FakeNormalizedVector(values)

    class _FakeNormalizedVector:
        def __init__(self, values: np.ndarray) -> None:
            self._values = np.asarray(values, dtype=np.float32)

        def to_track_quat(self, *_args):
            return types.SimpleNamespace(to_euler=lambda: (0.0, 0.0, 0.0))

    monkeypatch.setattr(pipeline, "Vector", _FakeVector)

    light = types.SimpleNamespace(
        name="PEMOINWrapFill",
        kind="POINT",
        role="wrap_key_fill",
        strength=6.0,
        color=np.array([0.95, 0.97, 1.0], dtype=np.float32),
        casts_shadow=False,
        placement_mode="subject_anchor_relative",
        placement_target="subject_root_dynamic",
        direction_world=np.array([0.2, 0.8, -0.56], dtype=np.float32),
        rotation_world=None,
        angular_size_deg=None,
        area_size=None,
        location_world=np.array([2.5, -1.5, 3.0], dtype=np.float32),
    )

    spec = pipeline._light_spec_from_standardized_light(
        light,
        anchor_world=(10.0, 20.0, 0.5),
    )

    assert spec.kind == "POINT"
    assert spec.role == "wrap_key_fill"
    assert spec.casts_shadow is False
    assert spec.location == pytest.approx((12.5, 18.5, 3.5))
    assert spec.relative_location == pytest.approx((2.5, -1.5, 3.0))
    assert spec.placement_mode == "subject_anchor_relative"
    assert spec.placement_target == "subject_root_dynamic"
    assert spec.area_size is None


def test_light_spec_from_standardized_light_accepts_location_only_point_light(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    light = types.SimpleNamespace(
        name="CARLASceneLight0000",
        kind="POINT",
        role="street_fill",
        strength=1.0,
        color=np.array([1.0, 0.9, 0.8], dtype=np.float32),
        casts_shadow=False,
        placement_mode="world_absolute",
        placement_target="world",
        direction_world=None,
        rotation_world=None,
        angular_size_deg=None,
        area_size=None,
        location_world=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        diagnostics={},
    )

    spec = pipeline._light_spec_from_standardized_light(light)

    assert spec.kind == "POINT"
    assert spec.location == pytest.approx((1.0, 2.0, 3.0))
    assert spec.rotation_euler_deg == pytest.approx((0.0, 0.0, 0.0))


def test_create_light_scales_wrap_subject_point_energy_by_offset_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    created_lights: list[types.SimpleNamespace] = []
    linked_objects: list[object] = []

    class _FakeObjectsFactory:
        @staticmethod
        def new(name: str, light_data):
            obj = types.SimpleNamespace(
                name=name,
                data=light_data,
                rotation_euler=None,
                location=None,
                _props={},
            )
            obj.__setitem__ = lambda key, value, store=obj._props: store.__setitem__(key, value)
            obj.__getitem__ = lambda key, store=obj._props: store.__getitem__(key)
            return obj

    class _FakeLightObject:
        def __init__(self, name: str, light_type: str) -> None:
            self.name = name
            self.type = light_type
            self.energy = 0.0
            self.color = None
            self.use_shadow = False

    class _FakeLightFactory:
        @staticmethod
        def new(name: str, type: str):
            light = _FakeLightObject(name, type)
            created_lights.append(light)
            return light

    class _FakeObject:
        def __init__(self, name: str, light_data) -> None:
            self.name = name
            self.data = light_data
            self.rotation_euler = None
            self.location = None
            self._props: dict[str, object] = {}

        def __setitem__(self, key: str, value: object) -> None:
            self._props[key] = value

        def __getitem__(self, key: str) -> object:
            return self._props[key]

    class _FakeObjects:
        @staticmethod
        def new(name: str, light_data):
            return _FakeObject(name, light_data)

    fake_collection = types.SimpleNamespace(
        objects=types.SimpleNamespace(link=lambda obj: linked_objects.append(obj))
    )
    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(lights=_FakeLightFactory(), objects=_FakeObjects()),
    )
    monkeypatch.setattr(pipeline, "bpy", fake_bpy)
    monkeypatch.setattr(pipeline, "ensure_collection", lambda name: fake_collection)

    spec = pipeline.LightSpec(
        name="PEMOINFill1",
        kind="POINT",
        energy=5.0,
        rotation_euler_deg=(0.0, 0.0, 0.0),
        color=(1.0, 1.0, 1.0),
        role="wrap_key_fill",
        casts_shadow=False,
        location=(2.0, 0.0, 0.0),
        relative_location=(2.0, 0.0, 0.0),
        transport_mode="wrap_subject_fill",
    )

    light_obj = pipeline.create_light(spec)

    assert len(created_lights) == 1
    expected = 5.0 * (2.0 ** 2) * (0.08 * (1.0 - 0.35 * 0.35)) * 2.0
    assert created_lights[0].energy == pytest.approx(expected)
    assert light_obj._props[pipeline._PEMOIN_LIGHT_SOURCE_ENERGY] == pytest.approx(5.0)
    assert light_obj._props[pipeline._PEMOIN_LIGHT_REALIZED_ENERGY] == pytest.approx(expected)
    assert light_obj._props[pipeline._PEMOIN_LIGHT_TRANSPORT_MODE] == "wrap_subject_fill"
    assert linked_objects == [light_obj]


def test_create_light_uses_configured_wrap_subject_fill_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    created_lights: list[types.SimpleNamespace] = []

    class _FakeLightData(types.SimpleNamespace):
        pass

    class _FakeLightFactory:
        @staticmethod
        def new(name: str, type: str):
            light = _FakeLightData(
                name=name,
                type=type,
                energy=None,
                color=None,
                angle=None,
                shape=None,
                size=None,
                size_y=None,
                use_shadow=None,
            )
            created_lights.append(light)
            return light

    class _FakeObject:
        def __init__(self, name: str, data) -> None:
            self.name = name
            self.data = data
            self.rotation_euler = None
            self.location = None
            self._props: dict[str, object] = {}

        def __setitem__(self, key: str, value: object) -> None:
            self._props[key] = value

        def __getitem__(self, key: str) -> object:
            return self._props[key]

    class _FakeObjects:
        @staticmethod
        def new(name: str, light_data):
            return _FakeObject(name, light_data)

    fake_collection = types.SimpleNamespace(
        objects=types.SimpleNamespace(link=lambda obj: None)
    )
    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(lights=_FakeLightFactory(), objects=_FakeObjects()),
    )
    monkeypatch.setattr(pipeline, "bpy", fake_bpy)
    monkeypatch.setattr(pipeline, "ensure_collection", lambda name: fake_collection)

    spec = pipeline.LightSpec(
        name="PEMOINFill2",
        kind="POINT",
        energy=5.0,
        rotation_euler_deg=(0.0, 0.0, 0.0),
        color=(1.0, 1.0, 1.0),
        role="counter_wrap_fill",
        casts_shadow=False,
        location=(2.0, 0.0, 0.0),
        relative_location=(2.0, 0.0, 0.0),
        transport_mode="wrap_subject_fill",
    )
    wrap_subject_fill = pipeline.WrapSubjectFillSpec(
        global_strength_scale=2.5,
        wrap_key_role_scale=0.1,
        counter_wrap_role_scale=0.05,
        sky_fill_role_scale=0.03,
        counter_side_lift_bias=0.8,
        sky_softness_bias=0.5,
        direct_preservation_bias=0.25,
        raw_exposure_trim=1.0,
    )

    pipeline.create_light(spec, wrap_subject_fill=wrap_subject_fill)

    expected = 5.0 * (2.0 ** 2) * (0.05 * (1.0 + 0.75 * 0.8)) * 2.5
    assert created_lights[0].energy == pytest.approx(expected)


def test_normalize_pedestrian_rgba_sequence_unpremultiplies_rendered_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    pedestrian_dir = tmp_path / "pedestrian_frames"
    pedestrian_dir.mkdir(parents=True, exist_ok=True)
    (pedestrian_dir / "frame_0000.png").write_bytes(b"x")

    premul = np.zeros((16, 16, 4), dtype=np.uint8)
    premul[2:14, 2:14, 3] = 128
    premul[2:14, 2:14, :3] = 40
    written: dict[str, np.ndarray] = {}

    def _fake_load(path: Path) -> np.ndarray:
        if str(path) in written:
            return written[str(path)].copy()
        return premul.copy()

    def _fake_write(path: Path, rgba: np.ndarray) -> None:
        written[str(path)] = np.asarray(rgba, dtype=np.uint8).copy()

    monkeypatch.setattr(pipeline, "_load_rgba_image", _fake_load)
    monkeypatch.setattr(pipeline, "_write_rgba_image", _fake_write)

    diagnostics = pipeline._normalize_pedestrian_rgba_sequence_to_straight_alpha(
        run_dir=tmp_path,
        pedestrian_frames_dir=pedestrian_dir,
    )

    corrected = written[str(pedestrian_dir / "frame_0000.png")]
    assert diagnostics["normalization_applied"] is True
    assert diagnostics["normalized_frame_count"] == 1
    assert corrected[4, 4, 0] == pytest.approx(80, abs=1)
    assert corrected[4, 4, 3] == 128
    assert Path(diagnostics["diagnostics_path"]).exists()


def test_relink_and_validate_mixamo_materials_relinks_missing_texture_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    asset_root = tmp_path / "mixamo_assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    texture_path = asset_root / "Ch33_1001_Diffuse.png"
    texture_path.write_bytes(b"png")

    class _FakeImage:
        def __init__(self) -> None:
            self.name = "Ch33_1001_Diffuse.png"
            self.filepath = "/missing/export/tmp/Ch33_1001_Diffuse.png"
            self.filepath_raw = self.filepath
            self.packed_file = None
            self.reloaded = False

        def reload(self) -> None:
            self.reloaded = True

    image = _FakeImage()
    node = types.SimpleNamespace(type="TEX_IMAGE", name="Image Texture", image=image)
    material = types.SimpleNamespace(
        name="Ch33_body",
        use_nodes=True,
        node_tree=types.SimpleNamespace(nodes=[node]),
    )
    mesh = types.SimpleNamespace(
        name="Ch33_Body",
        children=[],
        data=types.SimpleNamespace(materials=[material]),
    )
    fake_bpy = types.SimpleNamespace(path=types.SimpleNamespace(abspath=lambda path: path))
    monkeypatch.setattr(pipeline, "bpy", fake_bpy)

    diagnostics = pipeline._relink_and_validate_mixamo_materials(
        imported_objects=[mesh],
        asset_root=asset_root,
        run_dir=tmp_path,
    )

    assert image.filepath == str(texture_path.resolve())
    assert image.filepath_raw == str(texture_path.resolve())
    assert image.reloaded is True
    assert diagnostics["unresolved_entry_count"] == 0
    assert Path(diagnostics["diagnostics_path"]).exists()


def test_relink_and_validate_mixamo_materials_accepts_packed_embedded_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    asset_root = tmp_path / "mixamo_assets"
    asset_root.mkdir(parents=True, exist_ok=True)

    class _FakeImage:
        def __init__(self) -> None:
            self.name = "Ch33_1001_Diffuse.png"
            self.filepath = "/missing/export/tmp/Ch33_1001_Diffuse.png"
            self.filepath_raw = self.filepath
            self.packed_file = object()

    image = _FakeImage()
    node = types.SimpleNamespace(type="TEX_IMAGE", name="Image Texture", image=image)
    material = types.SimpleNamespace(
        name="Ch33_body",
        use_nodes=True,
        node_tree=types.SimpleNamespace(nodes=[node]),
    )
    mesh = types.SimpleNamespace(
        name="Ch33_Body",
        children=[],
        data=types.SimpleNamespace(materials=[material]),
    )
    fake_bpy = types.SimpleNamespace(path=types.SimpleNamespace(abspath=lambda path: path))
    monkeypatch.setattr(pipeline, "bpy", fake_bpy)

    diagnostics = pipeline._relink_and_validate_mixamo_materials(
        imported_objects=[mesh],
        asset_root=asset_root,
        run_dir=tmp_path,
    )

    assert diagnostics["unresolved_entry_count"] == 0
    assert diagnostics["entries"][0]["status"] == "packed_embedded"


def test_calibrate_raw_subject_exposure_brightens_pedestrian_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    original_dir = tmp_path / "standard" / "frames"
    pedestrian_dir = tmp_path / "pedestrian_frames"
    original_dir.mkdir(parents=True, exist_ok=True)
    pedestrian_dir.mkdir(parents=True, exist_ok=True)
    (original_dir / "000000.png").write_bytes(b"x")
    (pedestrian_dir / "000000.png").write_bytes(b"x")

    original = np.full((128, 128, 4), 85, dtype=np.uint8)
    original[:, :, 3] = 255
    pedestrian = np.zeros((128, 128, 4), dtype=np.uint8)
    pedestrian[40:88, 48:80, :3] = 60
    pedestrian[40:88, 48:80, 3] = 255
    written: dict[str, np.ndarray] = {}

    def _fake_load(path: Path) -> np.ndarray:
        if path.parent == original_dir:
            return original.copy()
        if path.parent == pedestrian_dir and str(path) in written:
            return written[str(path)].copy()
        return pedestrian.copy()

    def _fake_write(path: Path, rgba: np.ndarray) -> None:
        written[str(path)] = np.asarray(rgba, dtype=np.uint8).copy()

    monkeypatch.setattr(pipeline, "_load_rgba_image", _fake_load)
    monkeypatch.setattr(pipeline, "_write_rgba_image", _fake_write)

    settings = types.SimpleNamespace(
        enabled=True,
        target_match_strength=1.0,
        max_gain=3.0,
        validation_tolerance=0.25,
    )
    diagnostics = pipeline._calibrate_raw_subject_exposure(
        run_dir=tmp_path,
        original_frames_dir=original_dir,
        pedestrian_frames_dir=pedestrian_dir,
        settings=settings,
    )

    corrected = written[str(pedestrian_dir / "000000.png")]
    before_mean = float(pedestrian[40:88, 48:80, :3].mean())
    after_mean = float(corrected[40:88, 48:80, :3].mean())
    assert diagnostics["eligible_frame_count"] == 1
    assert diagnostics["applied_gain"] > 1.0
    assert diagnostics["applied_gain"] <= 1.15 + 1e-6
    assert after_mean > before_mean
    assert Path(diagnostics["diagnostics_path"]).exists()


def test_calibrate_raw_subject_exposure_applies_trim_after_auto_gain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    original_dir = tmp_path / "standard" / "frames"
    pedestrian_dir = tmp_path / "pedestrian_frames"
    original_dir.mkdir(parents=True, exist_ok=True)
    pedestrian_dir.mkdir(parents=True, exist_ok=True)
    (original_dir / "000000.png").write_bytes(b"x")
    (pedestrian_dir / "000000.png").write_bytes(b"x")

    original = np.full((128, 128, 4), 180, dtype=np.uint8)
    original[:, :, 3] = 255
    pedestrian = np.zeros((128, 128, 4), dtype=np.uint8)
    pedestrian[40:88, 48:80, :3] = 90
    pedestrian[40:88, 48:80, 3] = 255
    written: dict[str, np.ndarray] = {}

    def _fake_load(path: Path) -> np.ndarray:
        if path.parent == original_dir:
            return original.copy()
        if path.parent == pedestrian_dir and str(path) in written:
            return written[str(path)].copy()
        return pedestrian.copy()

    def _fake_write(path: Path, rgba: np.ndarray) -> None:
        written[str(path)] = np.asarray(rgba, dtype=np.uint8).copy()

    monkeypatch.setattr(pipeline, "_load_rgba_image", _fake_load)
    monkeypatch.setattr(pipeline, "_write_rgba_image", _fake_write)

    settings = types.SimpleNamespace(
        enabled=True,
        target_match_strength=1.0,
        max_gain=3.0,
        validation_tolerance=0.25,
    )
    diagnostics = pipeline._calibrate_raw_subject_exposure(
        run_dir=tmp_path,
        original_frames_dir=original_dir,
        pedestrian_frames_dir=pedestrian_dir,
        settings=settings,
        trim=1.1,
    )

    assert diagnostics["computed_gain"] == pytest.approx(1.15)
    assert diagnostics["applied_gain"] == pytest.approx(1.15)
    assert diagnostics["raw_exposure_trim"] == pytest.approx(1.1)


def test_calibrate_raw_subject_exposure_resizes_background_to_pedestrian_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    original_dir = tmp_path / "standard" / "frames"
    pedestrian_dir = tmp_path / "pedestrian_frames"
    original_dir.mkdir(parents=True, exist_ok=True)
    pedestrian_dir.mkdir(parents=True, exist_ok=True)
    (original_dir / "000000.png").write_bytes(b"x")
    (pedestrian_dir / "000000.png").write_bytes(b"x")

    original = np.full((405, 720, 4), 100, dtype=np.uint8)
    original[:, :, 3] = 255
    pedestrian = np.zeros((202, 360, 4), dtype=np.uint8)
    pedestrian[50:150, 130:230, :3] = 70
    pedestrian[50:150, 130:230, 3] = 255
    written: dict[str, np.ndarray] = {}

    def _fake_load(path: Path) -> np.ndarray:
        if path.parent == original_dir:
            return original.copy()
        if path.parent == pedestrian_dir and str(path) in written:
            return written[str(path)].copy()
        return pedestrian.copy()

    def _fake_write(path: Path, rgba: np.ndarray) -> None:
        written[str(path)] = np.asarray(rgba, dtype=np.uint8).copy()

    monkeypatch.setattr(pipeline, "_load_rgba_image", _fake_load)
    monkeypatch.setattr(pipeline, "_write_rgba_image", _fake_write)

    settings = types.SimpleNamespace(
        enabled=True,
        target_match_strength=1.0,
        max_gain=3.0,
        validation_tolerance=0.25,
    )
    diagnostics = pipeline._calibrate_raw_subject_exposure(
        run_dir=tmp_path,
        original_frames_dir=original_dir,
        pedestrian_frames_dir=pedestrian_dir,
        settings=settings,
    )

    assert diagnostics["eligible_frame_count"] == 1
    assert Path(diagnostics["diagnostics_path"]).exists()


def test_enforce_render_visibility_parity_rejects_projected_visible_empty_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    vis_dir = tmp_path / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "render_parity_diagnostics.json").write_text("{}", encoding="utf-8")

    frames = [
        pipeline.RenderVisibilityFrame(
            frame_index=0,
            rendered_visible=False,
            rendered_alpha_pixels=0,
            projected_visible=True,
            support_state="supported",
            visibility_contract_state="projected_visible",
        )
    ]

    with pytest.raises(ValueError, match="projected_visible_but_rendered_empty"):
        pipeline._enforce_render_visibility_parity(
            run_dir=tmp_path,
            frames=frames,
        )


def test_build_render_visibility_contract_reads_frame_prefixed_pngs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    frames_dir = tmp_path / "pedestrian_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "frame_0000.png").write_bytes(b"x")

    monkeypatch.setattr(
        pipeline,
        "_load_rgba_image",
        lambda _path: np.dstack(
            [
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.zeros((4, 4), dtype=np.uint8),
                np.full((4, 4), 255, dtype=np.uint8),
            ]
        ),
    )

    frames = pipeline._build_render_visibility_contract(
        frames_dir=frames_dir,
        grounding_diagnostics=[
            types.SimpleNamespace(
                frame_index=0,
                visibility_culled=False,
                support_state="supported",
                visibility_contract_state="projected_visible",
            )
        ],
    )

    assert len(frames) == 1
    assert frames[0].rendered_visible is True
    assert frames[0].rendered_alpha_pixels > 0


def test_classify_projected_actor_visibility_uses_upper_body_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    captured: dict[str, np.ndarray] = {}

    def _fake_project_world_to_image(
        points_world: np.ndarray,
        intrinsics_k: np.ndarray,
        *,
        camera_to_world_matrix: np.ndarray,
        camera_convention: str,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        del intrinsics_k, camera_to_world_matrix, camera_convention, image_shape
        points_world = np.asarray(points_world, dtype=np.float32)
        captured["points_world"] = points_world
        valid = points_world[:, 2] >= 1.0
        return np.zeros((points_world.shape[0], 2), dtype=np.float32), valid

    monkeypatch.setattr(pipeline, "project_world_to_image", _fake_project_world_to_image)
    mesh_obj = types.SimpleNamespace(
        type="MESH",
        hide_render=False,
        children=[],
        bound_box=[
            (-0.2, -0.2, -1.0),
            (-0.2, -0.2, 2.0),
            (-0.2, 0.2, -1.0),
            (-0.2, 0.2, 2.0),
            (0.2, -0.2, -1.0),
            (0.2, -0.2, 2.0),
            (0.2, 0.2, -1.0),
            (0.2, 0.2, 2.0),
        ],
        matrix_world=np.eye(4, dtype=np.float32),
    )
    mesh_obj.evaluated_get = lambda _deps: mesh_obj
    actor_root = types.SimpleNamespace(name="Pedestrian01", children=[mesh_obj])

    visible, state = pipeline._classify_projected_actor_visibility(
        frame_idx=0,
        intrinsics_k=np.eye(3, dtype=np.float32),
        frame_to_c2w={0: np.eye(4, dtype=np.float32)},
        actor_root=actor_root,
        depsgraph=object(),
        image_shape=(100, 100),
    )

    assert visible is True
    assert state == "projected_visible"
    assert "points_world" in captured
    assert captured["points_world"].shape[0] > 3


def test_classify_projected_actor_visibility_rejects_actor_without_renderable_meshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_blender_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "pemoin.visualization.blender_scene.pipeline", raising=False
    )
    pipeline = importlib.import_module("pemoin.visualization.blender_scene.pipeline")

    actor_root = types.SimpleNamespace(name="Pedestrian01", children=[])

    with pytest.raises(ValueError, match="no renderable mesh descendants"):
        pipeline._classify_projected_actor_visibility(
            frame_idx=0,
            intrinsics_k=np.eye(3, dtype=np.float32),
            frame_to_c2w={0: np.eye(4, dtype=np.float32)},
            actor_root=actor_root,
            depsgraph=object(),
            image_shape=(100, 100),
        )
