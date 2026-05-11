from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _FakeSocket:
    def __init__(self, name: str, socket_type: str):
        self.name = name
        self.type = socket_type
        self.default_value = None


class _FakeSocketList(list):
    def __init__(self, *args, fail_named_access: set[str] | None = None):
        super().__init__(*args)
        self._fail_named_access = set() if fail_named_access is None else fail_named_access

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._fail_named_access:
                raise KeyError(key)
            for socket in self:
                if socket.name == key:
                    return socket
            aliases = {"Image": "Color", "Color": "Image"}
            alias = aliases.get(key)
            if alias is not None:
                for socket in self:
                    if socket.name == alias:
                        return socket
            raise KeyError(key)
        return super().__getitem__(key)


class _FakeImageFormat:
    def __init__(self) -> None:
        self.file_format = "PNG"
        self.color_mode = "RGBA"
        self.color_depth = None
        self.exr_codec = None


class _FakeFileSlot:
    def __init__(self, path: str = "") -> None:
        self.path = path


class _FakeFileSlots(list):
    def __init__(self) -> None:
        super().__init__([_FakeFileSlot("Image")])


class _FakeFileOutputItem:
    def __init__(self, name: str) -> None:
        self.name = name
        self.path = None
        self.override_node_format = False
        self.format = _FakeImageFormat()


class _FakeFileOutputItems(list):
    allowed_socket_types = {
        "FLOAT",
        "INT",
        "BOOLEAN",
        "VECTOR",
        "RGBA",
        "ROTATION",
        "MATRIX",
        "STRING",
        "MENU",
        "SHADER",
        "OBJECT",
        "IMAGE",
        "GEOMETRY",
        "COLLECTION",
        "TEXTURE",
        "MATERIAL",
        "BUNDLE",
        "CLOSURE",
    }

    def __init__(
        self,
        node: "_FakeNode",
        *,
        create_input_socket: bool = True,
        missing_input_names: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._node = node
        self.calls: list[tuple[str, str]] = []
        self._create_input_socket = create_input_socket
        self._missing_input_names = set() if missing_input_names is None else missing_input_names

    def new(self, socket_type: str, name: str):
        if socket_type not in self.allowed_socket_types:
            raise TypeError(f"unsupported socket_type={socket_type}")
        self.calls.append((socket_type, name))
        item = _FakeFileOutputItem(name)
        self.append(item)
        if self._create_input_socket and name not in self._missing_input_names:
            self._node.inputs.insert(-1, _FakeSocket(name, socket_type))
        return item


class _FakeNode:
    def __init__(
        self,
        node_type: str,
        *,
        group_interface: "_FakeInterface | None" = None,
        create_item_inputs: bool = True,
        missing_item_inputs: set[str] | None = None,
        fail_output_named_access: set[str] | None = None,
    ) -> None:
        self.node_type = node_type
        self.base_path = None
        self.directory = None
        self.file_name = None
        self.format = _FakeImageFormat()
        self.inputs = _FakeSocketList()
        self.outputs = _FakeSocketList(fail_named_access=fail_output_named_access)
        if node_type == "CompositorNodeRLayers":
            self.outputs.extend(
                [
                    _FakeSocket("Image", "RGBA"),
                    _FakeSocket("Depth", "VALUE"),
                    _FakeSocket("Shadow", "VALUE"),
                ]
            )
        elif node_type == "CompositorNodeComposite":
            self.inputs.append(_FakeSocket("Image", "RGBA"))
        elif node_type == "CompositorNodeSetAlpha":
            self.inputs.extend(
                [_FakeSocket("Image", "RGBA"), _FakeSocket("Alpha", "VALUE")]
            )
            self.outputs.append(_FakeSocket("Image", "RGBA"))
        elif node_type == "CompositorNodeRGB":
            self.outputs.append(_FakeSocket("Color", "RGBA"))
        elif node_type == "NodeGroupOutput":
            if group_interface is not None:
                self.inputs.extend(group_interface.output_sockets())
        elif node_type == "CompositorNodeOutputFile":
            self.inputs.extend(
                [_FakeSocket("Image", "RGBA"), _FakeSocket("", "CUSTOM")]
            )
            self.file_slots = _FakeFileSlots()
            self.file_output_items = _FakeFileOutputItems(
                self,
                create_input_socket=create_item_inputs,
                missing_input_names=missing_item_inputs,
            )


class _FakeNodes:
    def __init__(
        self,
        *,
        group_interface: "_FakeInterface | None" = None,
        create_item_inputs: bool = True,
        missing_item_inputs: set[str] | None = None,
        fail_output_named_access: set[str] | None = None,
    ) -> None:
        self._group_interface = group_interface
        self._create_item_inputs = create_item_inputs
        self._missing_item_inputs = missing_item_inputs
        self._fail_output_named_access = fail_output_named_access
        self.created: list[_FakeNode] = []

    def clear(self) -> None:
        self.created.clear()

    def new(self, node_type: str) -> _FakeNode:
        node = _FakeNode(
            node_type,
            group_interface=self._group_interface,
            create_item_inputs=self._create_item_inputs,
            missing_item_inputs=self._missing_item_inputs,
            fail_output_named_access=self._fail_output_named_access,
        )
        self.created.append(node)
        return node


class _FakeLinks:
    def __init__(self) -> None:
        self.connections: list[tuple[_FakeSocket, _FakeSocket]] = []

    def new(self, output_socket: _FakeSocket, input_socket: _FakeSocket) -> None:
        self.connections.append((output_socket, input_socket))


class _FakeInterface:
    def __init__(self) -> None:
        self._outputs: list[tuple[str, str]] = []

    def new_socket(self, *, name: str, in_out: str, socket_type: str) -> None:
        if in_out == "OUTPUT":
            self._outputs.append((name, socket_type))

    def output_sockets(self) -> list[_FakeSocket]:
        return [_FakeSocket(name, socket_type) for name, socket_type in self._outputs]


class _FakeNodeTree:
    def __init__(
        self,
        *,
        create_item_inputs: bool = True,
        missing_item_inputs: set[str] | None = None,
        fail_output_named_access: set[str] | None = None,
    ) -> None:
        self.interface = _FakeInterface()
        self.nodes = _FakeNodes(
            group_interface=self.interface,
            create_item_inputs=create_item_inputs,
            missing_item_inputs=missing_item_inputs,
            fail_output_named_access=fail_output_named_access,
        )
        self.links = _FakeLinks()


class _FakeNodeGroups:
    def __init__(
        self,
        *,
        create_item_inputs: bool = True,
        missing_item_inputs: set[str] | None = None,
        fail_output_named_access: set[str] | None = None,
    ) -> None:
        self.created: list[_FakeNodeTree] = []
        self._create_item_inputs = create_item_inputs
        self._missing_item_inputs = missing_item_inputs
        self._fail_output_named_access = fail_output_named_access

    def new(self, _name: str, _tree_type: str) -> _FakeNodeTree:
        tree = _FakeNodeTree(
            create_item_inputs=self._create_item_inputs,
            missing_item_inputs=self._missing_item_inputs,
            fail_output_named_access=self._fail_output_named_access,
        )
        self.created.append(tree)
        return tree


class _FakeRenderSettings:
    def __init__(self) -> None:
        self.use_compositing = False


class _FakeViewLayer:
    def __init__(self) -> None:
        self.use_pass_z = False
        self.use_pass_shadow = False


def _install_pipeline_blender_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_item_inputs: bool = True,
    missing_item_inputs: set[str] | None = None,
    fail_output_named_access: set[str] | None = None,
):
    bpy_module = types.ModuleType("bpy")
    bpy_module.app = types.SimpleNamespace(version_string="5.0.1", version=(5, 0, 1))
    bpy_module.types = types.SimpleNamespace(
        Object=object,
        Scene=object,
        Collection=object,
        Camera=object,
        Material=object,
    )
    bpy_module.context = types.SimpleNamespace(
        scene=types.SimpleNamespace(
            render=_FakeRenderSettings(),
            compositing_node_group=None,
            node_tree=_FakeNodeTree(),
            use_nodes=False,
        ),
        view_layer=_FakeViewLayer(),
    )
    bpy_module.data = types.SimpleNamespace(
        node_groups=_FakeNodeGroups(
            create_item_inputs=create_item_inputs,
            missing_item_inputs=missing_item_inputs,
            fail_output_named_access=fail_output_named_access,
        )
    )
    mathutils_module = types.ModuleType("mathutils")
    mathutils_module.Matrix = type("Matrix", (), {})
    mathutils_module.Vector = type("Vector", (), {})

    monkeypatch.setitem(sys.modules, "bpy", bpy_module)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils_module)
    monkeypatch.delitem(
        sys.modules,
        "pemoin.visualization.blender_scene.pipeline",
        raising=False,
    )
    return importlib.import_module("pemoin.visualization.blender_scene.pipeline")


def test_configure_compositing_group_render_outputs_uses_rgba_shadow_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    export_api = pipeline._configure_compositing_group_render_outputs(
        tmp_path / "depth",
        tmp_path / "shadow",
    )

    assert export_api == "compositing_node_group"
    tree = pipeline.bpy.context.scene.compositing_node_group
    output_nodes = [
        node for node in tree.nodes.created if node.node_type == "CompositorNodeOutputFile"
    ]
    depth_out, shadow_out = output_nodes
    assert depth_out.file_output_items.calls == [("FLOAT", "Depth")]
    assert shadow_out.file_output_items.calls == [("RGBA", "Shadow")]
    assert depth_out.file_output_items[0].override_node_format is True
    assert shadow_out.file_output_items[0].override_node_format is True
    assert depth_out.file_output_items[0].format.file_format == "OPEN_EXR"
    assert depth_out.file_output_items[0].format.color_mode == "BW"
    assert depth_out.file_output_items[0].format.color_depth == "32"
    assert depth_out.file_output_items[0].format.exr_codec == "ZIP"
    assert shadow_out.file_output_items[0].format.file_format == "PNG"
    assert shadow_out.file_output_items[0].format.color_mode == "RGBA"
    assert depth_out.directory == str(tmp_path / "depth")
    assert depth_out.file_name == "depth_"
    assert shadow_out.directory == str(tmp_path / "shadow")
    assert shadow_out.file_name == "shadow_"


def test_min_support_confidence_for_projection_relaxes_persisted_blend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = pipeline.SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=None,
        cube_size=1.0,
        collection_name="Scene",
        foot_contact_min_plane_confidence_for_projection=0.35,
    )
    support = pipeline.SupportSurfaceResolution(
        mode="persisted_blend",
        normal=None,
        offset=0.0,
        confidence=0.0,
        source_frame_indices=(),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=None,
        persisted_blend_disagreement_m=None,
        held_from_previous=False,
        origin_mode="persisted_blend",
    )

    assert pipeline._min_support_confidence_for_projection(
        spec=spec,
        support_resolution=support,
    ) == pytest.approx(0.25)


def test_min_support_confidence_for_projection_keeps_local_fit_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = pipeline.SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=None,
        cube_size=1.0,
        collection_name="Scene",
        foot_contact_min_plane_confidence_for_projection=0.35,
    )
    support = pipeline.SupportSurfaceResolution(
        mode="local_fit",
        normal=None,
        offset=0.0,
        confidence=0.0,
        source_frame_indices=(),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=None,
        persisted_blend_disagreement_m=None,
        held_from_previous=False,
        origin_mode="local_fit",
    )

    assert pipeline._min_support_confidence_for_projection(
        spec=spec,
        support_resolution=support,
    ) == pytest.approx(0.35)


def test_min_support_confidence_for_projection_relaxes_held_persisted_support(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = pipeline.SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=None,
        cube_size=1.0,
        collection_name="Scene",
        foot_contact_min_plane_confidence_for_projection=0.35,
    )
    support = pipeline.SupportSurfaceResolution(
        mode="hold_prev",
        normal=None,
        offset=0.0,
        confidence=0.0,
        source_frame_indices=(),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=None,
        persisted_blend_disagreement_m=None,
        held_from_previous=True,
        origin_mode="persisted_blend",
    )

    assert pipeline._min_support_confidence_for_projection(
        spec=spec,
        support_resolution=support,
    ) == pytest.approx(0.25)


def test_stabilize_support_surface_accepts_same_plane_with_large_anchor_shift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    previous = pipeline.SupportSurfaceResolution(
        mode="persisted_blend",
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.25,
        confidence=0.30,
        source_frame_indices=(43,),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=3,
        persisted_blend_disagreement_m=0.01,
        held_from_previous=False,
        origin_mode="persisted_blend",
    )
    current = pipeline.SupportSurfaceResolution(
        mode="persisted_blend",
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.251,
        confidence=0.31,
        source_frame_indices=(44,),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=3,
        persisted_blend_disagreement_m=0.01,
        held_from_previous=False,
        origin_mode="persisted_blend",
    )

    stabilized, hold_count, normal_jump, height_jump, anchor_shift, current_signed, previous_signed = (
        pipeline._stabilize_support_surface(
            current=current,
            previous=previous,
            hold_count=0,
            comparison_anchor=np.array([1.8, 0.0, 0.05], dtype=np.float32),
            current_anchor=np.array([1.8, 0.0, 0.05], dtype=np.float32),
            previous_anchor=np.array([0.0, 0.0, 0.05], dtype=np.float32),
            max_hold_frames=6,
            max_anchor_shift_m=0.57,
        )
    )

    assert stabilized.mode == "persisted_blend"
    assert hold_count == 0
    assert normal_jump == pytest.approx(0.0)
    assert height_jump == pytest.approx(0.001, abs=1e-6)
    assert anchor_shift == pytest.approx(1.8)
    assert current_signed == pytest.approx(-0.201)
    assert previous_signed == pytest.approx(-0.2)


def test_closest_persisted_plane_for_point_uses_bootstrap_locality_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = pipeline.SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=None,
        cube_size=1.0,
        collection_name="Scene",
        global_plane_range_m=15.0,
        max_plane_center_xy_distance_m=8.0,
    )
    plane = pipeline.RoadPlaneSpec(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=0.0,
        center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scale_u=15.0,
        scale_v=15.0,
        frame_index=12,
        confidence=0.6,
    )
    trajectory_c2w = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)

    chosen, locality = pipeline._closest_persisted_plane_for_point(
        np.array([8.02, 0.0, 0.0], dtype=np.float32),
        planes={12: plane},
        current_frame_index=0,
        spec=spec,
        trajectory_c2w=trajectory_c2w,
    )

    assert chosen is plane
    assert locality.nearest_xy_distance_m == pytest.approx(8.02)
    assert locality.effective_limit_m == pytest.approx(8.27)
    assert locality.locality_mode == "bootstrap_relaxed"


def test_blend_persisted_support_planes_accepts_bootstrap_relaxed_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = pipeline.SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=None,
        cube_size=1.0,
        collection_name="Scene",
        global_plane_range_m=15.0,
        global_plane_frame_window=3,
        global_plane_confidence_threshold=0.3,
        max_plane_center_xy_distance_m=8.0,
    )
    plane = pipeline.RoadPlaneSpec(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.1,
        center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scale_u=15.0,
        scale_v=15.0,
        frame_index=12,
        confidence=0.6,
    )
    trajectory_c2w = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)

    support = pipeline._blend_persisted_support_planes(
        spec=spec,
        frame_idx=0,
        support_anchor_world=np.array([8.02, 0.0, 0.1], dtype=np.float32),
        planes={12: plane},
        trajectory_c2w=trajectory_c2w,
    )

    assert support.mode == "persisted_blend"
    assert support.confidence > 0.0
    assert support.source_frame_indices == (12,)


def test_raise_for_grounding_failures_reports_locality_without_resolution_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    diagnostic = pipeline.GroundingDiagnostic(
        frame_index=0,
        support_mode="none",
        support_confidence=None,
        support_source_frame_indices=tuple(),
        support_failure_reason=(
            "persisted_fallback_locality_rejected: "
            "nearest_xy_distance_m=8.0200 effective_limit_m=8.2700"
        ),
        sole_offset_m=0.06,
        chosen_plane_frame_index=None,
        chosen_plane_normal=None,
        chosen_plane_offset=None,
        chosen_plane_center=None,
        chosen_plane_center_xy_distance_m=8.02,
        selected_support_foot="left",
        left_foot_before=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        right_foot_before=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        left_foot_after=None,
        right_foot_after=None,
        support_point_before=np.array([8.02, 0.0, 0.1], dtype=np.float32),
        support_point_after=None,
        pre_correction_signed_distance_m=None,
        post_correction_signed_distance_m=None,
        left_post_signed_distance_m=None,
        right_post_signed_distance_m=None,
        support_jump_from_prev_deg=None,
        support_height_jump_from_prev_m=None,
        support_anchor_shift_from_prev_m=None,
        dynamic_anchor_shift_limit_m=0.41,
        applied_translation_world=np.zeros(3, dtype=np.float32),
        plane_selection_rejected_for_locality=True,
        missing_left_foot=False,
        missing_right_foot=False,
        no_plane=True,
        effective_persisted_plane_locality_limit_m=8.27,
        nearest_persisted_plane_center_xy_distance_m=8.02,
        persisted_plane_locality_mode="bootstrap_relaxed",
    )

    with pytest.raises(ValueError, match=r"locality>8\.000m=0") as exc_info:
        pipeline._raise_for_grounding_failures(
            diagnostics=[diagnostic],
            max_residual_m=0.08,
            max_plane_center_xy_distance_m=8.0,
        )

    assert "support_resolution_failed" not in str(exc_info.value)


def test_raise_for_grounding_failures_ignores_off_camera_locality_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    diagnostic = pipeline.GroundingDiagnostic(
        frame_index=166,
        support_mode="none",
        support_confidence=None,
        support_source_frame_indices=tuple(),
        support_failure_reason="persisted_fallback_locality_rejected",
        sole_offset_m=0.06,
        chosen_plane_frame_index=None,
        chosen_plane_normal=None,
        chosen_plane_offset=None,
        chosen_plane_center=None,
        chosen_plane_center_xy_distance_m=15.07,
        selected_support_foot="path",
        left_foot_before=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        right_foot_before=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        left_foot_after=None,
        right_foot_after=None,
        support_point_before=np.array([15.07, 0.0, 0.1], dtype=np.float32),
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
        plane_selection_rejected_for_locality=True,
        missing_left_foot=False,
        missing_right_foot=False,
        no_plane=True,
        visibility_culled=True,
        visibility_cull_reason="actor_off_camera",
        frame_requires_support=True,
        effective_persisted_plane_locality_limit_m=15.0,
        nearest_persisted_plane_center_xy_distance_m=15.07,
        persisted_plane_locality_mode="rejected",
        support_state="unsupported",
        visibility_contract_state="actor_off_camera",
    )

    pipeline._raise_for_grounding_failures(
        diagnostics=[diagnostic],
        max_residual_m=0.08,
        max_plane_center_xy_distance_m=8.0,
    )


def test_hold_previous_support_when_unresolved_returns_full_tuple_without_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    result = pipeline._hold_previous_support_when_unresolved(
        previous=None,
        hold_count=2,
        max_hold_frames=6,
        current_anchor=np.array([1.0, 2.0, 0.0], dtype=np.float32),
        previous_anchor=None,
    )

    assert len(result) == 7
    assert result == (None, 2, None, None, None, None, None)


def test_hold_previous_support_when_unresolved_returns_full_tuple_with_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    previous = pipeline.SupportSurfaceResolution(
        mode="persisted_blend",
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.25,
        confidence=0.30,
        source_frame_indices=(43,),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=3,
        persisted_blend_disagreement_m=0.01,
        held_from_previous=False,
        origin_mode="persisted_blend",
    )

    result = pipeline._hold_previous_support_when_unresolved(
        previous=previous,
        hold_count=1,
        max_hold_frames=6,
        current_anchor=np.array([1.8, 0.0, 0.0], dtype=np.float32),
        previous_anchor=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert len(result) == 7
    held, hold_count, support_jump, support_height, anchor_shift, current_signed, previous_signed = result
    assert held is not None
    assert held.mode == "hold_prev"
    assert held.held_from_previous is True
    assert hold_count == 2
    assert support_jump is None
    assert support_height is None
    assert anchor_shift == pytest.approx(1.8)
    assert current_signed is None
    assert previous_signed is None


def test_blend_support_anchor_uses_continuous_two_foot_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    plane = pipeline.RoadPlaneSpec(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=0.0,
        center=np.zeros(3, dtype=np.float32),
        scale_u=5.0,
        scale_v=5.0,
        frame_index=0,
        confidence=1.0,
    )
    anchor, left_weight, right_weight, label = pipeline._blend_support_anchor(
        np.array([0.0, 0.0, 0.20], dtype=np.float32),
        np.array([0.4, 0.0, 0.22], dtype=np.float32),
        plane=plane,
        previous_weights=(0.5, 0.5),
    )

    assert label == "both"
    assert 0.0 < left_weight < 1.0
    assert 0.0 < right_weight < 1.0
    assert left_weight + right_weight == pytest.approx(1.0)
    assert 0.0 < float(anchor[0]) < 0.4


def test_filter_support_anchor_height_clamps_small_flat_plane_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    previous_plane = pipeline.SupportSurfaceResolution(
        mode="local_fit",
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.25,
        confidence=0.8,
        source_frame_indices=(1,),
        local_fit_point_count=100,
        local_fit_radius_m=1.0,
        local_fit_residual_p90_m=0.01,
        local_fit_inlier_ratio=0.9,
        persisted_blend_candidate_count=None,
        persisted_blend_disagreement_m=None,
        held_from_previous=False,
        origin_mode="local_fit",
    )
    current_plane = pipeline.RoadPlaneSpec(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.252,
        center=np.zeros(3, dtype=np.float32),
        scale_u=5.0,
        scale_v=5.0,
        frame_index=2,
        confidence=1.0,
    )

    filtered_anchor, raw_height, filtered_height = pipeline._filter_support_anchor_height(
        anchor_world=np.array([1.0, 0.0, 0.29], dtype=np.float32),
        previous_anchor_world=np.array([1.0, 0.0, 0.25], dtype=np.float32),
        previous_plane=previous_plane,
        current_plane=current_plane,
    )

    assert raw_height == pytest.approx(0.29)
    assert filtered_height == pytest.approx(0.26)
    assert float(filtered_anchor[2]) == pytest.approx(0.26)


def test_resolve_support_anchor_selection_prefers_planted_foot_over_smoothed_blend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    plane = pipeline.RoadPlaneSpec(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=0.0,
        center=np.zeros(3, dtype=np.float32),
        scale_u=5.0,
        scale_v=5.0,
        frame_index=0,
        confidence=1.0,
    )
    spec = types.SimpleNamespace(
        support_anchor_dual_support_height_tol_m=0.035,
        support_anchor_switch_margin=0.12,
        support_anchor_transfer_frames=3,
    )

    selection = pipeline._resolve_support_anchor_selection(
        np.array([0.0, 0.0, 0.20], dtype=np.float32),
        np.array([0.4, 0.0, 0.11], dtype=np.float32),
        plane=plane,
        spec=spec,
        previous_label="right",
        previous_confidences=(0.5, 0.5),
    )

    assert selection.label == "right"
    assert selection.left_weight == pytest.approx(0.0)
    assert selection.right_weight == pytest.approx(1.0)
    assert float(selection.anchor[2]) == pytest.approx(0.11)


def test_filter_support_anchor_height_state_uses_raw_anchor_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    previous_plane = pipeline.SupportSurfaceResolution(
        mode="persisted_blend",
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.80,
        confidence=0.8,
        source_frame_indices=(1,),
        local_fit_point_count=None,
        local_fit_radius_m=None,
        local_fit_residual_p90_m=None,
        local_fit_inlier_ratio=None,
        persisted_blend_candidate_count=3,
        persisted_blend_disagreement_m=0.0,
        held_from_previous=False,
        origin_mode="persisted_blend",
    )
    current_plane = pipeline.RoadPlaneSpec(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.802,
        center=np.zeros(3, dtype=np.float32),
        scale_u=5.0,
        scale_v=5.0,
        frame_index=2,
        confidence=1.0,
    )
    spec = types.SimpleNamespace(
        support_anchor_flat_ground_normal_z_min=0.97,
        support_anchor_allow_vertical_motion_on_plane_change=True,
        support_anchor_plane_change_height_tol_m=0.04,
        support_anchor_max_z_step_m=0.01,
    )

    result = pipeline._filter_support_anchor_height_state(
        anchor_world=np.array([1.0, 0.0, 0.13], dtype=np.float32),
        previous_anchor_world=np.array([1.0, 0.0, 0.10], dtype=np.float32),
        previous_plane=previous_plane,
        current_plane=current_plane,
        spec=spec,
    )

    assert result.raw_height == pytest.approx(0.13)
    assert result.filtered_height == pytest.approx(0.11)
    assert float(result.anchor[2]) == pytest.approx(0.11)
    assert result.clamped is True


def test_clean_contact_state_sequence_cleans_short_state_islands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = types.SimpleNamespace(
        foot_contact_min_stance_frames=2,
        foot_contact_min_swing_frames=2,
    )
    cleaned = pipeline._clean_contact_state_sequence(
        [
            "left_stance",
            "left_stance",
            "swing",
            "right_stance",
            "right_stance",
            "swing",
        ],
        spec=spec,
    )

    assert cleaned == [
        "left_stance",
        "left_stance",
        "left_stance",
        "right_stance",
        "right_stance",
        "right_stance",
    ]


def test_clean_contact_state_sequence_preserves_single_frame_stance_at_low_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = types.SimpleNamespace(
        foot_contact_min_stance_frames=2,
        foot_contact_min_swing_frames=2,
        sampling_fps=10.0,
    )
    cleaned = pipeline._clean_contact_state_sequence(
        [
            "left_stance",
            "swing",
            "swing",
            "swing",
            "right_stance",
            "swing",
            "left_stance",
        ],
        spec=spec,
    )

    assert cleaned[4] == "right_stance"


def test_resolve_segment_support_weights_uses_stance_segments_and_transfer_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = types.SimpleNamespace(support_anchor_transfer_frames=3)
    dual_state = pipeline.ContactFrameState(
        frame_index=10,
        phase=0.4,
        raw_state="dual_support",
        clean_state="dual_support",
        segment_id=2,
        segment_kind="dual_support",
        segment_frame_index=0,
        segment_length=3,
        previous_stance_kind="left_stance",
        next_stance_kind="right_stance",
    )

    label, left_weight, right_weight, policy, transfer_state, authority = (
        pipeline._resolve_segment_support_weights(dual_state, spec=spec)
    )

    assert label == "both"
    assert left_weight > right_weight
    assert policy == "contact_segment_dual_support"
    assert transfer_state == "transfer"
    assert authority == "left_to_right"


def test_resolve_segment_support_weights_releases_long_swing_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    spec = types.SimpleNamespace(support_anchor_transfer_frames=3)
    swing_state = pipeline.ContactFrameState(
        frame_index=14,
        phase=0.62,
        raw_state="swing",
        clean_state="swing",
        segment_id=4,
        segment_kind="swing",
        segment_frame_index=3,
        segment_length=6,
        previous_stance_kind="left_stance",
        next_stance_kind="right_stance",
    )

    label, left_weight, right_weight, policy, transfer_state, authority = (
        pipeline._resolve_segment_support_weights(swing_state, spec=spec)
    )

    assert label == "left"
    assert left_weight == pytest.approx(1.0)
    assert right_weight == pytest.approx(0.0)
    assert policy == "contact_segment_swing_release"
    assert transfer_state == "swing_release"
    assert authority == "released_swing_left"


def test_support_planes_match_for_lock_uses_same_plane_tolerances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    previous = pipeline.SupportSurfaceResolution(
        mode="local_fit",
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=-0.1,
        confidence=0.9,
        source_frame_indices=(1,),
        local_fit_point_count=10,
        local_fit_radius_m=1.0,
        local_fit_residual_p90_m=0.01,
        local_fit_inlier_ratio=0.95,
        persisted_blend_candidate_count=None,
        persisted_blend_disagreement_m=None,
        held_from_previous=False,
        origin_mode="local_fit",
    )
    current = pipeline.SupportSurfaceResolution(
        mode="local_fit",
        normal=np.array([0.0, 0.02, 0.9998], dtype=np.float32),
        offset=-0.11,
        confidence=0.9,
        source_frame_indices=(2,),
        local_fit_point_count=10,
        local_fit_radius_m=1.0,
        local_fit_residual_p90_m=0.01,
        local_fit_inlier_ratio=0.95,
        persisted_blend_candidate_count=None,
        persisted_blend_disagreement_m=None,
        held_from_previous=False,
        origin_mode="local_fit",
    )
    spec = types.SimpleNamespace(
        support_anchor_same_plane_normal_tol_deg=3.0,
        support_anchor_same_plane_height_tol_m=0.015,
    )

    matches, reason = pipeline._support_planes_match_for_lock(
        previous_plane=previous,
        current_plane=current,
        comparison_anchor=np.array([0.0, 0.0, 0.1], dtype=np.float32),
        spec=spec,
    )

    assert matches is True
    assert reason is None


def test_configure_compositing_group_render_outputs_fails_fast_when_socket_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(
        monkeypatch,
        missing_item_inputs={"Shadow"},
    )

    with pytest.raises(RuntimeError, match="Expected input socket in \\['Shadow'\\]"):
        pipeline._configure_compositing_group_render_outputs(
            tmp_path / "depth",
            tmp_path / "shadow",
        )


def test_configure_compositing_group_render_outputs_uses_iterated_shadow_socket_when_named_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(
        monkeypatch,
        fail_output_named_access={"Shadow"},
    )

    export_api = pipeline._configure_compositing_group_render_outputs(
        tmp_path / "depth",
        tmp_path / "shadow",
    )

    assert export_api == "compositing_node_group"


def test_configure_legacy_render_output_nodes_reuses_single_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)
    scene = pipeline.bpy.context.scene
    scene.node_tree = _FakeNodeTree()

    export_api = pipeline._configure_legacy_render_output_nodes(
        tmp_path / "depth",
        tmp_path / "shadow",
    )

    assert export_api == "legacy_node_tree"
    output_nodes = [
        node
        for node in scene.node_tree.nodes.created
        if node.node_type == "CompositorNodeOutputFile"
    ]
    depth_out, shadow_out = output_nodes
    assert len(depth_out.file_slots) == 1
    assert len(shadow_out.file_slots) == 1
    assert depth_out.file_slots[0].path == "depth_"
    assert shadow_out.file_slots[0].path == "shadow_"
    assert depth_out.format.file_format == "OPEN_EXR"
    assert depth_out.format.color_mode == "BW"
    assert depth_out.format.color_depth == "32"
    assert depth_out.format.exr_codec == "ZIP"
    assert shadow_out.format.file_format == "PNG"
    assert shadow_out.format.color_mode == "RGBA"
    assert depth_out.base_path == str(tmp_path / "depth")
    assert shadow_out.base_path == str(tmp_path / "shadow")


def _overlay_diag(
    pipeline,
    *,
    frame_index: int = 0,
    support_to_contact_foot_px: float | None = None,
    state: str = "verified",
    trusted: bool = True,
    abort_relevant: bool = True,
    validation_passed: bool = True,
    failure_reason: str | None = None,
):
    return pipeline.OverlayValidationDiagnostic(
        frame_index=frame_index,
        has_visible_pedestrian=True,
        lowest_alpha_u=10,
        lowest_alpha_v=10,
        lowest_alpha_row_coverage_px=2,
        touches_image_bottom=False,
        left_foot_projected_uv=np.array([9.0, 9.0], dtype=np.float32),
        right_foot_projected_uv=np.array([11.0, 9.0], dtype=np.float32),
        left_foot_visible_expected=True,
        right_foot_visible_expected=True,
        support_point_projected_uv=np.array([10.0, 10.0], dtype=np.float32),
        support_point_projected_visible=True,
        support_point_depth_m=5.0,
        scene_depth_at_support_px_m=5.0,
        support_point_occluded_by_scene=False,
        selected_foot_projected_uv=np.array([10.0, 10.0], dtype=np.float32),
        selected_foot_projected_visible=True,
        support_to_left_foot_px=1.0,
        support_to_right_foot_px=1.0,
        support_to_contact_foot_px=support_to_contact_foot_px,
        contact_foot_comparison_mode="right",
        support_to_silhouette_bottom_px=1.0,
        support_to_selected_foot_px=1.0,
        support_patch_road_fraction=1.0,
        support_patch_nonroad_fraction=0.0,
        support_patch_size_px=16,
        road_region_validation_available=True,
        road_context_search_mode="direct_patch",
        validation_passed=validation_passed,
        failure_reason=failure_reason,
        support_mode="persisted_blend",
        selected_support_foot="right",
        warning_flags=tuple(),
        selected_support_foot_expected_visible=True,
        contact_validation_state=state,
        contact_validation_trusted=trusted,
        abort_relevant=abort_relevant,
    )


def test_overlay_validation_marks_fallback_other_visible_as_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    run_dir = tmp_path / "run"
    semantics_dir = run_dir / "standard" / "semantics_2d"
    depth_dir = run_dir / "standard" / "depth"
    semantics_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    np.savez_compressed(
        semantics_dir / "000000.npz",
        label_ids=np.zeros((20, 20), dtype=np.int32),
        metadata=np.array({"class_id_to_label": {0: "road"}}, dtype=object),
        segments_info=np.asarray([], dtype=object),
    )
    np.savez_compressed(
        depth_dir / "000000.npz",
        depth=np.full((20, 20), 10.0, dtype=np.float32),
    )

    ped_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    ped_rgba[12:19, 0:3, 3] = 255

    diag = pipeline._make_overlay_validation_diagnostic(
        run_dir=run_dir,
        frame_idx=0,
        ped_rgba=ped_rgba,
        left_uv=np.array([2.0, 15.0], dtype=np.float32),
        left_valid=True,
        right_uv=np.array([-4.0, 15.0], dtype=np.float32),
        right_valid=False,
        selected_support_foot="right",
        support_mode="persisted_blend",
        support_point_uv=np.array([0.5, 18.0], dtype=np.float32),
        support_point_visible=True,
        support_point_depth_m=10.0,
        road_labels=("road",),
    )

    assert diag.validation_passed is True
    assert diag.failure_reason is None
    assert diag.contact_foot_comparison_mode == "fallback_other_visible"
    assert diag.contact_validation_state == "unverifiable"
    assert diag.contact_validation_trusted is False
    assert diag.abort_relevant is False
    assert diag.selected_support_foot_expected_visible is False
    assert "selected_support_foot_not_visible" in diag.warning_flags
    assert "contact_validation_fallback_other_visible" in diag.warning_flags
    assert diag.touches_image_left is True


def test_overlay_validation_summary_ignores_unverifiable_frames_for_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    summary = pipeline._evaluate_overlay_validation_summary(
        [
            _overlay_diag(pipeline, frame_index=0, support_to_contact_foot_px=2.0),
            _overlay_diag(pipeline, frame_index=1, support_to_contact_foot_px=6.0),
            _overlay_diag(
                pipeline,
                frame_index=2,
                support_to_contact_foot_px=45.0,
                state="unverifiable",
                trusted=False,
                abort_relevant=False,
            ),
        ]
    )

    assert summary["hard_fail"] is False
    assert summary["median_contact"] == pytest.approx(4.0)
    assert summary["p90_contact"] == pytest.approx(5.6)
    assert summary["validation_degraded"] is True


def test_overlay_validation_summary_degrades_on_isolated_trusted_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    summary = pipeline._evaluate_overlay_validation_summary(
        [
            _overlay_diag(pipeline, frame_index=0, support_to_contact_foot_px=2.0),
            _overlay_diag(pipeline, frame_index=1, support_to_contact_foot_px=3.0),
            _overlay_diag(pipeline, frame_index=2, support_to_contact_foot_px=4.0),
            _overlay_diag(pipeline, frame_index=3, support_to_contact_foot_px=5.0),
            _overlay_diag(
                pipeline,
                frame_index=4,
                support_to_contact_foot_px=8.0,
                state="hard_failure",
                trusted=True,
                abort_relevant=True,
                validation_passed=False,
                failure_reason="contact_foot_mismatch",
            ),
        ]
    )

    assert summary["hard_fail"] is False
    assert summary["validation_degraded"] is True
    assert summary["failure_reason"] is None


def test_overlay_validation_summary_aborts_when_hard_failure_ratio_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    summary = pipeline._evaluate_overlay_validation_summary(
        [
            _overlay_diag(
                pipeline,
                frame_index=0,
                support_to_contact_foot_px=24.0,
                state="hard_failure",
                trusted=True,
                abort_relevant=True,
                validation_passed=False,
                failure_reason="contact_foot_mismatch",
            ),
            _overlay_diag(
                pipeline,
                frame_index=1,
                support_to_contact_foot_px=22.0,
                state="hard_failure",
                trusted=True,
                abort_relevant=True,
                validation_passed=False,
                failure_reason="contact_foot_mismatch",
            ),
            _overlay_diag(pipeline, frame_index=2, support_to_contact_foot_px=2.0),
            _overlay_diag(pipeline, frame_index=3, support_to_contact_foot_px=3.0),
        ]
    )

    assert summary["hard_fail"] is True
    assert summary["failure_reason"] == "contact_foot_mismatch"


def test_overlay_validation_summary_relaxes_thresholds_for_low_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)
    adaptive = pipeline.AdaptiveValidationContext.from_runtime(
        pipeline.ValidationPolicySettings(enabled=True, reference_sampling_fps=10.0),
        {"frame_provider_info": {"tool": "test", "settings": {"sampling_fps": 4.0}}},
    )

    summary = pipeline._evaluate_overlay_validation_summary(
        [
            _overlay_diag(pipeline, frame_index=0, support_to_contact_foot_px=4.0),
            _overlay_diag(pipeline, frame_index=1, support_to_contact_foot_px=7.0),
            _overlay_diag(pipeline, frame_index=2, support_to_contact_foot_px=12.0),
            _overlay_diag(pipeline, frame_index=3, support_to_contact_foot_px=20.0),
        ],
        adaptive=adaptive,
    )

    assert summary["hard_fail"] is False
    assert summary["validation_policy"]["enabled"] is True
    assert summary["effective_thresholds"]["p90_support_to_contact_foot_px"]["soft"] > 18.0


def test_build_actor_support_contract_uses_asset_native_root_to_support_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)

    contract = pipeline._build_actor_support_contract(
        root_positions_world=[
            np.array([0.0, 0.0, 1.05], dtype=np.float32),
            np.array([0.5, 0.0, 1.04], dtype=np.float32),
            np.array([1.0, 0.0, 1.06], dtype=np.float32),
        ],
        left_feet_world=[
            np.array([0.0, 0.0, 0.08], dtype=np.float32),
            np.array([0.5, 0.1, 0.09], dtype=np.float32),
            np.array([1.0, 0.0, 0.10], dtype=np.float32),
        ],
        right_feet_world=[
            np.array([0.1, 0.0, 0.10], dtype=np.float32),
            np.array([0.6, -0.1, 0.11], dtype=np.float32),
            np.array([1.1, 0.0, 0.12], dtype=np.float32),
        ],
    )

    assert contract.root_to_support_m == pytest.approx(1.02, abs=1e-6)
    assert contract.support_samples_used == 3


def test_enforce_render_visibility_parity_rejects_rendered_visible_projected_off_camera(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)
    vis_dir = tmp_path / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "render_parity_diagnostics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="rendered_visible_but_projected_off_camera=12"):
        pipeline._enforce_render_visibility_parity(
            run_dir=tmp_path,
            frames=[
                pipeline.RenderVisibilityFrame(
                    frame_index=12,
                    rendered_visible=True,
                    rendered_alpha_pixels=42,
                    projected_visible=False,
                    support_state="supported",
                    visibility_contract_state="actor_off_camera",
                )
            ],
        )

    payload = json.loads((vis_dir / "render_parity_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["render_visibility_parity"]["mismatch_count"] == 1


def test_enforce_render_visibility_parity_tolerates_isolated_boundary_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)
    vis_dir = tmp_path / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "render_parity_diagnostics.json").write_text("{}", encoding="utf-8")

    pipeline._enforce_render_visibility_parity(
        run_dir=tmp_path,
        frames=[
            pipeline.RenderVisibilityFrame(
                frame_index=11,
                rendered_visible=True,
                rendered_alpha_pixels=64,
                projected_visible=True,
                support_state="supported",
                visibility_contract_state="projected_visible",
            ),
            pipeline.RenderVisibilityFrame(
                frame_index=12,
                rendered_visible=True,
                rendered_alpha_pixels=42,
                projected_visible=False,
                support_state="supported",
                visibility_contract_state="actor_off_camera",
            ),
            pipeline.RenderVisibilityFrame(
                frame_index=13,
                rendered_visible=False,
                rendered_alpha_pixels=0,
                projected_visible=False,
                support_state="supported",
                visibility_contract_state="actor_off_camera",
            ),
        ],
    )

    payload = json.loads((vis_dir / "render_parity_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["render_visibility_parity"]["mismatch_count"] == 0
    assert payload["render_visibility_parity"][
        "rendered_visible_but_projected_off_camera_boundary_tolerated_frames"
    ] == [12]


def test_enforce_render_visibility_parity_tolerates_isolated_boundary_empty_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)
    vis_dir = tmp_path / "standard" / "visualizations" / "blender_scene"
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "render_parity_diagnostics.json").write_text("{}", encoding="utf-8")

    pipeline._enforce_render_visibility_parity(
        run_dir=tmp_path,
        frames=[
            pipeline.RenderVisibilityFrame(
                frame_index=43,
                rendered_visible=True,
                rendered_alpha_pixels=64,
                projected_visible=True,
                support_state="supported",
                visibility_contract_state="projected_visible",
            ),
            pipeline.RenderVisibilityFrame(
                frame_index=44,
                rendered_visible=False,
                rendered_alpha_pixels=0,
                projected_visible=True,
                support_state="supported",
                visibility_contract_state="projected_visible",
            ),
            pipeline.RenderVisibilityFrame(
                frame_index=45,
                rendered_visible=False,
                rendered_alpha_pixels=0,
                projected_visible=False,
                support_state="supported",
                visibility_contract_state="actor_off_camera",
            ),
        ],
    )

    payload = json.loads((vis_dir / "render_parity_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["render_visibility_parity"]["projected_visible_but_rendered_empty_count"] == 0
    assert payload["render_visibility_parity"][
        "projected_visible_but_rendered_empty_boundary_tolerated_frames"
    ] == [44]


def test_smooth_grounded_root_heights_preserves_xy_free_transition_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _install_pipeline_blender_stubs(monkeypatch)
    spec = pipeline.SceneSpec(
        run_dir=tmp_path,
        trajectory_path=tmp_path / "poses.npz",
        output_path=None,
        cube_size=1.0,
        collection_name="Scene",
        sampling_fps=10.0,
        trajectory_grounding_transition_frames=3,
        trajectory_grounding_max_step_m=0.05,
        trajectory_grounding_max_vertical_velocity_mps=1.0,
        trajectory_grounding_max_vertical_accel_mps2=10.0,
    )

    smoothed, velocities, accels, phases = pipeline._smooth_grounded_root_heights(
        raw_root_heights_m=[1.00, 1.00, 1.15, 1.15, 1.15],
        segment_ids=[0, 0, 1, 1, 1],
        spec=spec,
    )

    assert smoothed[0:2] == pytest.approx([1.0, 1.0])
    assert smoothed[2] < 1.15
    assert smoothed[3] <= 1.15
    assert phases[2] == "transition"
    assert velocities[2] is not None
    assert accels[2] is not None
