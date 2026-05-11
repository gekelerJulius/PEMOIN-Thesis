"""Tests for GeometryFusionProvider support-plane orientation helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    FrameData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    RoadPlaneData,
    SemanticsData,
)
from pemoin.providers.geometry_fusion import provider as geometry_fusion_provider_module
from pemoin.providers.geometry_fusion.provider import (
    GeometryFusionProvider,
    _orient_world_support_plane,
)
from pemoin.runtime.cache import CrossRunCacheManager
from pemoin.providers.geometry_fusion.stages.road_rectification import (
    FrameRectificationResult,
)
from pemoin.providers.geometry_fusion.stages import factor_graph as factor_graph_module
from pemoin.providers.geometry_fusion.utils import road_pixel_selection as road_pixel_selection_module


def test_orient_world_support_plane_chooses_camera_up_facing_sign():
    normal, offset = _orient_world_support_plane(
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        -1.6,
        np.zeros((3,), dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert abs(float(offset) - 1.6) < 1e-6


def test_orient_world_support_plane_rejects_non_support_plane():
    with pytest.raises(RuntimeError, match="cannot be oriented as a support surface"):
        _orient_world_support_plane(
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            1.6,
            np.zeros((3,), dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )


def test_geometry_fusion_preserves_metric_trajectory_when_configured(tmp_path, monkeypatch):
    store = ResourceStore("geometry_fusion_preserve_metric", root=tmp_path)
    frame_index = 0

    store.save_frame(
        FrameData(
            frame_id="000000",
            index=frame_index,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "unit-test"},
        )
    )
    store.save_depth(
        DepthData(
            frame_index=frame_index,
            depth=np.full((2, 2), 2.0, dtype=np.float32),
            confidence=np.full((2, 2), 0.9, dtype=np.float32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_semantics2d(
        SemanticsData(
            frame_index=frame_index,
            segments=[],
            segment_ids=np.zeros((2, 2), dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=frame_index,
            height_m=1.6,
            metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
        )
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0.0, 1.6, 0.0], dtype=np.float32)
    store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=frame_index,
                    camera_to_world=c2w,
                    world_to_camera=np.linalg.inv(c2w),
                    metadata={"source": "unit-test"},
                )
            ],
            metadata={"source": "unit-test", "metric_scale": True},
        )
    )

    rect_result = FrameRectificationResult(
        frame_index=frame_index,
        normal_cam=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        offset_cam=1.6,
        implied_height_m=1.6,
        scale=2.0,
        bias=0.3,
        inlier_ratio=1.0,
        residual_p90_m=0.0,
        support_count=4,
    )

    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "fit_per_frame_planes",
        lambda *args, **kwargs: [rect_result],
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "optimize_temporal_smoothness",
        lambda results, *args, **kwargs: results,
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "check_plateau_refit_needed",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "assess_quality",
        lambda *args, **kwargs: [SimpleNamespace(quality_ok=True)],
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "fit_quadratic_surfaces",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "select_road_pixels",
        lambda **kwargs: SimpleNamespace(
            points_cam=np.array([[0.0, 1.6, 2.0]], dtype=np.float32)
        ),
    )

    def _unexpected_factor_graph(*args, **kwargs):
        raise AssertionError("factor-graph fusion must be skipped for preserved metric trajectories")

    monkeypatch.setattr(
        factor_graph_module,
        "run_factor_graph_fusion",
        _unexpected_factor_graph,
    )

    provider = GeometryFusionProvider(
        {
            "preserve_metric_trajectory": True,
            "factor_graph_enabled": True,
            "quadratic_enabled": False,
        }
    )
    provider.run(store, {})

    with np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True) as data:
        saved_c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
        metadata = data["metadata"].item()
    np.testing.assert_allclose(saved_c2w[0], c2w)
    assert metadata["metric_scale"] is True
    assert metadata["scale_source"] == "geometry_fusion"
    assert metadata["trajectory_scale_mode"] == "preserved_metric_input"
    assert metadata["global_dpvo_scale"] == 1.0

    corrected_depth = store.load_depth(frame_index)
    np.testing.assert_allclose(corrected_depth.depth, np.full((2, 2), 4.3, dtype=np.float32))
    assert corrected_depth.metadata["metric_scale"] is True
    assert corrected_depth.metadata["scale_source"] == "geometry_fusion"

    plane = store.load_road_plane(frame_index)
    assert plane.metadata["source"] == "geometry_fusion"
    assert plane.metadata["enforce_height_anchor"] is True


def test_geometry_fusion_auto_verifies_metric_gt_inputs_without_rewriting_geometry(tmp_path, monkeypatch):
    store = ResourceStore("geometry_fusion_metric_verify_only", root=tmp_path)
    frame_index = 0

    store.save_frame(
        FrameData(
            frame_id="000000",
            index=frame_index,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "unit-test"},
        )
    )
    original_depth = np.full((2, 2), 2.0, dtype=np.float32)
    store.save_depth(
        DepthData(
            frame_index=frame_index,
            depth=original_depth.copy(),
            confidence=np.full((2, 2), 0.9, dtype=np.float32),
            metadata={"source": "carla"},
        )
    )
    store.save_semantics2d(
        SemanticsData(
            frame_index=frame_index,
            segments=[],
            segment_ids=np.zeros((2, 2), dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=frame_index,
            height_m=1.6,
            metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
        )
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0.0, 1.6, 0.0], dtype=np.float32)
    store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=frame_index,
                    camera_to_world=c2w,
                    world_to_camera=np.linalg.inv(c2w),
                    metadata={"source": "carla"},
                )
            ],
            metadata={"source": "carla", "metric_scale": True},
        )
    )

    rect_result = FrameRectificationResult(
        frame_index=frame_index,
        normal_cam=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        offset_cam=1.6,
        implied_height_m=1.6,
        scale=1.0,
        bias=0.0,
        inlier_ratio=1.0,
        residual_p90_m=0.0,
        support_count=4,
    )

    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "fit_per_frame_planes",
        lambda *args, **kwargs: [rect_result],
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "assess_quality",
        lambda *args, **kwargs: [SimpleNamespace(quality_ok=True, frame_index=frame_index)],
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("verify-only GT geometry path should not run this step")

    monkeypatch.setattr(geometry_fusion_provider_module, "optimize_temporal_smoothness", _unexpected)
    monkeypatch.setattr(geometry_fusion_provider_module, "check_plateau_refit_needed", _unexpected)
    monkeypatch.setattr(geometry_fusion_provider_module, "fit_quadratic_surfaces", _unexpected)
    monkeypatch.setattr(factor_graph_module, "run_factor_graph_fusion", _unexpected)
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "resolve_joint_consistent_global_scale",
        lambda *args, **kwargs: (
            1.0,
            {
                "source": "joint_depth_trajectory_height_consistency",
                "selected_scale": 1.0,
                "selected_road_consistency": {
                    "global_plane_residual_p90_m": 0.0,
                    "camera_height_median_abs_err_m": 0.0,
                    "frame_diagnostics": [],
                },
            },
        ),
    )

    provider = GeometryFusionProvider({})
    provider.run(
        store,
        {
            "geometry_validation": {
                "check_plane_anchor_consistency": True,
                "plane_anchor_tolerance_m": 0.05,
                "check_road_plane_residual_consistency": False,
                "write_visualizations": False,
            }
        },
    )

    with np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True) as data:
        saved_c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
        metadata = data["metadata"].item()
    np.testing.assert_allclose(saved_c2w[0], c2w)
    assert metadata == {"source": "carla", "metric_scale": True}

    saved_depth = store.load_depth(frame_index)
    np.testing.assert_allclose(saved_depth.depth, original_depth)
    assert saved_depth.metadata == {"source": "carla"}

    plane = store.load_road_plane(frame_index)
    assert plane.metadata["source"] == "geometry_fusion"

    summary = json.loads((store.raw_root / "geometry_fusion" / "summary.json").read_text(encoding="utf-8"))
    assert summary["trajectory_scale_mode"] == "joint_metric_input_verified"
    assert summary["scale_diagnostics_source"] == "joint_depth_trajectory_height_consistency"
    assert summary["verification_checked_frames"] == 1


def test_geometry_fusion_propagates_low_confidence_scale_metadata(tmp_path, monkeypatch):
    store = ResourceStore("geometry_fusion_low_confidence", root=tmp_path)
    frame_index = 0

    store.save_frame(
        FrameData(
            frame_id="000000",
            index=frame_index,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "unit-test"},
        )
    )
    store.save_depth(
        DepthData(
            frame_index=frame_index,
            depth=np.full((2, 2), 2.0, dtype=np.float32),
            confidence=np.full((2, 2), 0.9, dtype=np.float32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_semantics2d(
        SemanticsData(
            frame_index=frame_index,
            segments=[],
            segment_ids=np.zeros((2, 2), dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=frame_index,
            height_m=1.6,
            metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
        )
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0.0, 1.6, 0.0], dtype=np.float32)
    traj = PoseData(
        samples=[
            PoseSample(
                frame_index=frame_index,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"source": "unit-test"},
            )
        ],
        metadata={"source": "DPVO", "metric_scale": False},
    )
    store.save_trajectory(traj)

    rect_result = FrameRectificationResult(
        frame_index=frame_index,
        normal_cam=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        offset_cam=1.6,
        implied_height_m=1.6,
        scale=1.0,
        bias=0.0,
        inlier_ratio=1.0,
        residual_p90_m=0.0,
        support_count=4,
    )

    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "fit_per_frame_planes",
        lambda *args, **kwargs: [rect_result],
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "optimize_temporal_smoothness",
        lambda results, *args, **kwargs: results,
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "check_plateau_refit_needed",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "assess_quality",
        lambda *args, **kwargs: [SimpleNamespace(quality_ok=True, frame_index=frame_index)],
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "fit_quadratic_surfaces",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "select_road_pixels",
        lambda **kwargs: SimpleNamespace(
            points_cam=np.array([[0.0, 1.6, 2.0]], dtype=np.float32)
        ),
    )

    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "estimate_windowed_dpvo_local_scale",
        lambda *args, **kwargs: {
            "source": "windowed_local_scale_field",
            "global_scale": 2.0,
            "frame_local_scale_ratios": {frame_index: 1.0},
            "window_count": 1,
            "confident_window_count": 0,
            "low_confidence_ratio": 1.0,
            "degraded_mode": True,
            "metadata_flags_to_apply": {
                "metric_scale_low_confidence": True,
                "scale_confidence": "low",
            },
        },
    )
    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "apply_global_scale",
        lambda poses, scale: poses,
    )
    monkeypatch.setattr(
        factor_graph_module,
        "run_factor_graph_fusion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("factor-graph should not run in this test")
        ),
    )

    provider = GeometryFusionProvider(
        {
            "factor_graph_enabled": False,
            "quadratic_enabled": False,
        }
    )
    provider.run(store, {})

    with np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True) as data:
        metadata = data["metadata"].item()
    assert metadata["metric_scale_low_confidence"] is True
    assert metadata["scale_confidence"] == "low"
    assert metadata["trajectory_scale_mode"] == "windowed_local_scale_degraded"

    depth = store.load_depth(frame_index)
    assert depth.metadata["metric_scale_low_confidence"] is True
    assert depth.metadata["local_metric_scale_ratio"] == pytest.approx(1.0)


def test_geometry_fusion_writes_failure_diagnostics_for_road_selection_errors(
    tmp_path, monkeypatch
):
    store = ResourceStore("geometry_fusion_failure_diag", root=tmp_path)
    frame_index = 0

    store.save_frame(
        FrameData(
            frame_id="000000",
            index=frame_index,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "unit-test"},
        )
    )
    store.save_depth(
        DepthData(
            frame_index=frame_index,
            depth=np.full((2, 2), 2.0, dtype=np.float32),
            confidence=np.full((2, 2), 0.9, dtype=np.float32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_semantics2d(
        SemanticsData(
            frame_index=frame_index,
            segments=[],
            segment_ids=np.zeros((2, 2), dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=frame_index,
            height_m=1.6,
            metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
        )
    )
    store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=frame_index,
                    camera_to_world=np.eye(4, dtype=np.float32),
                    world_to_camera=np.eye(4, dtype=np.float32),
                    metadata={"source": "unit-test"},
                )
            ],
            metadata={"source": "unit-test", "metric_scale": True},
        )
    )

    def _raise_failure(*args, **kwargs):
        raise road_pixel_selection_module.RoadPixelSelectionError(
            "Geometry fusion frame 0: no road labels resolved from semantics. "
            "configured road_labels=['road']; available_labels=['roads', 'roadlines'].",
            diagnostic_payload={
                "frame_index": 0,
                "configured_road_labels": ["road"],
                "available_semantic_label_names": ["roads", "roadlines"],
            },
        )

    monkeypatch.setattr(
        geometry_fusion_provider_module,
        "fit_per_frame_planes",
        _raise_failure,
    )

    provider = GeometryFusionProvider(
        {
            "factor_graph_enabled": False,
            "quadratic_enabled": False,
        }
    )

    with pytest.raises(RuntimeError, match="Diagnostic written to"):
        provider.run(store, {})

    diag_path = store.raw_root / "geometry_fusion" / "failure_diagnostics.json"
    assert diag_path.exists()
    report = json.loads(diag_path.read_text(encoding="utf-8"))
    assert report["stage"] == "road_rectification"
    assert report["road_selection"]["configured_road_labels"] == ["road"]
    assert report["road_selection"]["available_semantic_label_names"] == [
        "roads",
        "roadlines",
    ]
    assert report["semantics_npz_path"].endswith("standard/semantics_2d/000000.npz")


def test_geometry_fusion_cross_run_cache_spec_requires_outputs(tmp_path) -> None:
    store = ResourceStore("geometry_fusion_cache_spec", root=tmp_path)
    frame_index = 0
    store.save_frame(
        FrameData(
            frame_id="000000",
            index=frame_index,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "unit-test"},
        )
    )
    store.save_depth(
        DepthData(
            frame_index=frame_index,
            depth=np.ones((2, 2), dtype=np.float32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_semantics2d(
        SemanticsData(
            frame_index=frame_index,
            segments=[],
            segment_ids=np.zeros((2, 2), dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=frame_index,
            height_m=1.6,
            metadata={"source": "unit-test"},
        )
    )
    c2w = np.eye(4, dtype=np.float32)
    store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=frame_index,
                    camera_to_world=c2w,
                    world_to_camera=np.linalg.inv(c2w),
                    metadata={"source": "unit-test"},
                )
            ],
            metadata={"source": "unit-test"},
        )
    )

    cache = CrossRunCacheManager(tmp_path / "cache")
    provider = GeometryFusionProvider({"factor_graph_enabled": False, "quadratic_enabled": False})
    provider.setup(
        {
            "cross_run_cache": cache,
            "cross_run_cache_stage_settings": {"geometry_fusion": {"enabled": True}},
            "profile_name": "test",
        }
    )
    payload = provider._cross_run_payload(store)
    assert payload is not None
    provider._cache_payload = payload
    provider._cache_signature = cache.signature("geometry_fusion", payload)

    not_ready = provider.get_cross_run_cache_spec(store)
    assert not_ready is not None
    assert not_ready["ready"] is False
    assert not_ready["not_ready_reason"] == "raw-geometry-fusion-missing"

    raw_dir = store.provider_dir("geometry_fusion")
    (raw_dir / "summary.json").write_text("{}", encoding="utf-8")
    store.save_road_plane(
        RoadPlaneData(
            frame_index=frame_index,
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            offset=-1.6,
            metadata={"source": "geometry_fusion"},
        )
    )

    ready = provider.get_cross_run_cache_spec(store)
    assert ready is not None
    assert ready["ready"] is True
    assert "standard/trajectory/poses.npz" in ready["artifacts"]
    assert any(path.startswith("standard/road_plane/") for path in ready["artifacts"])


def test_geometry_fusion_cross_run_payload_is_stable_for_equivalent_rewritten_npz_inputs(
    tmp_path,
) -> None:
    cache = CrossRunCacheManager(tmp_path / "cache")
    signatures: list[str] = []

    for run_name in ("run_a", "run_b"):
        store = ResourceStore(run_name, root=tmp_path)
        frame_index = 0
        store.save_frame(
            FrameData(
                frame_id="000000",
                index=frame_index,
                image=np.zeros((2, 2, 3), dtype=np.uint8),
            )
        )
        store.save_intrinsics(
            IntrinsicsData(
                matrix=np.array(
                    [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                metadata={"source": "unit-test"},
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_index,
                depth=np.ones((2, 2), dtype=np.float32),
                metadata={"source": "unit-test"},
            )
        )
        store.save_semantics2d(
            SemanticsData(
                frame_index=frame_index,
                segments=[],
                segment_ids=np.zeros((2, 2), dtype=np.int32),
                metadata={"source": "unit-test"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_index,
                height_m=1.6,
                metadata={"source": "unit-test"},
            )
        )
        c2w = np.eye(4, dtype=np.float32)
        store.save_trajectory(
            PoseData(
                samples=[
                    PoseSample(
                        frame_index=frame_index,
                        camera_to_world=c2w,
                        world_to_camera=np.linalg.inv(c2w),
                        metadata={"source": "unit-test"},
                    )
                ],
                metadata={"source": "unit-test"},
            )
        )

        provider = GeometryFusionProvider({"factor_graph_enabled": False, "quadratic_enabled": False})
        provider.setup(
            {
                "cross_run_cache": cache,
                "cross_run_cache_stage_settings": {"geometry_fusion": {"enabled": True}},
                "profile_name": "test",
            }
        )
        payload = provider._cross_run_payload(store)
        assert payload is not None
        signatures.append(cache.signature("geometry_fusion", payload))

    assert signatures[0] == signatures[1]
