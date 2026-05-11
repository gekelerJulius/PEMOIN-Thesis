"""Tests for DPVO match-graph metric scale alignment."""

from __future__ import annotations

import numpy as np
import pytest

from pemoin.data.contracts import (
    DepthData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    SemanticSegment,
    SemanticsData,
    TrajectoryMatchGraphData,
)
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.stages.road_rectification import FrameRectificationResult
from pemoin.providers.geometry_fusion.stages.scale_alignment import (
    apply_global_scale,
    estimate_global_dpvo_scale,
    estimate_windowed_dpvo_local_scale,
)


def _pose(frame_idx: int, x: float) -> PoseSample:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([x, 0.0, 0.0], dtype=np.float32)
    return PoseSample(
        frame_index=frame_idx,
        camera_to_world=c2w,
        world_to_camera=np.linalg.inv(c2w.astype(np.float64)).astype(np.float32),
    )


def _save_simple_run(tmp_path, *, true_scale: float = 20.0) -> tuple[ResourceStore, PoseData, np.ndarray]:
    store = ResourceStore("match_scale", root=tmp_path)
    h, w = 80, 120
    k = np.array([[90.0, 0.0, 60.0], [0.0, 90.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    store.save_intrinsics(IntrinsicsData(matrix=k, metadata={"camera_convention": "blender"}))

    depth = np.full((h, w), 12.0, dtype=np.float32)
    for fi in range(4):
        store.save_depth(DepthData(frame_index=fi, depth=depth, metadata={"metric_scale": True}))
        seg = np.zeros((h, w), dtype=np.int32)
        store.save_semantics2d(
            SemanticsData(
                frame_index=fi,
                segments=[
                    SemanticSegment(
                        segment_id=0,
                        label="road",
                        score=1.0,
                        mask=np.ones((h, w), dtype=bool),
                        label_id=0,
                    )
                ],
                frame_id=f"{fi:06d}",
                segment_ids=seg,
                label_ids=seg,
                metadata={},
            )
        )

    raw_step = 0.1
    poses = PoseData(
        samples=[_pose(i, i * raw_step) for i in range(4)],
        metadata={"source": "DPVO", "metric_scale": False},
    )

    # Build a synthetic match graph consistent with the estimator camera model.
    src_f = []
    tgt_f = []
    patch = []
    src_uv = []
    tgt_uv = []
    weight = []
    for i in range(3):
        for u in range(20, 101, 10):
            for v in range(45, 71, 8):
                z = 12.0
                x_i = (float(u) - float(k[0, 2])) / float(k[0, 0]) * z
                y_i = -((float(v) - float(k[1, 2])) / float(k[1, 1]) * z)
                z_i = -z
                x_j = x_i + true_scale * (-raw_step)
                y_j = y_i
                z_j = z_i
                u_j = float(k[0, 0]) * (x_j / (-z_j)) + float(k[0, 2])
                v_j = float(k[1, 1]) * ((-y_j) / (-z_j)) + float(k[1, 2])
                src_f.append(i)
                tgt_f.append(i + 1)
                patch.append(len(patch))
                src_uv.append([float(u), float(v)])
                tgt_uv.append([u_j, v_j])
                weight.append(1.0)
    store.save_trajectory_match_graph(
        TrajectoryMatchGraphData(
            payload={
                "schema_version": np.int32(2),
                "coord_space": np.array("full_res_pixels"),
                "res_factor": np.int32(4),
                "edge_src_frame_id": np.asarray(src_f, dtype=np.int32),
                "edge_tgt_frame_id": np.asarray(tgt_f, dtype=np.int32),
                "edge_src_node_idx": np.asarray(src_f, dtype=np.int32),
                "edge_tgt_node_idx": np.asarray(tgt_f, dtype=np.int32),
                "edge_patch_idx": np.asarray(patch, dtype=np.int32),
                "src_uv": np.asarray(src_uv, dtype=np.float32),
                "tgt_uv": np.asarray(tgt_uv, dtype=np.float32),
                "edge_weight": np.asarray(weight, dtype=np.float32),
                "edge_timestamp_src": np.asarray(src_f, dtype=np.int64),
                "edge_timestamp_tgt": np.asarray(tgt_f, dtype=np.int64),
            },
            metadata={"source": "unit-test"},
        )
    )
    return store, poses, k


def _rect_stub(n: int) -> list[FrameRectificationResult]:
    return [
        FrameRectificationResult(
            frame_index=i,
            normal_cam=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            offset_cam=1.6,
            implied_height_m=1.6,
            scale=1.0,
            bias=0.0,
            inlier_ratio=1.0,
            residual_p90_m=0.01,
            support_count=1000,
        )
        for i in range(n)
    ]


def test_match_graph_scale_recovers_synthetic_scale(tmp_path):
    store, poses, k = _save_simple_run(tmp_path, true_scale=20.0)
    settings = GeometryFusionSettings(
        dpvo_scale_mode="match_graph_global",
        dpvo_match_min_edges=20,
        dpvo_match_min_unique_frames=3,
        dpvo_match_min_valid_pairs=2,
        dpvo_match_min_edges_per_pair=8,
        dpvo_match_scale_min=1.0,
        dpvo_match_scale_max=60.0,
        dpvo_match_max_median_residual_px=1.0,
        dpvo_match_max_p90_residual_px=2.0,
        dpvo_match_static_filter_enabled=False,
        dpvo_match_debug_overlay_pairs=0,
    )
    scale, diag = estimate_global_dpvo_scale(
        store,
        poses,
        _rect_stub(4),
        k,
        camera_height_m=1.6,
        settings=settings,
    )
    assert scale == pytest.approx(20.0, abs=0.8)
    assert diag["pair_consistency"]["valid_pair_count"] >= 2


def test_match_graph_scale_fails_when_artifact_missing(tmp_path):
    store, poses, k = _save_simple_run(tmp_path, true_scale=20.0)
    store.path_for(ResourceKind.TRAJECTORY_MATCH_GRAPH).unlink()
    settings = GeometryFusionSettings(
        dpvo_scale_mode="match_graph_global",
        dpvo_match_static_filter_enabled=False,
    )
    with pytest.raises(RuntimeError, match="match graph missing"):
        estimate_global_dpvo_scale(
            store,
            poses,
            _rect_stub(4),
            k,
            camera_height_m=1.6,
            settings=settings,
        )


def test_match_graph_scale_fails_on_legacy_schema(tmp_path):
    store, poses, k = _save_simple_run(tmp_path, true_scale=20.0)
    path = store.path_for(ResourceKind.TRAJECTORY_MATCH_GRAPH)
    with np.load(path, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["schema_version"] = np.int32(1)
    np.savez_compressed(path, **payload)

    settings = GeometryFusionSettings(
        dpvo_scale_mode="match_graph_global",
        dpvo_match_static_filter_enabled=False,
    )
    with pytest.raises(RuntimeError, match="expected 2"):
        estimate_global_dpvo_scale(
            store,
            poses,
            _rect_stub(4),
            k,
            camera_height_m=1.6,
            settings=settings,
        )


def test_match_graph_scale_fallback_low_confidence_path(tmp_path):
    store, poses, k = _save_simple_run(tmp_path, true_scale=20.0)
    settings = GeometryFusionSettings(
        dpvo_scale_mode="match_graph_global",
        dpvo_match_min_edges=20,
        dpvo_match_min_unique_frames=3,
        dpvo_match_min_valid_pairs=6,
        dpvo_match_min_edges_per_pair=8,
        dpvo_match_scale_min=1.0,
        dpvo_match_scale_max=60.0,
        dpvo_match_max_median_residual_px=1e-6,  # force strict failure
        dpvo_match_max_p90_residual_px=1e-6,
        dpvo_match_fallback_enabled=True,
        dpvo_match_fallback_min_edges=20,
        dpvo_match_fallback_min_unique_frames=3,
        dpvo_match_fallback_min_valid_pairs=3,
        dpvo_match_fallback_max_median_residual_px=1.0,
        dpvo_match_fallback_max_p90_residual_px=2.0,
        dpvo_match_quality_filter_in_fallback=False,
        dpvo_match_pair_mode="undirected",
        dpvo_match_fallback_allow_low_confidence=True,
        dpvo_match_max_iqr_ratio=10.0,
        dpvo_match_static_filter_enabled=False,
        dpvo_match_debug_overlay_pairs=0,
    )
    _, diag = estimate_global_dpvo_scale(
        store,
        poses,
        _rect_stub(4),
        k,
        camera_height_m=1.6,
        settings=settings,
    )
    assert diag["final_decision"] == "fallback_success_low_confidence"
    assert diag["metadata_flags_to_apply"]["metric_scale_low_confidence"] is True
    assert diag["pair_consistency"]["pair_mode"] == "undirected"


def test_match_graph_scale_persists_diagnostics_on_failure(tmp_path):
    store, poses, k = _save_simple_run(tmp_path, true_scale=20.0)
    settings = GeometryFusionSettings(
        dpvo_scale_mode="match_graph_global",
        dpvo_match_min_edges=999999,  # guaranteed coverage failure
        dpvo_match_static_filter_enabled=False,
    )
    with pytest.raises(RuntimeError):
        estimate_global_dpvo_scale(
            store,
            poses,
            _rect_stub(4),
            k,
            camera_height_m=1.6,
            settings=settings,
        )
    diag_path = store.raw_root / "geometry_fusion" / "dpvo_match_scale_diagnostics.json"
    assert diag_path.exists()
    content = diag_path.read_text(encoding="utf-8")
    assert "\"final_decision\": \"failed\"" in content


def test_match_graph_scale_continues_degraded_for_low_fps_soft_overrun(tmp_path):
    store, poses, k = _save_simple_run(tmp_path, true_scale=20.0)
    settings = GeometryFusionSettings(
        dpvo_scale_mode="match_graph_global",
        dpvo_match_min_edges=180,
        dpvo_match_min_unique_frames=3,
        dpvo_match_min_valid_pairs=2,
        dpvo_match_min_edges_per_pair=8,
        dpvo_match_scale_min=1.0,
        dpvo_match_scale_max=60.0,
        dpvo_match_max_median_residual_px=1.0,
        dpvo_match_max_p90_residual_px=2.0,
        dpvo_match_static_filter_enabled=False,
        dpvo_match_debug_overlay_pairs=0,
    )
    _, diag = estimate_global_dpvo_scale(
        store,
        poses,
        _rect_stub(4),
        k,
        camera_height_m=1.6,
        settings=settings,
        context={
            "validation_policy": {"enabled": True, "reference_sampling_fps": 10.0},
            "frame_provider_info": {"tool": "test", "settings": {"sampling_fps": 4.0}},
        },
    )
    assert diag["final_decision"] in {"strict_success_degraded", "degraded_soft_threshold_exceeded"}
    assert "effective_thresholds" in diag
    assert diag["validation_policy"]["enabled"] is True


def test_apply_global_scale_metadata_and_translation():
    poses = PoseData(samples=[_pose(0, 0.0), _pose(1, 1.0)], metadata={"source": "DPVO"})
    scaled = apply_global_scale(poses, 3.0)
    assert scaled.metadata["scale_source"] == "geometry_fusion_dpvo_match_graph"
    assert scaled.metadata["global_scale_factor"] == pytest.approx(3.0)
    assert scaled.samples[1].camera_to_world[0, 3] == pytest.approx(3.0)


def test_windowed_local_scale_estimator_returns_confident_field(tmp_path):
    store, poses, k = _save_simple_run(tmp_path, true_scale=20.0)
    settings = GeometryFusionSettings(
        dpvo_local_window_size=3,
        dpvo_local_window_overlap=1,
        dpvo_local_window_min_edges=8,
        dpvo_local_window_min_confident_windows=1,
        dpvo_match_min_edge_weight=0.0,
        dpvo_match_scale_min=1.0,
        dpvo_match_scale_max=60.0,
        dpvo_match_static_filter_enabled=False,
        dpvo_match_debug_overlay_pairs=0,
        dpvo_match_gap_hard_max=3,
    )
    diag = estimate_windowed_dpvo_local_scale(
        store,
        poses,
        _rect_stub(4),
        k,
        settings,
    )
    assert diag["source"] == "windowed_local_scale_field"
    assert diag["global_scale"] == pytest.approx(20.0, abs=1.0)
    assert diag["confident_window_count"] >= 1
    assert diag["degraded_mode"] is False
    assert len(diag["frame_local_scales"]) == 4
