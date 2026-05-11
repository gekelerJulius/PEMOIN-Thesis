from __future__ import annotations

import numpy as np
import pytest

from pemoin.data.contracts import (
    DepthData,
    DynamicMaskData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceStore,
    SemanticSegment,
    SemanticsData,
)
import pemoin.validation.depth_pose_consistency as dpc
from pemoin.validation import (
    GeometryConsistencyValidationSettings,
    validate_depth_pose_intrinsics_consistency,
)


def _build_store(tmp_path, *, run_name: str, bad_frames: set[int]) -> ResourceStore:
    store = ResourceStore(run_name, root=tmp_path)
    k = np.array([[400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    store.save_intrinsics(IntrinsicsData(matrix=k, distortion=None, metadata={"source": "unit-test"}))

    samples = []
    for frame_idx in range(1, 9):
        c2w = np.eye(4, dtype=np.float32)
        c2w[1, 3] = float(frame_idx - 1) * 0.25
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender"},
            )
        )
        depth_val = 10.0 if frame_idx not in bad_frames else 60.0
        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((64, 64), depth_val, dtype=np.float32),
                metadata={"source": "unit-test"},
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata={"source": "unit-test"}))
    return store


def _base_settings(**overrides: object) -> GeometryConsistencyValidationSettings:
    values = {
        "enabled": True,
        "pixel_stride": 8,
        "min_overlap_points": 50,
        "min_static_overlap_points": 50,
        "reprojection_error_px": 2.0,
        "max_reprojection_rmse_px": 6.0,
        "max_reprojection_p90_px": 4.0,
        "max_reprojection_p95_px": 6.0,
        "min_inlier_ratio": 0.7,
        "max_depth_scale_drift": 0.05,
        "max_consecutive_catastrophic": 1,
        "max_skipped_frames": 3,
    }
    values.update(overrides)
    return GeometryConsistencyValidationSettings(**values)


def test_geometry_consistency_skips_up_to_three_frames(tmp_path):
    store = _build_store(tmp_path, run_name="geom_consistency_skip3", bad_frames={1, 8})
    context = {}
    result = validate_depth_pose_intrinsics_consistency(
        store,
        settings=_base_settings(max_skipped_frames=1),
        context=context,
    )
    assert result.status == "degraded"
    assert result.summary["replacement_budget_exceeded"] is True
    assert result.skipped_frames in {(1, 7), (1, 8)}
    assert "geometry_consistency_replacement_map" in context
    assert (store.visualizations_dir("geometry_consistency") / "summary.json").exists()


def test_geometry_consistency_ignores_rmse_tail_when_robust_metrics_are_healthy(tmp_path, monkeypatch):
    store = _build_store(tmp_path, run_name="geom_consistency_rmse_tail", bad_frames=set())

    def fake_metrics(**_: object) -> tuple[float, float, float, float, float, float, int, int]:
        return (12.0, 0.8, 2.5, 4.0, 0.96, 1.0, 220, 220)

    monkeypatch.setattr(dpc, "_estimate_pair_metrics", fake_metrics)

    result = validate_depth_pose_intrinsics_consistency(
        store,
        settings=_base_settings(),
        context={},
    )

    assert result.status == "ok"
    assert result.summary["num_catastrophic_pairs"] == 0


def test_geometry_consistency_uses_dynamic_masks_when_available(tmp_path, monkeypatch):
    store = _build_store(tmp_path, run_name="geom_consistency_dynamic_mask", bad_frames=set())
    for frame_idx in range(1, 9):
        mask = np.ones((64, 64), dtype=bool)
        mask[:, 48:] = False
        store.save_dynamic_mask(
            DynamicMaskData(
                frame_index=frame_idx,
                mask=mask,
                dynamic_classes=("car",),
                metadata={"source": "unit-test"},
            )
        )

    def fake_metrics(**kwargs: object) -> tuple[float, float, float, float, float, float, int, int]:
        assert kwargs["static_mask_a"] is not None
        assert kwargs["static_mask_b"] is not None
        return (1.0, 0.5, 1.0, 1.4, 0.98, 1.0, 220, 220)

    monkeypatch.setattr(dpc, "_estimate_pair_metrics", fake_metrics)

    result = validate_depth_pose_intrinsics_consistency(
        store,
        settings=_base_settings(exclude_dynamic_pixels=True, dynamic_mask_source="auto"),
        context={},
    )

    assert result.status == "ok"
    assert "dynamic_mask" in result.summary["dynamic_mask_sources_used"]


def test_geometry_consistency_falls_back_to_semantics_mobile_mask(tmp_path):
    store = _build_store(tmp_path, run_name="geom_consistency_semantics_mask", bad_frames=set())
    segment_ids = np.zeros((8, 8), dtype=np.int32)
    label_ids = np.zeros((8, 8), dtype=np.int32)
    label_ids[:, 4:] = 1
    store.save_semantics2d(
        SemanticsData(
            frame_index=1,
            frame_id="000001",
            segments=[
                SemanticSegment(segment_id=0, label="road", score=1.0, label_id=0, mask=label_ids == 0),
                SemanticSegment(segment_id=1, label="car", score=1.0, label_id=1, mask=label_ids == 1),
            ],
            segment_ids=segment_ids,
            label_ids=label_ids,
            metadata={
                "class_id_to_label": {0: "road", 1: "car"},
                "semantic_roles": {"mobile": ["car"]},
            },
        )
    )
    mask, source = dpc._resolve_static_mask(
        store,
        1,
        settings=_base_settings(dynamic_mask_source="semantics_mobile"),
        semantics_tool=None,
        expected_shape=(8, 8),
        cache={},
    )
    assert source == "semantics_mobile"
    assert mask is not None
    assert bool(mask[0, 0]) is True
    assert bool(mask[0, 7]) is False


def test_geometry_consistency_degrades_recoverable_catastrophic_pairs(tmp_path, monkeypatch):
    store = _build_store(tmp_path, run_name="geom_consistency_recoverable", bad_frames=set())
    metrics_by_pair = {
        (1, 2): (5.0, 2.5, 5.2, 7.5, 0.88, 1.0, 220, 220),
        (2, 3): (5.2, 2.7, 5.4, 7.8, 0.87, 1.0, 220, 220),
    }

    def fake_metrics(**kwargs: object) -> tuple[float, float, float, float, float, float, int, int]:
        return metrics_by_pair.get((int(kwargs["c2w_a"][1, 3] / 0.25) + 1, int(kwargs["c2w_b"][1, 3] / 0.25) + 1), (1.0, 0.5, 1.0, 1.2, 0.98, 1.0, 220, 220))

    monkeypatch.setattr(dpc, "_estimate_pair_metrics", fake_metrics)

    result = validate_depth_pose_intrinsics_consistency(
        store,
        settings=_base_settings(max_consecutive_catastrophic=1),
        context={},
    )

    assert result.status == "degraded"
    assert result.summary["num_recoverable_catastrophic_pairs"] == 2
    assert result.summary["num_severe_catastrophic_pairs"] == 0


def test_geometry_consistency_fails_on_severe_contiguous_run(tmp_path, monkeypatch):
    store = _build_store(tmp_path, run_name="geom_consistency_fail", bad_frames=set())

    def fake_metrics(**_: object) -> tuple[float, float, float, float, float, float, int, int]:
        return (10.0, 6.0, 12.0, 16.0, 0.3, 1.0, 220, 220)

    monkeypatch.setattr(dpc, "_estimate_pair_metrics", fake_metrics)

    with pytest.raises(RuntimeError, match="severe catastrophic pair run length"):
        validate_depth_pose_intrinsics_consistency(
            store,
            settings=_base_settings(max_consecutive_catastrophic=1),
            context={},
        )


def test_geometry_consistency_ok_when_clean(tmp_path):
    store = _build_store(tmp_path, run_name="geom_consistency_ok", bad_frames=set())
    result = validate_depth_pose_intrinsics_consistency(
        store,
        settings=_base_settings(),
        context={},
    )
    assert result.status == "ok"
    assert result.skipped_frames == ()
    assert result.replacement_map == {}


def test_geometry_consistency_adapts_thresholds_for_low_fps(tmp_path, monkeypatch):
    store = _build_store(tmp_path, run_name="geom_consistency_low_fps", bad_frames=set())

    def fake_metrics(**_: object) -> tuple[float, float, float, float, float, float, int, int]:
        return (5.0, 2.0, 5.1, 6.2, 0.6, 1.08, 220, 40)

    monkeypatch.setattr(dpc, "_estimate_pair_metrics", fake_metrics)

    result = validate_depth_pose_intrinsics_consistency(
        store,
        settings=_base_settings(
            min_static_overlap_points=50,
            max_reprojection_p90_px=4.0,
            max_reprojection_p95_px=6.0,
            min_inlier_ratio=0.7,
            max_depth_scale_drift=0.05,
        ),
        context={
            "validation_policy": {"enabled": True, "reference_sampling_fps": 10.0},
            "frame_provider_info": {"tool": "test", "settings": {"sampling_fps": 4.0}},
        },
    )

    assert result.status == "degraded"
    assert result.summary["validation_policy"]["enabled"] is True
    assert result.summary["threshold_reprojection_p90_px"] > 4.0
    assert result.summary["threshold_min_inlier_ratio"] < 0.7
