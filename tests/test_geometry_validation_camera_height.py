from __future__ import annotations

import numpy as np
import pytest

from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    FrameData,
    IntrinsicsData,
    PointCloud3DData,
    PoseData,
    PoseSample,
    ResourceStore,
    RoadPlaneData,
    RoadPlaneSupportData,
)
from pemoin.utils.geometry_validation import (
    GeometryValidationConfig,
    GeometryValidationError,
    _validate_optional_road_geometry_consistency,
    _validate_camera_height,
    validate_road_plane_anchor_consistency,
    validate_geometry_store,
)


def test_camera_height_validation_accounts_for_grounding_shift(tmp_path):
    store = ResourceStore("geom_val_grounding_shift", root=tmp_path)
    store.save_camera_height(
        CameraHeightData(
            frame_index=1,
            height_m=1.595,
            metadata={"axis": "z", "world_coordinate_system": "blender", "absolute": False},
        )
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 0.305

    cfg = GeometryValidationConfig(camera_height_tolerance_m=0.05)
    _validate_camera_height(store, 1, c2w, cfg, grounding_shift_z_m=-1.286)


def test_camera_height_validation_fails_without_grounding_shift(tmp_path):
    store = ResourceStore("geom_val_no_grounding_shift", root=tmp_path)
    store.save_camera_height(
        CameraHeightData(
            frame_index=1,
            height_m=1.595,
            metadata={"axis": "z", "world_coordinate_system": "blender", "absolute": False},
        )
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 0.305

    cfg = GeometryValidationConfig(camera_height_tolerance_m=0.05)
    with pytest.raises(GeometryValidationError, match="Camera height mismatch"):
        _validate_camera_height(store, 1, c2w, cfg, grounding_shift_z_m=0.0)


def test_camera_height_validation_uses_road_plane_anchor_for_canonical_mode(tmp_path):
    store = ResourceStore("geom_val_road_plane_anchor_mode", root=tmp_path)
    store.save_camera_height(
        CameraHeightData(
            frame_index=1,
            height_m=1.511,
            metadata={"axis": "z", "world_coordinate_system": "blender", "absolute": False},
        )
    )
    store.save_road_plane(
        RoadPlaneData(
            frame_index=1,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            offset=0.657,
            metadata={"source": "unit-test"},
        )
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 0.854

    cfg = GeometryValidationConfig(camera_height_tolerance_m=0.05, plane_anchor_tolerance_m=0.05)
    _validate_camera_height(
        store,
        1,
        c2w,
        cfg,
        trajectory_metadata={"height_fit_validation_mode": "road_plane_anchor"},
    )


def test_camera_height_validation_uses_road_plane_anchor_for_comparison_frame_metadata(tmp_path):
    store = ResourceStore("geom_val_comparison_frame_anchor_mode", root=tmp_path)
    store.save_camera_height(
        CameraHeightData(
            frame_index=1,
            height_m=1.511,
            metadata={"axis": "z", "world_coordinate_system": "blender", "absolute": False},
        )
    )
    store.save_road_plane(
        RoadPlaneData(
            frame_index=1,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            offset=0.657,
            metadata={"source": "unit-test"},
        )
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 0.854

    cfg = GeometryValidationConfig(camera_height_tolerance_m=0.05, plane_anchor_tolerance_m=0.05)
    _validate_camera_height(
        store,
        1,
        c2w,
        cfg,
        trajectory_metadata={"comparison_frame": {"enabled": True, "mode": "estimated"}},
    )


def _build_anchor_check_store(
    tmp_path: str,
    *,
    enforce_height_anchor: bool,
    metric_scale: bool,
) -> tuple[ResourceStore, np.ndarray]:
    store = ResourceStore("geom_val_anchor_nonmetric", root=tmp_path)
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array([[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            distortion=None,
            metadata={"source": "unit-test"},
        )
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    c2w = np.eye(4, dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    store.save_frame(FrameData(frame_id="000001", index=1, image=image))
    store.save_trajectory(
        PoseData(
            samples=[PoseSample(frame_index=1, camera_to_world=c2w, world_to_camera=w2c)],
            metadata={"source": "unit-test", "metric_scale": bool(metric_scale)},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=1,
            height_m=1.595,
            metadata={"axis": "z", "world_coordinate_system": "blender"},
        )
    )
    store.save_road_plane(
        RoadPlaneData(
            frame_index=1,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            offset=0.474,
            metadata={
                "enforce_height_anchor": enforce_height_anchor,
                "residual_p90": 0.0,
                "alignment_transform_id": "base",
            },
        )
    )
    store.save_point_cloud_3d(
        PointCloud3DData(
            points_world=np.array([[0.0, 0.0, -5.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            label_confidences=np.array([1.0], dtype=np.float32),
            colors=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            label_names={1: "road"},
            observation_counts=np.array([1], dtype=np.int32),
            metadata={},
        )
    )
    return store, c2w


def test_anchor_consistency_skipped_for_pre_metric_unanchored_planes(tmp_path):
    store, c2w = _build_anchor_check_store(tmp_path, enforce_height_anchor=False, metric_scale=False)
    cfg = GeometryValidationConfig(
        check_plane_anchor_consistency=True,
        check_road_plane_residual_consistency=False,
        point_cloud_vertical_check_max_frames=1,
    )
    _validate_optional_road_geometry_consistency(
        store=store,
        cfg=cfg,
        sample_frames=[1],
        traj_index={1: 0},
        c2w_all=np.asarray([c2w], dtype=np.float32),
    )


def test_anchor_consistency_kept_for_metric_plane_mode(tmp_path):
    store, c2w = _build_anchor_check_store(tmp_path, enforce_height_anchor=True, metric_scale=True)
    cfg = GeometryValidationConfig(
        check_plane_anchor_consistency=True,
        check_road_plane_residual_consistency=False,
        point_cloud_vertical_check_max_frames=1,
    )
    with pytest.raises(GeometryValidationError, match="Road-plane anchor consistency failed"):
        _validate_optional_road_geometry_consistency(
            store=store,
            cfg=cfg,
            sample_frames=[1],
            traj_index={1: 0},
            c2w_all=np.asarray([c2w], dtype=np.float32),
        )


def test_anchor_consistency_enforced_for_metric_even_if_unanchored_metadata(tmp_path):
    store, c2w = _build_anchor_check_store(tmp_path, enforce_height_anchor=False, metric_scale=True)
    cfg = GeometryValidationConfig(
        check_plane_anchor_consistency=True,
        check_road_plane_residual_consistency=False,
        point_cloud_vertical_check_max_frames=1,
    )
    with pytest.raises(GeometryValidationError, match="Road-plane anchor consistency failed"):
        _validate_optional_road_geometry_consistency(
            store=store,
            cfg=cfg,
            sample_frames=[1],
            traj_index={1: 0},
            c2w_all=np.asarray([c2w], dtype=np.float32),
        )


def test_validate_road_plane_anchor_consistency_returns_summary(tmp_path):
    store, _ = _build_anchor_check_store(tmp_path, enforce_height_anchor=True, metric_scale=True)
    store.save_road_plane(
        RoadPlaneData(
            frame_index=1,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            offset=1.595,
            metadata={
                "enforce_height_anchor": True,
                "residual_p90": 0.0,
                "alignment_transform_id": "base",
            },
        )
    )
    summary = validate_road_plane_anchor_consistency(
        store,
        config=GeometryValidationConfig(
            plane_anchor_tolerance_m=0.05,
            check_road_plane_residual_consistency=False,
            write_visualizations=False,
        ),
    )
    assert summary["checked_frames"] == 1
    assert summary["max_abs_anchor_delta_m"] == pytest.approx(0.0, abs=1e-6)


def test_validate_road_plane_anchor_consistency_fails_on_mismatch(tmp_path):
    store, _ = _build_anchor_check_store(tmp_path, enforce_height_anchor=True, metric_scale=True)
    with pytest.raises(GeometryValidationError, match="Road-plane anchor consistency failed"):
        validate_road_plane_anchor_consistency(
            store,
            config=GeometryValidationConfig(
                plane_anchor_tolerance_m=0.05,
                check_road_plane_residual_consistency=False,
                write_visualizations=False,
            ),
        )


def test_road_plane_residual_consistency_skips_frames_without_standardized_support(tmp_path):
    store, c2w = _build_anchor_check_store(tmp_path, enforce_height_anchor=True, metric_scale=True)
    cfg = GeometryValidationConfig(
        check_plane_anchor_consistency=False,
        check_road_plane_residual_consistency=True,
        point_cloud_vertical_check_max_frames=1,
    )
    _validate_optional_road_geometry_consistency(
        store=store,
        cfg=cfg,
        sample_frames=[1],
        traj_index={1: 0},
        c2w_all=np.asarray([c2w], dtype=np.float32),
    )


def test_road_plane_residual_consistency_uses_standardized_support_when_present(tmp_path):
    store, c2w = _build_anchor_check_store(tmp_path, enforce_height_anchor=True, metric_scale=True)
    store.save_road_plane_support(
        RoadPlaneSupportData(
            frame_index=1,
            points_world=np.array([[0.0, 0.0, -0.474]], dtype=np.float32),
            diagnostics={"alignment_transform_id": "base"},
            metadata={"source": "unit-test"},
        )
    )
    cfg = GeometryValidationConfig(
        check_plane_anchor_consistency=False,
        check_road_plane_residual_consistency=True,
        point_cloud_vertical_check_max_frames=1,
    )
    _validate_optional_road_geometry_consistency(
        store=store,
        cfg=cfg,
        sample_frames=[1],
        traj_index={1: 0},
        c2w_all=np.asarray([c2w], dtype=np.float32),
    )


def test_point_cloud_vertical_plausibility_uses_local_road_subset(tmp_path, monkeypatch):
    store = ResourceStore("geom_val_local_road_subset", root=tmp_path)
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array([[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            distortion=None,
            metadata={"source": "unit-test"},
        )
    )
    store.save_frame(FrameData(frame_id="000001", index=1, image=np.zeros((64, 64, 3), dtype=np.uint8)))
    c2w = np.eye(4, dtype=np.float32)
    store.save_trajectory(
        PoseData(
            samples=[PoseSample(frame_index=1, camera_to_world=c2w, world_to_camera=np.eye(4, dtype=np.float32))],
            metadata={"source": "unit-test", "metric_scale": True},
        )
    )
    store.save_road_plane(
        RoadPlaneData(
            frame_index=1,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            offset=1.0,
            metadata={
                "enforce_height_anchor": True,
                "residual_p90": 0.0,
                "alignment_transform_id": "base",
            },
        )
    )
    local_points = np.column_stack(
        [
            np.linspace(1.0, 4.0, 600, dtype=np.float32),
            np.zeros(600, dtype=np.float32),
            np.full(600, -1.0, dtype=np.float32),
        ]
    )
    distant_points = np.column_stack(
        [
            np.linspace(20.0, 30.0, 1000, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
            np.full(1000, 0.25, dtype=np.float32),
        ]
    )
    points = np.concatenate([local_points, distant_points], axis=0)
    store.save_point_cloud_3d(
        PointCloud3DData(
            points_world=points,
            labels=np.full(points.shape[0], 1, dtype=np.int32),
            label_confidences=np.ones(points.shape[0], dtype=np.float32),
            colors=np.zeros((points.shape[0], 3), dtype=np.uint8),
            label_names={1: "road"},
            observation_counts=np.full(points.shape[0], 2, dtype=np.int32),
            metadata={},
        )
    )

    monkeypatch.setattr(
        "pemoin.utils.geometry_validation.project_world_to_image",
        lambda *args, **kwargs: (np.zeros((points.shape[0], 2), dtype=np.float32), np.ones(points.shape[0], dtype=bool)),
    )

    cfg = GeometryValidationConfig(
        check_plane_anchor_consistency=False,
        check_road_plane_residual_consistency=False,
        point_cloud_vertical_check_max_frames=1,
        point_cloud_road_local_radius_m=10.0,
        point_cloud_road_local_min_points=100,
    )
    _validate_optional_road_geometry_consistency(
        store=store,
        cfg=cfg,
        sample_frames=[1],
        traj_index={1: 0},
        c2w_all=np.asarray([c2w], dtype=np.float32),
    )


def _build_minimal_validation_store(tmp_path: str, *, upside_down: bool) -> ResourceStore:
    store = ResourceStore("geom_val_canonical_up", root=tmp_path)
    store.save_frame(FrameData(frame_id="000001", index=1, image=np.zeros((32, 32, 3), dtype=np.uint8)))
    store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array([[25.0, 0.0, 16.0], [0.0, 25.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            distortion=None,
            metadata={"camera_convention": "blender"},
        )
    )
    store.save_depth(
        DepthData(
            frame_index=1,
            depth=np.full((32, 32), 5.0, dtype=np.float32),
            metadata={"camera_convention": "blender", "metric_scale": True},
        )
    )
    c2w = np.eye(4, dtype=np.float32)
    if upside_down:
        c2w[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
    else:
        c2w[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    store.save_trajectory(
        PoseData(
            samples=[PoseSample(frame_index=1, camera_to_world=c2w, world_to_camera=np.linalg.inv(c2w))],
            metadata={
                "camera_convention": "blender",
                "metric_scale": True,
                "canonical_world_frame": True,
            },
        )
    )
    return store


def test_geometry_validation_rejects_upside_down_canonical_trajectory(tmp_path):
    store = _build_minimal_validation_store(tmp_path, upside_down=True)
    with pytest.raises(GeometryValidationError, match="Canonical camera-up validation failed"):
        validate_geometry_store(store, config=GeometryValidationConfig(max_frames=1, write_visualizations=False))


def test_geometry_validation_accepts_upright_canonical_trajectory(tmp_path):
    store = _build_minimal_validation_store(tmp_path, upside_down=False)
    validate_geometry_store(store, config=GeometryValidationConfig(max_frames=1, write_visualizations=False))
