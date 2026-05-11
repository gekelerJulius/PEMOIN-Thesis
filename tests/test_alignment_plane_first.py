from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pemoin.coordinate_systems.alignment import (
    AlignmentSettings,
    align_trajectory_to_camera_height,
    canonicalize_metric_geometry_to_camera_height,
    verify_alignment_consistency,
)
from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    RoadPlaneData,
    RoadPlaneSupportData,
)


def _build_store_with_y_up_planes(tmp_path: Path) -> ResourceStore:
    store = ResourceStore("alignment_plane_first_regression", root=tmp_path)

    frame_indices = list(range(1, 7))
    samples = []
    for i, frame_idx in enumerate(frame_indices):
        c2w = np.eye(4, dtype=np.float32)
        # Raw trajectory is Y-up with camera at ~1.6m above road plane y=0.
        c2w[0, 3] = float(i) * 0.5
        c2w[1, 3] = 1.6
        c2w[2, 3] = 0.2 * float(i)
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender"},
            )
        )

        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((4, 4), 5.0, dtype=np.float32),
                metadata={"source": "unit-test"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.6,
                metadata={
                    "source": "unit-test",
                    "axis": "z",
                    "world_coordinate_system": "blender",
                    "absolute": False,
                },
            )
        )
        # Plane in the same raw frame as trajectory: y = 0.
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                offset=0.0,
                metadata={
                    "source": "unit-test",
                    "measurement_allowed": True,
                    "support_quality_ok": True,
                    "residual_median": 0.0,
                    "residual_p90": 0.0,
                    "inlier_ratio": 1.0,
                },
            )
        )

    store.save_trajectory(PoseData(samples=samples, metadata={"source": "unit-test"}))
    return store


def test_plane_first_alignment_uses_raw_frame_for_scale_regression(tmp_path):
    store = _build_store_with_y_up_planes(tmp_path)

    align_trajectory_to_camera_height(
        store,
        settings=AlignmentSettings(
            mode="plane_first",
            fail_on_consistency_error=True,
            min_plane_scale_samples=3,
            max_plane_scale_iqr_ratio=0.35,
            min_plane_scale_inlier_ratio=0.5,
            max_height_rmse_m=0.5,
            max_height_abs_err_m=1.0,
        ),
    )

    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    metadata = traj["metadata"].item()
    assert metadata.get("metric_scale") is True
    assert metadata.get("scale_source") == "road_plane_camera_height_piecewise"
    assert abs(float(metadata.get("scale_factor", 0.0)) - 1.0) < 0.02

    scale_diag = dict(metadata.get("scale_diagnostics", {}))
    assert float(scale_diag.get("scale_iqr_ratio", 1.0)) < 0.35

    verify_alignment_consistency(store, require_road_plane=True)


def test_plane_first_alignment_writes_alignment_debug_artifacts(tmp_path):
    store = _build_store_with_y_up_planes(tmp_path)
    align_trajectory_to_camera_height(store, settings=AlignmentSettings())

    vis_dir = store.visualizations_dir("alignment")
    assert (vis_dir / "height_raw.png").exists()
    assert (vis_dir / "height_corrected.png").exists()
    assert (vis_dir / "alignment_summary.json").exists()


def test_metric_canonicalization_can_apply_custom_target_up_before_geometry_fusion(tmp_path):
    store = _build_store_with_y_up_planes(tmp_path)
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    frame_indices = traj["frame_indices"]
    c2w_stack = np.asarray(traj["camera_to_world"], dtype=np.float32)
    metadata = traj["metadata"].item()
    samples = [
        PoseSample(
            frame_index=int(frame_indices[i]),
            camera_to_world=c2w_stack[i],
            world_to_camera=np.linalg.inv(c2w_stack[i]),
            metadata={"camera_convention": "blender"},
        )
        for i in range(len(frame_indices))
    ]
    store.save_trajectory(
        PoseData(
            samples=samples,
            metadata={**metadata, "metric_scale": True},
        )
    )

    target_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    canonicalize_metric_geometry_to_camera_height(
        store,
        target_up=target_up,
    )

    out = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    c2w = np.asarray(out["camera_to_world"], dtype=np.float32)
    up_avg = np.mean(c2w[:, :3, 1], axis=0)
    up_avg = up_avg / np.linalg.norm(up_avg)
    assert up_avg[2] == pytest.approx(1.0, abs=1e-5)


def test_plane_first_alignment_transforms_road_plane_sampled_points(tmp_path):
    store = _build_store_with_y_up_planes(tmp_path)
    raw_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    store.save_road_plane_support(
        RoadPlaneSupportData(
            frame_index=1,
            points_world=raw_points,
            weights=np.ones((raw_points.shape[0],), dtype=np.float32),
            diagnostics={"source": "unit-test"},
            metadata={"source": "unit-test"},
        )
    )

    align_trajectory_to_camera_height(store, settings=AlignmentSettings())
    support = store.load_road_plane_support(1)
    transformed = np.asarray(support.points_world, dtype=np.float32)
    diagnostics = dict(support.diagnostics)
    assert transformed.shape == raw_points.shape
    # The run applies a non-identity transform, so points must move.
    assert not np.allclose(transformed, raw_points)
    assert diagnostics.get("metric_scale") is True
    assert isinstance(diagnostics.get("alignment_transform_id"), str)
    assert diagnostics.get("alignment_transform_id")


def test_alignment_fails_when_all_planes_are_unanchored_or_low_quality(tmp_path):
    store = ResourceStore("alignment_nonmetric_primary", root=tmp_path)
    frame_indices = list(range(1, 7))
    samples = []
    for i, frame_idx in enumerate(frame_indices):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = 0.1 * float(i)
        c2w[1, 3] = 0.8 + 0.1 * float(i)
        c2w[2, 3] = 0.2 * float(i)
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender"},
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((4, 4), 5.0, dtype=np.float32),
                metadata={"source": "unit-test"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.2 + 0.1 * float(i),
                metadata={
                    "source": "unit-test",
                    "axis": "z",
                    "world_coordinate_system": "blender",
                    "absolute": False,
                },
            )
        )
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                offset=0.0,
                metadata={
                    "source": "unit-test",
                    "enforce_height_anchor": False,
                    "measurement_allowed": True,
                    "support_quality_ok": True,
                    "residual_median": 0.0,
                    "residual_p90": 0.0,
                    "inlier_ratio": 1.0,
                },
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata={"source": "unit-test"}))

    for frame_idx in frame_indices:
        plane = store.load_road_plane(frame_idx)
        meta = dict(plane.metadata or {})
        meta["measurement_allowed"] = False
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=plane.normal,
                offset=plane.offset,
                metadata=meta,
            )
        )
    try:
        align_trajectory_to_camera_height(
            store,
            settings=AlignmentSettings(
                mode="plane_first",
                fail_on_consistency_error=True,
                min_plane_scale_samples=3,
                max_plane_scale_iqr_ratio=0.35,
                min_plane_scale_inlier_ratio=0.5,
                max_height_rmse_m=0.5,
                max_height_abs_err_m=1.0,
            ),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_alignment_requires_road_plane_resource(tmp_path):
    store = ResourceStore("alignment_requires_road_plane", root=tmp_path)
    frame_indices = [1, 2, 3]
    samples = []
    for i, frame_idx in enumerate(frame_indices):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = float(i) * 0.2
        c2w[2, 3] = 1.6
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender"},
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((4, 4), 5.0, dtype=np.float32),
                metadata={"source": "unit-test"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.6,
                metadata={"axis": "z", "world_coordinate_system": "blender"},
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata={"source": "unit-test"}))

    try:
        align_trajectory_to_camera_height(
            store,
            settings=AlignmentSettings(mode="plane_first"),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_alignment_rejects_high_dispersion_plane_scales(tmp_path):
    store = _build_store_with_y_up_planes(tmp_path)
    for frame_idx in range(1, 7):
        plane = store.load_road_plane(frame_idx)
        offset = 0.0 if frame_idx % 2 == 0 else -0.8
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=plane.normal,
                offset=offset,
                metadata=dict(plane.metadata or {}),
            )
        )
    try:
        align_trajectory_to_camera_height(
            store,
            settings=AlignmentSettings(
                mode="plane_first",
                min_plane_scale_samples=3,
                max_plane_scale_iqr_ratio=0.1,
                allow_degraded_output=False,
            ),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_alignment_allows_high_dispersion_when_degraded_mode_enabled(tmp_path):
    store = _build_store_with_y_up_planes(tmp_path)
    for frame_idx in range(1, 7):
        plane = store.load_road_plane(frame_idx)
        offset = 0.0 if frame_idx % 2 == 0 else -0.8
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=plane.normal,
                offset=offset,
                metadata=dict(plane.metadata or {}),
            )
        )
    align_trajectory_to_camera_height(
        store,
        settings=AlignmentSettings(
            mode="plane_first",
            min_plane_scale_samples=3,
            max_plane_scale_iqr_ratio=0.1,
            allow_degraded_output=True,
        ),
    )
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    metadata = traj["metadata"].item()
    reasons = list((metadata.get("scale_diagnostics", {}) or {}).get("degraded_reasons", ()))
    assert "high_iqr_ratio" in reasons


def test_alignment_writes_plane_anchor_fit_diagnostics(tmp_path):
    store = _build_store_with_y_up_planes(tmp_path)
    align_trajectory_to_camera_height(
        store,
        settings=AlignmentSettings(
            mode="plane_first",
            min_plane_scale_samples=3,
            max_plane_scale_iqr_ratio=0.35,
            min_plane_scale_inlier_ratio=0.5,
            max_plane_anchor_rmse_m=0.3,
            max_plane_anchor_abs_err_m=0.6,
        ),
    )
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    metadata = traj["metadata"].item()
    assert "plane_anchor_fit" in metadata
    fit = dict(metadata["plane_anchor_fit"])
    assert float(fit["plane_anchor_rmse_m"]) >= 0.0


def test_metric_canonicalization_preserves_depth_and_aligns_camera_height(tmp_path):
    store = ResourceStore("metric_canonicalization", root=tmp_path)
    frame_indices = [1, 2, 3, 4]
    samples = []
    for i, frame_idx in enumerate(frame_indices):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = float(i) * 0.5
        c2w[1, 3] = 1.6
        c2w[2, 3] = 0.2 * float(i)
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender", "metric_scale": True},
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((3, 3), 5.0 + float(i), dtype=np.float32),
                metadata={
                    "source": "geometry_fusion",
                    "metric_scale": True,
                    "scale_source": "geometry_fusion",
                    "scale_factor": 1.2,
                    "bias_m": 0.3,
                },
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.6,
                metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
            )
        )
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                offset=0.0,
                metadata={
                    "source": "geometry_fusion",
                    "measurement_allowed": True,
                    "support_quality_ok": True,
                    "residual_p90": 0.0,
                    "inlier_ratio": 1.0,
                    "enforce_height_anchor": True,
                    "target_camera_height_m": 1.6,
                },
            )
        )

    raw_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], dtype=np.float32)
    store.save_road_plane_support(
        RoadPlaneSupportData(
            frame_index=1,
            points_world=raw_points,
            diagnostics={"source": "unit-test"},
            metadata={"source": "unit-test"},
        )
    )
    store.save_trajectory(
        PoseData(
            samples=samples,
            metadata={"source": "geometry_fusion", "metric_scale": True, "scale_source": "geometry_fusion"},
        )
    )

    original_depth = store.load_depth(1).depth.copy()
    canonicalize_metric_geometry_to_camera_height(store)

    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    metadata = traj["metadata"].item()
    c2w = np.asarray(traj["camera_to_world"], dtype=np.float32)
    assert metadata.get("metric_scale") is True
    assert metadata.get("alignment_mode") == "metric_rigid_canonicalization"
    assert metadata.get("scale_source") == "geometry_fusion"
    assert abs(float(metadata["alignment_transform"]["scale"]) - 1.0) < 1e-6
    assert np.allclose(c2w[:, 2, 3], 1.6, atol=1e-4)
    assert "up_direction_alignment_flipped_source" not in metadata
    assert float(metadata["camera_up_fit"]["up_dot_z_min"]) > 0.0
    assert np.all(c2w[:, 2, 1] > 0.0)

    depth = store.load_depth(1)
    assert np.allclose(depth.depth, original_depth)
    assert depth.metadata["scale_source"] == "geometry_fusion"
    assert depth.metadata["alignment_transform_id"] == metadata["alignment_transform_id"]

    plane = store.load_road_plane(1)
    anchor = float(np.dot(plane.normal, c2w[0, :3, 3]) + plane.offset)
    assert abs(anchor - 1.6) < 1e-4
    assert plane.metadata["support_plane_orientation_canonicalized"] is True

    support = store.load_road_plane_support(1)
    transformed = np.asarray(support.points_world, dtype=np.float32)
    diagnostics = dict(support.diagnostics)
    assert not np.allclose(transformed, raw_points)
    assert diagnostics["alignment_transform_id"] == metadata["alignment_transform_id"]

    verify_alignment_consistency(store, require_road_plane=True)


def test_metric_canonicalization_requires_metric_trajectory(tmp_path):
    store = ResourceStore("metric_canonicalization_requires_metric", root=tmp_path)
    c2w = np.eye(4, dtype=np.float32)
    store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(frame_index=1, camera_to_world=c2w, world_to_camera=np.linalg.inv(c2w)),
                PoseSample(frame_index=2, camera_to_world=c2w, world_to_camera=np.linalg.inv(c2w)),
            ],
            metadata={"metric_scale": False},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=1,
            height_m=1.6,
            metadata={"axis": "z", "world_coordinate_system": "blender"},
        )
    )
    store.save_camera_height(
        CameraHeightData(
            frame_index=2,
            height_m=1.6,
            metadata={"axis": "z", "world_coordinate_system": "blender"},
        )
    )

    try:
        canonicalize_metric_geometry_to_camera_height(store)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_metric_canonicalization_rejects_mirrored_support_surface(tmp_path):
    store = ResourceStore("metric_canonicalization_mirrored_support", root=tmp_path)
    samples = []
    for frame_idx in [1, 2, 3]:
        c2w = np.eye(4, dtype=np.float32)
        c2w[2, 3] = 1.6
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((2, 2), 5.0, dtype=np.float32),
                metadata={"metric_scale": True},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.6,
                metadata={"axis": "z", "world_coordinate_system": "blender"},
            )
        )
        # This plane sits above the camera after canonicalization.
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                offset=-2.0,
                metadata={
                    "measurement_allowed": True,
                    "support_quality_ok": True,
                    "residual_p90": 0.0,
                    "inlier_ratio": 1.0,
                },
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata={"metric_scale": True}))

    with np.testing.assert_raises(ValueError):
        canonicalize_metric_geometry_to_camera_height(store, settings=AlignmentSettings(allow_degraded_output=False))


def test_metric_canonicalization_prefers_road_plane_anchor_over_direct_height_gate(tmp_path):
    store = ResourceStore("metric_canonicalization_road_plane_gate", root=tmp_path)
    frame_indices = [1, 2, 3]
    samples = []
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    z_positions = [1.2, 1.5, 1.8]
    for i, frame_idx in enumerate(frame_indices):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation
        c2w[0, 3] = float(i) * 0.5
        c2w[2, 3] = float(z_positions[i])
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender", "metric_scale": True},
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((2, 2), 5.0, dtype=np.float32),
                metadata={"metric_scale": True, "scale_source": "geometry_fusion"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.5,
                metadata={"axis": "z", "world_coordinate_system": "blender"},
            )
        )
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                offset=-(float(z_positions[i]) - 1.5),
                metadata={
                    "source": "geometry_fusion",
                    "measurement_allowed": True,
                    "support_quality_ok": True,
                    "residual_p90": 0.0,
                    "inlier_ratio": 1.0,
                    "enforce_height_anchor": True,
                    "target_camera_height_m": 1.5,
                },
            )
        )

    store.save_trajectory(
        PoseData(
            samples=samples,
            metadata={"metric_scale": True, "camera_convention": "blender"},
        )
    )

    canonicalize_metric_geometry_to_camera_height(
        store,
        settings=AlignmentSettings(max_height_rmse_m=0.2, max_height_abs_err_m=0.4),
    )

    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    metadata = traj["metadata"].item()
    height_fit = dict(metadata["height_fit"])
    plane_anchor_fit = dict(metadata["plane_anchor_fit"])
    assert metadata["height_fit_validation_mode"] == "road_plane_anchor"
    assert height_fit["strict_thresholds_passed"] is False
    assert height_fit["diagnostic_only"] is True
    assert float(height_fit["height_rmse_m"]) > 0.2
    assert float(plane_anchor_fit["plane_anchor_signed_rmse_m"]) < 1e-4
    assert plane_anchor_fit["degraded"] is False


def test_metric_canonicalization_flips_plane_sign_without_flipping_camera_up(tmp_path):
    store = ResourceStore("metric_canonicalization_plane_sign_only", root=tmp_path)
    samples = []
    for frame_idx in [1, 2, 3]:
        c2w = np.eye(4, dtype=np.float32)
        c2w[1, 3] = 1.6
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender", "metric_scale": True},
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((2, 2), 4.0, dtype=np.float32),
                metadata={"metric_scale": True},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.6,
                metadata={"axis": "z", "world_coordinate_system": "blender"},
            )
        )
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                offset=0.0,
                metadata={"measurement_allowed": True, "support_quality_ok": True},
            )
        )
    store.save_trajectory(
        PoseData(samples=samples, metadata={"metric_scale": True, "camera_convention": "blender"})
    )

    canonicalize_metric_geometry_to_camera_height(store)

    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    c2w = np.asarray(traj["camera_to_world"], dtype=np.float32)
    meta = traj["metadata"].item()
    assert float(meta["camera_up_fit"]["up_dot_z_min"]) > 0.0
    assert np.all(c2w[:, 2, 1] > 0.0)
    plane = store.load_road_plane(1)
    assert plane.metadata["support_plane_orientation_canonicalized"] is True
    assert plane.metadata["support_plane_sign_flipped"] is True
