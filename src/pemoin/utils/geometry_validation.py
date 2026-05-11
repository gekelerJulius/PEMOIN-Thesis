"""
Geometry validation checks for standardized pipeline outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence
import logging

import numpy as np

from pemoin.data.contracts import ResourceKind, ResourceStore
from pemoin.geometry.camera_model import (
    backproject_uv_depth_to_camera,
    camera_to_world as transform_camera_to_world,
    project_world_to_image,
    world_to_camera as transform_world_to_camera,
)
from pemoin.providers.semantic_roles import resolve_role_label_ids
from pemoin.utils.geometry import up_direction_from_c2w, view_direction_from_c2w

LOG = logging.getLogger("pemoin")


@dataclass(frozen=True)
class GeometryValidationConfig:
    enabled: bool = True
    max_frames: int = 50
    max_depth_pixels: int = 5000
    min_positive_depth_ratio: float = 0.97
    min_front_ratio: float = 0.98
    reprojection_error_px: float = 2.0
    min_reprojection_ratio: float = 0.95
    motion_min_distance_m: float = 0.02
    min_view_motion_alignment: float = 0.2
    min_view_motion_ratio: float = 0.6
    rotation_orthonormal_tol: float = 1e-3
    det_tolerance: float = 0.05
    last_row_tolerance: float = 1e-3
    camera_height_tolerance_m: float = 0.25
    max_pose_jump_m: float = 50.0
    pose_inverse_frobenius_tol: float = 1e-2
    max_relative_rotation_deg: float = 120.0
    max_motion_translation_ratio: float = 120.0
    max_motion_rotation_ratio: float = 120.0
    min_motion_pairs_for_ratio_checks: int = 5
    max_static_rotation_deg: float = 30.0
    write_visualizations: bool = True
    max_visualization_frames: int = 24
    check_point_cloud_road_vertical_plausibility: bool = True
    point_cloud_road_max_median_above_camera_m: float = 0.05
    point_cloud_road_local_radius_m: float = 10.0
    point_cloud_road_local_min_points: int = 512
    point_cloud_vertical_check_max_frames: int = 64
    check_plane_anchor_consistency: bool = True
    plane_anchor_tolerance_m: float = 0.25
    check_road_plane_residual_consistency: bool = True
    road_plane_metadata_residual_tolerance_m: float = 0.02
    check_canonical_camera_up: bool = True
    canonical_camera_up_min_dot_z: float = 0.0
    canonical_camera_up_min_median_dot_z: float = 0.5

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> "GeometryValidationConfig":
        if not settings:
            return cls()
        payload: dict[str, Any] = {}
        for field in fields(cls):
            if field.name in settings:
                payload[field.name] = settings[field.name]
        return cls(**payload)

    def validate(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("geometry_validation.max_frames must be > 0.")
        if self.max_depth_pixels <= 0:
            raise ValueError("geometry_validation.max_depth_pixels must be > 0.")
        if not (0.0 < self.min_positive_depth_ratio <= 1.0):
            raise ValueError("geometry_validation.min_positive_depth_ratio must be in (0, 1].")
        if not (0.0 < self.min_front_ratio <= 1.0):
            raise ValueError("geometry_validation.min_front_ratio must be in (0, 1].")
        if self.reprojection_error_px <= 0:
            raise ValueError("geometry_validation.reprojection_error_px must be > 0.")
        if not (0.0 < self.min_reprojection_ratio <= 1.0):
            raise ValueError("geometry_validation.min_reprojection_ratio must be in (0, 1].")
        if self.motion_min_distance_m < 0:
            raise ValueError("geometry_validation.motion_min_distance_m must be >= 0.")
        if self.min_view_motion_alignment < -1.0 or self.min_view_motion_alignment > 1.0:
            raise ValueError("geometry_validation.min_view_motion_alignment must be in [-1, 1].")
        if not (0.0 <= self.min_view_motion_ratio <= 1.0):
            raise ValueError("geometry_validation.min_view_motion_ratio must be in [0, 1].")
        if self.rotation_orthonormal_tol <= 0:
            raise ValueError("geometry_validation.rotation_orthonormal_tol must be > 0.")
        if self.det_tolerance <= 0:
            raise ValueError("geometry_validation.det_tolerance must be > 0.")
        if self.last_row_tolerance <= 0:
            raise ValueError("geometry_validation.last_row_tolerance must be > 0.")
        if self.camera_height_tolerance_m <= 0:
            raise ValueError("geometry_validation.camera_height_tolerance_m must be > 0.")
        if self.max_pose_jump_m <= 0:
            raise ValueError("geometry_validation.max_pose_jump_m must be > 0.")
        if self.pose_inverse_frobenius_tol <= 0:
            raise ValueError("geometry_validation.pose_inverse_frobenius_tol must be > 0.")
        if self.max_relative_rotation_deg <= 0:
            raise ValueError("geometry_validation.max_relative_rotation_deg must be > 0.")
        if self.max_motion_translation_ratio <= 1.0:
            raise ValueError("geometry_validation.max_motion_translation_ratio must be > 1.")
        if self.max_motion_rotation_ratio <= 1.0:
            raise ValueError("geometry_validation.max_motion_rotation_ratio must be > 1.")
        if self.min_motion_pairs_for_ratio_checks < 2:
            raise ValueError("geometry_validation.min_motion_pairs_for_ratio_checks must be >= 2.")
        if self.max_static_rotation_deg <= 0:
            raise ValueError("geometry_validation.max_static_rotation_deg must be > 0.")
        if self.max_visualization_frames <= 0:
            raise ValueError("geometry_validation.max_visualization_frames must be > 0.")
        if self.point_cloud_road_max_median_above_camera_m < 0:
            raise ValueError("geometry_validation.point_cloud_road_max_median_above_camera_m must be >= 0.")
        if self.point_cloud_road_local_radius_m <= 0:
            raise ValueError("geometry_validation.point_cloud_road_local_radius_m must be > 0.")
        if self.point_cloud_road_local_min_points <= 0:
            raise ValueError("geometry_validation.point_cloud_road_local_min_points must be > 0.")
        if self.point_cloud_vertical_check_max_frames <= 0:
            raise ValueError("geometry_validation.point_cloud_vertical_check_max_frames must be > 0.")
        if self.plane_anchor_tolerance_m <= 0:
            raise ValueError("geometry_validation.plane_anchor_tolerance_m must be > 0.")
        if self.road_plane_metadata_residual_tolerance_m <= 0:
            raise ValueError("geometry_validation.road_plane_metadata_residual_tolerance_m must be > 0.")
        if not (-1.0 <= self.canonical_camera_up_min_dot_z < 1.0):
            raise ValueError("geometry_validation.canonical_camera_up_min_dot_z must be in [-1, 1).")
        if not (-1.0 <= self.canonical_camera_up_min_median_dot_z <= 1.0):
            raise ValueError("geometry_validation.canonical_camera_up_min_median_dot_z must be in [-1, 1].")
        if self.canonical_camera_up_min_median_dot_z < self.canonical_camera_up_min_dot_z:
            raise ValueError(
                "geometry_validation.canonical_camera_up_min_median_dot_z must be >= "
                "geometry_validation.canonical_camera_up_min_dot_z."
            )


class GeometryValidationError(RuntimeError):
    """Raised when geometry outputs fail validation."""


def _validate_road_plane_anchor_consistency(
    *,
    store: ResourceStore,
    cfg: GeometryValidationConfig,
    frame_indices: Sequence[int],
    traj_index: Mapping[int, int],
    c2w_all: np.ndarray,
    trajectory_metric_scale: bool,
) -> dict[str, Any]:
    checked_frames = 0
    max_abs_delta = 0.0
    for frame_idx in frame_indices:
        pose_idx = traj_index.get(int(frame_idx))
        if pose_idx is None:
            continue
        plane = store.load_road_plane(int(frame_idx))
        n = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        d = float(plane.offset)
        enforce_height_anchor = bool((plane.metadata or {}).get("enforce_height_anchor", True))
        if not trajectory_metric_scale and not enforce_height_anchor:
            continue
        height = float(store.load_camera_height(int(frame_idx)).height_m)
        anchor = float(n @ c2w_all[int(pose_idx), :3, 3] + d)
        delta = abs(anchor - height)
        checked_frames += 1
        max_abs_delta = max(max_abs_delta, delta)
        if delta > float(cfg.plane_anchor_tolerance_m):
            raise GeometryValidationError(
                "Road-plane anchor consistency failed "
                f"for frame {frame_idx}: anchor={anchor:.3f} camera_height={height:.3f} "
                f"tol={cfg.plane_anchor_tolerance_m:.3f}."
            )
    return {
        "checked_frames": int(checked_frames),
        "max_abs_anchor_delta_m": float(max_abs_delta),
        "plane_anchor_tolerance_m": float(cfg.plane_anchor_tolerance_m),
    }


def validate_road_plane_anchor_consistency(
    store: ResourceStore,
    *,
    config: GeometryValidationConfig | None = None,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    cfg = config or GeometryValidationConfig()
    cfg.validate()
    if not cfg.check_plane_anchor_consistency:
        return {
            "checked_frames": 0,
            "max_abs_anchor_delta_m": 0.0,
            "plane_anchor_tolerance_m": float(cfg.plane_anchor_tolerance_m),
            "enabled": False,
        }
    if not store.has(ResourceKind.TRAJECTORY):
        raise GeometryValidationError("Road-plane anchor consistency requires trajectory data.")
    if not store.has(ResourceKind.ROAD_PLANE):
        raise GeometryValidationError("Road-plane anchor consistency requires road-plane data.")
    if not store.has(ResourceKind.CAMERA_HEIGHT):
        raise GeometryValidationError("Road-plane anchor consistency requires camera-height data.")

    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as traj_data:
        frame_ids = np.asarray(traj_data["frame_indices"], dtype=np.int32)
        c2w_all = np.asarray(traj_data["camera_to_world"], dtype=np.float32)
        traj_meta = _coerce_metadata(traj_data)
    traj_index = {int(frame_idx): idx for idx, frame_idx in enumerate(frame_ids.tolist())}
    trajectory_metric_scale = bool(traj_meta.get("metric_scale", False)) if isinstance(traj_meta, Mapping) else False

    selected_frames = (
        [int(frame_idx) for frame_idx in frame_indices]
        if frame_indices is not None
        else [int(frame_idx) for frame_idx in store.frame_indices(ResourceKind.ROAD_PLANE)]
    )
    return _validate_road_plane_anchor_consistency(
        store=store,
        cfg=cfg,
        frame_indices=selected_frames,
        traj_index=traj_index,
        c2w_all=c2w_all,
        trajectory_metric_scale=trajectory_metric_scale,
    )


def _validate_canonical_camera_up(
    c2w_all: np.ndarray,
    frame_indices: Sequence[int],
    traj_index: Mapping[int, int],
    cfg: GeometryValidationConfig,
) -> None:
    if not frame_indices:
        return
    up_dots: list[float] = []
    for frame_idx in frame_indices:
        pose_idx = traj_index.get(int(frame_idx))
        if pose_idx is None:
            continue
        up = up_direction_from_c2w(np.asarray(c2w_all[int(pose_idx)], dtype=np.float32))
        norm = float(np.linalg.norm(up))
        if not np.isfinite(norm) or norm < 1e-6:
            raise GeometryValidationError(
                f"Canonical camera-up validation failed for frame {int(frame_idx)}: degenerate up vector."
            )
        dot_z = float(np.dot((up / norm).astype(np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        up_dots.append(dot_z)
        if dot_z <= float(cfg.canonical_camera_up_min_dot_z):
            raise GeometryValidationError(
                "Canonical camera-up validation failed "
                f"for frame {int(frame_idx)}: up·+Z={dot_z:.4f} "
                f"<= {float(cfg.canonical_camera_up_min_dot_z):.4f}."
            )
    if not up_dots:
        return
    median_dot = float(np.median(np.asarray(up_dots, dtype=np.float64)))
    if median_dot < float(cfg.canonical_camera_up_min_median_dot_z):
        raise GeometryValidationError(
            "Canonical camera-up validation failed: "
            f"median(up·+Z)={median_dot:.4f} < {float(cfg.canonical_camera_up_min_median_dot_z):.4f}."
        )


@dataclass(frozen=True)
class ReprojectionOverlayData:
    frame_index: int
    xs: np.ndarray
    ys: np.ndarray
    errors_px: np.ndarray


@dataclass(frozen=True)
class FrameGeometryMetrics:
    frame_index: int
    inverse_error_left_fro: float
    inverse_error_right_fro: float
    rotation_orthonormal_error: float
    rotation_determinant: float
    positive_depth_ratio: float
    front_ratio: float
    reprojection_inlier_ratio: float
    reprojection_rmse_px: float
    camera_position_x: float
    camera_position_y: float
    camera_position_z: float


@dataclass(frozen=True)
class MotionPairMetrics:
    prev_frame_index: int
    frame_index: int
    translation_m: float
    rotation_deg: float
    view_motion_alignment: float | None


@dataclass(frozen=True)
class GeometryValidationReport:
    frame_metrics: list[FrameGeometryMetrics]
    motion_metrics: list[MotionPairMetrics]
    reprojection_overlays: list[ReprojectionOverlayData]
    sampled_frame_indices: list[int]

    def to_summary(self) -> Mapping[str, Any]:
        frame_payload = [
            {
                "frame_index": m.frame_index,
                "inverse_error_left_fro": m.inverse_error_left_fro,
                "inverse_error_right_fro": m.inverse_error_right_fro,
                "rotation_orthonormal_error": m.rotation_orthonormal_error,
                "rotation_determinant": m.rotation_determinant,
                "positive_depth_ratio": m.positive_depth_ratio,
                "front_ratio": m.front_ratio,
                "reprojection_inlier_ratio": m.reprojection_inlier_ratio,
                "reprojection_rmse_px": m.reprojection_rmse_px,
                "camera_position_x": m.camera_position_x,
                "camera_position_y": m.camera_position_y,
                "camera_position_z": m.camera_position_z,
            }
            for m in self.frame_metrics
        ]
        motion_payload = [
            {
                "prev_frame_index": m.prev_frame_index,
                "frame_index": m.frame_index,
                "translation_m": m.translation_m,
                "rotation_deg": m.rotation_deg,
                "view_motion_alignment": m.view_motion_alignment,
            }
            for m in self.motion_metrics
        ]
        return {
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "num_frames": len(self.frame_metrics),
            "num_motion_pairs": len(self.motion_metrics),
            "frames": frame_payload,
            "motion_pairs": motion_payload,
        }


@dataclass(frozen=True)
class _PoseValidationResult:
    inverse_error_left_fro: float
    inverse_error_right_fro: float
    rotation_orthonormal_error: float
    rotation_determinant: float


@dataclass(frozen=True)
class _DepthValidationResult:
    positive_depth_ratio: float
    front_ratio: float
    reprojection_inlier_ratio: float
    reprojection_rmse_px: float
    overlay: ReprojectionOverlayData | None


def validate_geometry_store(
    store: ResourceStore,
    *,
    config: GeometryValidationConfig | None = None,
    expected_frames: int | None = None,
    logger: logging.Logger | None = None,
) -> GeometryValidationReport:
    cfg = config or GeometryValidationConfig()
    cfg.validate()
    if not cfg.enabled:
        return GeometryValidationReport([], [], [], [])

    log = logger or LOG
    log.info("Running geometry validation against standardized outputs.")

    required = [
        ResourceKind.FRAMES,
        ResourceKind.INTRINSICS,
        ResourceKind.DEPTH,
        ResourceKind.TRAJECTORY,
    ]
    missing = [kind.value for kind in required if not store.has(kind)]
    if missing:
        raise GeometryValidationError(
            "Geometry validation requires frames, intrinsics, depth, and trajectory outputs. "
            f"Missing: {', '.join(missing)}."
        )

    frame_indices = store.frame_indices(ResourceKind.FRAMES)
    if not frame_indices:
        raise GeometryValidationError("No frames found in the ResourceStore for validation.")
    if expected_frames is not None and len(frame_indices) < expected_frames:
        raise GeometryValidationError(
            f"Expected at least {expected_frames} frames but only found {len(frame_indices)}."
        )

    depth_indices = set(store.frame_indices(ResourceKind.DEPTH))
    if expected_frames is not None and len(depth_indices) < expected_frames:
        raise GeometryValidationError(
            f"Expected at least {expected_frames} depth maps but only found {len(depth_indices)}."
        )

    intrinsics = store.load_intrinsics()
    k = np.asarray(intrinsics.matrix, dtype=np.float32)
    intr_meta = intrinsics.metadata or {}
    log.info("Validating intrinsics matrix shape, values, and metadata.")
    _validate_intrinsics(k)
    _validate_convention_metadata(intr_meta, resource="intrinsics")

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        traj_frames = np.asarray(data["frame_indices"], dtype=np.int32)
        c2w_all = np.asarray(data["camera_to_world"], dtype=np.float32)
        w2c_raw = data["world_to_camera"] if "world_to_camera" in data.files else None
        w2c_all = _coerce_optional_matrix_array(w2c_raw)
        view_dir_all = _coerce_optional_matrix_array(data["view_direction"]) if "view_direction" in data.files else None
        up_dir_all = _coerce_optional_matrix_array(data["up_direction"]) if "up_direction" in data.files else None
        metadata = _coerce_metadata(data)

    if traj_frames.size == 0:
        raise GeometryValidationError("Trajectory file contains no frame indices.")
    if len(set(traj_frames.tolist())) != traj_frames.size:
        raise GeometryValidationError("Trajectory frame_indices contain duplicates.")
    if c2w_all.shape[0] != traj_frames.size:
        raise GeometryValidationError("camera_to_world array length does not match frame_indices.")
    _validate_convention_metadata(metadata, resource="trajectory")

    if metadata and metadata.get("metric_scale") is False:
        raise GeometryValidationError("Trajectory must be metric-scaled; metric_scale=False is not allowed.")

    traj_index = {int(frame): idx for idx, frame in enumerate(traj_frames.tolist())}

    log.info("Sampling up to %s frames for validation.", cfg.max_frames)
    sample_frames = _sample_indices(frame_indices, cfg.max_frames)
    missing_traj = [idx for idx in sample_frames if idx not in traj_index]
    if missing_traj:
        raise GeometryValidationError(
            f"Trajectory missing poses for sampled frames: {missing_traj[:10]}."
        )
    missing_depth = [idx for idx in sample_frames if idx not in depth_indices]
    if missing_depth:
        raise GeometryValidationError(
            f"Depth missing outputs for sampled frames: {missing_depth[:10]}."
        )

    if (
        cfg.check_canonical_camera_up
        and isinstance(metadata, Mapping)
        and bool(metadata.get("canonical_world_frame", False))
        and str(metadata.get("camera_convention", "")).strip().lower() == "blender"
    ):
        log.info("Validating canonical camera-up orientation.")
        _validate_canonical_camera_up(c2w_all, sample_frames, traj_index, cfg)

    log.info("Validating camera motion alignment and relative transform sanity.")
    motion_metrics = _validate_motion_and_relative_transforms(
        sample_frames,
        traj_index,
        c2w_all,
        cfg,
        log,
    )

    frame_metrics: list[FrameGeometryMetrics] = []
    overlays: list[ReprojectionOverlayData] = []
    grounding_shift_z_m = 0.0
    grounding_meta = metadata.get("grounding_to_z0", {}) if isinstance(metadata, Mapping) else {}
    if isinstance(grounding_meta, Mapping) and bool(grounding_meta.get("enabled", False)):
        try:
            grounding_shift_z_m = float(grounding_meta.get("shift_z_m", 0.0))
        except (TypeError, ValueError):
            raise GeometryValidationError(
                "Trajectory grounding_to_z0.shift_z_m metadata must be a finite float."
            ) from None
        if not np.isfinite(grounding_shift_z_m):
            raise GeometryValidationError(
                "Trajectory grounding_to_z0.shift_z_m metadata must be finite."
            )

    for frame_idx in sample_frames:
        log.debug("Validating frame %s", frame_idx)
        depth = store.load_depth(frame_idx)
        frame = store.load_frame(frame_idx)
        depth_meta = depth.metadata or {}
        _validate_convention_metadata(depth_meta, resource="depth", frame_idx=frame_idx)

        log.debug("Checking depth shape and metadata for frame %s.", frame_idx)
        _validate_depth_shape(depth.depth, frame_idx)
        image_shape = _resolve_image_shape(frame, depth.depth, intr_meta, depth_meta)
        if depth.depth.shape[:2] != image_shape:
            raise GeometryValidationError(
                f"Depth shape {depth.depth.shape[:2]} does not match frame shape {image_shape} "
                f"for frame {frame_idx}."
            )

        log.debug("Checking intrinsics bounds for frame %s.", frame_idx)
        _validate_intrinsics_image_bounds(fx, fy, cx, cy, image_shape)
        _validate_reference_resolution(image_shape, intr_meta, depth_meta, frame_idx)

        pose_idx = traj_index[int(frame_idx)]
        c2w = np.asarray(c2w_all[pose_idx], dtype=np.float32)
        w2c = _resolve_w2c(c2w, w2c_all, pose_idx)

        log.debug("Checking pose matrices for frame %s.", frame_idx)
        pose_result = _validate_pose_matrix(c2w, w2c, frame_idx, cfg)
        _validate_view_up_vectors(c2w, view_dir_all, up_dir_all, pose_idx, frame_idx)

        log.debug("Checking depth reprojection for frame %s.", frame_idx)
        capture_overlay = len(overlays) < cfg.max_visualization_frames
        depth_result = _validate_depth_consistency(
            store,
            depth.depth,
            c2w,
            w2c,
            k,
            image_shape,
            frame_idx,
            cfg,
            capture_overlay,
        )
        if depth_result.overlay is not None:
            overlays.append(depth_result.overlay)

        if store.has(ResourceKind.CAMERA_HEIGHT):
            log.debug("Checking camera height for frame %s.", frame_idx)
            _validate_camera_height(
                store,
                frame_idx,
                c2w,
                cfg,
                grounding_shift_z_m=grounding_shift_z_m,
                trajectory_metadata=metadata if isinstance(metadata, Mapping) else None,
            )

        frame_metrics.append(
            FrameGeometryMetrics(
                frame_index=int(frame_idx),
                inverse_error_left_fro=pose_result.inverse_error_left_fro,
                inverse_error_right_fro=pose_result.inverse_error_right_fro,
                rotation_orthonormal_error=pose_result.rotation_orthonormal_error,
                rotation_determinant=pose_result.rotation_determinant,
                positive_depth_ratio=depth_result.positive_depth_ratio,
                front_ratio=depth_result.front_ratio,
                reprojection_inlier_ratio=depth_result.reprojection_inlier_ratio,
                reprojection_rmse_px=depth_result.reprojection_rmse_px,
                camera_position_x=float(c2w[0, 3]),
                camera_position_y=float(c2w[1, 3]),
                camera_position_z=float(c2w[2, 3]),
            )
        )

    _validate_optional_road_geometry_consistency(
        store=store,
        cfg=cfg,
        sample_frames=sample_frames,
        traj_index=traj_index,
        c2w_all=c2w_all,
    )

    report = GeometryValidationReport(
        frame_metrics=frame_metrics,
        motion_metrics=motion_metrics,
        reprojection_overlays=overlays,
        sampled_frame_indices=list(sample_frames),
    )

    if cfg.write_visualizations:
        log.info("Writing geometry validation visualizations.")
        from pemoin.visualization.geometry_validation import (
            write_geometry_validation_visualizations,
        )

        try:
            write_geometry_validation_visualizations(store, report, logger=log)
        except Exception as exc:
            raise GeometryValidationError(
                f"Failed to write geometry validation visualizations: {exc}"
            ) from exc

    log.info("Geometry validation completed successfully.")
    return report


def _sample_indices(indices: Sequence[int], max_frames: int) -> list[int]:
    if len(indices) <= max_frames:
        return list(indices)
    stride = max(1, len(indices) // max_frames)
    sampled = list(indices[::stride])
    return sampled[:max_frames]


def _resolve_non_sky_mask(
    store: ResourceStore,
    *,
    frame_idx: int,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if not store.has(ResourceKind.SEMANTICS_2D):
        return None
    try:
        semantics = store.load_semantics2d(frame_idx)
    except Exception:
        return None
    label_map = {
        int(seg.label_id): str(seg.label).strip().lower()
        for seg in semantics.segments
        if getattr(seg, "label_id", None) is not None
    }
    if not label_map:
        return None
    sky_ids = resolve_role_label_ids(
        label_map,
        "sky",
        metadata=semantics.metadata,
    )
    if not sky_ids:
        return None

    if semantics.label_ids is not None:
        label_ids = np.asarray(semantics.label_ids)
        if label_ids.shape == shape:
            return ~np.isin(label_ids, np.asarray(sky_ids, dtype=label_ids.dtype))

    if semantics.segment_ids is None:
        return None
    segment_ids = np.asarray(semantics.segment_ids)
    if segment_ids.shape != shape:
        return None
    segment_to_label = {
        int(seg.segment_id): int(seg.label_id)
        for seg in semantics.segments
        if getattr(seg, "segment_id", None) is not None and getattr(seg, "label_id", None) is not None
    }
    if not segment_to_label:
        return None
    label_ids = np.full(segment_ids.shape, fill_value=-1, dtype=np.int32)
    for segment_id, label_id in segment_to_label.items():
        label_ids[segment_ids == int(segment_id)] = int(label_id)
    return ~np.isin(label_ids, np.asarray(sky_ids, dtype=np.int32))


def _validate_optional_road_geometry_consistency(
    *,
    store: ResourceStore,
    cfg: GeometryValidationConfig,
    sample_frames: Sequence[int],
    traj_index: Mapping[int, int],
    c2w_all: np.ndarray,
) -> None:
    if not (store.has(ResourceKind.POINT_CLOUD_3D) and store.has(ResourceKind.ROAD_PLANE)):
        return

    cloud = store.load_point_cloud_3d()
    points_world = np.asarray(cloud.points_world, dtype=np.float32)
    labels = np.asarray(cloud.labels, dtype=np.int32).reshape(-1)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise GeometryValidationError(f"POINT_CLOUD_3D points_world has invalid shape {points_world.shape}.")
    if labels.shape[0] != points_world.shape[0]:
        raise GeometryValidationError("POINT_CLOUD_3D labels length mismatch.")
    label_names = {int(k): str(v).lower() for k, v in (cloud.label_names or {}).items()}
    road_label_ids = [lid for lid, name in label_names.items() if "road" in name]
    if not road_label_ids:
        raise GeometryValidationError("POINT_CLOUD_3D has no road-like labels (name contains 'road').")
    road_mask = np.isin(labels, np.asarray(road_label_ids, dtype=np.int32))
    if not np.any(road_mask):
        raise GeometryValidationError("POINT_CLOUD_3D has zero points for road-like labels.")
    road_points_global = points_world[road_mask]

    intrinsics = store.load_intrinsics()
    selected = list(_sample_indices(sample_frames, cfg.point_cloud_vertical_check_max_frames))
    trajectory_metric_scale = False
    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as traj_data:
        traj_meta = _coerce_metadata(traj_data)
    if isinstance(traj_meta, Mapping):
        trajectory_metric_scale = bool(traj_meta.get("metric_scale", False))
    for frame_idx in selected:
        pose_idx = traj_index.get(int(frame_idx))
        if pose_idx is None:
            continue
        camera_z = float(c2w_all[int(pose_idx), 2, 3])
        pose = store.load_pose(int(frame_idx))
        frame = store.load_frame(int(frame_idx))
        if frame.image is None:
            continue
        _, valid = project_world_to_image(
            road_points_global,
            intrinsics.matrix,
            world_to_camera_matrix=pose.world_to_camera,
            camera_to_world_matrix=pose.camera_to_world,
            camera_convention="blender",
            image_shape=frame.image.shape[:2],
        )
        pts = road_points_global[valid]
        if cfg.check_point_cloud_road_vertical_plausibility:
            if pts.size == 0:
                raise GeometryValidationError(f"POINT_CLOUD_3D road subset empty at frame {frame_idx}.")
            camera_xy = np.asarray(c2w_all[int(pose_idx), :2, 3], dtype=np.float32).reshape(1, 2)
            horizontal_radius = np.linalg.norm(
                pts[:, :2] - camera_xy,
                axis=1,
            )
            local_pts = pts[horizontal_radius <= float(cfg.point_cloud_road_local_radius_m)]
            plausibility_points = (
                local_pts
                if local_pts.shape[0] >= int(cfg.point_cloud_road_local_min_points)
                else pts
            )
            road_med_z = float(np.median(plausibility_points[:, 2]))
            if road_med_z > camera_z + float(cfg.point_cloud_road_max_median_above_camera_m):
                raise GeometryValidationError(
                    "Point-cloud road vertical plausibility failed "
                    f"for frame {frame_idx}: road_median_z={road_med_z:.3f} camera_z={camera_z:.3f} "
                    f"max_above={cfg.point_cloud_road_max_median_above_camera_m:.3f}."
                )

        plane = store.load_road_plane(int(frame_idx))
        n = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        d = float(plane.offset)

        if cfg.check_road_plane_residual_consistency:
            support_path = store.path_for(ResourceKind.ROAD_PLANE_SUPPORT, int(frame_idx))
            if not support_path.exists():
                continue
            support = store.load_road_plane_support(int(frame_idx))
            diagnostics = dict(support.diagnostics or {})
            plane_meta = dict(plane.metadata or {})
            plane_transform_id = str(plane_meta.get("alignment_transform_id", "")).strip()
            sampled_transform_id = str(diagnostics.get("alignment_transform_id", "")).strip()
            if plane_transform_id and sampled_transform_id != plane_transform_id:
                raise GeometryValidationError(
                    "Road-plane support transform mismatch "
                    f"for frame {frame_idx}: plane_transform_id={plane_transform_id!r} "
                    f"support_transform_id={sampled_transform_id!r}. "
                    "Standardized support points are stale relative to aligned road planes."
                )
            residual_points = np.asarray(support.points_world, dtype=np.float32)
            if not (
                residual_points.ndim == 2
                and residual_points.shape[1] == 3
                and residual_points.shape[0] > 0
            ):
                raise GeometryValidationError(
                    f"Road-plane residual consistency has no valid support points for frame {frame_idx}."
                )
            if residual_points.size == 0:
                raise GeometryValidationError(
                    f"Road-plane residual consistency has no points for frame {frame_idx}."
                )
            residuals = np.abs(residual_points @ n + d)
            residual_p90 = float(np.percentile(residuals, 90))
            meta = dict(plane.metadata or {})
            if "residual_p90" not in meta:
                raise GeometryValidationError(
                    f"Road-plane metadata missing residual_p90 for frame {frame_idx}."
                )
            meta_p90 = float(meta["residual_p90"])
            if abs(residual_p90 - meta_p90) > float(cfg.road_plane_metadata_residual_tolerance_m):
                raise GeometryValidationError(
                    "Road-plane residual consistency failed "
                    f"for frame {frame_idx}: computed_p90={residual_p90:.3f} "
                    f"metadata_p90={meta_p90:.3f} tol={cfg.road_plane_metadata_residual_tolerance_m:.3f}."
                )

    if cfg.check_plane_anchor_consistency and store.has(ResourceKind.CAMERA_HEIGHT):
        _validate_road_plane_anchor_consistency(
            store=store,
            cfg=cfg,
            frame_indices=selected,
            traj_index=traj_index,
            c2w_all=c2w_all,
            trajectory_metric_scale=trajectory_metric_scale,
        )


def _coerce_optional_matrix_array(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.dtype == object and arr.size == 1:
        item = arr.item()
        if item is None:
            return None
        arr = np.asarray(item)
    return np.asarray(arr, dtype=np.float32)


def _coerce_metadata(data: np.lib.npyio.NpzFile) -> Mapping[str, Any]:
    if "metadata" not in data.files:
        return {}
    meta = data["metadata"]
    if isinstance(meta, np.ndarray) and meta.dtype == object:
        try:
            return meta.item()
        except Exception:
            return {}
    if isinstance(meta, Mapping):
        return meta
    return {}


def _validate_intrinsics(k: np.ndarray) -> None:
    if k.shape != (3, 3):
        raise GeometryValidationError(f"Intrinsics matrix must be 3x3, got {k.shape}.")
    if not np.isfinite(k).all():
        raise GeometryValidationError("Intrinsics matrix contains non-finite values.")

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    if fx <= 0 or fy <= 0:
        raise GeometryValidationError(f"Intrinsics focal lengths must be positive, got fx={fx}, fy={fy}.")

    if abs(k[2, 2] - 1.0) > 1e-3:
        raise GeometryValidationError(f"Intrinsics bottom-right entry must be 1, got {k[2, 2]}.")
    if abs(float(k[2, 0])) > 1e-6 or abs(float(k[2, 1])) > 1e-6:
        raise GeometryValidationError(
            "Intrinsics projective row must be [0, 0, 1] for pinhole camera model."
        )

    skew = float(k[0, 1])
    if abs(skew) > 1e-3:
        raise GeometryValidationError(
            f"Intrinsics skew K[0,1] must be ~0 for this pipeline, got {skew}."
        )


def _validate_convention_metadata(
    metadata: Mapping[str, Any],
    *,
    resource: str,
    frame_idx: int | None = None,
) -> None:
    if not metadata:
        return

    keys = (
        "camera_convention",
        "coordinate_system",
        "world_coordinate_system",
    )
    for key in keys:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        lower = str(value).lower()
        if lower != "blender":
            context = f" for frame {frame_idx}" if frame_idx is not None else ""
            raise GeometryValidationError(
                f"{resource} metadata key '{key}' must be 'blender'{context}, got '{value}'."
            )


def _resolve_image_shape(
    frame,
    depth: np.ndarray,
    intr_meta: Mapping[str, Any],
    depth_meta: Mapping[str, Any],
) -> tuple[int, int]:
    if frame.image is not None:
        return frame.image.shape[0], frame.image.shape[1]
    if depth.ndim >= 2:
        return depth.shape[0], depth.shape[1]
    ref = intr_meta.get("reference_resolution") or depth_meta.get("reference_resolution")
    if isinstance(ref, Sequence) and not isinstance(ref, (str, bytes)) and len(ref) >= 2:
        return int(ref[0]), int(ref[1])
    raise GeometryValidationError("Unable to resolve image shape for geometry validation.")


def _validate_intrinsics_image_bounds(fx: float, fy: float, cx: float, cy: float, shape: tuple[int, int]) -> None:
    height, width = shape
    if cx < 0 or cx > width or cy < 0 or cy > height:
        raise GeometryValidationError(
            f"Principal point out of bounds for resolution {width}x{height}: cx={cx}, cy={cy}."
        )
    # Allow very wide cameras (roughly up to ~127 degrees HFOV / VFOV) while
    # still rejecting clearly implausible near-zero focal lengths.
    min_fx = 0.25 * width
    max_fx = 10.0 * width
    min_fy = 0.25 * height
    max_fy = 10.0 * height
    if fx < min_fx or fx > max_fx:
        raise GeometryValidationError(
            f"Unusual fx={fx} for width={width} (expected {min_fx:.1f}-{max_fx:.1f})."
        )
    if fy < min_fy or fy > max_fy:
        raise GeometryValidationError(
            f"Unusual fy={fy} for height={height} (expected {min_fy:.1f}-{max_fy:.1f})."
        )


def _validate_reference_resolution(
    shape: tuple[int, int],
    intr_meta: Mapping[str, Any],
    depth_meta: Mapping[str, Any],
    frame_idx: int,
) -> None:
    for source, meta in ("intrinsics", intr_meta), ("depth", depth_meta):
        ref = meta.get("reference_resolution")
        if ref is None:
            continue
        if isinstance(ref, Sequence) and not isinstance(ref, (str, bytes)) and len(ref) >= 2:
            ref_shape = (int(ref[0]), int(ref[1]))
            if ref_shape != shape:
                raise GeometryValidationError(
                    f"{source} reference_resolution {ref_shape} does not match frame shape {shape} for frame {frame_idx}."
                )


def _validate_depth_shape(depth: np.ndarray, frame_idx: int) -> None:
    if depth.ndim != 2:
        raise GeometryValidationError(f"Depth for frame {frame_idx} must be 2D, got {depth.shape}.")
    if not np.isfinite(depth).any():
        raise GeometryValidationError(f"Depth for frame {frame_idx} contains no finite values.")


def _validate_pose_matrix(
    c2w: np.ndarray,
    w2c: np.ndarray,
    frame_idx: int,
    cfg: GeometryValidationConfig,
) -> _PoseValidationResult:
    if c2w.shape != (4, 4):
        raise GeometryValidationError(f"camera_to_world must be 4x4 for frame {frame_idx}.")
    if not np.isfinite(c2w).all():
        raise GeometryValidationError(f"camera_to_world contains non-finite values for frame {frame_idx}.")
    if w2c.shape != (4, 4):
        raise GeometryValidationError(f"world_to_camera must be 4x4 for frame {frame_idx}.")
    if not np.isfinite(w2c).all():
        raise GeometryValidationError(f"world_to_camera contains non-finite values for frame {frame_idx}.")

    expected_last_row = np.array([0, 0, 0, 1], dtype=np.float32)
    c2w_last_error = float(np.max(np.abs(c2w[3, :] - expected_last_row)))
    w2c_last_error = float(np.max(np.abs(w2c[3, :] - expected_last_row)))
    if c2w_last_error > cfg.last_row_tolerance:
        raise GeometryValidationError(f"camera_to_world last row invalid for frame {frame_idx}.")
    if w2c_last_error > cfg.last_row_tolerance:
        raise GeometryValidationError(f"world_to_camera last row invalid for frame {frame_idx}.")

    r = c2w[:3, :3]
    orth = r.T @ r
    orth_error = float(np.max(np.abs(orth - np.eye(3, dtype=np.float32))))
    if orth_error > cfg.rotation_orthonormal_tol:
        raise GeometryValidationError(
            f"Rotation matrix not orthonormal for frame {frame_idx}: max error {orth_error:.3e}."
        )
    det = float(np.linalg.det(r))
    if abs(det - 1.0) > cfg.det_tolerance:
        raise GeometryValidationError(f"Rotation determinant expected ~1, got {det} for frame {frame_idx}.")

    left_error = float(np.linalg.norm(c2w @ w2c - np.eye(4, dtype=np.float32), ord="fro"))
    right_error = float(np.linalg.norm(w2c @ c2w - np.eye(4, dtype=np.float32), ord="fro"))
    if left_error > cfg.pose_inverse_frobenius_tol or right_error > cfg.pose_inverse_frobenius_tol:
        raise GeometryValidationError(
            f"Pose invertibility check failed for frame {frame_idx}: "
            f"||c2w@w2c-I||_F={left_error:.3e}, ||w2c@c2w-I||_F={right_error:.3e}."
        )

    return _PoseValidationResult(
        inverse_error_left_fro=left_error,
        inverse_error_right_fro=right_error,
        rotation_orthonormal_error=orth_error,
        rotation_determinant=det,
    )


def _validate_view_up_vectors(
    c2w: np.ndarray,
    view_dir_all: np.ndarray | None,
    up_dir_all: np.ndarray | None,
    pose_idx: int,
    frame_idx: int,
) -> None:
    view_dir = view_direction_from_c2w(c2w)
    up_dir = up_direction_from_c2w(c2w)
    if abs(np.linalg.norm(view_dir) - 1.0) > 1e-3:
        raise GeometryValidationError(f"View direction not unit length for frame {frame_idx}.")
    if abs(np.linalg.norm(up_dir) - 1.0) > 1e-3:
        raise GeometryValidationError(f"Up direction not unit length for frame {frame_idx}.")
    if abs(float(np.dot(view_dir, up_dir))) > 1e-3:
        raise GeometryValidationError(f"View and up directions not orthogonal for frame {frame_idx}.")
    if view_dir_all is not None:
        stored = np.asarray(view_dir_all[pose_idx], dtype=np.float32)
        if np.linalg.norm(stored - view_dir) > 1e-2:
            raise GeometryValidationError(f"Stored view_direction mismatch for frame {frame_idx}.")
    if up_dir_all is not None:
        stored = np.asarray(up_dir_all[pose_idx], dtype=np.float32)
        if np.linalg.norm(stored - up_dir) > 1e-2:
            raise GeometryValidationError(f"Stored up_direction mismatch for frame {frame_idx}.")


def _validate_depth_consistency(
    store: ResourceStore,
    depth: np.ndarray,
    c2w: np.ndarray,
    w2c: np.ndarray,
    k: np.ndarray,
    image_shape: tuple[int, int],
    frame_idx: int,
    cfg: GeometryValidationConfig,
    capture_overlay: bool,
) -> _DepthValidationResult:
    height, width = image_shape
    valid = np.isfinite(depth) & (depth > 0)
    coverage_mask = _resolve_non_sky_mask(
        store,
        frame_idx=frame_idx,
        shape=image_shape,
    )
    coverage_scope = "full_frame"
    if coverage_mask is not None:
        coverage_mask = np.asarray(coverage_mask, dtype=bool)
        eligible = int(np.count_nonzero(coverage_mask))
        if eligible > 0:
            valid_ratio = float(np.count_nonzero(valid & coverage_mask)) / float(eligible)
            coverage_scope = "non_sky"
        else:
            valid_ratio = 0.0
    else:
        valid_ratio = float(valid.sum()) / float(depth.size)
    if valid_ratio < cfg.min_positive_depth_ratio:
        raise GeometryValidationError(
            f"Depth for frame {frame_idx} has only {valid_ratio:.3f} positive finite values "
            f"within {coverage_scope}."
        )

    ys, xs = _sample_grid_indices(height, width, cfg.max_depth_pixels)
    sample_depth = depth[ys, xs]
    sample_valid = np.isfinite(sample_depth) & (sample_depth > 0)
    if not np.any(sample_valid):
        raise GeometryValidationError(f"No valid depth samples for frame {frame_idx}.")
    xs = xs[sample_valid]
    ys = ys[sample_valid]
    sample_depth = sample_depth[sample_valid]

    uv = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    cam_pts = backproject_uv_depth_to_camera(
        uv,
        sample_depth.astype(np.float32),
        k,
        camera_convention="blender",
    )
    if not np.isfinite(cam_pts).all():
        raise GeometryValidationError(f"Backprojected camera points contain non-finite values for frame {frame_idx}.")

    world_xyz = transform_camera_to_world(cam_pts, c2w)
    cam_reproj_xyz = transform_world_to_camera(world_xyz, world_to_camera_matrix=w2c)
    cam_reproj = np.concatenate(
        [cam_reproj_xyz, np.ones((cam_reproj_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    z_reproj = cam_reproj[:, 2]
    front_mask = z_reproj < 0
    front_ratio = float(np.mean(front_mask))
    if front_ratio < cfg.min_front_ratio:
        raise GeometryValidationError(
            f"Only {front_ratio:.3f} of sampled points are in front of the camera for frame {frame_idx}."
        )

    cam_reproj = cam_reproj[front_mask]
    xs = xs[front_mask]
    ys = ys[front_mask]
    if cam_reproj.size == 0:
        raise GeometryValidationError(f"No valid reprojection points for frame {frame_idx}.")

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    u = fx * (cam_reproj[:, 0] / -cam_reproj[:, 2]) + cx
    v = fy * (-cam_reproj[:, 1] / -cam_reproj[:, 2]) + cy
    err = np.sqrt((u - xs) ** 2 + (v - ys) ** 2)
    reproj_ratio = float(np.mean(err <= cfg.reprojection_error_px))
    if reproj_ratio < cfg.min_reprojection_ratio:
        raise GeometryValidationError(
            f"Reprojection error too high for frame {frame_idx}: {reproj_ratio:.3f} <= {cfg.min_reprojection_ratio}."
        )

    reproj_rmse = float(np.sqrt(np.mean(err**2)))
    overlay = None
    if capture_overlay:
        overlay = ReprojectionOverlayData(
            frame_index=int(frame_idx),
            xs=xs.astype(np.int32),
            ys=ys.astype(np.int32),
            errors_px=err.astype(np.float32),
        )

    return _DepthValidationResult(
        positive_depth_ratio=valid_ratio,
        front_ratio=front_ratio,
        reprojection_inlier_ratio=reproj_ratio,
        reprojection_rmse_px=reproj_rmse,
        overlay=overlay,
    )


def _sample_grid_indices(height: int, width: int, max_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    total = height * width
    if total <= max_pixels:
        ys, xs = np.indices((height, width))
        return ys.reshape(-1), xs.reshape(-1)
    stride = int(np.sqrt(total / max_pixels))
    stride = max(1, stride)
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return grid_y.reshape(-1), grid_x.reshape(-1)


def _resolve_w2c(c2w: np.ndarray, w2c_all: np.ndarray | None, pose_idx: int) -> np.ndarray:
    if w2c_all is not None:
        return np.asarray(w2c_all[pose_idx], dtype=np.float32)
    return np.linalg.inv(c2w)


def _validate_motion_and_relative_transforms(
    sample_frames: Sequence[int],
    traj_index: Mapping[int, int],
    c2w_all: np.ndarray,
    cfg: GeometryValidationConfig,
    logger: logging.Logger,
) -> list[MotionPairMetrics]:
    ordered = sorted(sample_frames)
    if len(ordered) < 2:
        return []

    metrics: list[MotionPairMetrics] = []
    alignment_good = 0
    alignment_total = 0

    translation_values: list[float] = []
    rotation_values: list[float] = []

    prev = ordered[0]
    prev_c2w = np.asarray(c2w_all[traj_index[prev]], dtype=np.float32)
    prev_pos = prev_c2w[:3, 3]
    prev_view = view_direction_from_c2w(prev_c2w)

    for frame_idx in ordered[1:]:
        curr_c2w = np.asarray(c2w_all[traj_index[frame_idx]], dtype=np.float32)
        curr_pos = curr_c2w[:3, 3]
        delta = curr_pos - prev_pos
        dist = float(np.linalg.norm(delta))

        if dist > cfg.max_pose_jump_m:
            raise GeometryValidationError(
                f"Pose jump too large between frames {prev} and {frame_idx}: {dist:.2f}m."
            )

        rel = np.linalg.inv(prev_c2w) @ curr_c2w
        if not np.isfinite(rel).all():
            raise GeometryValidationError(
                f"Relative transform contains non-finite values between frames {prev} and {frame_idx}."
            )

        expected_last_row = np.array([0, 0, 0, 1], dtype=np.float32)
        rel_last_error = float(np.max(np.abs(rel[3, :] - expected_last_row)))
        if rel_last_error > cfg.last_row_tolerance:
            raise GeometryValidationError(
                f"Relative transform last row invalid between frames {prev} and {frame_idx}."
            )

        rel_rot = rel[:3, :3]
        rel_orth_error = float(
            np.max(np.abs(rel_rot.T @ rel_rot - np.eye(3, dtype=np.float32)))
        )
        if rel_orth_error > cfg.rotation_orthonormal_tol:
            raise GeometryValidationError(
                "Relative rotation not orthonormal between frames "
                f"{prev} and {frame_idx}: max error {rel_orth_error:.3e}."
            )

        rel_det = float(np.linalg.det(rel_rot))
        if abs(rel_det - 1.0) > cfg.det_tolerance:
            raise GeometryValidationError(
                f"Relative rotation determinant invalid between frames {prev} and {frame_idx}: {rel_det}."
            )

        trace = float(np.clip((np.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0))
        rotation_deg = float(np.degrees(np.arccos(trace)))

        if rotation_deg > cfg.max_relative_rotation_deg:
            raise GeometryValidationError(
                "Relative rotation too large between frames "
                f"{prev} and {frame_idx}: {rotation_deg:.2f} deg."
            )

        alignment = None
        if dist >= cfg.motion_min_distance_m:
            alignment_total += 1
            direction = delta / dist
            planar_delta = delta.copy()
            planar_delta[2] = 0.0
            planar_dist = float(np.linalg.norm(planar_delta))
            planar_view = prev_view.copy()
            planar_view[2] = 0.0
            planar_view_norm = float(np.linalg.norm(planar_view))
            if planar_dist >= cfg.motion_min_distance_m and planar_view_norm >= 1e-3:
                direction = planar_delta / planar_dist
                view_dir = planar_view / planar_view_norm
                alignment = float(np.dot(direction, view_dir))
            else:
                alignment = float(np.dot(direction, prev_view))

            if alignment >= cfg.min_view_motion_alignment:
                alignment_good += 1
        elif rotation_deg > cfg.max_static_rotation_deg:
            raise GeometryValidationError(
                "Large relative rotation with near-static translation between frames "
                f"{prev} and {frame_idx}: translation={dist:.4f}m, rotation={rotation_deg:.2f}deg."
            )

        translation_values.append(dist)
        rotation_values.append(rotation_deg)
        metrics.append(
            MotionPairMetrics(
                prev_frame_index=int(prev),
                frame_index=int(frame_idx),
                translation_m=dist,
                rotation_deg=rotation_deg,
                view_motion_alignment=alignment,
            )
        )

        prev_pos = curr_pos
        prev_view = view_direction_from_c2w(curr_c2w)
        prev = frame_idx
        prev_c2w = curr_c2w

    if alignment_total == 0:
        logger.info("Skipping motion alignment check: insufficient camera movement.")
    else:
        ratio = alignment_good / alignment_total
        if ratio < cfg.min_view_motion_ratio:
            raise GeometryValidationError(
                f"Camera motion alignment ratio {ratio:.3f} below minimum {cfg.min_view_motion_ratio}."
            )
        logger.info("Motion alignment check passed (ratio=%.3f).", ratio)

    _validate_motion_ratio_outliers(
        values=translation_values,
        threshold_ratio=cfg.max_motion_translation_ratio,
        min_pairs=cfg.min_motion_pairs_for_ratio_checks,
        value_name="relative translation",
    )
    _validate_motion_ratio_outliers(
        values=rotation_values,
        threshold_ratio=cfg.max_motion_rotation_ratio,
        min_pairs=cfg.min_motion_pairs_for_ratio_checks,
        value_name="relative rotation",
    )

    return metrics


def _validate_motion_ratio_outliers(
    *,
    values: Sequence[float],
    threshold_ratio: float,
    min_pairs: int,
    value_name: str,
) -> None:
    if len(values) < min_pairs:
        return
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    if median <= 1e-8:
        return
    max_value = float(np.max(arr))
    ratio = max_value / median
    if ratio > threshold_ratio:
        raise GeometryValidationError(
            f"Suspicious {value_name} spike: max/median ratio {ratio:.2f} exceeds {threshold_ratio:.2f}."
        )


def _validate_camera_height(
    store: ResourceStore,
    frame_idx: int,
    c2w: np.ndarray,
    cfg: GeometryValidationConfig,
    *,
    grounding_shift_z_m: float = 0.0,
    trajectory_metadata: Mapping[str, Any] | None = None,
) -> None:
    height = store.load_camera_height(frame_idx)
    axis = str(height.metadata.get("axis", "z")).lower()
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis)
    if axis_index is None:
        raise GeometryValidationError(f"Camera height axis invalid for frame {frame_idx}: {axis}.")
    world_cs = str(height.metadata.get("world_coordinate_system", "")).lower()
    if world_cs not in {"", "blender"}:
        raise GeometryValidationError(
            f"Camera height world_coordinate_system must be blender for frame {frame_idx}: {world_cs}."
        )
    if axis != "z":
        raise GeometryValidationError(
            f"Camera height axis must be 'z' (Blender world up) for frame {frame_idx}: {axis}."
        )
    validation_mode = ""
    if isinstance(trajectory_metadata, Mapping):
        validation_mode = str(trajectory_metadata.get("height_fit_validation_mode", "")).strip().lower()
        if not validation_mode:
            comparison_frame = trajectory_metadata.get("comparison_frame")
            if (
                isinstance(comparison_frame, Mapping)
                and bool(comparison_frame.get("enabled"))
                and store.has(ResourceKind.ROAD_PLANE)
            ):
                validation_mode = "road_plane_anchor"
    if validation_mode == "road_plane_anchor" and store.has(ResourceKind.ROAD_PLANE):
        plane = store.load_road_plane(frame_idx)
        normal = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-6:
            raise GeometryValidationError(
                f"Road-plane normal is degenerate for frame {frame_idx}; cannot validate camera height anchor."
            )
        normal = normal / norm
        anchor = float(normal @ np.asarray(c2w[:3, 3], dtype=np.float32) + float(plane.offset))
        delta = abs(float(height.height_m) - anchor)
        if delta > cfg.plane_anchor_tolerance_m:
            raise GeometryValidationError(
                f"Camera height mismatch for frame {frame_idx}: stored={height.height_m:.3f}m, "
                f"road_plane_anchor={anchor:.3f}m (tol={cfg.plane_anchor_tolerance_m:.3f})."
            )
        return

    pose_height = float(c2w[axis_index, 3])
    if axis == "z" and abs(float(grounding_shift_z_m)) > 0.0:
        # Camera heights are saved pre-grounding; de-ground trajectory pose height for comparison.
        pose_height -= float(grounding_shift_z_m)
    if bool(height.metadata.get("absolute", False)):
        pose_height = abs(pose_height)
    delta = abs(float(height.height_m) - pose_height)
    if delta > cfg.camera_height_tolerance_m:
        raise GeometryValidationError(
            f"Camera height mismatch for frame {frame_idx}: stored={height.height_m:.3f}m, "
            f"pose={pose_height:.3f}m (grounding_shift_z_m={grounding_shift_z_m:.3f})."
        )
