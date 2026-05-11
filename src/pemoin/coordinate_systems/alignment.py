"""Trajectory alignment and scene-specific correction utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender
from pemoin.coordinate_systems.trajectory_origin import compute_origin_anchor_translation
from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    PointCloud3DData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    RoadPlaneData,
    RoadPlaneSupportData,
)
from pemoin.utils.logging import get_logger
from pemoin.validation.policy import AdaptiveValidationContext, ValidationPolicySettings
from pemoin.visualization.debug_artifacts import (
    write_alignment_scale_diagnostics,
    write_camera_height_alignment_plots,
    write_comparison_frame_plots,
    write_trajectory_path_plots,
)

LOG = get_logger()


def _apply_origin_anchor_to_pose_stack(
    c2w: np.ndarray,
    *,
    anchor_height_m: float,
    metadata: Dict[str, Any],
    metadata_label: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply the standardized origin anchor and annotate trajectory metadata."""
    delta = compute_origin_anchor_translation(c2w, anchor_height_m=anchor_height_m)
    anchored = np.asarray(c2w, dtype=np.float32).copy()
    anchored[:, :3, 3] = np.asarray(anchored[:, :3, 3], dtype=np.float32) + delta.reshape(1, 3)
    meta = dict(metadata)
    meta.update(
        {
            "origin_anchor_enabled": True,
            "origin_anchor_mode": "first_frame_camera_height",
            "origin_anchor_target": [0.0, 0.0, float(anchor_height_m)],
            "origin_anchor_translation": delta.astype(float).tolist(),
            "origin_anchor_metadata_source": str(metadata_label),
        }
    )
    return anchored, delta, meta


@dataclass(frozen=True)
class AlignmentSettings:
    mode: str = "piecewise_plane_anchor"
    fail_on_consistency_error: bool = True
    min_plane_scale_samples: int = 5
    max_plane_scale_iqr_ratio: float = 0.35
    min_plane_scale_inlier_ratio: float = 0.6
    max_plane_anchor_rmse_m: float = 0.25
    max_plane_anchor_abs_err_m: float = 0.5
    segment_min_length: int = 5
    segment_change_zscore: float = 3.0
    segment_persistence_frames: int = 3
    max_segments: int = 6
    transition_frames: int = 2
    max_scale_jump_ratio: float = 1.8
    max_scale_rate_per_frame: float = 0.20
    min_valid_scale_candidates: int = 5
    allow_degraded_output: bool = True
    max_height_rmse_m: float = 0.20
    max_height_abs_err_m: float = 0.40

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "AlignmentSettings":
        raw = mapping or {}
        mode = str(raw.get("mode", cls.mode)).strip().lower()
        if mode == "plane_first":
            mode = "piecewise_plane_anchor"
        if mode not in {"piecewise_plane_anchor"}:
            raise ValueError("alignment.mode must be 'piecewise_plane_anchor'.")
        settings = cls(
            mode=mode,
            fail_on_consistency_error=bool(
                raw.get("fail_on_consistency_error", cls.fail_on_consistency_error)
            ),
            min_plane_scale_samples=int(
                raw.get("min_plane_scale_samples", cls.min_plane_scale_samples)
            ),
            max_plane_scale_iqr_ratio=float(
                raw.get("max_plane_scale_iqr_ratio", cls.max_plane_scale_iqr_ratio)
            ),
            min_plane_scale_inlier_ratio=float(
                raw.get("min_plane_scale_inlier_ratio", cls.min_plane_scale_inlier_ratio)
            ),
            max_plane_anchor_rmse_m=float(
                raw.get("max_plane_anchor_rmse_m", cls.max_plane_anchor_rmse_m)
            ),
            max_plane_anchor_abs_err_m=float(
                raw.get("max_plane_anchor_abs_err_m", cls.max_plane_anchor_abs_err_m)
            ),
            segment_min_length=int(raw.get("segment_min_length", cls.segment_min_length)),
            segment_change_zscore=float(
                raw.get("segment_change_zscore", cls.segment_change_zscore)
            ),
            segment_persistence_frames=int(
                raw.get("segment_persistence_frames", cls.segment_persistence_frames)
            ),
            max_segments=int(raw.get("max_segments", cls.max_segments)),
            transition_frames=int(raw.get("transition_frames", cls.transition_frames)),
            max_scale_jump_ratio=float(
                raw.get("max_scale_jump_ratio", cls.max_scale_jump_ratio)
            ),
            max_scale_rate_per_frame=float(
                raw.get("max_scale_rate_per_frame", cls.max_scale_rate_per_frame)
            ),
            min_valid_scale_candidates=int(
                raw.get("min_valid_scale_candidates", cls.min_valid_scale_candidates)
            ),
            allow_degraded_output=bool(
                raw.get("allow_degraded_output", cls.allow_degraded_output)
            ),
            max_height_rmse_m=float(raw.get("max_height_rmse_m", cls.max_height_rmse_m)),
            max_height_abs_err_m=float(
                raw.get("max_height_abs_err_m", cls.max_height_abs_err_m)
            ),
        )
        if settings.min_plane_scale_samples < 2:
            raise ValueError("alignment.min_plane_scale_samples must be >= 2.")
        if settings.max_plane_scale_iqr_ratio <= 0.0:
            raise ValueError("alignment.max_plane_scale_iqr_ratio must be > 0.")
        if settings.min_plane_scale_inlier_ratio <= 0.0 or settings.min_plane_scale_inlier_ratio > 1.0:
            raise ValueError("alignment.min_plane_scale_inlier_ratio must be in (0, 1].")
        if settings.max_plane_anchor_rmse_m <= 0.0:
            raise ValueError("alignment.max_plane_anchor_rmse_m must be > 0.")
        if settings.max_plane_anchor_abs_err_m <= 0.0:
            raise ValueError("alignment.max_plane_anchor_abs_err_m must be > 0.")
        if settings.segment_min_length < 2:
            raise ValueError("alignment.segment_min_length must be >= 2.")
        if settings.segment_change_zscore <= 0.0:
            raise ValueError("alignment.segment_change_zscore must be > 0.")
        if settings.segment_persistence_frames < 1:
            raise ValueError("alignment.segment_persistence_frames must be >= 1.")
        if settings.max_segments < 1:
            raise ValueError("alignment.max_segments must be >= 1.")
        if settings.transition_frames < 0:
            raise ValueError("alignment.transition_frames must be >= 0.")
        if settings.max_scale_jump_ratio <= 1.0:
            raise ValueError("alignment.max_scale_jump_ratio must be > 1.")
        if settings.max_scale_rate_per_frame <= 0.0:
            raise ValueError("alignment.max_scale_rate_per_frame must be > 0.")
        if settings.min_valid_scale_candidates < 2:
            raise ValueError("alignment.min_valid_scale_candidates must be >= 2.")
        if settings.max_height_rmse_m <= 0.0:
            raise ValueError("alignment.max_height_rmse_m must be > 0.")
        if settings.max_height_abs_err_m <= 0.0:
            raise ValueError("alignment.max_height_abs_err_m must be > 0.")
        return settings


@dataclass(frozen=True)
class GroundingSettings:
    enabled: bool = True
    source: str = "road_plane"
    fail_if_missing_ground: bool = True
    min_ground_samples: int = 5
    max_abs_ground_shift_m: float = 5.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "GroundingSettings":
        raw = mapping or {}
        source = str(raw.get("source", cls.source)).strip().lower()
        if source not in {"road_plane", "point_cloud_3d", "auto"}:
            raise ValueError("grounding_to_z0.source must be one of: road_plane, point_cloud_3d, auto.")
        settings = cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            source=source,
            fail_if_missing_ground=bool(raw.get("fail_if_missing_ground", cls.fail_if_missing_ground)),
            min_ground_samples=int(raw.get("min_ground_samples", cls.min_ground_samples)),
            max_abs_ground_shift_m=float(raw.get("max_abs_ground_shift_m", cls.max_abs_ground_shift_m)),
        )
        if settings.min_ground_samples <= 0:
            raise ValueError("grounding_to_z0.min_ground_samples must be > 0.")
        if settings.max_abs_ground_shift_m <= 0.0:
            raise ValueError("grounding_to_z0.max_abs_ground_shift_m must be > 0.")
        return settings


@dataclass(frozen=True)
class ComparisonFrameSettings:
    enabled: bool = True
    mode: str = "estimated"
    ground_source: str = "road_plane"
    fail_if_missing_ground: bool = True
    min_ground_samples: int = 5
    max_abs_ground_shift_m: float = 5.0
    min_motion_steps: int = 3
    min_total_xy_travel_m: float = 0.25
    min_direction_concentration: float = 0.2
    gt_max_height_rmse_m: float = 0.35
    gt_max_height_abs_err_m: float = 0.75
    gt_max_ground_drift_range_m: float = 0.5
    estimated_min_median_camera_height_m: float = 0.5
    up_direction_source: str = "estimated_camera_average"
    gravity_prior_provider: str = ""
    gravity_prior_fail_if_unavailable: bool = True
    gravity_prior_min_valid_frames: int = 5
    gravity_prior_max_outlier_angle_deg: float = 20.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ComparisonFrameSettings":
        raw = mapping or {}
        mode = str(raw.get("mode", cls.mode)).strip().lower()
        if mode not in {"gt", "estimated"}:
            raise ValueError("comparison_frame.mode must be 'gt' or 'estimated'.")
        ground_source = str(raw.get("ground_source", cls.ground_source)).strip().lower()
        if ground_source not in {"road_plane", "point_cloud_3d", "auto"}:
            raise ValueError(
                "comparison_frame.ground_source must be one of: road_plane, point_cloud_3d, auto."
            )
        up_direction_source = str(
            raw.get("up_direction_source", cls.up_direction_source)
        ).strip().lower()
        if up_direction_source not in {"estimated_camera_average", "gravity_prior"}:
            raise ValueError(
                "comparison_frame.up_direction_source must be one of: estimated_camera_average, gravity_prior."
            )
        gravity_prior_raw = raw.get("gravity_prior", {})
        if gravity_prior_raw is None:
            gravity_prior_raw = {}
        if not isinstance(gravity_prior_raw, Mapping):
            raise ValueError("comparison_frame.gravity_prior must be an object when provided.")
        gravity_provider = str(
            gravity_prior_raw.get("provider", cls.gravity_prior_provider)
        ).strip().lower()
        settings = cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            mode=mode,
            ground_source=ground_source,
            fail_if_missing_ground=bool(
                raw.get("fail_if_missing_ground", cls.fail_if_missing_ground)
            ),
            min_ground_samples=int(raw.get("min_ground_samples", cls.min_ground_samples)),
            max_abs_ground_shift_m=float(
                raw.get("max_abs_ground_shift_m", cls.max_abs_ground_shift_m)
            ),
            min_motion_steps=int(raw.get("min_motion_steps", cls.min_motion_steps)),
            min_total_xy_travel_m=float(
                raw.get("min_total_xy_travel_m", cls.min_total_xy_travel_m)
            ),
            min_direction_concentration=float(
                raw.get("min_direction_concentration", cls.min_direction_concentration)
            ),
            gt_max_height_rmse_m=float(
                raw.get("gt_max_height_rmse_m", cls.gt_max_height_rmse_m)
            ),
            gt_max_height_abs_err_m=float(
                raw.get("gt_max_height_abs_err_m", cls.gt_max_height_abs_err_m)
            ),
            gt_max_ground_drift_range_m=float(
                raw.get("gt_max_ground_drift_range_m", cls.gt_max_ground_drift_range_m)
            ),
            estimated_min_median_camera_height_m=float(
                raw.get(
                    "estimated_min_median_camera_height_m",
                    cls.estimated_min_median_camera_height_m,
                )
            ),
            up_direction_source=up_direction_source,
            gravity_prior_provider=gravity_provider,
            gravity_prior_fail_if_unavailable=bool(
                gravity_prior_raw.get(
                    "fail_if_unavailable", cls.gravity_prior_fail_if_unavailable
                )
            ),
            gravity_prior_min_valid_frames=int(
                gravity_prior_raw.get(
                    "min_valid_frames", cls.gravity_prior_min_valid_frames
                )
            ),
            gravity_prior_max_outlier_angle_deg=float(
                gravity_prior_raw.get(
                    "max_outlier_angle_deg", cls.gravity_prior_max_outlier_angle_deg
                )
            ),
        )
        if settings.min_ground_samples <= 0:
            raise ValueError("comparison_frame.min_ground_samples must be > 0.")
        if settings.max_abs_ground_shift_m <= 0.0:
            raise ValueError("comparison_frame.max_abs_ground_shift_m must be > 0.")
        if settings.min_motion_steps < 2:
            raise ValueError("comparison_frame.min_motion_steps must be >= 2.")
        if settings.min_total_xy_travel_m <= 0.0:
            raise ValueError("comparison_frame.min_total_xy_travel_m must be > 0.")
        if settings.min_direction_concentration <= 0.0 or settings.min_direction_concentration > 1.0:
            raise ValueError(
                "comparison_frame.min_direction_concentration must be in (0, 1]."
            )
        if settings.gt_max_height_rmse_m <= 0.0:
            raise ValueError("comparison_frame.gt_max_height_rmse_m must be > 0.")
        if settings.gt_max_height_abs_err_m <= 0.0:
            raise ValueError("comparison_frame.gt_max_height_abs_err_m must be > 0.")
        if settings.gt_max_ground_drift_range_m <= 0.0:
            raise ValueError("comparison_frame.gt_max_ground_drift_range_m must be > 0.")
        if settings.estimated_min_median_camera_height_m <= 0.0:
            raise ValueError(
                "comparison_frame.estimated_min_median_camera_height_m must be > 0."
            )
        if settings.up_direction_source == "gravity_prior" and settings.gravity_prior_provider not in {"unity_gt"}:
            raise ValueError(
                "comparison_frame.gravity_prior.provider must be 'unity_gt' when up_direction_source='gravity_prior'."
            )
        if settings.gravity_prior_min_valid_frames <= 0:
            raise ValueError("comparison_frame.gravity_prior.min_valid_frames must be > 0.")
        if settings.gravity_prior_max_outlier_angle_deg <= 0.0:
            raise ValueError(
                "comparison_frame.gravity_prior.max_outlier_angle_deg must be > 0."
            )
        return settings


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _estimate_dominant_motion_direction_xy(
    poses: np.ndarray,
    *,
    min_step_distance_m: float = 1e-3,
    max_step_percentile: float = 95.0,
    min_valid_steps: int = 3,
    min_total_xy_travel_m: float = 0.25,
    min_direction_concentration: float = 0.2,
) -> tuple[Optional[np.ndarray], Dict[str, float | int | str]]:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape [N,4,4], got {poses.shape}.")
    if poses.shape[0] < 2:
        return None, {
            "valid_steps": 0,
            "total_xy_travel_m": 0.0,
            "concentration": 0.0,
            "skipped_reason": "insufficient_frames",
        }

    positions_xy = np.asarray(poses[:, :2, 3], dtype=np.float32)
    step_vecs = np.diff(positions_xy, axis=0)
    step_mags = np.linalg.norm(step_vecs, axis=1)

    valid = np.isfinite(step_vecs).all(axis=1) & np.isfinite(step_mags) & (
        step_mags >= float(min_step_distance_m)
    )
    if not np.any(valid):
        return None, {
            "valid_steps": 0,
            "total_xy_travel_m": 0.0,
            "concentration": 0.0,
            "skipped_reason": "no_valid_motion_steps",
        }

    valid_vecs = step_vecs[valid]
    valid_mags = step_mags[valid]
    clip_threshold = float(np.percentile(valid_mags, float(max_step_percentile)))
    if not np.isfinite(clip_threshold) or clip_threshold <= 0.0:
        clip_threshold = float(np.max(valid_mags))
    robust_keep = valid_mags <= clip_threshold
    if np.any(robust_keep):
        valid_vecs = valid_vecs[robust_keep]
        valid_mags = valid_mags[robust_keep]

    valid_steps = int(valid_vecs.shape[0])
    total_xy_travel_m = float(np.sum(valid_mags))
    if valid_steps < int(min_valid_steps):
        return None, {
            "valid_steps": valid_steps,
            "total_xy_travel_m": total_xy_travel_m,
            "concentration": 0.0,
            "skipped_reason": "too_few_valid_steps",
        }
    if total_xy_travel_m < float(min_total_xy_travel_m):
        return None, {
            "valid_steps": valid_steps,
            "total_xy_travel_m": total_xy_travel_m,
            "concentration": 0.0,
            "skipped_reason": "low_total_xy_travel",
        }

    weighted_sum = np.sum(valid_vecs, axis=0)
    dominant_norm = float(np.linalg.norm(weighted_sum))
    if dominant_norm < 1e-8:
        return None, {
            "valid_steps": valid_steps,
            "total_xy_travel_m": total_xy_travel_m,
            "concentration": 0.0,
            "skipped_reason": "degenerate_dominant_direction",
        }

    concentration = dominant_norm / max(float(np.sum(valid_mags)), 1e-8)
    if concentration < float(min_direction_concentration):
        return None, {
            "valid_steps": valid_steps,
            "total_xy_travel_m": total_xy_travel_m,
            "concentration": concentration,
            "skipped_reason": "low_direction_concentration",
        }

    dominant_xy = (weighted_sum / dominant_norm).astype(np.float32)
    return dominant_xy, {
        "valid_steps": valid_steps,
        "total_xy_travel_m": total_xy_travel_m,
        "concentration": concentration,
        "skipped_reason": "",
    }


def _compute_yaw_to_target_axis_xy(dominant_xy: np.ndarray, *, target_xy: np.ndarray) -> float:
    current = np.asarray(dominant_xy, dtype=np.float32).reshape(2)
    target = np.asarray(target_xy, dtype=np.float32).reshape(2)

    current_norm = float(np.linalg.norm(current))
    target_norm = float(np.linalg.norm(target))
    if current_norm < 1e-8 or target_norm < 1e-8:
        raise ValueError("Cannot compute yaw: dominant or target direction is near zero.")

    current = current / current_norm
    target = target / target_norm
    cross_z = float(current[0] * target[1] - current[1] * target[0])
    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    return float(np.arctan2(cross_z, dot))


def _apply_global_yaw_rotation(poses: np.ndarray, yaw_rad: float) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape [N,4,4], got {poses.shape}.")
    r_yaw = _rotation_z(float(yaw_rad))
    updated = np.asarray(poses, dtype=np.float32).copy()
    updated[:, :3, :3] = np.einsum("ij,fjk->fik", r_yaw, updated[:, :3, :3])
    updated[:, :3, 3] = (r_yaw @ updated[:, :3, 3].T).T
    return updated


def normalize_trajectory_yaw_by_dominant_motion(
    poses: np.ndarray,
    *,
    fail_on_weak_motion: bool = False,
    min_valid_steps: int = 3,
    min_total_xy_travel_m: float = 0.25,
    min_direction_concentration: float = 0.2,
) -> tuple[np.ndarray, Dict[str, Any]]:
    target_xy = np.array([0.0, 1.0], dtype=np.float32)
    dominant_xy, stats = _estimate_dominant_motion_direction_xy(
        poses,
        min_valid_steps=min_valid_steps,
        min_total_xy_travel_m=min_total_xy_travel_m,
        min_direction_concentration=min_direction_concentration,
    )

    meta: Dict[str, Any] = {
        "enabled": True,
        "mode": "dominant_motion_global",
        "target_axis": "+Y",
        "applied": False,
        "yaw_deg": 0.0,
        "confidence": float(stats.get("concentration", 0.0)),
        "valid_steps": int(stats.get("valid_steps", 0)),
        "total_xy_travel_m": float(stats.get("total_xy_travel_m", 0.0)),
        "skipped_reason": str(stats.get("skipped_reason", "")),
    }

    if dominant_xy is None:
        if fail_on_weak_motion:
            raise ValueError(
                "Dominant-motion yaw normalization is required but motion is insufficient: "
                f"reason={meta['skipped_reason'] or 'unknown'} "
                f"valid_steps={int(meta['valid_steps'])} "
                f"total_xy_travel_m={float(meta['total_xy_travel_m']):.4f} "
                f"concentration={float(meta['confidence']):.4f}."
            )
        LOG.warning(
            "!!! YAW NORMALIZATION SKIPPED !!! reason=%s valid_steps=%d total_xy_travel_m=%.4f concentration=%.4f",
            meta["skipped_reason"] or "unknown",
            int(meta["valid_steps"]),
            float(meta["total_xy_travel_m"]),
            float(meta["confidence"]),
        )
        return np.asarray(poses, dtype=np.float32).copy(), meta

    yaw_rad = _compute_yaw_to_target_axis_xy(dominant_xy, target_xy=target_xy)
    yaw_deg = float(np.degrees(yaw_rad))
    normalized = _apply_global_yaw_rotation(poses, yaw_rad)
    meta["applied"] = True
    meta["yaw_deg"] = yaw_deg
    meta["skipped_reason"] = ""
    return normalized, meta


def compute_up_direction_alignment(up_vectors: np.ndarray, target_up: Optional[np.ndarray] = None) -> np.ndarray:
    if target_up is None:
        target_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    up_avg = np.asarray(up_vectors, dtype=np.float32)
    if up_avg.ndim == 0 or up_avg.size != 3:
        raise ValueError("Expected a 3-element average up direction vector to compute alignment.")
    norm = float(np.linalg.norm(up_avg))
    if norm < 1e-6:
        raise ValueError("Cannot determine up axis from trajectory - average up vector is zero.")

    up_avg = up_avg / norm
    target_up = np.asarray(target_up, dtype=np.float32)
    if target_up.ndim == 0 or target_up.size != 3:
        raise ValueError("Target up direction must be a 3-element vector.")
    target_up = target_up / float(np.linalg.norm(target_up))

    v = np.cross(up_avg, target_up)
    s = float(np.linalg.norm(v))
    c = float(np.dot(up_avg, target_up))

    if s < 1e-6:
        return np.eye(3, dtype=np.float32)

    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1 - c) / (s**2))


def _write_height_debug_visualization(
    store: ResourceStore,
    frame_indices: np.ndarray,
    c2w_raw: np.ndarray,
    c2w_aligned: np.ndarray,
    heights: Dict[int, Any],
) -> None:
    axis = str(next(iter(heights.values())).metadata.get("axis", "z")).lower()
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, 2)

    raw = c2w_raw[:, axis_index, 3]
    corrected = c2w_aligned[:, axis_index, 3]
    target = np.array([heights[int(idx)].height_m for idx in frame_indices], dtype=np.float32)

    vis_dir = store.visualizations_dir("alignment")
    vis_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_camera_height_alignment_plots(
            vis_dir,
            frame_indices=frame_indices,
            raw_height=raw,
            corrected_height=corrected,
            target_height=target,
        )
    except Exception:
        return


def _validate_alignment_inputs(
    frame_indices: np.ndarray,
    c2w: np.ndarray,
    heights: Dict[int, CameraHeightData],
) -> None:
    if frame_indices.ndim != 1:
        raise ValueError(f"Trajectory frame_indices must be 1D, got {frame_indices.shape}.")
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Trajectory camera_to_world must have shape [N,4,4], got {c2w.shape}.")
    if c2w.shape[0] != frame_indices.shape[0]:
        raise ValueError(
            "Trajectory frame count mismatch: "
            f"frame_indices={frame_indices.shape[0]} camera_to_world={c2w.shape[0]}."
        )
    if frame_indices.size < 2:
        raise ValueError("Alignment requires at least 2 trajectory frames.")
    if np.unique(frame_indices).size != frame_indices.size:
        raise ValueError("Trajectory frame_indices contain duplicates.")
    if not np.all(np.isfinite(c2w)):
        raise ValueError("Trajectory contains non-finite pose values.")
    missing_heights = [int(idx) for idx in frame_indices.tolist() if int(idx) not in heights]
    if missing_heights:
        preview = ", ".join(str(v) for v in missing_heights[:8])
        raise ValueError(
            "Camera height data is missing for trajectory frames: "
            f"{preview}{'...' if len(missing_heights) > 8 else ''}."
        )


def _write_alignment_debug_artifacts(
    store: ResourceStore,
    *,
    frame_indices: np.ndarray,
    scale_diag: Mapping[str, Any],
    height_fit_diag: Mapping[str, Any],
    transform_metadata: Mapping[str, Any],
) -> None:
    vis_dir = store.visualizations_dir("alignment")
    vis_dir.mkdir(parents=True, exist_ok=True)

    frame_details = scale_diag.get("frame_details", [])
    if not isinstance(frame_details, list):
        frame_details = []
    frame_map: Dict[int, Dict[str, Any]] = {}
    for item in frame_details:
        if not isinstance(item, Mapping):
            continue
        frame_val = item.get("frame")
        if frame_val is None:
            continue
        frame_map[int(frame_val)] = dict(item)

    ordered_frames: list[int] = []
    ordered_scales: list[float] = []
    ordered_apparent: list[float] = []
    ordered_target: list[float] = []
    for frame_idx in frame_indices.tolist():
        detail = frame_map.get(int(frame_idx))
        if detail is None:
            continue
        ordered_frames.append(int(frame_idx))
        ordered_scales.append(float(detail.get("frame_scale", np.nan)))
        ordered_apparent.append(float(detail.get("apparent_height", np.nan)))
        ordered_target.append(float(detail.get("gt_height", np.nan)))

    summary_payload: Dict[str, Any] = {
        "scale_diagnostics": dict(scale_diag),
        "height_fit": dict(height_fit_diag),
        "transform": dict(transform_metadata),
    }
    try:
        write_alignment_scale_diagnostics(
            vis_dir,
            frame_indices=ordered_frames,
            frame_scales=ordered_scales,
            apparent_heights=ordered_apparent,
            target_heights=ordered_target,
            summary=summary_payload,
        )
    except Exception as exc:
        LOG.warning("[Alignment] Failed to write alignment scale diagnostics: %s", exc)

    # Keep a plain JSON copy for quick grep/filtering.
    try:
        (vis_dir / "alignment_diagnostics.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        LOG.warning("[Alignment] Failed to write alignment diagnostics json: %s", exc)


def apply_height_correction(
    poses: np.ndarray,
    heights: Dict[int, CameraHeightData],
    frame_indices: np.ndarray,
    *,
    axis: str = "z",
) -> Tuple[np.ndarray, float]:
    if not heights:
        raise ValueError("Camera height data is required for height correction.")

    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if axis_index is None:
        raise ValueError(f"Height correction axis must be one of x, y, z; got {axis!r}.")

    height_vals = []
    for idx, frame_idx in enumerate(frame_indices):
        height = heights[int(frame_idx)]
        z = float(poses[idx, axis_index, 3])
        if bool(height.metadata.get("absolute", False)):
            z = abs(z)
        height_vals.append(z)

    target_vals = np.asarray([float(heights[int(frame_idx)].height_m) for frame_idx in frame_indices], dtype=np.float32)
    observed_vals = np.asarray(height_vals, dtype=np.float32)
    offset = float(np.median(target_vals - observed_vals))

    corrected_poses = poses.copy()
    corrected_poses[:, axis_index, 3] += offset
    return corrected_poses, offset


def _normalize_camera_heights_to_blender(heights: Dict[int, CameraHeightData]) -> Dict[int, CameraHeightData]:
    normalized: Dict[int, CameraHeightData] = {}
    for frame_idx, height in heights.items():
        meta = dict(height.metadata or {})
        source_world = str(meta.get("world_coordinate_system", "")).lower()
        source_axis = str(meta.get("axis", "z")).lower()
        if source_axis not in {"x", "y", "z"}:
            source_axis = "z"
        meta.update(
            {
                "world_coordinate_system": "blender",
                "axis": "z",
                "source_world_coordinate_system": source_world or None,
                "source_axis": source_axis,
                "conversion": "axis_remap",
            }
        )
        normalized[frame_idx] = CameraHeightData(
            frame_index=height.frame_index,
            height_m=float(height.height_m),
            metadata=meta,
        )
    return normalized


def _safe_float(meta: Mapping[str, Any], key: str) -> float | None:
    value = meta.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _estimate_scale_from_road_planes(
    store: ResourceStore,
    aligned_c2w: np.ndarray,
    heights: Dict[int, CameraHeightData],
    frame_indices: np.ndarray,
    *,
    min_samples: int,
    max_iqr_ratio: float,
    min_inlier_ratio: float,
    allow_unstable: bool,
) -> tuple[float, Dict[str, Any]]:
    if not store.has(ResourceKind.ROAD_PLANE):
        raise ValueError("Road-plane data is required for plane-first scale estimation.")

    scales: list[float] = []
    per_frame: list[Dict[str, Any]] = []
    skipped: Dict[str, int] = {
        "load_error": 0,
        "metadata_missing": 0,
        "measurement_rejected": 0,
        "support_quality_rejected": 0,
        "fit_quality_rejected": 0,
        "degenerate_normal": 0,
        "invalid_apparent_height": 0,
        "invalid_frame_scale": 0,
    }
    for i, frame_idx in enumerate(frame_indices.tolist()):
        try:
            plane = store.load_road_plane(int(frame_idx))
        except Exception:
            skipped["load_error"] += 1
            per_frame.append({"frame": int(frame_idx), "accepted": False, "reason": "load_error"})
            continue
        meta = dict(plane.metadata or {})
        required = ("measurement_allowed", "support_quality_ok", "residual_p90", "inlier_ratio")
        missing = [key for key in required if key not in meta]
        if missing:
            skipped["metadata_missing"] += 1
            per_frame.append(
                {
                    "frame": int(frame_idx),
                    "accepted": False,
                    "reason": "metadata_missing",
                    "missing_keys": tuple(missing),
                }
            )
            continue
        if bool(meta.get("measurement_allowed", True)) is False:
            skipped["measurement_rejected"] += 1
            per_frame.append({"frame": int(frame_idx), "accepted": False, "reason": "measurement_rejected"})
            continue
        if bool(meta.get("support_quality_ok", True)) is False:
            skipped["support_quality_rejected"] += 1
            per_frame.append({"frame": int(frame_idx), "accepted": False, "reason": "support_quality_rejected"})
            continue
        residual_p90 = _safe_float(meta, "residual_p90")
        inlier_ratio = _safe_float(meta, "inlier_ratio")
        if residual_p90 is None or inlier_ratio is None:
            skipped["metadata_missing"] += 1
            per_frame.append({"frame": int(frame_idx), "accepted": False, "reason": "invalid_quality_metadata"})
            continue
        fit_quality_ok = bool(residual_p90 <= 0.40 and inlier_ratio >= 0.30)
        if not fit_quality_ok:
            skipped["fit_quality_rejected"] += 1
            per_frame.append(
                {
                    "frame": int(frame_idx),
                    "accepted": False,
                    "reason": "fit_quality_rejected",
                    "residual_p90": float(residual_p90),
                    "inlier_ratio": float(inlier_ratio),
                }
            )
            continue
        normal = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            skipped["degenerate_normal"] += 1
            per_frame.append({"frame": int(frame_idx), "accepted": False, "reason": "degenerate_normal"})
            continue
        normal = normal / norm
        offset = float(plane.offset)
        cam_pos = np.asarray(aligned_c2w[i, :3, 3], dtype=np.float32)
        apparent_height = float(np.dot(normal, cam_pos) + offset)
        if apparent_height < 0.0:
            apparent_height = -apparent_height
        if not np.isfinite(apparent_height) or apparent_height < 1e-6:
            skipped["invalid_apparent_height"] += 1
            per_frame.append({"frame": int(frame_idx), "accepted": False, "reason": "invalid_apparent_height"})
            continue
        gt_height = float(heights[int(frame_idx)].height_m)
        frame_scale = float(gt_height / apparent_height)
        if not np.isfinite(frame_scale) or frame_scale <= 0.0:
            skipped["invalid_frame_scale"] += 1
            per_frame.append({"frame": int(frame_idx), "accepted": False, "reason": "invalid_frame_scale"})
            continue
        scales.append(frame_scale)
        per_frame.append(
            {
                "frame": int(frame_idx),
                "accepted": True,
                "reason": "ok",
                "gt_height": float(gt_height),
                "apparent_height": float(apparent_height),
                "frame_scale": float(frame_scale),
                "residual_p90": float(residual_p90),
                "inlier_ratio": float(inlier_ratio),
            }
        )

    if len(scales) < int(min_samples):
        raise ValueError(
            f"Plane-first scale estimation has only {len(scales)} valid samples; need at least {min_samples}."
        )

    arr = np.asarray(scales, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1)
    low = float(q1 - 1.5 * iqr)
    high = float(q3 + 1.5 * iqr)
    keep = (arr >= low) & (arr <= high)
    arr_f = arr[keep]
    degraded_reasons: list[str] = []
    if arr_f.size < int(min_samples):
        if not allow_unstable:
            raise ValueError(
                f"Plane-first scale estimation kept only {arr_f.size} inliers after IQR filtering; need {min_samples}."
            )
        degraded_reasons.append("insufficient_iqr_inliers")
        arr_f = arr

    scale = float(np.median(arr_f))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Plane-first scale estimation returned non-finite or non-positive scale.")

    iqr_ratio = float(iqr / max(abs(scale), 1e-8))
    if iqr_ratio > float(max_iqr_ratio):
        if not allow_unstable:
            raise ValueError(
                "Plane-first scale estimation is unstable: "
                f"IQR ratio {iqr_ratio:.4f} exceeds threshold {max_iqr_ratio:.4f}."
            )
        degraded_reasons.append("high_iqr_ratio")
    inlier_fraction = float(arr_f.size / max(arr.size, 1))
    if inlier_fraction < float(min_inlier_ratio):
        if not allow_unstable:
            raise ValueError(
                "Plane-first scale estimation has insufficient inlier support: "
                f"inlier_fraction {inlier_fraction:.4f} < {min_inlier_ratio:.4f}."
            )
        degraded_reasons.append("low_inlier_fraction")
    keep_idx = 0
    for item in per_frame:
        if not bool(item.get("accepted", False)):
            item["inlier"] = False
            continue
        item["inlier"] = bool(keep[keep_idx])
        keep_idx += 1
    inlier_frames = [int(item["frame"]) for item in per_frame if bool(item.get("inlier", False))]

    diag = {
        "method": "road_plane_camera_height",
        "sampled_frames": int(frame_indices.size),
        "valid_samples": int(arr.size),
        "inlier_samples": int(arr_f.size),
        "inlier_fraction": float(inlier_fraction),
        "scale_median": float(scale),
        "scale_std": float(np.std(arr_f)),
        "scale_min": float(np.min(arr_f)),
        "scale_max": float(np.max(arr_f)),
        "scale_iqr": iqr,
        "scale_iqr_ratio": iqr_ratio,
        "inlier_frames": inlier_frames,
        "degraded_reasons": degraded_reasons,
        "skipped": skipped,
        "frame_details": per_frame,
    }
    LOG.info(
        "[Alignment] Plane-first scale fit: sampled=%d valid=%d inliers=%d "
        "scale=%.6f std=%.6f iqr_ratio=%.6f skipped=%s",
        int(diag["sampled_frames"]),
        int(diag["valid_samples"]),
        int(diag["inlier_samples"]),
        float(diag["scale_median"]),
        float(diag["scale_std"]),
        float(diag["scale_iqr_ratio"]),
        skipped,
    )
    if degraded_reasons:
        LOG.warning(
            "[Alignment] Plane-first candidate set is unstable; continuing in degraded mode with reasons=%s",
            ",".join(degraded_reasons),
        )
    return scale, diag


def _validate_camera_height_fit(
    poses: np.ndarray,
    heights: Dict[int, CameraHeightData],
    frame_indices: np.ndarray,
    *,
    max_rmse_m: float,
    max_abs_err_m: float,
) -> Dict[str, float]:
    observed = np.asarray(poses[:, 2, 3], dtype=np.float64)
    target = np.asarray([float(heights[int(frame_idx)].height_m) for frame_idx in frame_indices], dtype=np.float64)
    errors = observed - target
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_abs = float(np.max(np.abs(errors)))
    if rmse > float(max_rmse_m):
        raise ValueError(
            f"Alignment RMSE against camera height too high: {rmse:.4f} m > {max_rmse_m:.4f} m."
        )
    if max_abs > float(max_abs_err_m):
        raise ValueError(
            f"Alignment max absolute error too high: {max_abs:.4f} m > {max_abs_err_m:.4f} m."
        )
    return {
        "height_rmse_m": rmse,
        "height_max_abs_err_m": max_abs,
        "height_median_err_m": float(np.median(errors)),
    }


def _validate_canonical_camera_up(
    poses: np.ndarray,
    *,
    min_up_dot_z: float = 0.0,
    min_median_up_dot_z: float = 0.5,
) -> Dict[str, float | bool]:
    up_vectors = np.asarray(poses[:, :3, 1], dtype=np.float64)
    norms = np.linalg.norm(up_vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-6):
        raise ValueError("Canonical camera-up validation failed: degenerate up vectors.")
    up_unit = up_vectors / norms.reshape(-1, 1)
    dots = up_unit @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    min_dot = float(np.min(dots))
    median_dot = float(np.median(dots))
    max_dot = float(np.max(dots))
    if min_dot <= float(min_up_dot_z):
        raise ValueError(
            "Canonical camera-up validation failed: "
            f"min(up·+Z)={min_dot:.4f} <= {float(min_up_dot_z):.4f}."
        )
    if median_dot < float(min_median_up_dot_z):
        raise ValueError(
            "Canonical camera-up validation failed: "
            f"median(up·+Z)={median_dot:.4f} < {float(min_median_up_dot_z):.4f}."
        )
    return {
        "up_dot_z_min": min_dot,
        "up_dot_z_median": median_dot,
        "up_dot_z_max": max_dot,
        "frames_checked": float(dots.size),
        "passed": True,
    }


def _canonicalize_support_plane_orientation(
    normal: np.ndarray,
    offset: float,
    cam_pos: np.ndarray,
    *,
    target_height_m: float | None = None,
) -> tuple[np.ndarray, float, bool, float, float]:
    n = np.asarray(normal, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(n))
    if norm < 1e-6:
        raise ValueError("Road plane normal is degenerate.")
    n = n / norm
    c = np.asarray(cam_pos, dtype=np.float32).reshape(3)

    best_valid: tuple[np.ndarray, float, bool, float, float, float] | None = None
    for sign, flipped in ((1.0, False), (-1.0, True)):
        n_cand = (sign * n).astype(np.float32)
        d_cand = float(sign * float(offset))
        signed_anchor = float(np.dot(n_cand, c) + d_cand)
        ground_point = c - signed_anchor * n_cand
        if not np.isfinite(ground_point).all():
            continue
        anchor_positive = signed_anchor > 0.0
        ground_below = float(ground_point[2]) < float(c[2])
        if not (anchor_positive and ground_below):
            continue
        if target_height_m is None or not np.isfinite(float(target_height_m)):
            height_err = abs(signed_anchor)
        else:
            height_err = abs(signed_anchor - float(target_height_m))
        candidate = (
            n_cand,
            d_cand,
            flipped,
            signed_anchor,
            float(c[2] - ground_point[2]),
            float(height_err),
        )
        if best_valid is None or candidate[-1] < best_valid[-1]:
            best_valid = candidate

    if best_valid is None:
        raise ValueError("Road plane cannot be oriented as a valid support surface below the camera.")

    normal_out, offset_out, flipped_out, anchor_out, support_delta_out, _ = best_valid
    return normal_out, float(offset_out), bool(flipped_out), float(anchor_out), float(support_delta_out)


def _compute_rigid_canonical_alignment(
    c2w: np.ndarray,
    heights: Dict[int, CameraHeightData],
    frame_indices: np.ndarray,
    *,
    target_up: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any], Dict[str, float]]:
    up_vectors = np.asarray(c2w[:, :3, 1], dtype=np.float32)
    up_avg = np.mean(up_vectors, axis=0)
    if target_up is None:
        target_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        target_up = np.asarray(target_up, dtype=np.float32).reshape(3)
    r_align = compute_up_direction_alignment(up_avg, target_up)
    LOG.debug(
        "[Alignment] Up alignment: up_avg=%s target_up=%s",
        np.array2string(up_avg, precision=4, suppress_small=True),
        np.array2string(target_up, precision=4, suppress_small=True),
    )

    aligned_c2w = np.asarray(c2w, dtype=np.float32).copy()
    aligned_c2w[:, :3, :3] = np.einsum("ij,fjk->fik", r_align, aligned_c2w[:, :3, :3])
    aligned_c2w[:, :3, 3] = (r_align @ aligned_c2w[:, :3, 3].T).T

    aligned_c2w, offset = apply_height_correction(aligned_c2w, heights, frame_indices, axis="z")
    aligned_c2w, yaw_meta = normalize_trajectory_yaw_by_dominant_motion(aligned_c2w)
    LOG.info(
        "[Alignment] Applied rigid canonicalization: z_offset=%.6f yaw_applied=%s yaw_deg=%.4f",
        float(offset),
        bool(yaw_meta.get("applied", False)),
        float(yaw_meta.get("yaw_deg", 0.0)),
    )

    yaw_rad = np.radians(float(yaw_meta.get("yaw_deg", 0.0))) if bool(yaw_meta.get("applied", False)) else 0.0
    r_yaw = _rotation_z(float(yaw_rad))
    r_total = (r_yaw @ r_align).astype(np.float32)
    t_total = (r_yaw @ np.array([0.0, 0.0, float(offset)], dtype=np.float32)).astype(np.float32)

    height_fit_diag = _validate_camera_height_fit(
        aligned_c2w,
        heights,
        frame_indices,
        max_rmse_m=float(np.inf),
        max_abs_err_m=float(np.inf),
    )
    LOG.info(
        "[Alignment] Height fit: rmse=%.6f max_abs=%.6f median_err=%.6f",
        float(height_fit_diag.get("height_rmse_m", np.nan)),
        float(height_fit_diag.get("height_max_abs_err_m", np.nan)),
        float(height_fit_diag.get("height_median_err_m", np.nan)),
    )

    aligned_w2c = np.linalg.inv(aligned_c2w)
    return aligned_c2w, aligned_w2c, r_total, t_total, float(offset), dict(yaw_meta), height_fit_diag


def _evaluate_transformed_road_plane_anchor_fit(
    store: ResourceStore,
    aligned_c2w: np.ndarray,
    heights: Dict[int, CameraHeightData],
    frame_indices: np.ndarray,
    *,
    inlier_frames: set[int],
    r_total: np.ndarray,
    scale: float,
    t_total: np.ndarray,
    max_rmse_m: float,
    max_abs_err_m: float,
) -> Dict[str, float | bool]:
    if not store.has(ResourceKind.ROAD_PLANE):
        raise ValueError("Road-plane data is required for post-alignment anchor validation.")
    abs_errors: list[float] = []
    signed_errors: list[float] = []
    support_delta_z: list[float] = []
    sign_flips = 0
    for pose_idx, frame_idx in enumerate(frame_indices.tolist()):
        if int(frame_idx) not in inlier_frames:
            continue
        plane = store.load_road_plane(int(frame_idx))
        normal = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal = normal / norm
        normal_t = (np.asarray(r_total, dtype=np.float32) @ normal).astype(np.float32)
        offset_t = float(scale * float(plane.offset) - float(np.dot(normal_t, np.asarray(t_total, dtype=np.float32))))
        cam_pos = np.asarray(aligned_c2w[pose_idx, :3, 3], dtype=np.float32)
        target = float(heights[int(frame_idx)].height_m)
        try:
            normal_t, offset_t, flipped, signed_anchor, support_delta = _canonicalize_support_plane_orientation(
                normal_t,
                offset_t,
                cam_pos,
                target_height_m=target,
            )
        except ValueError as exc:
            raise ValueError(
                f"Post-alignment road-plane support is invalid for frame {int(frame_idx)}: {exc}"
            ) from exc
        abs_errors.append(abs(signed_anchor - target))
        signed_errors.append(signed_anchor - target)
        support_delta_z.append(float(support_delta))
        if flipped:
            sign_flips += 1
    if not signed_errors:
        raise ValueError("Post-alignment anchor validation has no eligible inlier frames.")
    abs_arr = np.asarray(abs_errors, dtype=np.float64)
    signed_arr = np.asarray(signed_errors, dtype=np.float64)
    delta_arr = np.asarray(support_delta_z, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(abs_arr**2)))
    max_abs = float(np.max(np.abs(abs_arr)))
    signed_rmse = float(np.sqrt(np.mean(signed_arr**2)))
    signed_max_abs = float(np.max(np.abs(signed_arr)))
    if signed_rmse > float(max_rmse_m):
        raise ValueError(
            f"Post-alignment road-plane signed anchor RMSE too high: {signed_rmse:.4f} m > {max_rmse_m:.4f} m."
        )
    if signed_max_abs > float(max_abs_err_m):
        raise ValueError(
            f"Post-alignment road-plane signed anchor max absolute error too high: {signed_max_abs:.4f} m > {max_abs_err_m:.4f} m."
        )
    return {
        "plane_anchor_rmse_m": rmse,
        "plane_anchor_max_abs_err_m": max_abs,
        "plane_anchor_median_err_m": float(np.median(abs_arr)),
        "plane_anchor_signed_rmse_m": signed_rmse,
        "plane_anchor_signed_max_abs_err_m": signed_max_abs,
        "plane_anchor_signed_median_err_m": float(np.median(signed_arr)),
        "plane_anchor_samples": float(signed_arr.size),
        "support_surface_below_camera": True,
        "support_surface_median_delta_z_m": float(np.median(delta_arr)),
        "support_plane_sign_flips": float(sign_flips),
        "support_plane_frames_checked": float(signed_arr.size),
        "support_plane_orientation_canonicalized": True,
    }


def _robust_mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad


def _segment_scale_regimes(
    accepted_frames: np.ndarray,
    accepted_scales: np.ndarray,
    *,
    segment_min_length: int,
    segment_change_zscore: float,
    segment_persistence_frames: int,
    max_segments: int,
) -> list[dict[str, Any]]:
    if accepted_frames.size != accepted_scales.size:
        raise ValueError("accepted frame/scale arrays must have equal size.")
    if accepted_frames.size == 0:
        return []
    regimes: list[dict[str, Any]] = []
    start = 0
    pending_start = -1
    pending_count = 0
    i = 0
    while i < accepted_scales.size:
        current = accepted_scales[start : i + 1]
        med = float(np.median(current))
        sigma = max(_robust_mad(current), 1e-6)
        z = abs(float(accepted_scales[i] - med) / sigma)
        can_split = (i - start + 1) >= int(segment_min_length)
        if can_split and z > float(segment_change_zscore):
            if pending_start < 0:
                pending_start = i
                pending_count = 1
            else:
                pending_count += 1
            if pending_count >= int(segment_persistence_frames):
                split_idx = pending_start
                regime_scales = accepted_scales[start:split_idx]
                if regime_scales.size >= int(segment_min_length):
                    regimes.append(
                        {
                            "start_idx": int(start),
                            "end_idx": int(split_idx - 1),
                            "start_frame": int(accepted_frames[start]),
                            "end_frame": int(accepted_frames[split_idx - 1]),
                            "scale": float(np.median(regime_scales)),
                            "samples": int(regime_scales.size),
                            "scale_iqr": float(np.percentile(regime_scales, 75) - np.percentile(regime_scales, 25)),
                        }
                    )
                    start = split_idx
                    pending_start = -1
                    pending_count = 0
                    if len(regimes) >= int(max_segments - 1):
                        break
                    i = start
                    continue
        else:
            pending_start = -1
            pending_count = 0
        i += 1
    tail = accepted_scales[start:]
    if tail.size > 0:
        regimes.append(
            {
                "start_idx": int(start),
                "end_idx": int(accepted_scales.size - 1),
                "start_frame": int(accepted_frames[start]),
                "end_frame": int(accepted_frames[-1]),
                "scale": float(np.median(tail)),
                "samples": int(tail.size),
                "scale_iqr": float(np.percentile(tail, 75) - np.percentile(tail, 25)),
            }
        )
    return regimes


def _build_piecewise_scale_series(
    frame_indices: np.ndarray,
    frame_details: Sequence[Mapping[str, Any]],
    *,
    settings: AlignmentSettings,
) -> tuple[np.ndarray, Dict[str, Any]]:
    frame_to_detail: Dict[int, Mapping[str, Any]] = {}
    for item in frame_details:
        frame = int(item.get("frame", -1))
        if frame > 0:
            frame_to_detail[frame] = item
    accepted_frames: list[int] = []
    accepted_scales: list[float] = []
    for frame_idx in frame_indices.tolist():
        item = frame_to_detail.get(int(frame_idx))
        if not item or not bool(item.get("accepted", False)):
            continue
        fs = float(item.get("frame_scale", np.nan))
        if not np.isfinite(fs) or fs <= 0.0:
            continue
        accepted_frames.append(int(frame_idx))
        accepted_scales.append(fs)
    if len(accepted_frames) < int(settings.min_valid_scale_candidates):
        raise ValueError(
            "Insufficient valid per-frame scale candidates: "
            f"{len(accepted_frames)} < {settings.min_valid_scale_candidates}."
        )
    acc_frames = np.asarray(accepted_frames, dtype=np.int32)
    acc_scales = np.asarray(accepted_scales, dtype=np.float64)
    regimes = _segment_scale_regimes(
        acc_frames,
        acc_scales,
        segment_min_length=settings.segment_min_length,
        segment_change_zscore=settings.segment_change_zscore,
        segment_persistence_frames=settings.segment_persistence_frames,
        max_segments=settings.max_segments,
    )
    if not regimes:
        raise ValueError("No stable scale regimes could be identified.")
    if len(regimes) > int(settings.max_segments):
        raise ValueError(
            f"Scale regime count {len(regimes)} exceeds max_segments={settings.max_segments}."
        )

    # Assign accepted frames to regime scales.
    acc_piecewise = np.zeros_like(acc_scales, dtype=np.float64)
    for reg in regimes:
        s = int(reg["start_idx"])
        e = int(reg["end_idx"]) + 1
        acc_piecewise[s:e] = float(reg["scale"])
    # Smooth transitions between adjacent regimes.
    t = int(settings.transition_frames)
    if t > 0 and len(regimes) > 1:
        for ridx in range(len(regimes) - 1):
            left = regimes[ridx]
            right = regimes[ridx + 1]
            boundary = int(left["end_idx"])
            for k in range(1, t + 1):
                li = boundary - t + k
                if li < int(left["start_idx"]) or li >= acc_piecewise.size:
                    continue
                alpha = float(k / (t + 1))
                acc_piecewise[li] = (1.0 - alpha) * float(left["scale"]) + alpha * float(right["scale"])

    # Clamp abrupt jumps/rates on accepted sequence.
    for i in range(1, acc_piecewise.size):
        prev = float(acc_piecewise[i - 1])
        cur = float(acc_piecewise[i])
        if prev <= 0.0:
            continue
        ratio = max(cur / prev, prev / cur) if cur > 0 else np.inf
        if ratio > float(settings.max_scale_jump_ratio):
            cur = prev * (settings.max_scale_jump_ratio if cur > prev else 1.0 / settings.max_scale_jump_ratio)
        max_delta = float(settings.max_scale_rate_per_frame) * prev
        cur = float(np.clip(cur, prev - max_delta, prev + max_delta))
        acc_piecewise[i] = cur

    # Expand to all frames (forward/backward fill from accepted).
    scale_series = np.zeros((frame_indices.size,), dtype=np.float64)
    frame_pos = {int(frame): idx for idx, frame in enumerate(frame_indices.tolist())}
    for frame, scale in zip(acc_frames.tolist(), acc_piecewise.tolist()):
        scale_series[frame_pos[int(frame)]] = float(scale)
    last = 0.0
    for i in range(scale_series.size):
        if scale_series[i] > 0.0:
            last = float(scale_series[i])
        elif last > 0.0:
            scale_series[i] = last
    next_val = 0.0
    for i in range(scale_series.size - 1, -1, -1):
        if scale_series[i] > 0.0:
            next_val = float(scale_series[i])
        elif next_val > 0.0:
            scale_series[i] = next_val
    if not np.all(scale_series > 0.0):
        raise ValueError("Failed to construct positive scale series for all frames.")

    diag = {
        "regimes": regimes,
        "accepted_frames": accepted_frames,
        "accepted_scales_raw": accepted_scales,
        "accepted_scales_piecewise": acc_piecewise.astype(float).tolist(),
        "scale_series": scale_series.astype(float).tolist(),
        "num_regimes": int(len(regimes)),
        "degraded": bool(len(regimes) > 1),
    }
    return scale_series.astype(np.float32), diag


def _validate_post_alignment_plane_anchor_fit(
    store: ResourceStore,
    aligned_c2w: np.ndarray,
    heights: Dict[int, CameraHeightData],
    frame_indices: np.ndarray,
    *,
    inlier_frames: set[int],
    max_rmse_m: float,
    max_abs_err_m: float,
) -> Dict[str, float]:
    return _evaluate_transformed_road_plane_anchor_fit(
        store,
        aligned_c2w,
        heights,
        frame_indices,
        inlier_frames=inlier_frames,
        r_total=np.eye(3, dtype=np.float32),
        scale=1.0,
        t_total=np.zeros((3,), dtype=np.float32),
        max_rmse_m=max_rmse_m,
        max_abs_err_m=max_abs_err_m,
    )


def _apply_transform_to_point_cloud(
    store: ResourceStore,
    *,
    r_total: np.ndarray,
    scale: float,
    t_total: np.ndarray,
    transform_id: str,
) -> None:
    if not store.has(ResourceKind.POINT_CLOUD_3D):
        return
    if not np.all(np.isfinite(r_total)):
        raise ValueError("Alignment rotation contains non-finite values.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Alignment scale must be positive and finite, got {scale}.")
    if not np.all(np.isfinite(t_total)):
        raise ValueError("Alignment translation contains non-finite values.")
    cloud = store.load_point_cloud_3d()
    points = np.asarray(cloud.points_world, dtype=np.float32)
    transformed = ((r_total @ points.T).T * float(scale)) + t_total.reshape(1, 3)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Transformed point cloud contains non-finite coordinates.")
    metadata = dict(cloud.metadata or {})
    metadata["alignment_transform_id"] = transform_id
    metadata["metric_scale"] = True
    store.save_point_cloud_3d(
        PointCloud3DData(
            points_world=transformed.astype(np.float32),
            labels=np.asarray(cloud.labels, dtype=np.int32),
            label_confidences=np.asarray(cloud.label_confidences, dtype=np.float32),
            colors=np.asarray(cloud.colors, dtype=np.uint8),
            label_names=dict(cloud.label_names or {}),
            observation_counts=np.asarray(cloud.observation_counts, dtype=np.int32),
            metadata=metadata,
        )
    )


def _apply_transform_to_road_planes(
    store: ResourceStore,
    *,
    r_total: np.ndarray,
    scale: float,
    t_total: np.ndarray,
    transform_id: str,
) -> None:
    if not store.has(ResourceKind.ROAD_PLANE):
        return
    if not store.has(ResourceKind.TRAJECTORY):
        raise ValueError("Road-plane transforms require trajectory data to canonicalize support orientation.")
    if not np.all(np.isfinite(r_total)):
        raise ValueError("Alignment rotation contains non-finite values.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Alignment scale must be positive and finite, got {scale}.")
    if not np.all(np.isfinite(t_total)):
        raise ValueError("Alignment translation contains non-finite values.")
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    traj_frames = np.asarray(traj["frame_indices"], dtype=np.int32)
    traj_c2w = np.asarray(traj["camera_to_world"], dtype=np.float32)
    camera_map = {
        int(traj_frames[i]): np.asarray(traj_c2w[i, :3, 3], dtype=np.float32)
        for i in range(traj_frames.size)
    }
    for frame_idx in store.frame_indices(ResourceKind.ROAD_PLANE):
        plane = store.load_road_plane(int(frame_idx))
        n = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-6:
            raise ValueError(f"Road plane normal is degenerate for frame {frame_idx}.")
        n = n / n_norm
        n_t = (r_total @ n).astype(np.float32)
        d_t = float(scale * float(plane.offset) - float(np.dot(n_t, t_total)))
        if not np.all(np.isfinite(n_t)) or not np.isfinite(d_t):
            raise ValueError(f"Transformed road plane is non-finite for frame {frame_idx}.")
        cam_pos = camera_map.get(int(frame_idx))
        if cam_pos is None:
            raise ValueError(
                f"Road-plane frame {int(frame_idx)} is missing a matching trajectory pose for support canonicalization."
            )
        target_height = None
        if store.has(ResourceKind.CAMERA_HEIGHT):
            target_height = float(store.load_camera_height(int(frame_idx)).height_m)
        n_t, d_t, flipped, _, _ = _canonicalize_support_plane_orientation(
            n_t,
            d_t,
            cam_pos,
            target_height_m=target_height,
        )
        meta = dict(plane.metadata or {})
        meta["alignment_transform_id"] = transform_id
        meta["metric_scale"] = True
        meta["support_plane_orientation_canonicalized"] = True
        meta["support_plane_sign_flipped"] = bool(flipped)
        store.save_road_plane(
            RoadPlaneData(
                frame_index=int(frame_idx),
                normal=n_t,
                offset=d_t,
                metadata=meta,
            )
        )


def _apply_piecewise_transform_to_road_planes(
    store: ResourceStore,
    *,
    r_total: np.ndarray,
    scale_by_frame: Mapping[int, float],
    t_total: np.ndarray,
    transform_id: str,
) -> None:
    if not store.has(ResourceKind.ROAD_PLANE):
        return
    if not store.has(ResourceKind.TRAJECTORY):
        raise ValueError("Road-plane transforms require trajectory data to canonicalize support orientation.")
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    traj_frames = np.asarray(traj["frame_indices"], dtype=np.int32)
    traj_c2w = np.asarray(traj["camera_to_world"], dtype=np.float32)
    camera_map = {
        int(traj_frames[i]): np.asarray(traj_c2w[i, :3, 3], dtype=np.float32)
        for i in range(traj_frames.size)
    }
    for frame_idx in store.frame_indices(ResourceKind.ROAD_PLANE):
        frame_idx_i = int(frame_idx)
        if frame_idx_i not in scale_by_frame:
            raise ValueError(f"Missing piecewise scale for road-plane frame {frame_idx_i}.")
        scale = float(scale_by_frame[frame_idx_i])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid piecewise scale for road-plane frame {frame_idx_i}: {scale}.")
        plane = store.load_road_plane(frame_idx_i)
        n = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-6:
            raise ValueError(f"Road plane normal is degenerate for frame {frame_idx_i}.")
        n = n / n_norm
        n_t = (r_total @ n).astype(np.float32)
        d_t = float(scale * float(plane.offset) - float(np.dot(n_t, t_total)))
        cam_pos = camera_map.get(frame_idx_i)
        if cam_pos is None:
            raise ValueError(
                f"Road-plane frame {frame_idx_i} is missing a matching trajectory pose for support canonicalization."
            )
        target_height = None
        if store.has(ResourceKind.CAMERA_HEIGHT):
            target_height = float(store.load_camera_height(frame_idx_i).height_m)
        n_t, d_t, flipped, _, _ = _canonicalize_support_plane_orientation(
            n_t,
            d_t,
            cam_pos,
            target_height_m=target_height,
        )
        meta = dict(plane.metadata or {})
        meta["alignment_transform_id"] = transform_id
        meta["metric_scale"] = True
        meta["piecewise_scale_factor"] = float(scale)
        meta["support_plane_orientation_canonicalized"] = True
        meta["support_plane_sign_flipped"] = bool(flipped)
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx_i,
                normal=n_t,
                offset=d_t,
                metadata=meta,
            )
        )


def _apply_transform_to_road_plane_sampled_points(
    store: ResourceStore,
    *,
    r_total: np.ndarray,
    scale: float,
    t_total: np.ndarray,
    transform_id: str,
) -> int:
    frame_indices = store.frame_indices(ResourceKind.ROAD_PLANE_SUPPORT)
    if not frame_indices:
        return 0
    updated = 0
    for frame_idx in frame_indices:
        support = store.load_road_plane_support(int(frame_idx))
        points = np.asarray(support.points_world, dtype=np.float32)
        weights = (
            np.asarray(support.weights, dtype=np.float32)
            if support.weights is not None
            else None
        )
        diagnostics = dict(support.diagnostics or {})
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Invalid standardized road-plane support shape for frame {frame_idx}: {points.shape}."
            )
        transformed = ((r_total @ points.T).T * float(scale)) + t_total.reshape(1, 3)
        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                f"Transformed standardized road-plane support is non-finite for frame {frame_idx}."
            )
        store.save_road_plane_support(
            RoadPlaneSupportData(
                frame_index=int(frame_idx),
                points_world=transformed.astype(np.float32),
                weights=weights,
                source_frame_index=support.source_frame_index,
                diagnostics={
                    **(diagnostics if isinstance(diagnostics, Mapping) else {}),
                    "alignment_transform_id": transform_id,
                    "metric_scale": True,
                },
                metadata=dict(support.metadata or {}),
            )
        )
        updated += 1
    return updated


def _apply_piecewise_transform_to_road_plane_sampled_points(
    store: ResourceStore,
    *,
    r_total: np.ndarray,
    scale_by_frame: Mapping[int, float],
    t_total: np.ndarray,
    transform_id: str,
) -> int:
    frame_indices = store.frame_indices(ResourceKind.ROAD_PLANE_SUPPORT)
    if not frame_indices:
        return 0
    updated = 0
    for frame_idx in frame_indices:
        if frame_idx not in scale_by_frame:
            continue
        scale = float(scale_by_frame[frame_idx])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid piecewise scale for sampled points frame {frame_idx}: {scale}.")
        support = store.load_road_plane_support(int(frame_idx))
        points = np.asarray(support.points_world, dtype=np.float32)
        weights = (
            np.asarray(support.weights, dtype=np.float32)
            if support.weights is not None
            else None
        )
        diagnostics = dict(support.diagnostics or {})
        transformed = ((r_total @ points.T).T * scale) + t_total.reshape(1, 3)
        store.save_road_plane_support(
            RoadPlaneSupportData(
                frame_index=int(frame_idx),
                points_world=transformed.astype(np.float32),
                weights=weights,
                source_frame_index=support.source_frame_index,
                diagnostics={
                    **(diagnostics if isinstance(diagnostics, Mapping) else {}),
                    "alignment_transform_id": transform_id,
                    "metric_scale": True,
                    "piecewise_scale_factor": float(scale),
                },
                metadata=dict(support.metadata or {}),
            )
        )
        updated += 1
    return updated


def verify_alignment_consistency(
    store: ResourceStore,
    *,
    require_road_plane: bool = False,
) -> None:
    if not store.has(ResourceKind.TRAJECTORY):
        return
    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        metadata = (
            data["metadata"].item()
            if "metadata" in data.files and isinstance(data["metadata"], np.ndarray)
            else {}
        )
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)

    transform_id = str(metadata.get("alignment_transform_id", "")).strip()
    if not transform_id:
        raise ValueError("Trajectory metadata is missing alignment_transform_id.")
    if metadata.get("metric_scale") is not True:
        raise ValueError("Trajectory metadata must set metric_scale=true.")

    checked_depth = 0
    for frame_idx in frame_indices.tolist():
        depth = store.load_depth(int(frame_idx))
        dmeta = dict(depth.metadata or {})
        if str(dmeta.get("alignment_transform_id", "")).strip() != transform_id:
            raise ValueError(
                f"Depth frame {int(frame_idx)} has mismatched alignment_transform_id."
            )
        if dmeta.get("metric_scale") is not True:
            raise ValueError(f"Depth frame {int(frame_idx)} is not marked metric_scale=true.")
        checked_depth += 1

    checked_point_cloud = 0
    if store.has(ResourceKind.POINT_CLOUD_3D):
        pc = store.load_point_cloud_3d()
        pc_id = str((pc.metadata or {}).get("alignment_transform_id", "")).strip()
        if pc_id and pc_id != transform_id:
            raise ValueError("POINT_CLOUD_3D has mismatched alignment_transform_id.")
        if not pc_id:
            LOG.debug(
                "[Alignment] POINT_CLOUD_3D has no alignment_transform_id (likely rebuilt post-alignment)."
            )
        checked_point_cloud = 1

    if require_road_plane and not store.has(ResourceKind.ROAD_PLANE):
        raise ValueError("Road-plane resource is required but missing.")

    checked_road_planes = 0
    if store.has(ResourceKind.ROAD_PLANE):
        for frame_idx in store.frame_indices(ResourceKind.ROAD_PLANE):
            plane = store.load_road_plane(int(frame_idx))
            rp_id = str((plane.metadata or {}).get("alignment_transform_id", "")).strip()
            if rp_id and rp_id != transform_id:
                raise ValueError(
                    f"Road-plane frame {int(frame_idx)} has mismatched alignment_transform_id."
                )
            if not rp_id:
                LOG.debug(
                    "[Alignment] Road-plane frame %d has no alignment_transform_id (likely rebuilt post-alignment).",
                    int(frame_idx),
                )
            checked_road_planes += 1
    LOG.info(
        "[Alignment] Consistency checks passed: transform_id=%s depth_frames=%d point_cloud=%d road_planes=%d",
        transform_id,
        checked_depth,
        checked_point_cloud,
        checked_road_planes,
    )


def align_trajectory_to_camera_height(
    store: ResourceStore,
    target_up: Optional[np.ndarray] = None,
    *,
    road_labels: Tuple[str, ...] = ("road",),
    settings: AlignmentSettings | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    del road_labels  # Kept for stable API, no longer used in plane-first alignment.

    cfg = settings or AlignmentSettings()
    raw_policy = context.get("validation_policy") if isinstance(context, Mapping) else None
    policy = ValidationPolicySettings.from_mapping(raw_policy if isinstance(raw_policy, Mapping) else None)
    adaptive = AdaptiveValidationContext.from_runtime(policy, context)
    max_plane_anchor_rmse_soft, max_plane_anchor_rmse_hard = adaptive.max_thresholds(
        float(cfg.max_plane_anchor_rmse_m)
    )
    max_plane_anchor_abs_soft, max_plane_anchor_abs_hard = adaptive.max_thresholds(
        float(cfg.max_plane_anchor_abs_err_m)
    )

    if not store.has(ResourceKind.TRAJECTORY):
        raise ValueError("Trajectory data is required for alignment.")
    if not store.has(ResourceKind.CAMERA_HEIGHT):
        raise ValueError("Camera height data is required for alignment.")
    if not store.has(ResourceKind.ROAD_PLANE):
        raise ValueError("Road-plane data is required for plane-first metric scale recovery.")

    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        frame_indices = np.asarray(data["frame_indices"], dtype=int)
        c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
        metadata = (
            data["metadata"].item()
            if "metadata" in data.files and isinstance(data["metadata"], np.ndarray)
            else {}
        )
        confidence = np.asarray(data["confidence"]) if "confidence" in data.files else None

    if metadata.get("metric_scale") is True:
        LOG.info("[Alignment] Trajectory already metric-scaled; skipping.")
        return

    heights = {int(frame_idx): store.load_camera_height(int(frame_idx)) for frame_idx in frame_indices}
    if not heights:
        raise ValueError("Camera height data is required to metric-scale trajectory.")
    heights = _normalize_camera_heights_to_blender(heights)
    _validate_alignment_inputs(frame_indices, c2w, heights)
    LOG.info(
        "[Alignment] Starting alignment mode=%s frames=%d require_road_plane=%s",
        cfg.mode,
        int(frame_indices.size),
        True,
    )

    up_vectors = np.asarray(c2w[:, :3, 1], dtype=np.float32)
    up_avg = np.mean(up_vectors, axis=0)
    if target_up is None:
        target_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    r_align = compute_up_direction_alignment(up_avg, target_up)
    LOG.debug(
        "[Alignment] Up alignment: up_avg=%s target_up=%s",
        np.array2string(up_avg, precision=4, suppress_small=True),
        np.array2string(np.asarray(target_up, dtype=np.float32), precision=4, suppress_small=True),
    )

    aligned_c2w = np.asarray(c2w, dtype=np.float32).copy()
    aligned_c2w[:, :3, :3] = np.einsum("ij,fjk->fik", r_align, aligned_c2w[:, :3, :3])
    aligned_c2w[:, :3, 3] = (r_align @ aligned_c2w[:, :3, 3].T).T

    # Road planes were estimated before alignment in the raw trajectory frame.
    # Scale candidates must be estimated in the same raw frame to avoid pose/plane mismatch.
    nominal_scale, scale_diag = _estimate_scale_from_road_planes(
        store,
        c2w,
        heights,
        frame_indices,
        min_samples=cfg.min_plane_scale_samples,
        max_iqr_ratio=cfg.max_plane_scale_iqr_ratio,
        min_inlier_ratio=cfg.min_plane_scale_inlier_ratio,
        allow_unstable=bool(cfg.allow_degraded_output),
    )
    scale_series, piecewise_diag = _build_piecewise_scale_series(
        frame_indices,
        scale_diag.get("frame_details", ()),
        settings=cfg,
    )
    scale_diag["piecewise"] = piecewise_diag
    frame_scale_map = {int(frame_indices[i]): float(scale_series[i]) for i in range(frame_indices.size)}
    if bool(piecewise_diag.get("degraded", False)) and not bool(cfg.allow_degraded_output):
        raise ValueError(
            "Piecewise alignment required degraded segmented scaling, but allow_degraded_output=false."
        )

    for i in range(frame_indices.size):
        aligned_c2w[i, :3, 3] *= float(scale_series[i])
    aligned_c2w, offset = apply_height_correction(aligned_c2w, heights, frame_indices, axis="z")
    aligned_c2w, yaw_meta = normalize_trajectory_yaw_by_dominant_motion(aligned_c2w)
    anchor_frame_idx = int(frame_indices[0])
    anchor_height_m = float(heights[anchor_frame_idx].height_m)
    aligned_c2w, origin_anchor_delta, origin_anchor_meta = _apply_origin_anchor_to_pose_stack(
        aligned_c2w,
        anchor_height_m=anchor_height_m,
        metadata={},
        metadata_label="alignment",
    )
    LOG.info(
        "[Alignment] Applied transforms: nominal_scale=%.6f segmented=%s z_offset=%.6f yaw_applied=%s yaw_deg=%.4f",
        float(nominal_scale),
        bool(piecewise_diag.get("degraded", False)),
        float(offset),
        bool(yaw_meta.get("applied", False)),
        float(yaw_meta.get("yaw_deg", 0.0)),
    )

    yaw_rad = np.radians(float(yaw_meta.get("yaw_deg", 0.0))) if bool(yaw_meta.get("applied", False)) else 0.0
    r_yaw = _rotation_z(float(yaw_rad))
    r_total = (r_yaw @ r_align).astype(np.float32)
    t_total = (
        (r_yaw @ np.array([0.0, 0.0, float(offset)], dtype=np.float32)).astype(np.float32)
        + origin_anchor_delta.astype(np.float32)
    )

    height_fit_diag = _validate_camera_height_fit(
        aligned_c2w,
        heights,
        frame_indices,
        max_rmse_m=cfg.max_height_rmse_m,
        max_abs_err_m=cfg.max_height_abs_err_m,
    )
    LOG.info(
        "[Alignment] Height fit: rmse=%.6f max_abs=%.6f median_err=%.6f",
        float(height_fit_diag.get("height_rmse_m", np.nan)),
        float(height_fit_diag.get("height_max_abs_err_m", np.nan)),
        float(height_fit_diag.get("height_median_err_m", np.nan)),
    )

    aligned_w2c = np.linalg.inv(aligned_c2w)
    transform_id = uuid4().hex
    metadata = dict(metadata or {})
    metadata.update(
        {
            "metric_scale": True,
            "scale_factor": float(nominal_scale),
            "scale_source": "road_plane_camera_height_piecewise",
            "scale_series_per_frame": {str(k): float(v) for k, v in frame_scale_map.items()},
            "up_direction_alignment": "camera_height",
            "height_correction_offset": float(offset),
            "yaw_normalization": dict(yaw_meta),
            "alignment_mode": cfg.mode,
            "alignment_transform_id": transform_id,
            "alignment_transform": {
                "rotation": r_total.astype(float).tolist(),
                "scale": float(nominal_scale),
                "translation": t_total.astype(float).tolist(),
            },
            "alignment_settings": {
                "mode": cfg.mode,
                "min_plane_scale_samples": int(cfg.min_plane_scale_samples),
                "max_plane_scale_iqr_ratio": float(cfg.max_plane_scale_iqr_ratio),
                "min_plane_scale_inlier_ratio": float(cfg.min_plane_scale_inlier_ratio),
                "max_plane_anchor_rmse_m": float(cfg.max_plane_anchor_rmse_m),
                "max_plane_anchor_abs_err_m": float(cfg.max_plane_anchor_abs_err_m),
                "segment_min_length": int(cfg.segment_min_length),
                "segment_change_zscore": float(cfg.segment_change_zscore),
                "segment_persistence_frames": int(cfg.segment_persistence_frames),
                "max_segments": int(cfg.max_segments),
                "transition_frames": int(cfg.transition_frames),
                "max_scale_jump_ratio": float(cfg.max_scale_jump_ratio),
                "max_scale_rate_per_frame": float(cfg.max_scale_rate_per_frame),
                "min_valid_scale_candidates": int(cfg.min_valid_scale_candidates),
                "allow_degraded_output": bool(cfg.allow_degraded_output),
                "max_height_rmse_m": float(cfg.max_height_rmse_m),
                "max_height_abs_err_m": float(cfg.max_height_abs_err_m),
            },
            "scale_diagnostics": scale_diag,
            "height_fit": height_fit_diag,
        }
    )
    metadata.update(origin_anchor_meta)
    metadata["origin_anchor_frame_index"] = anchor_frame_idx
    LOG.debug(
        "[Alignment] Global transform summary: id=%s rotation=%s translation=%s",
        transform_id,
        np.array2string(r_total, precision=4, suppress_small=True),
        np.array2string(t_total, precision=4, suppress_small=True),
    )

    samples = []
    for idx, frame_idx in enumerate(frame_indices):
        conf = None
        if confidence is not None and np.asarray(confidence).size > idx:
            conf = float(np.asarray(confidence)[idx])
        samples.append(
            PoseSample(
                frame_index=int(frame_idx),
                camera_to_world=aligned_c2w[idx],
                world_to_camera=aligned_w2c[idx],
                confidence=conf,
                metadata=dict(metadata),
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata=metadata))
    LOG.info("[Alignment] Saved aligned trajectory: frames=%d", len(samples))

    for frame_idx, height in heights.items():
        hmeta = dict(height.metadata or {})
        hmeta["alignment_transform_id"] = transform_id
        store.save_camera_height(
            CameraHeightData(
                frame_index=height.frame_index,
                height_m=height.height_m,
                metadata=hmeta,
            )
        )
    LOG.debug("[Alignment] Saved normalized camera heights for %d frames.", len(heights))

    for frame_idx in frame_indices:
        depth = store.load_depth(int(frame_idx))
        depth_meta = dict(depth.metadata or {})
        depth_meta["metric_scale"] = True
        depth_scale = float(frame_scale_map[int(frame_idx)])
        depth_meta["scale_factor"] = depth_scale
        depth_meta["scale_source"] = "road_plane_camera_height_piecewise"
        depth_meta["alignment_transform_id"] = transform_id
        store.save_depth(
            DepthData(
                frame_index=depth.frame_index,
                depth=np.asarray(depth.depth, dtype=np.float32) * depth_scale,
                confidence=depth.confidence,
                metadata=depth_meta,
            )
        )
    LOG.debug("[Alignment] Saved metric-scaled depth for %d frames.", int(frame_indices.size))

    # Point cloud fused pre-alignment cannot be corrected by a single transform under piecewise scaling.
    # Runtime will rebuild it from corrected depth + trajectory.
    _apply_piecewise_transform_to_road_planes(
        store,
        r_total=r_total,
        scale_by_frame=frame_scale_map,
        t_total=t_total,
        transform_id=transform_id,
    )
    updated_sampled = _apply_piecewise_transform_to_road_plane_sampled_points(
        store,
        r_total=r_total,
        scale_by_frame=frame_scale_map,
        t_total=t_total,
        transform_id=transform_id,
    )
    LOG.debug(
        "[Alignment] Updated piecewise-transformed road-plane sampled point files: %d",
        int(updated_sampled),
    )

    inlier_frames = set(int(v) for v in (scale_diag.get("inlier_frames") or []))
    if not inlier_frames:
        raise RuntimeError("Alignment scale diagnostics did not retain any inlier frames.")
    try:
        plane_anchor_fit_diag = _validate_post_alignment_plane_anchor_fit(
            store,
            aligned_c2w,
            heights,
            frame_indices,
            inlier_frames=inlier_frames,
            max_rmse_m=max_plane_anchor_rmse_soft,
            max_abs_err_m=max_plane_anchor_abs_soft,
        )
        plane_anchor_fit_diag["degraded"] = False
    except ValueError as exc:
        if not bool(cfg.allow_degraded_output):
            raise
        plane_anchor_fit_diag = _validate_post_alignment_plane_anchor_fit(
            store,
            aligned_c2w,
            heights,
            frame_indices,
            inlier_frames=inlier_frames,
            max_rmse_m=max_plane_anchor_rmse_hard if adaptive.enabled else float(np.inf),
            max_abs_err_m=max_plane_anchor_abs_hard if adaptive.enabled else float(np.inf),
        )
        if adaptive.enabled and (
            float(plane_anchor_fit_diag.get("plane_anchor_rmse_m", np.inf)) > max_plane_anchor_rmse_hard
            or float(plane_anchor_fit_diag.get("plane_anchor_max_abs_err_m", np.inf)) > max_plane_anchor_abs_hard
        ):
            raise
        plane_anchor_fit_diag["degraded"] = True
        plane_anchor_fit_diag["degraded_reason"] = str(exc)
        plane_anchor_fit_diag["validation_policy"] = adaptive.diagnostic_summary()
        LOG.warning(
            "[Alignment] Post-alignment anchor fit exceeds soft thresholds; continuing in degraded mode: %s",
            exc,
        )
    metadata["plane_anchor_fit"] = plane_anchor_fit_diag
    metadata["geometry_refresh_required"] = {
        "point_cloud_3d": True,
        "road_plane": True,
    }
    for sample in samples:
        sample.metadata = dict(metadata)
    store.save_trajectory(PoseData(samples=samples, metadata=metadata))
    LOG.info(
        "[Alignment] Plane anchor fit: rmse=%.6f max_abs=%.6f samples=%d",
        float(plane_anchor_fit_diag.get("plane_anchor_rmse_m", np.nan)),
        float(plane_anchor_fit_diag.get("plane_anchor_max_abs_err_m", np.nan)),
        int(plane_anchor_fit_diag.get("plane_anchor_samples", 0)),
    )

    _write_height_debug_visualization(store, frame_indices, c2w, aligned_c2w, heights)
    _write_alignment_debug_artifacts(
        store,
        frame_indices=frame_indices,
        scale_diag=scale_diag,
        height_fit_diag={**height_fit_diag, **plane_anchor_fit_diag},
        transform_metadata=metadata.get("alignment_transform", {}),
    )
    LOG.info("[Alignment] Completed alignment transform_id=%s", transform_id)


def canonicalize_metric_geometry_to_camera_height(
    store: ResourceStore,
    target_up: Optional[np.ndarray] = None,
    *,
    settings: AlignmentSettings | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    cfg = settings or AlignmentSettings()
    raw_policy = context.get("validation_policy") if isinstance(context, Mapping) else None
    policy = ValidationPolicySettings.from_mapping(raw_policy if isinstance(raw_policy, Mapping) else None)
    adaptive = AdaptiveValidationContext.from_runtime(policy, context)
    max_plane_anchor_rmse_soft, max_plane_anchor_rmse_hard = adaptive.max_thresholds(
        float(cfg.max_plane_anchor_rmse_m)
    )
    max_plane_anchor_abs_soft, max_plane_anchor_abs_hard = adaptive.max_thresholds(
        float(cfg.max_plane_anchor_abs_err_m)
    )

    if not store.has(ResourceKind.TRAJECTORY):
        raise ValueError("Trajectory data is required for canonicalization.")
    if not store.has(ResourceKind.CAMERA_HEIGHT):
        raise ValueError("Camera height data is required for canonicalization.")

    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        frame_indices = np.asarray(data["frame_indices"], dtype=int)
        c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
        metadata = (
            data["metadata"].item()
            if "metadata" in data.files and isinstance(data["metadata"], np.ndarray)
            else {}
        )
        confidence = np.asarray(data["confidence"]) if "confidence" in data.files else None

    if metadata.get("metric_scale") is not True:
        raise ValueError(
            "Metric canonicalization requires a metric trajectory; trajectory metadata must set metric_scale=true."
        )

    heights = {int(frame_idx): store.load_camera_height(int(frame_idx)) for frame_idx in frame_indices}
    if not heights:
        raise ValueError("Camera height data is required to canonicalize trajectory.")
    heights = _normalize_camera_heights_to_blender(heights)
    _validate_alignment_inputs(frame_indices, c2w, heights)
    LOG.info(
        "[Alignment] Starting metric canonicalization: frames=%d require_road_plane=%s",
        int(frame_indices.size),
        bool(store.has(ResourceKind.ROAD_PLANE)),
    )

    all_frames = {int(v) for v in frame_indices.tolist()}
    aligned_c2w, aligned_w2c, r_total, t_total, offset, yaw_meta, _ = _compute_rigid_canonical_alignment(
        c2w,
        heights,
        frame_indices,
        target_up=target_up,
    )
    anchor_frame_idx = int(frame_indices[0])
    anchor_height_m = float(heights[anchor_frame_idx].height_m)
    aligned_c2w, origin_anchor_delta, origin_anchor_meta = _apply_origin_anchor_to_pose_stack(
        aligned_c2w,
        anchor_height_m=anchor_height_m,
        metadata={},
        metadata_label="metric_canonicalization",
    )
    aligned_w2c = np.linalg.inv(aligned_c2w)
    t_total = np.asarray(t_total, dtype=np.float32) + origin_anchor_delta.astype(np.float32)
    has_road_plane = bool(store.has(ResourceKind.ROAD_PLANE))
    height_fit_diag = _validate_camera_height_fit(
        aligned_c2w,
        heights,
        frame_indices,
        max_rmse_m=float(np.inf),
        max_abs_err_m=float(np.inf),
    )
    camera_up_fit_diag = _validate_canonical_camera_up(aligned_c2w)
    pre_plane_anchor_fit_diag: Dict[str, float | bool] = {}
    if has_road_plane:
        pre_plane_anchor_fit_diag = _evaluate_transformed_road_plane_anchor_fit(
            store,
            aligned_c2w,
            heights,
            frame_indices,
            inlier_frames=all_frames,
            r_total=r_total,
            scale=1.0,
            t_total=t_total,
            max_rmse_m=cfg.max_plane_anchor_rmse_m,
            max_abs_err_m=cfg.max_plane_anchor_abs_err_m,
        )
    height_fit_strict_ok = True
    height_fit_validation_mode = "road_plane_anchor" if has_road_plane else "direct_axis_height"
    height_rmse = float(height_fit_diag.get("height_rmse_m", np.inf))
    height_max_abs = float(height_fit_diag.get("height_max_abs_err_m", np.inf))
    if height_rmse > float(cfg.max_height_rmse_m):
        height_fit_strict_ok = False
        if not has_road_plane:
            raise ValueError(
                f"Metric canonicalization RMSE against camera height too high: "
                f"{height_rmse:.4f} m > {float(cfg.max_height_rmse_m):.4f} m."
            )
        LOG.warning(
            "[Alignment] Metric canonicalization direct camera-axis RMSE exceeds strict threshold "
            "(diagnostic-only because road-plane anchor fit is present): %.4f m > %.4f m.",
            height_rmse,
            float(cfg.max_height_rmse_m),
        )
    if height_max_abs > float(cfg.max_height_abs_err_m):
        height_fit_strict_ok = False
        if not has_road_plane:
            raise ValueError(
                f"Metric canonicalization max absolute error too high: "
                f"{height_max_abs:.4f} m > {float(cfg.max_height_abs_err_m):.4f} m."
            )
        LOG.warning(
            "[Alignment] Metric canonicalization direct camera-axis max absolute error exceeds "
            "strict threshold (diagnostic-only because road-plane anchor fit is present): "
            "%.4f m > %.4f m.",
            height_max_abs,
            float(cfg.max_height_abs_err_m),
        )

    transform_id = uuid4().hex
    meta = dict(metadata or {})
    meta.update(
        {
            "alignment_mode": "metric_rigid_canonicalization",
            "alignment_transform_id": transform_id,
            "alignment_transform": {
                "rotation": r_total.astype(float).tolist(),
                "scale": 1.0,
                "translation": t_total.astype(float).tolist(),
            },
            "up_direction_alignment": "camera_height",
            "height_correction_offset": float(offset),
            "yaw_normalization": dict(yaw_meta),
            "height_fit": {
                **dict(height_fit_diag),
                "strict_thresholds_passed": bool(height_fit_strict_ok),
                "diagnostic_only": bool(has_road_plane),
            },
            "height_fit_validation_mode": height_fit_validation_mode,
            "camera_up_fit": dict(camera_up_fit_diag),
            "canonical_world_frame": True,
            "alignment_settings": {
                "mode": cfg.mode,
                "max_height_rmse_m": float(cfg.max_height_rmse_m),
                "max_height_abs_err_m": float(cfg.max_height_abs_err_m),
            },
            "geometry_refresh_required": {
                "point_cloud_3d": False,
                "road_plane": False,
            },
        }
    )
    if pre_plane_anchor_fit_diag:
        meta["pre_transform_plane_anchor_fit"] = dict(pre_plane_anchor_fit_diag)
    meta.update(origin_anchor_meta)
    meta["origin_anchor_frame_index"] = anchor_frame_idx

    samples = []
    for idx, frame_idx in enumerate(frame_indices):
        conf = None
        if confidence is not None and np.asarray(confidence).size > idx:
            conf = float(np.asarray(confidence)[idx])
        samples.append(
            PoseSample(
                frame_index=int(frame_idx),
                camera_to_world=aligned_c2w[idx],
                world_to_camera=aligned_w2c[idx],
                confidence=conf,
                metadata=dict(meta),
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata=meta))
    LOG.info("[Alignment] Saved canonicalized metric trajectory: frames=%d", len(samples))

    for frame_idx, height in heights.items():
        hmeta = dict(height.metadata or {})
        hmeta["alignment_transform_id"] = transform_id
        hmeta["canonical_world_frame"] = True
        store.save_camera_height(
            CameraHeightData(
                frame_index=height.frame_index,
                height_m=height.height_m,
                metadata=hmeta,
            )
        )

    for frame_idx in frame_indices:
        depth = store.load_depth(int(frame_idx))
        depth_meta = dict(depth.metadata or {})
        depth_meta["alignment_transform_id"] = transform_id
        depth_meta["metric_scale"] = True
        depth_meta["canonical_world_frame"] = True
        depth_meta["world_frame_alignment"] = "camera_height_rigid"
        store.save_depth(
            DepthData(
                frame_index=depth.frame_index,
                depth=np.asarray(depth.depth, dtype=np.float32),
                confidence=depth.confidence,
                metadata=depth_meta,
            )
        )

    _apply_transform_to_point_cloud(
        store,
        r_total=r_total,
        scale=1.0,
        t_total=t_total,
        transform_id=transform_id,
    )

    plane_anchor_fit_diag: Dict[str, float | bool | str] = {}
    if store.has(ResourceKind.ROAD_PLANE):
        _apply_transform_to_road_planes(
            store,
            r_total=r_total,
            scale=1.0,
            t_total=t_total,
            transform_id=transform_id,
        )
        _apply_transform_to_road_plane_sampled_points(
            store,
            r_total=r_total,
            scale=1.0,
            t_total=t_total,
            transform_id=transform_id,
        )
        try:
            plane_anchor_fit_diag = _validate_post_alignment_plane_anchor_fit(
                store,
                aligned_c2w,
                heights,
                frame_indices,
                inlier_frames=all_frames,
                max_rmse_m=max_plane_anchor_rmse_soft,
                max_abs_err_m=max_plane_anchor_abs_soft,
            )
            plane_anchor_fit_diag["degraded"] = False
        except ValueError as exc:
            if not bool(cfg.allow_degraded_output):
                raise
            plane_anchor_fit_diag = _validate_post_alignment_plane_anchor_fit(
                store,
                aligned_c2w,
                heights,
                frame_indices,
                inlier_frames=all_frames,
                max_rmse_m=max_plane_anchor_rmse_hard if adaptive.enabled else float(np.inf),
                max_abs_err_m=max_plane_anchor_abs_hard if adaptive.enabled else float(np.inf),
            )
            if adaptive.enabled and (
                float(plane_anchor_fit_diag.get("plane_anchor_rmse_m", np.inf)) > max_plane_anchor_rmse_hard
                or float(plane_anchor_fit_diag.get("plane_anchor_max_abs_err_m", np.inf)) > max_plane_anchor_abs_hard
            ):
                raise
            plane_anchor_fit_diag["degraded"] = True
            plane_anchor_fit_diag["degraded_reason"] = str(exc)
            plane_anchor_fit_diag["validation_policy"] = adaptive.diagnostic_summary()
            LOG.warning(
                "[Alignment] Metric canonicalization plane-anchor fit exceeds soft thresholds; continuing in degraded mode: %s",
                exc,
            )
        meta["plane_anchor_fit"] = dict(plane_anchor_fit_diag)
        for sample in samples:
            sample.metadata = dict(meta)
        store.save_trajectory(PoseData(samples=samples, metadata=meta))

    _write_height_debug_visualization(store, frame_indices, c2w, aligned_c2w, heights)
    _write_alignment_debug_artifacts(
        store,
        frame_indices=frame_indices,
        scale_diag={},
        height_fit_diag={**height_fit_diag, **dict(camera_up_fit_diag), **plane_anchor_fit_diag},
        transform_metadata=meta.get("alignment_transform", {}),
    )
    LOG.info("[Alignment] Completed metric canonicalization transform_id=%s", transform_id)


def _estimate_ground_z_from_road_planes(store: ResourceStore) -> tuple[np.ndarray, Dict[str, int]]:
    stats = {
        "valid_road_plane_samples": 0,
        "rejected_nonpositive_anchor": 0,
        "rejected_ground_not_below_camera": 0,
    }
    if not store.has(ResourceKind.ROAD_PLANE):
        return np.zeros((0,), dtype=np.float32), stats
    if not store.has(ResourceKind.TRAJECTORY):
        return np.zeros((0,), dtype=np.float32), stats
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    frame_indices = np.asarray(traj["frame_indices"], dtype=np.int32)
    c2w = np.asarray(traj["camera_to_world"], dtype=np.float32)
    camera_map = {int(frame_indices[i]): c2w[i, :3, 3].astype(np.float32) for i in range(frame_indices.size)}
    ground_z: list[float] = []
    for frame_idx in store.frame_indices(ResourceKind.ROAD_PLANE):
        if int(frame_idx) not in camera_map:
            continue
        plane = store.load_road_plane(int(frame_idx))
        n = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            continue
        n = n / norm
        c = camera_map[int(frame_idx)]
        h = float(np.dot(n, c) + float(plane.offset))
        if h <= 0.0:
            stats["rejected_nonpositive_anchor"] += 1
            continue
        g = c - h * n
        if not np.isfinite(g).all():
            continue
        if float(g[2]) >= float(c[2]):
            stats["rejected_ground_not_below_camera"] += 1
            continue
        ground_z.append(float(g[2]))
        stats["valid_road_plane_samples"] += 1
    return np.asarray(ground_z, dtype=np.float32), stats


def _estimate_ground_z_from_point_cloud(
    store: ResourceStore,
    *,
    road_labels: Tuple[str, ...],
    sidewalk_labels: Tuple[str, ...],
) -> np.ndarray:
    if not store.has(ResourceKind.POINT_CLOUD_3D):
        return np.zeros((0,), dtype=np.float32)
    cloud = store.load_point_cloud_3d()
    points = np.asarray(cloud.points_world, dtype=np.float32)
    labels = np.asarray(cloud.labels, dtype=np.int32).reshape(-1)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    wanted = {str(v).strip().lower() for v in road_labels + sidewalk_labels if str(v).strip()}
    label_ids = [
        int(label_id)
        for label_id, label_name in (cloud.label_names or {}).items()
        if str(label_name).strip().lower() in wanted
    ]
    if not label_ids:
        return np.zeros((0,), dtype=np.float32)
    mask = np.isin(labels, np.asarray(label_ids, dtype=np.int32))
    if int(np.count_nonzero(mask)) == 0:
        return np.zeros((0,), dtype=np.float32)
    return points[mask, 2].astype(np.float32)


def _estimate_ground_z_from_point_cloud_data(
    cloud: PointCloud3DData,
    *,
    road_labels: Tuple[str, ...],
    sidewalk_labels: Tuple[str, ...],
) -> np.ndarray:
    points = np.asarray(cloud.points_world, dtype=np.float32)
    labels = np.asarray(cloud.labels, dtype=np.int32).reshape(-1)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    wanted = {str(v).strip().lower() for v in road_labels + sidewalk_labels if str(v).strip()}
    label_ids = [
        int(label_id)
        for label_id, label_name in (cloud.label_names or {}).items()
        if str(label_name).strip().lower() in wanted
    ]
    if not label_ids:
        return np.zeros((0,), dtype=np.float32)
    mask = np.isin(labels, np.asarray(label_ids, dtype=np.int32))
    if int(np.count_nonzero(mask)) == 0:
        return np.zeros((0,), dtype=np.float32)
    return points[mask, 2].astype(np.float32)


def _estimate_support_up_from_road_planes(
    store: ResourceStore,
    heights: Mapping[int, CameraHeightData],
) -> np.ndarray:
    if not store.has(ResourceKind.ROAD_PLANE):
        raise ValueError("GT comparison-frame canonicalization requires road-plane data.")
    if not store.has(ResourceKind.TRAJECTORY):
        raise ValueError("GT comparison-frame canonicalization requires trajectory data.")
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    frame_indices = np.asarray(traj["frame_indices"], dtype=np.int32)
    c2w = np.asarray(traj["camera_to_world"], dtype=np.float32)
    camera_map = {
        int(frame_indices[i]): np.asarray(c2w[i, :3, 3], dtype=np.float32)
        for i in range(frame_indices.size)
    }
    normals: list[np.ndarray] = []
    for frame_idx in store.frame_indices(ResourceKind.ROAD_PLANE):
        cam_pos = camera_map.get(int(frame_idx))
        if cam_pos is None:
            continue
        plane = store.load_road_plane(int(frame_idx))
        target_height = None
        height = heights.get(int(frame_idx))
        if height is not None:
            target_height = float(height.height_m)
        try:
            n_canon, _, _, _, _ = _canonicalize_support_plane_orientation(
                np.asarray(plane.normal, dtype=np.float32),
                float(plane.offset),
                cam_pos,
                target_height_m=target_height,
            )
        except ValueError:
            continue
        if np.all(np.isfinite(n_canon)):
            normals.append(np.asarray(n_canon, dtype=np.float32))
    if not normals:
        raise ValueError(
            "Could not derive a stable support-plane up axis from road-plane geometry."
        )
    normal_stack = np.stack(normals, axis=0)
    up_avg = np.mean(normal_stack, axis=0)
    norm = float(np.linalg.norm(up_avg))
    if norm < 1e-6:
        raise ValueError("Support-plane up-axis estimate is degenerate.")
    return (up_avg / norm).astype(np.float32)


def _resolve_unity_sequence_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.name.startswith("sequence.") and candidate.is_dir():
        return candidate
    children = sorted(candidate.glob("sequence.*"))
    if children:
        return children[0]
    raise FileNotFoundError(f"Unity sequence directory not found under {path}.")


def _pick_unity_camera_capture(captures: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for capture in captures:
        if capture.get("id") == "camera":
            return capture
    return None


def _unity_gt_blender_pose_for_frame(sequence_dir: Path, frame_idx: int) -> np.ndarray:
    json_path = sequence_dir / f"step{int(frame_idx)}.frame_data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Unity frame metadata missing for frame {frame_idx}: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    capture = _pick_unity_camera_capture(payload.get("captures", []))
    if capture is None:
        raise ValueError(f"Unity frame {frame_idx} is missing a camera capture.")
    position = np.asarray(capture.get("position", [0, 0, 0]), dtype=np.float32)
    rotation = np.asarray(capture.get("rotation", [0, 0, 0, 1]), dtype=np.float32)
    if rotation.shape != (4,):
        raise ValueError(f"Unity frame {frame_idx} has malformed camera rotation.")
    x, y, z, w = [float(v) for v in rotation]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    r_unity = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    c = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    r_cv = c @ r_unity @ c
    t_cv = c @ position.reshape(3, 1)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = r_cv
    c2w[:3, 3] = t_cv[:, 0]
    w2c = np.linalg.inv(c2w)
    c2w_bl, _ = convert_pose_opencv_to_blender(c2w, w2c)
    return np.asarray(c2w_bl, dtype=np.float32)


def _unity_gt_blender_rotation_for_frame(sequence_dir: Path, frame_idx: int) -> np.ndarray:
    return np.asarray(
        _unity_gt_blender_pose_for_frame(sequence_dir, frame_idx)[:3, :3],
        dtype=np.float32,
    )


def _load_unity_gt_blender_positions(
    sequence_dir: Path,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    used_frame_indices: list[int] = []
    for frame_idx in frame_indices.tolist():
        try:
            pose = _unity_gt_blender_pose_for_frame(sequence_dir, int(frame_idx))
        except FileNotFoundError:
            continue
        positions.append(np.asarray(pose[:3, 3], dtype=np.float32))
        used_frame_indices.append(int(frame_idx))
    if len(positions) < 2:
        raise ValueError(
            "Unity authoring-frame alignment requires at least 2 matching Unity frames."
        )
    return np.stack(positions, axis=0), np.asarray(used_frame_indices, dtype=np.int32)


def _fit_planar_rigid_transform(
    source_xz: np.ndarray,
    target_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source = np.asarray(source_xz, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 2 or source.shape[0] < 2:
        raise ValueError("source_xz must have shape (N, 2) with N >= 2.")
    if target.shape != source.shape:
        raise ValueError(
            "target_xy must match source_xz shape, "
            f"got {target.shape} vs {source.shape}."
        )
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center.reshape(1, 2)
    target_centered = target - target_center.reshape(1, 2)
    if float(np.linalg.norm(source_centered)) <= 1e-6:
        raise ValueError("Unity authoring-frame alignment is degenerate in the source plane.")
    if float(np.linalg.norm(target_centered)) <= 1e-6:
        raise ValueError("Unity authoring-frame alignment is degenerate in the target plane.")
    cov = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(cov.astype(np.float64), full_matrices=False)
    rotation = (vt.T @ u.T).astype(np.float32)
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1, :] *= -1.0
        rotation = (vt.T @ u.T).astype(np.float32)
    translation = (
        target_center.reshape(2, 1) - rotation @ source_center.reshape(2, 1)
    ).reshape(2).astype(np.float32)
    predicted = (rotation @ source.T).T + translation.reshape(1, 2)
    residuals = np.linalg.norm(predicted - target, axis=1)
    return rotation, translation, {
        "rmse_m": float(np.sqrt(np.mean(np.square(residuals, dtype=np.float64)))),
        "max_abs_residual_m": float(np.max(residuals)),
    }


def _resolve_unity_authoring_to_canonical_transform(
    *,
    frame_indices: np.ndarray,
    final_c2w: np.ndarray,
    context: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    frame_source_raw = context.get("frame_source") if isinstance(context, Mapping) else None
    if frame_source_raw is None:
        return None
    try:
        sequence_dir = _resolve_unity_sequence_dir(Path(str(frame_source_raw)))
    except Exception:
        return None
    unity_positions_blender, used_frame_indices = _load_unity_gt_blender_positions(
        sequence_dir,
        frame_indices,
    )
    index_lookup = {int(frame_idx): idx for idx, frame_idx in enumerate(frame_indices.tolist())}
    target_positions: list[np.ndarray] = []
    aligned_unity_positions: list[np.ndarray] = []
    for pos, frame_idx in zip(unity_positions_blender, used_frame_indices.tolist()):
        target_idx = index_lookup.get(int(frame_idx))
        if target_idx is None:
            continue
        aligned_unity_positions.append(np.asarray(pos, dtype=np.float32))
        target_positions.append(np.asarray(final_c2w[target_idx, :3, 3], dtype=np.float32))
    if len(aligned_unity_positions) < 2:
        raise ValueError(
            "Unity authoring-frame alignment requires at least 2 overlapping trajectory frames."
        )
    unity_stack = np.stack(aligned_unity_positions, axis=0)
    target_stack = np.stack(target_positions, axis=0)
    rotation_2d, translation_xy, fit_stats = _fit_planar_rigid_transform(
        np.stack(
            [
                unity_stack[:, 0],
                -unity_stack[:, 2],
            ],
            axis=1,
        ),
        target_stack[:, :2],
    )
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = float(rotation_2d[0, 0])
    transform[0, 2] = float(rotation_2d[0, 1])
    transform[1, 0] = float(rotation_2d[1, 0])
    transform[1, 2] = float(rotation_2d[1, 1])
    transform[2, :] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    transform[2, 1] = 1.0
    transform[0, 3] = float(translation_xy[0])
    transform[1, 3] = float(translation_xy[1])
    angle_deg = float(
        np.degrees(np.arctan2(float(rotation_2d[1, 0]), float(rotation_2d[0, 0])))
    )
    return {
        "supported": True,
        "mode": "unity_world_horizontal",
        "sequence_dir": str(sequence_dir),
        "matching_frame_count": int(len(aligned_unity_positions)),
        "used_frame_indices": [int(v) for v in used_frame_indices.tolist()],
        "authoring_up_axis": "+Y",
        "authoring_horizontal_axes": {
            "+X": [1.0, 0.0, 0.0],
            "+Z": [0.0, 0.0, 1.0],
        },
        "canonical_up_axis": "+Z",
        "planar_rotation_deg": angle_deg,
        "fit_rmse_m": float(fit_stats["rmse_m"]),
        "fit_max_abs_residual_m": float(fit_stats["max_abs_residual_m"]),
        "authoring_to_canonical_transform": transform.astype(float).tolist(),
    }


def _aggregate_gravity_prior_vectors(
    vectors: Sequence[np.ndarray],
    *,
    min_valid_frames: int,
    max_outlier_angle_deg: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    normalized: list[np.ndarray] = []
    for vec in vectors:
        arr = np.asarray(vec, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(arr)):
            continue
        norm = float(np.linalg.norm(arr))
        if norm < 1e-6:
            continue
        normalized.append(arr / norm)
    if len(normalized) < int(min_valid_frames):
        raise ValueError(
            f"Insufficient valid gravity-prior frames: {len(normalized)} < {int(min_valid_frames)}."
        )
    stack = np.stack(normalized, axis=0)
    pairwise_dots = np.clip(stack @ stack.T, -1.0, 1.0)
    pairwise_angles = np.degrees(np.arccos(pairwise_dots))
    best_idx = -1
    best_count = -1
    best_median = float("inf")
    threshold = float(max_outlier_angle_deg)
    for idx in range(stack.shape[0]):
        row = pairwise_angles[idx]
        count = int(np.count_nonzero(row <= threshold))
        median = float(np.median(row[row <= threshold])) if count > 0 else float("inf")
        if count > best_count or (count == best_count and median < best_median):
            best_idx = idx
            best_count = count
            best_median = median
    if best_idx < 0:
        raise ValueError("Gravity-prior aggregate could not identify a consensus frame.")
    seed = stack[best_idx]
    dots = np.clip(stack @ seed, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    inlier_mask = angles <= threshold
    inliers = stack[inlier_mask]
    if inliers.shape[0] < int(min_valid_frames):
        raise ValueError(
            f"Gravity-prior inliers after angular rejection are insufficient: "
            f"{int(inliers.shape[0])} < {int(min_valid_frames)}."
        )
    resolved = np.mean(inliers, axis=0)
    resolved_norm = float(np.linalg.norm(resolved))
    if resolved_norm < 1e-6:
        raise ValueError("Gravity-prior aggregate is degenerate after outlier rejection.")
    resolved = resolved / resolved_norm
    diagnostics = {
        "candidate_frames": int(stack.shape[0]),
        "inlier_frames": int(inliers.shape[0]),
        "rejected_frames": int(stack.shape[0] - inliers.shape[0]),
        "max_outlier_angle_deg": float(max_outlier_angle_deg),
        "max_inlier_angle_deg": float(np.max(angles[inlier_mask])) if np.any(inlier_mask) else None,
        "median_inlier_angle_deg": float(np.median(angles[inlier_mask])) if np.any(inlier_mask) else None,
    }
    return resolved.astype(np.float32), diagnostics


def _resolve_unity_gt_gravity_prior(
    *,
    frame_indices: np.ndarray,
    c2w: np.ndarray,
    cfg: ComparisonFrameSettings,
    context: Mapping[str, Any] | None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    frame_source_raw = context.get("frame_source") if isinstance(context, Mapping) else None
    if frame_source_raw is None:
        raise ValueError("Unity GT gravity prior requires runtime context.frame_source.")
    sequence_dir = _resolve_unity_sequence_dir(Path(str(frame_source_raw)))
    # Unity GT poses are first converted into the repo's Blender-facing convention.
    # In that converted GT world, vertical is +Y and forward depth is +Z.
    gt_world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    candidates: list[np.ndarray] = []
    used_frames: list[int] = []
    for idx, frame_idx in enumerate(frame_indices.tolist()):
        try:
            r_gt = _unity_gt_blender_rotation_for_frame(sequence_dir, int(frame_idx))
        except Exception:
            continue
        r_est = np.asarray(c2w[idx, :3, :3], dtype=np.float32)
        if not np.all(np.isfinite(r_est)):
            continue
        gravity_in_est = r_est @ r_gt.T @ gt_world_up
        if not np.all(np.isfinite(gravity_in_est)):
            continue
        candidates.append(np.asarray(gravity_in_est, dtype=np.float32))
        used_frames.append(int(frame_idx))
    resolved, diagnostics = _aggregate_gravity_prior_vectors(
        candidates,
        min_valid_frames=cfg.gravity_prior_min_valid_frames,
        max_outlier_angle_deg=cfg.gravity_prior_max_outlier_angle_deg,
    )
    diagnostics.update(
        {
            "provider": "unity_gt",
            "sequence_dir": str(sequence_dir),
            "used_frame_indices": used_frames,
            "resolved_target_up": resolved.astype(float).tolist(),
        }
    )
    return resolved, diagnostics


def _resolve_comparison_frame_target_up(
    *,
    store: ResourceStore,
    frame_indices: np.ndarray,
    c2w: np.ndarray,
    heights: Mapping[int, CameraHeightData],
    cfg: ComparisonFrameSettings,
    context: Mapping[str, Any] | None,
) -> tuple[np.ndarray, str, Dict[str, Any]]:
    if cfg.mode == "gt":
        target_up = _estimate_support_up_from_road_planes(store, heights)
        return target_up, "support_plane", {}
    if cfg.up_direction_source == "gravity_prior":
        try:
            if cfg.gravity_prior_provider == "unity_gt":
                target_up, diagnostics = _resolve_unity_gt_gravity_prior(
                    frame_indices=frame_indices,
                    c2w=c2w,
                    cfg=cfg,
                    context=context,
                )
                return target_up, "gravity_prior_unity_gt", diagnostics
            raise ValueError(
                f"Unsupported comparison-frame gravity prior provider: {cfg.gravity_prior_provider}."
            )
        except Exception as exc:
            if cfg.gravity_prior_fail_if_unavailable:
                raise
            LOG.warning(
                "[Alignment] Gravity prior unavailable (%s). Falling back to estimated camera-up average.",
                exc,
            )
    up_vectors = np.asarray(c2w[:, :3, 1], dtype=np.float32)
    target_up = np.mean(up_vectors, axis=0)
    return target_up, "camera_up_average", {}


def _estimate_ground_samples_from_transformed_road_planes(
    store: ResourceStore,
    *,
    r_total: np.ndarray,
    t_total: np.ndarray,
) -> tuple[np.ndarray, Dict[str, int]]:
    stats = {
        "valid_road_plane_samples": 0,
        "rejected_nonpositive_anchor": 0,
        "rejected_ground_not_below_camera": 0,
    }
    if not store.has(ResourceKind.ROAD_PLANE):
        return np.zeros((0,), dtype=np.float32), stats
    if not store.has(ResourceKind.TRAJECTORY):
        return np.zeros((0,), dtype=np.float32), stats
    traj = np.load(store.path_for(ResourceKind.TRAJECTORY), allow_pickle=True)
    frame_indices = np.asarray(traj["frame_indices"], dtype=np.int32)
    c2w = np.asarray(traj["camera_to_world"], dtype=np.float32)
    camera_map = {
        int(frame_indices[i]): ((np.asarray(r_total, dtype=np.float32) @ c2w[i, :3, 3]) + t_total).astype(np.float32)
        for i in range(frame_indices.size)
    }
    ground_z: list[float] = []
    for frame_idx in store.frame_indices(ResourceKind.ROAD_PLANE):
        cam_pos = camera_map.get(int(frame_idx))
        if cam_pos is None:
            continue
        plane = store.load_road_plane(int(frame_idx))
        n = np.asarray(plane.normal, dtype=np.float32).reshape(3)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-6:
            continue
        n = n / n_norm
        n_t = (np.asarray(r_total, dtype=np.float32) @ n).astype(np.float32)
        d_t = float(float(plane.offset) - float(np.dot(n_t, np.asarray(t_total, dtype=np.float32))))
        try:
            n_t, d_t, _, signed_anchor, _ = _canonicalize_support_plane_orientation(
                n_t,
                d_t,
                cam_pos,
            )
        except ValueError:
            stats["rejected_ground_not_below_camera"] += 1
            continue
        if signed_anchor <= 0.0:
            stats["rejected_nonpositive_anchor"] += 1
            continue
        ground_point = cam_pos - signed_anchor * n_t
        if not np.all(np.isfinite(ground_point)):
            continue
        ground_z.append(float(ground_point[2]))
        stats["valid_road_plane_samples"] += 1
    return np.asarray(ground_z, dtype=np.float32), stats


def _estimate_ground_shift_for_comparison_frame(
    store: ResourceStore,
    *,
    cfg: ComparisonFrameSettings,
    r_total: np.ndarray,
    t_total: np.ndarray,
    road_labels: Tuple[str, ...],
    sidewalk_labels: Tuple[str, ...],
) -> tuple[float, np.ndarray, Dict[str, Any]]:
    candidates: list[np.ndarray] = []
    ground_stats: Dict[str, Any] = {"source": cfg.ground_source}
    if cfg.ground_source in {"road_plane", "auto"}:
        road_samples, road_stats = _estimate_ground_samples_from_transformed_road_planes(
            store,
            r_total=r_total,
            t_total=t_total,
        )
        if road_samples.size > 0:
            candidates.append(road_samples)
        ground_stats.update(road_stats)
    if cfg.ground_source in {"point_cloud_3d", "auto"} and store.has(ResourceKind.POINT_CLOUD_3D):
        cloud = store.load_point_cloud_3d()
        shifted_cloud = PointCloud3DData(
            points_world=((np.asarray(r_total, dtype=np.float32) @ np.asarray(cloud.points_world, dtype=np.float32).T).T + t_total.reshape(1, 3)).astype(np.float32),
            labels=np.asarray(cloud.labels, dtype=np.int32),
            label_confidences=np.asarray(cloud.label_confidences, dtype=np.float32),
            colors=np.asarray(cloud.colors, dtype=np.uint8),
            label_names=dict(cloud.label_names or {}),
            observation_counts=np.asarray(cloud.observation_counts, dtype=np.int32),
            metadata=dict(cloud.metadata or {}),
        )
        point_samples = _estimate_ground_z_from_point_cloud_data(
            shifted_cloud,
            road_labels=road_labels,
            sidewalk_labels=sidewalk_labels,
        )
        if point_samples.size > 0:
            candidates.append(point_samples)
            ground_stats["valid_point_cloud_samples"] = int(point_samples.size)

    merged = (
        np.concatenate([arr for arr in candidates if arr.size > 0], axis=0)
        if any(arr.size > 0 for arr in candidates)
        else np.zeros((0,), dtype=np.float32)
    )
    if merged.size < int(cfg.min_ground_samples):
        msg = (
            "Comparison-frame canonicalization has insufficient ground samples: "
            f"{int(merged.size)} < {int(cfg.min_ground_samples)}."
        )
        if cfg.fail_if_missing_ground:
            raise RuntimeError(msg)
        LOG.warning(msg)
        return 0.0, merged, ground_stats
    shift_reference = "median_ground"
    shift_value = float(np.median(merged))
    shift_z = -float(shift_value)
    if abs(shift_z) > float(cfg.max_abs_ground_shift_m):
        raise RuntimeError(
            f"Comparison-frame grounding z-shift {shift_z:.4f}m exceeds configured limit "
            f"{float(cfg.max_abs_ground_shift_m):.4f}m."
        )
    ground_stats["median_ground_z_before_m"] = float(np.median(merged))
    ground_stats["ground_shift_reference"] = shift_reference
    ground_stats["ground_shift_z_m"] = float(shift_z)
    return shift_z, merged, ground_stats


def _write_comparison_frame_debug_artifacts(
    store: ResourceStore,
    *,
    frame_indices: np.ndarray,
    pre_c2w: np.ndarray,
    post_c2w: np.ndarray,
    support_ground_z_before: np.ndarray,
    support_ground_z_after: np.ndarray,
    camera_height_above_support: np.ndarray,
    diagnostics: Mapping[str, Any],
) -> None:
    vis_dir = store.visualizations_dir("comparison_frame")
    vis_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_comparison_frame_plots(
            vis_dir,
            frame_indices=frame_indices,
            support_ground_z_before=support_ground_z_before,
            support_ground_z_after=support_ground_z_after,
            camera_height_above_support=camera_height_above_support,
        )
    except Exception as exc:
        LOG.warning("[ComparisonFrame] Failed to write comparison-frame plots: %s", exc)
    try:
        write_trajectory_path_plots(vis_dir, post_c2w)
    except Exception as exc:
        LOG.warning("[ComparisonFrame] Failed to write trajectory path plots: %s", exc)
    try:
        (vis_dir / "comparison_frame_summary.json").write_text(
            json.dumps(dict(diagnostics), indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        LOG.warning("[ComparisonFrame] Failed to write summary json: %s", exc)


def canonicalize_geometry_to_comparison_frame(
    store: ResourceStore,
    *,
    settings: ComparisonFrameSettings | None = None,
    road_labels: Tuple[str, ...] = ("road",),
    sidewalk_labels: Tuple[str, ...] = ("sidewalk",),
    context: Mapping[str, Any] | None = None,
) -> None:
    cfg = settings or ComparisonFrameSettings()
    if not cfg.enabled:
        return
    if not store.has(ResourceKind.TRAJECTORY):
        raise ValueError("Comparison-frame canonicalization requires trajectory data.")
    if not store.has(ResourceKind.CAMERA_HEIGHT):
        raise ValueError("Comparison-frame canonicalization requires camera height data.")

    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
        c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
        confidence = np.asarray(data["confidence"]) if "confidence" in data.files else None
        metadata = (
            data["metadata"].item()
            if "metadata" in data.files and isinstance(data["metadata"], np.ndarray)
            else {}
        )
    if metadata.get("metric_scale") is not True:
        raise ValueError(
            "Comparison-frame canonicalization requires metric geometry; trajectory metadata must set metric_scale=true."
        )

    heights = {int(frame_idx): store.load_camera_height(int(frame_idx)) for frame_idx in frame_indices.tolist()}
    heights = _normalize_camera_heights_to_blender(heights)
    _validate_alignment_inputs(frame_indices, c2w, heights)

    target_up, up_alignment_source, up_alignment_diagnostics = _resolve_comparison_frame_target_up(
        store=store,
        frame_indices=frame_indices,
        c2w=c2w,
        heights=heights,
        cfg=cfg,
        context=context,
    )
    r_up = compute_up_direction_alignment(target_up, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    rotated_c2w = np.asarray(c2w, dtype=np.float32).copy()
    rotated_c2w[:, :3, :3] = np.einsum("ij,fjk->fik", r_up, rotated_c2w[:, :3, :3])
    rotated_c2w[:, :3, 3] = (r_up @ rotated_c2w[:, :3, 3].T).T
    yawed_c2w, yaw_meta = normalize_trajectory_yaw_by_dominant_motion(
        rotated_c2w,
        fail_on_weak_motion=True,
        min_valid_steps=cfg.min_motion_steps,
        min_total_xy_travel_m=cfg.min_total_xy_travel_m,
        min_direction_concentration=cfg.min_direction_concentration,
    )
    yaw_rad = np.radians(float(yaw_meta.get("yaw_deg", 0.0))) if bool(yaw_meta.get("applied", False)) else 0.0
    r_yaw = _rotation_z(float(yaw_rad))
    r_total = (r_yaw @ r_up).astype(np.float32)

    pre_ground_samples, _ = _estimate_ground_samples_from_transformed_road_planes(
        store,
        r_total=r_total,
        t_total=np.zeros((3,), dtype=np.float32),
    )
    shift_z, merged_ground_samples, ground_stats = _estimate_ground_shift_for_comparison_frame(
        store,
        cfg=cfg,
        r_total=r_total,
        t_total=np.zeros((3,), dtype=np.float32),
        road_labels=road_labels,
        sidewalk_labels=sidewalk_labels,
    )
    positions = np.asarray(yawed_c2w[:, :3, 3], dtype=np.float32)
    xy_anchor = -positions[0, :2]
    t_total = np.array([float(xy_anchor[0]), float(xy_anchor[1]), float(shift_z)], dtype=np.float32)

    final_c2w = np.asarray(yawed_c2w, dtype=np.float32).copy()
    final_c2w[:, 0, 3] += float(xy_anchor[0])
    final_c2w[:, 1, 3] += float(xy_anchor[1])
    final_c2w[:, 2, 3] += float(shift_z)
    final_w2c = np.linalg.inv(final_c2w)

    transform_id = uuid4().hex
    post_ground_samples = pre_ground_samples + float(shift_z) if pre_ground_samples.size else pre_ground_samples

    camera_height_above_support = np.zeros((frame_indices.size,), dtype=np.float32)
    plane_height_errors: list[float] = []
    frame_ground_z_after: list[float] = []
    if store.has(ResourceKind.ROAD_PLANE):
        for idx, frame_idx in enumerate(frame_indices.tolist()):
            try:
                plane = store.load_road_plane(int(frame_idx))
            except Exception:
                continue
            n = np.asarray(plane.normal, dtype=np.float32)
            n = n / max(float(np.linalg.norm(n)), 1e-8)
            n_t = (r_total @ n).astype(np.float32)
            d_t = float(float(plane.offset) - float(np.dot(n_t, t_total)))
            cam_pos = np.asarray(final_c2w[idx, :3, 3], dtype=np.float32)
            try:
                n_t, d_t, _, signed_anchor, _ = _canonicalize_support_plane_orientation(
                    n_t,
                    d_t,
                    cam_pos,
                    target_height_m=float(heights[int(frame_idx)].height_m),
                )
            except ValueError:
                continue
            plane_height_errors.append(signed_anchor - float(heights[int(frame_idx)].height_m))
            camera_height_above_support[idx] = float(signed_anchor)
            ground_point = cam_pos - float(signed_anchor) * n_t
            frame_ground_z_after.append(float(ground_point[2]))

    if cfg.mode == "gt":
        if not plane_height_errors:
            raise ValueError(
                "GT comparison-frame canonicalization requires valid road-plane anchor checks."
            )
        errors = np.asarray(plane_height_errors, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_abs = float(np.max(np.abs(errors)))
        if rmse > float(cfg.gt_max_height_rmse_m):
            raise ValueError(
                f"GT comparison-frame camera-height RMSE {rmse:.4f}m exceeds "
                f"{float(cfg.gt_max_height_rmse_m):.4f}m."
            )
        if max_abs > float(cfg.gt_max_height_abs_err_m):
            raise ValueError(
                f"GT comparison-frame camera-height max abs error {max_abs:.4f}m exceeds "
                f"{float(cfg.gt_max_height_abs_err_m):.4f}m."
            )
        if frame_ground_z_after:
            ground_after = np.asarray(frame_ground_z_after, dtype=np.float64)
            drift_range = float(np.max(ground_after) - np.min(ground_after))
            if drift_range > float(cfg.gt_max_ground_drift_range_m):
                raise ValueError(
                    f"GT comparison-frame residual support-surface drift {drift_range:.4f}m exceeds "
                    f"{float(cfg.gt_max_ground_drift_range_m):.4f}m."
                )
    elif np.count_nonzero(camera_height_above_support > 0.0) > 0:
        median_height = float(np.median(camera_height_above_support[camera_height_above_support > 0.0]))
        if median_height < float(cfg.estimated_min_median_camera_height_m):
            raise ValueError(
                f"Estimated comparison-frame median camera height {median_height:.4f}m is implausibly low."
            )

    authoring_frame_meta = _resolve_unity_authoring_to_canonical_transform(
        frame_indices=frame_indices,
        final_c2w=final_c2w,
        context=context,
    )
    meta = dict(metadata or {})
    meta.update(
        {
            "canonical_world_frame": True,
            "comparison_frame": {
                "enabled": True,
                "mode": cfg.mode,
                "up_alignment_source": up_alignment_source,
                "up_alignment_diagnostics": dict(up_alignment_diagnostics),
                "resolved_target_up": np.asarray(target_up, dtype=np.float32).astype(float).tolist(),
                "target_up": [0.0, 0.0, 1.0],
                "origin_anchor_mode": "first_frame_xy_grounded",
                "origin_anchor_translation": t_total.astype(float).tolist(),
                "ground_shift_z_m": float(shift_z),
                "yaw_normalization": dict(yaw_meta),
                "ground_statistics": ground_stats,
                "authoring_frame": authoring_frame_meta,
            },
            "alignment_mode": f"comparison_frame_{cfg.mode}",
            "alignment_transform_id": transform_id,
            "height_fit_validation_mode": "road_plane_anchor" if store.has(ResourceKind.ROAD_PLANE) else "direct_axis_height",
            "alignment_transform": {
                "rotation": r_total.astype(float).tolist(),
                "scale": 1.0,
                "translation": t_total.astype(float).tolist(),
            },
            "world_frame_alignment": f"comparison_frame_{cfg.mode}",
            "geometry_refresh_required": {
                "point_cloud_3d": False,
                "road_plane": False,
            },
        }
    )
    samples: list[PoseSample] = []
    for idx, frame_idx in enumerate(frame_indices.tolist()):
        conf = float(confidence[idx]) if confidence is not None and confidence.size > idx else None
        samples.append(
            PoseSample(
                frame_index=int(frame_idx),
                camera_to_world=final_c2w[idx],
                world_to_camera=final_w2c[idx],
                confidence=conf,
                metadata=dict(meta),
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata=meta))

    for frame_idx, height in heights.items():
        hmeta = dict(height.metadata or {})
        hmeta["alignment_transform_id"] = transform_id
        hmeta["canonical_world_frame"] = True
        hmeta["world_frame_alignment"] = f"comparison_frame_{cfg.mode}"
        store.save_camera_height(
            CameraHeightData(
                frame_index=height.frame_index,
                height_m=height.height_m,
                metadata=hmeta,
            )
        )
    for frame_idx in frame_indices.tolist():
        depth = store.load_depth(int(frame_idx))
        dmeta = dict(depth.metadata or {})
        dmeta["alignment_transform_id"] = transform_id
        dmeta["metric_scale"] = True
        dmeta["canonical_world_frame"] = True
        dmeta["world_frame_alignment"] = f"comparison_frame_{cfg.mode}"
        store.save_depth(
            DepthData(
                frame_index=depth.frame_index,
                depth=np.asarray(depth.depth, dtype=np.float32),
                confidence=depth.confidence,
                metadata=dmeta,
            )
        )

    _apply_transform_to_point_cloud(
        store,
        r_total=r_total,
        scale=1.0,
        t_total=t_total,
        transform_id=transform_id,
    )
    _apply_transform_to_road_planes(
        store,
        r_total=r_total,
        scale=1.0,
        t_total=t_total,
        transform_id=transform_id,
    )
    _apply_transform_to_road_plane_sampled_points(
        store,
        r_total=r_total,
        scale=1.0,
        t_total=t_total,
        transform_id=transform_id,
    )
    verify_alignment_consistency(
        store,
        require_road_plane=store.has(ResourceKind.ROAD_PLANE),
    )
    diagnostics = {
        "mode": cfg.mode,
        "frame_count": int(frame_indices.size),
        "yaw_normalization": dict(yaw_meta),
        "ground_statistics": ground_stats,
        "camera_height_above_support_median_m": float(
            np.median(camera_height_above_support[camera_height_above_support > 0.0])
        )
        if np.count_nonzero(camera_height_above_support > 0.0) > 0
        else None,
    }
    _write_comparison_frame_debug_artifacts(
        store,
        frame_indices=frame_indices,
        pre_c2w=c2w,
        post_c2w=final_c2w,
        support_ground_z_before=merged_ground_samples if merged_ground_samples.size else pre_ground_samples,
        support_ground_z_after=post_ground_samples,
        camera_height_above_support=camera_height_above_support,
        diagnostics=diagnostics,
    )
    LOG.info(
        "[ComparisonFrame] Applied %s comparison-frame transform_id=%s shift_z_m=%.6f",
        cfg.mode,
        transform_id,
        float(shift_z),
    )


def ground_scene_to_z0(
    store: ResourceStore,
    *,
    settings: GroundingSettings | None = None,
    road_labels: Tuple[str, ...] = ("road",),
    sidewalk_labels: Tuple[str, ...] = ("sidewalk",),
) -> None:
    cfg = settings or GroundingSettings()
    if not cfg.enabled:
        return
    if not store.has(ResourceKind.TRAJECTORY):
        raise ValueError("Grounding requires trajectory data.")

    candidates: list[np.ndarray] = []
    road_plane_stats = {
        "valid_road_plane_samples": 0,
        "rejected_nonpositive_anchor": 0,
        "rejected_ground_not_below_camera": 0,
    }
    if cfg.source in {"road_plane", "auto"}:
        road_plane_samples, road_plane_stats = _estimate_ground_z_from_road_planes(store)
        candidates.append(road_plane_samples)
    if cfg.source in {"point_cloud_3d", "auto"}:
        candidates.append(
            _estimate_ground_z_from_point_cloud(
                store,
                road_labels=road_labels,
                sidewalk_labels=sidewalk_labels,
            )
        )
    merged = np.concatenate([arr for arr in candidates if arr.size > 0], axis=0) if any(
        arr.size > 0 for arr in candidates
    ) else np.zeros((0,), dtype=np.float32)
    if merged.size < int(cfg.min_ground_samples):
        msg = (
            "Grounding to z=0 has insufficient ground samples: "
            f"{int(merged.size)} < {int(cfg.min_ground_samples)}."
        )
        if cfg.fail_if_missing_ground:
            raise RuntimeError(msg)
        LOG.warning(msg)
        return

    shift_z = -float(np.median(merged))
    if abs(shift_z) > float(cfg.max_abs_ground_shift_m):
        raise RuntimeError(
            f"Grounding z-shift {shift_z:.4f}m exceeds configured limit {cfg.max_abs_ground_shift_m:.4f}m."
        )

    r_total = np.eye(3, dtype=np.float32)
    t_total = np.array([0.0, 0.0, float(shift_z)], dtype=np.float32)
    transform_id = uuid4().hex

    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
        c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
        w2c = np.asarray(data["world_to_camera"], dtype=np.float32) if "world_to_camera" in data.files else None
        confidence = np.asarray(data["confidence"]) if "confidence" in data.files else None
        metadata = (
            data["metadata"].item()
            if "metadata" in data.files and isinstance(data["metadata"], np.ndarray)
            else {}
        )
    c2w[:, :3, 3] += t_total.reshape(1, 3)
    anchor_frame_idx = int(frame_indices[0])
    anchor_height = float(store.load_camera_height(anchor_frame_idx).height_m)
    c2w, origin_anchor_delta, origin_anchor_meta = _apply_origin_anchor_to_pose_stack(
        c2w,
        anchor_height_m=anchor_height,
        metadata={},
        metadata_label="grounding_to_z0",
    )
    t_total = np.asarray(t_total, dtype=np.float32) + origin_anchor_delta.astype(np.float32)
    w2c = np.linalg.inv(c2w)
    ground_residual_z = float(np.median(merged) + float(t_total[2]))

    meta = dict(metadata or {})
    meta["grounding_transform_id"] = transform_id
    meta["grounding_to_z0"] = {
        "enabled": True,
        "source": cfg.source,
        "shift_z_m": float(shift_z),
        "origin_anchor_correction": origin_anchor_delta.astype(float).tolist(),
        "ground_residual_z_m": float(ground_residual_z),
        "sample_count": int(merged.size),
    }
    meta.update(origin_anchor_meta)
    meta["origin_anchor_frame_index"] = anchor_frame_idx
    samples: list[PoseSample] = []
    for idx, frame_idx in enumerate(frame_indices.tolist()):
        conf = float(confidence[idx]) if confidence is not None and confidence.size > idx else None
        samples.append(
            PoseSample(
                frame_index=int(frame_idx),
                camera_to_world=c2w[idx],
                world_to_camera=w2c[idx],
                confidence=conf,
                metadata=dict(meta),
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata=meta))

    _apply_transform_to_point_cloud(
        store,
        r_total=r_total,
        scale=1.0,
        t_total=t_total,
        transform_id=transform_id,
    )
    _apply_transform_to_road_planes(
        store,
        r_total=r_total,
        scale=1.0,
        t_total=t_total,
        transform_id=transform_id,
    )
    _apply_transform_to_road_plane_sampled_points(
        store,
        r_total=r_total,
        scale=1.0,
        t_total=t_total,
        transform_id=transform_id,
    )
    vis_dir = store.visualizations_dir("alignment")
    vis_dir.mkdir(parents=True, exist_ok=True)
    (vis_dir / "grounding_summary.json").write_text(
        json.dumps(
            {
                "grounding_transform_id": transform_id,
                "shift_z_m": float(shift_z),
                "origin_anchor_correction": origin_anchor_delta.astype(float).tolist(),
                "source": cfg.source,
                "sample_count": int(merged.size),
                "median_ground_z_before_m": float(np.median(merged)),
                "median_ground_z_after_m": float(np.median(merged + float(t_total[2]))),
                "ground_residual_z_m": float(ground_residual_z),
                **road_plane_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info(
        "[Alignment] Applied grounding-to-z0 transform_id=%s shift_z_m=%.6f samples=%d",
        transform_id,
        float(shift_z),
        int(merged.size),
    )
