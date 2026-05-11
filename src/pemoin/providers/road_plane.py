"""Upgraded robust road-plane estimation provider.

This module implements:
- robust global road-plane estimation (confidence/ROI/IRLS/trim)
- optional state-space filtering over roll/pitch/height
- adaptive temporal windows
- optional multi-hypothesis fallback under mixture/outlier failure modes
- optional road-aligned metric grid summaries
"""

from __future__ import annotations

import json
import math
import shutil
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import cv2
import numpy as np

from pemoin.data.contracts import (
    HEIGHT_METADATA_REQUIRED_FIELDS,
    DepthData,
    IntrinsicsData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    RoadPlaneData,
    RoadPlaneSupportData,
    SemanticsData,
    SemanticsAuxData,
)
from pemoin.geometry.camera_model import (
    backproject_uv_depth_to_camera,
    camera_to_world,
    project_world_to_image,
)
from pemoin.providers.road_plane_internal import (
    SimpleRoadStateFilter,
    assert_plane_residual_metadata_consistency,
    compute_plane_quality,
    huber_weights,
    solve_plane_weighted,
)
from pemoin.providers.base import Provider, ProviderExecutionMode
from pemoin.providers.semantic_roles import resolve_semantic_role_labels, resolve_role_label_ids
from pemoin.utils.logging import get_logger
from pemoin.validation.policy import AdaptiveValidationContext, ValidationPolicySettings
from pemoin.visualization.road_geometry_debug import write_road_geometry_debug_artifacts
from pemoin.visualization.road_plane import (
    write_road_plane_overlay_image,
    write_road_plane_residuals_image,
)
from pemoin.visualization.video import write_video

LOG = get_logger()


def _resolve_sampling_fps(context: Mapping[str, object] | None) -> float:
    """Resolve road-plane video FPS from runtime frame provider settings."""
    if not isinstance(context, Mapping):
        raise ValueError(
            "Runtime context must be provided when generating road-plane videos."
        )
    frame_provider_info = context.get("frame_provider_info")
    if not isinstance(frame_provider_info, Mapping):
        raise ValueError(
            "frame_provider_info must be provided in runtime context for road-plane videos."
        )
    settings = frame_provider_info.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError(
            "frame_provider_info.settings must be provided for road-plane videos."
        )
    sampling_fps = settings.get("sampling_fps")
    if sampling_fps is None:
        raise ValueError(
            "frame_provider_info.settings.sampling_fps must be resolved for road-plane videos."
        )
    fps = float(sampling_fps)
    if fps <= 0.0:
        raise ValueError(
            "frame_provider_info.settings.sampling_fps must be > 0 for road-plane videos."
        )
    return fps


@dataclass(frozen=True)
class RoadPlaneSettings:
    """Configuration for upgraded road-plane estimation."""

    # Global robust baseline
    road_labels: Tuple[str, ...] = ("road",)
    sidewalk_labels: Tuple[str, ...] = ()
    include_sidewalk_in_support: bool = True
    support_source: str = "frame_depth_semantics"
    support_pixel_stride: int = 2
    support_min_confidence: float = 0.3
    support_min_points: int = 800
    support_max_layering_ratio: float = 0.40
    support_max_depth_spread_p90_m: float = 1.5
    support_adaptive_forward_min_enabled: bool = True
    support_forward_min_floor_m: float = 1.0
    support_forward_min_step_m: float = 0.5
    support_target_min_points: int = 1000
    support_adaptive_forward_max_iters: int = 5
    auto_disable_height_anchor_for_nonmetric_depth: bool = True
    max_reused_frames: int = 3
    points_per_frame: int = 12000
    forward_min_m: float = 3.0
    forward_max_m: float = 25.0
    lateral_max_m: float = 6.0
    vertical_max_m: float = 3.0
    sampling_weight_power: float = 1.0
    enforce_anchor_from_camera_height: bool = True

    # Temporal windowing
    window_half_width: int = 6
    adaptive_window_enabled: bool = False
    window_causal_only: bool = True
    window_exclude_catastrophic_frames: bool = True
    window_exclude_low_support_frames: bool = True
    window_min_support_points_for_inclusion: int = 600
    window_min_frames_required: int = 3
    window_min_half_width: int = 6
    window_max_half_width: int = 6
    motion_turn_weight: float = 2.0
    motion_speed_weight: float = 1.0

    # Robust fitting
    huber_delta: float = 0.06
    lambda_height: float = 200.0  # Backward-compatible alias for lambda_up.
    lambda_up: float = 200.0
    lambda_temp: float = 25.0
    trim_ratio: float = 0.2
    irls_iters: int = 2
    state_causal_smoothing: int = 3

    # Temporal smoothing/filtering
    temporal_mode: str = "state_filter"  # "ema" or "state_filter"
    ema_alpha: float = 0.3
    state_process_noise_roll: float = 5.0e-4
    state_process_noise_pitch: float = 5.0e-4
    state_process_noise_height: float = 2.0e-3
    state_meas_noise_roll: float = 4.0e-2
    state_meas_noise_pitch: float = 4.0e-2
    state_meas_noise_height: float = 8.0e-2
    state_innovation_gate: float = 4.0

    # Robust gating + jump clamps
    min_window_points: int = 600
    min_lateral_span_m: float = 3.0
    min_forward_span_m: float = 6.0
    max_condition_number: float = 1.0e4
    min_left_right_balance_ratio: float = 0.25
    gating_max_residual_p90_m: float = 0.25
    gating_min_inlier_ratio: float = 0.55
    catastrophic_residual_p90_m: float = 1.5
    catastrophic_min_inlier_ratio: float = 0.02
    saved_point_max_residual_p90_m: float = 0.30
    saved_point_min_inlier_ratio: float = 0.50
    saved_point_startup_grace_frames: int = 5
    saved_point_startup_max_residual_p90_m: float = 0.60
    saved_point_startup_min_inlier_ratio: float = 0.20
    max_saved_point_skips: int = 3
    allow_degraded_output: bool = True
    recovery_fit_enabled: bool = True
    recovery_fit_min_points: int = 400
    recovery_fit_huber_delta_scale: float = 1.25
    recovery_fit_accept_max_residual_p90_m: float = 0.45
    recovery_fit_accept_min_inlier_ratio: float = 0.30
    max_roll_pitch_step_deg: float = 1.0
    max_height_step_m: float = 0.05

    # Multi-hypothesis fallback
    multi_hypothesis_enabled: bool = True
    multi_hypothesis_min_inlier_ratio: float = 0.55
    multi_hypothesis_p90_trigger_m: float = 0.20

    # Optional metric grid
    metric_grid_enabled: bool = False
    metric_grid_cell_size_m: float = 0.5
    metric_grid_max_points_per_frame: int = 1200

    # Visualization/debug
    overlay_extent_m: float = 30.0
    overlay_max_points: int = 40000
    seed: int = 123
    viz_point_stride_m: float = 0.1
    viz_confidence_threshold: float = 0.7
    viz_residual_clamp: float = 0.3
    viz_video_codec: str = "mp4v"
    residual_metadata_tolerance_m: float = 1e-5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RoadPlaneSettings":
        for forbidden_key in ("road_labels", "sidewalk_labels"):
            if mapping.get(forbidden_key) is not None:
                raise ValueError(
                    f"road_plane.{forbidden_key} is no longer supported; semantic roles are resolved automatically."
                )
        def _tuple(value: Any, default: Tuple[str, ...]) -> Tuple[str, ...]:
            if isinstance(value, (list, tuple)):
                return tuple(str(v).strip().lower() for v in value if str(v).strip())
            if isinstance(value, str):
                return tuple(v.strip().lower() for v in value.split(",") if v.strip())
            return default

        def _bool(key: str, default: bool) -> bool:
            return bool(mapping.get(key, default))

        return cls(
            include_sidewalk_in_support=_bool("include_sidewalk_in_support", cls.include_sidewalk_in_support),
            support_source=str(mapping.get("support_source", cls.support_source)).strip().lower(),
            support_pixel_stride=int(mapping.get("support_pixel_stride", cls.support_pixel_stride)),
            support_min_confidence=float(mapping.get("support_min_confidence", cls.support_min_confidence)),
            support_min_points=int(mapping.get("support_min_points", cls.support_min_points)),
            support_max_layering_ratio=float(
                mapping.get("support_max_layering_ratio", cls.support_max_layering_ratio)
            ),
            support_max_depth_spread_p90_m=float(
                mapping.get("support_max_depth_spread_p90_m", cls.support_max_depth_spread_p90_m)
            ),
            support_adaptive_forward_min_enabled=_bool(
                "support_adaptive_forward_min_enabled", cls.support_adaptive_forward_min_enabled
            ),
            support_forward_min_floor_m=float(
                mapping.get("support_forward_min_floor_m", cls.support_forward_min_floor_m)
            ),
            support_forward_min_step_m=float(
                mapping.get("support_forward_min_step_m", cls.support_forward_min_step_m)
            ),
            support_target_min_points=int(
                mapping.get("support_target_min_points", cls.support_target_min_points)
            ),
            support_adaptive_forward_max_iters=int(
                mapping.get(
                    "support_adaptive_forward_max_iters",
                    cls.support_adaptive_forward_max_iters,
                )
            ),
            auto_disable_height_anchor_for_nonmetric_depth=_bool(
                "auto_disable_height_anchor_for_nonmetric_depth",
                cls.auto_disable_height_anchor_for_nonmetric_depth,
            ),
            max_reused_frames=int(mapping.get("max_reused_frames", cls.max_reused_frames)),
            points_per_frame=int(mapping.get("points_per_frame", cls.points_per_frame)),
            forward_min_m=float(mapping.get("forward_min_m", cls.forward_min_m)),
            forward_max_m=float(mapping.get("forward_max_m", cls.forward_max_m)),
            lateral_max_m=float(mapping.get("lateral_max_m", cls.lateral_max_m)),
            vertical_max_m=float(mapping.get("vertical_max_m", cls.vertical_max_m)),
            sampling_weight_power=float(mapping.get("sampling_weight_power", cls.sampling_weight_power)),
            enforce_anchor_from_camera_height=bool(
                mapping.get("enforce_anchor_from_camera_height", cls.enforce_anchor_from_camera_height)
            ),
            window_half_width=int(mapping.get("window_half_width", cls.window_half_width)),
            adaptive_window_enabled=_bool("adaptive_window_enabled", cls.adaptive_window_enabled),
            window_causal_only=_bool("window_causal_only", cls.window_causal_only),
            window_exclude_catastrophic_frames=_bool(
                "window_exclude_catastrophic_frames", cls.window_exclude_catastrophic_frames
            ),
            window_exclude_low_support_frames=_bool(
                "window_exclude_low_support_frames", cls.window_exclude_low_support_frames
            ),
            window_min_support_points_for_inclusion=int(
                mapping.get(
                    "window_min_support_points_for_inclusion",
                    cls.window_min_support_points_for_inclusion,
                )
            ),
            window_min_frames_required=int(
                mapping.get("window_min_frames_required", cls.window_min_frames_required)
            ),
            window_min_half_width=int(mapping.get("window_min_half_width", cls.window_min_half_width)),
            window_max_half_width=int(mapping.get("window_max_half_width", cls.window_max_half_width)),
            motion_turn_weight=float(mapping.get("motion_turn_weight", cls.motion_turn_weight)),
            motion_speed_weight=float(mapping.get("motion_speed_weight", cls.motion_speed_weight)),
            huber_delta=float(mapping.get("huber_delta", cls.huber_delta)),
            lambda_height=float(mapping.get("lambda_height", cls.lambda_height)),
            lambda_up=float(mapping.get("lambda_up", mapping.get("lambda_height", cls.lambda_up))),
            lambda_temp=float(mapping.get("lambda_temp", cls.lambda_temp)),
            trim_ratio=float(mapping.get("trim_ratio", cls.trim_ratio)),
            irls_iters=int(mapping.get("irls_iters", cls.irls_iters)),
            state_causal_smoothing=int(mapping.get("state_causal_smoothing", cls.state_causal_smoothing)),
            temporal_mode=str(mapping.get("temporal_mode", cls.temporal_mode)).strip().lower(),
            ema_alpha=float(mapping.get("ema_alpha", cls.ema_alpha)),
            state_process_noise_roll=float(mapping.get("state_process_noise_roll", cls.state_process_noise_roll)),
            state_process_noise_pitch=float(mapping.get("state_process_noise_pitch", cls.state_process_noise_pitch)),
            state_process_noise_height=float(mapping.get("state_process_noise_height", cls.state_process_noise_height)),
            state_meas_noise_roll=float(mapping.get("state_meas_noise_roll", cls.state_meas_noise_roll)),
            state_meas_noise_pitch=float(mapping.get("state_meas_noise_pitch", cls.state_meas_noise_pitch)),
            state_meas_noise_height=float(mapping.get("state_meas_noise_height", cls.state_meas_noise_height)),
            state_innovation_gate=float(mapping.get("state_innovation_gate", cls.state_innovation_gate)),
            min_window_points=int(mapping.get("min_window_points", cls.min_window_points)),
            min_lateral_span_m=float(mapping.get("min_lateral_span_m", cls.min_lateral_span_m)),
            min_forward_span_m=float(mapping.get("min_forward_span_m", cls.min_forward_span_m)),
            max_condition_number=float(mapping.get("max_condition_number", cls.max_condition_number)),
            min_left_right_balance_ratio=float(
                mapping.get("min_left_right_balance_ratio", cls.min_left_right_balance_ratio)
            ),
            gating_max_residual_p90_m=float(
                mapping.get("gating_max_residual_p90_m", cls.gating_max_residual_p90_m)
            ),
            gating_min_inlier_ratio=float(
                mapping.get("gating_min_inlier_ratio", cls.gating_min_inlier_ratio)
            ),
            catastrophic_residual_p90_m=float(
                mapping.get("catastrophic_residual_p90_m", cls.catastrophic_residual_p90_m)
            ),
            catastrophic_min_inlier_ratio=float(
                mapping.get("catastrophic_min_inlier_ratio", cls.catastrophic_min_inlier_ratio)
            ),
            saved_point_max_residual_p90_m=float(
                mapping.get("saved_point_max_residual_p90_m", cls.saved_point_max_residual_p90_m)
            ),
            saved_point_min_inlier_ratio=float(
                mapping.get("saved_point_min_inlier_ratio", cls.saved_point_min_inlier_ratio)
            ),
            saved_point_startup_grace_frames=int(
                mapping.get("saved_point_startup_grace_frames", cls.saved_point_startup_grace_frames)
            ),
            saved_point_startup_max_residual_p90_m=float(
                mapping.get(
                    "saved_point_startup_max_residual_p90_m",
                    cls.saved_point_startup_max_residual_p90_m,
                )
            ),
            saved_point_startup_min_inlier_ratio=float(
                mapping.get(
                    "saved_point_startup_min_inlier_ratio",
                    cls.saved_point_startup_min_inlier_ratio,
                )
            ),
            max_saved_point_skips=int(mapping.get("max_saved_point_skips", cls.max_saved_point_skips)),
            allow_degraded_output=_bool("allow_degraded_output", cls.allow_degraded_output),
            recovery_fit_enabled=_bool("recovery_fit_enabled", cls.recovery_fit_enabled),
            recovery_fit_min_points=int(
                mapping.get("recovery_fit_min_points", cls.recovery_fit_min_points)
            ),
            recovery_fit_huber_delta_scale=float(
                mapping.get("recovery_fit_huber_delta_scale", cls.recovery_fit_huber_delta_scale)
            ),
            recovery_fit_accept_max_residual_p90_m=float(
                mapping.get(
                    "recovery_fit_accept_max_residual_p90_m",
                    cls.recovery_fit_accept_max_residual_p90_m,
                )
            ),
            recovery_fit_accept_min_inlier_ratio=float(
                mapping.get(
                    "recovery_fit_accept_min_inlier_ratio",
                    cls.recovery_fit_accept_min_inlier_ratio,
                )
            ),
            max_roll_pitch_step_deg=float(
                mapping.get("max_roll_pitch_step_deg", cls.max_roll_pitch_step_deg)
            ),
            max_height_step_m=float(mapping.get("max_height_step_m", cls.max_height_step_m)),
            multi_hypothesis_enabled=_bool("multi_hypothesis_enabled", cls.multi_hypothesis_enabled),
            multi_hypothesis_min_inlier_ratio=float(mapping.get("multi_hypothesis_min_inlier_ratio", cls.multi_hypothesis_min_inlier_ratio)),
            multi_hypothesis_p90_trigger_m=float(mapping.get("multi_hypothesis_p90_trigger_m", cls.multi_hypothesis_p90_trigger_m)),
            metric_grid_enabled=_bool("metric_grid_enabled", cls.metric_grid_enabled),
            metric_grid_cell_size_m=float(mapping.get("metric_grid_cell_size_m", cls.metric_grid_cell_size_m)),
            metric_grid_max_points_per_frame=int(mapping.get("metric_grid_max_points_per_frame", cls.metric_grid_max_points_per_frame)),
            overlay_extent_m=float(mapping.get("overlay_extent_m", cls.overlay_extent_m)),
            overlay_max_points=int(mapping.get("overlay_max_points", cls.overlay_max_points)),
            seed=int(mapping.get("seed", cls.seed)),
            viz_point_stride_m=float(mapping.get("viz_point_stride_m", cls.viz_point_stride_m)),
            viz_confidence_threshold=float(mapping.get("viz_confidence_threshold", cls.viz_confidence_threshold)),
            viz_residual_clamp=float(mapping.get("viz_residual_clamp", cls.viz_residual_clamp)),
            viz_video_codec=str(mapping.get("viz_video_codec", cls.viz_video_codec)),
            residual_metadata_tolerance_m=float(
                mapping.get("residual_metadata_tolerance_m", cls.residual_metadata_tolerance_m)
            ),
        )


@dataclass
class FrameBundle:
    frame_idx: int
    pose: PoseSample
    camera_center: np.ndarray
    camera_height: float


@dataclass
class PlaneResult:
    normal: np.ndarray
    offset: float
    residuals: np.ndarray
    cov_diag: np.ndarray
    quality: Dict[str, float]
    hypothesis: str = "single"


@dataclass(frozen=True)
class FrameRoadPointSample:
    points: np.ndarray
    weights: np.ndarray
    diagnostics: Dict[str, float]


class RobustRoadPlaneProvider(Provider):
    execution_mode = ProviderExecutionMode.BATCH
    required_resources = frozenset(
        {
            ResourceKind.DEPTH,
            ResourceKind.INTRINSICS,
            ResourceKind.TRAJECTORY,
            ResourceKind.CAMERA_HEIGHT,
            ResourceKind.FRAMES,
            ResourceKind.SEMANTICS_2D,
        }
    )
    produced_resources = frozenset({ResourceKind.ROAD_PLANE})

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = RoadPlaneSettings.from_mapping(settings)

    def setup(self, context: MutableMapping[str, Any]):
        self._semantics_tool = (
            str(context.get("semantics_tool")) if context.get("semantics_tool") is not None else None
        )
        self._semantic_role_defaults = (
            dict(context.get("semantic_role_defaults"))
            if isinstance(context.get("semantic_role_defaults"), Mapping)
            else None
        )
        return None

    def process(self, frame: Any):
        raise NotImplementedError("RobustRoadPlaneProvider runs in batch mode.")

    def teardown(self):
        return None

    def run(
        self,
        resources: ResourceStore,
        context: MutableMapping[str, object] | None = None,
    ) -> None:
        self.validate_requirements(resources)
        self._validate_sampling_settings()
        raw_policy = context.get("validation_policy") if isinstance(context, Mapping) else None
        policy = ValidationPolicySettings.from_mapping(raw_policy if isinstance(raw_policy, Mapping) else None)
        adaptive = AdaptiveValidationContext.from_runtime(policy, context)
        max_reused_frames_soft, max_reused_frames_hard = adaptive.max_count_thresholds(
            int(self.settings.max_reused_frames)
        )
        max_saved_point_skips_soft, max_saved_point_skips_hard = adaptive.max_count_thresholds(
            int(self.settings.max_saved_point_skips)
        )
        if self.settings.temporal_mode != "state_filter":
            raise RuntimeError(
                "RobustRoadPlaneProvider requires temporal_mode='state_filter'. "
                "EMA mode is disabled for robustness."
            )
        self._reset_visualization_artifacts(resources)
        frame_indices = self._trajectory_frames(resources)
        bundles = self._load_frame_bundles(resources, frame_indices)
        LOG.info(
            "[RoadPlane] Starting upgraded estimation: frames=%d temporal_mode=%s adaptive_window=%s "
            "multi_hypothesis=%s metric_grid=%s",
            len(frame_indices),
            self.settings.temporal_mode,
            self.settings.adaptive_window_enabled,
            self.settings.multi_hypothesis_enabled,
            self.settings.metric_grid_enabled,
        )

        intrinsics = resources.load_intrinsics()
        replacement_map: Dict[int, int] = {}
        skipped_frames: tuple[int, ...] = tuple()
        if isinstance(context, Mapping):
            raw_rep = context.get("geometry_consistency_replacement_map")
            if isinstance(raw_rep, Mapping):
                replacement_map = {int(k): int(v) for k, v in raw_rep.items()}
            raw_skip = context.get("geometry_consistency_skipped_frames")
            if isinstance(raw_skip, (list, tuple)):
                skipped_frames = tuple(sorted(int(v) for v in raw_skip))
        if len(skipped_frames) > int(max_reused_frames_hard):
            raise RuntimeError(
                f"Road-plane requires reusing {len(skipped_frames)} catastrophic frames, "
                f"limit is max_reused_frames={max_reused_frames_hard}."
            )
        if len(skipped_frames) > int(max_reused_frames_soft):
            LOG.warning(
                "[RoadPlane] Reused-frame count exceeds soft limit: %d > %d (hard=%d).",
                len(skipped_frames),
                int(max_reused_frames_soft),
                int(max_reused_frames_hard),
            )
        if skipped_frames:
            LOG.warning(
                "[RoadPlane] Reusing nearby frame data for catastrophic frames: %s",
                skipped_frames,
            )

        rng = np.random.default_rng(self.settings.seed)
        sampled_points: Dict[int, np.ndarray] = {}
        sampled_weights: Dict[int, np.ndarray] = {}
        sampled_diagnostics: Dict[int, Dict[str, float]] = {}
        for frame_idx in frame_indices:
            source_idx = int(replacement_map.get(int(frame_idx), int(frame_idx)))
            pose = resources.load_pose(source_idx)
            depth_data = resources.load_depth(source_idx)
            semantics = resources.load_semantics2d(source_idx)
            sample = self._sample_frame_points_from_frame_resources(
                resources=resources,
                frame_idx=int(frame_idx),
                source_idx=int(source_idx),
                depth_data=depth_data,
                semantics=semantics,
                pose=pose,
                intrinsics=intrinsics,
                rng=rng,
                context=context,
            )
            sampled_points[frame_idx] = sample.points
            sampled_weights[frame_idx] = sample.weights
            sampled_diagnostics[frame_idx] = dict(sample.diagnostics)
            resources.save_road_plane_support(
                RoadPlaneSupportData(
                    frame_index=int(frame_idx),
                    points_world=sample.points.astype(np.float32),
                    weights=sample.weights.astype(np.float32),
                    source_frame_index=int(source_idx),
                    diagnostics=dict(sample.diagnostics),
                    metadata={"source": "road_plane"},
                )
            )
            LOG.info(
                "[RoadPlane][Support] frame=%d source_frame=%d support_pixels=%d valid_pixels=%d "
                "points=%d layering_ratio=%.3f depth_spread_p90=%.3f",
                int(frame_idx),
                int(source_idx),
                int(sample.diagnostics.get("support_pixels", 0.0)),
                int(sample.diagnostics.get("valid_pixels", 0.0)),
                int(sample.points.shape[0]),
                float(sample.diagnostics.get("layering_ratio", 0.0)),
                float(sample.diagnostics.get("layering_depth_spread_p90_m", 0.0)),
            )
        point_counts = np.asarray([sampled_points[idx].shape[0] for idx in frame_indices], dtype=np.int32)
        non_empty = int(np.count_nonzero(point_counts > 0))
        min_points = int(point_counts.min()) if point_counts.size else 0
        max_points = int(point_counts.max()) if point_counts.size else 0
        med_points = int(np.median(point_counts)) if point_counts.size else 0
        layering_ratio = np.asarray(
            [float(sampled_diagnostics[idx].get("layering_ratio", 0.0)) for idx in frame_indices],
            dtype=np.float32,
        )
        depth_spread = np.asarray(
            [float(sampled_diagnostics[idx].get("layering_depth_spread_p90_m", 0.0)) for idx in frame_indices],
            dtype=np.float32,
        )
        LOG.info(
            "[RoadPlane] Sampled support points per frame: non_empty=%d/%d min=%d median=%d max=%d "
            "layering_ratio[med=%.3f p95=%.3f] depth_spread_p90[med=%.3f p95=%.3f]",
            non_empty,
            len(frame_indices),
            min_points,
            med_points,
            max_points,
            float(np.median(layering_ratio)) if layering_ratio.size else 0.0,
            float(np.percentile(layering_ratio, 95)) if layering_ratio.size else 0.0,
            float(np.median(depth_spread)) if depth_spread.size else 0.0,
            float(np.percentile(depth_spread, 95)) if depth_spread.size else 0.0,
        )
        metric_scale_series = np.asarray(
            [float(sampled_diagnostics[idx].get("metric_scale", 0.0)) for idx in frame_indices],
            dtype=np.float32,
        )
        nonmetric_support_mode = bool(np.median(metric_scale_series) < 0.5)
        if nonmetric_support_mode and self.settings.auto_disable_height_anchor_for_nonmetric_depth:
            LOG.warning(
                "[RoadPlane] Non-metric depth detected in support sampling (median metric_scale=%.3f); "
                "disabling hard camera-height anchoring for pre-alignment fit.",
                float(np.median(metric_scale_series)),
            )
        enforce_height_anchor = bool(
            self.settings.enforce_anchor_from_camera_height
            and not (
                nonmetric_support_mode
                and self.settings.auto_disable_height_anchor_for_nonmetric_depth
            )
        )

        motion_scores = self._compute_motion_scores(frame_indices, bundles)
        if motion_scores:
            arr = np.asarray(motion_scores, dtype=np.float32)
            LOG.info(
                "[RoadPlane] Motion score summary for adaptive windows: min=%.3f median=%.3f max=%.3f",
                float(np.min(arr)),
                float(np.median(arr)),
                float(np.max(arr)),
            )

        state_filter = SimpleRoadStateFilter(self.settings)
        prev_plane: tuple[np.ndarray, float] | None = None

        residuals_frames: List[np.ndarray] = []
        overlay_frames: List[np.ndarray] = []

        global_history: List[Dict[str, Any]] = []
        grid_cells: Dict[tuple[int, int], Dict[str, Any]] = {}
        debug_camera_height_series: List[float] = []
        debug_plane_height_series: List[float] = []
        debug_fit_residual_p90_series: List[float] = []
        debug_saved_residual_p90_series: List[float] = []
        debug_support_layering_ratio_series: List[float] = []
        debug_support_depth_spread_p90_series: List[float] = []
        debug_adaptive_forward_min_series: List[float] = []
        debug_window_included_frame_count_series: List[float] = []
        debug_recovery_used_series: List[float] = []
        debug_recovery_accepted_series: List[float] = []
        state_roll_history: List[float] = []
        state_pitch_history: List[float] = []
        measurement_skip_count = 0
        catastrophic_frames_set = set(int(v) for v in skipped_frames)
        excluded_for_window: set[int] = set()
        if self.settings.window_exclude_catastrophic_frames:
            excluded_for_window.update(catastrophic_frames_set)
        support_point_counts = {
            int(idx): int(sampled_diagnostics.get(int(idx), {}).get("roi_points", 0.0))
            for idx in frame_indices
        }

        for i, frame_idx in enumerate(frame_indices):
            half_width = self._window_half_width(i, motion_scores)
            (
                points,
                weights,
                centers,
                camera_height_m,
                anchor_camera_center,
                window_meta,
            ) = self._collect_window_points_from_cache(
                frame_indices=frame_indices,
                center_idx=i,
                half_width=half_width,
                sampled_points=sampled_points,
                sampled_weights=sampled_weights,
                bundles=bundles,
                excluded_frames=frozenset(excluded_for_window),
                support_point_counts=support_point_counts,
            )
            if points.size == 0:
                raise RuntimeError(f"No road points available for frame {frame_idx}.")
            LOG.info(
                "[RoadPlane] Frame %d: window_half_width=%d window_points=%d window_centers=%d included_frames=%d "
                "excluded[cat=%d low_support=%d] camera_height=%.3f",
                frame_idx,
                half_width,
                points.shape[0],
                centers.shape[0],
                int(window_meta.get("window_included_frame_count", 0)),
                int(window_meta.get("window_excluded_catastrophic_count", 0)),
                int(window_meta.get("window_excluded_low_support_count", 0)),
                camera_height_m,
            )

            global_result = self._fit_global_plane(
                points=points,
                weights=weights,
                centers=centers,
                camera_height_m=camera_height_m,
                anchor_camera_center=anchor_camera_center,
                prev_plane=prev_plane,
                enforce_height_anchor=enforce_height_anchor,
            )
            LOG.info(
                "[RoadPlane][Fit] frame=%d residual_median=%.4f residual_p90=%.4f inlier_ratio=%.3f",
                frame_idx,
                float(global_result.quality.get("residual_median", -1.0)),
                float(global_result.quality.get("residual_p90", -1.0)),
                float(global_result.quality.get("inlier_ratio", -1.0)),
            )

            gate_ok, gate_meta, pre_quality_score = self._evaluate_pre_fit_gates(
                points=points,
                center_bundle=bundles[frame_idx],
            )
            if (
                gate_ok
                and self.settings.multi_hypothesis_enabled
                and self._needs_multi_hypothesis(global_result)
            ):
                LOG.info(
                    "[RoadPlane] Frame %d triggering multi-hypothesis fallback (inlier_ratio=%.3f, residual_p90=%.4f).",
                    frame_idx,
                    float(global_result.quality.get("inlier_ratio", -1.0)),
                    float(global_result.quality.get("residual_p90", -1.0)),
                )
                alt = self._fit_multi_hypothesis(
                    points=points,
                    weights=weights,
                    centers=centers,
                    camera_height_m=camera_height_m,
                    anchor_camera_center=anchor_camera_center,
                    prev_plane=prev_plane,
                    base=global_result,
                    enforce_height_anchor=enforce_height_anchor,
                )
                if alt is not None:
                    global_result = alt
                    LOG.info(
                        "[RoadPlane] Frame %d selected %s hypothesis: residual_p90=%.4f inlier_ratio=%.3f",
                        frame_idx,
                        global_result.hypothesis,
                        float(global_result.quality.get("residual_p90", -1.0)),
                        float(global_result.quality.get("inlier_ratio", -1.0)),
                    )

            post_gate_ok, post_gate_reason, post_quality_score = self._evaluate_post_fit_gates(global_result)
            quality_score = float(np.clip(pre_quality_score * post_quality_score, 0.0, 1.0))
            measurement_allowed = gate_ok and post_gate_ok
            gate_reason = "ok"
            gate_reasons_all: List[str] = []
            if not gate_ok:
                reason = str(gate_meta.get("gate_reason", "pre_fit_gate_failed"))
                gate_reason = self._resolve_gate_reason(current=gate_reason, candidate=reason)
                gate_reasons_all.append(reason)
            elif not post_gate_ok:
                gate_reason = self._resolve_gate_reason(current=gate_reason, candidate=post_gate_reason)
                gate_reasons_all.append(str(post_gate_reason))
            elif post_gate_reason != "ok":
                gate_reasons_all.append(str(post_gate_reason))
            support_diag = sampled_diagnostics.get(frame_idx, {})
            support_points = int(support_diag.get("roi_points", 0.0))
            support_layering = float(support_diag.get("layering_ratio", 0.0))
            support_depth_spread = float(support_diag.get("layering_depth_spread_p90_m", 0.0))
            support_quality_ok = self._evaluate_support_quality_gate(
                support_points=support_points,
                layering_ratio=support_layering,
                depth_spread_p90_m=support_depth_spread,
            )
            if not support_quality_ok:
                measurement_allowed = False
                gate_reason = self._resolve_gate_reason(
                    current=gate_reason,
                    candidate="support_quality",
                )
                gate_reasons_all.append("support_quality")
            save_points = sampled_points.get(frame_idx, points)
            _, pre_saved_quality = compute_plane_quality(
                save_points,
                global_result.normal,
                global_result.offset,
                inlier_threshold_m=max(self.settings.huber_delta * 2.0, 0.1),
            )
            saved_gate_ok, startup_grace_used = self._evaluate_saved_point_gate(
                residual_p90=float(pre_saved_quality.residual_p90),
                inlier_ratio=float(pre_saved_quality.inlier_ratio),
                frame_order_index=i,
            )
            if not saved_gate_ok:
                measurement_allowed = False
                gate_reason = self._resolve_gate_reason(
                    current=gate_reason,
                    candidate="saved_point_quality",
                )
                gate_reasons_all.append("saved_point_quality")
            elif measurement_allowed and startup_grace_used:
                gate_reason = "saved_point_quality_startup_grace"
                LOG.info(
                    "[RoadPlane] Frame %d accepted by startup grace: residual_p90=%.4f inlier_ratio=%.3f "
                    "(strict<=%.3f/>=%.3f; startup<=%.3f/>=%.3f; order=%d/%d).",
                    frame_idx,
                    float(pre_saved_quality.residual_p90),
                    float(pre_saved_quality.inlier_ratio),
                    float(self.settings.saved_point_max_residual_p90_m),
                    float(self.settings.saved_point_min_inlier_ratio),
                    float(self.settings.saved_point_startup_max_residual_p90_m),
                    float(self.settings.saved_point_startup_min_inlier_ratio),
                    i + 1,
                    int(self.settings.saved_point_startup_grace_frames),
                )
            recovery_meta = {
                "recovery_fit_used": 0.0,
                "recovery_fit_accepted": 0.0,
            }
            if not measurement_allowed:
                recovery_ok, recovery_result, recovery_meta = self._try_recovery_fit(
                    frame_idx=int(frame_idx),
                    center_bundle=bundles[frame_idx],
                    support_points_world=save_points,
                    support_weights=sampled_weights.get(frame_idx, np.ones((save_points.shape[0],), dtype=np.float32)),
                    pre_saved_quality=pre_saved_quality,
                    frame_order_index=i,
                    prev_plane=prev_plane,
                    enforce_height_anchor=enforce_height_anchor,
                )
                if recovery_ok and recovery_result is not None:
                    measurement_allowed = True
                    global_result = recovery_result
                    gate_reason = self._resolve_gate_reason(
                        current=gate_reason,
                        candidate="recovery_fit",
                    )
                    gate_reasons_all.append("recovery_fit")
                    LOG.info(
                        "[RoadPlane] Frame %d recovery fit accepted: residual_p90=%.4f inlier_ratio=%.3f",
                        frame_idx,
                        float(recovery_result.quality.get("residual_p90", -1.0)),
                        float(recovery_result.quality.get("inlier_ratio", -1.0)),
                    )
            if not measurement_allowed:
                measurement_skip_count += 1
            else:
                measurement_skip_count = 0
            if measurement_skip_count > int(max_saved_point_skips_hard):
                if not bool(self.settings.allow_degraded_output):
                    raise RuntimeError(
                        "Road-plane measurement gating exceeded skip limit at frame "
                        f"{frame_idx}: consecutive_skips={measurement_skip_count} > "
                        f"max_saved_point_skips={max_saved_point_skips_hard}."
                    )
            if measurement_skip_count > int(max_saved_point_skips_soft):
                LOG.warning(
                    "[RoadPlane] Measurement gating exceeded skip limit at frame %d, "
                    "continuing in degraded predict-only mode: consecutive_skips=%d > max_saved_point_skips=%d (hard=%d)",
                    int(frame_idx),
                    int(measurement_skip_count),
                    int(max_saved_point_skips_soft),
                    int(max_saved_point_skips_hard),
                )
                measurement_skip_count = int(max_saved_point_skips_soft)
                gate_reason = self._resolve_gate_reason(
                    current=gate_reason,
                    candidate="degraded_saved_point_skip_limit",
                )
                gate_reasons_all.append("degraded_saved_point_skip_limit")
            if not measurement_allowed:
                LOG.info(
                    "[RoadPlane][Filter] frame=%d mode=predict_only gate_reason=%s reasons=%s",
                    frame_idx,
                    gate_reason,
                    ",".join(gate_reasons_all) if gate_reasons_all else "none",
                )

            filtered_normal, filtered_offset, filter_meta = self._apply_temporal_model(
                frame_idx=frame_idx,
                result=global_result,
                bundle=bundles[frame_idx],
                prev_plane=prev_plane,
                state_filter=state_filter,
                measurement_allowed=measurement_allowed,
                gate_reason=gate_reason,
                quality_score=quality_score,
                state_roll_history=state_roll_history,
                state_pitch_history=state_pitch_history,
            )
            filter_meta["anchor_enforced"] = True
            prev_plane = (filtered_normal, filtered_offset)
            filtered_residuals, filtered_quality = compute_plane_quality(
                points,
                filtered_normal,
                filtered_offset,
                inlier_threshold_m=max(self.settings.huber_delta * 2.0, 0.1),
            )
            _, saved_quality = compute_plane_quality(
                save_points,
                filtered_normal,
                filtered_offset,
                inlier_threshold_m=max(self.settings.huber_delta * 2.0, 0.1),
            )
            assert_plane_residual_metadata_consistency(
                residuals=filtered_residuals,
                metadata_residual_median=filtered_quality.residual_median,
                metadata_residual_p90=filtered_quality.residual_p90,
                tolerance_m=float(self.settings.residual_metadata_tolerance_m),
            )
            debug_camera_height_series.append(float(camera_height_m))
            debug_plane_height_series.append(float(filtered_normal @ bundles[frame_idx].camera_center + filtered_offset))
            debug_fit_residual_p90_series.append(float(global_result.quality.get("residual_p90", 0.0)))
            debug_saved_residual_p90_series.append(float(saved_quality.residual_p90))
            debug_support_layering_ratio_series.append(
                float(sampled_diagnostics.get(frame_idx, {}).get("layering_ratio", 0.0))
            )
            debug_support_depth_spread_p90_series.append(
                float(sampled_diagnostics.get(frame_idx, {}).get("layering_depth_spread_p90_m", 0.0))
            )
            debug_adaptive_forward_min_series.append(
                float(sampled_diagnostics.get(frame_idx, {}).get("adaptive_forward_min_used_m", self.settings.forward_min_m))
            )
            debug_window_included_frame_count_series.append(float(window_meta.get("window_included_frame_count", 0)))
            debug_recovery_used_series.append(float(recovery_meta.get("recovery_fit_used", 0.0)))
            debug_recovery_accepted_series.append(float(recovery_meta.get("recovery_fit_accepted", 0.0)))

            if self.settings.metric_grid_enabled:
                self._update_metric_grid(
                    frame_idx=frame_idx,
                    bundle=bundles[frame_idx],
                    points=points,
                    normal=filtered_normal,
                    offset=filtered_offset,
                    grid_cells=grid_cells,
                )

            metadata = {
                "source": "robust_road_plane_upgraded",
                "num_points": int(points.shape[0]),
                "window_half_width": int(half_width),
                "window_included_frame_count": int(window_meta.get("window_included_frame_count", 0)),
                "window_excluded_catastrophic_count": int(window_meta.get("window_excluded_catastrophic_count", 0)),
                "window_excluded_low_support_count": int(window_meta.get("window_excluded_low_support_count", 0)),
                "measurement_allowed": bool(measurement_allowed),
                "measurement_quality_score": float(quality_score),
                "gate_reason": gate_reason,
                "gate_reason_primary": gate_reason,
                "gate_reasons_all": tuple(gate_reasons_all),
                "enforce_height_anchor": bool(enforce_height_anchor),
                "saved_point_gate_ok": bool(saved_gate_ok),
                "saved_point_startup_grace_used": bool(startup_grace_used),
                "saved_point_skip_count": int(measurement_skip_count),
                "saved_point_gate_residual_p90": float(pre_saved_quality.residual_p90),
                "saved_point_gate_inlier_ratio": float(pre_saved_quality.inlier_ratio),
                "support_quality_ok": bool(support_quality_ok),
                "support_points": float(support_points),
                "support_layering_ratio": float(support_layering),
                "support_depth_spread_p90_m": float(support_depth_spread),
                **{k: float(v) for k, v in recovery_meta.items() if isinstance(v, (float, int))},
                **{
                    f"sampling_{k}": float(v)
                    for k, v in sampled_diagnostics.get(frame_idx, {}).items()
                },
                **gate_meta,
                "hypothesis": global_result.hypothesis,
                **saved_quality.to_metadata(),
                "window_residual_median": float(filtered_quality.residual_median),
                "window_residual_p90": float(filtered_quality.residual_p90),
                "window_inlier_ratio": float(filtered_quality.inlier_ratio),
                "window_fit_point_count": float(filtered_quality.fit_point_count),
                "fit_residual_median": float(global_result.quality.get("residual_median", 0.0)),
                "fit_residual_p90": float(global_result.quality.get("residual_p90", 0.0)),
                "fit_inlier_ratio": float(global_result.quality.get("inlier_ratio", 0.0)),
                "fit_point_count": float(global_result.quality.get("fit_point_count", 0.0)),
                **filter_meta,
            }
            resources.save_road_plane(
                RoadPlaneData(
                    frame_index=int(frame_idx),
                    normal=filtered_normal.astype(np.float32),
                    offset=float(filtered_offset),
                    metadata=metadata,
                )
            )

            viz_result = self._write_debug(
                resources=resources,
                frame_idx=frame_idx,
                points=points,
                normal=filtered_normal,
                offset=filtered_offset,
            )
            if viz_result is not None:
                residuals_frames.append(viz_result[0])
                overlay_frames.append(viz_result[1])

            global_history.append(
                {
                    "frame_index": int(frame_idx),
                    "normal": filtered_normal.astype(float).tolist(),
                    "offset": float(filtered_offset),
                    "measurement_allowed": bool(measurement_allowed),
                    "measurement_quality_score": float(quality_score),
                    "gate_reason": gate_reason,
                    "gate_reasons_all": list(gate_reasons_all),
                    "gate_metrics": gate_meta,
                    "quality": saved_quality.to_metadata(),
                    "window_quality": filtered_quality.to_metadata(),
                    "fit_quality": global_result.quality,
                    "hypothesis": global_result.hypothesis,
                    "temporal": filter_meta,
                    "window": dict(window_meta),
                    "recovery": dict(recovery_meta),
                    "sampling": sampled_diagnostics.get(frame_idx, {}),
                }
            )

        self._generate_videos(
            resources,
            residuals_frames,
            overlay_frames,
            fps=_resolve_sampling_fps(context),
        )
        self._write_upgrade_artifacts(
            resources=resources,
            frame_indices=frame_indices,
            global_history=global_history,
            grid_cells=grid_cells,
            validation_policy=adaptive.diagnostic_summary(),
            effective_max_reused_frames=int(max_reused_frames_soft),
            effective_max_saved_point_skips=int(max_saved_point_skips_soft),
        )
        write_road_geometry_debug_artifacts(
            output_dir=resources.visualizations_dir("road_plane"),
            frame_indices=frame_indices,
            camera_height_m=debug_camera_height_series,
            plane_height_at_camera_m=debug_plane_height_series,
            fit_residual_p90_m=debug_fit_residual_p90_series,
            saved_residual_p90_m=debug_saved_residual_p90_series,
            support_layering_ratio=debug_support_layering_ratio_series,
            support_depth_spread_p90_m=debug_support_depth_spread_p90_series,
            adaptive_forward_min_used_m=debug_adaptive_forward_min_series,
            window_included_frame_count=debug_window_included_frame_count_series,
            recovery_fit_used=debug_recovery_used_series,
            recovery_fit_accepted=debug_recovery_accepted_series,
        )
        LOG.info(
            "[RoadPlane] Completed upgraded estimation: frames=%d metric_grid_cells=%d "
            "residual_p90_fit[med=%.3f] residual_p90_saved[med=%.3f]",
            len(frame_indices),
            len(grid_cells),
            float(np.median(np.asarray(debug_fit_residual_p90_series, dtype=np.float32)))
            if debug_fit_residual_p90_series
            else 0.0,
            float(np.median(np.asarray(debug_saved_residual_p90_series, dtype=np.float32)))
            if debug_saved_residual_p90_series
            else 0.0,
        )

    def refresh_debug_visualizations(
        self,
        resources: ResourceStore,
        context: MutableMapping[str, object] | None = None,
    ) -> None:
        """Regenerate road-plane debug artifacts using currently persisted resources."""
        if not resources.has(ResourceKind.ROAD_PLANE):
            raise RuntimeError("Cannot refresh road-plane visualizations: ROAD_PLANE is missing.")

        frame_indices = self._trajectory_frames(resources)
        support_frames = set(resources.frame_indices(ResourceKind.ROAD_PLANE_SUPPORT))
        if not support_frames:
            LOG.warning(
                "[RoadPlane] Skipping post-alignment debug refresh: ROAD_PLANE_SUPPORT is missing.",
            )
            return

        self._reset_visualization_artifacts(resources)
        residuals_frames: List[np.ndarray] = []
        overlay_frames: List[np.ndarray] = []
        refreshed = 0

        for frame_idx in frame_indices:
            if int(frame_idx) not in support_frames:
                LOG.warning(
                    "[RoadPlane] Missing sampled points for frame %d in standardized road_plane_support; skipping visualization frame.",
                    frame_idx,
                )
                continue
            support = resources.load_road_plane_support(int(frame_idx))
            points = np.asarray(support.points_world, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3:
                raise RuntimeError(
                    f"Invalid sampled points shape for frame {frame_idx}: {points.shape}."
                )

            plane = resources.load_road_plane(int(frame_idx))
            viz_result = self._write_debug(
                resources=resources,
                frame_idx=int(frame_idx),
                points=points,
                normal=np.asarray(plane.normal, dtype=np.float32),
                offset=float(plane.offset),
            )
            if viz_result is None:
                continue
            residuals_frames.append(viz_result[0])
            overlay_frames.append(viz_result[1])
            refreshed += 1

        self._generate_videos(
            resources,
            residuals_frames,
            overlay_frames,
            fps=_resolve_sampling_fps(context),
        )
        LOG.info(
            "[RoadPlane] Refreshed post-alignment debug visualizations: frames=%d/%d",
            refreshed,
            len(frame_indices),
        )

    def _validate_sampling_settings(self) -> None:
        if self.settings.support_source != "frame_depth_semantics":
            raise ValueError("road_plane.support_source must be 'frame_depth_semantics'.")
        if self.settings.support_pixel_stride <= 0:
            raise ValueError("road_plane.support_pixel_stride must be > 0.")
        if not (0.0 <= self.settings.support_min_confidence <= 1.0):
            raise ValueError("road_plane.support_min_confidence must be in [0, 1].")
        if self.settings.support_min_points <= 0:
            raise ValueError("road_plane.support_min_points must be > 0.")
        if not (0.0 <= self.settings.support_max_layering_ratio <= 1.0):
            raise ValueError("road_plane.support_max_layering_ratio must be in [0, 1].")
        if self.settings.support_max_depth_spread_p90_m <= 0.0:
            raise ValueError("road_plane.support_max_depth_spread_p90_m must be > 0.")
        if self.settings.support_forward_min_floor_m < 0.0:
            raise ValueError("road_plane.support_forward_min_floor_m must be >= 0.")
        if self.settings.support_forward_min_floor_m > self.settings.forward_min_m:
            raise ValueError(
                "road_plane.support_forward_min_floor_m must be <= road_plane.forward_min_m."
            )
        if self.settings.support_forward_min_step_m <= 0.0:
            raise ValueError("road_plane.support_forward_min_step_m must be > 0.")
        if self.settings.support_target_min_points <= 0:
            raise ValueError("road_plane.support_target_min_points must be > 0.")
        if self.settings.support_adaptive_forward_max_iters < 0:
            raise ValueError("road_plane.support_adaptive_forward_max_iters must be >= 0.")
        if self.settings.max_reused_frames < 0:
            raise ValueError("road_plane.max_reused_frames must be >= 0.")
        if self.settings.points_per_frame <= 0:
            raise ValueError("road_plane.points_per_frame must be > 0.")
        if self.settings.forward_min_m < 0.0:
            raise ValueError("road_plane.forward_min_m must be >= 0.")
        if self.settings.forward_max_m <= self.settings.forward_min_m:
            raise ValueError("road_plane.forward_max_m must be > road_plane.forward_min_m.")
        if self.settings.lateral_max_m <= 0.0:
            raise ValueError("road_plane.lateral_max_m must be > 0.")
        if self.settings.vertical_max_m <= 0.0:
            raise ValueError("road_plane.vertical_max_m must be > 0.")
        if self.settings.sampling_weight_power < 0.0:
            raise ValueError("road_plane.sampling_weight_power must be >= 0.")
        if self.settings.lambda_up < 0.0:
            raise ValueError("road_plane.lambda_up must be >= 0.")
        if self.settings.catastrophic_residual_p90_m <= 0.0:
            raise ValueError("road_plane.catastrophic_residual_p90_m must be > 0.")
        if self.settings.catastrophic_min_inlier_ratio < 0.0:
            raise ValueError("road_plane.catastrophic_min_inlier_ratio must be >= 0.")
        if self.settings.saved_point_max_residual_p90_m <= 0.0:
            raise ValueError("road_plane.saved_point_max_residual_p90_m must be > 0.")
        if not (0.0 <= self.settings.saved_point_min_inlier_ratio <= 1.0):
            raise ValueError("road_plane.saved_point_min_inlier_ratio must be in [0, 1].")
        if self.settings.saved_point_startup_grace_frames < 0:
            raise ValueError("road_plane.saved_point_startup_grace_frames must be >= 0.")
        if self.settings.saved_point_startup_max_residual_p90_m <= 0.0:
            raise ValueError("road_plane.saved_point_startup_max_residual_p90_m must be > 0.")
        if not (0.0 <= self.settings.saved_point_startup_min_inlier_ratio <= 1.0):
            raise ValueError("road_plane.saved_point_startup_min_inlier_ratio must be in [0, 1].")
        if self.settings.max_saved_point_skips < 0:
            raise ValueError("road_plane.max_saved_point_skips must be >= 0.")
        if self.settings.window_min_support_points_for_inclusion <= 0:
            raise ValueError("road_plane.window_min_support_points_for_inclusion must be > 0.")
        if self.settings.window_min_frames_required <= 0:
            raise ValueError("road_plane.window_min_frames_required must be > 0.")
        if self.settings.recovery_fit_min_points <= 0:
            raise ValueError("road_plane.recovery_fit_min_points must be > 0.")
        if self.settings.recovery_fit_huber_delta_scale <= 0.0:
            raise ValueError("road_plane.recovery_fit_huber_delta_scale must be > 0.")
        if self.settings.recovery_fit_accept_max_residual_p90_m <= 0.0:
            raise ValueError("road_plane.recovery_fit_accept_max_residual_p90_m must be > 0.")
        if not (0.0 <= self.settings.recovery_fit_accept_min_inlier_ratio <= 1.0):
            raise ValueError("road_plane.recovery_fit_accept_min_inlier_ratio must be in [0, 1].")
        if self.settings.state_causal_smoothing < 1:
            raise ValueError("road_plane.state_causal_smoothing must be >= 1.")

    # ------------------------------------------------------------------
    # Loading and windowing
    # ------------------------------------------------------------------
    def _trajectory_frames(self, resources: ResourceStore) -> List[int]:
        traj_path = resources.path_for(ResourceKind.TRAJECTORY)
        if not traj_path.exists():
            raise RuntimeError("No trajectory file available for road plane estimation.")
        with np.load(traj_path, allow_pickle=True) as data:
            frame_indices = np.asarray(data["frame_indices"]).astype(int).tolist()
        if not frame_indices:
            raise RuntimeError("No trajectory frames available for road plane estimation.")
        return frame_indices

    def _load_frame_bundles(
        self,
        resources: ResourceStore,
        frame_indices: Sequence[int],
    ) -> Dict[int, FrameBundle]:
        bundles: Dict[int, FrameBundle] = {}
        for frame_idx in frame_indices:
            pose = resources.load_pose(frame_idx)
            height_data = resources.load_camera_height(frame_idx)
            meta = dict(height_data.metadata or {})
            missing = sorted(k for k in HEIGHT_METADATA_REQUIRED_FIELDS if k not in meta)
            if missing:
                raise RuntimeError(
                    f"Camera height metadata missing required fields for frame {frame_idx}: {missing}."
                )
            bundles[frame_idx] = FrameBundle(
                frame_idx=frame_idx,
                pose=pose,
                camera_center=np.asarray(pose.camera_to_world[:3, 3], dtype=np.float32),
                camera_height=float(height_data.height_m),
            )
        return bundles

    def _compute_motion_scores(
        self,
        frame_indices: Sequence[int],
        bundles: Mapping[int, FrameBundle],
    ) -> List[float]:
        centers = np.stack([bundles[idx].camera_center for idx in frame_indices], axis=0)
        speeds = np.zeros((len(frame_indices),), dtype=np.float32)
        turns = np.zeros((len(frame_indices),), dtype=np.float32)

        if len(frame_indices) <= 2:
            return [0.0 for _ in frame_indices]

        deltas = centers[1:] - centers[:-1]
        step = np.linalg.norm(deltas, axis=1)
        speeds[1:] = step

        for i in range(1, len(frame_indices) - 1):
            v0 = deltas[i - 1]
            v1 = deltas[i]
            n0 = float(np.linalg.norm(v0))
            n1 = float(np.linalg.norm(v1))
            if n0 < 1e-5 or n1 < 1e-5:
                turns[i] = 0.0
                continue
            cosang = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
            turns[i] = abs(math.acos(cosang))

        speed_norm = speeds / max(float(np.percentile(speeds, 75)), 1e-4)
        turn_norm = turns / max(float(np.percentile(turns, 75)), 1e-4)
        score = self.settings.motion_speed_weight * speed_norm + self.settings.motion_turn_weight * turn_norm
        score = np.clip(score, 0.0, 1.0)
        return score.astype(float).tolist()

    def _window_half_width(self, center_idx: int, motion_scores: Sequence[float]) -> int:
        if not self.settings.adaptive_window_enabled:
            return int(max(0, self.settings.window_half_width))

        min_w = int(max(0, self.settings.window_min_half_width))
        max_w = int(max(min_w, self.settings.window_max_half_width))
        if min_w == max_w:
            return min_w

        score = float(motion_scores[center_idx]) if center_idx < len(motion_scores) else 0.0
        dynamic = max_w - int(round(score * (max_w - min_w)))
        base = int(max(min_w, min(max_w, self.settings.window_half_width)))
        return int(max(min_w, min(max_w, dynamic if self.settings.adaptive_window_enabled else base)))

    def _collect_window_points_from_cache(
        self,
        *,
        frame_indices: Sequence[int],
        center_idx: int,
        half_width: int,
        sampled_points: Mapping[int, np.ndarray],
        sampled_weights: Mapping[int, np.ndarray],
        bundles: Mapping[int, FrameBundle],
        excluded_frames: frozenset[int],
        support_point_counts: Mapping[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, Dict[str, Any]]:
        start = max(0, center_idx - half_width)
        if self.settings.window_causal_only:
            end = center_idx + 1
        else:
            end = min(len(frame_indices), center_idx + half_width + 1)
        window = frame_indices[start:end]

        points: List[np.ndarray] = []
        weights: List[np.ndarray] = []
        centers: List[np.ndarray] = []
        included_frames: List[int] = []
        excluded_catastrophic = 0
        excluded_low_support = 0

        center_frame = frame_indices[center_idx]
        camera_height_m = bundles[center_frame].camera_height
        anchor_camera_center = bundles[center_frame].camera_center

        for frame_idx in window:
            if frame_idx in excluded_frames:
                excluded_catastrophic += 1
                continue
            if (
                self.settings.window_exclude_low_support_frames
                and int(support_point_counts.get(frame_idx, 0))
                < int(self.settings.window_min_support_points_for_inclusion)
                and frame_idx != center_frame
            ):
                excluded_low_support += 1
                continue
            pts = sampled_points[frame_idx]
            wts = sampled_weights[frame_idx]
            if pts.size:
                points.append(pts)
                weights.append(wts)
                included_frames.append(int(frame_idx))
            centers.append(bundles[frame_idx].camera_center)

        if len(included_frames) < int(self.settings.window_min_frames_required):
            pts_center = sampled_points.get(center_frame)
            wts_center = sampled_weights.get(center_frame)
            if pts_center is not None and wts_center is not None and pts_center.size > 0:
                points = [pts_center]
                weights = [wts_center]
                centers = [bundles[center_frame].camera_center]
                included_frames = [int(center_frame)]

        window_meta = {
            "window_frame_count": int(len(window)),
            "window_included_frame_count": int(len(included_frames)),
            "window_excluded_catastrophic_count": int(excluded_catastrophic),
            "window_excluded_low_support_count": int(excluded_low_support),
            "window_is_causal": bool(self.settings.window_causal_only),
            "window_included_frames": [int(v) for v in included_frames],
        }
        if not points:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.stack(centers, axis=0) if centers else np.zeros((0, 3), dtype=np.float32),
                float(camera_height_m),
                np.asarray(anchor_camera_center, dtype=np.float32).reshape(3),
                window_meta,
            )

        return (
            np.concatenate(points, axis=0),
            np.concatenate(weights, axis=0),
            np.stack(centers, axis=0),
            float(camera_height_m),
            np.asarray(anchor_camera_center, dtype=np.float32).reshape(3),
            window_meta,
        )

    def _sample_frame_points_from_frame_resources(
        self,
        *,
        resources: ResourceStore,
        frame_idx: int,
        source_idx: int,
        depth_data: DepthData,
        semantics: SemanticsData,
        pose: PoseSample,
        intrinsics: IntrinsicsData,
        rng: np.random.Generator,
        context: Mapping[str, object] | None,
    ) -> FrameRoadPointSample:
        depth = np.asarray(depth_data.depth, dtype=np.float32)
        if depth.ndim != 2:
            raise RuntimeError(f"Depth for frame {source_idx} must be 2D, got {depth.shape}.")
        ids = semantics.label_ids if semantics.label_ids is not None else semantics.segment_ids
        if ids is None:
            raise RuntimeError(
                f"Semantics for frame {source_idx} is missing both label_ids and segment_ids; "
                "road-plane support sampling cannot continue."
            )
        if ids.shape != depth.shape:
            raise RuntimeError(
                f"Semantics/depth shape mismatch for frame {source_idx}: {ids.shape} vs {depth.shape}."
            )
        label_map = self._label_map(semantics)
        support_ids = self._support_label_ids_from_semantics(
            semantics=semantics,
            label_map=label_map,
            context=context,
        )
        if not support_ids:
            raise RuntimeError(
                f"Frame {source_idx} contains none of configured support labels "
                f"(road={self.settings.road_labels}, sidewalk={self.settings.sidewalk_labels})."
            )
        support_mask = np.isin(ids, np.asarray(support_ids, dtype=np.int32))
        confidence = self._support_confidence(
            semantics,
            support_mask=support_mask,
            resources=resources,
        )

        h, w = depth.shape
        stride = int(max(1, self.settings.support_pixel_stride))
        yy = np.arange(0, h, stride, dtype=np.int32)
        xx = np.arange(0, w, stride, dtype=np.int32)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        ys = grid_y.reshape(-1)
        xs = grid_x.reshape(-1)
        sampled_depth = depth[ys, xs]
        sampled_support = support_mask[ys, xs]
        sampled_conf = confidence[ys, xs]

        valid = (
            sampled_support
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.isfinite(sampled_conf)
            & (sampled_conf >= float(self.settings.support_min_confidence))
        )
        support_pixels = int(np.count_nonzero(sampled_support))
        valid_pixels = int(np.count_nonzero(valid))
        if valid_pixels == 0:
            return FrameRoadPointSample(
                points=np.zeros((0, 3), dtype=np.float32),
                weights=np.zeros((0,), dtype=np.float32),
                diagnostics={
                    "source_frame_index": float(source_idx),
                    "reused_source": float(1 if int(frame_idx) != int(source_idx) else 0),
                    "support_pixels": float(support_pixels),
                    "valid_pixels": 0.0,
                    "roi_points": 0.0,
                    "adaptive_forward_min_used_m": float(self.settings.forward_min_m),
                    "adaptive_forward_iters": 0.0,
                    "adaptive_forward_fallback_used": 0.0,
                    "metric_scale": float(
                        1.0 if bool((depth_data.metadata or {}).get("metric_scale", False)) else 0.0
                    ),
                    "layering_ratio": 1.0,
                    "layering_depth_spread_p90_m": 1.0e6,
                },
            )

        uv = np.stack([xs[valid], ys[valid]], axis=1).astype(np.float32)
        valid_depth = sampled_depth[valid].astype(np.float32)
        valid_conf = sampled_conf[valid].astype(np.float32)
        cam_points = backproject_uv_depth_to_camera(
            uv,
            valid_depth,
            intrinsics.matrix,
            camera_convention="blender",
        )
        forward = -cam_points[:, 2]
        lateral_abs = np.abs(cam_points[:, 0])
        vertical_abs = np.abs(cam_points[:, 1])
        forward_min_used = float(self.settings.forward_min_m)
        adaptive_iters = 0
        fallback_used = False
        roi = np.zeros_like(forward, dtype=bool)
        if self.settings.support_adaptive_forward_min_enabled:
            floor = float(self.settings.support_forward_min_floor_m)
            step = float(self.settings.support_forward_min_step_m)
            max_iters = int(self.settings.support_adaptive_forward_max_iters)
            target = int(max(1, self.settings.support_target_min_points))
            for it in range(max_iters + 1):
                roi = (
                    (forward >= forward_min_used)
                    & (forward <= float(self.settings.forward_max_m))
                    & (lateral_abs <= float(self.settings.lateral_max_m))
                    & (vertical_abs <= float(self.settings.vertical_max_m))
                )
                if int(np.count_nonzero(roi)) >= target or forward_min_used <= floor + 1e-6:
                    adaptive_iters = it
                    break
                forward_min_used = max(floor, forward_min_used - step)
                fallback_used = True
                adaptive_iters = it + 1
        else:
            roi = (
                (forward >= float(self.settings.forward_min_m))
                & (forward <= float(self.settings.forward_max_m))
                & (lateral_abs <= float(self.settings.lateral_max_m))
                & (vertical_abs <= float(self.settings.vertical_max_m))
            )
        if int(np.count_nonzero(roi)) == 0:
            return FrameRoadPointSample(
                points=np.zeros((0, 3), dtype=np.float32),
                weights=np.zeros((0,), dtype=np.float32),
                diagnostics={
                    "source_frame_index": float(source_idx),
                    "reused_source": float(1 if int(frame_idx) != int(source_idx) else 0),
                    "support_pixels": float(support_pixels),
                    "valid_pixels": float(valid_pixels),
                    "roi_points": 0.0,
                    "adaptive_forward_min_used_m": float(forward_min_used),
                    "adaptive_forward_iters": float(adaptive_iters),
                    "adaptive_forward_fallback_used": float(1.0 if fallback_used else 0.0),
                    "metric_scale": float(
                        1.0 if bool((depth_data.metadata or {}).get("metric_scale", False)) else 0.0
                    ),
                    "layering_ratio": 1.0,
                    "layering_depth_spread_p90_m": 1.0e6,
                },
            )
        uv_roi = uv[roi]
        cam_roi = cam_points[roi]
        depth_roi = valid_depth[roi]
        conf_roi = valid_conf[roi]
        points_world = camera_to_world(cam_roi, pose.camera_to_world).astype(np.float32)
        layering_ratio, layering_spread_p90 = self._layering_metrics(
            uv=uv_roi,
            depth=depth_roi,
            image_width=w,
        )

        max_points = max(1, int(self.settings.points_per_frame))
        sampled_idx = self._weighted_sample_indices(
            weights=np.asarray(conf_roi, dtype=np.float32),
            count=points_world.shape[0],
            max_points=max_points,
            rng=rng,
        )
        sampled_points = points_world[sampled_idx].astype(np.float32)
        sampled_weights = conf_roi[sampled_idx].astype(np.float32)
        return FrameRoadPointSample(
            points=sampled_points,
            weights=sampled_weights,
            diagnostics={
                "source_frame_index": float(source_idx),
                "reused_source": float(1 if int(frame_idx) != int(source_idx) else 0),
                "support_pixels": float(support_pixels),
                "valid_pixels": float(valid_pixels),
                "roi_points": float(points_world.shape[0]),
                "adaptive_forward_min_used_m": float(forward_min_used),
                "adaptive_forward_iters": float(adaptive_iters),
                "adaptive_forward_fallback_used": float(1.0 if fallback_used else 0.0),
                "metric_scale": float(
                    1.0 if bool((depth_data.metadata or {}).get("metric_scale", False)) else 0.0
                ),
                "forward_p95_m": float(np.percentile(forward[roi], 95)),
                "lateral_p95_m": float(np.percentile(lateral_abs[roi], 95)),
                "vertical_p95_m": float(np.percentile(vertical_abs[roi], 95)),
                "layering_ratio": float(layering_ratio),
                "layering_depth_spread_p90_m": float(layering_spread_p90),
            },
        )

    def _weighted_sample_indices(
        self,
        *,
        weights: np.ndarray,
        count: int,
        max_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count <= max_points:
            return np.arange(count, dtype=np.int32)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != count:
            raise RuntimeError(f"Sampling weight shape mismatch: {w.shape[0]} != {count}.")
        power = max(0.0, float(self.settings.sampling_weight_power))
        if power != 1.0:
            w = np.power(np.maximum(w, 1e-12), power)
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            return rng.choice(count, size=max_points, replace=False).astype(np.int32)
        probs = w / total
        return rng.choice(count, size=max_points, replace=False, p=probs).astype(np.int32)

    @staticmethod
    def _layering_metrics(
        *,
        uv: np.ndarray,
        depth: np.ndarray,
        image_width: int,
    ) -> tuple[float, float]:
        if uv.size == 0:
            return 1.0, float("inf")
        px = np.rint(uv[:, 0]).astype(np.int32)
        py = np.rint(uv[:, 1]).astype(np.int32)
        key = py.astype(np.int64) * np.int64(max(1, image_width)) + px.astype(np.int64)
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        duplicate_mask = counts[inverse] > 1
        layering_ratio = float(np.mean(duplicate_mask)) if duplicate_mask.size else 0.0
        if not np.any(counts > 1):
            return layering_ratio, 0.0
        spreads: List[float] = []
        for idx, cnt in enumerate(counts):
            if cnt <= 1:
                continue
            vals = depth[inverse == idx]
            if vals.size <= 1:
                continue
            spreads.append(float(np.max(vals) - np.min(vals)))
        if not spreads:
            return layering_ratio, 0.0
        return layering_ratio, float(np.percentile(np.asarray(spreads, dtype=np.float32), 90.0))

    # ------------------------------------------------------------------
    # Label/confidence helpers.
    # ------------------------------------------------------------------
    def _load_semantics_aux(
        self,
        resources: ResourceStore,
        semantics: SemanticsData,
    ) -> SemanticsAuxData | None:
        try:
            return resources.load_semantics_aux(int(semantics.frame_index))
        except Exception:
            return None

    def _road_confidence(
        self,
        semantics: SemanticsData,
        *,
        resources: ResourceStore,
    ) -> np.ndarray | None:
        label_ids = semantics.label_ids
        segment_ids = semantics.segment_ids
        if label_ids is None and segment_ids is None:
            return None

        label_map = self._label_map(semantics)
        road_ids = resolve_role_label_ids(
            label_map,
            "road",
            metadata=semantics.metadata,
            tool=getattr(self, "_semantics_tool", None),
            defaults=getattr(self, "_semantic_role_defaults", None),
            required=True,
            source_name=f"road-plane frame {semantics.frame_index}",
        )
        if not road_ids:
            msg = f"No road labels found in semantics data for frame {semantics.frame_index}."
            msg += f"\nAvailable labels: {list(label_map.values())}."
            msg += "\nWe were looking for canonical semantic role 'road'."
            raise ValueError(msg)
        road_id = int(road_ids[0])
        ids = label_ids if label_ids is not None else segment_ids
        if ids is None:
            return None
        road_binary = np.zeros_like(ids, dtype=np.float32)
        road_binary[ids == road_id] = 1.0
        return self._support_confidence(
            semantics,
            support_mask=road_binary > 0.0,
            resources=resources,
        )

    def _support_confidence(
        self,
        semantics: SemanticsData,
        *,
        support_mask: np.ndarray,
        resources: ResourceStore,
    ) -> np.ndarray:
        if support_mask.ndim != 2:
            raise RuntimeError(f"support_mask must be 2D, got {support_mask.shape}.")
        support_binary = support_mask.astype(np.float32, copy=False)

        aux = self._load_semantics_aux(resources, semantics)
        if aux is not None:
            if aux.road_confidence is not None and aux.road_confidence.shape == support_binary.shape:
                return np.asarray(aux.road_confidence, dtype=np.float32)
            if aux.class_probabilities is not None and aux.class_probabilities.shape[1:] == support_binary.shape:
                conf = np.max(np.asarray(aux.class_probabilities, dtype=np.float32), axis=0)
                return conf * support_binary
            if aux.confidence is not None and aux.confidence.shape == support_binary.shape:
                return np.asarray(aux.confidence, dtype=np.float32) * support_binary
        return support_binary

    @staticmethod
    def _label_map(semantics: SemanticsData) -> Dict[int, str]:
        result: Dict[int, str] = {}
        for seg in semantics.segments:
            if seg.label_id is None:
                result[int(seg.segment_id)] = str(seg.label).lower()
            else:
                result[int(seg.label_id)] = str(seg.label).lower()
        return result

    def _support_label_ids_from_semantics(
        self,
        *,
        semantics: SemanticsData,
        label_map: Mapping[int, str],
        context: Mapping[str, object] | None = None,
    ) -> List[int]:
        del context
        road = set(
            resolve_semantic_role_labels(
                "road",
                metadata=semantics.metadata,
                tool=getattr(self, "_semantics_tool", None),
                defaults=getattr(self, "_semantic_role_defaults", None),
                required=True,
                source_name=f"road-plane frame {semantics.frame_index}",
            )
        )
        sidewalk = set(
            resolve_semantic_role_labels(
                "sidewalk",
                metadata=semantics.metadata,
                tool=getattr(self, "_semantics_tool", None),
                defaults=getattr(self, "_semantic_role_defaults", None),
            )
        )
        wanted = set(road)
        if self.settings.include_sidewalk_in_support:
            wanted.update(sidewalk)
        if not wanted:
            return []
        result: List[int] = []
        for label_id, label_name in label_map.items():
            if str(label_name).strip().lower() in wanted:
                result.append(int(label_id))
        return sorted(set(result))

    # ------------------------------------------------------------------
    # Global plane fitting and temporal filtering
    # ------------------------------------------------------------------
    def _fit_global_plane(
        self,
        *,
        points: np.ndarray,
        weights: np.ndarray,
        centers: np.ndarray,
        camera_height_m: float,
        anchor_camera_center: np.ndarray,
        prev_plane: tuple[np.ndarray, float] | None,
        huber_delta_override: float | None = None,
        enforce_height_anchor: bool = True,
    ) -> PlaneResult:
        pts = points
        wts = weights
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for iteration in range(max(1, self.settings.irls_iters)):
            normal, plane_offset_d, cov_diag = solve_plane_weighted(
                points=pts,
                weights=wts,
                anchor_camera_center=anchor_camera_center,
                camera_height_m=camera_height_m,
                lambda_up=self.settings.lambda_up,
                lambda_temp=self.settings.lambda_temp,
                prev_plane=prev_plane,
                up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                enforce_height_anchor=bool(enforce_height_anchor),
            )
            residuals = np.abs(pts @ normal + plane_offset_d)
            if iteration == 0 and self.settings.trim_ratio > 0:
                cutoff = float(np.quantile(residuals, 1.0 - self.settings.trim_ratio))
                keep = residuals <= cutoff
                if int(np.count_nonzero(keep)) >= 3:
                    pts = pts[keep]
                    wts = wts[keep]
                    residuals = residuals[keep]
            huber_delta = float(
                huber_delta_override if huber_delta_override is not None else self.settings.huber_delta
            )
            huber = huber_weights(residuals, huber_delta)
            wts = wts * huber

        inlier_threshold = max(
            float((huber_delta_override if huber_delta_override is not None else self.settings.huber_delta) * 2.0),
            0.1,
        )
        residuals, quality = compute_plane_quality(
            pts,
            normal,
            plane_offset_d,
            inlier_threshold_m=inlier_threshold,
        )

        return PlaneResult(
            normal=normal.astype(np.float32),
            offset=float(plane_offset_d),
            residuals=residuals.astype(np.float32),
            cov_diag=cov_diag.astype(np.float32),
            quality=quality.to_metadata(),
        )

    def _needs_multi_hypothesis(self, result: PlaneResult) -> bool:
        return (
            float(result.quality.get("inlier_ratio", 1.0)) < self.settings.multi_hypothesis_min_inlier_ratio
            or float(result.quality.get("residual_p90", 0.0)) > self.settings.multi_hypothesis_p90_trigger_m
        )

    def _fit_multi_hypothesis(
        self,
        *,
        points: np.ndarray,
        weights: np.ndarray,
        centers: np.ndarray,
        camera_height_m: float,
        anchor_camera_center: np.ndarray,
        prev_plane: tuple[np.ndarray, float] | None,
        base: PlaneResult,
        enforce_height_anchor: bool,
    ) -> PlaneResult | None:
        signed = points @ base.normal + base.offset
        if points.shape[0] < 60:
            return None

        groups = [signed >= np.median(signed), signed < np.median(signed)]
        candidates: List[PlaneResult] = [base]
        for mask in groups:
            if int(np.count_nonzero(mask)) < 30:
                continue
            try:
                candidate = self._fit_global_plane(
                    points=points[mask],
                    weights=weights[mask],
                    centers=centers,
                    camera_height_m=camera_height_m,
                    anchor_camera_center=anchor_camera_center,
                    prev_plane=prev_plane,
                    enforce_height_anchor=enforce_height_anchor,
                )
                candidate.hypothesis = "multi"
                candidates.append(candidate)
            except Exception:
                continue

        if len(candidates) == 1:
            LOG.info("[RoadPlane] Multi-hypothesis fallback had no valid alternative candidates.")
            return None

        def _score(item: PlaneResult) -> float:
            q = item.quality
            anchor_err = abs(float(np.mean(centers @ item.normal + item.offset) - camera_height_m))
            return (
                float(q.get("inlier_ratio", 0.0))
                - float(q.get("residual_p90", 1.0))
                - 0.5 * float(q.get("residual_median", 1.0))
                - 2.0 * anchor_err
            )

        best = max(candidates, key=_score)
        LOG.info("[RoadPlane] Multi-hypothesis candidates=%d selected=%s", len(candidates), best.hypothesis)
        return best

    def _apply_temporal_model(
        self,
        *,
        frame_idx: int,
        result: PlaneResult,
        bundle: FrameBundle,
        prev_plane: tuple[np.ndarray, float] | None,
        state_filter: SimpleRoadStateFilter,
        measurement_allowed: bool,
        gate_reason: str,
        quality_score: float,
        state_roll_history: List[float],
        state_pitch_history: List[float],
    ) -> tuple[np.ndarray, float, Dict[str, Any]]:
        roll, pitch = self._roll_pitch_from_normal(result.normal)
        plane_height_at_camera_m = float(result.normal @ bundle.camera_center + result.offset)
        residual_p90 = float(result.quality.get("residual_p90", 0.05))
        residual_scale = max(1.0, residual_p90 / 0.20)
        confidence_scale = 1.0 / max(0.05, float(quality_score))
        quality_scale = max(1.0, residual_scale * confidence_scale)
        measurement = np.array([roll, pitch, plane_height_at_camera_m], dtype=np.float32)

        state_filter.ensure_initialized(measurement)
        prev_state = state_filter.x.copy() if state_filter.x is not None else measurement.copy()
        state_filter.predict()

        update_used = bool(measurement_allowed)
        if measurement_allowed:
            state, state_cov = state_filter.update(measurement, quality_scale=quality_scale)
            if not state_filter.last_update_accepted:
                update_used = False
        else:
            state = state_filter.x.copy()
            state_cov = np.diag(state_filter.P).copy()

        max_step_rad = math.radians(max(0.0, self.settings.max_roll_pitch_step_deg))
        max_height_step = float(max(0.0, self.settings.max_height_step_m))
        state_clamped = state.copy()
        jump_clamp_applied = False
        dr = float(state_clamped[0] - prev_state[0])
        dp = float(state_clamped[1] - prev_state[1])
        dh = float(state_clamped[2] - prev_state[2])
        if abs(dr) > max_step_rad:
            state_clamped[0] = prev_state[0] + np.sign(dr) * max_step_rad
            jump_clamp_applied = True
        if abs(dp) > max_step_rad:
            state_clamped[1] = prev_state[1] + np.sign(dp) * max_step_rad
            jump_clamp_applied = True
        if abs(dh) > max_height_step:
            state_clamped[2] = prev_state[2] + np.sign(dh) * max_height_step
            jump_clamp_applied = True
        if jump_clamp_applied:
            state_filter.x = state_clamped.astype(np.float32)
            state = state_filter.x.copy()

        state_roll_history.append(float(state[0]))
        state_pitch_history.append(float(state[1]))
        hist = max(1, int(self.settings.state_causal_smoothing))
        if hist > 1:
            state_roll_smoothed = float(np.median(np.asarray(state_roll_history[-hist:], dtype=np.float32)))
            state_pitch_smoothed = float(np.median(np.asarray(state_pitch_history[-hist:], dtype=np.float32)))
        else:
            state_roll_smoothed = float(state[0])
            state_pitch_smoothed = float(state[1])

        n = self._normal_from_roll_pitch(state_roll_smoothed, state_pitch_smoothed)
        d = float(state[2] - np.dot(n, bundle.camera_center))

        return n, d, {
            "temporal_mode": "state_filter",
            "state_roll": float(state[0]),
            "state_pitch": float(state[1]),
            "state_roll_smoothed": float(state_roll_smoothed),
            "state_pitch_smoothed": float(state_pitch_smoothed),
            "state_plane_height_at_camera_m": float(state[2]),
            "state_cov_roll": float(state_cov[0]),
            "state_cov_pitch": float(state_cov[1]),
            "state_cov_height": float(state_cov[2]),
            "state_update_accepted": bool(state_filter.last_update_accepted),
            "state_update_used": bool(update_used and state_filter.last_update_accepted),
            "predict_only": bool(not (update_used and state_filter.last_update_accepted)),
            "gate_reason": str(gate_reason),
            "measurement_quality_score": float(quality_score),
            "measurement_quality_scale": float(quality_scale),
            "jump_clamp_applied": bool(jump_clamp_applied),
            "max_roll_pitch_step_deg": float(self.settings.max_roll_pitch_step_deg),
            "max_height_step_m": float(self.settings.max_height_step_m),
        }

    def _evaluate_pre_fit_gates(
        self,
        *,
        points: np.ndarray,
        center_bundle: FrameBundle,
    ) -> tuple[bool, Dict[str, Any], float]:
        count = int(points.shape[0])
        if count < int(self.settings.min_window_points):
            return (
                False,
                {
                    "gate_reason": "min_window_points",
                    "window_point_count": count,
                    "min_window_points": int(self.settings.min_window_points),
                },
                0.0,
            )

        rel = points - center_bundle.camera_center[None, :]
        cam_x = np.asarray(center_bundle.pose.camera_to_world[:3, 0], dtype=np.float32)
        cam_forward = -np.asarray(center_bundle.pose.camera_to_world[:3, 2], dtype=np.float32)
        lateral = rel @ cam_x
        forward = rel @ cam_forward

        lat_span = float(np.max(lateral) - np.min(lateral))
        fwd_span = float(np.max(forward) - np.min(forward))
        left = int(np.count_nonzero(lateral < 0.0))
        right = int(np.count_nonzero(lateral >= 0.0))
        balance = float(min(left, right)) / float(max(left, right, 1))

        centered = points - np.mean(points, axis=0, keepdims=True)
        _, svals, _ = np.linalg.svd(centered, full_matrices=False)
        cond = float(svals[0] / max(float(svals[-1]), 1e-8))

        # Soft quality scoring (0..1) for non-catastrophic geometry issues.
        lat_quality = float(np.clip(lat_span / max(float(self.settings.min_lateral_span_m), 1e-4), 0.0, 1.0))
        fwd_quality = float(np.clip(fwd_span / max(float(self.settings.min_forward_span_m), 1e-4), 0.0, 1.0))
        bal_quality = float(
            np.clip(
                balance / max(float(self.settings.min_left_right_balance_ratio), 1e-4),
                0.0,
                1.0,
            )
        )
        cond_quality = float(
            np.clip(
                float(self.settings.max_condition_number) / max(cond, 1e-6),
                0.0,
                1.0,
            )
        )
        quality_score = float(np.clip(0.35 * lat_quality + 0.35 * fwd_quality + 0.20 * bal_quality + 0.10 * cond_quality, 0.0, 1.0))
        soft_reasons: List[str] = []
        if lat_span < float(self.settings.min_lateral_span_m):
            soft_reasons.append("min_lateral_span_m")
        if fwd_span < float(self.settings.min_forward_span_m):
            soft_reasons.append("min_forward_span_m")
        if balance < float(self.settings.min_left_right_balance_ratio):
            soft_reasons.append("left_right_balance")
        if cond > float(self.settings.max_condition_number):
            soft_reasons.append("condition_number")

        return True, {
            "gate_reason": "ok",
            "soft_gate_reasons": tuple(soft_reasons),
            "window_point_count": count,
            "lateral_span_m": lat_span,
            "forward_span_m": fwd_span,
            "left_right_balance_ratio": balance,
            "condition_number": cond,
            "pre_quality_score": quality_score,
        }, quality_score

    def _evaluate_post_fit_gates(self, result: PlaneResult) -> tuple[bool, str, float]:
        residual_p90 = float(result.quality.get("residual_p90", 0.0))
        inlier = float(result.quality.get("inlier_ratio", 0.0))
        if residual_p90 > float(self.settings.catastrophic_residual_p90_m):
            return False, "catastrophic_residual_p90", 0.0
        if inlier < float(self.settings.catastrophic_min_inlier_ratio):
            return False, "catastrophic_min_inlier_ratio", 0.0

        # Soft quality score; low score still allows update with higher measurement noise.
        residual_good = max(float(self.settings.gating_max_residual_p90_m), 1e-4)
        inlier_good = max(float(self.settings.gating_min_inlier_ratio), 1e-4)
        residual_score = float(np.clip(residual_good / max(residual_p90, residual_good), 0.0, 1.0))
        inlier_score = float(np.clip(inlier / inlier_good, 0.0, 1.0))
        quality_score = float(np.clip(0.55 * residual_score + 0.45 * inlier_score, 0.0, 1.0))
        gate_reason = "ok"
        if residual_p90 > float(self.settings.gating_max_residual_p90_m):
            gate_reason = "soft_max_residual_p90"
        if inlier < float(self.settings.gating_min_inlier_ratio):
            gate_reason = "soft_min_inlier_ratio"
        return True, gate_reason, quality_score

    def _evaluate_saved_point_gate(
        self,
        *,
        residual_p90: float,
        inlier_ratio: float,
        frame_order_index: int,
    ) -> tuple[bool, bool]:
        strict_ok = (
            residual_p90 <= float(self.settings.saved_point_max_residual_p90_m)
            and inlier_ratio >= float(self.settings.saved_point_min_inlier_ratio)
        )
        if strict_ok:
            return True, False
        grace_frames = int(self.settings.saved_point_startup_grace_frames)
        if frame_order_index >= grace_frames:
            return False, False
        startup_ok = (
            residual_p90 <= float(self.settings.saved_point_startup_max_residual_p90_m)
            and inlier_ratio >= float(self.settings.saved_point_startup_min_inlier_ratio)
        )
        return bool(startup_ok), bool(startup_ok)

    def _evaluate_support_quality_gate(
        self,
        *,
        support_points: int,
        layering_ratio: float,
        depth_spread_p90_m: float,
    ) -> bool:
        return bool(
            support_points >= int(self.settings.support_min_points)
            and layering_ratio <= float(self.settings.support_max_layering_ratio)
            and depth_spread_p90_m <= float(self.settings.support_max_depth_spread_p90_m)
        )

    @staticmethod
    def _gate_reason_priority(reason: str) -> int:
        if reason.startswith("support_quality"):
            return 100
        if reason.startswith("catastrophic"):
            return 90
        if reason.startswith("saved_point_quality"):
            return 80
        if reason.startswith("recovery_fit"):
            return 85
        if reason.startswith("pre_fit"):
            return 70
        if reason.startswith("soft_"):
            return 50
        if reason == "ok":
            return 0
        return 40

    def _resolve_gate_reason(
        self,
        *,
        current: str,
        candidate: str,
    ) -> str:
        if self._gate_reason_priority(candidate) > self._gate_reason_priority(current):
            return candidate
        return current

    def _try_recovery_fit(
        self,
        *,
        frame_idx: int,
        center_bundle: FrameBundle,
        support_points_world: np.ndarray,
        support_weights: np.ndarray,
        pre_saved_quality: Any,
        frame_order_index: int,
        prev_plane: tuple[np.ndarray, float] | None,
        enforce_height_anchor: bool,
    ) -> tuple[bool, PlaneResult | None, Dict[str, float]]:
        if not self.settings.recovery_fit_enabled:
            return False, None, {"recovery_fit_used": 0.0, "recovery_fit_accepted": 0.0}
        if support_points_world.shape[0] < int(self.settings.recovery_fit_min_points):
            return False, None, {
                "recovery_fit_used": 0.0,
                "recovery_fit_accepted": 0.0,
                "recovery_fit_reason": 1.0,  # insufficient points
            }
        recovery = self._fit_global_plane(
            points=support_points_world,
            weights=support_weights,
            centers=np.asarray(center_bundle.camera_center, dtype=np.float32).reshape(1, 3),
            camera_height_m=float(center_bundle.camera_height),
            anchor_camera_center=np.asarray(center_bundle.camera_center, dtype=np.float32).reshape(3),
            prev_plane=prev_plane,
            huber_delta_override=float(self.settings.huber_delta * self.settings.recovery_fit_huber_delta_scale),
            enforce_height_anchor=bool(enforce_height_anchor),
        )
        _, post_reason, _post_q = self._evaluate_post_fit_gates(recovery)
        _, recovery_saved_quality = compute_plane_quality(
            support_points_world,
            recovery.normal,
            recovery.offset,
            inlier_threshold_m=max(self.settings.huber_delta * 2.0, 0.1),
        )
        saved_gate_ok, _startup = self._evaluate_saved_point_gate(
            residual_p90=float(recovery_saved_quality.residual_p90),
            inlier_ratio=float(recovery_saved_quality.inlier_ratio),
            frame_order_index=frame_order_index,
        )
        rec_res_p90 = float(recovery.quality.get("residual_p90", 0.0))
        rec_inlier = float(recovery.quality.get("inlier_ratio", 0.0))
        accepted = bool(
            post_reason != "catastrophic_residual_p90"
            and post_reason != "catastrophic_min_inlier_ratio"
            and saved_gate_ok
            and rec_res_p90 <= float(self.settings.recovery_fit_accept_max_residual_p90_m)
            and rec_inlier >= float(self.settings.recovery_fit_accept_min_inlier_ratio)
        )
        return accepted, recovery, {
            "recovery_fit_used": 1.0,
            "recovery_fit_accepted": 1.0 if accepted else 0.0,
            "recovery_fit_residual_p90": rec_res_p90,
            "recovery_fit_inlier_ratio": rec_inlier,
            "recovery_fit_saved_residual_p90": float(recovery_saved_quality.residual_p90),
            "recovery_fit_saved_inlier_ratio": float(recovery_saved_quality.inlier_ratio),
        }

    @staticmethod
    def _roll_pitch_from_normal(normal: np.ndarray) -> tuple[float, float]:
        n = np.asarray(normal, dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            return 0.0, 0.0
        n = n / n_norm
        if n[2] < 0:
            n = -n
        pitch = math.atan2(-float(n[0]), max(1e-6, float(n[2])))
        roll = math.atan2(float(n[1]), max(1e-6, float(n[2])))
        return roll, pitch

    @staticmethod
    def _normal_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
        n = np.array(
            [
                -math.sin(pitch),
                math.sin(roll) * math.cos(pitch),
                math.cos(roll) * math.cos(pitch),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return n / norm

    def _solve_plane(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        anchor_camera_center: np.ndarray,
        camera_height_m: float,
        prev_plane: tuple[np.ndarray, float] | None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        return solve_plane_weighted(
            points=points,
            weights=weights,
            anchor_camera_center=anchor_camera_center,
            camera_height_m=camera_height_m,
            lambda_up=self.settings.lambda_up,
            lambda_temp=self.settings.lambda_temp,
            prev_plane=prev_plane,
            up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    @staticmethod
    def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
        return huber_weights(residuals, delta)

    # ------------------------------------------------------------------
    # Optional road-aligned metric grid
    # ------------------------------------------------------------------
    def _update_metric_grid(
        self,
        *,
        frame_idx: int,
        bundle: FrameBundle,
        points: np.ndarray,
        normal: np.ndarray,
        offset: float,
        grid_cells: MutableMapping[tuple[int, int], Dict[str, Any]],
    ) -> None:
        if points.shape[0] == 0:
            return

        pts = points
        if pts.shape[0] > self.settings.metric_grid_max_points_per_frame:
            step = max(1, pts.shape[0] // self.settings.metric_grid_max_points_per_frame)
            pts = pts[::step]

        cam_pos = bundle.camera_center
        cam_forward = -np.asarray(bundle.pose.camera_to_world[:3, 2], dtype=np.float32)
        u = cam_forward - float(np.dot(cam_forward, normal)) * normal
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-6:
            u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            u = u / u_norm
        v = np.cross(normal, u)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-6:
            v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            v = v / v_norm

        origin = cam_pos - float(np.dot(cam_pos, normal) + offset) * normal
        rel = pts - origin[None, :]
        uu = rel @ u
        vv = rel @ v
        signed = pts @ normal + offset

        cell = float(max(self.settings.metric_grid_cell_size_m, 1e-3))
        i_idx = np.floor(uu / cell).astype(np.int32)
        j_idx = np.floor(vv / cell).astype(np.int32)

        for ii, jj, s in zip(i_idx, j_idx, signed):
            key = (int(ii), int(jj))
            cell_data = grid_cells.get(key)
            if cell_data is None:
                grid_cells[key] = {
                    "sum_height": float(s),
                    "count": 1,
                    "normal_sum": normal.astype(np.float32).tolist(),
                    "last_frame": int(frame_idx),
                }
                continue
            cell_data["sum_height"] += float(s)
            cell_data["count"] += 1
            nsum = np.asarray(cell_data["normal_sum"], dtype=np.float32) + normal
            cell_data["normal_sum"] = nsum.astype(float).tolist()
            cell_data["last_frame"] = int(frame_idx)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    def _write_debug(
        self,
        resources: ResourceStore,
        frame_idx: int,
        points: np.ndarray,
        normal: np.ndarray,
        offset: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        vis_dir = resources.visualizations_dir("road_plane")
        vis_dir.mkdir(parents=True, exist_ok=True)

        frame_dir = vis_dir / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        try:
            frame = resources.load_frame(frame_idx)
            intrinsics = resources.load_intrinsics()
            pose = resources.load_pose(frame_idx)
            semantics = resources.load_semantics2d(frame_idx)
        except Exception as exc:
            LOG.warning("[RoadPlane] Failed to load resources for frame %d: %s", frame_idx, exc)
            return None

        if frame.image is None:
            return None

        residuals_img = self._create_residuals_visualization(points, normal, offset, str(frame_dir), frame_idx)
        overlay_img = self._create_road_plane_overlay(
            resources,
            semantics,
            frame.image,
            intrinsics,
            pose,
            normal,
            offset,
            str(frame_dir),
            frame_idx,
        )
        if residuals_img is None or overlay_img is None:
            return None
        return residuals_img, overlay_img

    def _create_residuals_visualization(
        self,
        points: np.ndarray,
        normal: np.ndarray,
        offset: float,
        vis_dir: str,
        frame_idx: int,
    ) -> np.ndarray | None:
        if points.size == 0:
            return None
        residuals = points @ normal + offset
        residuals_clamped = np.clip(
            residuals,
            -self.settings.viz_residual_clamp,
            self.settings.viz_residual_clamp,
        )
        return write_road_plane_residuals_image(
            Path(vis_dir) / "plane_residuals.png",
            points_xy=points[:, :2],
            residuals_clamped=residuals_clamped,
            clamp_abs=float(self.settings.viz_residual_clamp),
            frame_idx=int(frame_idx),
        )

    def _create_road_plane_overlay(
        self,
        resources: ResourceStore,
        semantics: SemanticsData,
        frame: np.ndarray,
        intrinsics: IntrinsicsData,
        pose: PoseSample,
        normal: np.ndarray,
        offset: float,
        vis_dir: str,
        frame_idx: int,
    ) -> np.ndarray | None:
        image = np.asarray(frame).copy()
        h, w = image.shape[:2]
        c2w = np.asarray(pose.camera_to_world, dtype=np.float32)
        w2c = np.asarray(pose.world_to_camera, dtype=np.float32) if pose.world_to_camera is not None else np.linalg.inv(c2w)
        n = normal.astype(np.float32)
        d = float(offset)

        cam_pos = c2w[:3, 3]
        proj = cam_pos - (np.dot(n, cam_pos) + d) * n
        basis_u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(basis_u, n))) > 0.9:
            basis_u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        basis_u = basis_u - np.dot(basis_u, n) * n
        basis_u /= max(1e-6, float(np.linalg.norm(basis_u)))
        basis_v = np.cross(n, basis_u)
        basis_v /= max(1e-6, float(np.linalg.norm(basis_v)))

        extent = float(self.settings.overlay_extent_m)
        spacing = float(self.settings.viz_point_stride_m)
        coords = np.arange(-extent, extent + spacing * 0.5, spacing, dtype=np.float32)
        uu, vv = np.meshgrid(coords, coords, indexing="ij")
        grid = proj[None, None, :] + uu[..., None] * basis_u + vv[..., None] * basis_v
        grid = grid.reshape(-1, 3)

        if grid.shape[0] > self.settings.overlay_max_points:
            step = max(1, grid.shape[0] // self.settings.overlay_max_points)
            grid = grid[::step]

        uv, valid = project_world_to_image(
            grid,
            intrinsics.matrix,
            world_to_camera_matrix=w2c,
            camera_convention="blender",
            image_shape=(h, w),
        )
        if not np.any(valid):
            return None
        uv = uv[valid]
        grid = grid[valid]
        u = uv[:, 0].astype(np.int32)
        v = uv[:, 1].astype(np.int32)
        if u.size == 0:
            return None

        road_conf = self._road_confidence(semantics, resources=resources)
        if road_conf is None:
            return None
        mask_inside = road_conf[v, u] > self.settings.viz_confidence_threshold
        u = u[mask_inside]
        v = v[mask_inside]
        grid = grid[mask_inside]
        if u.size == 0:
            return None

        residuals = (grid @ n + d).astype(np.float32)
        colors = _residual_colors(residuals)

        radius = 3
        if image.dtype != np.uint8:
            img_min = float(np.nanmin(image))
            img_max = float(np.nanmax(image))
            if not np.isfinite(img_min) or not np.isfinite(img_max) or abs(img_max - img_min) < 1e-6:
                image = np.zeros_like(image, dtype=np.uint8)
            else:
                scale = 255.0 / (img_max - img_min)
                image = np.clip((image - img_min) * scale, 0, 255).astype(np.uint8)

        for x, y, color in zip(u, v, colors):
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.circle(image, (int(x), int(y)), radius, bgr, thickness=-1, lineType=cv2.LINE_AA)

        return write_road_plane_overlay_image(Path(vis_dir) / "road_overlay.png", image)

    def _generate_videos(
        self,
        resources: ResourceStore,
        residuals_frames: List[np.ndarray],
        overlay_frames: List[np.ndarray],
        *,
        fps: float,
    ) -> None:
        vis_dir = resources.visualizations_dir("road_plane")
        vis_dir.mkdir(parents=True, exist_ok=True)

        if residuals_frames:
            write_video(
                residuals_frames,
                vis_dir / "plane_residuals.mp4",
                fps,
                self.settings.viz_video_codec,
            )
            LOG.info("[RoadPlane] Generated plane_residuals.mp4 with %d frames", len(residuals_frames))

        if overlay_frames:
            write_video(
                overlay_frames,
                vis_dir / "road_overlay.mp4",
                fps,
                self.settings.viz_video_codec,
            )
            LOG.info("[RoadPlane] Generated road_overlay.mp4 with %d frames", len(overlay_frames))

    # ------------------------------------------------------------------
    # Output artifacts for upgraded features
    # ------------------------------------------------------------------
    def _write_upgrade_artifacts(
        self,
        *,
        resources: ResourceStore,
        frame_indices: Sequence[int],
        global_history: Sequence[Mapping[str, Any]],
        grid_cells: Mapping[tuple[int, int], Mapping[str, Any]],
        validation_policy: Mapping[str, Any],
        effective_max_reused_frames: int,
        effective_max_saved_point_skips: int,
    ) -> None:
        provider_dir = resources.provider_dir("road_plane")
        vis_dir = resources.visualizations_dir("road_plane")

        global_path = provider_dir / "global_state_history.json"
        global_payload = {
            "frame_count": len(frame_indices),
            "temporal_mode": self.settings.temporal_mode,
            "entries": list(global_history),
        }
        global_path.write_text(json.dumps(global_payload, indent=2), encoding="utf-8")

        grid_path = provider_dir / "metric_grid.json"
        grid_payload = {
            "enabled": self.settings.metric_grid_enabled,
            "cell_size_m": float(self.settings.metric_grid_cell_size_m),
            "cells": [
                {
                    "i": int(key[0]),
                    "j": int(key[1]),
                    "mean_height": float(value["sum_height"]) / max(int(value["count"]), 1),
                    "count": int(value["count"]),
                    "normal": (
                        np.asarray(value["normal_sum"], dtype=np.float32)
                        / max(float(np.linalg.norm(np.asarray(value["normal_sum"], dtype=np.float32))), 1e-6)
                    ).astype(float).tolist(),
                    "last_frame": int(value["last_frame"]),
                }
                for key, value in sorted(grid_cells.items(), key=lambda kv: (kv[0][0], kv[0][1]))
            ],
        }
        grid_path.write_text(json.dumps(grid_payload, indent=2), encoding="utf-8")

        summary_path = vis_dir / "road_plane_upgrade_summary.json"
        summary = {
            "global_state_history": str(global_path),
            "metric_grid": str(grid_path),
            "global_frame_count": len(frame_indices),
            "metric_grid_cell_count": len(grid_cells),
            "settings": {
                "temporal_mode": self.settings.temporal_mode,
                "road_labels": list(self.settings.road_labels),
                "sidewalk_labels": list(self.settings.sidewalk_labels),
                "include_sidewalk_in_support": bool(self.settings.include_sidewalk_in_support),
                "support_source": self.settings.support_source,
                "support_pixel_stride": self.settings.support_pixel_stride,
                "support_min_confidence": self.settings.support_min_confidence,
                "support_min_points": self.settings.support_min_points,
                "support_max_layering_ratio": self.settings.support_max_layering_ratio,
                "support_max_depth_spread_p90_m": self.settings.support_max_depth_spread_p90_m,
                "support_adaptive_forward_min_enabled": self.settings.support_adaptive_forward_min_enabled,
                "support_forward_min_floor_m": self.settings.support_forward_min_floor_m,
                "support_forward_min_step_m": self.settings.support_forward_min_step_m,
                "support_target_min_points": self.settings.support_target_min_points,
                "support_adaptive_forward_max_iters": self.settings.support_adaptive_forward_max_iters,
                "auto_disable_height_anchor_for_nonmetric_depth": self.settings.auto_disable_height_anchor_for_nonmetric_depth,
                "max_reused_frames": self.settings.max_reused_frames,
                "adaptive_window_enabled": self.settings.adaptive_window_enabled,
                "window_causal_only": self.settings.window_causal_only,
                "window_exclude_catastrophic_frames": self.settings.window_exclude_catastrophic_frames,
                "window_exclude_low_support_frames": self.settings.window_exclude_low_support_frames,
                "window_min_support_points_for_inclusion": self.settings.window_min_support_points_for_inclusion,
                "window_min_frames_required": self.settings.window_min_frames_required,
                "multi_hypothesis_enabled": self.settings.multi_hypothesis_enabled,
                "metric_grid_enabled": self.settings.metric_grid_enabled,
                "min_window_points": self.settings.min_window_points,
                "min_lateral_span_m": self.settings.min_lateral_span_m,
                "min_forward_span_m": self.settings.min_forward_span_m,
                "max_condition_number": self.settings.max_condition_number,
                "min_left_right_balance_ratio": self.settings.min_left_right_balance_ratio,
                "gating_max_residual_p90_m": self.settings.gating_max_residual_p90_m,
                "gating_min_inlier_ratio": self.settings.gating_min_inlier_ratio,
                "catastrophic_residual_p90_m": self.settings.catastrophic_residual_p90_m,
                "catastrophic_min_inlier_ratio": self.settings.catastrophic_min_inlier_ratio,
                "saved_point_max_residual_p90_m": self.settings.saved_point_max_residual_p90_m,
                "saved_point_min_inlier_ratio": self.settings.saved_point_min_inlier_ratio,
                "saved_point_startup_grace_frames": self.settings.saved_point_startup_grace_frames,
                "saved_point_startup_max_residual_p90_m": self.settings.saved_point_startup_max_residual_p90_m,
                "saved_point_startup_min_inlier_ratio": self.settings.saved_point_startup_min_inlier_ratio,
                "max_saved_point_skips": self.settings.max_saved_point_skips,
                "allow_degraded_output": self.settings.allow_degraded_output,
                "validation_policy": dict(validation_policy),
                "effective_max_reused_frames": int(effective_max_reused_frames),
                "effective_max_saved_point_skips": int(effective_max_saved_point_skips),
                "recovery_fit_enabled": self.settings.recovery_fit_enabled,
                "recovery_fit_min_points": self.settings.recovery_fit_min_points,
                "recovery_fit_huber_delta_scale": self.settings.recovery_fit_huber_delta_scale,
                "recovery_fit_accept_max_residual_p90_m": self.settings.recovery_fit_accept_max_residual_p90_m,
                "recovery_fit_accept_min_inlier_ratio": self.settings.recovery_fit_accept_min_inlier_ratio,
                "lambda_up": self.settings.lambda_up,
                "max_roll_pitch_step_deg": self.settings.max_roll_pitch_step_deg,
                "max_height_step_m": self.settings.max_height_step_m,
                "state_causal_smoothing": self.settings.state_causal_smoothing,
                "residual_metadata_tolerance_m": self.settings.residual_metadata_tolerance_m,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOG.info(
            "[RoadPlane] Wrote upgraded artifacts: global=%s grid=%s summary=%s",
            global_path,
            grid_path,
            summary_path,
        )

    def _reset_visualization_artifacts(self, resources: ResourceStore) -> None:
        vis_dir = resources.visualizations_dir("road_plane")
        if not vis_dir.exists():
            return
        for frame_dir in vis_dir.glob("frame_*"):
            if frame_dir.is_dir():
                shutil.rmtree(frame_dir, ignore_errors=True)
        for video_name in ("plane_residuals.mp4", "road_overlay.mp4"):
            path = vis_dir / video_name
            if path.exists():
                with contextlib.suppress(Exception):
                    path.unlink()


def _residual_colors(residuals: np.ndarray) -> np.ndarray:
    res = np.clip(residuals, -0.5, 0.5)
    norm = (res + 0.5) / 1.0
    r = (norm * 255).astype(np.uint8)
    b = (255 - r).astype(np.uint8)
    g = np.clip(255 - np.abs(residuals) * 510, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


def register_road_plane_provider_builders(factory) -> None:
    factory.register(
        "RobustRoadPlaneProvider",
        lambda binding, context: RobustRoadPlaneProvider(binding.settings),
    )
