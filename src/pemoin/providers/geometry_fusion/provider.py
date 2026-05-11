"""GeometryFusionProvider: SOTA geometry fusion batch provider.

Orchestrates per-frame affine depth rectification, DPVO metric scale alignment,
quadratic road surface modelling, optional GTSAM factor-graph fusion, and
quality gating. Replaces the older piecewise-plane alignment path with one
maintained metric geometry fusion stage.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np

from pemoin.coordinate_systems.trajectory_origin import save_origin_anchored_trajectory
from pemoin.data.contracts import (
    DepthData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    RoadPlaneData,
)
from pemoin.providers.base import Provider, ProviderExecutionMode
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.semantic_roles import resolve_semantic_role_labels
from pemoin.runtime.cache import CrossRunCacheManager
from pemoin.providers.geometry_fusion.stages.quality_gating import (
    assess_quality,
    check_plateau_refit_needed,
)
from pemoin.providers.geometry_fusion.stages.quadratic_surface import (
    fit_quadratic_surfaces,
)
from pemoin.providers.geometry_fusion.stages.road_rectification import (
    FrameRectificationResult,
    fit_per_frame_planes,
    optimize_temporal_smoothness,
)
from pemoin.providers.geometry_fusion.stages.scale_alignment import (
    apply_global_scale,
    evaluate_dpvo_scale_candidate,
    estimate_global_dpvo_scale,
    estimate_windowed_dpvo_local_scale,
)
from pemoin.providers.geometry_fusion.utils.plane_fitting import ransac_irls_plane_fit
from pemoin.providers.geometry_fusion.utils.road_pixel_selection import select_road_pixels
from pemoin.utils.geometry_validation import (
    GeometryValidationConfig,
    validate_road_plane_anchor_consistency,
)
from pemoin.utils.logging import get_logger

LOG = get_logger()
_GT_METRIC_SOURCES = frozenset({"carla", "unity", "vkitti2"})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _geometry_fusion_failure_report(
    resources: ResourceStore,
    *,
    settings: GeometryFusionSettings,
    frame_indices: list[int],
    exc: Exception,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "geometry_fusion",
        "stage": "road_rectification",
        "reason": str(exc),
        "frame_indices": list(frame_indices),
        "settings": _jsonable(asdict(settings)),
    }
    diagnostic = getattr(exc, "diagnostic_payload", None)
    if isinstance(diagnostic, Mapping):
        payload["road_selection"] = _jsonable(dict(diagnostic))
        frame_index = diagnostic.get("frame_index")
        if isinstance(frame_index, (int, np.integer)):
            frame_idx = int(frame_index)
            semantics_path = resources.path_for(ResourceKind.SEMANTICS_2D, frame_idx)
            if semantics_path.exists():
                payload["semantics_npz_path"] = str(semantics_path)
            debug_dir = resources.visualizations_dir("semantics_debug") / "carla"
            debug_artifacts = {
                "labels_json": debug_dir / f"{frame_idx:06d}_labels.json",
                "labels_png": debug_dir / f"{frame_idx:06d}_labels.png",
                "road_mask_png": debug_dir / f"{frame_idx:06d}_road_mask.png",
                "road_candidate_png": debug_dir / f"{frame_idx:06d}_road_candidate.png",
                "road_overlay_png": debug_dir / f"{frame_idx:06d}_road_overlay.png",
            }
            payload["semantics_debug_artifacts"] = {
                name: str(path)
                for name, path in debug_artifacts.items()
                if path.exists()
            }
    return payload


def _normalized_source(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("source", "")).strip().lower()


def _geometry_validation_config_from_context(
    context: Mapping[str, Any] | None,
) -> GeometryValidationConfig:
    raw = context.get("geometry_validation") if isinstance(context, Mapping) else None
    return GeometryValidationConfig.from_settings(raw if isinstance(raw, Mapping) else None)


def _select_geometry_input_mode(
    *,
    settings: GeometryFusionSettings,
    trajectory: PoseData,
    depth_metadata: Mapping[str, Any] | None,
) -> str:
    trajectory_metadata = trajectory.metadata or {}
    trajectory_source = _normalized_source(trajectory_metadata)
    depth_source = _normalized_source(depth_metadata)
    trajectory_metric = bool(trajectory_metadata.get("metric_scale", False))

    if settings.preserve_metric_trajectory:
        if not trajectory_metric:
            raise RuntimeError(
                "Geometry fusion: preserve_metric_trajectory requires a metric trajectory input."
            )
        return "preserved_metric_input"

    if trajectory_source == "dpvo":
        return (
            "dpvo_windowed_local_scale"
            if settings.dpvo_scale_mode == "windowed_local"
            else "dpvo_match_graph_global_scale"
        )

    if (
        trajectory_metric
        and trajectory_source in _GT_METRIC_SOURCES
        and depth_source == trajectory_source
    ):
        return "metric_input_verified"

    if trajectory_metric:
        raise RuntimeError(
            "Geometry fusion: automatic metric-input verification requires matching GT depth and "
            f"trajectory sources, got trajectory={trajectory_source!r} depth={depth_source!r}."
        )
    return (
        "dpvo_windowed_local_scale"
        if settings.dpvo_scale_mode == "windowed_local"
        else "dpvo_match_graph_global_scale"
    )


def _orient_world_support_plane(
    normal_world: np.ndarray,
    offset_world: float,
    camera_pos_world: np.ndarray,
    camera_up_world: np.ndarray,
) -> tuple[np.ndarray, float]:
    n = np.asarray(normal_world, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-8:
        raise RuntimeError("Geometry fusion: road-plane normal is degenerate.")
    n = n / n_norm
    c = np.asarray(camera_pos_world, dtype=np.float64).reshape(3)
    u = np.asarray(camera_up_world, dtype=np.float64).reshape(3)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        raise RuntimeError("Geometry fusion: camera up vector is degenerate.")
    u = u / u_norm

    for sign in (1.0, -1.0):
        n_cand = sign * n
        d_cand = float(sign * float(offset_world))
        anchor = float(np.dot(n_cand, c) + d_cand)
        if anchor <= 0.0:
            continue
        if float(np.dot(n_cand, u)) <= 0.0:
            continue
        return n_cand.astype(np.float32), float(d_cand)

    raise RuntimeError(
        "Geometry fusion: world road plane cannot be oriented as a support surface below the camera."
    )


def _sample_joint_consistency_frames(
    frame_indices: list[int],
    *,
    max_frames: int,
) -> list[int]:
    if len(frame_indices) <= max_frames:
        return [int(v) for v in frame_indices]
    lin = np.linspace(0, len(frame_indices) - 1, num=max_frames, dtype=np.int32)
    return [int(frame_indices[int(idx)]) for idx in np.unique(lin).tolist()]


def _evaluate_global_road_consistency(
    resources: ResourceStore,
    *,
    scaled_traj: PoseData,
    frame_indices: list[int],
    camera_height_m: float,
    K: np.ndarray,
    settings: GeometryFusionSettings,
) -> dict[str, Any]:
    sampled_frames = _sample_joint_consistency_frames(
        frame_indices,
        max_frames=int(settings.joint_consistency_max_sampled_frames),
    )
    scaled_pose_by_frame = {int(s.frame_index): s for s in scaled_traj.samples}
    per_frame: list[dict[str, Any]] = []
    world_points_batches: list[np.ndarray] = []
    world_weight_batches: list[np.ndarray] = []
    first_camera_pos: np.ndarray | None = None
    first_camera_up: np.ndarray | None = None

    for frame_idx in sampled_frames:
        if frame_idx not in scaled_pose_by_frame:
            continue
        depth_data = resources.load_depth(frame_idx)
        semantics = resources.load_semantics2d(frame_idx)
        try:
            selection = select_road_pixels(
                resources=resources,
                depth=depth_data.depth,
                semantics=semantics,
                K=K,
                road_labels=settings.road_labels,
                conf_thresh=settings.road_conf_thresh,
                roi_bottom_frac=settings.roi_bottom_frac,
                z_max_m=settings.z_max_m,
                min_points=max(32, min(int(settings.min_support_points), 256)),
            )
        except RuntimeError as exc:
            per_frame.append(
                {
                    "frame_index": int(frame_idx),
                    "point_count": 0,
                    "status": "selection_failed",
                    "error": str(exc),
                }
            )
            continue

        points_cam = np.asarray(selection.points_cam, dtype=np.float32)
        weights = np.asarray(selection.weights, dtype=np.float32)
        if points_cam.shape[0] > int(settings.joint_consistency_max_points_per_frame):
            rng = np.random.default_rng(int(frame_idx) + 17)
            subset = rng.choice(
                points_cam.shape[0],
                size=int(settings.joint_consistency_max_points_per_frame),
                replace=False,
            )
            points_cam = points_cam[subset]
            weights = weights[subset]

        c2w = np.asarray(scaled_pose_by_frame[frame_idx].camera_to_world, dtype=np.float32)
        rot = c2w[:3, :3]
        trans = c2w[:3, 3]
        points_world = (rot @ points_cam.T).T + trans[None, :]
        world_points_batches.append(np.asarray(points_world, dtype=np.float32))
        world_weight_batches.append(np.asarray(weights, dtype=np.float32))
        if first_camera_pos is None:
            first_camera_pos = np.asarray(trans, dtype=np.float32)
            first_camera_up = np.asarray(rot[:, 1], dtype=np.float32)
        per_frame.append(
            {
                "frame_index": int(frame_idx),
                "point_count": int(points_world.shape[0]),
                "status": "ok",
            }
        )

    if not world_points_batches or first_camera_pos is None or first_camera_up is None:
        raise RuntimeError(
            "Joint geometry consistency could not assemble enough corrected road support points."
        )

    points_world_all = np.concatenate(world_points_batches, axis=0)
    weights_all = np.concatenate(world_weight_batches, axis=0)
    plane = ransac_irls_plane_fit(
        points=points_world_all,
        weights=weights_all,
        iters=max(256, int(settings.ransac_iters // 2)),
        inlier_thresh=float(settings.inlier_thresh_m),
        huber_delta=float(settings.huber_delta_plane_m),
        irls_iters=max(4, int(settings.irls_iters)),
        seed=20260409,
    )
    normal_world, offset_world = _orient_world_support_plane(
        plane.normal,
        plane.offset,
        first_camera_pos,
        first_camera_up,
    )
    point_residuals = np.abs(points_world_all @ normal_world.astype(np.float32) + float(offset_world))

    anchor_errors: list[float] = []
    batch_cursor = 0
    for frame_diag, pts_world in zip(per_frame, world_points_batches):
        if frame_diag.get("status") != "ok":
            continue
        count = int(pts_world.shape[0])
        batch = point_residuals[batch_cursor : batch_cursor + count]
        batch_cursor += count
        frame_idx = int(frame_diag["frame_index"])
        cam_pos = np.asarray(scaled_pose_by_frame[frame_idx].camera_to_world[:3, 3], dtype=np.float32)
        anchor = float(normal_world @ cam_pos + float(offset_world))
        frame_diag["camera_height_anchor_m"] = anchor
        frame_diag["camera_height_abs_error_m"] = abs(anchor - float(camera_height_m))
        frame_diag["point_residual_median_m"] = float(np.median(batch)) if batch.size else None
        frame_diag["point_residual_p90_m"] = float(np.percentile(batch, 90)) if batch.size else None
        anchor_errors.append(abs(anchor - float(camera_height_m)))

    return {
        "sampled_frames": [int(v) for v in sampled_frames],
        "valid_sampled_frame_count": int(sum(1 for item in per_frame if item.get("status") == "ok")),
        "global_plane_normal": normal_world.astype(float).tolist(),
        "global_plane_offset": float(offset_world),
        "global_plane_residual_median_m": float(np.median(point_residuals)),
        "global_plane_residual_p90_m": float(np.percentile(point_residuals, 90)),
        "camera_height_median_abs_err_m": float(np.median(anchor_errors)) if anchor_errors else None,
        "camera_height_max_abs_err_m": float(np.max(anchor_errors)) if anchor_errors else None,
        "frame_diagnostics": per_frame,
    }


def resolve_joint_consistent_global_scale(
    resources: ResourceStore,
    *,
    traj: PoseData,
    frame_indices: list[int],
    K: np.ndarray,
    camera_height_m: float,
    settings: GeometryFusionSettings,
    input_mode: str,
    quality_reports: Sequence[Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve one global trajectory scale under corrected-depth road consistency."""
    src = _normalized_source(traj.metadata)
    diagnostics: dict[str, Any] = {
        "source": "joint_depth_trajectory_height_consistency",
        "camera_height_m": float(camera_height_m),
        "camera_height_source": "frame0_constant_assumption",
        "input_mode": str(input_mode),
    }

    base_scale = 1.0
    match_graph_diagnostics: dict[str, Any] | None = None
    if src == "dpvo":
        base_scale, match_graph_diagnostics = estimate_global_dpvo_scale(
            resources,
            traj,
            [],
            K,
            camera_height_m,
            settings,
            quality_reports=quality_reports,
            context=context,
        )
        diagnostics["match_graph_scale"] = float(base_scale)
        diagnostics["match_graph_diagnostics"] = match_graph_diagnostics
        low = max(float(settings.dpvo_match_scale_min), float(base_scale) * 0.25)
        high = min(float(settings.dpvo_match_scale_max), float(base_scale) * 4.0)
        candidates = np.geomspace(low, high, num=17, dtype=np.float64)
        candidates = np.unique(
            np.concatenate([candidates, np.asarray([1.0, float(base_scale)], dtype=np.float64)])
        )
    else:
        candidates = np.asarray(
            [0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1],
            dtype=np.float64,
        )

    candidate_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    for candidate_scale in candidates.tolist():
        scaled_traj = (
            traj
            if abs(float(candidate_scale) - 1.0) <= 1e-8
            else apply_global_scale(traj, float(candidate_scale))
        )
        road_consistency = _evaluate_global_road_consistency(
            resources,
            scaled_traj=scaled_traj,
            frame_indices=frame_indices,
            camera_height_m=camera_height_m,
            K=K,
            settings=settings,
        )
        record: dict[str, Any] = {
            "scale": float(candidate_scale),
            "road_consistency": road_consistency,
        }
        score = float(road_consistency["global_plane_residual_p90_m"])
        anchor_median = road_consistency.get("camera_height_median_abs_err_m")
        if anchor_median is not None:
            score += float(anchor_median)

        if src == "dpvo":
            reproj_eval = evaluate_dpvo_scale_candidate(
                resources,
                traj,
                K,
                settings,
                scale=float(candidate_scale),
            )
            record["dpvo_reprojection"] = reproj_eval
            if (
                float(reproj_eval["median_residual_px"])
                > float(settings.joint_consistency_hard_max_median_residual_px)
                or float(reproj_eval["p90_residual_px"])
                > float(settings.joint_consistency_hard_max_p90_residual_px)
            ):
                record["rejected_reason"] = "reprojection_limit_exceeded"
                candidate_records.append(record)
                continue
            score += float(settings.joint_consistency_reprojection_weight) * (
                float(reproj_eval["median_residual_px"])
                + 0.5 * float(reproj_eval["p90_residual_px"])
            )
        record["joint_score"] = float(score)
        candidate_records.append(record)
        if best_record is None or float(record["joint_score"]) < float(best_record["joint_score"]):
            best_record = record

    diagnostics["candidates"] = candidate_records
    if best_record is None:
        raise RuntimeError(
            "Joint geometry consistency could not find a scale that satisfies the hard reprojection bounds."
        )

    selected_scale = float(best_record["scale"])
    diagnostics["selected_scale"] = selected_scale
    diagnostics["selected_road_consistency"] = best_record.get("road_consistency", {})
    if "dpvo_reprojection" in best_record:
        diagnostics["selected_dpvo_reprojection"] = best_record["dpvo_reprojection"]

    if src != "dpvo":
        gt_scale_delta = abs(selected_scale - 1.0)
        diagnostics["gt_scale_delta"] = float(gt_scale_delta)
        if gt_scale_delta >= float(settings.joint_consistency_gt_fail_scale_delta):
            raise RuntimeError(
                "Joint geometry consistency requires a large GT trajectory correction "
                f"({gt_scale_delta:.4f} >= {float(settings.joint_consistency_gt_fail_scale_delta):.4f})."
            )
        if gt_scale_delta >= float(settings.joint_consistency_gt_warn_scale_delta):
            diagnostics["gt_warning"] = (
                "GT trajectory required a non-trivial consistency correction."
            )
            LOG.warning(
                "Geometry fusion: GT input required a non-trivial consistency correction "
                "(scale=%.5f delta=%.5f).",
                selected_scale,
                gt_scale_delta,
            )

    return selected_scale, diagnostics


def _apply_frame_local_scale_ratios(
    rect_results: Sequence[FrameRectificationResult],
    *,
    local_scale_ratios: Mapping[int, float] | None,
) -> list[FrameRectificationResult]:
    if not local_scale_ratios:
        return list(rect_results)
    adjusted: list[FrameRectificationResult] = []
    for result in rect_results:
        ratio = float(local_scale_ratios.get(int(result.frame_index), 1.0))
        if not np.isfinite(ratio) or ratio <= 0.0:
            ratio = 1.0
        adjusted.append(
            FrameRectificationResult(
                frame_index=result.frame_index,
                normal_cam=result.normal_cam,
                offset_cam=result.offset_cam,
                implied_height_m=result.implied_height_m,
                scale=float(result.scale) * ratio,
                bias=result.bias,
                inlier_ratio=result.inlier_ratio,
                residual_p90_m=result.residual_p90_m,
                support_count=result.support_count,
            )
        )
    return adjusted


class GeometryFusionProvider(Provider):
    """Batch provider implementing SOTA geometry fusion pipeline."""

    execution_mode = ProviderExecutionMode.BATCH
    required_resources = frozenset(
        {
            ResourceKind.FRAMES,
            ResourceKind.INTRINSICS,
            ResourceKind.DEPTH,
            ResourceKind.TRAJECTORY,
            ResourceKind.CAMERA_HEIGHT,
            ResourceKind.SEMANTICS_2D,
        }
    )
    produced_resources = frozenset(
        {ResourceKind.DEPTH, ResourceKind.TRAJECTORY, ResourceKind.ROAD_PLANE}
    )

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = GeometryFusionSettings.from_mapping(settings)
        self._cache_manager: CrossRunCacheManager | None = None
        self._profile_name: str | None = None
        self._cache_enabled = False
        self._cache_signature: str | None = None
        self._cache_payload: dict[str, Any] | None = None
        self._cache_status: dict[str, Any] = {
            "cross_run_cache_enabled": False,
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled",
        }

    def setup(self, context: MutableMapping[str, Any]) -> None:
        cache_manager = context.get("cross_run_cache")
        stage_settings = context.get("cross_run_cache_stage_settings")
        stage_enabled = True
        if isinstance(stage_settings, Mapping):
            geometry_settings = stage_settings.get("geometry_fusion")
            if isinstance(geometry_settings, Mapping) and "enabled" in geometry_settings:
                stage_enabled = bool(geometry_settings.get("enabled"))
        self._cache_manager = (
            cache_manager if isinstance(cache_manager, CrossRunCacheManager) else None
        )
        self._cache_enabled = bool(
            self._cache_manager is not None and self._cache_manager.enabled and stage_enabled
        )
        self._profile_name = (
            str(context.get("profile_name"))
            if context.get("profile_name") is not None
            else None
        )
        self._cache_status = {
            "cross_run_cache_enabled": self._cache_enabled,
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled" if not self._cache_enabled else "not-checked",
        }

    def process(self, frame: Any) -> None:
        raise RuntimeError("GeometryFusionProvider is batch-oriented; use run().")

    def teardown(self) -> None:
        return None

    def run(self, resources: ResourceStore, context: MutableMapping[str, object] | None = None) -> None:
        """Execute the full geometry fusion pipeline.

        Stages:
        1. Per-frame plane fit + affine correction parameters
        2. Temporal smoothness optimization (L-BFGS-B)
        3. Pre-quality gating
        4. Apply affine correction to depth maps
        5. Global trajectory scaling (or metric-GT preservation)
        6. Backproject corrected road depth to world, fit quadratic surfaces
        7. Factor-graph fusion (optional, requires GTSAM)
        8. Post-quality validation
        9. Save all outputs
        """
        self.validate_requirements(resources)
        self._cache_payload = self._cross_run_payload(resources)
        if self._cache_payload is not None and self._cache_manager is not None:
            self._cache_signature = self._cache_manager.signature(
                "geometry_fusion",
                self._cache_payload,
            )
            lookup = self._cache_manager.lookup(
                "geometry_fusion",
                self._cache_signature,
                required_relpaths=[
                    "standard/trajectory/poses.npz",
                ],
            )
            self._cache_status.update(
                {
                    "cross_run_cache_signature": self._cache_signature,
                    "cross_run_cache_hit": lookup.hit,
                    "cross_run_cache_entry": str(lookup.entry_dir),
                    "cross_run_cache_validation": lookup.reason,
                }
            )
            if lookup.hit:
                materialized = self._cache_manager.materialize(
                    "geometry_fusion",
                    self._cache_signature,
                    run_root=resources.root,
                )
                self._cache_status["cross_run_cache_materialized"] = materialized
                LOG.info("Reused cross-run geometry fusion cache at '%s'.", lookup.entry_dir)
                return
            self._cache_status["cross_run_cache_reason"] = lookup.reason
        semantics_tool = None
        role_defaults = None
        if isinstance(context, Mapping):
            if context.get("semantics_tool") is not None:
                semantics_tool = str(context.get("semantics_tool"))
            if isinstance(context.get("semantic_role_defaults"), Mapping):
                role_defaults = context.get("semantic_role_defaults")
        settings = replace(
            self.settings,
            road_labels=resolve_semantic_role_labels(
                "road",
                tool=semantics_tool,
                defaults=role_defaults,
                required=True,
                source_name="GeometryFusionProvider",
            ),
            dpvo_match_dynamic_labels=resolve_semantic_role_labels(
                "mobile",
                tool=semantics_tool,
                defaults=role_defaults,
                required=True,
                source_name="GeometryFusionProvider",
            ),
        )

        # --- Load shared data ---
        intrinsics = resources.load_intrinsics()
        K = np.asarray(intrinsics.matrix, dtype=np.float32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if not (np.isfinite(fx) and np.isfinite(fy) and abs(fx) > 1e-9 and abs(fy) > 1e-9):
            raise RuntimeError("Geometry fusion: invalid intrinsics (fx/fy).")

        traj = self._load_trajectory(resources)
        pose_by_frame: Dict[int, PoseSample] = {int(s.frame_index): s for s in traj.samples}

        frame_indices = sorted(
            set(resources.frame_indices(ResourceKind.DEPTH))
            & set(resources.frame_indices(ResourceKind.SEMANTICS_2D))
            & set(resources.frame_indices(ResourceKind.CAMERA_HEIGHT))
            & set(pose_by_frame.keys())
        )
        if not frame_indices:
            raise RuntimeError("Geometry fusion: no overlapping depth/semantics/height/trajectory frames.")

        # Use first frame's camera height (assumed constant for now)
        height_data = resources.load_camera_height(frame_indices[0])
        camera_height_m = float(height_data.height_m)

        diag_dir = resources.provider_dir("geometry_fusion")
        diag_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: Per-frame plane fit ---
        LOG.info("Geometry fusion: fitting per-frame road planes (%d frames).", len(frame_indices))
        try:
            rect_results = fit_per_frame_planes(
                resources, frame_indices, K, camera_height_m, settings
            )
        except Exception as exc:
            failure_path = diag_dir / "failure_diagnostics.json"
            failure_report = _geometry_fusion_failure_report(
                resources,
                settings=settings,
                frame_indices=frame_indices,
                exc=exc,
            )
            failure_path.write_text(
                json.dumps(_jsonable(failure_report), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            message = (
                "Geometry fusion failed during road rectification. "
                f"{exc} Diagnostic written to {failure_path}."
            )
            LOG.error(message)
            raise RuntimeError(message) from exc

        input_mode = _select_geometry_input_mode(
            settings=settings,
            trajectory=traj,
            depth_metadata=resources.load_depth(frame_indices[0]).metadata,
        )
        validation_config = _geometry_validation_config_from_context(context)
        global_scale = 1.0
        scaled_traj = traj
        tmeta = dict(traj.metadata or {})

        if input_mode == "metric_input_verified":
            LOG.info(
                "Geometry fusion: metric GT depth+trajectory detected; running joint consistency verification path."
            )
            quality_reports = assess_quality(rect_results, camera_height_m, settings)
            n_ok = sum(1 for q in quality_reports if q.quality_ok)
            LOG.info(
                "Geometry fusion: raw metric-input road-fit quality passed for %d/%d frames.",
                n_ok,
                len(quality_reports),
            )
            post_quality = quality_reports
            global_scale, scale_diagnostics = resolve_joint_consistent_global_scale(
                resources,
                traj=traj,
                frame_indices=frame_indices,
                K=K,
                camera_height_m=camera_height_m,
                settings=settings,
                input_mode=input_mode,
                quality_reports=quality_reports,
                context=context,
            )
            trajectory_scale_mode = "joint_metric_input_verified"
            metadata_flags: Mapping[str, Any] = {}
            if abs(float(global_scale) - 1.0) > 1e-8:
                scaled_traj = apply_global_scale(traj, global_scale)
        else:
            # --- Stage 2: Temporal smoothness optimization ---
            LOG.info("Geometry fusion: optimizing temporal smoothness via L-BFGS-B.")
            rect_results = optimize_temporal_smoothness(rect_results, camera_height_m, settings)

            if check_plateau_refit_needed(rect_results, settings):
                LOG.info("Geometry fusion: scale plateaus detected, refitting with 2x lambda_s.")
                boosted = replace(settings, lambda_s=settings.lambda_s * 2.0)
                rect_results = optimize_temporal_smoothness(rect_results, camera_height_m, boosted)

            # --- Stage 3: Pre-quality gating ---
            LOG.info("Geometry fusion: assessing per-frame quality.")
            quality_reports = assess_quality(rect_results, camera_height_m, settings)
            n_ok = sum(1 for q in quality_reports if q.quality_ok)
            LOG.info("Geometry fusion: %d/%d frames passed quality gating.", n_ok, len(quality_reports))

            local_scale_ratios: Mapping[int, float] = {}

            # --- Stage 4: Trajectory scaling selection ---
            if input_mode == "preserved_metric_input":
                LOG.info("Geometry fusion: preserving metric input trajectory.")
                scale_diagnostics = {
                    "source": "preserved_metric_input",
                    "camera_height_m": float(camera_height_m),
                    "camera_height_source": "frame0_constant_assumption",
                    "trajectory_scale_mode": "preserved_metric_input",
                    "candidate_pair_count": 0,
                    "valid_pair_count": 0,
                    "pairs": [],
                }
                trajectory_scale_mode = "preserved_metric_input"
            elif input_mode == "dpvo_windowed_local_scale":
                LOG.info("Geometry fusion: estimating windowed local DPVO scale field.")
                scale_diagnostics = estimate_windowed_dpvo_local_scale(
                    resources,
                    traj,
                    rect_results,
                    K,
                    settings,
                    quality_reports=quality_reports,
                )
                global_scale = float(scale_diagnostics.get("global_scale", 1.0))
                local_scale_ratios = {
                    int(k): float(v)
                    for k, v in scale_diagnostics.get("frame_local_scale_ratios", {}).items()
                }
                rect_results = _apply_frame_local_scale_ratios(
                    rect_results,
                    local_scale_ratios=local_scale_ratios,
                )
                scaled_traj = apply_global_scale(traj, global_scale)
                trajectory_scale_mode = (
                    "windowed_local_scale_degraded"
                    if bool(scale_diagnostics.get("degraded_mode", False))
                    else "windowed_local_scale"
                )
            else:
                LOG.info("Geometry fusion: resolving joint global metric scale.")
                global_scale, scale_diagnostics = resolve_joint_consistent_global_scale(
                    resources,
                    traj=traj,
                    frame_indices=frame_indices,
                    K=K,
                    camera_height_m=camera_height_m,
                    settings=settings,
                    input_mode=input_mode,
                    quality_reports=quality_reports,
                    context=context,
                )
                scaled_traj = apply_global_scale(traj, global_scale)
                trajectory_scale_mode = "joint_depth_trajectory_height_scale"

            # --- Stage 5: Apply affine correction to depth maps ---
            LOG.info("Geometry fusion: applying affine depth correction.")
            for r in rect_results:
                depth_data = resources.load_depth(r.frame_index)
                corrected = np.asarray(depth_data.depth, dtype=np.float32) * r.scale + r.bias
                dmeta = dict(depth_data.metadata or {})
                dmeta.update({
                    "scale_source": "geometry_fusion",
                    "scale_factor": float(r.scale),
                    "bias_m": float(r.bias),
                    "metric_scale": True,
                })
                if local_scale_ratios:
                    local_ratio = float(local_scale_ratios.get(int(r.frame_index), 1.0))
                    dmeta.update(
                        {
                            "local_metric_scale_ratio": float(local_ratio),
                            "local_metric_scale_source": str(scale_diagnostics.get("source", "")),
                        }
                    )
                resources.save_depth(
                    DepthData(
                        frame_index=r.frame_index,
                        depth=corrected,
                        confidence=depth_data.confidence,
                        metadata=dmeta,
                    )
                )

            # --- Stage 6: Backproject road depth to world, fit quadratic surfaces ---
            road_points_world: list[np.ndarray] = []
            plane_normals_world: list[np.ndarray] = []
            scaled_pose_by_frame = {int(s.frame_index): s for s in scaled_traj.samples}

            for r in rect_results:
                fi = r.frame_index
                depth_data = resources.load_depth(fi)
                semantics = resources.load_semantics2d(fi)

                try:
                    selection = select_road_pixels(
                        resources=resources,
                        depth=depth_data.depth,
                        semantics=semantics,
                        K=K,
                        road_labels=settings.road_labels,
                        conf_thresh=settings.road_conf_thresh,
                        roi_bottom_frac=settings.roi_bottom_frac,
                        z_max_m=settings.z_max_m,
                        min_points=max(10, settings.min_support_points // 10),
                    )
                except RuntimeError:
                    road_points_world.append(np.zeros((0, 3), dtype=np.float32))
                    plane_normals_world.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                    continue

                if fi in scaled_pose_by_frame:
                    c2w = scaled_pose_by_frame[fi].camera_to_world.astype(np.float32)
                else:
                    road_points_world.append(np.zeros((0, 3), dtype=np.float32))
                    plane_normals_world.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                    continue

                rot = c2w[:3, :3]
                trans = c2w[:3, 3]
                pts_world = (rot @ selection.points_cam.T).T + trans[None, :]
                road_points_world.append(pts_world)

                n_world = rot @ r.normal_cam
                n_world = n_world / max(float(np.linalg.norm(n_world)), 1e-8)
                plane_normals_world.append(n_world)

            if settings.quadratic_enabled:
                LOG.info("Geometry fusion: fitting quadratic road surfaces.")
                scaled_poses_ordered = [scaled_pose_by_frame[fi] for fi in frame_indices if fi in scaled_pose_by_frame]
                quad_results = fit_quadratic_surfaces(
                    road_points_world,
                    scaled_poses_ordered,
                    plane_normals_world,
                    settings,
                )
            else:
                quad_results = []

            # --- Stage 7: Factor-graph fusion (optional) ---
            if settings.factor_graph_enabled and input_mode in {
                "dpvo_match_graph_global_scale",
                "dpvo_windowed_local_scale",
            }:
                LOG.info("Geometry fusion: running factor-graph fusion.")
                from pemoin.providers.geometry_fusion.stages.factor_graph import run_factor_graph_fusion
                scaled_traj, rect_results, quad_results = run_factor_graph_fusion(
                    scaled_traj,
                    rect_results,
                    quad_results,
                    camera_height_m,
                    K,
                    resources,
                    frame_indices,
                    settings,
                )

                for r in rect_results:
                    depth_data = resources.load_depth(r.frame_index)
                    orig_meta = depth_data.metadata or {}
                    if orig_meta.get("scale_source") == "geometry_fusion":
                        old_scale = float(orig_meta.get("scale_factor", 1.0))
                        old_bias = float(orig_meta.get("bias_m", 0.0))
                        raw_depth = (depth_data.depth - old_bias) / max(old_scale, 1e-8)
                        corrected = raw_depth * r.scale + r.bias
                        dmeta = dict(depth_data.metadata or {})
                        dmeta.update({
                            "scale_factor": float(r.scale),
                            "bias_m": float(r.bias),
                            "factor_graph_corrected": True,
                        })
                        resources.save_depth(
                            DepthData(
                                frame_index=r.frame_index,
                                depth=corrected,
                                confidence=depth_data.confidence,
                                metadata=dmeta,
                            )
                        )
            elif settings.factor_graph_enabled:
                LOG.info(
                    "Geometry fusion: skipping factor-graph fusion because trajectory scaling is not DPVO-based."
                )

            # --- Save metric trajectory ---
            LOG.info("Geometry fusion: saving metric trajectory.")
            metadata_flags = (
                scale_diagnostics.get("metadata_flags_to_apply", {})
                if isinstance(scale_diagnostics, Mapping)
                else {}
            )
            tmeta = dict(scaled_traj.metadata or {})
            tmeta.update({
                "metric_scale": True,
                "scale_source": (
                    "geometry_fusion"
                    if input_mode == "preserved_metric_input"
                    else "geometry_fusion_joint_consistency"
                ),
                "global_dpvo_scale": float(global_scale),
                "geometry_fusion_frames": len(frame_indices),
                "trajectory_scale_mode": trajectory_scale_mode,
            })
            if input_mode == "dpvo_windowed_local_scale":
                tmeta.update(
                    {
                        "scale_source": "geometry_fusion_windowed_local_scale",
                        "local_scale_window_count": int(scale_diagnostics.get("window_count", 0)),
                        "local_scale_confident_window_count": int(
                            scale_diagnostics.get("confident_window_count", 0)
                        ),
                        "local_scale_low_confidence_ratio": float(
                            scale_diagnostics.get("low_confidence_ratio", 0.0)
                        ),
                    }
                )
            if isinstance(metadata_flags, Mapping):
                tmeta.update(dict(metadata_flags))
            if input_mode == "preserved_metric_input":
                scaled_traj = PoseData(samples=scaled_traj.samples, metadata=tmeta)
                resources.save_trajectory(scaled_traj)
            else:
                scaled_traj, _ = save_origin_anchored_trajectory(
                    resources,
                    PoseData(samples=scaled_traj.samples, metadata=tmeta),
                    metadata_label="geometry_fusion",
                )

            if isinstance(metadata_flags, Mapping) and metadata_flags:
                for fi in frame_indices:
                    depth_data = resources.load_depth(fi)
                    dmeta = dict(depth_data.metadata or {})
                    dmeta.update(dict(metadata_flags))
                    resources.save_depth(
                        DepthData(
                            frame_index=fi,
                            depth=np.asarray(depth_data.depth, dtype=np.float32),
                            confidence=depth_data.confidence,
                            metadata=dmeta,
                        )
                    )

            LOG.info("Geometry fusion: running post-quality validation.")
            post_quality = assess_quality(rect_results, camera_height_m, settings)
            n_ok_post = sum(1 for q in post_quality if q.quality_ok)
            LOG.info(
                "Geometry fusion: post-optimization quality: %d/%d frames OK.",
                n_ok_post,
                len(post_quality),
            )

        # --- Save per-frame road planes ---
        LOG.info("Geometry fusion: saving per-frame road planes.")
        scaled_pose_by_frame = {int(s.frame_index): s for s in scaled_traj.samples}
        for i, r in enumerate(rect_results):
            fi = r.frame_index
            if fi not in scaled_pose_by_frame:
                continue

            c2w = scaled_pose_by_frame[fi].camera_to_world.astype(np.float64)
            rot = c2w[:3, :3]
            trans = c2w[:3, 3]
            cam_up_world = rot[:, 1]

            # Transform plane from camera to world coordinates
            n_world = (rot @ r.normal_cam.astype(np.float64)).astype(np.float32)
            n_world = n_world / max(float(np.linalg.norm(n_world)), 1e-8)

            # Offset in world: n_world · p_world + d_world = 0
            # For a point on the plane in camera coords: n_cam · p_cam + d_cam = 0
            # p_world = R @ p_cam + t, so n_world · (R @ p_cam + t) + d_world = 0
            # n_cam · p_cam + n_world · t + d_world = 0 (since n_world = R @ n_cam)
            # d_world = d_cam_corrected - n_world · t
            d_cam = (
                r.offset_cam
                if input_mode == "metric_input_verified"
                else r.offset_cam * r.scale
            )
            d_world = float(d_cam - np.dot(n_world, trans))
            n_world, d_world = _orient_world_support_plane(
                n_world,
                d_world,
                trans,
                cam_up_world,
            )

            height_data = resources.load_camera_height(fi)
            resources.save_road_plane(
                RoadPlaneData(
                    frame_index=fi,
                    normal=n_world,
                    offset=d_world,
                    metadata={
                        "source": "geometry_fusion",
                        "measurement_allowed": True,
                        "support_quality_ok": r.inlier_ratio >= settings.gate_min_inlier,
                        "inlier_ratio": float(r.inlier_ratio),
                        "residual_p90": float(r.residual_p90_m),
                        "world_coordinate_system": "blender",
                        "axis": "z",
                        "target_camera_height_m": float(height_data.height_m),
                        "enforce_height_anchor": True,
                    },
                )
            )

        if input_mode == "metric_input_verified":
            LOG.info("Geometry fusion: verifying metric input consistency against road plane and camera height.")
        if input_mode != "preserved_metric_input":
            verification_summary = validate_road_plane_anchor_consistency(
                resources,
                config=validation_config,
                frame_indices=frame_indices,
            )
            scale_diagnostics["verification_summary"] = verification_summary

        # --- Save diagnostics ---
        self._save_diagnostics(
            diag_dir,
            frame_indices,
            rect_results,
            quality_reports,
            post_quality,
            global_scale,
            scale_diagnostics,
            camera_height_m,
            tmeta.get("trajectory_scale_mode", trajectory_scale_mode),
        )

        LOG.info("Geometry fusion complete.")

    def _cross_run_payload(self, resources: ResourceStore) -> dict[str, Any] | None:
        if self._cache_manager is None or not self._cache_enabled:
            return None
        repo_root = Path(__file__).resolve().parents[4]
        payload: dict[str, Any] = {
            "settings": _jsonable(asdict(self.settings)),
            "intrinsics": self._cache_manager.resource_file_key_signature(
                resources.path_for(ResourceKind.INTRINSICS),
                logical_name="standard/intrinsics/intrinsics.npz",
            ),
            "depth_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.DEPTH),
                canonicalize_npz=True,
            ),
            "trajectory": self._cache_manager.resource_file_key_signature(
                resources.path_for(ResourceKind.TRAJECTORY),
                logical_name="standard/trajectory/poses.npz",
            ),
            "camera_height_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.CAMERA_HEIGHT),
                canonicalize_npz=True,
            ),
            "semantics_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.SEMANTICS_2D),
                canonicalize_npz=True,
            ),
            "provider_script": self._cache_manager.script_key_signature(
                Path(__file__),
                repo_root=repo_root,
            ),
        }
        for key, script_path in (
            (
                "road_rectification_script",
                Path(__file__).resolve().parent / "stages" / "road_rectification.py",
            ),
            (
                "scale_alignment_script",
                Path(__file__).resolve().parent / "stages" / "scale_alignment.py",
            ),
            (
                "quadratic_surface_script",
                Path(__file__).resolve().parent / "stages" / "quadratic_surface.py",
            ),
            (
                "quality_gating_script",
                Path(__file__).resolve().parent / "stages" / "quality_gating.py",
            ),
        ):
            if script_path.exists():
                payload[key] = self._cache_manager.script_key_signature(
                    script_path,
                    repo_root=repo_root,
                )
        factor_graph_script = Path(__file__).resolve().parent / "stages" / "factor_graph.py"
        if factor_graph_script.exists() and bool(self.settings.factor_graph_enabled):
            payload["factor_graph_script"] = self._cache_manager.script_key_signature(
                factor_graph_script,
                repo_root=repo_root,
            )
        return payload

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._cache_status)

    def get_cross_run_cache_spec(self, resources: ResourceStore | None) -> dict[str, Any] | None:
        if (
            resources is None
            or self._cache_manager is None
            or not self._cache_enabled
            or self._cache_signature is None
            or self._cache_payload is None
        ):
            return None
        raw_dir = resources.provider_dir("geometry_fusion")
        trajectory_path = resources.path_for(ResourceKind.TRAJECTORY)
        road_plane_dir = resources.base_dir(ResourceKind.ROAD_PLANE)
        depth_dir = resources.base_dir(ResourceKind.DEPTH)
        artifacts = self._cache_manager.collect_tree(
            raw_dir,
            rel_prefix="raw/geometry_fusion",
        )
        artifacts.update(
            self._cache_manager.collect_tree(
                depth_dir,
                rel_prefix="standard/depth",
            )
        )
        artifacts.update(
            self._cache_manager.collect_file(
                trajectory_path,
                relpath="standard/trajectory/poses.npz",
            )
        )
        artifacts.update(
            self._cache_manager.collect_tree(
                road_plane_dir,
                rel_prefix="standard/road_plane",
            )
        )
        ready = True
        not_ready_reason: str | None = None
        if not (raw_dir.exists() and any(path.is_file() for path in raw_dir.rglob("*"))):
            ready = False
            not_ready_reason = "raw-geometry-fusion-missing"
        elif not (depth_dir.exists() and any(depth_dir.glob("*.npz"))):
            ready = False
            not_ready_reason = "standard-depth-missing"
        elif not trajectory_path.exists():
            ready = False
            not_ready_reason = "standard-trajectory-missing"
        elif not (road_plane_dir.exists() and any(road_plane_dir.glob("*.npz"))):
            ready = False
            not_ready_reason = "standard-road-plane-missing"
        spec: dict[str, Any] = {
            "provider_id": "geometry_fusion",
            "signature": self._cache_signature,
            "payload": self._cache_payload,
            "artifacts": artifacts,
            "ready": ready,
            "source_summary": {
                "profile": self._profile_name,
                "run_root": str(resources.root),
            },
            "provenance": {
                "intrinsics_path": str(resources.path_for(ResourceKind.INTRINSICS)),
                "trajectory_path": str(trajectory_path),
            },
        }
        if not_ready_reason is not None:
            spec["not_ready_reason"] = not_ready_reason
        return spec

    @staticmethod
    def _load_trajectory(resources: ResourceStore) -> PoseData:
        """Load trajectory from resource store."""
        path = resources.path_for(ResourceKind.TRAJECTORY)
        if not path.exists():
            raise RuntimeError(f"Geometry fusion: trajectory file missing at {path}.")
        with np.load(path, allow_pickle=True) as data:
            frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
            c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
            w2c_arr = None
            if "world_to_camera" in data.files:
                raw_w2c = data["world_to_camera"]
                if not (
                    isinstance(raw_w2c, np.ndarray)
                    and raw_w2c.dtype == object
                    and raw_w2c.size == 1
                    and raw_w2c.item() is None
                ):
                    w2c_arr = np.asarray(raw_w2c, dtype=np.float32)
            conf_arr = (
                np.asarray(data["confidence"], dtype=np.float32)
                if "confidence" in data.files
                else None
            )
            metadata: Dict[str, Any] = {}
            if "metadata" in data.files:
                raw = data["metadata"]
                if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
                    try:
                        raw = raw.item()
                    except Exception:
                        raw = {}
                if isinstance(raw, Mapping):
                    metadata = dict(raw)
        samples: list[PoseSample] = []
        for i, frame_idx in enumerate(frame_indices.tolist()):
            w2c = None
            if w2c_arr is not None and i < w2c_arr.shape[0]:
                w2c = np.asarray(w2c_arr[i], dtype=np.float32)
            conf = None
            if conf_arr is not None and i < conf_arr.shape[0]:
                value = float(conf_arr[i])
                if np.isfinite(value):
                    conf = value
            samples.append(
                PoseSample(
                    frame_index=int(frame_idx),
                    camera_to_world=np.asarray(c2w[i], dtype=np.float32),
                    world_to_camera=w2c,
                    confidence=conf,
                    metadata={},
                )
            )
        return PoseData(samples=samples, metadata=metadata)

    @staticmethod
    def _save_diagnostics(
        diag_dir,
        frame_indices,
        rect_results,
        pre_quality,
        post_quality,
        global_scale,
        scale_diagnostics,
        camera_height_m,
        trajectory_scale_mode,
    ) -> None:
        """Save diagnostic artifacts for inspection."""
        summary = {
            "source": "geometry_fusion",
            "frames": [int(f) for f in frame_indices],
            "global_dpvo_scale": float(global_scale),
            "trajectory_scale_mode": str(trajectory_scale_mode),
            "scale_diagnostics_source": str(scale_diagnostics.get("source", "")),
            "camera_height_m": float(camera_height_m),
            "camera_height_source": str(
                scale_diagnostics.get("camera_height_source", "frame0_constant_assumption")
            ),
            "scale_mean": float(np.mean([r.scale for r in rect_results])),
            "scale_std": float(np.std([r.scale for r in rect_results])),
            "bias_mean": float(np.mean([r.bias for r in rect_results])),
            "pre_quality_ok_ratio": float(
                sum(1 for q in pre_quality if q.quality_ok) / max(len(pre_quality), 1)
            ),
            "post_quality_ok_ratio": float(
                sum(1 for q in post_quality if q.quality_ok) / max(len(post_quality), 1)
            ),
            "dpvo_scale_valid_pairs": int(
                scale_diagnostics.get("pair_consistency", {}).get("valid_pair_count", 0)
            ),
            "dpvo_scale_median_residual_m": (
                None
                if scale_diagnostics.get("optimizer_summary", {}).get("median_residual_px") is None
                else float(scale_diagnostics["optimizer_summary"]["median_residual_px"])
            ),
            "dpvo_scale_iqr_ratio": (
                None
                if scale_diagnostics.get("pair_consistency", {}).get("pair_scale_iqr_ratio") is None
                else float(scale_diagnostics["pair_consistency"]["pair_scale_iqr_ratio"])
            ),
            "verification_checked_frames": int(
                scale_diagnostics.get("verification_summary", {}).get("checked_frames", 0)
            ),
            "verification_max_abs_anchor_delta_m": (
                None
                if scale_diagnostics.get("verification_summary", {}).get("max_abs_anchor_delta_m") is None
                else float(scale_diagnostics["verification_summary"]["max_abs_anchor_delta_m"])
            ),
            "joint_selected_scale": (
                None
                if scale_diagnostics.get("selected_scale") is None
                else float(scale_diagnostics.get("selected_scale"))
            ),
            "joint_global_plane_residual_p90_m": (
                None
                if scale_diagnostics.get("selected_road_consistency", {}).get("global_plane_residual_p90_m") is None
                else float(scale_diagnostics["selected_road_consistency"]["global_plane_residual_p90_m"])
            ),
            "joint_camera_height_median_abs_err_m": (
                None
                if scale_diagnostics.get("selected_road_consistency", {}).get("camera_height_median_abs_err_m") is None
                else float(scale_diagnostics["selected_road_consistency"]["camera_height_median_abs_err_m"])
            ),
            "local_scale_window_count": int(scale_diagnostics.get("window_count", 0)),
            "local_scale_confident_window_count": int(
                scale_diagnostics.get("confident_window_count", 0)
            ),
            "local_scale_low_confidence_ratio": float(
                scale_diagnostics.get("low_confidence_ratio", 0.0)
            ),
            "local_scale_degraded_mode": bool(scale_diagnostics.get("degraded_mode", False)),
        }
        (diag_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (diag_dir / "dpvo_scale_diagnostics.json").write_text(
            json.dumps(scale_diagnostics, indent=2),
            encoding="utf-8",
        )
        if isinstance(scale_diagnostics.get("selected_road_consistency"), Mapping):
            joint_frames = scale_diagnostics["selected_road_consistency"].get("frame_diagnostics", [])
            (diag_dir / "joint_consistency_frame_diagnostics.json").write_text(
                json.dumps(_jsonable(joint_frames), indent=2),
                encoding="utf-8",
            )

        per_frame = {}
        for r in rect_results:
            per_frame[str(r.frame_index)] = {
                "scale": float(r.scale),
                "bias": float(r.bias),
                "implied_height_m": float(r.implied_height_m),
                "inlier_ratio": float(r.inlier_ratio),
                "residual_p90_m": float(r.residual_p90_m),
                "support_count": int(r.support_count),
            }
        (diag_dir / "frame_diagnostics.json").write_text(
            json.dumps(per_frame, indent=2), encoding="utf-8"
        )

        np.savez_compressed(
            diag_dir / "scale_diagnostics.npz",
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            scales=np.asarray([r.scale for r in rect_results], dtype=np.float32),
            biases=np.asarray([r.bias for r in rect_results], dtype=np.float32),
            implied_heights=np.asarray(
                [r.implied_height_m for r in rect_results], dtype=np.float32
            ),
            inlier_ratios=np.asarray(
                [r.inlier_ratio for r in rect_results], dtype=np.float32
            ),
        )
