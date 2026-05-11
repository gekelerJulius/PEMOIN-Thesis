"""Runtime entry point for quality metrics computation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pemoin.data.contracts import ResourceKind, ResourceStore
from pemoin.metrics.settings import QualityMetricsSettings

LOG = logging.getLogger("pemoin")


def _load_trajectory_poses(store: ResourceStore) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (poses, frame_indices) from trajectory resource. Returns None if missing."""
    path = store.path_for(ResourceKind.TRAJECTORY)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        c2w = np.asarray(data["camera_to_world"], dtype=np.float64)
        indices = np.asarray(data["frame_indices"], dtype=np.int64)
    return c2w, indices


def _load_gt_trajectory(store: ResourceStore) -> tuple[np.ndarray, np.ndarray] | None:
    """Load ground-truth trajectory from trajectory_gt.npz alongside poses.npz."""
    traj_dir = store.path_for(ResourceKind.TRAJECTORY).parent
    gt_path = traj_dir / "trajectory_gt.npz"
    if not gt_path.exists():
        return None
    with np.load(gt_path, allow_pickle=True) as data:
        c2w = np.asarray(data["camera_to_world"], dtype=np.float64)
        indices = np.asarray(data["frame_indices"], dtype=np.int64)
    return c2w, indices


def _extract_road_points(store: ResourceStore) -> np.ndarray | None:
    """Extract road-labeled points from 3D point cloud."""
    try:
        pc = store.load_point_cloud_3d()
    except Exception:
        return None

    road_label_ids = [
        lid for lid, name in pc.label_names.items()
        if "road" in name.lower()
    ]
    if not road_label_ids:
        return None

    mask = np.isin(pc.labels, road_label_ids)
    if np.sum(mask) < 4:
        return None

    return pc.points_world[mask]


def run_quality_metrics(
    store: ResourceStore,
    settings: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Orchestrate all quality metric categories.

    Parameters
    ----------
    store : ResourceStore with pipeline outputs
    settings : raw settings dict (parsed into QualityMetricsSettings)
    logger : optional logger (defaults to pemoin logger)

    Returns
    -------
    Dict of all computed metric results.
    """
    log = logger or LOG
    config = QualityMetricsSettings.from_mapping(settings)

    if not config.enabled:
        log.debug("Quality metrics disabled, skipping.")
        return {}

    vis_root = store.visualizations_dir() / "quality_metrics"
    vis_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    # --- Trajectory metrics (require GT) ---
    if config.trajectory.enabled:
        try:
            summary["trajectory"] = _run_trajectory_metrics(
                store, config, vis_root, log
            )
        except Exception as exc:
            log.warning("Trajectory metrics failed: %s", exc, exc_info=True)
            summary["trajectory"] = {"error": str(exc)}

    # --- Road metrics ---
    if config.road.enabled:
        try:
            summary["road"] = _run_road_metrics(store, config, vis_root, log)
        except Exception as exc:
            log.warning("Road metrics failed: %s", exc, exc_info=True)
            summary["road"] = {"error": str(exc)}

    # --- Visual artifacts ---
    if config.artifacts.enabled:
        try:
            summary["artifacts"] = _run_artifacts(store, config, vis_root, log)
        except Exception as exc:
            log.warning("Artifact generation failed: %s", exc, exc_info=True)
            summary["artifacts"] = {"error": str(exc)}

    # Write summary
    summary_path = vis_root / "metrics_summary.json"
    _write_summary(summary, summary_path)
    log.info("Quality metrics summary written to %s", summary_path)

    return summary


def _run_trajectory_metrics(
    store: ResourceStore,
    config: QualityMetricsSettings,
    vis_root: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    from pemoin.metrics.trajectory import (
        compute_ate,
        compute_rpe,
        compute_scale_drift,
        visualize_trajectory_metrics,
    )

    est_data = _load_trajectory_poses(store)
    if est_data is None:
        log.info("No trajectory found, skipping trajectory metrics.")
        return {"skipped": "no_trajectory"}

    gt_data = _load_gt_trajectory(store)
    if gt_data is None:
        log.info("No GT trajectory found, skipping trajectory metrics.")
        return {"skipped": "no_gt_trajectory"}

    est_poses, est_indices = est_data
    gt_poses, gt_indices = gt_data

    # Align on common frame indices
    common = np.intersect1d(est_indices, gt_indices)
    if len(common) < 3:
        log.warning("Fewer than 3 common frames between est and GT trajectories.")
        return {"skipped": "insufficient_common_frames", "common_frames": int(len(common))}

    est_mask = np.isin(est_indices, common)
    gt_mask = np.isin(gt_indices, common)
    # Sort both by frame index
    est_order = np.argsort(est_indices[est_mask])
    gt_order = np.argsort(gt_indices[gt_mask])
    est_aligned = est_poses[est_mask][est_order]
    gt_aligned = gt_poses[gt_mask][gt_order]

    traj_cfg = config.trajectory
    result: dict[str, Any] = {"common_frames": int(len(common))}

    # ATE
    ate = compute_ate(
        est_aligned, gt_aligned,
        align=traj_cfg.umeyama_align,
        with_scale=traj_cfg.umeyama_with_scale,
    )
    result["ate"] = {
        "rmse_m": ate.rmse_m,
        "mean_m": ate.mean_m,
        "median_m": ate.median_m,
        "std_m": ate.std_m,
        "max_m": ate.max_m,
    }
    log.info("ATE: RMSE=%.4fm, mean=%.4fm, median=%.4fm", ate.rmse_m, ate.mean_m, ate.median_m)

    # RPE
    rpe_results = []
    result["rpe"] = {}
    for delta in traj_cfg.rpe_deltas:
        if delta >= len(common):
            continue
        rpe = compute_rpe(
            est_aligned, gt_aligned, delta,
            align=traj_cfg.umeyama_align,
            with_scale=traj_cfg.umeyama_with_scale,
        )
        rpe_results.append(rpe)
        result["rpe"][f"delta_{delta}"] = {
            "trans_rmse": rpe.trans_rmse,
            "rot_rmse_deg": rpe.rot_rmse_deg,
        }
        log.info(
            "RPE(delta=%d): trans_rmse=%.4fm, rot_rmse=%.3f°",
            delta, rpe.trans_rmse, rpe.rot_rmse_deg,
        )

    # Scale drift
    scale_drift = None
    if traj_cfg.scale_drift_window <= len(common):
        scale_drift = compute_scale_drift(
            est_aligned, gt_aligned,
            window=traj_cfg.scale_drift_window,
            stride=traj_cfg.scale_drift_stride,
        )
        result["scale_drift"] = {
            "drift_per_100m": scale_drift.drift_per_100m,
            "n_windows": int(scale_drift.scale_factors.size),
        }
        log.info("Scale drift: %.4f per 100m", scale_drift.drift_per_100m)

    # Visualizations
    traj_vis_dir = vis_root / "trajectory"
    plots = visualize_trajectory_metrics(
        traj_vis_dir, ate=ate, rpe_results=rpe_results, scale_drift=scale_drift
    )
    result["plots"] = [str(p) for p in plots]

    return result


def _run_road_metrics(
    store: ResourceStore,
    config: QualityMetricsSettings,
    vis_root: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    from pemoin.metrics.road import (
        compute_normal_stability,
        compute_plane_residuals,
        compute_smoothness,
        visualize_road_metrics,
    )

    result: dict[str, Any] = {}

    # Load road planes
    road_indices = store.frame_indices(ResourceKind.ROAD_PLANE)
    if not road_indices:
        log.info("No road plane data found, skipping road metrics.")
        return {"skipped": "no_road_planes"}

    normals = []
    offsets = []
    for fi in road_indices:
        rp = store.load_road_plane(fi)
        normals.append(rp.normal)
        offsets.append(rp.offset)

    normals_arr = np.array(normals, dtype=np.float64)
    frame_arr = np.array(road_indices, dtype=np.int64)

    # Normal stability
    stability = compute_normal_stability(normals_arr, frame_arr)
    result["normal_stability"] = {
        "mean_angle_deg": stability.mean_angle_deg,
        "p95_angle_deg": stability.p95_angle_deg,
        "p99_angle_deg": stability.p99_angle_deg,
    }
    log.info(
        "Normal stability: mean=%.3f°, p95=%.3f°, p99=%.3f°",
        stability.mean_angle_deg, stability.p95_angle_deg, stability.p99_angle_deg,
    )

    # Plane residuals (need road points from 3D point cloud)
    road_points = _extract_road_points(store)
    residuals = None
    smoothness = None
    if road_points is not None:
        # Use median plane for residuals
        median_normal = np.median(normals_arr, axis=0)
        median_normal /= np.linalg.norm(median_normal)
        median_offset = float(np.median(offsets))

        residuals = compute_plane_residuals(
            road_points, median_normal, median_offset,
            percentiles=config.road.residual_percentiles,
        )
        result["plane_residuals"] = {
            "mean_m": residuals.mean_m,
            "rmse_m": residuals.rmse_m,
            "median_m": residuals.median_m,
            "percentiles": residuals.percentiles,
        }
        log.info(
            "Plane residuals: RMSE=%.4fm, median=%.4fm",
            residuals.rmse_m, residuals.median_m,
        )

        # Smoothness (need trajectory)
        est_data = _load_trajectory_poses(store)
        if est_data is not None:
            traj_pos = est_data[0][:, :3, 3]
            smoothness = compute_smoothness(
                road_points, traj_pos,
                window_size=config.road.smoothness_window,
            )
            result["smoothness"] = {
                "mean_curvature": smoothness.mean_curvature,
                "max_curvature": smoothness.max_curvature,
            }
            log.info(
                "Smoothness: mean=%.4f, max=%.4f",
                smoothness.mean_curvature, smoothness.max_curvature,
            )
    else:
        log.info("No road points in point cloud, skipping residual/smoothness metrics.")

    # Visualizations
    road_vis_dir = vis_root / "road"
    plots = visualize_road_metrics(
        road_vis_dir,
        residuals=residuals,
        stability=stability,
        smoothness=smoothness,
    )
    result["plots"] = [str(p) for p in plots]

    return result


def _run_artifacts(
    store: ResourceStore,
    config: QualityMetricsSettings,
    vis_root: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    from pemoin.metrics.artifacts import (
        generate_confidence_overlay,
        generate_point_cloud_slices,
        generate_reprojection_heatmaps,
        generate_road_model_overlay,
        generate_temporal_flicker,
    )

    artifact_cfg = config.artifacts
    result: dict[str, Any] = {}

    # Load common data
    est_data = _load_trajectory_poses(store)
    has_trajectory = est_data is not None

    try:
        intrinsics = store.load_intrinsics()
        K = intrinsics.matrix
    except Exception:
        log.info("No intrinsics found, skipping pixel-based artifacts.")
        K = None

    try:
        pc = store.load_point_cloud_3d()
    except Exception:
        pc = None

    # Load depth maps for reprojection/flicker
    depths = []
    depth_indices = store.frame_indices(ResourceKind.DEPTH)
    if K is not None and has_trajectory and depth_indices:
        # Only load depths that overlap with trajectory frames
        traj_indices = set(est_data[1].tolist())
        common_depth = sorted(set(depth_indices) & traj_indices)
        for fi in common_depth:
            try:
                dd = store.load_depth(fi)
                depths.append(dd.depth)
            except Exception:
                break

    # Reprojection heatmaps
    if (
        artifact_cfg.reprojection_heatmaps
        and pc is not None
        and has_trajectory
        and K is not None
        and depths
    ):
        try:
            traj_poses, traj_fi = est_data
            # Use only frames with depths
            traj_indices_list = traj_fi.tolist()
            common_depth = sorted(set(depth_indices) & set(traj_indices_list))
            pose_sel = []
            depth_sel = []
            fi_sel = []
            for i, fi in enumerate(common_depth):
                if i < len(depths):
                    tidx = traj_indices_list.index(fi) if fi in traj_indices_list else None
                    if tidx is not None:
                        pose_sel.append(traj_poses[tidx])
                        depth_sel.append(depths[i])
                        fi_sel.append(fi)

            if pose_sel:
                hm_dir = vis_root / "reprojection_heatmaps"
                paths = generate_reprojection_heatmaps(
                    hm_dir,
                    pc.points_world, np.array(pose_sel), K,
                    depth_sel, np.array(fi_sel), artifact_cfg,
                )
                result["reprojection_heatmaps"] = len(paths)
                log.info("Generated %d reprojection heatmaps.", len(paths))
        except Exception as exc:
            log.warning("Reprojection heatmap generation failed: %s", exc)

    # Temporal flicker
    if (
        artifact_cfg.temporal_flicker
        and has_trajectory
        and K is not None
        and depths
    ):
        try:
            traj_poses, traj_fi = est_data
            common_depth = sorted(set(depth_indices) & set(traj_fi.tolist()))
            traj_indices_list = traj_fi.tolist()
            pose_sel = []
            depth_sel = []
            fi_sel = []
            for i, fi in enumerate(common_depth):
                if i < len(depths):
                    tidx = traj_indices_list.index(fi) if fi in traj_indices_list else None
                    if tidx is not None:
                        pose_sel.append(traj_poses[tidx])
                        depth_sel.append(depths[i])
                        fi_sel.append(fi)

            if len(pose_sel) >= 3:
                fl_dir = vis_root / "temporal_flicker"
                paths = generate_temporal_flicker(
                    fl_dir,
                    np.array(pose_sel), K, depth_sel,
                    np.array(fi_sel), artifact_cfg,
                )
                result["temporal_flicker"] = len(paths)
                log.info("Generated %d temporal flicker maps.", len(paths))
        except Exception as exc:
            log.warning("Temporal flicker generation failed: %s", exc)

    # Point cloud slices
    if artifact_cfg.point_cloud_slices and pc is not None and has_trajectory:
        try:
            traj_pos = est_data[0][:, :3, 3]
            sl_dir = vis_root / "point_cloud_slices"
            paths = generate_point_cloud_slices(
                sl_dir, pc.points_world, traj_pos, artifact_cfg,
            )
            result["point_cloud_slices"] = len(paths)
            log.info("Generated %d point cloud slice plots.", len(paths))
        except Exception as exc:
            log.warning("Point cloud slice generation failed: %s", exc)

    # Road model overlay
    if artifact_cfg.road_model_overlay and pc is not None:
        try:
            road_indices = store.frame_indices(ResourceKind.ROAD_PLANE)
            if road_indices:
                rp = store.load_road_plane(road_indices[len(road_indices) // 2])
                road_points = _extract_road_points(store)
                if road_points is not None:
                    ro_dir = vis_root / "road_model_overlay"
                    paths = generate_road_model_overlay(
                        ro_dir, road_points, rp.normal, rp.offset, artifact_cfg,
                    )
                    result["road_model_overlay"] = len(paths)
                    log.info("Generated road model overlay.")
        except Exception as exc:
            log.warning("Road model overlay generation failed: %s", exc)

    # Confidence overlay
    if artifact_cfg.confidence_overlay and pc is not None:
        try:
            co_dir = vis_root / "confidence_overlay"
            paths = generate_confidence_overlay(
                co_dir, pc.points_world, pc.observation_counts, artifact_cfg,
            )
            result["confidence_overlay"] = len(paths)
            log.info("Generated confidence overlay.")
        except Exception as exc:
            log.warning("Confidence overlay generation failed: %s", exc)

    return result


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    """Write metrics summary to JSON, handling numpy types."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_convert(summary), indent=2), encoding="utf-8")
