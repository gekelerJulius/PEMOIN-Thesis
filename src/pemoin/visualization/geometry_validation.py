"""Geometry validation visualization artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from pemoin.data.contracts import ResourceStore
from pemoin.utils.geometry_validation import GeometryValidationReport
from pemoin.visualization.debug_artifacts import save_rgb_image

LOG = logging.getLogger(__name__)


def write_geometry_validation_visualizations(
    store: ResourceStore,
    report: GeometryValidationReport,
    *,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Write geometry validation metrics and plots into run visualizations."""
    log = logger or LOG
    out_dir = store.visualizations_dir("geometry_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(report.to_summary(), indent=2), encoding="utf-8")
    outputs.append(summary_path)

    outputs.extend(_write_metric_plots(out_dir, report, log))
    outputs.extend(_write_motion_plots(out_dir, report, log))
    outputs.extend(_write_reprojection_overlays(store, out_dir / "reprojection", report, log))

    ply_path = _write_trajectory_point_cloud(out_dir / "trajectory_points.ply", report)
    if ply_path is not None:
        outputs.append(ply_path)

    log.info("Geometry validation visualizations written to %s (%d artifact(s)).", out_dir, len(outputs))
    return outputs


def _write_metric_plots(out_dir: Path, report: GeometryValidationReport, logger: logging.Logger) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.debug("matplotlib unavailable; skipping geometry metric plots.")
        return []

    if not report.frame_metrics:
        return []

    frames = np.asarray([m.frame_index for m in report.frame_metrics], dtype=np.int32)
    inverse_left = np.asarray([m.inverse_error_left_fro for m in report.frame_metrics], dtype=np.float32)
    inverse_right = np.asarray([m.inverse_error_right_fro for m in report.frame_metrics], dtype=np.float32)
    rot_orth = np.asarray([m.rotation_orthonormal_error for m in report.frame_metrics], dtype=np.float32)
    reproj_rmse = np.asarray([m.reprojection_rmse_px for m in report.frame_metrics], dtype=np.float32)
    reproj_inlier = np.asarray([m.reprojection_inlier_ratio for m in report.frame_metrics], dtype=np.float32)
    positive_depth = np.asarray([m.positive_depth_ratio for m in report.frame_metrics], dtype=np.float32)
    front_ratio = np.asarray([m.front_ratio for m in report.frame_metrics], dtype=np.float32)

    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, inverse_left, label="||c2w@w2c-I||F", linewidth=1.2)
    ax.plot(frames, inverse_right, label="||w2c@c2w-I||F", linewidth=1.2)
    ax.plot(frames, rot_orth, label="max|R^TR-I|", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("frame")
    ax.set_ylabel("error")
    ax.set_title("Pose Consistency Errors")
    ax.legend()
    fig.tight_layout()
    pose_path = out_dir / "pose_consistency.png"
    fig.savefig(pose_path, dpi=150)
    plt.close(fig)
    outputs.append(pose_path)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, reproj_rmse, label="reprojection_rmse_px", linewidth=1.2)
    ax.plot(frames, reproj_inlier, label="reprojection_inlier_ratio", linewidth=1.2)
    ax.plot(frames, positive_depth, label="positive_depth_ratio", linewidth=1.2)
    ax.plot(frames, front_ratio, label="front_ratio", linewidth=1.2)
    ax.set_xlabel("frame")
    ax.set_ylabel("value")
    ax.set_title("Depth/Reprojection Metrics")
    ax.legend()
    fig.tight_layout()
    depth_path = out_dir / "depth_reprojection_metrics.png"
    fig.savefig(depth_path, dpi=150)
    plt.close(fig)
    outputs.append(depth_path)

    return outputs


def _write_motion_plots(out_dir: Path, report: GeometryValidationReport, logger: logging.Logger) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.debug("matplotlib unavailable; skipping geometry motion plots.")
        return []

    if not report.motion_metrics:
        return []

    frames = np.asarray([m.frame_index for m in report.motion_metrics], dtype=np.int32)
    translation = np.asarray([m.translation_m for m in report.motion_metrics], dtype=np.float32)
    rotation = np.asarray([m.rotation_deg for m in report.motion_metrics], dtype=np.float32)
    alignment = np.asarray(
        [np.nan if m.view_motion_alignment is None else float(m.view_motion_alignment) for m in report.motion_metrics],
        dtype=np.float32,
    )

    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, translation, label="translation_m", linewidth=1.2)
    ax.plot(frames, rotation, label="rotation_deg", linewidth=1.2)
    ax.set_xlabel("frame")
    ax.set_ylabel("value")
    ax.set_title("Relative Motion")
    ax.legend()
    fig.tight_layout()
    rel_path = out_dir / "relative_motion.png"
    fig.savefig(rel_path, dpi=150)
    plt.close(fig)
    outputs.append(rel_path)

    if not np.all(np.isnan(alignment)):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(frames, alignment, label="view_motion_alignment", linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("frame")
        ax.set_ylabel("cosine")
        ax.set_title("View-Motion Alignment")
        ax.legend()
        fig.tight_layout()
        align_path = out_dir / "view_motion_alignment.png"
        fig.savefig(align_path, dpi=150)
        plt.close(fig)
        outputs.append(align_path)

    return outputs


def _write_reprojection_overlays(
    store: ResourceStore,
    out_dir: Path,
    report: GeometryValidationReport,
    logger: logging.Logger,
) -> list[Path]:
    if not report.reprojection_overlays:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for overlay in report.reprojection_overlays:
        try:
            frame = store.load_frame(int(overlay.frame_index))
            if frame.image is None:
                continue
            image = np.asarray(frame.image, dtype=np.uint8).copy()
            _draw_error_points(image, overlay.xs, overlay.ys, overlay.errors_px)
            path = out_dir / f"{int(overlay.frame_index):06d}.png"
            save_rgb_image(path, image)
            outputs.append(path)
        except Exception as exc:
            logger.warning(
                "Failed to write reprojection overlay for frame %s: %s",
                overlay.frame_index,
                exc,
            )

    return outputs


def _draw_error_points(image: np.ndarray, xs: np.ndarray, ys: np.ndarray, errors: np.ndarray) -> None:
    h, w = image.shape[:2]
    if errors.size == 0:
        return

    max_err = float(np.percentile(errors, 95)) if errors.size > 1 else float(errors[0])
    max_err = max(max_err, 1e-6)

    for x, y, err in zip(xs.astype(int), ys.astype(int), errors.astype(float), strict=False):
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        t = float(np.clip(err / max_err, 0.0, 1.0))
        color = np.array([255.0 * t, 255.0 * (1.0 - t), 0.0], dtype=np.float32)
        image[y, x, :3] = np.clip(0.4 * image[y, x, :3] + 0.6 * color, 0.0, 255.0).astype(np.uint8)


def _write_trajectory_point_cloud(path: Path, report: GeometryValidationReport) -> Path | None:
    if not report.frame_metrics:
        return None

    points = np.asarray(
        [
            [m.camera_position_x, m.camera_position_y, m.camera_position_z]
            for m in report.frame_metrics
        ],
        dtype=np.float32,
    )
    if points.size == 0:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    return path
