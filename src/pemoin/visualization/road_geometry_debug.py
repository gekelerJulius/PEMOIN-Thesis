"""Road-geometry debug plots and summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def write_road_geometry_debug_artifacts(
    *,
    output_dir: Path,
    frame_indices: Sequence[int],
    camera_height_m: Sequence[float],
    plane_height_at_camera_m: Sequence[float],
    fit_residual_p90_m: Sequence[float],
    saved_residual_p90_m: Sequence[float],
    support_layering_ratio: Sequence[float] | None = None,
    support_depth_spread_p90_m: Sequence[float] | None = None,
    adaptive_forward_min_used_m: Sequence[float] | None = None,
    window_included_frame_count: Sequence[float] | None = None,
    recovery_fit_used: Sequence[float] | None = None,
    recovery_fit_accepted: Sequence[float] | None = None,
) -> Path:
    """Write road-geometry diagnostics to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    cam_h = np.asarray(camera_height_m, dtype=np.float32).reshape(-1)
    plane_h = np.asarray(plane_height_at_camera_m, dtype=np.float32).reshape(-1)
    fit_p90 = np.asarray(fit_residual_p90_m, dtype=np.float32).reshape(-1)
    saved_p90 = np.asarray(saved_residual_p90_m, dtype=np.float32).reshape(-1)
    if not (frames.size == cam_h.size == plane_h.size == fit_p90.size == saved_p90.size):
        raise ValueError("Road geometry debug series length mismatch.")
    if support_layering_ratio is not None:
        layer = np.asarray(support_layering_ratio, dtype=np.float32).reshape(-1)
        if layer.size != frames.size:
            raise ValueError("support_layering_ratio length mismatch.")
    else:
        layer = None
    if support_depth_spread_p90_m is not None:
        spread = np.asarray(support_depth_spread_p90_m, dtype=np.float32).reshape(-1)
        if spread.size != frames.size:
            raise ValueError("support_depth_spread_p90_m length mismatch.")
    else:
        spread = None
    adaptive_forward = None
    if adaptive_forward_min_used_m is not None:
        adaptive_forward = np.asarray(adaptive_forward_min_used_m, dtype=np.float32).reshape(-1)
        if adaptive_forward.size != frames.size:
            raise ValueError("adaptive_forward_min_used_m length mismatch.")
    window_size = None
    if window_included_frame_count is not None:
        window_size = np.asarray(window_included_frame_count, dtype=np.float32).reshape(-1)
        if window_size.size != frames.size:
            raise ValueError("window_included_frame_count length mismatch.")
    recovery_used_arr = None
    if recovery_fit_used is not None:
        recovery_used_arr = np.asarray(recovery_fit_used, dtype=np.float32).reshape(-1)
        if recovery_used_arr.size != frames.size:
            raise ValueError("recovery_fit_used length mismatch.")
    recovery_accepted_arr = None
    if recovery_fit_accepted is not None:
        recovery_accepted_arr = np.asarray(recovery_fit_accepted, dtype=np.float32).reshape(-1)
        if recovery_accepted_arr.size != frames.size:
            raise ValueError("recovery_fit_accepted length mismatch.")

    abs_height_delta = np.abs(cam_h - plane_h)
    abs_residual_delta = np.abs(fit_p90 - saved_p90)
    summary = {
        "frame_count": int(frames.size),
        "checks": {
            "plane_height_matches_camera_height": bool(np.percentile(abs_height_delta, 90) < 0.20),
            "fit_vs_saved_residuals_consistent": bool(np.percentile(abs_residual_delta, 90) < 0.05),
        },
        "metrics": {
            "height_delta_median_m": float(np.median(abs_height_delta)) if abs_height_delta.size else 0.0,
            "height_delta_p90_m": float(np.percentile(abs_height_delta, 90)) if abs_height_delta.size else 0.0,
            "residual_p90_delta_median_m": float(np.median(abs_residual_delta)) if abs_residual_delta.size else 0.0,
            "residual_p90_delta_p90_m": float(np.percentile(abs_residual_delta, 90)) if abs_residual_delta.size else 0.0,
            "support_layering_ratio_median": float(np.median(layer)) if layer is not None and layer.size else 0.0,
            "support_layering_ratio_p90": float(np.percentile(layer, 90)) if layer is not None and layer.size else 0.0,
            "support_depth_spread_p90_median_m": float(np.median(spread)) if spread is not None and spread.size else 0.0,
            "support_depth_spread_p90_p90_m": float(np.percentile(spread, 90)) if spread is not None and spread.size else 0.0,
            "adaptive_forward_min_used_median_m": float(np.median(adaptive_forward)) if adaptive_forward is not None and adaptive_forward.size else 0.0,
            "window_included_frame_count_median": float(np.median(window_size)) if window_size is not None and window_size.size else 0.0,
            "recovery_fit_used_count": int(np.count_nonzero(recovery_used_arr > 0.5)) if recovery_used_arr is not None else 0,
            "recovery_fit_accepted_count": int(np.count_nonzero(recovery_accepted_arr > 0.5)) if recovery_accepted_arr is not None else 0,
        },
        "plots": {},
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(frames, cam_h, label="camera_height_m", linewidth=1.2)
        ax.plot(frames, plane_h, label="plane_height_at_camera_m", linewidth=1.2)
        ax.set_title("Camera Height vs Plane Height")
        ax.set_xlabel("frame")
        ax.set_ylabel("meters")
        ax.grid(alpha=0.25)
        ax.legend()
        trend_path = output_dir / "plane_vs_camera_height.png"
        fig.savefig(trend_path, dpi=140)
        plt.close(fig)
        summary["plots"]["plane_vs_camera_height"] = str(trend_path)

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(frames, fit_p90, label="fit_residual_p90_m", linewidth=1.2)
        ax.plot(frames, saved_p90, label="saved_residual_p90_m", linewidth=1.2)
        ax.set_title("Fit Residual p90 vs Saved Plane Residual p90")
        ax.set_xlabel("frame")
        ax.set_ylabel("meters")
        ax.grid(alpha=0.25)
        ax.legend()
        residual_path = output_dir / "fit_vs_saved_residual_p90.png"
        fig.savefig(residual_path, dpi=140)
        plt.close(fig)
        summary["plots"]["fit_vs_saved_residual_p90"] = str(residual_path)

        if layer is not None and spread is not None:
            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            ax.plot(frames, layer, label="support_layering_ratio", linewidth=1.2)
            ax.set_title("Support Layering Ratio")
            ax.set_xlabel("frame")
            ax.set_ylabel("ratio")
            ax.grid(alpha=0.25)
            ax.legend()
            layer_path = output_dir / "support_layering_ratio.png"
            fig.savefig(layer_path, dpi=140)
            plt.close(fig)
            summary["plots"]["support_layering_ratio"] = str(layer_path)

            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            ax.plot(frames, spread, label="support_depth_spread_p90_m", linewidth=1.2)
            ax.set_title("Support Depth Spread (p90)")
            ax.set_xlabel("frame")
            ax.set_ylabel("meters")
            ax.grid(alpha=0.25)
            ax.legend()
            spread_path = output_dir / "support_depth_spread_p90.png"
            fig.savefig(spread_path, dpi=140)
            plt.close(fig)
            summary["plots"]["support_depth_spread_p90"] = str(spread_path)

        if adaptive_forward is not None:
            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            ax.plot(frames, adaptive_forward, label="adaptive_forward_min_used_m", linewidth=1.2)
            ax.set_title("Adaptive Forward-Min Used")
            ax.set_xlabel("frame")
            ax.set_ylabel("meters")
            ax.grid(alpha=0.25)
            ax.legend()
            adaptive_path = output_dir / "adaptive_forward_min_vs_frame.png"
            fig.savefig(adaptive_path, dpi=140)
            plt.close(fig)
            summary["plots"]["adaptive_forward_min_vs_frame"] = str(adaptive_path)

        if window_size is not None:
            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            ax.plot(frames, window_size, label="window_included_frame_count", linewidth=1.2)
            ax.set_title("Effective Window Size")
            ax.set_xlabel("frame")
            ax.set_ylabel("frames")
            ax.grid(alpha=0.25)
            ax.legend()
            window_path = output_dir / "window_effective_size_vs_frame.png"
            fig.savefig(window_path, dpi=140)
            plt.close(fig)
            summary["plots"]["window_effective_size_vs_frame"] = str(window_path)

        if recovery_used_arr is not None and recovery_accepted_arr is not None:
            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            ax.plot(frames, recovery_used_arr, label="recovery_fit_used", linewidth=1.2)
            ax.plot(frames, recovery_accepted_arr, label="recovery_fit_accepted", linewidth=1.2)
            ax.set_title("Recovery Fit Events")
            ax.set_xlabel("frame")
            ax.set_ylabel("flag")
            ax.grid(alpha=0.25)
            ax.legend()
            recovery_path = output_dir / "recovery_fit_events.png"
            fig.savefig(recovery_path, dpi=140)
            plt.close(fig)
            summary["plots"]["recovery_fit_events"] = str(recovery_path)

        early = min(5, frames.size)
        if early > 0:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            ax.hist(cam_h[:early], bins=16, alpha=0.6, label="camera_height_m")
            ax.hist(plane_h[:early], bins=16, alpha=0.6, label="plane_height_at_camera_m")
            ax.set_title("First 5 Frames Height Distribution")
            ax.set_xlabel("meters")
            ax.set_ylabel("count")
            ax.legend()
            hist_path = output_dir / "first5_height_histogram.png"
            fig.savefig(hist_path, dpi=140)
            plt.close(fig)
            summary["plots"]["first5_height_histogram"] = str(hist_path)
    except Exception as exc:
        summary["plots_error"] = str(exc)

    summary_path = output_dir / "road_geometry_debug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
