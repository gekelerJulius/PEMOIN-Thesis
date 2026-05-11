"""Visualization helpers for depth-pose-intrinsics consistency diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save_series_plot(
    path: Path,
    *,
    x: Sequence[int],
    y: Sequence[float],
    title: str,
    ylabel: str,
    threshold: float | None = None,
    highlighted_x: Sequence[int] | None = None,
    marked_frames: Sequence[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xx = np.asarray(list(x), dtype=np.int32)
    yy = np.asarray(list(y), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 4))
    if xx.size:
        ax.plot(xx, yy, marker="o", linewidth=1.2)
    if highlighted_x:
        highlight_set = {int(v) for v in highlighted_x}
        mask = np.asarray([int(v) in highlight_set for v in xx], dtype=bool)
        if np.any(mask):
            ax.scatter(xx[mask], yy[mask], color="red", s=36, zorder=3, label="catastrophic pair")
    if marked_frames:
        marked = sorted({int(v) for v in marked_frames})
        for frame in marked:
            ax.axvline(float(frame), color="orange", linestyle=":", linewidth=0.9, alpha=0.8)
    if threshold is not None:
        ax.axhline(float(threshold), color="red", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if highlighted_x:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_geometry_consistency_artifacts(
    out_dir: Path,
    *,
    pair_frames: Sequence[int],
    reproj_rmse_px: Sequence[float],
    reproj_p90_px: Sequence[float],
    reproj_p95_px: Sequence[float],
    inlier_ratio: Sequence[float],
    depth_scale: Sequence[float],
    static_overlap_points: Sequence[float],
    catastrophic_pair_frames: Sequence[int],
    replaced_frames: Sequence[int],
    summary: Mapping[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_series_plot(
        out_dir / "pairwise_reprojection_rmse.png",
        x=pair_frames,
        y=reproj_rmse_px,
        title="Pairwise Reprojection RMSE",
        ylabel="RMSE (px)",
        threshold=float(summary.get("threshold_reprojection_rmse_px", np.nan)),
        highlighted_x=catastrophic_pair_frames,
        marked_frames=replaced_frames,
    )
    _save_series_plot(
        out_dir / "pairwise_reprojection_p90.png",
        x=pair_frames,
        y=reproj_p90_px,
        title="Pairwise Reprojection P90",
        ylabel="P90 (px)",
        threshold=float(summary.get("threshold_reprojection_p90_px", np.nan)),
        highlighted_x=catastrophic_pair_frames,
        marked_frames=replaced_frames,
    )
    _save_series_plot(
        out_dir / "pairwise_reprojection_p95.png",
        x=pair_frames,
        y=reproj_p95_px,
        title="Pairwise Reprojection P95",
        ylabel="P95 (px)",
        threshold=float(summary.get("threshold_reprojection_p95_px", np.nan)),
        highlighted_x=catastrophic_pair_frames,
        marked_frames=replaced_frames,
    )
    _save_series_plot(
        out_dir / "pairwise_inlier_ratio.png",
        x=pair_frames,
        y=inlier_ratio,
        title="Pairwise Reprojection Inlier Ratio",
        ylabel="Inlier Ratio",
        threshold=float(summary.get("threshold_min_inlier_ratio", np.nan)),
        highlighted_x=catastrophic_pair_frames,
        marked_frames=replaced_frames,
    )
    _save_series_plot(
        out_dir / "depth_scale_drift.png",
        x=pair_frames,
        y=depth_scale,
        title="Pairwise Depth Scale Drift",
        ylabel="Scale t->t+1",
        threshold=None,
        highlighted_x=catastrophic_pair_frames,
        marked_frames=replaced_frames,
    )
    _save_series_plot(
        out_dir / "pairwise_static_overlap.png",
        x=pair_frames,
        y=static_overlap_points,
        title="Pairwise Static Overlap",
        ylabel="Static Points",
        threshold=float(summary.get("threshold_min_static_overlap_points", np.nan)),
        highlighted_x=catastrophic_pair_frames,
        marked_frames=replaced_frames,
    )
    (out_dir / "summary.json").write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
