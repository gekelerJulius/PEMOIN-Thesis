"""Road model quality metrics: plane residuals, normal stability, smoothness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass
class PlaneResidualResult:
    """Plane fit residual statistics."""

    mean_m: float
    rmse_m: float
    median_m: float
    percentiles: Mapping[float, float]  # percentile → value in meters


@dataclass
class NormalStabilityResult:
    """Normal stability across frames."""

    mean_angle_deg: float
    p95_angle_deg: float
    p99_angle_deg: float
    per_frame_angles_deg: np.ndarray  # (N-1,) angle between consecutive normals


@dataclass
class SmoothnessResult:
    """Road surface smoothness (curvature) statistics."""

    mean_curvature: float
    max_curvature: float
    per_window_curvatures: np.ndarray


def compute_plane_residuals(
    points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    *,
    percentiles: Sequence[float] = (50.0, 90.0, 95.0, 99.0),
) -> PlaneResidualResult:
    """Compute signed-distance residual statistics for points against a plane.

    Plane equation: n^T x + d = 0.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        raise ValueError("Degenerate plane normal.")
    n = n / n_norm
    d = float(offset) / n_norm

    distances = np.abs(pts @ n + d)
    pct_values = np.percentile(distances, list(percentiles))
    pct_map = {float(p): float(v) for p, v in zip(percentiles, pct_values)}

    return PlaneResidualResult(
        mean_m=float(np.mean(distances)),
        rmse_m=float(np.sqrt(np.mean(distances ** 2))),
        median_m=float(np.median(distances)),
        percentiles=pct_map,
    )


def compute_normal_stability(
    normals: np.ndarray,
    frame_indices: np.ndarray | None = None,
) -> NormalStabilityResult:
    """Measure angular change between consecutive per-frame road normals.

    Parameters
    ----------
    normals : (N, 3) unit normals, one per frame
    frame_indices : optional (N,) frame indices for sorting
    """
    n = np.asarray(normals, dtype=np.float64)
    if n.ndim != 2 or n.shape[1] != 3:
        raise ValueError(f"normals must be (N, 3), got {n.shape}.")
    if n.shape[0] < 2:
        return NormalStabilityResult(
            mean_angle_deg=0.0,
            p95_angle_deg=0.0,
            p99_angle_deg=0.0,
            per_frame_angles_deg=np.array([], dtype=np.float64),
        )

    if frame_indices is not None:
        order = np.argsort(frame_indices)
        n = n[order]

    # Normalize
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    n = n / norms

    # Angle between consecutive normals
    dots = np.sum(n[:-1] * n[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))

    return NormalStabilityResult(
        mean_angle_deg=float(np.mean(angles)),
        p95_angle_deg=float(np.percentile(angles, 95.0)),
        p99_angle_deg=float(np.percentile(angles, 99.0)),
        per_frame_angles_deg=angles,
    )


def compute_smoothness(
    road_points: np.ndarray,
    trajectory_positions: np.ndarray,
    window_size: int = 10,
) -> SmoothnessResult:
    """Measure road surface smoothness by local plane-fit residuals along trajectory.

    Divides road points into spatial windows along the trajectory and computes
    per-window plane fit residual as a proxy for curvature.
    """
    pts = np.asarray(road_points, dtype=np.float64)
    traj = np.asarray(trajectory_positions, dtype=np.float64)

    if pts.shape[0] < 4:
        return SmoothnessResult(
            mean_curvature=0.0, max_curvature=0.0, per_window_curvatures=np.array([])
        )

    # Project road points onto nearest trajectory segment
    # Use trajectory positions as window centers
    n_windows = max(1, len(traj) // window_size)
    window_indices = np.linspace(0, len(traj) - 1, n_windows + 1, dtype=int)

    curvatures: list[float] = []
    for i in range(len(window_indices) - 1):
        start_pos = traj[window_indices[i]]
        end_pos = traj[window_indices[i + 1]]
        center = (start_pos + end_pos) / 2.0
        radius = np.linalg.norm(end_pos - start_pos) / 2.0 + 1.0

        dists_to_center = np.linalg.norm(pts - center, axis=1)
        mask = dists_to_center < radius
        local_pts = pts[mask]

        if local_pts.shape[0] < 4:
            continue

        # Fit plane via SVD
        centroid = local_pts.mean(axis=0)
        centered = local_pts - centroid
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[2]
        residuals = np.abs(centered @ normal)
        curvatures.append(float(np.sqrt(np.mean(residuals ** 2))))

    curv_arr = np.array(curvatures)
    return SmoothnessResult(
        mean_curvature=float(np.mean(curv_arr)) if curv_arr.size > 0 else 0.0,
        max_curvature=float(np.max(curv_arr)) if curv_arr.size > 0 else 0.0,
        per_window_curvatures=curv_arr,
    )


def visualize_road_metrics(
    output_dir: Path,
    residuals: PlaneResidualResult | None = None,
    stability: NormalStabilityResult | None = None,
    smoothness: SmoothnessResult | None = None,
) -> list[Path]:
    """Generate PNG plots for road metrics. Returns paths of created files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    if residuals is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        pcts = sorted(residuals.percentiles.keys())
        vals = [residuals.percentiles[p] for p in pcts]
        ax.bar(
            range(len(pcts) + 2),
            [residuals.mean_m, residuals.median_m] + vals,
            tick_label=["Mean", "Median"] + [f"P{p:.0f}" for p in pcts],
        )
        ax.set_ylabel("Residual (m)")
        ax.set_title(f"Plane Residuals (RMSE={residuals.rmse_m:.4f}m)")
        ax.grid(True, alpha=0.3)
        path = output_dir / "plane_residuals.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    if stability is not None and stability.per_frame_angles_deg.size > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(stability.per_frame_angles_deg, linewidth=0.8)
        ax.axhline(
            stability.mean_angle_deg, color="r", linestyle="--",
            label=f"Mean={stability.mean_angle_deg:.3f}°",
        )
        ax.set_xlabel("Frame Pair")
        ax.set_ylabel("Angle Change (deg)")
        ax.set_title("Normal Stability (consecutive frame angle change)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = output_dir / "normal_stability.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    if smoothness is not None and smoothness.per_window_curvatures.size > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(smoothness.per_window_curvatures, "o-", markersize=3)
        ax.axhline(
            smoothness.mean_curvature, color="r", linestyle="--",
            label=f"Mean={smoothness.mean_curvature:.4f}",
        )
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Local Plane RMSE (m)")
        ax.set_title("Road Surface Smoothness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = output_dir / "smoothness.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    return created
