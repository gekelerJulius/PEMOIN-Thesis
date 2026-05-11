"""Trajectory quality metrics: ATE, RPE, scale drift with Umeyama alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class UmeyamaResult:
    """Result of Umeyama alignment."""

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)


@dataclass
class ATEResult:
    """Absolute Trajectory Error result."""

    rmse_m: float
    mean_m: float
    median_m: float
    std_m: float
    max_m: float
    per_frame_errors: np.ndarray  # (N,)


@dataclass
class RPEResult:
    """Relative Pose Error result for a single delta."""

    delta_frames: int
    trans_rmse: float
    rot_rmse_deg: float
    per_pair_trans_errors: np.ndarray
    per_pair_rot_errors_deg: np.ndarray


@dataclass
class ScaleDriftResult:
    """Scale drift detection result."""

    scale_factors: np.ndarray  # per-window scale factors
    window_centers: np.ndarray  # frame indices at window centers
    drift_per_100m: float  # scale change rate normalized to 100m travel


def align_trajectories_umeyama(
    est_positions: np.ndarray,
    gt_positions: np.ndarray,
    *,
    with_scale: bool = True,
) -> UmeyamaResult:
    """SVD-based Umeyama alignment of estimated positions to ground truth.

    Parameters
    ----------
    est_positions : (N, 3) estimated trajectory positions
    gt_positions : (N, 3) ground-truth trajectory positions
    with_scale : whether to estimate and apply a scale factor

    Returns
    -------
    UmeyamaResult with scale, rotation, translation such that
    gt ≈ scale * R @ est + t
    """
    est = np.asarray(est_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)
    if est.shape != gt.shape or est.ndim != 2 or est.shape[1] != 3:
        raise ValueError(
            f"Position arrays must be (N, 3), got est={est.shape}, gt={gt.shape}."
        )
    n = est.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 points for Umeyama, got {n}.")

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_centered = est - mu_est
    gt_centered = gt - mu_gt

    sigma_est_sq = np.sum(est_centered ** 2) / n
    cov = (gt_centered.T @ est_centered) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        scale = float(np.trace(np.diag(D) @ S) / sigma_est_sq)
    else:
        scale = 1.0

    t = mu_gt - scale * R @ mu_est
    return UmeyamaResult(
        scale=scale,
        rotation=R.astype(np.float64),
        translation=t.astype(np.float64),
    )


def _apply_alignment(positions: np.ndarray, alignment: UmeyamaResult) -> np.ndarray:
    """Apply Umeyama alignment: aligned = scale * R @ pos + t."""
    return (alignment.scale * (alignment.rotation @ positions.T).T + alignment.translation)


def compute_ate(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    *,
    align: bool = True,
    with_scale: bool = True,
) -> ATEResult:
    """Compute Absolute Trajectory Error.

    Parameters
    ----------
    est_poses : (N, 4, 4) estimated camera-to-world matrices
    gt_poses : (N, 4, 4) ground-truth camera-to-world matrices
    align : whether to Umeyama-align before computing errors
    with_scale : whether alignment includes scale correction
    """
    est = np.asarray(est_poses, dtype=np.float64)
    gt = np.asarray(gt_poses, dtype=np.float64)
    if est.ndim != 3 or est.shape[1:] != (4, 4):
        raise ValueError(f"est_poses must be (N, 4, 4), got {est.shape}.")
    if gt.shape != est.shape:
        raise ValueError(f"Shape mismatch: est={est.shape}, gt={gt.shape}.")

    est_pos = est[:, :3, 3]
    gt_pos = gt[:, :3, 3]

    if align and est_pos.shape[0] >= 3:
        alignment = align_trajectories_umeyama(est_pos, gt_pos, with_scale=with_scale)
        est_pos = _apply_alignment(est_pos, alignment)

    errors = np.linalg.norm(est_pos - gt_pos, axis=1)
    return ATEResult(
        rmse_m=float(np.sqrt(np.mean(errors ** 2))),
        mean_m=float(np.mean(errors)),
        median_m=float(np.median(errors)),
        std_m=float(np.std(errors)),
        max_m=float(np.max(errors)),
        per_frame_errors=errors.astype(np.float64),
    )


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Rotation angle in degrees from a 3x3 rotation matrix."""
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_rpe(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    delta_frames: int = 1,
    *,
    align: bool = True,
    with_scale: bool = True,
) -> RPEResult:
    """Compute Relative Pose Error for a given frame delta.

    Parameters
    ----------
    est_poses : (N, 4, 4) estimated camera-to-world matrices
    gt_poses : (N, 4, 4) ground-truth camera-to-world matrices
    delta_frames : frame gap between pose pairs
    """
    est = np.asarray(est_poses, dtype=np.float64)
    gt = np.asarray(gt_poses, dtype=np.float64)
    if est.ndim != 3 or est.shape[1:] != (4, 4):
        raise ValueError(f"est_poses must be (N, 4, 4), got {est.shape}.")
    if gt.shape != est.shape:
        raise ValueError(f"Shape mismatch: est={est.shape}, gt={gt.shape}.")

    n = est.shape[0]
    if delta_frames >= n:
        raise ValueError(
            f"delta_frames={delta_frames} >= number of poses={n}."
        )

    # Optionally align before computing relative errors
    if align and n >= 3:
        est_pos = est[:, :3, 3].copy()
        gt_pos = gt[:, :3, 3].copy()
        alignment = align_trajectories_umeyama(est_pos, gt_pos, with_scale=with_scale)
        aligned_est = est.copy()
        for i in range(n):
            R_est = est[i, :3, :3]
            t_est = est[i, :3, 3]
            aligned_est[i, :3, :3] = alignment.rotation @ R_est
            aligned_est[i, :3, 3] = alignment.scale * (alignment.rotation @ t_est) + alignment.translation
        est = aligned_est

    trans_errors = []
    rot_errors = []
    for i in range(n - delta_frames):
        j = i + delta_frames
        # Relative motion in GT
        gt_rel = np.linalg.inv(gt[i]) @ gt[j]
        # Relative motion in estimated
        est_rel = np.linalg.inv(est[i]) @ est[j]
        # Error transform
        err = np.linalg.inv(gt_rel) @ est_rel
        trans_errors.append(float(np.linalg.norm(err[:3, 3])))
        rot_errors.append(_rotation_angle_deg(err[:3, :3]))

    trans_arr = np.array(trans_errors)
    rot_arr = np.array(rot_errors)
    return RPEResult(
        delta_frames=delta_frames,
        trans_rmse=float(np.sqrt(np.mean(trans_arr ** 2))),
        rot_rmse_deg=float(np.sqrt(np.mean(rot_arr ** 2))),
        per_pair_trans_errors=trans_arr,
        per_pair_rot_errors_deg=rot_arr,
    )


def compute_scale_drift(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    window: int = 20,
    stride: int = 5,
) -> ScaleDriftResult:
    """Detect scale drift by comparing local scale factors over sliding windows.

    For each window of consecutive poses, computes the ratio of estimated
    to ground-truth path lengths. Drift is measured as scale change per 100m
    of GT travel.
    """
    est = np.asarray(est_poses, dtype=np.float64)
    gt = np.asarray(gt_poses, dtype=np.float64)
    n = est.shape[0]

    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}.")
    if window > n:
        raise ValueError(f"window={window} > number of poses={n}.")

    est_pos = est[:, :3, 3]
    gt_pos = gt[:, :3, 3]

    scale_factors = []
    centers = []

    for start in range(0, n - window + 1, stride):
        end = start + window
        est_segment = est_pos[start:end]
        gt_segment = gt_pos[start:end]

        est_dists = np.linalg.norm(np.diff(est_segment, axis=0), axis=1)
        gt_dists = np.linalg.norm(np.diff(gt_segment, axis=0), axis=1)

        est_length = float(np.sum(est_dists))
        gt_length = float(np.sum(gt_dists))

        if gt_length < 1e-8:
            continue

        scale_factors.append(est_length / gt_length)
        centers.append(start + window // 2)

    scale_arr = np.array(scale_factors)
    centers_arr = np.array(centers)

    # Compute drift rate: scale change per 100m of GT travel
    if len(scale_factors) >= 2:
        gt_cumulative = np.cumsum(
            np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
        )
        gt_cumulative = np.concatenate([[0.0], gt_cumulative])
        center_distances = gt_cumulative[centers_arr]
        total_travel = center_distances[-1] - center_distances[0]
        if total_travel > 1e-3:
            scale_change = abs(scale_arr[-1] - scale_arr[0])
            drift_per_100m = float(scale_change / total_travel * 100.0)
        else:
            drift_per_100m = 0.0
    else:
        drift_per_100m = 0.0

    return ScaleDriftResult(
        scale_factors=scale_arr,
        window_centers=centers_arr,
        drift_per_100m=drift_per_100m,
    )


def visualize_trajectory_metrics(
    output_dir: Path,
    ate: ATEResult | None = None,
    rpe_results: Sequence[RPEResult] | None = None,
    scale_drift: ScaleDriftResult | None = None,
) -> list[Path]:
    """Generate PNG plots for trajectory metrics. Returns paths of created files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    if ate is not None and ate.per_frame_errors.size > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ate.per_frame_errors, linewidth=0.8)
        ax.axhline(ate.rmse_m, color="r", linestyle="--", label=f"RMSE={ate.rmse_m:.4f}m")
        ax.set_xlabel("Frame")
        ax.set_ylabel("ATE (m)")
        ax.set_title("Absolute Trajectory Error per Frame")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = output_dir / "ate_per_frame.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    if rpe_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        deltas = [r.delta_frames for r in rpe_results]
        trans_rmses = [r.trans_rmse for r in rpe_results]
        rot_rmses = [r.rot_rmse_deg for r in rpe_results]

        ax1.bar(range(len(deltas)), trans_rmses, tick_label=[str(d) for d in deltas])
        ax1.set_xlabel("Delta (frames)")
        ax1.set_ylabel("Trans RMSE (m)")
        ax1.set_title("RPE Translation vs Delta")
        ax1.grid(True, alpha=0.3)

        ax2.bar(range(len(deltas)), rot_rmses, tick_label=[str(d) for d in deltas])
        ax2.set_xlabel("Delta (frames)")
        ax2.set_ylabel("Rot RMSE (deg)")
        ax2.set_title("RPE Rotation vs Delta")
        ax2.grid(True, alpha=0.3)

        path = output_dir / "rpe_vs_delta.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    if scale_drift is not None and scale_drift.scale_factors.size > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(scale_drift.window_centers, scale_drift.scale_factors, "o-", markersize=3)
        ax.axhline(1.0, color="r", linestyle="--", alpha=0.5, label="Ideal scale=1.0")
        ax.set_xlabel("Frame (window center)")
        ax.set_ylabel("Local Scale Factor (est/gt)")
        ax.set_title(f"Scale Drift (drift/100m = {scale_drift.drift_per_100m:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = output_dir / "scale_drift.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    return created
