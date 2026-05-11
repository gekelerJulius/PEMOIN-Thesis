"""Diagnostics helpers for road-plane quality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneQuality:
    residual_median: float
    residual_p90: float
    inlier_ratio: float
    fit_point_count: int

    def to_metadata(self) -> dict:
        return {
            "residual_median": float(self.residual_median),
            "residual_p90": float(self.residual_p90),
            "inlier_ratio": float(self.inlier_ratio),
            "fit_point_count": float(self.fit_point_count),
        }


def compute_plane_quality(
    points: np.ndarray,
    normal: np.ndarray,
    plane_offset_d: float,
    *,
    inlier_threshold_m: float,
) -> tuple[np.ndarray, PlaneQuality]:
    """Compute residuals and robust quality metrics for a plane."""
    residuals = np.abs(np.asarray(points, dtype=np.float32) @ np.asarray(normal, dtype=np.float32) + float(plane_offset_d))
    if residuals.size == 0:
        quality = PlaneQuality(0.0, 0.0, 0.0, 0)
        return residuals.astype(np.float32), quality
    quality = PlaneQuality(
        residual_median=float(np.median(residuals)),
        residual_p90=float(np.percentile(residuals, 90)),
        inlier_ratio=float(np.mean(residuals <= float(inlier_threshold_m))),
        fit_point_count=int(points.shape[0]),
    )
    return residuals.astype(np.float32), quality


def assert_plane_residual_metadata_consistency(
    *,
    residuals: np.ndarray,
    metadata_residual_median: float,
    metadata_residual_p90: float,
    tolerance_m: float,
) -> None:
    """Raise if plane residual metadata diverges from computed residuals."""
    if residuals.size == 0:
        return
    computed_median = float(np.median(residuals))
    computed_p90 = float(np.percentile(residuals, 90))
    if abs(computed_median - float(metadata_residual_median)) > float(tolerance_m):
        raise RuntimeError(
            "Road-plane residual metadata mismatch for median: "
            f"computed={computed_median:.6f} metadata={float(metadata_residual_median):.6f} "
            f"tol={float(tolerance_m):.6f}."
        )
    if abs(computed_p90 - float(metadata_residual_p90)) > float(tolerance_m):
        raise RuntimeError(
            "Road-plane residual metadata mismatch for p90: "
            f"computed={computed_p90:.6f} metadata={float(metadata_residual_p90):.6f} "
            f"tol={float(tolerance_m):.6f}."
        )

