"""Per-frame road-anchored depth rectification.

Fits a plane to road pixels in camera coordinates, derives per-frame affine
depth correction (scale + optional bias), and optimises temporal smoothness
via L-BFGS-B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pemoin.data.contracts import ResourceStore
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.utils.plane_fitting import (
    ransac_irls_plane_fit,
)
from pemoin.providers.geometry_fusion.utils.road_pixel_selection import select_road_pixels
from pemoin.utils.logging import get_logger

LOG = get_logger()


@dataclass
class FrameRectificationResult:
    """Result of per-frame plane fit and affine depth correction."""

    frame_index: int
    normal_cam: np.ndarray  # (3,) plane normal in camera coords
    offset_cam: float  # d in n·p + d = 0
    implied_height_m: float  # |offset| / |normal_y|
    scale: float  # s_t = h / h_hat
    bias: float  # b_t (0 if scale_only mode)
    inlier_ratio: float
    residual_p90_m: float
    support_count: int


def fit_per_frame_planes(
    resources: ResourceStore,
    frame_indices: list[int],
    K: np.ndarray,
    camera_height_m: float,
    settings: GeometryFusionSettings,
) -> list[FrameRectificationResult]:
    """Fit road planes per frame and compute initial affine correction parameters.

    For each frame:
    1. Select road pixels and backproject to camera coordinates.
    2. RANSAC + IRLS plane fit.
    3. Compute implied height from plane: h_hat = |offset| / |normal[1]|.
    4. Initial scale s_t = camera_height / h_hat, bias b_t = 0.

    Args:
        resources: ResourceStore with depth, semantics, camera_height.
        frame_indices: Sorted frame indices to process.
        K: 3x3 intrinsics matrix.
        camera_height_m: Known camera height in meters.
        settings: Geometry fusion settings.

    Returns:
        List of FrameRectificationResult, one per frame.
    """
    results: list[FrameRectificationResult] = []

    for frame_idx in frame_indices:
        depth_data = resources.load_depth(frame_idx)
        semantics = resources.load_semantics2d(frame_idx)

        selection = select_road_pixels(
            resources=resources,
            depth=depth_data.depth,
            semantics=semantics,
            K=K,
            road_labels=settings.road_labels,
            conf_thresh=settings.road_conf_thresh,
            roi_bottom_frac=settings.roi_bottom_frac,
            z_max_m=settings.z_max_m,
            min_points=settings.min_support_points,
        )

        plane = ransac_irls_plane_fit(
            points=selection.points_cam,
            weights=selection.weights,
            iters=settings.ransac_iters,
            inlier_thresh=settings.inlier_thresh_m,
            huber_delta=settings.huber_delta_plane_m,
            irls_iters=settings.irls_iters,
            seed=int(frame_idx) + 1337,
        )

        # In PEMOIN's standardized Blender camera coordinates, +Y is camera up
        # and the road lies below the camera. The plane equation is n·p + d = 0
        # with camera at the origin. Implied height remains |d| / |n_y|.
        normal = plane.normal
        offset = plane.offset
        n_y = float(abs(normal[1]))
        if n_y < 1e-6:
            LOG.warning(
                "Frame %d: road plane nearly horizontal in camera Y (n_y=%.4f), using fallback.",
                frame_idx,
                n_y,
            )
            n_y = max(n_y, 1e-4)

        h_hat = float(abs(offset)) / n_y
        if not np.isfinite(h_hat) or h_hat <= 1e-6:
            raise RuntimeError(
                f"Geometry fusion frame {frame_idx}: invalid implied height {h_hat}."
            )

        s = camera_height_m / h_hat
        if not np.isfinite(s) or s <= 0.0:
            raise RuntimeError(
                f"Geometry fusion frame {frame_idx}: invalid scale {s}."
            )

        results.append(
            FrameRectificationResult(
                frame_index=frame_idx,
                normal_cam=normal,
                offset_cam=offset,
                implied_height_m=h_hat,
                scale=s,
                bias=0.0,
                inlier_ratio=plane.inlier_ratio,
                residual_p90_m=plane.residual_p90,
                support_count=selection.points_cam.shape[0],
            )
        )

    return results


def optimize_temporal_smoothness(
    results: list[FrameRectificationResult],
    camera_height_m: float,
    settings: GeometryFusionSettings,
) -> list[FrameRectificationResult]:
    """Optimize scale (and optional bias) for temporal smoothness via L-BFGS-B.

    Minimises:
      sum_t  Huber(h_hat(s_t, b_t) - h)
      + lambda_s * sum_t Huber(s_t - s_{t-1}, delta_ds)
      + lambda_b * sum_t Huber(b_t - b_{t-1}, delta_db)  [affine mode only]

    Args:
        results: Per-frame rectification results with initial s_t, b_t.
        camera_height_m: Target camera height.
        settings: Geometry fusion settings.

    Returns:
        Updated results with smoothed scale and bias values.
    """
    from scipy.optimize import minimize

    n = len(results)
    if n <= 1:
        return results

    raw_h_hats = np.array([r.implied_height_m for r in results], dtype=np.float64)
    affine = settings.affine_mode == "affine"
    n_vars = 2 * n if affine else n

    # Initial values
    x0 = np.zeros(n_vars, dtype=np.float64)
    for i, r in enumerate(results):
        x0[i] = r.scale
        if affine:
            x0[n + i] = r.bias

    h = float(camera_height_m)
    lambda_s = float(settings.lambda_s)
    lambda_b = float(settings.lambda_b)
    delta_ds = float(settings.huber_delta_ds)
    delta_db = float(settings.huber_delta_db_m)

    def _huber(x: float, delta: float) -> float:
        ax = abs(x)
        if ax <= delta:
            return 0.5 * x * x
        return delta * (ax - 0.5 * delta)

    def objective(x: np.ndarray) -> float:
        scales = x[:n]
        biases = x[n:2 * n] if affine else np.zeros(n, dtype=np.float64)
        cost = 0.0
        # Data term: how well the corrected height matches target
        for i in range(n):
            h_corrected = scales[i] * raw_h_hats[i] + biases[i]
            cost += _huber(h_corrected - h, 0.05)
        # Scale smoothness
        for i in range(1, n):
            cost += lambda_s * _huber(scales[i] - scales[i - 1], delta_ds)
        # Bias smoothness
        if affine:
            for i in range(1, n):
                cost += lambda_b * _huber(biases[i] - biases[i - 1], delta_db)
        return cost

    # Bounds: scale > 0, bias unconstrained
    bounds = [(0.01, 10.0)] * n
    if affine:
        bounds += [(-5.0, 5.0)] * n

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": settings.lbfgs_maxiter, "ftol": 1e-10},
    )

    opt_scales = result.x[:n]
    opt_biases = result.x[n:2 * n] if affine else np.zeros(n, dtype=np.float64)

    updated: list[FrameRectificationResult] = []
    for i, r in enumerate(results):
        updated.append(
            FrameRectificationResult(
                frame_index=r.frame_index,
                normal_cam=r.normal_cam,
                offset_cam=r.offset_cam,
                implied_height_m=r.implied_height_m,
                scale=float(opt_scales[i]),
                bias=float(opt_biases[i]),
                inlier_ratio=r.inlier_ratio,
                residual_p90_m=r.residual_p90_m,
                support_count=r.support_count,
            )
        )

    return updated
