"""Quadratic road surface model.

Fits z = ax^2 + bxy + cy^2 + dx + ey + f in a local road-aligned frame per frame,
with regularization on curvature and slope terms, plus temporal smoothness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pemoin.data.contracts import PoseSample
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.utils.logging import get_logger

LOG = get_logger()


@dataclass
class QuadraticSurfaceResult:
    """Result of quadratic road surface fit for one frame."""

    frame_index: int
    coeffs: np.ndarray  # [a, b, c, d, e, f] for z = ax^2+bxy+cy^2+dx+ey+f
    local_to_world: np.ndarray  # 4x4 transform from local road frame to world
    confidence: float


def _build_local_frame(
    camera_pos: np.ndarray,
    forward: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Build a 4x4 local road-aligned frame transform.

    Origin: camera ground projection (along plane normal).
    X: forward direction projected onto road plane.
    Y: cross-road direction.
    Z: plane normal (up from road).

    Returns:
        4x4 local-to-world transform.
    """
    n = plane_normal / max(float(np.linalg.norm(plane_normal)), 1e-8)

    # Project forward onto plane
    fwd_proj = forward - float(np.dot(forward, n)) * n
    fwd_norm = float(np.linalg.norm(fwd_proj))
    if fwd_norm < 1e-6:
        # Fallback: use X axis
        fwd_proj = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fwd_proj = fwd_proj - float(np.dot(fwd_proj, n)) * n
        fwd_norm = float(np.linalg.norm(fwd_proj))
    x_axis = fwd_proj / max(fwd_norm, 1e-8)
    y_axis = np.cross(n, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)

    # Origin: project camera onto road plane
    dist_to_plane = float(np.dot(camera_pos, n))
    origin = camera_pos - dist_to_plane * n

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = n
    T[:3, 3] = origin
    return T


def fit_quadratic_surfaces(
    road_points_world_per_frame: list[np.ndarray],
    poses: list[PoseSample],
    plane_normals: list[np.ndarray],
    settings: GeometryFusionSettings,
) -> list[QuadraticSurfaceResult]:
    """Fit per-frame quadratic road surface models.

    For each frame:
    1. Build local road-aligned frame.
    2. Transform road points to local coords.
    3. Fit z = ax^2 + bxy + cy^2 + dx + ey + f with regularization.

    Args:
        road_points_world_per_frame: List of (N_i, 3) world-frame road points per frame.
        poses: Camera poses (C2W) per frame.
        plane_normals: Road plane normals per frame (world coords).
        settings: Geometry fusion settings.

    Returns:
        List of QuadraticSurfaceResult per frame.
    """
    if not settings.quadratic_enabled:
        return []

    n_frames = len(road_points_world_per_frame)
    results: list[QuadraticSurfaceResult] = []
    prev_coeffs: np.ndarray | None = None

    for i in range(n_frames):
        pts_world = road_points_world_per_frame[i]
        pose = poses[i]
        normal = plane_normals[i]
        c2w = pose.camera_to_world

        camera_pos = c2w[:3, 3].astype(np.float64)
        # Forward direction: negative Z in Blender convention (Z-backward)
        forward = -c2w[:3, 2].astype(np.float64)

        local_to_world = _build_local_frame(camera_pos, forward, normal.astype(np.float64))
        world_to_local = np.linalg.inv(local_to_world)

        if pts_world.shape[0] < 6:
            LOG.warning("Frame %d: too few road points (%d) for quadratic fit.", pose.frame_index, pts_world.shape[0])
            coeffs = np.zeros(6, dtype=np.float32)
            results.append(
                QuadraticSurfaceResult(
                    frame_index=pose.frame_index,
                    coeffs=coeffs,
                    local_to_world=local_to_world.astype(np.float32),
                    confidence=0.0,
                )
            )
            continue

        # Transform to local coords
        pts_h = np.hstack([pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float64)])
        pts_local = (world_to_local @ pts_h.T).T[:, :3]

        x_l = pts_local[:, 0]
        y_l = pts_local[:, 1]
        z_l = pts_local[:, 2]

        # Design matrix: [x^2, xy, y^2, x, y, 1]
        A = np.column_stack([
            x_l ** 2,
            x_l * y_l,
            y_l ** 2,
            x_l,
            y_l,
            np.ones_like(x_l),
        ])

        # Regularization matrix
        n_cols = 6
        lambda_curv = float(settings.quadratic_lambda_curv)
        lambda_lin = float(settings.quadratic_lambda_lin)

        reg = np.zeros((n_cols, n_cols), dtype=np.float64)
        reg[0, 0] = lambda_curv  # a (x^2)
        reg[1, 1] = lambda_curv  # b (xy)
        reg[2, 2] = lambda_curv  # c (y^2)
        reg[3, 3] = lambda_lin  # d (x)
        reg[4, 4] = lambda_lin  # e (y)
        # f (constant) is free

        # Augmented system
        ATA = A.T @ A + reg
        ATb = A.T @ z_l
        if prev_coeffs is not None:
            # Temporal regularization
            lambda_temp = lambda_curv * 0.5
            ATA += lambda_temp * np.eye(n_cols, dtype=np.float64)
            ATb += lambda_temp * prev_coeffs.astype(np.float64)

        try:
            coeffs = np.linalg.solve(ATA, ATb).astype(np.float32)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(ATA, ATb, rcond=None)[0].astype(np.float32)

        # Confidence: based on residual quality
        z_pred = A @ coeffs.astype(np.float64)
        residuals = np.abs(z_l - z_pred)
        confidence = float(np.mean(residuals < 0.1))

        prev_coeffs = coeffs

        results.append(
            QuadraticSurfaceResult(
                frame_index=pose.frame_index,
                coeffs=coeffs,
                local_to_world=local_to_world.astype(np.float32),
                confidence=confidence,
            )
        )

    return results
