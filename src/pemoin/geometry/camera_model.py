"""Canonical camera geometry operations.

All camera projection/backprojection code in PEMOIN should use this module.
The equations are defined once here to avoid convention drift across providers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .conventions import (
    CAMERA_CONVENTION_BLENDER,
    CAMERA_CONVENTION_OPENCV,
    normalize_camera_convention,
)


def backproject_uv_depth_to_camera(
    uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsics_k: np.ndarray,
    *,
    camera_convention: str = CAMERA_CONVENTION_BLENDER,
) -> np.ndarray:
    """Backproject pixels to camera coordinates.

    Equations (u right, v down):
    - x = (u - cx) / fx * d
    - y = s_y * (v - cy) / fy * d
    - z = s_z * d

    where `(s_y, s_z)` depend on `camera_convention`:
    - `blender`: `s_y = -1`, `s_z = -1`
    - `opencv`: `s_y = +1`, `s_z = +1`
    """
    conv = normalize_camera_convention(camera_convention)
    uv_arr = np.asarray(uv, dtype=np.float32)
    d = np.asarray(depth_m, dtype=np.float32).reshape(-1)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        raise ValueError(f"uv must be Nx2, got shape {uv_arr.shape}.")
    if uv_arr.shape[0] != d.shape[0]:
        raise ValueError("uv and depth_m length mismatch.")
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("depth_m must be finite and strictly positive.")
    k = np.asarray(intrinsics_k, dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"intrinsics_k must be 3x3, got {k.shape}.")
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid intrinsics focal lengths fx={fx}, fy={fy}.")

    x = (uv_arr[:, 0] - cx) / fx * d
    y = (uv_arr[:, 1] - cy) / fy * d
    z = d.copy()
    if conv == CAMERA_CONVENTION_BLENDER:
        y = -y
        z = -z
    elif conv == CAMERA_CONVENTION_OPENCV:
        pass
    else:
        raise ValueError(
            f"backproject_uv_depth_to_camera currently supports only blender/opencv, got {conv!r}."
        )
    return np.stack([x, y, z], axis=1).astype(np.float32)


def camera_to_world(points_camera: np.ndarray, camera_to_world_matrix: np.ndarray) -> np.ndarray:
    """Transform camera-frame points to world coordinates."""
    pts = np.asarray(points_camera, dtype=np.float32)
    c2w = np.asarray(camera_to_world_matrix, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_camera must be Nx3, got {pts.shape}.")
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world_matrix must be 4x4, got {c2w.shape}.")
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    return (c2w @ pts_h.T).T[:, :3].astype(np.float32)


def world_to_camera(
    points_world: np.ndarray,
    *,
    world_to_camera_matrix: np.ndarray | None = None,
    camera_to_world_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Transform world-frame points to camera coordinates."""
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_world must be Nx3, got {pts.shape}.")
    if world_to_camera_matrix is None:
        if camera_to_world_matrix is None:
            raise ValueError("Either world_to_camera_matrix or camera_to_world_matrix must be provided.")
        c2w = np.asarray(camera_to_world_matrix, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera_to_world_matrix must be 4x4, got {c2w.shape}.")
        w2c = np.linalg.inv(c2w)
    else:
        w2c = np.asarray(world_to_camera_matrix, dtype=np.float32)
        if w2c.shape != (4, 4):
            raise ValueError(f"world_to_camera_matrix must be 4x4, got {w2c.shape}.")
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    return (w2c @ pts_h.T).T[:, :3].astype(np.float32)


def project_world_to_image(
    points_world: np.ndarray,
    intrinsics_k: np.ndarray,
    *,
    world_to_camera_matrix: np.ndarray | None = None,
    camera_to_world_matrix: np.ndarray | None = None,
    camera_convention: str = CAMERA_CONVENTION_BLENDER,
    image_shape: Tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to image coordinates.

    Returns `(uv, valid_mask)` where `uv` has shape `Nx2` and `valid_mask`
    indicates points in front of camera (and inside image if `image_shape` set).
    """
    conv = normalize_camera_convention(camera_convention)
    cam_pts = world_to_camera(
        points_world,
        world_to_camera_matrix=world_to_camera_matrix,
        camera_to_world_matrix=camera_to_world_matrix,
    )
    k = np.asarray(intrinsics_k, dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"intrinsics_k must be 3x3, got {k.shape}.")
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    z = cam_pts[:, 2]
    if conv == CAMERA_CONVENTION_BLENDER:
        in_front = z < -1e-6
        denom = -z
        y_img = -cam_pts[:, 1]
    elif conv == CAMERA_CONVENTION_OPENCV:
        in_front = z > 1e-6
        denom = z
        y_img = cam_pts[:, 1]
    else:
        raise ValueError(
            f"project_world_to_image currently supports only blender/opencv, got {conv!r}."
        )
    denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    u = fx * (cam_pts[:, 0] / denom) + cx
    v = fy * (y_img / denom) + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    finite = np.isfinite(uv).all(axis=1)
    valid = in_front & finite
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        valid = valid & inside
    return uv, valid.astype(bool)

