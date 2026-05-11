"""Backprojection helpers for converting masked pixels to world points."""
from __future__ import annotations


import numpy as np

from pemoin.utils.geometry import ProjectionHelper


def backproject_mask_to_world(
    depth: np.ndarray,
    intrinsics,
    pose,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Backproject masked depth pixels to world points.

    Args:
        depth: (H, W) depth map.
        intrinsics: IntrinsicsData or object with .matrix.
        pose: PoseSample for the frame.
        mask: (H, W) boolean mask of pixels to backproject.

    Returns:
        (N, 3) array of world points. Empty if mask has no valid pixels.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
        depth_arr = depth_arr[:, :, 0]
    if depth_arr.ndim != 2:
        raise ValueError("Depth must be a 2D array for backprojection.")

    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != depth_arr.shape:
        raise ValueError("Mask shape must match depth shape for backprojection.")

    valid = mask_arr & np.isfinite(depth_arr) & (depth_arr > 1e-4)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    ys, xs = np.where(valid)
    uv = np.stack([xs, ys], axis=1).astype(np.float32)
    depths = depth_arr[ys, xs].astype(np.float32)
    helper = ProjectionHelper(intrinsics=intrinsics, pose=pose)
    return helper.image_to_world(uv, depths)


def backproject_points_from_uv(
    uv: np.ndarray,
    depth: np.ndarray,
    intrinsics,
    pose,
) -> np.ndarray:
    """
    Backproject explicit pixel coordinates to world points.

    Args:
        uv: (N, 2) pixel coordinates.
        depth: (H, W) depth map.
        intrinsics: IntrinsicsData or object with .matrix.
        pose: PoseSample for the frame.

    Returns:
        (N, 3) array of world points.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
        depth_arr = depth_arr[:, :, 0]
    if depth_arr.ndim != 2:
        raise ValueError("Depth must be a 2D array for backprojection.")

    uv_arr = np.asarray(uv, dtype=np.int32)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        raise ValueError("uv must be an (N, 2) array.")
    h, w = depth_arr.shape[:2]
    u = np.clip(uv_arr[:, 0], 0, w - 1)
    v = np.clip(uv_arr[:, 1], 0, h - 1)
    depths = depth_arr[v, u].astype(np.float32)
    helper = ProjectionHelper(intrinsics=intrinsics, pose=pose)
    return helper.image_to_world(uv_arr.astype(np.float32), depths)
