"""Foot-anchor estimation helpers."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pemoin.utils.geometry import ProjectionHelper


def estimate_anchor_world(
    depth: np.ndarray,
    intrinsics,
    pose,
    pixel: Tuple[int, int],
    *,
    patch_size: int = 7,
) -> Optional[np.ndarray]:
    """
    Estimate a stable 3D anchor at the foot pixel by median depth.

    Args:
        depth: (H, W) depth map.
        intrinsics: IntrinsicsData or object with .matrix.
        pose: PoseSample for the frame.
        pixel: (u, v) anchor pixel.
        patch_size: odd patch size for median depth.

    Returns:
        (3,) world position or None if no valid depth is available.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
        depth_arr = depth_arr[:, :, 0]
    if depth_arr.ndim != 2:
        raise ValueError("Depth must be a 2D array for anchor estimation.")

    u0, v0 = int(pixel[0]), int(pixel[1])
    h, w = depth_arr.shape[:2]
    if not (0 <= u0 < w and 0 <= v0 < h):
        return None

    if patch_size < 1:
        raise ValueError("patch_size must be >= 1.")
    radius = patch_size // 2
    x0 = max(0, u0 - radius)
    x1 = min(w, u0 + radius + 1)
    y0 = max(0, v0 - radius)
    y1 = min(h, v0 + radius + 1)
    patch = depth_arr[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch) & (patch > 1e-4)]
    if patch.size == 0:
        return None

    depth_med = float(np.median(patch))
    if not np.isfinite(depth_med) or depth_med <= 0:
        return None

    helper = ProjectionHelper(intrinsics=intrinsics, pose=pose)
    anchor = helper.image_to_world(
        np.array([[u0, v0]], dtype=np.float32),
        np.array([depth_med], dtype=np.float32),
    )
    if anchor.shape[0] != 1:
        return None
    return anchor[0]
