"""Projection utilities for reprojection consistency checks."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from pemoin.utils.geometry import camera_frame_transform


def project_world_to_image(
    points_world: np.ndarray,
    intrinsics,
    pose,
    image_shape: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world points into image space.

    Returns (uv, z_cam, valid) where valid is inside the image and z_cam > 0.
    """
    pts = np.asarray(points_world, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    world_to_camera = (
        np.asarray(pose.world_to_camera, dtype=np.float32)
        if pose.world_to_camera is not None
        else np.linalg.inv(np.asarray(pose.camera_to_world, dtype=np.float32))
    )
    pts_cam = (world_to_camera @ pts_h.T).T[:, :3]
    cam_to_pose = camera_frame_transform(pose)
    if cam_to_pose is not None:
        pts_cam = (cam_to_pose.T @ pts_cam.T).T
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1) & (z > 1e-6)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        k = np.asarray(intrinsics.matrix, dtype=np.float32)
        proj = (k @ pts_cam[valid].T).T
        uv[valid] = proj[:, :2] / proj[:, 2:3]
    h, w = image_shape
    inside = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    return uv, z, inside


def sample_depth_bilinear(
    depth: np.ndarray,
    uv: np.ndarray,
    *,
    min_depth: float = 1e-6,
) -> np.ndarray:
    """Bilinearly sample depth at floating uv coordinates; returns nan for invalid/out-of-bounds."""
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        raise ValueError("Depth map must be 2D.")
    if uv.size == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = depth_arr.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1
    inside = (u0 >= 0) & (v0 >= 0) & (u1 < w) & (v1 < h)
    if not np.any(inside):
        return np.full((uv.shape[0],), np.nan, dtype=np.float32)
    z00 = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    z01 = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    z10 = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    z11 = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    idx = np.flatnonzero(inside)
    z00[idx] = depth_arr[v0[idx], u0[idx]]
    z01[idx] = depth_arr[v1[idx], u0[idx]]
    z10[idx] = depth_arr[v0[idx], u1[idx]]
    z11[idx] = depth_arr[v1[idx], u1[idx]]
    valid = (
        np.isfinite(z00)
        & np.isfinite(z01)
        & np.isfinite(z10)
        & np.isfinite(z11)
        & (z00 > min_depth)
        & (z01 > min_depth)
        & (z10 > min_depth)
        & (z11 > min_depth)
    )
    if not np.any(valid):
        return np.full((uv.shape[0],), np.nan, dtype=np.float32)
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)
    w00 = (1.0 - du) * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w10 = du * (1.0 - dv)
    w11 = du * dv
    samples = w00 * z00 + w01 * z01 + w10 * z10 + w11 * z11
    samples[~valid] = np.nan
    return samples


def sample_scalar_bilinear(values: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample a scalar map; returns nan for out-of-bounds."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Scalar map must be 2D.")
    if uv.size == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = arr.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1
    inside = (u0 >= 0) & (v0 >= 0) & (u1 < w) & (v1 < h)
    samples = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    if not np.any(inside):
        return samples
    idx = np.flatnonzero(inside)
    z00 = arr[v0[idx], u0[idx]]
    z01 = arr[v1[idx], u0[idx]]
    z10 = arr[v0[idx], u1[idx]]
    z11 = arr[v1[idx], u1[idx]]
    du = (u[idx] - u0[idx]).astype(np.float32)
    dv = (v[idx] - v0[idx]).astype(np.float32)
    w00 = (1.0 - du) * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w10 = du * (1.0 - dv)
    w11 = du * dv
    samples[idx] = w00 * z00 + w01 * z01 + w10 * z10 + w11 * z11
    return samples
