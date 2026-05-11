"""
Lightweight projection helpers for standard PEMOIN geometry.

These helpers work with the canonical IntrinsicsData/PoseSample contracts and
avoid duplication of camera math across providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from pemoin.data.models import IntrinsicsData, PoseSample
from pemoin.geometry.camera_model import (
    backproject_uv_depth_to_camera,
    camera_to_world as transform_camera_to_world,
    project_world_to_image as canonical_project_world_to_image,
)


def _pose_convention(metadata: Optional[dict]) -> str:
    meta = metadata or {}
    explicit = str(meta.get("camera_convention", "") or meta.get("camera_frame", "")).lower()
    if explicit:
        return explicit
    explicit = str(meta.get("pose_coordinate_system", "")).lower()
    if explicit:
        return explicit
    return ""


def pose_uses_opengl(pose: PoseSample) -> bool:
    metadata = getattr(pose, "metadata", None) or {}
    explicit = _pose_convention(metadata)
    if explicit in {"opengl", "gl"}:
        return True
    if explicit in {"opencv", "cv"}:
        return False
    if explicit in {"blender"}:
        return True
    if str(metadata.get("source", "")).lower() == "depthanything3":
        return False
    export_format = str(metadata.get("export_format", "")).lower()
    if "opengl" in export_format or "glb" in export_format:
        return True
    return False


def camera_frame_transform(pose: PoseSample) -> np.ndarray:
    """
    Return a 3x3 transform that maps OpenCV camera coords into the pose's camera frame.
    """
    explicit = _pose_convention(getattr(pose, "metadata", None))
    if explicit in {"opencv", "cv"}:
        from pemoin.coordinate_systems.conversions import opencv_to_blender_matrix

        return opencv_to_blender_matrix()
    return np.eye(3, dtype=np.float32)


def view_direction_from_c2w(camera_to_world: np.ndarray) -> np.ndarray:
    """Return Blender view direction (-Z) in world coordinates."""
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    return -c2w[:3, 2]


def up_direction_from_c2w(camera_to_world: np.ndarray) -> np.ndarray:
    """
    Return Blender up direction (+Y) in world coordinates.
    
    Extracts the up direction vector from a camera-to-world matrix in Blender convention.
    The up direction corresponds to the Y-axis of the camera coordinate system
    transformed into world coordinates.
    
    Args:
        camera_to_world: 4x4 camera-to-world matrix in Blender convention
        
    Returns:
        3D up direction vector as numpy array
        
    See Also:
        - view_direction_from_c2w(): Extract view direction from pose matrix
        - validate_up_direction_consistency(): Validate up direction properties
        - compute_up_direction_alignment(): Compute alignment rotation for up direction
        
    Example:
        >>> c2w = np.eye(4)  # Identity pose
        >>> up_dir = up_direction_from_c2w(c2w)
        >>> # up_dir should be [0, 1, 0] (pointing straight up)
    """
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    return c2w[:3, 1]


@dataclass
class ProjectionHelper:
    """Utility for projecting between image and world coordinates."""

    intrinsics: IntrinsicsData
    pose: PoseSample
    _c2w: np.ndarray | None = None
    _w2c: np.ndarray | None = None
    _k: np.ndarray | None = None
    _cam_to_pose: np.ndarray | None = None

    def __post_init__(self) -> None:
        c2w = np.asarray(self.pose.camera_to_world, dtype=np.float32)
        w2c = (
            np.asarray(self.pose.world_to_camera, dtype=np.float32)
            if self.pose.world_to_camera is not None
            else None
        )
        if c2w.shape == (4, 4) and w2c is None:
            w2c = np.linalg.inv(c2w)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera_to_world must be 4x4, got {c2w.shape}")
        object.__setattr__(self, "_c2w", c2w)
        object.__setattr__(self, "_w2c", w2c)
        object.__setattr__(self, "_k", np.asarray(self.intrinsics.matrix, dtype=np.float32))
        object.__setattr__(self, "_cam_to_pose", camera_frame_transform(self.pose))

    @property
    def camera_to_world(self) -> np.ndarray:
        return self._c2w

    @property
    def world_to_camera(self) -> np.ndarray:
        return self._w2c if self._w2c is not None else np.linalg.inv(self._c2w)

    def project_world_to_image(self, points_world: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        convention = _pose_convention(getattr(self.pose, "metadata", None)) or "blender"
        uv, valid = canonical_project_world_to_image(
            pts,
            self._k,
            world_to_camera_matrix=self.world_to_camera,
            camera_convention=convention,
            image_shape=image_shape,
        )
        if not np.any(valid):
            return np.zeros((0, 2), dtype=np.float32)
        return uv[valid]

    def image_to_world(self, uv: np.ndarray, depths: Iterable[float]) -> np.ndarray:
        uv_arr = np.asarray(uv, dtype=np.float32)
        depth_arr = np.asarray(list(depths), dtype=np.float32).reshape(-1, 1)
        if uv_arr.shape[0] != depth_arr.shape[0]:
            raise ValueError("uv and depths must have matching lengths")
        convention = _pose_convention(getattr(self.pose, "metadata", None)) or "blender"
        cam_pts = backproject_uv_depth_to_camera(
            uv_arr,
            depth_arr.reshape(-1),
            self._k,
            camera_convention=convention,
        )
        cam_to_pose = self._cam_to_pose
        if cam_to_pose is not None:
            cam_pts = (cam_to_pose @ cam_pts.T).T
        return transform_camera_to_world(cam_pts, self.camera_to_world)

    def backproject_depth_map(self, depth: np.ndarray) -> np.ndarray:
        depth_arr = np.asarray(depth, dtype=np.float32)
        h, w = depth_arr.shape[:2]
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        valid = np.isfinite(depth_arr) & (depth_arr > 0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)
        uv = np.stack([xs[valid], ys[valid]], axis=1)
        depths = depth_arr[valid]
        return self.image_to_world(uv, depths)
