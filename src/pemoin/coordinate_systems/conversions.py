"""
Coordinate system conversion functions.

This module provides functions for converting poses and geometry between
different coordinate system conventions. The conversions are implemented
as clean, well-documented functions that handle the mathematical transformations
while preserving important metadata.

Key Functions:
- convert_pose_opencv_to_blender(): Convert OpenCV poses to Blender convention
- convert_pose_carla_to_blender(): Convert CARLA poses to Blender convention  
- convert_pose_unity_to_blender(): Convert Unity poses to Blender convention
- opencv_to_blender_matrix(): Get 3x3 transformation matrix
- opencv_to_blender_matrix4(): Get 4x4 transformation matrix

The conversion functions follow a consistent pattern:
1. Extract the transformation matrix for the source convention
2. Apply the transformation to camera-to-world and world-to-camera matrices
3. Preserve or update metadata to reflect the new convention
4. Return the converted matrices with updated metadata
"""

import logging
from typing import Optional, Tuple
import numpy as np


LOG = logging.getLogger("pemoin")


# Transformation Matrices

def opencv_to_blender_matrix() -> np.ndarray:
    """
    Get 3x3 transformation matrix from OpenCV to Blender convention.
    
    The matrix performs:
    - Y-axis inversion (OpenCV Y-down → Blender Y-up)
    - Z-axis inversion (OpenCV Z-forward → Blender Z-backward)
    - X-axis unchanged
    
    Returns:
        3x3 transformation matrix as numpy array
    
    See Also:
        - opencv_to_blender_matrix4(): 4x4 version of this matrix
        - convert_pose_opencv_to_blender(): Full pose conversion function
    """
    return np.array([
        [1.0, 0.0, 0.0],   # X: unchanged
        [0.0, -1.0, 0.0],  # Y: inverted (down → up)
        [0.0, 0.0, -1.0],  # Z: inverted (forward → backward)
    ], dtype=np.float32)


def opencv_to_blender_matrix4() -> np.ndarray:
    """
    Get 4x4 transformation matrix from OpenCV to Blender convention.
    
    Returns:
        4x4 transformation matrix as numpy array
        
    See Also:
        - opencv_to_blender_matrix(): 3x3 version of this matrix
        - convert_pose_opencv_to_blender(): Full pose conversion function
    """
    t3 = opencv_to_blender_matrix()
    t4 = np.eye(4, dtype=np.float32)
    t4[:3, :3] = t3
    return t4


def carla_to_blender_matrix() -> np.ndarray:
    """
    Get 4x4 transformation matrix from CARLA to Blender convention.
    
    CARLA uses a left-handed system (X-forward, Y-right, Z-up) while
    Blender uses right-handed (X-right, Y-up, Z-backward).
    
    Returns:
        4x4 transformation matrix for CARLA → Blender conversion
        
    See Also:
        - convert_pose_carla_to_blender(): Full CARLA pose conversion
        - _flip_carla_world_y(): CARLA-specific Y-axis flip
    """
    # Legacy compatibility matrix for axis remap visualization. For full pose
    # conversion, use convert_pose_carla_to_blender(), which applies both world
    # and camera basis transforms.
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _carla_world_to_blender_world_matrix4() -> np.ndarray:
    """Map CARLA world axes to Blender world axes while preserving vertical Z."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix


def _blender_camera_to_carla_camera_matrix4() -> np.ndarray:
    """Map Blender camera basis into CARLA camera basis."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return matrix


def unity_to_blender_matrix() -> np.ndarray:
    """
    Get 4x4 transformation matrix from Unity to Blender convention.
    
    Unity uses left-handed (X-right, Y-up, Z-forward) while
    Blender uses right-handed (X-right, Y-up, Z-backward).
    
    Returns:
        4x4 transformation matrix for Unity → Blender conversion
        
    See Also:
        - convert_pose_unity_to_blender(): Full Unity pose conversion
    """
    return np.array([
        [1.0, 0.0, 0.0, 0.0],  # X: unchanged
        [0.0, 1.0, 0.0, 0.0],  # Y: unchanged
        [0.0, 0.0, -1.0, 0.0], # Z: inverted (forward → backward)
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


# Conversion Functions

def convert_pose_opencv_to_blender(
    camera_to_world: np.ndarray,
    world_to_camera: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert pose matrices from OpenCV convention to Blender convention.
    
    This function applies the standard OpenCV → Blender transformation:
    - Inverts Y-axis (OpenCV Y-down → Blender Y-up)
    - Inverts Z-axis (OpenCV Z-forward → Blender Z-backward)
    - Preserves X-axis (right direction)
    
    Args:
        camera_to_world: 4x4 camera-to-world matrix in OpenCV convention
        world_to_camera: Optional 4x4 world-to-camera matrix in OpenCV convention
        
    Returns:
        Tuple of (camera_to_world_blender, world_to_camera_blender)
        
    See Also:
        - opencv_to_blender_matrix4(): The underlying transformation matrix
        - Runtime._align_trajectory_to_camera_height(): Scene-specific alignment
        - validate_up_direction_consistency(): Validation of up direction
        
    Example:
        >>> c2w_opencv = np.eye(4)
        >>> c2w_blender, w2c_blender = convert_pose_opencv_to_blender(c2w_opencv)
        >>> # c2w_blender now has Y-up, Z-backward convention
    """
    t4 = opencv_to_blender_matrix4()
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    c2w_bl = t4 @ c2w @ t4
    
    w2c_bl = None
    if world_to_camera is not None:
        w2c = np.asarray(world_to_camera, dtype=np.float32)
        w2c_bl = t4 @ w2c @ t4
        
    return c2w_bl, w2c_bl


def convert_pose_opencv_camera_to_blender_world(
    camera_to_world: np.ndarray,
    world_to_camera: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert only the camera basis from OpenCV to Blender while preserving world basis.

    This is intended for mixed-basis poses where the camera uses the OpenCV basis
    but the world frame is already a metric world frame (for example, nuScenes
    global coordinates). In that case, we must remap only the camera basis:

    p_world = C2W_cv * p_cam_cv
    p_cam_cv = T * p_cam_bl
    => C2W_mixed = C2W_cv * T

    Args:
        camera_to_world: 4x4 camera-to-world matrix with OpenCV camera basis.
        world_to_camera: Optional 4x4 world-to-camera matrix with OpenCV camera basis.

    Returns:
        Tuple of (camera_to_world_mixed, world_to_camera_mixed)
    """
    t4 = opencv_to_blender_matrix4()
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    c2w_bl = c2w @ t4

    w2c_bl = None
    if world_to_camera is not None:
        w2c = np.asarray(world_to_camera, dtype=np.float32)
        w2c_bl = t4 @ w2c

    return c2w_bl, w2c_bl


def convert_pose_carla_to_blender(
    camera_to_world: np.ndarray,
    world_to_camera: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert pose matrices from CARLA convention to Blender convention.
    
    CARLA uses Unreal Engine's left-handed coordinate system:
    - X: Forward, Y: Right, Z: Up
    
    This function converts to Blender's right-handed system:
    - X: Right, Y: Up, Z: Backward
    
    Args:
        camera_to_world: 4x4 camera-to-world matrix in CARLA convention
        world_to_camera: Optional 4x4 world-to-camera matrix in CARLA convention
        
    Returns:
        Tuple of (camera_to_world_blender, world_to_camera_blender)
        
    See Also:
        - carla_to_blender_matrix(): The underlying transformation matrix
        - _flip_carla_world_y(): CARLA-specific Y-axis correction
        - convert_pose_opencv_to_blender(): OpenCV conversion for comparison
    """
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world must be 4x4 for CARLA conversion, got {c2w.shape}.")

    world_basis = _carla_world_to_blender_world_matrix4()
    camera_basis = _blender_camera_to_carla_camera_matrix4()
    c2w_bl = world_basis @ c2w @ camera_basis

    if world_to_camera is None:
        w2c_bl = None
    else:
        w2c = np.asarray(world_to_camera, dtype=np.float32)
        if w2c.shape != (4, 4):
            raise ValueError(f"world_to_camera must be 4x4 for CARLA conversion, got {w2c.shape}.")
        camera_basis_inv = np.linalg.inv(camera_basis)
        world_basis_inv = np.linalg.inv(world_basis)
        w2c_bl = camera_basis_inv @ w2c @ world_basis_inv
        residual = float(np.max(np.abs((w2c_bl @ c2w_bl) - np.eye(4, dtype=np.float32))))
        LOG.debug(
            "[Conversion][CARLA->Blender] inverse residual max=%.6e",
            residual,
        )
        if residual > 1e-3:
            raise ValueError(
                "CARLA pose conversion produced inconsistent c2w/w2c matrices "
                f"(max residual={residual:.6e})."
            )

    LOG.debug(
        "[Conversion][CARLA->Blender] translation raw=[%.6f, %.6f, %.6f] converted=[%.6f, %.6f, %.6f]",
        float(c2w[0, 3]),
        float(c2w[1, 3]),
        float(c2w[2, 3]),
        float(c2w_bl[0, 3]),
        float(c2w_bl[1, 3]),
        float(c2w_bl[2, 3]),
    )
    return c2w_bl.astype(np.float32), None if w2c_bl is None else w2c_bl.astype(np.float32)


def convert_pose_unity_to_blender(
    camera_to_world: np.ndarray,
    world_to_camera: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert pose matrices from Unity convention to Blender convention.
    
    Unity uses left-handed coordinate system:
    - X: Right, Y: Up, Z: Forward
    
    This function converts to Blender's right-handed system:
    - X: Right, Y: Up, Z: Backward (only Z-axis inverted)
    
    Args:
        camera_to_world: 4x4 camera-to-world matrix in Unity convention
        world_to_camera: Optional 4x4 world-to-camera matrix in Unity convention
        
    Returns:
        Tuple of (camera_to_world_blender, world_to_camera_blender)
        
    See Also:
        - unity_to_blender_matrix(): The underlying transformation matrix
        - convert_pose_opencv_to_blender(): Similar conversion for OpenCV
    """
    t4 = unity_to_blender_matrix()
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    c2w_bl = t4 @ c2w @ t4
    
    w2c_bl = None
    if world_to_camera is not None:
        w2c = np.asarray(world_to_camera, dtype=np.float32)
        w2c_bl = t4 @ w2c @ t4
        
    return c2w_bl, w2c_bl
