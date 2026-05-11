"""
Geometry validation functions for coordinate system consistency.

This module provides validation functions that ensure geometric consistency
across different coordinate systems and after coordinate transformations.
The validations help catch errors in coordinate system conversions and
ensure that the final geometry meets quality standards.

Key Functions:
- validate_geometry_consistency(): Comprehensive geometry validation
- validate_camera_height_consistency(): Camera height validation  
- validate_up_direction_consistency(): Up direction validation

The validation functions work with the ResourceStore to access all necessary
data and provide detailed error messages when inconsistencies are detected.
"""

from typing import Optional
import numpy as np

from pemoin.data.contracts import ResourceKind, ResourceStore
from pemoin.utils.geometry import up_direction_from_c2w, view_direction_from_c2w


class GeometryValidationError(ValueError):
    """
    Exception raised when geometry validation fails.
    
    This exception provides detailed information about what went wrong
    during geometry validation, including which specific check failed
    and what the expected vs actual values were.
    """
    pass


def validate_up_direction_consistency(
    c2w: np.ndarray,
    frame_idx: int,
    tolerance: float = 1e-3
) -> None:
    """
    Validate that the up direction is consistent and properly oriented.
    
    This function checks:
    1. Up direction is a unit vector
    2. Up direction is orthogonal to view direction
    3. Up direction points in a reasonable direction (not too close to horizontal)
    
    Args:
        c2w: 4x4 camera-to-world matrix
        frame_idx: Frame index for error reporting
        tolerance: Tolerance for validation checks
        
    Raises:
        GeometryValidationError: If any validation check fails
        
    See Also:
        - up_direction_from_c2w(): Extract up direction from pose matrix
        - compute_up_direction_alignment(): Function that corrects up direction
        - align_trajectory_to_camera_height(): Main alignment function
    """
    view_dir = view_direction_from_c2w(c2w)
    up_dir = up_direction_from_c2w(c2w)
    
    # Check unit length
    if abs(np.linalg.norm(view_dir) - 1.0) > tolerance:
        raise GeometryValidationError(
            f"View direction not unit length for frame {frame_idx}: "
            f"norm={np.linalg.norm(view_dir):.6f}"
        )
    
    if abs(np.linalg.norm(up_dir) - 1.0) > tolerance:
        raise GeometryValidationError(
            f"Up direction not unit length for frame {frame_idx}: "
            f"norm={np.linalg.norm(up_dir):.6f}"
        )
    
    # Check orthogonality
    dot_product = abs(float(np.dot(view_dir, up_dir)))
    if dot_product > tolerance:
        raise GeometryValidationError(
            f"View and up directions not orthogonal for frame {frame_idx}: "
            f"dot={dot_product:.6f}"
        )
    
    # Check up direction is reasonably vertical (not too horizontal)
    horizontal_component = np.linalg.norm(up_dir[:2])  # X and Y components
    if horizontal_component > 0.9:  # Mostly horizontal
        raise GeometryValidationError(
            f"Up direction too horizontal for frame {frame_idx}: "
            f"horizontal={horizontal_component:.3f}"
        )


def validate_camera_height_consistency(
    store: ResourceStore,
    frame_idx: int,
    c2w: np.ndarray,
    tolerance: float = 0.25
) -> None:
    """
    Validate that camera height matches ground truth data.
    
    This function compares the vertical position extracted from the pose matrix
    with the ground truth camera height data to ensure consistency.
    
    Args:
        store: ResourceStore containing camera height data
        frame_idx: Frame index to validate
        c2w: 4x4 camera-to-world matrix
        tolerance: Maximum allowed difference in meters
        
    Raises:
        GeometryValidationError: If height difference exceeds tolerance
        
    See Also:
        - CameraHeightData: Data structure containing height information
        - apply_height_correction(): Function that corrects camera height
        - align_trajectory_to_camera_height(): Main alignment function
    """
    height_data = store.load_camera_height(frame_idx)
    axis = str(height_data.metadata.get("axis", "z")).lower()
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis)
    
    if axis_index is None:
        raise GeometryValidationError(
            f"Camera height axis invalid for frame {frame_idx}: {axis}"
        )
    
    # Extract height from pose matrix
    pose_height = float(c2w[axis_index, 3])
    if bool(height_data.metadata.get("absolute", False)):
        pose_height = abs(pose_height)
    
    # Compare with ground truth
    delta = abs(float(height_data.height_m) - pose_height)
    if delta > tolerance:
        raise GeometryValidationError(
            f"Camera height mismatch for frame {frame_idx}: "
            f"stored={height_data.height_m:.3f}m, pose={pose_height:.3f}m, "
            f"delta={delta:.3f}m"
        )


def validate_geometry_consistency(
    store: ResourceStore,
    frame_indices: Optional[np.ndarray] = None,
    max_frames: int = 50,
    up_direction_tolerance: float = 1e-3,
    height_tolerance: float = 0.25
) -> None:
    """
    Comprehensive geometry validation across multiple frames.
    
    This function performs a series of validation checks to ensure geometric
    consistency across the trajectory. It validates:
    
    1. Up direction consistency for all frames
    2. Camera height consistency (if camera height data available)
    3. Pose matrix quality (orthonormality, invertibility)
    
    Args:
        store: ResourceStore containing geometry data
        frame_indices: Specific frame indices to validate (None for all)
        max_frames: Maximum number of frames to validate
        up_direction_tolerance: Tolerance for up direction validation
        height_tolerance: Tolerance for camera height validation
        
    Raises:
        GeometryValidationError: If any validation check fails
        
    See Also:
        - validate_up_direction_consistency(): Up direction validation
        - validate_camera_height_consistency(): Camera height validation
        - align_trajectory_to_camera_height(): Function that ensures consistency
    """
    # Load trajectory data
    traj_path = store.path_for(ResourceKind.TRAJECTORY)
    with np.load(traj_path, allow_pickle=True) as data:
        all_frame_indices = np.asarray(data["frame_indices"], dtype=int)
        c2w_array = np.asarray(data["camera_to_world"], dtype=np.float32)
    
    # Select frames to validate
    if frame_indices is None:
        frame_indices = all_frame_indices
    
    # Limit to max_frames
    if len(frame_indices) > max_frames:
        stride = max(1, len(frame_indices) // max_frames)
        frame_indices = frame_indices[::stride][:max_frames]
    
    # Create mapping from frame index to array index
    index_map = {frame: i for i, frame in enumerate(all_frame_indices)}
    
    # Validate each frame
    for frame_idx in frame_indices:
        if frame_idx not in index_map:
            continue
        
        array_idx = index_map[frame_idx]
        c2w = c2w_array[array_idx]
        
        # Validate up direction
        validate_up_direction_consistency(c2w, frame_idx, up_direction_tolerance)
        
        # Validate camera height if available
        if store.has(ResourceKind.CAMERA_HEIGHT):
            validate_camera_height_consistency(
                store, frame_idx, c2w, height_tolerance
            )
        
        # Validate pose matrix quality
        rot = c2w[:3, :3]
        det = float(np.linalg.det(rot))
        if abs(det - 1.0) > 0.01:
            raise GeometryValidationError(
                f"Rotation matrix not orthonormal for frame {frame_idx}: "
                f"det={det:.6f}"
            )
        
        # Check invertibility
        try:
            w2c = np.linalg.inv(c2w)
            identity = w2c @ c2w
            if not np.allclose(identity, np.eye(4), atol=1e-3):
                raise GeometryValidationError(
                    f"Pose matrix not invertible for frame {frame_idx}"
                )
        except np.linalg.LinAlgError:
            raise GeometryValidationError(
                f"Pose matrix singular for frame {frame_idx}"
            )
