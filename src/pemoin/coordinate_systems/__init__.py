"""
Coordinate system management and conversion utilities for PEMOIN.

This module provides comprehensive support for handling different coordinate
system conventions and converting between them. It serves as the central
location for all coordinate system-related functionality in PEMOIN.

Key Components:
- conventions: Definition of coordinate system conventions
- conversions: Conversion functions between coordinate systems  
- alignment: Scene-specific trajectory alignment utilities
- validation: Geometry validation functions

Usage:
    from pemoin.coordinate_systems import (
        convert_pose_opencv_to_blender,
        canonicalize_geometry_to_comparison_frame,
        validate_geometry_consistency
    )
"""

from .conventions import (
    CoordinateConvention,
    OPENCV_CONVENTION,
    BLENDER_CONVENTION,
    CARLA_CONVENTION,
    UNITY_CONVENTION,
    get_convention_by_name
)
from .conversions import (
    convert_pose_opencv_to_blender,
    convert_pose_carla_to_blender,
    convert_pose_unity_to_blender,
    opencv_to_blender_matrix,
    opencv_to_blender_matrix4,
    carla_to_blender_matrix,
    unity_to_blender_matrix
)
from .alignment import (
    ComparisonFrameSettings,
    canonicalize_geometry_to_comparison_frame,
    compute_up_direction_alignment,
    apply_height_correction,
    verify_alignment_consistency,
)
from .trajectory_origin import (
    anchor_pose_data_to_origin,
    compute_origin_anchor_translation,
    resolve_anchor_height_from_store,
    save_origin_anchored_trajectory,
)
from .validation import (
    validate_geometry_consistency,
    validate_camera_height_consistency,
    validate_up_direction_consistency
)

__all__ = [
    "CoordinateConvention",
    "OPENCV_CONVENTION",
    "BLENDER_CONVENTION",
    "CARLA_CONVENTION",
    "UNITY_CONVENTION",
    "get_convention_by_name",
    "convert_pose_opencv_to_blender",
    "convert_pose_carla_to_blender",
    "convert_pose_unity_to_blender",
    "opencv_to_blender_matrix",
    "opencv_to_blender_matrix4",
    "carla_to_blender_matrix",
    "unity_to_blender_matrix",
    "ComparisonFrameSettings",
    "canonicalize_geometry_to_comparison_frame",
    "compute_up_direction_alignment",
    "apply_height_correction",
    "verify_alignment_consistency",
    "anchor_pose_data_to_origin",
    "compute_origin_anchor_translation",
    "resolve_anchor_height_from_store",
    "save_origin_anchored_trajectory",
    "validate_geometry_consistency",
    "validate_camera_height_consistency",
    "validate_up_direction_consistency",
]
