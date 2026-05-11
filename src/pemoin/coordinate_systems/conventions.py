"""
Coordinate system conventions used in PEMOIN.

This module defines the standard coordinate system conventions and provides
utilities for working with different coordinate systems. Each convention
specifies the axis orientations and handedness of a coordinate system.

Coordinate System Conventions:
- OpenCV: Computer vision standard (Y-down, Z-forward)
- Blender: PEMOIN target convention (Y-up, Z-backward)  
- CARLA: Unreal Engine convention (left-handed, Z-up)
- Unity: Unity game engine convention (left-handed, Y-up)

The conventions are defined as dataclasses with clear axis definitions and
transformation matrices for easy conversion between systems.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class CoordinateConvention:
    """
    Definition of a coordinate system convention.
    
    Attributes:
        name: Human-readable name of the convention
        description: Detailed description of the convention
        handedness: 'right' or 'left' handed coordinate system
        axes: Dictionary mapping axis names to their directions
        up_axis: Primary up axis (usually 'Y' or 'Z')
        forward_axis: Primary forward axis (usually 'Z' or 'X')
        right_axis: Primary right axis (usually 'X' or 'Y')
    """
    name: str
    description: str
    handedness: str
    axes: Dict[str, np.ndarray]
    up_axis: str
    forward_axis: str
    right_axis: str


# OpenCV Convention (Computer Vision Standard)
OPENCV_CONVENTION = CoordinateConvention(
    name="opencv",
    description="OpenCV computer vision convention. Right-handed coordinate system "
                "used by MegaSAM, DepthAnything3, and other CV libraries.",
    handedness="right",
    axes={
        "X": np.array([1.0, 0.0, 0.0]),  # Right
        "Y": np.array([0.0, -1.0, 0.0]), # Down (image coordinates)
        "Z": np.array([0.0, 0.0, 1.0]),  # Forward (into scene)
    },
    up_axis="Y",
    forward_axis="Z",
    right_axis="X"
)


# Blender Convention (PEMOIN Target)
BLENDER_CONVENTION = CoordinateConvention(
    name="blender",
    description="Blender 3D convention. Right-handed coordinate system used as "
                "the target convention for all PEMOIN geometry outputs.",
    handedness="right",
    axes={
        "X": np.array([1.0, 0.0, 0.0]),  # Right
        "Y": np.array([0.0, 1.0, 0.0]),  # Up
        "Z": np.array([0.0, 0.0, -1.0]), # Backward (out of scene)
    },
    up_axis="Y",
    forward_axis="Z",
    right_axis="X"
)


# CARLA Convention (Unreal Engine)
CARLA_CONVENTION = CoordinateConvention(
    name="carla",
    description="CARLA simulator convention. Left-handed coordinate system "
                "based on Unreal Engine conventions.",
    handedness="left",
    axes={
        "X": np.array([1.0, 0.0, 0.0]),  # Forward
        "Y": np.array([0.0, 1.0, 0.0]),  # Right
        "Z": np.array([0.0, 0.0, 1.0]),  # Up
    },
    up_axis="Z",
    forward_axis="X",
    right_axis="Y"
)


# Unity Convention
UNITY_CONVENTION = CoordinateConvention(
    name="unity",
    description="Unity game engine convention. Left-handed coordinate system.",
    handedness="left",
    axes={
        "X": np.array([1.0, 0.0, 0.0]),  # Right
        "Y": np.array([0.0, 1.0, 0.0]),  # Up
        "Z": np.array([0.0, 0.0, 1.0]),  # Forward
    },
    up_axis="Y",
    forward_axis="Z",
    right_axis="X"
)


def get_convention_by_name(name: str) -> Optional[CoordinateConvention]:
    """
    Get coordinate convention by name.
    
    Args:
        name: Name of the convention (case-insensitive)
        
    Returns:
        CoordinateConvention if found, None otherwise
    """
    name_lower = name.lower()
    conventions = {
        "opencv": OPENCV_CONVENTION,
        "cv": OPENCV_CONVENTION,
        "blender": BLENDER_CONVENTION,
        "carla": CARLA_CONVENTION,
        "unreal": CARLA_CONVENTION,
        "unity": UNITY_CONVENTION,
    }
    return conventions.get(name_lower)


def list_all_conventions() -> Dict[str, CoordinateConvention]:
    """
    Get all available coordinate conventions.
    
    Returns:
        Dictionary mapping convention names to CoordinateConvention objects
    """
    return {
        "opencv": OPENCV_CONVENTION,
        "blender": BLENDER_CONVENTION,
        "carla": CARLA_CONVENTION,
        "unity": UNITY_CONVENTION,
    }