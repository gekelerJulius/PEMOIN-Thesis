"""Canonical geometry conventions used across PEMOIN.

This module is the single source of truth for image/camera/world axis semantics.
All projection/backprojection helpers should reference these definitions.
"""

from __future__ import annotations

from dataclasses import dataclass


IMAGE_U_AXIS = "right"
IMAGE_V_AXIS = "down"

CAMERA_CONVENTION_BLENDER = "blender"
CAMERA_CONVENTION_OPENCV = "opencv"
CAMERA_CONVENTION_CARLA = "carla"
CAMERA_CONVENTION_UNITY = "unity"

KNOWN_CAMERA_CONVENTIONS = frozenset(
    {
        CAMERA_CONVENTION_BLENDER,
        CAMERA_CONVENTION_OPENCV,
        CAMERA_CONVENTION_CARLA,
        CAMERA_CONVENTION_UNITY,
    }
)


@dataclass(frozen=True)
class CameraAxes:
    """Axis directions for a camera convention.

    For image coordinates we always assume:
    - `u` increases to the right
    - `v` increases downward

    Depth passed to backprojection is always metric distance from camera center
    along view rays.
    """

    name: str
    x_axis: str
    y_axis: str
    z_axis: str
    forward_sign: int


CAMERA_AXES = {
    CAMERA_CONVENTION_BLENDER: CameraAxes(
        name=CAMERA_CONVENTION_BLENDER,
        x_axis="right",
        y_axis="up",
        z_axis="backward",
        forward_sign=-1,
    ),
    CAMERA_CONVENTION_OPENCV: CameraAxes(
        name=CAMERA_CONVENTION_OPENCV,
        x_axis="right",
        y_axis="down",
        z_axis="forward",
        forward_sign=1,
    ),
}


def normalize_camera_convention(value: object, *, default: str = CAMERA_CONVENTION_BLENDER) -> str:
    """Normalize a camera convention string to a supported key."""
    raw = str(value or "").strip().lower()
    if raw in {"cv", "open_cv"}:
        raw = CAMERA_CONVENTION_OPENCV
    if raw in {"gl", "opengl"}:
        raw = CAMERA_CONVENTION_BLENDER
    if not raw:
        raw = default
    if raw not in KNOWN_CAMERA_CONVENTIONS:
        raise ValueError(
            f"Unsupported camera convention {raw!r}. "
            f"Expected one of {sorted(KNOWN_CAMERA_CONVENTIONS)}."
        )
    return raw


def world_up_axis_for_profile() -> str:
    """Return expected world up axis for standardized PEMOIN geometry."""
    return "z"

