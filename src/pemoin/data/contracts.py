"""Compatibility facade for PEMOIN standardized data contracts."""

from __future__ import annotations

from importlib import import_module

from .layouts import _STANDARD_LAYOUTS
from .models import (
    CAMERA_CONVENTION_KEY,
    CAMERA_CONVENTION_VALUES,
    HEIGHT_AXIS_KEY,
    HEIGHT_METADATA_REQUIRED_FIELDS,
    PLANE_METADATA_REQUIRED_FIELDS,
    POSE_COORDINATE_SYSTEM_KEY,
    WORLD_COORDINATE_SYSTEM_KEY,
    CameraHeightData,
    ConfidenceVolumeData,
    DepthData,
    DynamicMaskData,
    FrameData,
    IntrinsicsData,
    LightingLightData,
    LightingData,
    PointCloud3DData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceLayout,
    RoadPlaneData,
    RoadPlaneSupportData,
    SemanticSegment,
    SemanticsAuxData,
    SemanticsData,
    TrajectoryMatchGraphData,
    lighting_from_payload,
    lighting_light_from_mapping,
    lighting_light_to_payload,
    lighting_to_payload,
)
from .store import ResourceMissingError, ResourceStore, _normalize_json_payload

__all__ = [
    "CAMERA_CONVENTION_KEY",
    "CAMERA_CONVENTION_VALUES",
    "HEIGHT_AXIS_KEY",
    "HEIGHT_METADATA_REQUIRED_FIELDS",
    "PLANE_METADATA_REQUIRED_FIELDS",
    "POSE_COORDINATE_SYSTEM_KEY",
    "WORLD_COORDINATE_SYSTEM_KEY",
    "CameraHeightData",
    "ConfidenceVolumeData",
    "DepthData",
    "DynamicMaskData",
    "FrameData",
    "IntrinsicsData",
    "LightingLightData",
    "LightingData",
    "PointCloud3DData",
    "PoseData",
    "PoseSample",
    "ResourceKind",
    "ResourceLayout",
    "ResourceMissingError",
    "ResourceStore",
    "RoadPlaneData",
    "RoadPlaneSupportData",
    "SemanticSegment",
    "SemanticsAuxData",
    "SemanticsData",
    "TrajectoryMatchGraphData",
    "_STANDARD_LAYOUTS",
    "_normalize_json_payload",
    "lighting_from_payload",
    "lighting_light_from_mapping",
    "lighting_light_to_payload",
    "lighting_to_payload",
]


def __getattr__(name: str):
    if name in {"ResourceMissingError", "ResourceStore", "_normalize_json_payload"}:
        store_module = import_module("pemoin.data.store")
        value = getattr(store_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
