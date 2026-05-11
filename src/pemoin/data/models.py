"""Typed standardized resource models used across PEMOIN."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

# Metadata schema constants used across providers and validators.
CAMERA_CONVENTION_KEY = "camera_convention"
POSE_COORDINATE_SYSTEM_KEY = "pose_coordinate_system"
WORLD_COORDINATE_SYSTEM_KEY = "world_coordinate_system"
HEIGHT_AXIS_KEY = "axis"

CAMERA_CONVENTION_VALUES = frozenset({"blender", "opencv", "carla", "unity"})
PLANE_METADATA_REQUIRED_FIELDS = frozenset(
    {
        "source",
        "residual_median",
        "residual_p90",
        "inlier_ratio",
    }
)
HEIGHT_METADATA_REQUIRED_FIELDS = frozenset(
    {
        "source",
        HEIGHT_AXIS_KEY,
        WORLD_COORDINATE_SYSTEM_KEY,
    }
)


@dataclass(slots=True)
class FrameData:
    """RGB frame data along with timestamp and frame identifier."""

    frame_id: str
    index: int
    timestamp: Optional[float] = None
    image: Optional[np.ndarray] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntrinsicsData:
    """Camera intrinsics shared across a sequence."""

    matrix: np.ndarray
    distortion: Optional[np.ndarray] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DepthData:
    """Depth map result including optional confidence estimates."""

    frame_index: int
    depth: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Sequence[int]:
        return self.depth.shape


@dataclass(slots=True)
class PoseSample:
    """Single camera pose sample."""

    frame_index: int
    camera_to_world: np.ndarray
    world_to_camera: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PoseData:
    """Camera pose representation for one or multiple estimates."""

    samples: List[PoseSample]
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def by_frame(self) -> Mapping[int, PoseSample]:
        return {sample.frame_index: sample for sample in self.samples}


@dataclass(slots=True)
class CameraHeightData:
    """Camera height sample for a single frame."""

    frame_index: int
    height_m: float
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoadPlaneData:
    """Estimated road plane for a single frame."""

    frame_index: int
    normal: np.ndarray
    offset: float
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoadPlaneSupportData:
    """Persisted per-frame road-plane support points and diagnostics."""

    frame_index: int
    points_world: np.ndarray
    weights: Optional[np.ndarray] = None
    source_frame_index: Optional[int] = None
    diagnostics: MutableMapping[str, Any] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DynamicMaskData:
    """Per-frame binary mask separating static and dynamic regions."""

    frame_index: int
    mask: np.ndarray
    dynamic_classes: tuple[str, ...]
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PointCloud3DData:
    """Dense 3D point cloud with per-point semantic labels and colors."""

    points_world: np.ndarray
    labels: np.ndarray
    label_confidences: np.ndarray
    colors: np.ndarray
    label_names: Dict[int, str]
    observation_counts: np.ndarray
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LightingLightData:
    """One analytic light extracted from the clip-level lighting package."""

    name: str
    kind: str
    role: str
    strength: float
    color: np.ndarray
    casts_shadow: bool = False
    placement_mode: str = "world_absolute"
    placement_target: str = "world"
    direction_world: Optional[np.ndarray] = None
    rotation_world: Optional[np.ndarray] = None
    location_world: Optional[np.ndarray] = None
    angular_size_deg: Optional[float] = None
    area_size: Optional[np.ndarray] = None
    confidence: float = 1.0
    diagnostics: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LightingData:
    """Clip-level lighting package used by Blender and downstream consumers."""

    sun_direction_world: np.ndarray
    sun_strength: float
    sun_color: np.ndarray
    envmap_path: str
    envmap_rotation_world: np.ndarray
    ambient_strength: float
    mode: str = "full_sun"
    schema_version: int = 2
    rig_mode: str = "envmap_only"
    light_rig: List[LightingLightData] = field(default_factory=list)
    decomposition: MutableMapping[str, Any] = field(default_factory=dict)
    quality: MutableMapping[str, float] = field(default_factory=dict)
    sun_diagnostics: MutableMapping[str, Any] = field(default_factory=dict)
    validation: MutableMapping[str, Any] = field(default_factory=dict)
    recovery: MutableMapping[str, Any] = field(default_factory=dict)
    selected_frame_indices: List[int] = field(default_factory=list)
    per_keyframe_diagnostics: List[MutableMapping[str, Any]] = field(default_factory=list)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


def lighting_light_to_payload(light: LightingLightData) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": str(light.name),
        "kind": str(light.kind),
        "role": str(light.role),
        "strength": float(light.strength),
        "color": np.asarray(light.color, dtype=np.float32).reshape(3),
        "casts_shadow": bool(light.casts_shadow),
        "placement_mode": str(light.placement_mode or "world_absolute"),
        "placement_target": str(light.placement_target or "world"),
        "confidence": float(light.confidence),
        "diagnostics": dict(light.diagnostics or {}),
    }
    if light.direction_world is not None:
        payload["direction_world"] = np.asarray(light.direction_world, dtype=np.float32).reshape(3)
    if light.rotation_world is not None:
        payload["rotation_world"] = np.asarray(light.rotation_world, dtype=np.float32).reshape(3)
    if light.location_world is not None:
        payload["location_world"] = np.asarray(light.location_world, dtype=np.float32).reshape(3)
    if light.angular_size_deg is not None:
        payload["angular_size_deg"] = float(light.angular_size_deg)
    if light.area_size is not None:
        payload["area_size"] = np.asarray(light.area_size, dtype=np.float32).reshape(2)
    return payload


def lighting_light_from_mapping(payload: Mapping[str, Any], *, key: str = "light_rig[]") -> LightingLightData:
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"Invalid {key}.name: expected non-empty string.")
    kind = str(payload.get("kind", "")).strip().upper()
    if kind not in {"SUN", "AREA", "POINT"}:
        raise ValueError(f"Invalid {key}.kind: expected 'SUN', 'AREA', or 'POINT'.")
    role = str(payload.get("role", "")).strip().lower()
    if not role:
        raise ValueError(f"Invalid {key}.role: expected non-empty string.")
    placement_mode = str(payload.get("placement_mode", "world_absolute")).strip().lower()
    if placement_mode not in {"world_absolute", "subject_anchor_relative"}:
        raise ValueError(
            f"Invalid {key}.placement_mode: expected 'world_absolute' or 'subject_anchor_relative'."
        )
    placement_target_default = (
        "world" if placement_mode == "world_absolute" else "subject_root_dynamic"
    )
    placement_target = str(
        payload.get("placement_target", placement_target_default)
    ).strip().lower()
    if placement_target not in {"world", "subject_spawn_static", "subject_root_dynamic"}:
        raise ValueError(
            f"Invalid {key}.placement_target: expected 'world', "
            "'subject_spawn_static', or 'subject_root_dynamic'."
        )
    if placement_mode == "world_absolute" and placement_target != "world":
        raise ValueError(
            f"Invalid {key}: placement_target must be 'world' when placement_mode is "
            "'world_absolute'."
        )
    if placement_mode == "subject_anchor_relative" and placement_target == "world":
        raise ValueError(
            f"Invalid {key}: placement_target must not be 'world' when placement_mode is "
            "'subject_anchor_relative'."
        )
    strength = float(payload.get("strength", 0.0))
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError(f"Invalid {key}.strength: expected finite value >= 0.")
    color = np.asarray(payload.get("color", (1.0, 1.0, 1.0)), dtype=np.float32).reshape(-1)
    if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0.0):
        raise ValueError(f"Invalid {key}.color: expected finite RGB triple >= 0.")
    direction_world: Optional[np.ndarray] = None
    rotation_world: Optional[np.ndarray] = None
    location_world: Optional[np.ndarray] = None
    angular_size_deg: Optional[float] = None
    area_size: Optional[np.ndarray] = None
    if kind == "SUN":
        direction_world = np.asarray(payload.get("direction_world", (0.0, 0.0, 1.0)), dtype=np.float32).reshape(-1)
        if direction_world.shape != (3,) or not np.isfinite(direction_world).all():
            raise ValueError(f"Invalid {key}.direction_world: expected finite XYZ triple.")
        norm = float(np.linalg.norm(direction_world))
        if norm <= 1e-6:
            raise ValueError(f"Invalid {key}.direction_world: zero vector is not allowed.")
        direction_world = direction_world / norm
        angular_size_deg = float(payload.get("angular_size_deg", 2.0))
        if not np.isfinite(angular_size_deg) or angular_size_deg <= 0.0:
            raise ValueError(f"Invalid {key}.angular_size_deg: expected finite value > 0.")
    elif kind == "AREA":
        direction_raw = payload.get("direction_world")
        if direction_raw is not None:
            direction_world = np.asarray(direction_raw, dtype=np.float32).reshape(-1)
            if direction_world.shape != (3,) or not np.isfinite(direction_world).all():
                raise ValueError(f"Invalid {key}.direction_world: expected finite XYZ triple.")
            norm = float(np.linalg.norm(direction_world))
            if norm <= 1e-6:
                raise ValueError(f"Invalid {key}.direction_world: zero vector is not allowed.")
            direction_world = direction_world / norm
        rotation_world = np.asarray(payload.get("rotation_world", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(-1)
        location_world = np.asarray(payload.get("location_world", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(-1)
        area_size = np.asarray(payload.get("area_size", (10.0, 10.0)), dtype=np.float32).reshape(-1)
        if rotation_world.shape != (3,) or not np.isfinite(rotation_world).all():
            raise ValueError(f"Invalid {key}.rotation_world: expected finite XYZ triple.")
        if location_world.shape != (3,) or not np.isfinite(location_world).all():
            raise ValueError(f"Invalid {key}.location_world: expected finite XYZ triple.")
        if area_size.shape != (2,) or not np.isfinite(area_size).all() or np.any(area_size <= 0.0):
            raise ValueError(f"Invalid {key}.area_size: expected positive finite [width, height].")
    else:
        direction_raw = payload.get("direction_world")
        if direction_raw is not None:
            direction_world = np.asarray(direction_raw, dtype=np.float32).reshape(-1)
            if direction_world.shape != (3,) or not np.isfinite(direction_world).all():
                raise ValueError(f"Invalid {key}.direction_world: expected finite XYZ triple.")
            norm = float(np.linalg.norm(direction_world))
            if norm <= 1e-6:
                raise ValueError(f"Invalid {key}.direction_world: zero vector is not allowed.")
            direction_world = direction_world / norm
        location_world = np.asarray(payload.get("location_world", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(-1)
        if location_world.shape != (3,) or not np.isfinite(location_world).all():
            raise ValueError(f"Invalid {key}.location_world: expected finite XYZ triple.")
    confidence = float(payload.get("confidence", 1.0))
    if not np.isfinite(confidence) or confidence < 0.0:
        raise ValueError(f"Invalid {key}.confidence: expected finite value >= 0.")
    diagnostics_payload = payload.get("diagnostics", {})
    return LightingLightData(
        name=name,
        kind=kind,
        role=role,
        strength=strength,
        color=color,
        casts_shadow=bool(payload.get("casts_shadow", False)),
        placement_mode=placement_mode,
        placement_target=placement_target,
        direction_world=direction_world,
        rotation_world=rotation_world,
        location_world=location_world,
        angular_size_deg=angular_size_deg,
        area_size=area_size,
        confidence=confidence,
        diagnostics=dict(diagnostics_payload or {}) if isinstance(diagnostics_payload, Mapping) else {},
    )


def lighting_to_payload(lighting: LightingData, *, envmap_relative_path: str) -> dict[str, Any]:
    metadata = dict(lighting.metadata or {})
    return {
        "provider": str(metadata.get("provider", "")),
        "schema_version": int(lighting.schema_version),
        "rig_mode": str(lighting.rig_mode or "envmap_only"),
        "mode": str(lighting.mode or "ambient_only"),
        "sun_direction_world": np.asarray(lighting.sun_direction_world, dtype=np.float32).reshape(3),
        "sun_strength": float(lighting.sun_strength),
        "sun_color": np.asarray(lighting.sun_color, dtype=np.float32).reshape(3),
        "envmap_path": envmap_relative_path,
        "envmap_rotation_world": np.asarray(lighting.envmap_rotation_world, dtype=np.float32).reshape(3),
        "ambient_strength": float(lighting.ambient_strength),
        "light_rig": [lighting_light_to_payload(light) for light in lighting.light_rig],
        "decomposition": dict(lighting.decomposition or {}),
        "quality": dict(lighting.quality or {}),
        "sun_diagnostics": dict(lighting.sun_diagnostics or {}),
        "validation": dict(lighting.validation or {}),
        "recovery": dict(lighting.recovery or {}),
        "selected_frame_indices": list(lighting.selected_frame_indices or []),
        "per_keyframe_diagnostics": list(lighting.per_keyframe_diagnostics or []),
        "metadata": metadata,
    }


def lighting_from_payload(
    payload: Mapping[str, Any],
    *,
    envmap_path: str,
    key: str = "lighting",
) -> LightingData:
    validation_payload = payload.get("validation", {})
    if not isinstance(validation_payload, Mapping):
        raise ValueError(f"Invalid {key}.validation: expected object.")
    rig_mode = str(payload.get("rig_mode", "")).strip().lower()
    if rig_mode not in {"analytic_rig", "sun_plus_fill", "envmap_only"}:
        raise ValueError(
            f"Invalid {key}.rig_mode: expected 'analytic_rig', 'sun_plus_fill', or 'envmap_only'."
        )
    schema_version = int(payload.get("schema_version", 0))
    if schema_version < 2:
        raise ValueError(f"Invalid {key}.schema_version: expected integer >= 2.")
    light_rig_raw = payload.get("light_rig", [])
    if not isinstance(light_rig_raw, list):
        raise ValueError(f"Invalid {key}.light_rig: expected list.")
    light_rig = [
        lighting_light_from_mapping(item, key=f"{key}.light_rig[{idx}]")
        for idx, item in enumerate(light_rig_raw)
        if isinstance(item, Mapping)
    ]
    if len(light_rig) != len(light_rig_raw):
        raise ValueError(f"Invalid {key}.light_rig: every entry must be an object.")
    decomposition_payload = payload.get("decomposition", {})
    quality_payload = payload.get("quality", {})
    sun_diagnostics_payload = payload.get("sun_diagnostics", {})
    recovery_payload = payload.get("recovery", {})
    metadata = dict(payload.get("metadata", {}) or {})
    metadata.setdefault("provider", str(payload.get("provider", metadata.get("provider", ""))))
    return LightingData(
        sun_direction_world=np.asarray(payload.get("sun_direction_world", (0.0, 0.0, 1.0)), dtype=np.float32).reshape(3),
        sun_strength=float(payload.get("sun_strength", 0.0)),
        sun_color=np.asarray(payload.get("sun_color", (1.0, 1.0, 1.0)), dtype=np.float32).reshape(3),
        envmap_path=str(envmap_path),
        envmap_rotation_world=np.asarray(payload.get("envmap_rotation_world", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(3),
        ambient_strength=float(payload.get("ambient_strength", 0.0)),
        mode=str(payload.get("mode", "ambient_only")),
        schema_version=schema_version,
        rig_mode=rig_mode,
        light_rig=light_rig,
        decomposition=dict(decomposition_payload or {}) if isinstance(decomposition_payload, Mapping) else {},
        quality={str(k): float(v) for k, v in dict(quality_payload or {}).items()},
        sun_diagnostics=dict(sun_diagnostics_payload or {}) if isinstance(sun_diagnostics_payload, Mapping) else {},
        validation=dict(validation_payload or {}),
        recovery=dict(recovery_payload or {}) if isinstance(recovery_payload, Mapping) else {},
        selected_frame_indices=[int(idx) for idx in payload.get("selected_frame_indices", []) or []],
        per_keyframe_diagnostics=[
            dict(item)
            for item in payload.get("per_keyframe_diagnostics", []) or []
            if isinstance(item, Mapping)
        ],
        metadata=metadata,
    )


@dataclass(slots=True)
class SemanticsAuxData:
    """Standardized per-frame semantics sidecars consumed by later stages."""

    frame_index: int
    class_probabilities: Optional[np.ndarray] = None
    class_ids: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    road_confidence: Optional[np.ndarray] = None
    validity_mask: Optional[np.ndarray] = None
    debug_maps: MutableMapping[str, np.ndarray] = field(default_factory=dict)
    model_outputs: MutableMapping[str, MutableMapping[str, np.ndarray]] = field(
        default_factory=dict
    )
    road_prior_outputs: MutableMapping[str, np.ndarray] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryMatchGraphData:
    """Canonical persisted match-graph payload consumed by later stages."""

    payload: MutableMapping[str, np.ndarray]
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticSegment:
    """Single semantic or panoptic segment predicted for a frame."""

    segment_id: int
    label: str
    score: float
    mask: np.ndarray
    label_id: Optional[int] = None
    area: Optional[int] = None
    bbox: Optional[Sequence[int]] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.mask.size == 0 or not bool(np.any(self.mask))


@dataclass(slots=True)
class SemanticsData:
    """Semantic segmentation result for a single frame."""

    frame_index: int
    segments: List[SemanticSegment]
    frame_id: Optional[str] = None
    segment_ids: Optional[np.ndarray] = None
    label_ids: Optional[np.ndarray] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def labels(self) -> List[str]:
        return [seg.label for seg in self.segments]

    def __str__(self) -> str:
        header_parts = [f"SemanticsData(frame_index={self.frame_index})"]
        if self.frame_id is not None:
            header_parts.append(f"frame_id={self.frame_id}")
        header_parts.append(f"num_segments={len(self.segments)}")
        header = " | ".join(header_parts)

        if not self.segments:
            return f"{header}\n  (no segments)"

        labels = [seg.label for seg in self.segments]
        unique_labels = sorted(set(labels))

        lines = [header, "-" * max(40, len(header))]
        for seg in self.segments:
            lines.append(
                f"  - SegmentID={seg.segment_id}  Label={seg.label!s}  Score={seg.score:.3f}  "
                f"Category={seg.label_id}  Area={seg.area}"
            )

        lines.append("")
        lines.append(f"Labels ({len(unique_labels)}): " + ", ".join(unique_labels))
        lines.append("")
        lines.append(f"Label IDs: {np.unique(self.label_ids)}")
        return "\n".join(lines)


@dataclass(slots=True)
class ConfidenceVolumeData:
    """Multi-model confidence volumes for semantic fusion."""

    log_probabilities: np.ndarray
    confidence: np.ndarray
    validity_mask: np.ndarray
    frame_id: int
    model_name: str
    timestamp: Optional[float] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


class ResourceKind(str, Enum):
    """Standard resource kinds produced/consumed by providers."""

    FRAMES = "frames"
    DEPTH = "depth"
    INTRINSICS = "intrinsics"
    TRAJECTORY = "trajectory"
    TRAJECTORY_MATCH_GRAPH = "trajectory_match_graph"
    SEMANTICS_2D = "semantics_2d"
    SEMANTICS_AUX = "semantics_aux"
    POINT_CLOUD_3D = "point_cloud_3d"
    CAMERA_HEIGHT = "camera_height"
    ROAD_PLANE = "road_plane"
    ROAD_PLANE_SUPPORT = "road_plane_support"
    DYNAMIC_MASK = "dynamic_mask"
    LIGHTING = "lighting"


@dataclass(frozen=True, slots=True)
class ResourceLayout:
    """Filesystem layout for a resource kind."""

    subdir: str
    pattern: Optional[str] = None
    filename: Optional[str] = None
    description: str = ""

    def path_for(self, root: Path, frame_index: Optional[int] = None) -> Path:
        base = root / self.subdir
        if self.filename:
            return base / self.filename
        if self.pattern and frame_index is not None:
            return base / self.pattern.format(frame=frame_index)
        return base
