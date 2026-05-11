"""Configuration dataclasses for quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, List, Mapping


@dataclass(frozen=True)
class TrajectoryMetricsSettings:
    """Settings for trajectory quality metrics (ATE, RPE, scale drift)."""

    enabled: bool = True
    rpe_deltas: List[int] = field(default_factory=lambda: [1, 5, 10])
    scale_drift_window: int = 20
    scale_drift_stride: int = 5
    umeyama_align: bool = True
    umeyama_with_scale: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> TrajectoryMetricsSettings:
        if not data:
            return cls()
        payload: dict[str, Any] = {}
        valid = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in valid:
                payload[k] = v
        return cls(**payload)


@dataclass(frozen=True)
class RoadMetricsSettings:
    """Settings for road model quality metrics."""

    enabled: bool = True
    residual_percentiles: List[float] = field(default_factory=lambda: [50.0, 90.0, 95.0, 99.0])
    normal_stability_window: int = 5
    smoothness_window: int = 10

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> RoadMetricsSettings:
        if not data:
            return cls()
        payload: dict[str, Any] = {}
        valid = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in valid:
                payload[k] = v
        return cls(**payload)


@dataclass(frozen=True)
class ArtifactSettings:
    """Settings for human-validation visualization artifacts."""

    enabled: bool = True
    reprojection_heatmaps: bool = True
    temporal_flicker: bool = True
    point_cloud_slices: bool = True
    road_model_overlay: bool = True
    confidence_overlay: bool = True
    max_frames: int = 16
    colormap: str = "viridis"
    slice_thickness_m: float = 1.0
    flicker_neighbor_frames: int = 5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ArtifactSettings:
        if not data:
            return cls()
        payload: dict[str, Any] = {}
        valid = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in valid:
                payload[k] = v
        return cls(**payload)


@dataclass(frozen=True)
class QualityMetricsSettings:
    """Root settings for the quality metrics module."""

    enabled: bool = True
    trajectory: TrajectoryMetricsSettings = field(default_factory=TrajectoryMetricsSettings)
    road: RoadMetricsSettings = field(default_factory=RoadMetricsSettings)
    artifacts: ArtifactSettings = field(default_factory=ArtifactSettings)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> QualityMetricsSettings:
        if not data:
            return cls()
        enabled = data.get("enabled", True)
        trajectory = TrajectoryMetricsSettings.from_mapping(data.get("trajectory"))
        road = RoadMetricsSettings.from_mapping(data.get("road"))
        artifacts = ArtifactSettings.from_mapping(data.get("artifacts"))
        return cls(
            enabled=enabled,
            trajectory=trajectory,
            road=road,
            artifacts=artifacts,
        )
