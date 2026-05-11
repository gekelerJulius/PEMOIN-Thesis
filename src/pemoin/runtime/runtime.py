"""
Runtime entry point that orchestrates frame processing through geometry providers.

This module contains the core PEMOIN runtime system, which coordinates the complete
pipeline lifecycle from frame ingestion to final output. The runtime manages:

1. **Initialization**: Profile loading, provider factory setup, resource store creation
2. **Frame Processing**: Per-frame execution through all configured providers
3. **Post-Processing**: Comparison-frame canonicalization, geometry validation, visualization
4. **Resource Management**: Disk-based storage and caching of pipeline outputs

## Pipeline Phases

### Phase 1: Initialization
- Load profile configuration from `config/profiles.json`
- Create provider factory and register provider builders
- Set up resource store for output persistence
- Initialize frame provider based on input source

### Phase 2: Frame Processing Loop
- Process each frame through all configured providers
- Convert coordinate systems to Blender convention
- Store results in resource store with standardized layout
- Maintain state cache for temporal consistency

### Phase 3: Post-Processing
- **Comparison-Frame Canonicalization**: Canonicalize complete clip geometry into the shared comparison frame
  - See: `canonicalize_geometry_to_comparison_frame()` for GT-vs-estimated world-frame normalization
- **Geometry Validation**: Validate consistency across all frames
  - See: `validate_geometry_store()` for quality checks
- **Visualization**: Generate debug plots and Blender scenes

### Phase 4: Cleanup
- Teardown providers and release resources
- Close frame provider and finalize outputs

## Key Components

### Runtime Class
Main orchestration class that manages the complete pipeline execution.

**Key Methods**:
- `run()`: Main pipeline execution entry point
- `build_providers()`: Create provider instances from profile bindings
- `_ensure_resource_store()`: Set up disk-based resource storage

### SceneStateCache
Manages the sliding window of frame states for temporal processing.

### Provider System
- **Provider Factory**: Creates provider instances based on profile
- **Provider Bindings**: Maps provider names to implementations in profile
- **Provider Lifecycle**: setup() → process() → teardown()

## Coordinate System Handling

The runtime implements a two-stage coordinate system pipeline:

1. **Provider-Level Conversion**: Each provider converts from its native convention
   to Blender convention during frame processing
   - Example: `convert_pose_opencv_to_blender()` in MegaSAM adapter

2. **Runtime-Level Alignment**: After all frames are processed, the runtime performs
  scene-specific comparison-frame canonicalization after geometry fusion
   - Example: GT and estimated profiles both land in the same grounded comparison frame

## Standard Output Layout

The runtime persists all outputs in a standardized directory structure:

```
outputs/<run>/
├── standard/
│   ├── frames/              # RGB frames
│   ├── depth/               # Depth maps
│   ├── intrinsics/          # Camera intrinsics
│   ├── trajectory/          # Camera poses
│   ├── semantics_2d/        # 2D semantics
│   ├── camera_height/       # Camera heights
│   ├── visualizations/      # Debug visualizations
│   └── logs/                # Log files
└── scene.blend              # Blender scene
```

## Entry Points

- **CLI Entry**: `src/pemoin/cli.py` - Main command-line interface
- **Runtime Entry**: `Runtime.run()` - Core pipeline execution
- **Provider Entry**: Provider factory creates instances from profile bindings

## Related Documentation

- **Overview**: `docs/system-overview.md`
- **Profiles**: `docs/profile-reference.md`
- **Data Contract**: `docs/data-contract.md`
- **Geometry**: `docs/geometry-reference.md`

## Key Functions

- `Runtime.run()`: Main pipeline execution
- `canonicalize_geometry_to_comparison_frame()`: maintained comparison-frame canonicalization
- `validate_geometry_store()`: Geometry validation
- `create_default_provider_factory()`: Provider factory setup
"""

from __future__ import annotations

import contextlib
import gc
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from pemoin.data.contracts import ResourceKind, ResourceStore
from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    SemanticsAuxData,
    SemanticsData,
)
from pemoin.providers import ProviderFactory, create_default_provider_factory
from pemoin.providers.base import ProviderExecutionMode
from pemoin.providers.semantic_roles import semantic_role_defaults_for_tool
from pemoin.runtime.cache import CrossRunCacheManager, RenderArtifactCacheManager
from pemoin.utils.instance_tracking import GeometryAwareInstanceTracker
from pemoin.utils.camera_calibration import validate_and_normalize_intrinsics
from pemoin.utils.geometry_validation import (
    GeometryValidationConfig,
    validate_geometry_store,
)
from pemoin.utils.logging import RuntimeStageRecord, RuntimeTimeline, iter_with_progress
from pemoin.coordinate_systems import (
    save_origin_anchored_trajectory,
)
from pemoin.coordinate_systems.alignment import (
    ComparisonFrameSettings,
    canonicalize_geometry_to_comparison_frame,
)
from pemoin.validation import (
    GeometryConsistencyValidationSettings,
    ValidationPolicySettings,
    validate_depth_pose_intrinsics_consistency,
)
from pemoin.utils.resolution import (
    normalize_frame_resolution,
    resize_depth,
    resize_semantics,
    _resize_array,
)
from pemoin.visualization.video import (
    VideoExportSettings,
    copy_canonical_output_video,
    generate_flat_video_from_dir,
    generate_visualization_videos,
)
from pemoin.visualization.blender_runner import (
    build_blender_trajectory_command,
    render_trajectory_scene,
    validate_blender_scene_inputs,
)
from pemoin.visualization.semantics import (
    SemanticsVisualizationSettings,
    generate_semantics_visualizations,
)
from pemoin.visualization.semantics_debug import (
    SemanticsDebugSettings,
    generate_semantics_debug_visualizations,
)
from pemoin.visualization.harmonized_overlay_grid import (
    generate_harmonized_ground_grid_video,
)
from pemoin.visualization.pedestrian_placement import (
    detect_mixamo_animation_motion_category,
)
from pemoin.utils.harmonisation import HarmonisationSettings, run_harmonisation

from .context import RuntimeContext
from .orchestration.frame_provider import FrameProvider
from .orchestration.state_cache import SceneFrameState, SceneStateCache
from .profiles.config import ProfileConfig

LOG = logging.getLogger(__name__)

_GT_TRAJECTORY_TOOLS = frozenset(
    {
        "UnityGTTrajectoryProvider",
        "CarlaTrajectoryProvider",
        "NuScenesTrajectoryProvider",
    }
)


@dataclass(slots=True)
class RuntimeResult:
    """Structured result returned by ``Runtime.run()``."""

    state_cache: SceneStateCache
    processed_frames: int
    expected_frames: Optional[int]
    provider_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class _ProviderTimingAggregate:
    name: str
    display_name: str
    duration_s: float = 0.0
    calls: int = 0
    first_started_at: str | None = None
    last_ended_at: str | None = None

    def add_sample(self, *, started_at: str, ended_at: str, duration_s: float) -> None:
        if self.first_started_at is None:
            self.first_started_at = started_at
        self.last_ended_at = ended_at
        self.duration_s += float(duration_s)
        self.calls += 1


def _cleanup_cuda_memory() -> None:
    """
    Free any lingering CUDA allocations to give providers a clean slate at startup.
    """
    # Avoid importing torch here, which can initialize CUDA in the runtime
    # process and consume VRAM that subprocess-based providers (UniDepth/DPVO)
    # need. Only clean if torch is already loaded and CUDA is initialized.
    torch = sys.modules.get("torch")
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    with contextlib.suppress(Exception):
        if not cuda.is_available():
            return
    with contextlib.suppress(Exception):
        if not cuda.is_initialized():
            return
    try:
        gc.collect()
    except Exception:
        return
    LOG.debug("Releasing unused CUDA memory before pipeline start.")
    with contextlib.suppress(Exception):
        cuda.empty_cache()
    with contextlib.suppress(Exception):
        cuda.ipc_collect()


def _normalize_mapping(obj: Any) -> Any:
    """Convert profile/settings objects into JSON-serialisable payloads."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _normalize_mapping(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_mapping(item) for item in obj]
    return obj


def _cross_run_cache_stage_settings(
    runtime_settings: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    raw = {}
    if isinstance(runtime_settings, Mapping):
        raw = runtime_settings.get("cross_run_cache", {}) or {}
    stage_settings: dict[str, Mapping[str, Any]] = {}
    for name in ("lighting", "geometry_fusion", "blender_scene", "harmonisation", "ground_grid"):
        value = raw.get(name)
        stage_settings[name] = value if isinstance(value, Mapping) else {}
    return stage_settings


def _expected_frame_count(
    frame_provider, resource_store: Optional[ResourceStore]
) -> Optional[int]:
    """Infer how many frames should exist for this run."""
    count: Optional[int] = None
    length_fn = getattr(frame_provider, "__len__", None)
    if callable(length_fn):
        try:
            count = len(frame_provider)  # type: ignore[arg-type]
        except Exception:
            count = None
    if resource_store is not None and resource_store.has(ResourceKind.FRAMES):
        stored = len(resource_store.frame_indices(ResourceKind.FRAMES))
        if stored > 0:
            count = max(count or 0, stored)
    return count


def _parse_working_resolution(value: object) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            size = int(value[0])
            return (size, size)
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
    if isinstance(value, (int, float)):
        size = int(value)
        return (size, size)
    return None


def _reject_legacy_semantic_label_settings(
    runtime_settings: Mapping[str, Any] | None,
    semantics_settings: Mapping[str, Any] | None = None,
) -> None:
    if isinstance(runtime_settings, Mapping) and runtime_settings.get("road_labels") is not None:
        raise ValueError("runtime.settings.road_labels is no longer supported.")
    if isinstance(runtime_settings, Mapping) and runtime_settings.get("mobile_labels") is not None:
        raise ValueError("runtime.settings.mobile_labels is no longer supported.")
    if isinstance(runtime_settings, Mapping) and runtime_settings.get("sidewalk_labels") is not None:
        raise ValueError("runtime.settings.sidewalk_labels is no longer supported.")
    if isinstance(semantics_settings, Mapping) and semantics_settings.get("dynamic_labels") is not None:
        raise ValueError("providers.semantics.settings.dynamic_labels is no longer supported.")
    for key in ("road_labels", "mobile_labels", "sidewalk_labels", "sky_labels", "large_vehicle_labels"):
        if isinstance(semantics_settings, Mapping) and semantics_settings.get(key) is not None:
            raise ValueError(
                f"providers.semantics.settings.{key} is no longer supported; semantic roles come from provider defaults."
            )


def _semantic_role_labels_from_runtime(
    role: str,
    runtime_settings: Mapping[str, Any] | None,
    semantics_tool: str | None,
    semantics_settings: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    _reject_legacy_semantic_label_settings(runtime_settings, semantics_settings)
    labels = tuple(
        semantic_role_defaults_for_tool(semantics_tool).get(str(role).strip().lower(), ())
    )
    if labels:
        return labels
    if role == "road":
        return ("road",)
    if role == "mobile":
        return ("human", "car")
    if role == "sidewalk":
        return ("sidewalk",)
    return ()


def _provider_metadata_payload(
    binding,
    *,
    frame_source: object,
    frame_provider_info: Mapping[str, object] | None,
    working_resolution: Optional[tuple[int, int]],
    max_frames: Optional[int],
) -> Dict[str, Any]:
    """Create a deterministic metadata payload describing provider inputs."""
    return {
        "tool": binding.tool,
        "settings": _normalize_mapping(binding.settings),
        "frame_source": str(frame_source) if frame_source is not None else None,
        "frame_provider": _normalize_mapping(frame_provider_info or {}),
        "working_resolution": working_resolution,
        "max_frames": max_frames,
    }


def _resolve_sampling_fps(provider_context: Mapping[str, Any]) -> float:
    """Resolve video FPS from runtime frame provider settings."""
    frame_provider_info = provider_context.get("frame_provider_info")
    if not isinstance(frame_provider_info, Mapping):
        raise ValueError(
            "frame_provider_info must be provided in runtime context when video export is enabled."
        )
    settings = frame_provider_info.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError(
            "frame_provider_info.settings must be provided when video export is enabled."
        )
    sampling_fps = settings.get("resolved_sampling_fps")
    if sampling_fps is None:
        sampling_fps = settings.get("sampling_fps")
    if sampling_fps is None:
        sampling_fps = settings.get("source_sampling_fps")
    if sampling_fps is None:
        sampling_fps = settings.get("frame_rate")
    if sampling_fps is None:
        sampling_fps = settings.get("frame_rate_hint")
    if sampling_fps is None:
        raise ValueError(
            "frame_provider_info.settings must resolve video FPS when video export is enabled."
        )
    fps = float(sampling_fps)
    if fps <= 0.0:
        raise ValueError(
            "frame_provider_info.settings must resolve a video FPS > 0 when video export is enabled."
        )
    return fps


class Runtime:
    """Coordinates frame ingestion and provider execution for geometry outputs."""

    def __init__(self, profile: ProfileConfig):
        self.profile = profile
        semantics_settings: Mapping[str, Any] | None = None
        semantics_binding = profile.providers.get("semantics")
        semantics_tool = semantics_binding.tool if semantics_binding is not None else None
        self._semantics_tool = semantics_tool
        if semantics_binding is not None and isinstance(semantics_binding.settings, Mapping):
            semantics_settings = semantics_binding.settings
        runtime_res = (
            profile.runtime.settings.get("working_resolution")
            if isinstance(profile.runtime.settings, Mapping)
            else None
        )
        working_res = profile.working_resolution or runtime_res
        self._working_resolution = _parse_working_resolution(working_res)
        self._semantic_role_defaults = semantic_role_defaults_for_tool(semantics_tool)
        self._road_labels = _semantic_role_labels_from_runtime(
            "road",
            profile.runtime.settings if isinstance(profile.runtime.settings, Mapping) else None,
            semantics_tool,
            semantics_settings,
        )
        self._mobile_labels = _semantic_role_labels_from_runtime(
            "mobile",
            profile.runtime.settings if isinstance(profile.runtime.settings, Mapping) else None,
            semantics_tool,
            semantics_settings,
        )
        self._sidewalk_labels = _semantic_role_labels_from_runtime(
            "sidewalk",
            profile.runtime.settings if isinstance(profile.runtime.settings, Mapping) else None,
            semantics_tool,
            semantics_settings,
        )
        comparison_frame_raw: Mapping[str, Any] | None = None
        consistency_raw: Mapping[str, Any] | None = None
        validation_policy_raw: Mapping[str, Any] | None = None
        if isinstance(profile.runtime.settings, Mapping):
            candidate = profile.runtime.settings.get("comparison_frame", {})
            if isinstance(candidate, Mapping):
                comparison_frame_raw = candidate
            candidate = profile.runtime.settings.get("geometry_consistency_validation", {})
            if isinstance(candidate, Mapping):
                consistency_raw = candidate
            candidate = profile.runtime.settings.get("validation_policy", {})
            if isinstance(candidate, Mapping):
                validation_policy_raw = candidate
        self._comparison_frame_settings = ComparisonFrameSettings.from_mapping(comparison_frame_raw)
        self._consistency_settings = GeometryConsistencyValidationSettings.from_mapping(
            consistency_raw
        )
        self._validation_policy = ValidationPolicySettings.from_mapping(validation_policy_raw)

        from pemoin.providers.depth_stabilization import DepthStabilizationSettings

        depth_stab_raw: Mapping[str, Any] | None = None
        if isinstance(profile.runtime.settings, Mapping):
            candidate = profile.runtime.settings.get("depth_stabilization", {})
            if isinstance(candidate, Mapping):
                depth_stab_raw = candidate
        self._depth_stabilization_settings = DepthStabilizationSettings.from_mapping(
            depth_stab_raw
        )

        from pemoin.utils.trajectory_cleanup import PoseConditioningSettings

        pose_cond_raw: Mapping[str, Any] | None = None
        if isinstance(profile.runtime.settings, Mapping):
            candidate = profile.runtime.settings.get("pose_conditioning", {})
            if isinstance(candidate, Mapping):
                pose_cond_raw = candidate
        self._pose_conditioning_settings = PoseConditioningSettings.from_mapping(
            pose_cond_raw
        )
        self._validate_comparison_frame_policy()

    @classmethod
    def from_registry(cls, registry, name: str) -> "Runtime":
        return cls(profile=registry.get(name))

    def _validate_comparison_frame_policy(self) -> None:
        if not self._comparison_frame_settings.enabled:
            return
        geometry_fusion_binding = self.profile.providers.get("geometry_fusion")
        if geometry_fusion_binding is None:
            raise ValueError(
                "runtime.settings.comparison_frame requires providers.geometry_fusion."
            )
        trajectory_binding = self.profile.providers.get("trajectory")
        if trajectory_binding is None:
            raise ValueError(
                "runtime.settings.comparison_frame requires a configured trajectory provider."
            )
        if self._comparison_frame_settings.mode == "gt" and trajectory_binding.tool not in _GT_TRAJECTORY_TOOLS:
            raise ValueError(
                "runtime.settings.comparison_frame.mode='gt' requires a GT trajectory provider, "
                f"got {trajectory_binding.tool!r}."
            )

    def _export_videos(
        self,
        *,
        resource_store: ResourceStore,
        provider_context: Mapping[str, Any],
        timeline: RuntimeTimeline | None = None,
        timeline_parent: RuntimeStageRecord | None = None,
    ) -> None:
        """Generate post-processing MP4 outputs for this run."""
        video_settings = {}
        if isinstance(self.profile.runtime.settings, Mapping):
            video_settings = dict(
                self.profile.runtime.settings.get("video_export", {}) or {}
            )
        video_override = provider_context.get("video_export_override")
        if video_override:
            video_settings.update(video_override)
        video_enabled = bool(video_settings.get("enabled", True))
        video_fps = _resolve_sampling_fps(provider_context) if video_enabled else 1.0
        video_config = VideoExportSettings.from_mapping(video_settings, fps=video_fps)
        render_artifact_cache = provider_context.get("render_artifact_cache")
        if not isinstance(render_artifact_cache, RenderArtifactCacheManager):
            render_artifact_cache = None
        if not video_config.enabled:
            if timeline is not None:
                timeline.add_completed_stage(
                    "runtime.post.video_export",
                    display_name="Video export",
                    status="disabled",
                    duration_s=0.0,
                    metadata={"enabled": False},
                    parent=timeline_parent,
                )
            return

        vis_root = resource_store.visualizations_dir()
        videos_dir = resource_store.videos_dir()
        harmonized_dir_name = "artifacts/harmonisation/harmonized_overlays"
        blender_scene_enabled = False
        harmonisation_enabled = False
        if isinstance(self.profile.runtime.settings, Mapping):
            blender_scene_settings = (
                self.profile.runtime.settings.get("blender_scene", {}) or {}
            )
            if isinstance(blender_scene_settings, Mapping):
                blender_scene_enabled = bool(blender_scene_settings.get("enabled", False))
            harmonisation_settings = (
                self.profile.runtime.settings.get("harmonisation", {}) or {}
            )
            if isinstance(harmonisation_settings, Mapping) and harmonisation_settings.get("output_dir"):
                harmonized_dir_name = str(harmonisation_settings.get("output_dir"))
            if isinstance(harmonisation_settings, Mapping):
                harmonisation_enabled = bool(harmonisation_settings.get("enabled", False))

        run_root = resource_store.root

        def _resolve_optional_flat_export_source(
            canonical_source_dir: Path,
            legacy_source_dir: Path,
        ) -> Path:
            if canonical_source_dir.exists():
                return canonical_source_dir
            return legacy_source_dir

        optional_flat_exports = [
            (
                "overlayed_frames",
                _resolve_optional_flat_export_source(
                    resource_store.blender_artifacts_dir("overlayed_frames"),
                    run_root / "overlayed_frames",
                ),
                blender_scene_enabled,
            ),
            (
                "shadow_frames",
                _resolve_optional_flat_export_source(
                    resource_store.blender_artifacts_dir("shadow_frames"),
                    run_root / "shadow_frames",
                ),
                blender_scene_enabled,
            ),
            (
                "overlayed_frames_support_local_grid",
                _resolve_optional_flat_export_source(
                    resource_store.blender_artifacts_dir("overlayed_frames_support_local_grid"),
                    run_root / "overlayed_frames_support_local_grid",
                ),
                blender_scene_enabled,
            ),
            (
                "harmonized_overlays",
                _resolve_optional_flat_export_source(
                    run_root / harmonized_dir_name,
                    run_root / "harmonized_overlays",
                ),
                harmonisation_enabled,
            ),
        ]

        video_scope_cm = (
            timeline.stage(
                "runtime.post.video_export",
                display_name="Video export",
                metadata={"enabled": True, "fps": float(video_config.fps)},
            )
            if timeline is not None
            else contextlib.nullcontext(None)
        )
        with video_scope_cm as _video_scope:
            vis_scope_cm = (
                timeline.stage(
                    "video_export.visualization_videos",
                    display_name="Visualization videos",
                    metadata={"visualizations_dir": str(vis_root), "videos_dir": str(videos_dir)},
                )
                if timeline is not None
                else contextlib.nullcontext(None)
            )
            with vis_scope_cm:
                generated = generate_visualization_videos(vis_root, videos_dir, video_config)
            if generated:
                LOG.info(
                    "Generated %d visualization video(s) in %s",
                    len(generated),
                    videos_dir,
                )

            skipped_not_expected: list[str] = []
            expected_missing: list[str] = []
            generated_flat_exports = 0
            flat_scope_cm = (
                timeline.stage(
                    "video_export.flat_exports",
                    display_name="Flat video exports",
                )
                if timeline is not None
                else contextlib.nullcontext(None)
            )
            with flat_scope_cm as flat_scope:
                for name, source_dir, expected in optional_flat_exports:
                    if not expected:
                        skipped_not_expected.append(name)
                        continue
                    if not source_dir.exists():
                        expected_missing.append(f"{name}={source_dir}")
                        continue
                    generate_flat_video_from_dir(
                        source_dir,
                        videos_dir,
                        video_config,
                        name=name,
                    )
                    generated_flat_exports += 1
                if flat_scope is not None:
                    flat_scope.record.metadata.update(
                        {
                            "generated_exports": generated_flat_exports,
                            "skipped_not_expected": skipped_not_expected,
                            "missing_expected": expected_missing,
                        }
                    )
            if skipped_not_expected:
                LOG.debug(
                    "Skipping optional flat-video exports not expected for this run: %s",
                    ", ".join(skipped_not_expected),
                )
            if expected_missing:
                LOG.warning(
                    "Optional flat-video sources expected for this run were missing: %s",
                    "; ".join(expected_missing),
                )

            harmonized_dir = run_root / harmonized_dir_name
            ground_grid_settings = video_settings.get("ground_grid", {}) or {}
            if not isinstance(ground_grid_settings, Mapping):
                ground_grid_settings = {}
            ground_grid_num_workers = int(ground_grid_settings.get("num_workers", 0) or 0)
            ground_grid_output_path = videos_dir / "harmonized_overlays_ground_grid.mp4"
            ground_grid_scope_cm = (
                timeline.stage(
                    "video_export.ground_grid",
                    display_name="Ground-grid video",
                )
                if timeline is not None
                else contextlib.nullcontext(None)
            )
            with ground_grid_scope_cm as ground_grid_scope:
                if harmonized_dir.exists() and resource_store.has(ResourceKind.ROAD_PLANE):
                    overlay_extent_m = 30.0
                    road_plane_binding = self.profile.providers.get("road_plane")
                    if (
                        road_plane_binding is not None
                        and isinstance(road_plane_binding.settings, Mapping)
                        and "overlay_extent_m" in road_plane_binding.settings
                    ):
                        overlay_extent_m = float(road_plane_binding.settings["overlay_extent_m"])
                    try:
                        cache_enabled = (
                            render_artifact_cache is not None
                            and render_artifact_cache.enabled_for("ground_grid")
                        )
                        ground_grid_payload = None
                        ground_grid_signature = None
                        cache_hit = False
                        ground_grid_relpath = str(ground_grid_output_path.relative_to(run_root))
                        if cache_enabled:
                            ground_grid_payload = self._ground_grid_bundle_payload(
                                resource_store=resource_store,
                                source_dir=harmonized_dir,
                                output_path=ground_grid_output_path,
                                fps=video_config.fps,
                                codec=video_config.codec,
                                grid_spacing_m=0.2,
                                extent_m=overlay_extent_m,
                                min_frames=video_config.min_frames,
                                road_labels=self._road_labels,
                                num_workers=ground_grid_num_workers,
                                render_cache=render_artifact_cache,
                            )
                            ground_grid_signature = render_artifact_cache.signature(
                                "ground_grid_video",
                                ground_grid_payload,
                            )
                            lookup_scope_cm = (
                                timeline.stage(
                                    "video_export.ground_grid.cache_lookup",
                                    display_name="Ground-grid cache lookup",
                                )
                                if timeline is not None
                                else contextlib.nullcontext(None)
                            )
                            with lookup_scope_cm as lookup_scope:
                                lookup = render_artifact_cache.lookup(
                                    "ground_grid_video",
                                    ground_grid_signature,
                                    required_relpaths=[ground_grid_relpath],
                                )
                                cache_hit = bool(lookup.hit)
                                if lookup_scope is not None:
                                    lookup_scope.record.metadata.update(
                                        {"hit": lookup.hit, "reason": lookup.reason}
                                    )
                                if ground_grid_scope is not None:
                                    ground_grid_scope.record.metadata.update(
                                        {
                                            "cross_run_cache_enabled": True,
                                            "cross_run_cache_hit": lookup.hit,
                                            "cross_run_cache_validation": lookup.reason,
                                            "cross_run_cache_signature": ground_grid_signature,
                                            "cross_run_cache_entry": str(lookup.entry_dir),
                                        }
                                    )
                            if cache_hit:
                                materialize_scope_cm = (
                                    timeline.stage(
                                        "video_export.ground_grid.cache_materialize",
                                        display_name="Ground-grid cache materialize",
                                    )
                                    if timeline is not None
                                    else contextlib.nullcontext(None)
                                )
                                with materialize_scope_cm as materialize_scope:
                                    materialized = render_artifact_cache.materialize(
                                        "ground_grid_video",
                                        ground_grid_signature,
                                        run_root=run_root,
                                    )
                                    if materialize_scope is not None:
                                        materialize_scope.record.metadata["materialized_files"] = materialized
                                    if ground_grid_scope is not None:
                                        ground_grid_scope.record.metadata["cross_run_cache_materialized"] = materialized
                        elif ground_grid_scope is not None:
                            ground_grid_scope.record.metadata.update(
                                {
                                    "cross_run_cache_enabled": False,
                                    "cross_run_cache_hit": False,
                                    "cross_run_cache_validation": "disabled",
                                }
                            )
                        if not cache_hit:
                            generate_harmonized_ground_grid_video(
                                resource_store,
                                harmonized_dir,
                                ground_grid_output_path,
                                fps=video_config.fps,
                                codec=video_config.codec,
                                grid_spacing_m=0.2,
                                extent_m=overlay_extent_m,
                                min_frames=video_config.min_frames,
                                road_labels=self._road_labels,
                                num_workers=ground_grid_num_workers,
                            )
                            if (
                                cache_enabled
                                and ground_grid_payload is not None
                                and ground_grid_signature is not None
                                and ground_grid_output_path.exists()
                            ):
                                publish_scope_cm = (
                                    timeline.stage(
                                        "video_export.ground_grid.cache_publish",
                                        display_name="Ground-grid cache publish",
                                    )
                                    if timeline is not None
                                    else contextlib.nullcontext(None)
                                )
                                with publish_scope_cm:
                                    render_artifact_cache.publish(
                                        "ground_grid_video",
                                        ground_grid_signature,
                                        payload=ground_grid_payload,
                                        artifacts=self._ground_grid_bundle_artifacts(
                                            run_root,
                                            ground_grid_output_path,
                                            render_artifact_cache,
                                        ),
                                        source_summary={
                                            "run_root": str(run_root),
                                            "source_dir": str(harmonized_dir),
                                        },
                                        provenance={
                                            "road_plane_dir": str(resource_store.base_dir(ResourceKind.ROAD_PLANE)),
                                            "trajectory_path": str(resource_store.path_for(ResourceKind.TRAJECTORY)),
                                        },
                                    )
                        if ground_grid_scope is not None:
                            ground_grid_scope.record.metadata["generated"] = True
                            ground_grid_scope.record.metadata["num_workers"] = max(1, ground_grid_num_workers)
                    except Exception as exc:
                        if ground_grid_scope is not None:
                            ground_grid_scope.record.metadata["generated"] = False
                            ground_grid_scope.record.metadata["error"] = str(exc)
                        LOG.error(
                            "Failed to generate harmonized ground-grid overlay video: %s",
                            exc,
                        )
                elif ground_grid_scope is not None:
                    ground_grid_scope.finish(
                        status="skipped",
                        metadata={"reason": "harmonized-overlays-or-road-plane-missing"},
                    )

            canonical_video_candidates = [
                videos_dir / "harmonized_overlays.mp4",
                videos_dir / "overlayed_frames.mp4",
            ]
            canonical_scope_cm = (
                timeline.stage(
                    "video_export.canonical_copy",
                    display_name="Canonical output copy",
                )
                if timeline is not None
                else contextlib.nullcontext(None)
            )
            with canonical_scope_cm as canonical_scope:
                for candidate in canonical_video_candidates:
                    if not candidate.exists():
                        continue
                    output_mp4 = copy_canonical_output_video(
                        candidate,
                        resource_store.output_video_path(),
                    )
                    if canonical_scope is not None:
                        canonical_scope.record.metadata.update(
                            {"source": str(candidate), "output": str(output_mp4)}
                        )
                    LOG.info("Copied canonical output video to %s", output_mp4)
                    break
                else:
                    if canonical_scope is not None:
                        canonical_scope.finish(
                            status="skipped",
                            metadata={"reason": "canonical-video-source-missing"},
                        )
                    LOG.warning(
                        "Canonical output video was not created because neither %s nor %s exists.",
                        canonical_video_candidates[0],
                        canonical_video_candidates[1],
                    )

    def run(
        self,
        frame_provider: FrameProvider,
        *,
        provider_factory: Optional[ProviderFactory] = None,
        context: MutableMapping[str, Any] | None = None,
        max_frames: Optional[int] = None,
        state_cache: Optional[SceneStateCache] = None,
        on_frame: Optional[Callable[[SceneFrameState], None]] = None,
    ) -> RuntimeResult:
        _cleanup_cuda_memory()
        factory = provider_factory or create_default_provider_factory()
        provider_context = RuntimeContext.coerce(context)
        if self._working_resolution:
            provider_context.setdefault("working_resolution", self._working_resolution)
        provider_context.setdefault("semantic_role_defaults", dict(self._semantic_role_defaults))
        provider_context.setdefault("semantics_tool", self._semantics_tool)
        provider_context.setdefault("validation_policy", self._validation_policy.to_mapping())
        provider_context.setdefault(
            "cross_run_cache_stage_settings",
            _cross_run_cache_stage_settings(self.profile.runtime.settings),
        )
        provider_context.setdefault(
            "cross_run_cache",
            CrossRunCacheManager.from_runtime_settings(
                self.profile.runtime.settings,
                base_root=Path.cwd(),
            ),
        )
        provider_context.setdefault(
            "render_artifact_cache",
            RenderArtifactCacheManager.from_runtime_settings(
                self.profile.runtime.settings,
                base_root=Path.cwd(),
            ),
        )
        resource_store = self._ensure_resource_store(provider_context)
        expected_frames = _expected_frame_count(frame_provider, resource_store)
        if max_frames is not None:
            expected_frames = min(expected_frames, int(max_frames))
        provider_context.setdefault("expected_frame_count", expected_frames)
        cross_run_cache = provider_context.get("cross_run_cache")
        render_artifact_cache = provider_context.get("render_artifact_cache")
        providers = self.build_providers(factory, provider_context)
        LOG.info(
            "Runtime comparison-frame config: enabled=%s mode=%s ground_source=%s",
            bool(self._comparison_frame_settings.enabled),
            self._comparison_frame_settings.mode,
            self._comparison_frame_settings.ground_source,
        )
        cache = state_cache or SceneStateCache(self.profile.runtime.state_window or 1)
        frame_source = provider_context.frame_source
        frame_provider_info = provider_context.get("frame_provider_info")
        provider_metadata: Dict[str, Dict[str, Any]] = {}
        published_cache_metadata: Dict[str, Dict[str, Any]] = {}
        runtime_stage_metadata: Dict[str, Dict[str, Any]] = {}
        provider_timing: dict[str, _ProviderTimingAggregate] = {}
        timeline = RuntimeTimeline(logger=LOG)
        timeline.set_metadata(
            profile_name=self.profile.name,
            frame_source=str(frame_source) if frame_source is not None else None,
            expected_frames=expected_frames,
            run_dir=str(provider_context.get("run_dir")) if provider_context.get("run_dir") is not None else None,
        )
        if resource_store is not None:
            for name, binding in self.profile.providers.items():
                provider_metadata[name] = _provider_metadata_payload(
                    binding,
                    frame_source=frame_source,
                    frame_provider_info=frame_provider_info
                    if isinstance(frame_provider_info, Mapping)
                    else {},
                    working_resolution=self._working_resolution,
                    max_frames=max_frames,
                )

        intr_provider = providers.get("intrinsics")
        depth_provider = providers.get("depth")
        trajectory_provider = providers.get("trajectory")
        semantics_provider = providers.get("semantics")
        camera_height_provider = providers.get("camera_height")
        point_cloud_provider = providers.get("point_cloud_3d")
        road_plane_provider = providers.get("road_plane")
        lighting_provider = providers.get("lighting")
        batch_semantics_provider = None
        if semantics_provider is not None and (
            getattr(semantics_provider, "execution_mode", ProviderExecutionMode.PER_FRAME)
            == ProviderExecutionMode.BATCH
            or bool(getattr(semantics_provider, "batch_oriented", False))
        ):
            batch_semantics_provider = semantics_provider
            semantics_provider = None
        semantics_tracker = (
            GeometryAwareInstanceTracker()
            if semantics_provider is not None
            else None
        )
        # Providers that declare deferred_batch=True are skipped in the per-frame
        # loop and run only after all sampled frames have been persisted. This is
        # required for adapters such as DPVO/UniDepth that consume the full
        # standard/frames sequence from disk rather than the in-memory frame.
        # When batch semantics are present, they still run first so deferred
        # providers can consume any semantics-derived artifacts (for example
        # DPVO dynamic masks).
        defer_trajectory = (
            trajectory_provider is not None
            and (
                getattr(trajectory_provider, "execution_mode", ProviderExecutionMode.PER_FRAME)
                == ProviderExecutionMode.DEFERRED_BATCH
                or bool(getattr(trajectory_provider, "deferred_batch", False))
            )
        )
        defer_depth = (
            depth_provider is not None
            and (
                getattr(depth_provider, "execution_mode", ProviderExecutionMode.PER_FRAME)
                == ProviderExecutionMode.DEFERRED_BATCH
                or bool(getattr(depth_provider, "deferred_batch", False))
            )
        )
        intr_cache: Optional[IntrinsicsData] = None
        intr_dynamic: Optional[bool] = None
        semantics_viz_config: Optional[SemanticsVisualizationSettings] = None
        semantics_debug_config: Optional[SemanticsDebugSettings] = None
        if resource_store is not None and isinstance(self.profile.runtime.settings, Mapping):
            semantics_viz_settings = dict(
                self.profile.runtime.settings.get("semantics_visualization", {}) or {}
            )
            semantics_debug_settings = dict(
                self.profile.runtime.settings.get("semantics_debug", {}) or {}
            )
            semantics_debug_settings["road_label_tokens"] = list(self._road_labels)
            semantics_viz_config = SemanticsVisualizationSettings.from_mapping(
                semantics_viz_settings
            )
            semantics_debug_config = SemanticsDebugSettings.from_mapping(
                semantics_debug_settings
            )

        def _run_semantics_outputs(frame_indices: Optional[Sequence[int]]) -> None:
            if resource_store is None:
                return
            if semantics_viz_config is not None and semantics_viz_config.enabled:
                try:
                    generate_semantics_visualizations(
                        resource_store,
                        semantics_viz_config,
                        frame_indices=frame_indices,
                    )
                except Exception as exc:
                    LOG.error(
                        "Failed to generate semantics visualizations: %s",
                        exc,
                        exc_info=True,
                    )
            if semantics_debug_config is not None and semantics_debug_config.enabled:
                try:
                    generate_semantics_debug_visualizations(
                        resource_store,
                        semantics_debug_config,
                        frame_indices=frame_indices,
                    )
                except Exception as exc:
                    LOG.error(
                        "Failed to generate semantics debug visualizations: %s",
                        exc,
                        exc_info=True,
                    )

        def _publish_provider_cache(name: str, provider: Any, *, phase: str) -> None:
            if (
                resource_store is None
                or not isinstance(cross_run_cache, CrossRunCacheManager)
                or not cross_run_cache.enabled
            ):
                return
            spec_fn = getattr(provider, "get_cross_run_cache_spec", None)
            if not callable(spec_fn):
                return
            try:
                spec = spec_fn(resource_store)
            except Exception as exc:
                published_cache_metadata[name] = {
                    "cross_run_cache_publish": "error",
                    "cross_run_cache_publish_phase": phase,
                    "cross_run_cache_publish_error": str(exc),
                }
                return
            if not isinstance(spec, Mapping):
                return
            ready = spec.get("ready", True)
            if not isinstance(ready, bool):
                ready = bool(ready)
            if not ready:
                published_cache_metadata.setdefault(
                    name,
                    {
                        "cross_run_cache_publish": "not-ready",
                        "cross_run_cache_publish_phase": phase,
                        "cross_run_cache_publish_reason": str(
                            spec.get("not_ready_reason", "not-ready")
                        ),
                    },
                )
                return
            provider_id = spec.get("provider_id")
            signature = spec.get("signature")
            payload = spec.get("payload")
            artifacts = spec.get("artifacts")
            if (
                not isinstance(provider_id, str)
                or not provider_id
                or not isinstance(signature, str)
                or not signature
                or not isinstance(payload, Mapping)
                or not isinstance(artifacts, Mapping)
            ):
                published_cache_metadata[name] = {
                    "cross_run_cache_publish": "invalid-spec",
                    "cross_run_cache_publish_phase": phase,
                }
                return
            publish_result = cross_run_cache.publish(
                provider_id,
                signature,
                payload=payload,
                artifacts=artifacts,
                source_summary=spec.get("source_summary")
                if isinstance(spec.get("source_summary"), Mapping)
                else None,
                provenance=spec.get("provenance")
                if isinstance(spec.get("provenance"), Mapping)
                else None,
            )
            metadata = published_cache_metadata.setdefault(name, {})
            metadata["cross_run_cache_publish"] = str(
                publish_result.get("reason", "unknown")
            )
            metadata["cross_run_cache_publish_entry"] = str(
                publish_result.get("entry_dir", "")
            )
            metadata["cross_run_cache_publish_phase"] = phase
            if (
                str(publish_result.get("reason", "")) == "published"
                and "cross_run_cache_first_publish_phase" not in metadata
            ):
                metadata["cross_run_cache_first_publish_phase"] = phase

        def _record_runtime_stage_metadata(stage_name: str, values: Mapping[str, Any]) -> None:
            metadata = runtime_stage_metadata.setdefault(stage_name, {})
            metadata.update(dict(values))

        def _try_materialize_standardized_outputs(provider: Any) -> bool:
            if resource_store is None:
                return False
            materialize_fn = getattr(provider, "try_materialize_standardized_outputs", None)
            if not callable(materialize_fn):
                return False
            try:
                return bool(materialize_fn(resource_store))
            except Exception:
                return False

        def _record_provider_timing_sample(
            name: str,
            *,
            duration_s: float,
            started_at: str,
            ended_at: str,
        ) -> None:
            aggregate = provider_timing.get(name)
            if aggregate is None:
                aggregate = _ProviderTimingAggregate(
                    name=f"runtime.frame_loop.providers.{name}",
                    display_name=f"Provider: {name}",
                )
                provider_timing[name] = aggregate
            aggregate.add_sample(
                started_at=started_at,
                ended_at=ended_at,
                duration_s=duration_s,
            )

        @contextlib.contextmanager
        def _time_provider_call(name: str):
            started_at = datetime.now(timezone.utc).isoformat()
            started_perf = time.perf_counter()
            try:
                yield
            finally:
                ended_at = datetime.now(timezone.utc).isoformat()
                duration_s = time.perf_counter() - started_perf
                _record_provider_timing_sample(
                    name,
                    duration_s=duration_s,
                    started_at=started_at,
                    ended_at=ended_at,
                )

        def _publish_render_bundle(
            stage_name: str,
            *,
            bundle_id: str,
            signature: str,
            payload: Mapping[str, Any],
            artifacts: Mapping[str, Path],
            phase: str,
            source_summary: Mapping[str, Any],
            provenance: Mapping[str, Any],
            ready: bool = True,
            not_ready_reason: str | None = None,
        ) -> None:
            if (
                resource_store is None
                or not isinstance(render_artifact_cache, RenderArtifactCacheManager)
                or not render_artifact_cache.enabled
            ):
                return
            stage_meta = runtime_stage_metadata.setdefault(stage_name, {})
            if not ready:
                stage_meta.setdefault("cross_run_cache_publish", "not-ready")
                stage_meta.setdefault("cross_run_cache_publish_phase", phase)
                if not_ready_reason is not None:
                    stage_meta.setdefault("cross_run_cache_publish_reason", not_ready_reason)
                return
            publish_result = render_artifact_cache.publish(
                bundle_id,
                signature,
                payload=payload,
                artifacts=artifacts,
                source_summary=source_summary,
                provenance=provenance,
            )
            stage_meta["cross_run_cache_publish"] = str(publish_result.get("reason", "unknown"))
            stage_meta["cross_run_cache_publish_phase"] = phase
            stage_meta["cross_run_cache_publish_entry"] = str(
                publish_result.get("entry_dir", "")
            )
            if (
                str(publish_result.get("reason", "")) == "published"
                and "cross_run_cache_first_publish_phase" not in stage_meta
            ):
                stage_meta["cross_run_cache_first_publish_phase"] = phase

        final_status = "completed"
        final_error: str | None = None

        try:
            with timeline.stage(
                "runtime.setup",
                display_name="Runtime setup",
                metadata={"provider_count": len(providers)},
            ):
                for name, provider in providers.items():
                    provider.setup(provider_context)

                if road_plane_provider is not None and resource_store is None:
                    raise RuntimeError(
                        "Road plane provider requires a run directory to persist outputs. "
                        "Please supply --output-root/--run-dir so resources can be stored."
                    )
                if point_cloud_provider is not None and resource_store is None:
                    raise RuntimeError(
                        "Point cloud provider requires a run directory to persist outputs. "
                        "Please supply --output-root/--run-dir so resources can be stored."
                    )
                if lighting_provider is not None and resource_store is None:
                    raise RuntimeError(
                        "Lighting provider requires a run directory to persist outputs. "
                        "Please supply --output-root/--run-dir so resources can be stored."
                    )

            pose_samples: dict[int, PoseSample] = {}
            trajectory_metadata: dict[str, Any] = {}
            processed = 0
            working_shape: Optional[tuple[int, int]] = None
            show_progress_bars = bool(
                (provider_context.get("logging") or {}).get("show_progress_bars", True)
            )
            with timeline.stage(
                "runtime.frame_loop",
                display_name="Frame loop",
                metadata={"expected_frames": expected_frames},
            ) as frame_loop_scope:
                frame_iterable = iter_with_progress(
                    frame_provider,
                    enabled=show_progress_bars,
                    total=expected_frames,
                    desc="Process frames",
                    unit="frame",
                )
                for frame in frame_iterable:
                    if max_frames is not None and processed >= max_frames:
                        break
                    if self._working_resolution:
                        frame = normalize_frame_resolution(frame, self._working_resolution)
                    working_shape = (
                        frame.image.shape[:2] if frame.image is not None else None
                    )
                    if (
                        resource_store is not None
                        and frame.image is not None
                    ):
                        # Persist the frame so downstream batch consumers can rely on outputs/<run>/standard/frames.
                        resource_store.save_frame(frame)

                    intrinsics = None
                    if intr_provider is not None:
                        with _time_provider_call("intrinsics"):
                            if intr_cache is not None and intr_dynamic is False:
                                intrinsics = intr_cache
                            else:
                                intrinsics = self._expect_intrinsics(
                                    intr_provider.process(frame)
                                )
                                intr_dynamic = bool(intrinsics.metadata.get("dynamic", False))
                                if not intr_dynamic:
                                    intr_cache = intrinsics
                        frame.metadata["intrinsics"] = intrinsics
                    else:
                        intrinsics = frame.metadata.get("intrinsics")
                    if intrinsics is not None:
                        cache_hit = intr_cache is intrinsics
                        frame_shape = (
                            frame.image.shape[:2]
                            if getattr(frame, "image", None) is not None
                            else working_shape
                        )
                        normalized_matrix, normalized_metadata, _ = (
                            validate_and_normalize_intrinsics(
                                intrinsics.matrix,
                                intrinsics.metadata,
                                frame_shape=frame_shape,
                                allow_principal_point_fallback=False,
                                fail_on_heuristic=True,
                            )
                        )
                        intrinsics = IntrinsicsData(
                            matrix=normalized_matrix,
                            distortion=intrinsics.distortion,
                            metadata=normalized_metadata,
                        )
                        frame.metadata["intrinsics"] = intrinsics
                        if cache_hit:
                            intr_cache = intrinsics
                        if resource_store is not None:
                            resource_store.save_intrinsics(intrinsics)

                    depth = None
                    if depth_provider is not None and not defer_depth:
                        with _time_provider_call("depth"):
                            depth = self._expect_depth(
                                depth_provider.process(frame), frame.index
                            )
                        if working_shape is not None:
                            depth = resize_depth(depth, working_shape)
                        if resource_store is not None:
                            resource_store.save_depth(depth)
                        frame.metadata["depth"] = depth

                    # Process camera_height BEFORE trajectory (trajectory may depend on it)
                    camera_height = None
                    if camera_height_provider is not None:
                        with _time_provider_call("camera_height"):
                            camera_height = self._expect_camera_height(
                                camera_height_provider.process(frame),
                                frame.index,
                            )
                        if resource_store is not None:
                            resource_store.save_camera_height(camera_height)
                        frame.metadata["camera_height"] = camera_height

                    # Process trajectory AFTER camera_height (may require camera_height data)
                    pose = None
                    if trajectory_provider is not None and not defer_trajectory:
                        # On the final frame, DPVO may launch as a subprocess; ensure
                        # prior providers have released any stale CUDA allocations.
                        if expected_frames is not None and (processed + 1) >= expected_frames:
                            _cleanup_cuda_memory()
                        with _time_provider_call("trajectory"):
                            pose_result = trajectory_provider.process(frame)
                        if isinstance(pose_result, PoseData):
                            for sample in pose_result.samples:
                                pose_samples[sample.frame_index] = sample
                            if (
                                pose_result.metadata
                                and not trajectory_metadata
                            ):
                                for key, value in pose_result.metadata.items():
                                    if key in {
                                        "frame_index",
                                        "index_clamped_from",
                                        "source_frame_index",
                                    }:
                                        continue
                                    trajectory_metadata[key] = value
                        pose = self._expect_pose(pose_result, frame.index)
                        if (
                            pose is not None
                            and pose.metadata
                            and not trajectory_metadata
                        ):
                            for key, value in pose.metadata.items():
                                if key in {
                                    "frame_index",
                                    "index_clamped_from",
                                    "source_frame_index",
                                }:
                                    continue
                                trajectory_metadata[key] = value
                        if pose is not None:
                            frame.metadata["pose"] = pose
                            pose_samples[pose.frame_index] = pose
                    if pose is not None and camera_height is not None:
                        frame.metadata["pose"] = pose

                    semantics = None
                    if semantics_provider is not None:
                        try:
                            with _time_provider_call("semantics"):
                                semantics = self._expect_semantics(
                                    semantics_provider.process(frame), frame.index
                                )
                            if semantics_tracker is not None:
                                semantics = semantics_tracker.assign_tracks(
                                    semantics, frame
                                )
                        except Exception:
                            _run_semantics_outputs([frame.index])
                            raise
                        if semantics is not None:
                            if working_shape is not None:
                                semantics = resize_semantics(semantics, working_shape)
                                if resource_store is not None:
                                    aux_path = resource_store.path_for(
                                        ResourceKind.SEMANTICS_AUX, frame.index
                                    )
                                    if aux_path.exists():
                                        try:
                                            aux = resource_store.load_semantics_aux(frame.index)

                                            def _resize_sem_aux_array(
                                                arr: np.ndarray,
                                            ) -> np.ndarray:
                                                if arr.ndim == 2:
                                                    return _resize_array(
                                                        arr, tuple(working_shape)
                                                    ).astype(arr.dtype, copy=False)
                                                if arr.ndim == 3:
                                                    return np.stack(
                                                        [
                                                            _resize_array(
                                                                layer,
                                                                tuple(working_shape),
                                                            )
                                                            for layer in arr
                                                        ],
                                                        axis=0,
                                                    ).astype(arr.dtype, copy=False)
                                                return arr

                                            resource_store.save_semantics_aux(
                                                SemanticsAuxData(
                                                    frame_index=frame.index,
                                                    class_probabilities=(
                                                        _resize_sem_aux_array(
                                                            aux.class_probabilities
                                                        )
                                                        if aux.class_probabilities is not None
                                                        else None
                                                    ),
                                                    class_ids=aux.class_ids,
                                                    confidence=(
                                                        _resize_sem_aux_array(aux.confidence)
                                                        if aux.confidence is not None
                                                        else None
                                                    ),
                                                    road_confidence=(
                                                        _resize_sem_aux_array(
                                                            aux.road_confidence
                                                        )
                                                        if aux.road_confidence is not None
                                                        else None
                                                    ),
                                                    validity_mask=(
                                                        _resize_sem_aux_array(
                                                            aux.validity_mask.astype(
                                                                np.float32
                                                            )
                                                        )
                                                        > 0.5
                                                        if aux.validity_mask is not None
                                                        else None
                                                    ),
                                                    debug_maps={
                                                        key: _resize_sem_aux_array(value)
                                                        for key, value in aux.debug_maps.items()
                                                    },
                                                    model_outputs={
                                                        name: {
                                                            key: _resize_sem_aux_array(value)
                                                            if isinstance(value, np.ndarray)
                                                            and value.ndim in {2, 3}
                                                            else value
                                                            for key, value in output.items()
                                                        }
                                                        for name, output in aux.model_outputs.items()
                                                    },
                                                    road_prior_outputs={
                                                        name: _resize_sem_aux_array(value)
                                                        for name, value in aux.road_prior_outputs.items()
                                                    },
                                                    metadata=dict(aux.metadata or {}),
                                                )
                                            )
                                        except Exception as exc:
                                            LOG.warning(
                                                "Failed to resize standardized semantics aux (%s); continuing.",
                                                exc,
                                            )
                            frame.metadata["semantics"] = semantics
                            if resource_store is not None:
                                resource_store.save_semantics2d(semantics)
                            _run_semantics_outputs([frame.index])

                    state = SceneFrameState(
                        frame=frame,
                        depth=depth,
                        pose=pose,
                        intrinsics=intrinsics,
                        camera_height=camera_height,
                        semantics=semantics,
                    )
                    cache.update(state)
                    processed += 1

                    if on_frame:
                        on_frame(state)

                    # Periodic allocator cleanup during long runs.
                    if processed > 0 and processed % 5 == 0:
                        _cleanup_cuda_memory()

                frame_loop_scope.record.metadata["processed_frames"] = processed
                provider_parent = timeline.add_completed_stage(
                    "runtime.frame_loop.providers",
                    display_name="Per-frame provider totals",
                    status="completed",
                    duration_s=0.0,
                    metadata={"provider_count": len(provider_timing)},
                    parent=frame_loop_scope.record,
                )
                for provider_name in ("intrinsics", "depth", "camera_height", "trajectory", "semantics"):
                    aggregate = provider_timing.get(provider_name)
                    if aggregate is None:
                        continue
                    timeline.add_completed_stage(
                        aggregate.name,
                        display_name=aggregate.display_name,
                        duration_s=aggregate.duration_s,
                        started_at=aggregate.first_started_at,
                        ended_at=aggregate.last_ended_at,
                        metadata={"calls": aggregate.calls},
                        parent=provider_parent,
                    )

            _cleanup_cuda_memory()

            # Run batch semantics immediately after frames are saved,
            # before trajectory assembly and other batch providers.
            if batch_semantics_provider is not None:
                if resource_store is None:
                    raise RuntimeError(
                        "Batch-oriented semantics providers require a ResourceStore; "
                        "ensure a run directory is configured."
                    )
                with timeline.stage(
                    "runtime.post.batch_semantics",
                    display_name="Batch semantics",
                ) as batch_semantics_scope:
                    try:
                        if _try_materialize_standardized_outputs(batch_semantics_provider):
                            batch_semantics_scope.finish(
                                status="cache_materialized",
                                metadata={"reason": "standardized-semantics-materialized"},
                            )
                        else:
                            batch_semantics_provider.run(resource_store, provider_context)
                    finally:
                        _run_semantics_outputs(None)
                _publish_provider_cache(
                    "semantics",
                    batch_semantics_provider,
                    phase="after_batch_semantics",
                )

            # Run deferred trajectory (e.g. DPVO) after all frames are persisted
            # and after any batch semantics so it can use masks produced from the
            # semantics output.
            if defer_trajectory and trajectory_provider is not None:
                flush_fn = getattr(trajectory_provider, "flush", None)
                if flush_fn is not None:
                    _cleanup_cuda_memory()
                    LOG.info(
                        "Running deferred trajectory provider after batch semantics.",
                        extra={"summary": True},
                    )
                    with timeline.stage(
                        "runtime.post.deferred_trajectory",
                        display_name="Deferred trajectory",
                    ):
                        pose_result = flush_fn()
                    if isinstance(pose_result, PoseData):
                        for sample in pose_result.samples:
                            pose_samples[sample.frame_index] = sample
                        if pose_result.metadata and not trajectory_metadata:
                            for k, v in pose_result.metadata.items():
                                if k not in {
                                    "frame_index",
                                    "index_clamped_from",
                                    "source_frame_index",
                                }:
                                    trajectory_metadata[k] = v

            # Run deferred depth (e.g. UniDepth) after all frames are persisted
            # and after trajectory so GPU memory from the trajectory subprocess
            # is released first.
            if defer_depth and depth_provider is not None and resource_store is not None:
                _cleanup_cuda_memory()
                LOG.info(
                    "Running deferred depth provider after trajectory.",
                    extra={"summary": True},
                )
                from types import SimpleNamespace
                frame_indices_iterable = list(resource_store.frame_indices(ResourceKind.FRAMES))
                with timeline.stage(
                    "runtime.post.deferred_depth",
                    display_name="Deferred depth",
                    metadata={"frames": len(frame_indices_iterable)},
                ) as deferred_depth_scope:
                    if _try_materialize_standardized_outputs(depth_provider):
                        deferred_depth_scope.finish(
                            status="cache_materialized",
                            metadata={"reason": "standardized-depth-materialized"},
                        )
                    else:
                        for frame_idx in iter_with_progress(
                            frame_indices_iterable,
                            enabled=show_progress_bars,
                            total=len(frame_indices_iterable),
                            desc="Deferred depth",
                            unit="frame",
                        ):
                            dummy = SimpleNamespace(index=frame_idx, metadata={})
                            depth = self._expect_depth(
                                depth_provider.process(dummy), frame_idx
                            )
                            if working_shape is not None:
                                depth = resize_depth(depth, working_shape)
                            resource_store.save_depth(depth)
                _publish_provider_cache(
                    "depth",
                    depth_provider,
                    phase="after_deferred_depth",
                )

            if (
                resource_store is not None
                and pose_samples
            ):
                trajectory = PoseData(
                    samples=[pose_samples[idx] for idx in sorted(pose_samples)],
                    metadata=dict(trajectory_metadata),
                )

                if self._pose_conditioning_settings.enabled:
                    from pemoin.utils.trajectory_cleanup import condition_poses

                    LOG.info(
                        "Running pose conditioning on raw trajectory.",
                        extra={"summary": True},
                    )
                    with timeline.stage(
                        "runtime.post.pose_conditioning",
                        display_name="Pose conditioning",
                    ):
                        c2w_stack = np.stack(
                            [s.camera_to_world for s in trajectory.samples]
                        )
                        c2w_conditioned, cond_meta = condition_poses(
                            c2w_stack, self._pose_conditioning_settings
                        )
                        for i, sample in enumerate(trajectory.samples):
                            sample.camera_to_world = c2w_conditioned[i]
                            sample.world_to_camera = np.linalg.inv(
                                c2w_conditioned[i].astype(np.float64)
                            ).astype(np.float32)
                        trajectory.metadata.update(cond_meta)

                save_origin_anchored_trajectory(
                    resource_store,
                    trajectory,
                    metadata_label="runtime_initial_save",
                )
                if trajectory_provider is not None:
                    _publish_provider_cache(
                        "trajectory",
                        trajectory_provider,
                        phase="after_trajectory_save",
                    )

            if (
                self._depth_stabilization_settings.enabled
                and resource_store is not None
                and resource_store.has(ResourceKind.DEPTH)
                and resource_store.has(ResourceKind.TRAJECTORY)
            ):
                from pemoin.providers.depth_stabilization import stabilize_depth_sequence

                LOG.info("Running temporal depth stabilization.", extra={"summary": True})
                _cleanup_cuda_memory()
                with timeline.stage(
                    "runtime.post.depth_stabilization",
                    display_name="Depth stabilization",
                ):
                    stabilize_depth_sequence(resource_store, self._depth_stabilization_settings)
                _cleanup_cuda_memory()

            geometry_fusion_provider = providers.get("geometry_fusion")

            if geometry_fusion_provider is not None:
                # --- SOTA path: geometry fusion handles depth rectification + scale + road planes ---
                if resource_store is None:
                    raise RuntimeError(
                        "GeometryFusionProvider requires a ResourceStore; "
                        "ensure a run directory is configured."
                    )
                LOG.info(
                    "Running SOTA geometry fusion provider.",
                    extra={"summary": True},
                )
                with timeline.stage(
                    "runtime.post.geometry_fusion",
                    display_name="Geometry fusion",
                ):
                    geometry_fusion_provider.run(resource_store, provider_context)
                _publish_provider_cache(
                    "geometry_fusion",
                    geometry_fusion_provider,
                    phase="after_geometry_fusion",
                )

            # Point cloud runs after geometry fusion (uses corrected depth+poses)
            if point_cloud_provider is not None:
                LOG.info("Running geometry consistency validation before point_cloud_3d.")
                with timeline.stage(
                    "runtime.post.geometry_consistency_validation",
                    display_name="Geometry consistency validation",
                ) as consistency_scope:
                    consistency_result = validate_depth_pose_intrinsics_consistency(
                        resource_store,
                        settings=self._consistency_settings,
                        context=provider_context,
                    )
                    consistency_scope.record.metadata["status"] = consistency_result.status
                if consistency_result.status == "degraded":
                    summary_path = (
                        resource_store.visualizations_dir("geometry_consistency") / "summary.json"
                    )
                    LOG.warning(
                        "Geometry consistency validation degraded: catastrophic_pairs=%d "
                        "replaced_frames=%d warning_budget_exceeded=%s summary=%s",
                        int(consistency_result.summary.get("num_catastrophic_pairs", 0)),
                        int(consistency_result.summary.get("num_replaced_frames", 0)),
                        bool(consistency_result.summary.get("replacement_budget_exceeded", False)),
                        summary_path,
                    )
                LOG.info("Running point_cloud_3d provider.")
                with timeline.stage(
                    "runtime.post.point_cloud_3d",
                    display_name="Point cloud 3D",
                ):
                    point_cloud_provider.run(resource_store, provider_context)
                _publish_provider_cache(
                    "point_cloud_3d",
                    point_cloud_provider,
                    phase="after_point_cloud_3d",
                )

            # Road plane only if geometry_fusion didn't already produce planes
            if road_plane_provider is not None and geometry_fusion_provider is None:
                LOG.info("Running road_plane provider before post-frame canonicalization.")
                with timeline.stage(
                    "runtime.post.road_plane",
                    display_name="Road plane",
                ):
                    road_plane_provider.run(resource_store, provider_context)

            if (
                geometry_fusion_provider is not None
                and resource_store is not None
                and resource_store.has(ResourceKind.TRAJECTORY)
                and resource_store.has(ResourceKind.CAMERA_HEIGHT)
            ):
                LOG.info(
                    "Canonicalizing geometry-fusion outputs to the shared comparison frame (mode=%s).",
                    self._comparison_frame_settings.mode,
                )
                with timeline.stage(
                    "runtime.post.comparison_frame",
                    display_name="Comparison-frame canonicalization",
                ):
                    canonicalize_geometry_to_comparison_frame(
                        resource_store,
                        settings=self._comparison_frame_settings,
                        road_labels=self._road_labels,
                        sidewalk_labels=self._sidewalk_labels,
                        context=provider_context,
                    )

            if resource_store is not None:
                validation_settings = {}
                if isinstance(self.profile.runtime.settings, Mapping):
                    validation_settings = (
                        self.profile.runtime.settings.get("geometry_validation", {})
                        or {}
                    )
                validation_config = GeometryValidationConfig.from_settings(
                    validation_settings
                )
                with timeline.stage(
                    "runtime.post.geometry_validation",
                    display_name="Geometry validation",
                ):
                    validate_geometry_store(
                        resource_store,
                        config=validation_config,
                        expected_frames=expected_frames,
                    )
                if lighting_provider is not None:
                    LOG.info("Running clip-level lighting provider.")
                    with timeline.stage(
                        "runtime.post.lighting",
                        display_name="Lighting",
                    ):
                        lighting_provider.run(resource_store, provider_context)
                    _publish_provider_cache(
                        "lighting",
                        lighting_provider,
                        phase="after_lighting",
                    )
                run_dir = provider_context.get("run_dir")
                if run_dir is not None:
                    blender_settings = {}
                    if isinstance(self.profile.runtime.settings, Mapping):
                        blender_settings = dict(
                            self.profile.runtime.settings.get("blender_scene", {}) or {}
                        )
                    if blender_settings.get("enabled", False):
                        config_path = None
                        profile_name = self.profile.name
                        if isinstance(provider_context, Mapping):
                            config_path = provider_context.get("profiles_config_path")
                            context_profile = provider_context.get("profile_name")
                            if isinstance(context_profile, str) and context_profile:
                                profile_name = context_profile
                        if config_path is None:
                            raise ValueError(
                                "profiles_config_path must be provided in the runtime context "
                                "when blender_scene is enabled."
                            )
                        config_path = Path(config_path).expanduser().resolve()
                        run_root = Path(run_dir)
                        if point_cloud_provider is None:
                            raise RuntimeError(
                                "Blender scene export now requires a configured point_cloud_3d "
                                "provider so aligned geometry can publish rgb_pointcloud.glb "
                                "and semantic_pointcloud.glb before Blender starts."
                            )
                        self._require_point_cloud_debug_outputs(resource_store)
                        with timeline.stage(
                            "runtime.post.blender_scene",
                            display_name="Blender scene",
                        ):
                            blender_cache_enabled = (
                                isinstance(render_artifact_cache, RenderArtifactCacheManager)
                                and render_artifact_cache.enabled_for("blender_scene")
                            )
                            blender_bundle_payloads: dict[str, dict[str, Any]] = {}
                            blender_bundle_signatures: dict[str, str] = {}
                            blender_cache_hit = False
                            if blender_cache_enabled:
                                for bundle_id in (
                                    "blender_scene_export",
                                    "blender_fbx_export",
                                    "blender_pedestrian_render_outputs",
                                    "blender_shadow_outputs",
                                    "blender_composition_outputs",
                                ):
                                    bundle_payload = self._blender_bundle_payload(
                                        run_dir=run_root,
                                        profile_name=profile_name,
                                        blender_settings=blender_settings,
                                        render_cache=render_artifact_cache,
                                        bundle_id=bundle_id,
                                    )
                                    blender_bundle_payloads[bundle_id] = bundle_payload
                                    blender_bundle_signatures[bundle_id] = render_artifact_cache.signature(
                                        bundle_id,
                                        bundle_payload,
                                    )
                                with timeline.stage(
                                    "blender_scene.cache_lookup",
                                    display_name="Blender cache lookup",
                                ) as cache_lookup_scope:
                                    lookup = render_artifact_cache.lookup(
                                        "blender_composition_outputs",
                                        blender_bundle_signatures["blender_composition_outputs"],
                                    )
                                    blender_cache_hit = bool(lookup.hit)
                                    cache_lookup_scope.record.metadata.update(
                                        {"hit": lookup.hit, "reason": lookup.reason}
                                    )
                                _record_runtime_stage_metadata(
                                    "blender_scene",
                                    {
                                        "cross_run_cache_enabled": True,
                                        "cross_run_cache_signature": blender_bundle_signatures["blender_composition_outputs"],
                                        "cross_run_cache_hit": lookup.hit,
                                        "cross_run_cache_validation": lookup.reason,
                                        "cross_run_cache_entry": str(lookup.entry_dir),
                                        "cross_run_cache_bundle_signatures": {
                                            key: str(value)
                                            for key, value in blender_bundle_signatures.items()
                                        },
                                    },
                                )
                                if lookup.hit:
                                    with timeline.stage(
                                        "blender_scene.cache_materialize",
                                        display_name="Blender cache materialize",
                                    ) as cache_materialize_scope:
                                        materialized = 0
                                        materialized += render_artifact_cache.materialize(
                                            "blender_composition_outputs",
                                            blender_bundle_signatures["blender_composition_outputs"],
                                            run_root=run_root,
                                        )
                                        materialized += render_artifact_cache.materialize(
                                            "blender_shadow_outputs",
                                            blender_bundle_signatures["blender_shadow_outputs"],
                                            run_root=run_root,
                                        )
                                        materialized += render_artifact_cache.materialize(
                                            "blender_pedestrian_render_outputs",
                                            blender_bundle_signatures["blender_pedestrian_render_outputs"],
                                            run_root=run_root,
                                        )
                                        materialized += render_artifact_cache.materialize(
                                            "blender_scene_export",
                                            blender_bundle_signatures["blender_scene_export"],
                                            run_root=run_root,
                                        )
                                        materialized += render_artifact_cache.materialize(
                                            "blender_fbx_export",
                                            blender_bundle_signatures["blender_fbx_export"],
                                            run_root=run_root,
                                        )
                                        cache_materialize_scope.record.metadata["materialized_files"] = materialized
                                    _record_runtime_stage_metadata(
                                        "blender_scene",
                                        {"cross_run_cache_materialized": materialized},
                                    )
                                else:
                                    _record_runtime_stage_metadata(
                                        "blender_scene",
                                        {"cross_run_cache_reason": lookup.reason},
                                    )
                            else:
                                _record_runtime_stage_metadata(
                                    "blender_scene",
                                    {
                                        "cross_run_cache_enabled": False,
                                        "cross_run_cache_hit": False,
                                        "cross_run_cache_validation": "disabled",
                                    },
                                )
                                timeline.add_completed_stage(
                                    "blender_scene.cache_lookup",
                                    display_name="Blender cache lookup",
                                    status="disabled",
                                    duration_s=0.0,
                                    metadata={"reason": "cache-disabled"},
                                )
                            try:
                                with timeline.stage(
                                    "blender_scene.input_validation",
                                    display_name="Blender input validation",
                                ):
                                    self._validate_blender_scene_inputs(run_root)
                                if not blender_cache_hit:
                                    with timeline.stage(
                                        "blender_scene.subprocess_render",
                                        display_name="Blender subprocess render",
                                    ) as render_scope:
                                        result = self._render_trajectory_scene(
                                            run_root,
                                            config_path=config_path,
                                            profile_name=profile_name,
                                            stream_output=bool(
                                                (provider_context.get("logging") or {}).get(
                                                    "stream_blender_subprocess_output", False
                                                )
                                            ),
                                            show_progress=bool(
                                                (provider_context.get("logging") or {}).get(
                                                    "show_progress_bars", True
                                                )
                                            ),
                                        )
                                        if result is not None:
                                            stdout_log_path = getattr(result, "stdout_log_path", None)
                                            stderr_log_path = getattr(result, "stderr_log_path", None)
                                            if stdout_log_path is not None:
                                                render_scope.record.metadata["stdout_log_path"] = str(stdout_log_path)
                                            if stderr_log_path is not None:
                                                render_scope.record.metadata["stderr_log_path"] = str(stderr_log_path)
                                    LOG.info(
                                        "Blender scene saved to %s", run_root / "scene.blend"
                                    )
                                    if blender_cache_enabled:
                                        with timeline.stage(
                                            "blender_scene.cache_publish",
                                            display_name="Blender cache publish",
                                        ):
                                            for bundle_id in (
                                                "blender_scene_export",
                                                "blender_fbx_export",
                                                "blender_pedestrian_render_outputs",
                                                "blender_shadow_outputs",
                                                "blender_composition_outputs",
                                            ):
                                                _publish_render_bundle(
                                                    "blender_scene",
                                                    bundle_id=bundle_id,
                                                    signature=blender_bundle_signatures[bundle_id],
                                                    payload=blender_bundle_payloads[bundle_id],
                                                    artifacts=self._blender_bundle_artifacts(
                                                        run_root,
                                                        bundle_id,
                                                        render_artifact_cache,
                                                    ),
                                                    phase=f"after_{bundle_id}",
                                                    source_summary={
                                                        "profile": profile_name,
                                                        "run_root": str(run_root),
                                                    },
                                                    provenance={
                                                        "config_path": str(config_path),
                                                        "scene_path": str(run_root / "scene.blend"),
                                                    },
                                                )
                                else:
                                    timeline.add_completed_stage(
                                        "blender_scene.subprocess_render",
                                        display_name="Blender subprocess render",
                                        status="cache_hit",
                                        duration_s=0.0,
                                        metadata={"reason": "composition-cache-hit"},
                                    )
                            except Exception as exc:
                                raise RuntimeError(
                                    "Blender scene generation failed; aborting pipeline."
                                ) from exc

                harmonisation_settings = {}
                if isinstance(self.profile.runtime.settings, Mapping):
                    harmonisation_settings = dict(
                        self.profile.runtime.settings.get("harmonisation", {}) or {}
                    )
                harmonisation_config = HarmonisationSettings.from_mapping(harmonisation_settings)
                if harmonisation_config.enabled:
                    if run_dir is None:
                        raise ValueError(
                            "run_dir must be provided in the runtime context when harmonisation is enabled."
                        )
                    run_root = Path(run_dir)
                    with timeline.stage(
                        "runtime.post.harmonisation",
                        display_name="Harmonisation",
                    ):
                        harmonisation_cache_enabled = (
                            isinstance(render_artifact_cache, RenderArtifactCacheManager)
                            and render_artifact_cache.enabled_for("harmonisation")
                        )
                        harmonisation_payload = None
                        harmonisation_signature = None
                        harmonisation_cache_hit = False
                        if harmonisation_cache_enabled:
                            harmonisation_payload = self._harmonisation_bundle_payload(
                                run_dir=run_root,
                                settings=harmonisation_config,
                                render_cache=render_artifact_cache,
                            )
                            harmonisation_signature = render_artifact_cache.signature(
                                "harmonisation_outputs",
                                harmonisation_payload,
                            )
                            with timeline.stage(
                                "harmonisation.cache_lookup",
                                display_name="Harmonisation cache lookup",
                            ) as cache_lookup_scope:
                                lookup = render_artifact_cache.lookup(
                                    "harmonisation_outputs",
                                    harmonisation_signature,
                                )
                                harmonisation_cache_hit = bool(lookup.hit)
                                cache_lookup_scope.record.metadata.update(
                                    {"hit": lookup.hit, "reason": lookup.reason}
                                )
                            _record_runtime_stage_metadata(
                                "harmonisation",
                                {
                                    "cross_run_cache_enabled": True,
                                    "cross_run_cache_signature": harmonisation_signature,
                                    "cross_run_cache_hit": lookup.hit,
                                    "cross_run_cache_validation": lookup.reason,
                                    "cross_run_cache_entry": str(lookup.entry_dir),
                                },
                            )
                            if lookup.hit:
                                with timeline.stage(
                                    "harmonisation.cache_materialize",
                                    display_name="Harmonisation cache materialize",
                                ) as materialize_scope:
                                    materialized = render_artifact_cache.materialize(
                                        "harmonisation_outputs",
                                        harmonisation_signature,
                                        run_root=run_root,
                                    )
                                    materialize_scope.record.metadata["materialized_files"] = materialized
                                _record_runtime_stage_metadata(
                                    "harmonisation",
                                    {"cross_run_cache_materialized": materialized},
                                )
                            else:
                                _record_runtime_stage_metadata(
                                    "harmonisation",
                                    {"cross_run_cache_reason": lookup.reason},
                                )
                        else:
                            _record_runtime_stage_metadata(
                                "harmonisation",
                                {
                                    "cross_run_cache_enabled": False,
                                    "cross_run_cache_hit": False,
                                    "cross_run_cache_validation": "disabled",
                                },
                            )
                            timeline.add_completed_stage(
                                "harmonisation.cache_lookup",
                                display_name="Harmonisation cache lookup",
                                status="disabled",
                                duration_s=0.0,
                                metadata={"reason": "cache-disabled"},
                            )
                        if not harmonisation_cache_hit:
                            with timeline.stage(
                                "harmonisation.run",
                                display_name="Harmonisation run",
                            ):
                                output_dir = run_harmonisation(run_root, harmonisation_config)
                            LOG.info("Harmonisation outputs saved to %s", output_dir)
                            if harmonisation_cache_enabled and harmonisation_payload is not None and harmonisation_signature is not None:
                                with timeline.stage(
                                    "harmonisation.cache_publish",
                                    display_name="Harmonisation cache publish",
                                ):
                                    _publish_render_bundle(
                                        "harmonisation",
                                        bundle_id="harmonisation_outputs",
                                        signature=harmonisation_signature,
                                        payload=harmonisation_payload,
                                        artifacts=self._harmonisation_bundle_artifacts(
                                            run_root,
                                            harmonisation_config,
                                            render_artifact_cache,
                                        ),
                                        phase="after_harmonisation",
                                        source_summary={
                                            "run_root": str(run_root),
                                            "output_dir": harmonisation_config.output_dir,
                                        },
                                        provenance={
                                            "overlay_dir": harmonisation_config.overlay_dir,
                                            "occlusion_mask_dir": harmonisation_config.occlusion_mask_dir,
                                        },
                                    )
                        else:
                            timeline.add_completed_stage(
                                "harmonisation.run",
                                display_name="Harmonisation run",
                                status="cache_hit",
                                duration_s=0.0,
                                metadata={"reason": "harmonisation-cache-hit"},
                            )

            # Quality metrics
            if resource_store is not None:
                quality_metrics_settings = {}
                if isinstance(self.profile.runtime.settings, Mapping):
                    quality_metrics_settings = (
                        self.profile.runtime.settings.get("quality_metrics", {})
                        or {}
                    )
                from pemoin.metrics.integration import run_quality_metrics
                try:
                    with timeline.stage(
                        "runtime.post.quality_metrics",
                        display_name="Quality metrics",
                    ):
                        run_quality_metrics(
                            resource_store,
                            settings=quality_metrics_settings,
                            logger=LOG,
                        )
                except Exception as exc:
                    LOG.error("Quality metrics failed: %s", exc, exc_info=True)

            # Generate visualization videos
            if resource_store is not None:
                for name, provider in providers.items():
                    _publish_provider_cache(
                        name,
                        provider,
                        phase="final_fallback",
                    )
                self._export_videos(
                    resource_store=resource_store,
                    provider_context=provider_context,
                    timeline=timeline,
                )

            if resource_store is not None and provider_metadata:
                for name, payload_base in provider_metadata.items():
                    payload = dict(payload_base)
                    provider_obj = providers.get(name)
                    if provider_obj is not None:
                        payload["produced_resources"] = sorted(
                            kind.value for kind in provider_obj.produced_resources
                        )
                        cache_status_fn = getattr(
                            provider_obj, "get_cross_run_cache_status", None
                        )
                        if callable(cache_status_fn):
                            try:
                                cache_status = cache_status_fn()
                            except Exception as exc:
                                cache_status = {
                                    "cross_run_cache_status_error": str(exc),
                                }
                            if isinstance(cache_status, Mapping):
                                payload.update(cache_status)
                    if name in published_cache_metadata:
                        payload.update(published_cache_metadata[name])
                    payload["expected_frames"] = expected_frames
                    payload["processed_frames"] = processed
                    resource_store.save_provider_settings(name, payload)
                for stage_name, stage_payload in runtime_stage_metadata.items():
                    payload = dict(stage_payload)
                    payload["expected_frames"] = expected_frames
                    payload["processed_frames"] = processed
                    resource_store.save_provider_settings(stage_name, payload)

        except Exception as exc:
            final_status = "failed"
            final_error = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            with timeline.stage(
                "runtime.teardown",
                display_name="Runtime teardown",
            ):
                for name, provider in providers.items():
                    with contextlib.suppress(Exception):
                        provider.teardown()
                with contextlib.suppress(Exception):
                    frame_provider.close()
                _cleanup_cuda_memory()
            timeline.finalize(
                status=final_status,
                metadata={
                    "processed_frames": locals().get("processed", 0),
                    **({"error": final_error} if final_error else {}),
                },
            )
            if resource_store is not None:
                with contextlib.suppress(Exception):
                    resource_store.save_runtime_timeline(timeline.to_mapping())

        return RuntimeResult(
            state_cache=cache,
            processed_frames=processed,
            expected_frames=expected_frames,
            provider_metadata=provider_metadata,
        )

    def build_providers(
        self, factory: ProviderFactory, context: MutableMapping[str, Any] | None = None
    ) -> Dict[str, Any]:
        provider_context = RuntimeContext.coerce(context)
        provider_context.setdefault(
            "cross_run_cache",
            CrossRunCacheManager.from_runtime_settings(
                self.profile.runtime.settings,
                base_root=Path.cwd(),
            ),
        )
        provider_context.setdefault(
            "cross_run_cache_stage_settings",
            _cross_run_cache_stage_settings(self.profile.runtime.settings),
        )
        provider_context.setdefault(
            "render_artifact_cache",
            RenderArtifactCacheManager.from_runtime_settings(
                self.profile.runtime.settings,
                base_root=Path.cwd(),
            ),
        )
        if self.profile.depthanything3:
            provider_context.setdefault(
                "depthanything3_settings", self.profile.depthanything3
            )
        provider_context.setdefault("semantic_role_defaults", dict(self._semantic_role_defaults))
        providers: Dict[str, Any] = {}
        for name, binding in self.profile.providers.items():
            providers[name] = factory.create(binding, provider_context)
        return providers

    def _ensure_resource_store(
        self, context: MutableMapping[str, Any]
    ) -> ResourceStore | None:
        """
        Create and cache a ResourceStore when a run directory is available so that
        frames are persisted to outputs/<run>/standard/frames for downstream providers.
        """
        runtime_context = RuntimeContext.coerce(context)
        existing = runtime_context.get("resource_store")
        if isinstance(existing, ResourceStore):
            return existing
        run_dir = runtime_context.get("run_dir")
        if run_dir is None:
            return None
        run_path = Path(run_dir)
        store = ResourceStore(run_path.name, root=run_path.parent)
        runtime_context.resource_store = store
        return store

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _blender_bundle_payload(
        self,
        *,
        run_dir: Path,
        profile_name: str,
        blender_settings: Mapping[str, Any],
        render_cache: RenderArtifactCacheManager,
        bundle_id: str,
    ) -> dict[str, Any]:
        repo_root = self._repo_root()
        standard_root = run_dir / "standard"
        mixamo_settings = (
            dict(self.profile.mixamo)
            if isinstance(getattr(self.profile, "mixamo", None), Mapping)
            else {}
        )
        if bundle_id == "blender_fbx_export":
            payload: dict[str, Any] = {
                "bundle_id": bundle_id,
                "profile_name": profile_name,
                "settings": _normalize_mapping(
                    {
                        "mixamo_source_fps": mixamo_settings.get("source_fps", 30.0),
                        "mixamo_debug": mixamo_settings.get("debug", True),
                    }
                ),
                "profile_snapshot": render_cache.file_key_signature(
                    standard_root / "profile.json",
                    logical_name="standard/profile.json",
                    include_sha256=True,
                    include_mtime=False,
                ),
                "frames_dir": render_cache.directory_signature(standard_root / "frames"),
            }
        else:
            payload = {
                "bundle_id": bundle_id,
                "profile_name": profile_name,
                "settings": _normalize_mapping(dict(blender_settings)),
                "profile_snapshot": render_cache.file_key_signature(
                    standard_root / "profile.json",
                    logical_name="standard/profile.json",
                    include_sha256=True,
                    include_mtime=False,
                ),
                "frames_dir": render_cache.directory_signature(standard_root / "frames"),
                "intrinsics": render_cache.resource_file_key_signature(
                    standard_root / "intrinsics" / "intrinsics.npz",
                    logical_name="standard/intrinsics/intrinsics.npz",
                ),
                "trajectory": render_cache.resource_file_key_signature(
                    standard_root / "trajectory" / "poses.npz",
                    logical_name="standard/trajectory/poses.npz",
                ),
                "depth_dir": render_cache.resource_directory_signature(standard_root / "depth"),
                "road_plane_dir": render_cache.resource_directory_signature(standard_root / "road_plane"),
                "semantics_dir": render_cache.resource_directory_signature(standard_root / "semantics_2d"),
                "lighting_dir": render_cache.directory_signature(standard_root / "lighting"),
            }
        animation_path_raw = mixamo_settings.get("animation_fbx_path")
        if isinstance(animation_path_raw, str) and animation_path_raw.strip():
            animation_path = Path(animation_path_raw).expanduser()
            if not animation_path.is_absolute():
                animation_path = (repo_root / animation_path).resolve()
            payload["mixamo_animation_path"] = str(animation_path)
            payload["mixamo_animation_motion_category"] = (
                detect_mixamo_animation_motion_category(animation_path) or "unknown"
            )
            if animation_path.exists():
                payload["mixamo_animation_fbx"] = render_cache.file_key_signature(
                    animation_path,
                    logical_name="mixamo.animation_fbx",
                    include_sha256=True,
                    include_mtime=False,
                )
        character_path_raw = mixamo_settings.get("character_fbx_path")
        if isinstance(character_path_raw, str) and character_path_raw.strip():
            character_path = Path(character_path_raw).expanduser()
            if not character_path.is_absolute():
                character_path = (repo_root / character_path).resolve()
            payload["mixamo_character_path"] = str(character_path)
            if character_path.exists():
                payload["mixamo_character_fbx"] = render_cache.file_key_signature(
                    character_path,
                    logical_name="mixamo.character_fbx",
                    include_sha256=True,
                    include_mtime=False,
                )
        for key, script_path in (
            (
                "blender_runner_script",
                repo_root / "src" / "pemoin" / "visualization" / "blender_runner.py",
            ),
            (
                "blender_entry_script",
                repo_root / "src" / "pemoin" / "scripts" / "blender_trajectory_scene.py",
            ),
            (
                "blender_scene_app",
                repo_root / "src" / "pemoin" / "visualization" / "blender_scene" / "app.py",
            ),
            (
                "blender_scene_pipeline",
                repo_root / "src" / "pemoin" / "visualization" / "blender_scene" / "pipeline.py",
            ),
        ):
            if script_path.exists():
                payload[key] = render_cache.script_key_signature(
                    script_path,
                    repo_root=repo_root,
                )
        return payload

    @staticmethod
    def _blender_bundle_artifacts(
        run_dir: Path,
        bundle_id: str,
        render_cache: RenderArtifactCacheManager,
    ) -> dict[str, Path]:
        if bundle_id == "blender_scene_export":
            return render_cache.collect_file(run_dir / "scene.blend", relpath="scene.blend")
        if bundle_id == "blender_fbx_export":
            artifacts: dict[str, Path] = {}
            artifacts.update(
                render_cache.collect_tree(
                    ResourceStore.blender_artifact_dir_for(run_dir, "fbx_exports"),
                    rel_prefix="artifacts/blender/fbx_exports",
                )
            )
            artifacts.update(
                render_cache.collect_file(
                    run_dir / "character_root_motion.fbx",
                    relpath="character_root_motion.fbx",
                )
            )
            return artifacts
        if bundle_id == "blender_pedestrian_render_outputs":
            artifacts: dict[str, Path] = {}
            for dirname in (
                "pedestrian_frames",
                "pedestrian_depth_frames",
                "_pedestrian_depth_exr",
            ):
                artifacts.update(
                    render_cache.collect_tree(
                        ResourceStore.blender_artifact_dir_for(run_dir, dirname),
                        rel_prefix=f"artifacts/blender/{dirname}",
                    )
                )
            return artifacts
        if bundle_id == "blender_shadow_outputs":
            return render_cache.collect_tree(
                ResourceStore.blender_artifact_dir_for(run_dir, "shadow_frames"),
                rel_prefix="artifacts/blender/shadow_frames",
            )
        if bundle_id == "blender_composition_outputs":
            artifacts: dict[str, Path] = {}
            for dirname in (
                "overlayed_frames",
                "overlayed_frames_support_local_grid",
                "occlusion_masks",
                "occlusion_debug",
            ):
                artifacts.update(
                    render_cache.collect_tree(
                        ResourceStore.blender_artifact_dir_for(run_dir, dirname),
                        rel_prefix=f"artifacts/blender/{dirname}",
                    )
                )
            return artifacts
        raise ValueError(f"Unknown Blender bundle id '{bundle_id}'.")

    def _harmonisation_bundle_payload(
        self,
        *,
        run_dir: Path,
        settings: HarmonisationSettings,
        render_cache: RenderArtifactCacheManager,
    ) -> dict[str, Any]:
        repo_root = self._repo_root()
        payload: dict[str, Any] = {
            "settings": _normalize_mapping(asdict(settings)),
            "overlay_dir": render_cache.directory_signature(run_dir / settings.overlay_dir),
            "occlusion_mask_dir": render_cache.directory_signature(
                run_dir / settings.occlusion_mask_dir
            ),
            "pedestrian_frames_dir": render_cache.directory_signature(
                ResourceStore.blender_artifact_dir_for(run_dir, "pedestrian_frames")
            ),
        }
        checkpoint = Path(settings.pretrained_path).expanduser()
        if checkpoint.exists():
            payload["checkpoint"] = render_cache.file_key_signature(
                checkpoint,
                logical_name=checkpoint.name,
                include_sha256=True,
                include_mtime=False,
            )
        for key, script_path in (
            (
                "harmonisation_helper",
                repo_root / "src" / "pemoin" / "utils" / "harmonisation.py",
            ),
            (
                "harmonisation_runner",
                repo_root / "src" / "pemoin" / "scripts" / "harmonisation_runner.py",
            ),
        ):
            if script_path.exists():
                payload[key] = render_cache.script_key_signature(
                    script_path,
                    repo_root=repo_root,
                )
        return payload

    @staticmethod
    def _harmonisation_bundle_artifacts(
        run_dir: Path,
        settings: HarmonisationSettings,
        render_cache: RenderArtifactCacheManager,
    ) -> dict[str, Path]:
        artifacts = render_cache.collect_tree(
            run_dir / settings.output_dir,
            rel_prefix=settings.output_dir,
        )
        diagnostics_dir = run_dir / f"{settings.output_dir}_diagnostics"
        artifacts.update(
            render_cache.collect_tree(
                diagnostics_dir,
                rel_prefix=f"{settings.output_dir}_diagnostics",
            )
        )
        return artifacts

    def _ground_grid_bundle_payload(
        self,
        *,
        resource_store: ResourceStore,
        source_dir: Path,
        output_path: Path,
        fps: float,
        codec: str,
        grid_spacing_m: float,
        extent_m: float,
        min_frames: int,
        road_labels: Sequence[str],
        num_workers: int,
        render_cache: RenderArtifactCacheManager,
    ) -> dict[str, Any]:
        repo_root = self._repo_root()
        run_dir = resource_store.root
        payload: dict[str, Any] = {
            "source_dir": render_cache.directory_signature(source_dir),
            "occlusion_mask_dir": render_cache.directory_signature(
                resource_store.blender_artifacts_dir("occlusion_masks")
            ),
            "semantics_dir": render_cache.resource_directory_signature(
                resource_store.base_dir(ResourceKind.SEMANTICS_2D)
            ),
            "road_plane_dir": render_cache.resource_directory_signature(
                resource_store.base_dir(ResourceKind.ROAD_PLANE)
            ),
            "trajectory": render_cache.resource_file_key_signature(
                resource_store.path_for(ResourceKind.TRAJECTORY),
                logical_name="standard/trajectory/poses.npz",
            ),
            "intrinsics": render_cache.resource_file_key_signature(
                resource_store.path_for(ResourceKind.INTRINSICS),
                logical_name="standard/intrinsics/camera_intrinsics.npz",
            ),
            "settings": {
                "fps": float(fps),
                "codec": str(codec),
                "grid_spacing_m": float(grid_spacing_m),
                "extent_m": float(extent_m),
                "min_frames": int(min_frames),
                "road_labels": [str(label) for label in road_labels],
                "num_workers": int(num_workers),
                "output_path": str(output_path.relative_to(run_dir)),
            },
        }
        for key, script_path in (
            (
                "ground_grid_video_helper",
                repo_root / "src" / "pemoin" / "visualization" / "harmonized_overlay_grid.py",
            ),
            (
                "ground_grid_renderer",
                repo_root / "src" / "pemoin" / "visualization" / "ground_grid.py",
            ),
        ):
            if script_path.exists():
                payload[key] = render_cache.script_key_signature(
                    script_path,
                    repo_root=repo_root,
                )
        return payload

    @staticmethod
    def _ground_grid_bundle_artifacts(
        run_dir: Path,
        output_path: Path,
        render_cache: RenderArtifactCacheManager,
    ) -> dict[str, Path]:
        return render_cache.collect_file(
            output_path,
            relpath=str(output_path.relative_to(run_dir)),
        )

    @staticmethod
    def _expect_intrinsics(result: Any) -> IntrinsicsData:
        if isinstance(result, IntrinsicsData):
            return result
        raise TypeError(
            "Intrinsics provider returned unexpected result. "
            f"Expected IntrinsicsData, received {type(result)!r}."
        )

    @staticmethod
    def _expect_depth(result: Any, frame_index: int) -> DepthData:
        if isinstance(result, DepthData):
            return result
        if isinstance(result, Iterable):
            for item in result:
                if isinstance(item, DepthData) and item.frame_index == frame_index:
                    return item
        raise TypeError(
            "Depth provider returned unexpected result. "
            f"Expected DepthData, received {type(result)!r}."
        )

    @staticmethod
    def _expect_pose(result: Any, frame_index: int) -> Optional[PoseSample]:
        if isinstance(result, PoseData):
            return Runtime._pose_for_frame(result, frame_index)
        if isinstance(result, Iterable):
            for item in result:
                if isinstance(item, PoseData):
                    sample = Runtime._pose_for_frame(item, frame_index)
                    if sample is not None:
                        return sample
        raise TypeError(
            "Trajectory provider returned unexpected result. "
            f"Expected PoseData, received {type(result)!r}."
        )

    @staticmethod
    def _pose_for_frame(pose_data: PoseData, frame_index: int) -> Optional[PoseSample]:
        for sample in pose_data.samples:
            if sample.frame_index == frame_index:
                return sample
        return None

    @staticmethod
    def _expect_semantics(result: Any, frame_index: int) -> Optional[SemanticsData]:
        if isinstance(result, SemanticsData):
            if result.frame_index != frame_index:
                raise ValueError(
                    f"Semantics result index {result.frame_index} does not match frame index {frame_index}."
                )
            return result
        if isinstance(result, Iterable):
            for item in result:
                if isinstance(item, SemanticsData) and item.frame_index == frame_index:
                    return item
        raise TypeError(
            "Semantics provider returned unexpected result. "
            f"Expected SemanticsData, received {type(result)!r}."
        )

    @staticmethod
    def _expect_camera_height(result: Any, frame_index: int) -> CameraHeightData:
        if isinstance(result, CameraHeightData):
            if result.frame_index != frame_index:
                raise ValueError(
                    f"Camera height result index {result.frame_index} does not match frame index {frame_index}."
                )
            return result
        raise TypeError(
            "Camera height provider returned unexpected result. "
            f"Expected CameraHeightData, received {type(result)!r}."
        )

    @staticmethod
    def _validate_blender_scene_inputs(run_dir: Path) -> None:
        validate_blender_scene_inputs(run_dir)

    @staticmethod
    def _require_point_cloud_debug_outputs(resource_store: ResourceStore) -> None:
        missing: list[str] = []
        if not resource_store.has(ResourceKind.POINT_CLOUD_3D):
            missing.append(str(resource_store.path_for(ResourceKind.POINT_CLOUD_3D)))
        for path in (
            resource_store.rgb_pointcloud_artifact_path(),
            resource_store.semantic_pointcloud_artifact_path(),
            resource_store.rgb_pointcloud_path(),
            resource_store.semantic_pointcloud_path(),
        ):
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise RuntimeError(
                "Point-cloud debug outputs are required before Blender scene export, "
                f"but the following paths were missing: {', '.join(missing)}"
            )

    @staticmethod
    def _render_trajectory_scene(
        run_dir: Path,
        config_path: Path,
        profile_name: str,
        stream_output: bool = False,
        show_progress: bool = True,
    ):
        _ = config_path
        Runtime._validate_blender_scene_inputs(run_dir)
        cmd = build_blender_trajectory_command(run_dir, profile_name=profile_name)
        LOG.info("Running Blender trajectory visualization: %s", " ".join(cmd))
        return render_trajectory_scene(
            run_dir,
            profile_name=profile_name,
            stream_output=stream_output,
            show_progress=show_progress,
        )
