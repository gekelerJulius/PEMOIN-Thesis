# Data Contract

## Purpose

This is the canonical contract for persisted standardized data under `outputs/<run>/standard/`.

Code authority:

- `src/pemoin/data/contracts.py`

## Governing Rules

1. Downstream stages must consume persisted data only from `standard/`.
2. If a later stage needs persisted intermediate data, that data must be standardized through `ResourceKind`, `_STANDARD_LAYOUTS`, and `ResourceStore`.
3. Downstream stages must not consume `raw/`; it is for provider-native outputs, caches, and diagnostics only.
4. Metadata may reference `raw/` paths for provenance, but downstream execution must not depend on them.

## Standard Layout

All standardized resources live under `outputs/<run>/standard/`.

| Resource | Path | Format | Core payload |
| --- | --- | --- | --- |
| Frames | `standard/frames/{frame:06d}.png` | PNG | RGB frame |
| Intrinsics | `standard/intrinsics/intrinsics.npz` | NPZ | `matrix`, optional `distortion`, `metadata` |
| Depth | `standard/depth/{frame:06d}.npz` | NPZ | `depth`, optional `confidence`, `metadata` |
| Trajectory | `standard/trajectory/poses.npz` | NPZ | `frame_indices`, `camera_to_world`, optional `world_to_camera`, optional `confidence`, metadata |
| Trajectory Match Graph | `standard/trajectory_match_graph/dpvo_match_graph.npz` | NPZ | canonical DPVO match-graph payload plus `metadata` |
| Semantics 2D | `standard/semantics_2d/{frame:06d}.npz` | NPZ | `segment_ids`, optional `label_ids`, `segments_info`, `frame_id`, `metadata` |
| Semantics Aux | `standard/semantics_aux/{frame:06d}.npz` | NPZ | optional probabilities/confidence sidecars plus `metadata` |
| Point Cloud 3D | `standard/point_cloud_3d/cloud.npz` | NPZ | `points_world`, `labels`, `label_confidences`, `colors`, `observation_counts`, `label_names`, `metadata` |
| Camera Height | `standard/camera_height/{frame:06d}.npz` | NPZ | `height_m`, `metadata` |
| Road Plane | `standard/road_plane/{frame:06d}.npz` | NPZ | `normal`, `offset`, `metadata` |
| Road Plane Support | `standard/road_plane_support/{frame:06d}.npz` | NPZ | `points_world`, optional `weights`, optional `source_frame_index`, optional `diagnostics`, `metadata` |
| Dynamic Mask | `standard/dynamic_mask/{frame:06d}.png` | PNG | `255=static`, `0=dynamic` |
| Lighting Package | `standard/lighting/lighting.json` | JSON | validated clip-level lighting parameters, analytic `light_rig`, decomposition diagnostics, validation, and recovery metadata |
| Lighting Envmap | `standard/lighting/envmap.exr` | EXR | fused clip-level HDR environment map |

Additional standardized metadata:

- `standard/profile.json`
- `standard/runtime/timeline.json`
- `standard/providers/*.json`
- `standard/visualizations/...`
- `standard/videos/...`

Current standardized videos are encoded from top-origin PNG frame sequences without geometric inversion. When source frames have odd width or height, video export pads to an even codec-safe raster instead of cropping persisted pixels.

There is no canonical `semantics_3d` standardized resource in current code.

Current Blender, harmonisation, and geometry debug artifacts live under `outputs/<run>/artifacts/`, for example `artifacts/geometry/point_cloud/rgb_pointcloud.glb`, `artifacts/geometry/point_cloud/semantic_pointcloud.glb`, `artifacts/blender/fbx_exports/character_root_motion.fbx`, `artifacts/blender/fbx_exports/character_root_motion.export.json`, `artifacts/blender/pedestrian_frames/`, `artifacts/blender/pedestrian_depth_frames/`, `artifacts/blender/shadow_frames/`, `artifacts/blender/overlayed_frames/`, `artifacts/blender/overlayed_frames_support_local_grid/`, `artifacts/blender/occlusion_masks/`, `artifacts/blender/occlusion_debug/`, `artifacts/harmonisation/harmonized_overlays/`, and `artifacts/harmonisation/harmonized_overlays_diagnostics/`. These are run artifacts, not standardized resources. Current diagnostics may include local-crop metadata, eligibility-gating decisions, actor-track harmonisation policy, applied-parameter sources such as direct-reference/backfilled/interpolated/forward-propagated markers, tiny-object conservative-path decisions, local color-matching metadata, raw vs smoothed temporal harmonisation parameters, FBX export metadata such as clip span, exporter settings, and texture-embedding diagnostics, post-check rejection reasons, reset reasons, fallback-transform traces, render-backend fixed-scale diagnostics, render-vs-grounding visibility contract diagnostics, and trajectory-first grounding debug outputs such as grounded root poses, support-plane traversal segments, and trajectory height profiles. Downstream execution must not depend on them as a cross-stage contract.
Geometry-fusion provider-native diagnostics under `raw/geometry_fusion/` may now also include shared-solver artifacts such as `joint_consistency_frame_diagnostics.json`, joint scale candidate summaries, corrected-depth vs trajectory road-consistency metrics, and GT correction warnings/fail reasons. Those remain provider-native diagnostics rather than standardized cross-stage resources.
Runtime may materialize those run artifacts from the shared cross-run cache when `runtime.settings.cross_run_cache.blender_scene.enabled`, `runtime.settings.cross_run_cache.harmonisation.enabled`, or `runtime.settings.cross_run_cache.ground_grid.enabled` is active, but that reuse layer does not promote them into standardized resources.
Current Blender cache publishing is split across scene-export, pedestrian/depth-render, shadow, and composition artifact bundles so runtime diagnostics can distinguish which late-stage slice was reused or invalidated.

The run root may also contain convenience outputs such as `scene.blend`, `character_root_motion.fbx`, a copied final `output.mp4`, `rgb_pointcloud.glb`, and `semantic_pointcloud.glb`, but those are not standardized contracts either.

### Runtime Timeline

Canonical run-level execution report persisted at `standard/runtime/timeline.json`.

Current payload includes:

- run-level `status`
- run-level `started_at`, `ended_at`, `duration_s`
- run-level `metadata` such as profile, frame-source summary, expected/processed frame counts, and final error on failed runs
- hierarchical `stages`
  - each stage records `name`, `display_name`, `status`, `started_at`, `ended_at`, `duration_s`, `metadata`, and nested `children`

Current runtime stage metadata may include aggregate per-provider frame-loop timing, cache lookup/hit/materialization results, generated artifact counts, subprocess log paths, and stage-specific skip reasons.

## Resource Details

### Frames

- Producer: frame-provider persistence in runtime
- Main consumers: image-based providers, visualizations, deferred batch providers
- NuScenes frames may carry source metadata such as `cam_sd_token`, `sample_token`, `source_timestamp`, `source_is_key_frame`, and `sampling_mode`; downstream NuScenes-specific providers may use that metadata to resolve the exact source `sample_data`, but cross-stage contracts still depend only on the standardized persisted frame sequence

### Intrinsics

- `matrix`: `(3, 3)`
- `distortion`: optional
- runtime validates and normalizes intrinsics before saving

### Depth

- `depth`: `(H, W)` float array in meters
- `confidence`: optional `(H, W)` float array

### Trajectory

- `camera_to_world`: `(N, 4, 4)`
- `world_to_camera`: optional `(N, 4, 4)`
- `frame_indices`: `(N,)`
- saved trajectories preserve the standardized origin anchor at frame 0

### Trajectory Match Graph

Canonical persisted match-graph resource used by later metric-scale stages.

Expected schema includes:

- `schema_version`
- `coord_space`
- `res_factor`
- edge/frame/node index arrays
- `src_uv`, `tgt_uv`
- `edge_weight`
- timestamps
- `metadata`

### Semantics 2D

- `segment_ids`: per-pixel segment map
- `label_ids`: optional per-pixel class map
- `segments_info`: per-segment dictionaries
- `frame_id`
- `metadata`
- `metadata.semantic_roles`: canonical role groups such as `road`, `sidewalk`, `mobile`, `sky`, and `large_vehicle`; newly generated metadata stores normalized lowercase labels and includes the built-in alias floor for `road`/`roads`, `sidewalk`/`sidewalks`, and `pedestrian`/`pedestrians`/`human`/`person`; downstream stages must resolve semantic groups from this metadata instead of profile label lists

### Semantics Aux

Optional downstream sidecar for semantics-derived signals such as:

- `class_probabilities`
- `class_ids`
- `confidence`
- `road_confidence`
- `validity_mask`
- `debug_maps`
- `model_outputs`
- `road_prior_outputs`
- `metadata`

### Point Cloud 3D

Global dense world-space cloud used by later validation and visualization.

Main fields:

- `points_world`
- `labels`
- `label_confidences`
- `colors`
- `label_names`
- `observation_counts`
- `metadata`

### Camera Height

Per-frame metric camera-height samples.

Metadata should identify:

- source
- axis
- world coordinate system

NuScenes camera-height exports currently represent a stream-constant value derived from calibrated sensor translation, but they are still published through the per-frame standardized camera-height contract.

### Road Plane

Canonical plane equation is stored as:

- `normal`
- `offset`
- `metadata`

See `geometry-reference.md` for the plane convention.

### Road Plane Support

Persisted per-frame support points and diagnostics used by validation and road-plane debug refresh.

### Dynamic Mask

Binary static-vs-dynamic mask used by some geometry stages.

### Lighting Package

Canonical clip-level lighting payload used by Blender scene export and any later rendering-oriented stages.

The JSON contract is provider-agnostic and currently carries fields such as:

- `provider`
- `schema_version`
- `rig_mode` (`analytic_rig`, `sun_plus_fill`, or `envmap_only`)
- `mode` (legacy summary, currently `full_sun` or `ambient_only`)
- `selected_frame_indices`
- `sun_direction_world`
- `sun_strength`
- `sun_color`
- `envmap_path`
- `envmap_rotation_world`
- `ambient_strength`
- `light_rig`
  - each light carries a concrete `kind` such as `SUN`, `AREA`, or `POINT`
  - each light also carries `placement_mode` and `placement_target`; diffuse fills may now publish `subject_anchor_relative` offsets with `placement_target=subject_root_dynamic` instead of absolute world-space locations
  - current PEMOIN diffuse subject fills use explicit roles such as `wrap_key_fill`, `counter_wrap_fill`, and `sky_fill`
  - for `transport_mode=wrap_subject_fill`, `strength` is abstract transport strength and renderer adapters may convert it into renderer-specific light power
- `decomposition`
  - planner diagnostics such as `mode`, diffuse/direct scores, direct-to-fill ratio, demotion reason, rebalanced energy summary, fill transport mode, view direction, view-facing and visible-dark-side irradiance, dark-to-bright ratio, and world-strength adjustments
- `quality`
- `sun_diagnostics`
- `validation`
- `recovery`
- `per_keyframe_diagnostics`
- `metadata`

This is the only supported lighting contract. The EXR remains the canonical fused envmap. Downstream renderers should consume the versioned `light_rig` decomposition from `lighting.json` rather than reconstructing lights from the legacy `mode` summary alone. `rig_mode=analytic_rig` means use the published compact analytic rig. `rig_mode=sun_plus_fill` means use the validated direct light and any available broad fills. `rig_mode=envmap_only` means the envmap is valid but the analytic rig degraded and renderers should rely on world env lighting only. Diffuse fills are no longer assumed to be literal distant emitters in world space; `placement_mode=subject_anchor_relative` means the stored location is a subject-relative offset, and `placement_target` determines whether that offset is anchored to a static insertion spawn or to the time-varying subject root. Current PEMOIN diffuse transport fills use `placement_target=subject_root_dynamic`, `transport_mode=wrap_subject_fill`, and explicit wrap roles so downstream renderers can preserve dark-side support instead of re-inferring transport from literal area-light geometry. Renderer consumers may adapt wrap-fill `strength` into renderer-specific power units, but must preserve the published placement and role semantics.

The paired EXR at `standard/lighting/envmap.exr` is the standardized fused environment map. The provider must validate the lighting package before publish; malformed or implausible envmaps must still fail fast instead of writing standardized lighting. Analytic-rig decomposition may degrade to a simpler rig without aborting if the canonical envmap remains valid. DiffusionLight-Turbo raw probes, intermediate envmaps, HDR variants, recovery artifacts, and scoring diagnostics remain provider-native under `raw/lighting/`.

For current `carla_gt` runs, the upstream CARLA export may also include additive simulator-ground-truth lighting provenance under `<carla_export>/lighting_gt/`, including `run_lighting.json`, `scene_lights.json`, and `frame_lighting.jsonl`. Those files are dataset-native provenance, not PEMOIN standardized cross-stage resources; downstream PEMOIN stages must still consume only the standardized `standard/lighting/lighting.json` plus `standard/lighting/envmap.exr` contract. Current CARLA GT lighting ignores vehicle lights entirely, derives daylight primarily from simulator weather, emits an analytic `SUN` light for direct sun when the sun is above the horizon, and keeps the EXR as sky/ambient illumination rather than the primary direct-sun carrier.

For current Unity GT-backed runs, the upstream Unity export may also include additive lighting provenance under `<unity_export>/lighting_gt/`, including `run_lighting.json`, `scene_lights.json`, `frame_lighting.jsonl`, and `reflection_probe_faces/*.exr`. Those files are dataset-native provenance, not PEMOIN standardized cross-stage resources; downstream PEMOIN stages must still consume only the standardized `standard/lighting/lighting.json` plus `standard/lighting/envmap.exr` contract. Current Unity GT lighting is daylight-first and reflection-probe-backed: it emits one analytic `SUN` light from the exported directional light, converts the exported reflection probe cubemap faces into the canonical envmap, and derives ambient world strength from exported ambient/probe SH data.

## Producer And Consumer Discipline

- Providers may write provider-native diagnostics under `raw/<provider>/...`.
- If another stage needs persisted access to a resource, it must use a standardized `ResourceKind`.
- Do not invent sidecar files outside `ResourceStore` and then make later stages depend on them.

## Metadata Conventions

Common metadata keys used across contracts include:

- `camera_convention`
- `pose_coordinate_system`
- `world_coordinate_system`
- `axis`

Required height/plane metadata rules are enforced by validation code.

## Related Docs

- `system-overview.md`
- `profile-reference.md`
- `geometry-reference.md`
- `validation.md`
