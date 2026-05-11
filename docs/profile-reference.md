# Profile Reference

## Purpose

Profiles define runtime behavior, provider bindings, and tool-specific settings.

Code authority:

- `config/profiles.json`
- `src/pemoin/runtime/profiles/config.py`

## Top-Level Shape

Each profile is defined under the top-level `profiles` object and supports:

- `working_resolution`
- `runtime`
- `providers`
- optional `frame_provider`
- optional `effects`
- optional top-level blocks such as `megasam`, `depthanything3`, `panst3r`, `unity_import`, and `mixamo`

Minimal shape:

```json
{
  "profiles": {
    "profile_name": {
      "working_resolution": 640,
      "runtime": {
        "state_window": 24,
        "degradation_policy": "OfflineDegradationPolicy",
        "settings": {}
      },
      "providers": {
        "intrinsics": {"tool": "...", "settings": {}},
        "depth": {"tool": "...", "settings": {}}
      }
    }
  }
}
```

## Parser Rules

The parser enforces:

- `working_resolution` must exist and be positive
- `working_resolution` may be a single number or `[height, width]`
- `runtime` must be an object
- every binding must contain a non-empty `tool`
- every `settings` block must be an object
- known path-backed profile inputs fail fast during profile load
  - current loader checks `mixamo.character_fbx_path`, `mixamo.animation_fbx_path`, optional `mixamo.asset_root`, `providers.lighting.settings.repo_root`, `runtime.settings.harmonisation.pretrained_path`, `providers.semantics.settings.label_map_path`, and adapter `checkpoint_path` / `config_path` entries when present
  - relative paths are resolved from the repository root implied by the loaded config file

## Current Profiles

Current entries in `config/profiles.json`:

- `unity_gt`
- `unity_dpvo`
- `carla_dpvo`
- `carla_gt`
- `nuscenes_gt`
- `nuscenes_dpvo`

Current frame-provider usage by profile:

- `unity_gt`: Unity import workflow populates frames before runtime
- `unity_dpvo`: `UnityFrameProvider`
- `carla_dpvo`: `CarlaFrameProvider`
- `carla_gt`: `CarlaFrameProvider`
- `nuscenes_gt`: `NuScenesFrameProvider`
- `nuscenes_dpvo`: `NuScenesFrameProvider`

For video export, runtime uses the active frame provider's resolved emitted FPS from `frame_provider_info.settings`. Unity-import-backed runs therefore propagate the imported cadence into runtime frame-provider metadata instead of depending on `unity_import` as a separate video-export-only config source.

Current NuScenes profile defaults:

- `nuscenes_gt`: `sampling_mode=all_camera_frames`, `sampling_fps=30`
- `nuscenes_dpvo`: `sampling_mode=all_camera_frames`, positive profile-tuned `sampling_fps`

Current `carla_dpvo` alignment:

- `carla_dpvo` mirrors `nuscenes_dpvo` for shared DPVO/runtime/blender/harmonisation/geometry-fusion/lighting tuning
- dataset-specific exceptions stay CARLA-native: `CarlaFrameProvider`, `CarlaIntrinsicsProvider`, `CarlaSemanticsProvider`, and fixed-height `CameraHeightProvider { height: 1.6 }`
- `carla_gt` now mirrors the same CARLA runtime/blender/harmonisation/geometry-fusion tuning as `carla_dpvo`, keeps `CarlaDepthProvider` and `CarlaTrajectoryProvider` as the GT geometry sources, and now binds `providers.lighting` to `CarlaGTLightingProvider` instead of DiffusionLight-Turbo
- `unity_dpvo` now mirrors `carla_dpvo` for the same runtime/blender/harmonisation/geometry-fusion/validation tuning while keeping Unity-native ingestion plus `UnityGTIntrinsicsProvider`, `UnityGTSemanticsProvider`, `UniDepthDepthProvider`, `DPVOTrajectoryProvider`, `UnityGTLightingProvider`, and a Unity GT gravity prior for estimated comparison-frame up alignment
- `unity_gt` now mirrors `carla_gt` for the same runtime/blender/harmonisation/geometry-fusion/validation tuning while keeping Unity import ingestion plus Unity GT intrinsics/depth/trajectory/semantics and `UnityGTLightingProvider`
- current `CarlaGTLightingProvider` is weather-driven and daylight-first; it ignores vehicle lights entirely, emits one analytic `SUN` light for direct sunlight when the sun is above the horizon, and uses scene lights only as conservative non-vehicle fills when present
- current `UnityGTLightingProvider` is daylight-first and reflection-probe-backed; it emits one analytic `SUN` light from the exported Unity directional light, converts exported reflection-probe cubemap faces into the canonical envmap, derives ambient world strength from exported ambient/probe SH data, and currently does not synthesize Unity-specific analytic fill lights beyond that direct sun

## Current Provider Families

Provider tools present in current profiles and/or current factory registration include:

- MegaSAM: `MegaSAMIntrinsicsProvider`, `MegaSAMDepthProvider`, `MegaSAMTrajectoryProvider`
- PanSt3R: `PanSt3RIntrinsicsProvider`, `PanSt3RDepthProvider`, `PanSt3RTrajectoryProvider`, `PanSt3RCompositeProvider`
- Unity GT: `UnityGTIntrinsicsProvider`, `UnityGTDepthProvider`, `UnityGTTrajectoryProvider`, `UnityGTSemanticsProvider`
- CARLA: `CarlaIntrinsicsProvider`, `CarlaDepthProvider`, `CarlaTrajectoryProvider`, `CarlaSemanticsProvider`
- NuScenes: `NuScenesIntrinsicsProvider`, `NuScenesTrajectoryProvider`, `NuScenesCameraHeightProvider`
- Virtual KITTI 2: `VirtualKitty2IntrinsicsProvider`, `VirtualKitty2DepthProvider`, `VirtualKitty2TrajectoryProvider`, `VirtualKitty2SemanticsProvider`
- learned geometry: `UniDepthIntrinsicsProvider`, `UniDepthDepthProvider`, `DPVOTrajectoryProvider`, `DepthAnything3*`
- semantics: `TemporalFusionSemanticsProvider`, `CAVISSemanticsProvider`, `Mask2FormerSemanticsProvider`, `TwinLiteSegFormerSemanticsProvider`, `VideoKMaXSemanticsProvider`
- batch geometry: `DensePointCloud3DProvider`, `RobustRoadPlaneProvider`, `GeometryFusionProvider`
- camera height: `CameraHeightProvider`
- lighting: `DiffusionLightTurboLightingProvider`, `CarlaGTLightingProvider`, `UnityGTLightingProvider`

See `src/pemoin/providers/factory.py` for the exact registered tool strings.

Semantic-role labels still come from provider-owned defaults, but PEMOIN extends those defaults with a shared built-in alias floor so all semantics tools recognize `road`/`roads`, `sidewalk`/`sidewalks`, and `pedestrian`/`pedestrians`/`human`/`person` on newly generated metadata. Current Unity GT semantics defaults also classify `ParkedCar` as a `mobile` label for downstream masking and fusion.

## DPVO Adapter Settings

`providers.trajectory.settings.adapter` for `DPVOTrajectoryProvider` supports:

- `stride`
- `skip`
- `precision_mode`
  - `amp_fp16`
  - `fp32`
- `memory_preset`
  - `balanced`
  - `low_vram`
- `allocator_mode`
  - `native` (current default and current profile setting for `unity_dpvo` / `carla_dpvo` / `nuscenes_dpvo`)
  - `expandable_segments`
  - `cuda_malloc_async`
- `allocator_max_split_size_mb`
  - optional allocator tuning knob; only applied when `allocator_mode` is not `native`
- `cfg_opts`
  - DPVO `KEY VALUE` override tokens passed through to `demo.py`
- `memory_diagnostics.sample_every_n_frames`
- `memory_diagnostics.warn_vram_used_ratio`
- `memory_guard.enabled`
- `memory_guard.warmup_frames`
- `memory_guard.abort_reserved_vram_ratio`
- `memory_guard.abort_reserved_to_allocated_ratio`

DPVO now fails fast on allocator instability. The bridge writes software/build information, effective allocator config, per-frame memory samples, and failure classification to `raw/dpvo/dpvo_memory_diagnostics.json`. There is no automatic retry or transparent fallback to a different allocator or precision mode.

## Common Runtime Settings

`runtime.settings` commonly contains:

- `comparison_frame`
- `validation_policy`
- `geometry_consistency_validation`
- `geometry_validation`
- `quality_metrics`
- `semantics_debug`
- `semantics_visualization`
- `video_export`
- `blender_scene`
- `harmonisation`

`runtime.settings.geometry_consistency_validation` currently supports recoverable-vs-definitive semantics:

- `exclude_dynamic_pixels` enables static-scene validation by masking dynamic/mobile pixels before pairwise checks
- `dynamic_mask_source` selects `dynamic_mask`, `semantics_mobile`, `auto`, or `none`
- robust reprojection thresholds use `max_reprojection_p90_px` / `max_reprojection_p95_px` as the primary catastrophic signal, while `max_reprojection_rmse_px` remains a supporting tail diagnostic
- `max_consecutive_catastrophic` is the contiguous severe-catastrophic hard-fail limit
- `max_skipped_frames` is the replacement warning budget used for degraded diagnostics
- isolated or short recoverable catastrophic spans can continue with minimal frame replacement and a loud warning instead of forcing an automatic abort

`runtime.settings.validation_policy` supports adaptive low-FPS quality gating for selected validators.

Important keys:

- `enabled`
- `reference_sampling_fps`
- `minimum_sampling_fps`
- `threshold_curve`
  - currently `sqrt_inverse_ratio`
- `max_threshold_scale`
- `min_count_scale`
- `hard_fail_margin`
- `continue_on_soft_failure`
- `emit_loud_warnings`

Current semantics:

- `sampling_fps >= 10` keeps current soft thresholds effectively unchanged
- below `10 FPS`, selected geometry-quality thresholds are relaxed gradually and selected frame-count budgets are expanded
- soft-limit overruns continue in degraded mode when `continue_on_soft_failure=true`
- hard-limit overruns and invariant failures still abort

## NuScenes Frame-Provider Settings

`NuScenesFrameProvider` supports:

- `sampling_mode`
  - `keyframes_only`
  - `all_camera_frames`
- `sampling_fps`
  - target/cap FPS for emitted frames; runtime stores the resolved emitted cadence derived from timestamps
- `start_frame`
- `end_frame`
- `frame_stride`

When `sampling_mode=all_camera_frames`, NuScenes GT providers consume exact sweep `sample_data` poses from the dataset, validate that intrinsics stay constant within the selected stream, and derive constant camera height from calibrated sensor translation.

`runtime.settings.blender_scene.occlusion` controls contact-aware overlay compositing for inserted pedestrians.

Imported character assets consumed by `runtime.settings.blender_scene` are treated as metric-authoritative. PEMOIN expects user-supplied character assets to already be correctly metric-sized and does not expose character-rescaling policy knobs in current profiles.

Important keys:

- `depth_source`
- `default_front_margin_m`
- `relative_margin`
- `contact_ground_roles` / `contact_ground_labels`
- `contact_plane_band_m`
- `contact_patch_radius_m`
- `contact_coplanar_tolerance_m`
- `write_debug`
- `edge_treatment.enabled`
- `edge_treatment.boundary_band_px`
- `edge_treatment.feather_radius_px`
- `edge_treatment.feather_strength`
- `edge_treatment.blur_enabled`
- `edge_treatment.blur_radius_px`
- `edge_treatment.blur_strength`
- `edge_treatment.despill_enabled`
- `edge_treatment.despill_strength`
- `edge_treatment.regrain_enabled`
- `edge_treatment.regrain_strength`
- `edge_treatment.tiny_object_disable_feather`
- `edge_treatment.tiny_object_disable_blur`
- `edge_treatment.tiny_object_disable_despill`
- `edge_treatment.tiny_object_disable_regrain`
- `edge_treatment.tiny_object_max_boundary_fraction`
- `edge_treatment.tiny_object_disable_all_below_short_side_px`
- `edge_treatment.tiny_object_disable_all_below_visible_pixels`
- `edge_treatment.disable_when_boundary_fraction_above`
- `temporal_stabilization.enabled`
- `temporal_stabilization.base_hysteresis_margin_m`
- `temporal_stabilization.state_flip_persist_frames`
- `temporal_stabilization.edge_exit_hold_frames`
- `temporal_stabilization.max_single_frame_visible_area_drop_ratio`

`runtime.settings.blender_scene.shadow` controls the shadow-catcher render and Python shadow composite.

Important keys:

- `enabled`
- `receiver_patch_size_m`
- `map_resolution`
- `softness`
- `opacity`
- `tint_rgb`

Shadow mode is no longer configurable. PEMOIN always uses the single-pass receiver-luma shadow path.

`runtime.settings.blender_scene.render` controls the fast raster render budget.

Important keys:

- `engine`
- `resolution_scale`
- `samples`
- `material_policy`
- `dynamic_light_binding`
- `salience_adaptive.enabled`
- `salience_adaptive.low_salience_resolution_scale`
- `salience_adaptive.protect_below_visible_pixels`
- `salience_adaptive.protect_below_bbox_short_side_px`
- `salience_adaptive.protect_when_center_distance_ratio_below`
- `salience_adaptive.reduce_only_when_boundary_fraction_above`
- `salience_adaptive.reduce_only_near_visibility_transition`
- `salience_adaptive.shadow_quality_reduction_enabled`
- `salience_adaptive.fill_light_reduction_enabled`
- `performance.persistent_data`
- `performance.fast_png_compression`
- `performance.disable_raytracing`
- `performance.disable_volumetric_shadows`
- `performance.disable_volumetric_lighting`
- `performance.disable_bloom`
- `performance.disable_screen_space_reflections`
- `performance.disable_gtao`
- `performance.disable_motion_blur`
- `performance.disable_high_quality_normals`
- `raw_subject_exposure.enabled`
- `raw_subject_exposure.target_match_strength`
- `raw_subject_exposure.max_gain`
- `raw_subject_exposure.validation_tolerance`
- `raw_subject_exposure.pedestrian_reference_weight`
- `raw_subject_exposure.min_pedestrian_reference_pixels`

Current render behavior:

- PEMOIN renders the pedestrian exactly once at the configured `resolution_scale`.
- `resolution_scale` is the authoritative fixed internal raster scale and may exceed `1.0` for profiles that need more tiny-actor coverage.
- Adaptive `render.tiny_object.*` rerender settings were removed and now fail fast during profile parsing.
- Current Blender raster runs favor persistent render data and low-overhead intermediate PNG encoding because the rendered pedestrian/shadow sequences are pipeline-internal artifacts that are consumed immediately by later stages.
- Current Blender raster runs now also expose an explicit subject-material simplification policy. The default `preserve_base_alpha_normal` path keeps base-color, transparency, and normal-map response while flattening lower-value roughness/specular/gloss map branches to cheaper constant shading inputs.
- Current Blender raster runs also expose an explicit dynamic subject-light binding policy. The default `copy_location_constraint` path keeps subject-relative fill lights attached to the grounded actor through location-only Blender evaluation rather than inserting per-frame light-location keyframes.
- Current Blender raster runs also disable several costly EEVEE screen-space features by default for these pipeline-internal renders, including bloom, screen-space reflections, GTAO, motion blur, and optional volumetric/high-quality shading toggles when the Blender build exposes them.
- When grounding/visibility diagnostics already prove the actor is fully off-camera for some frames, PEMOIN may skip rasterizing those frames and materialize empty pedestrian/depth/shadow outputs instead.
- `standard/visualizations/blender_scene/render_backend_diagnostics.json` records the effective backend fast-path settings plus the split between rendered frames and visibility-culled frames.
- This path changes render sampling only; it must not move or rescale the pedestrian in world space.
- For `trajectory_path` support mode, a projected support anchor that cannot be trusted in that canonical overlay space is recorded as `unverifiable` unless stronger contradictory evidence exists; verified off-road or contact-mismatch evidence still hard-fails.

## Alignment And World-Frame Settings

`runtime.settings.comparison_frame` controls the maintained post-geometry-fusion canonicalization path shared by all active profiles.

Important keys:

- `enabled`
- `mode`
  - `gt`
  - `estimated`
- `ground_source`
- `fail_if_missing_ground`
- `min_ground_samples`
- `max_abs_ground_shift_m`
- `min_motion_steps`
- `min_total_xy_travel_m`
- `min_direction_concentration`
- `gt_max_height_rmse_m`
- `gt_max_height_abs_err_m`
- `gt_max_ground_drift_range_m`
- `estimated_min_median_camera_height_m`
- `up_direction_source`
- `gravity_prior`
  - `provider`
  - `fail_if_unavailable`
  - `min_valid_frames`
  - `max_outlier_angle_deg`

Current semantics:

- all active profiles use `providers.geometry_fusion` first and then one comparison-frame path
- GT profiles trust the incoming metric trajectory but canonicalize into PEMOIN's comparison frame instead of preserving dataset-native world axes
- DPVO / estimated profiles canonicalize into the same comparison frame through the estimated path
- `unity_dpvo` currently keeps estimated DPVO translation/orientation as the trajectory source but resolves canonical up from a Unity GT gravity prior expressed in the estimated trajectory world; this is a privileged IMU-like prior, not GT trajectory replacement
- both maintained paths fail fast when motion is too weak to derive a reliable canonical yaw
- the final comparison frame always targets a grounded support surface at `z=0`

## Geometry Consistency Validation

`runtime.settings.geometry_consistency_validation` runs before point-cloud fusion and controls:

- pairwise reprojection sampling
- static-vs-dynamic masking policy
- overlap/inlier thresholds
- depth-scale drift limits
- recoverable-vs-severe catastrophic classification
- catastrophic-frame skip/replacement budget

See `validation.md`.

## Geometry Validation

`runtime.settings.geometry_validation` controls post-processing validation of:

- intrinsics
- trajectory matrices
- depth reprojection
- road-plane and point-cloud consistency
- optional canonical camera orientation checks

See `validation.md`.

## Quality Metrics

`runtime.settings.quality_metrics` controls the opt-in quality-metrics module.

Current nested sections:

- `trajectory`
  - `enabled`
  - `rpe_deltas`
  - `scale_drift_window`
  - `scale_drift_stride`
  - `umeyama_align`
  - `umeyama_with_scale`
- `road`
  - `enabled`
  - `residual_percentiles`
  - `normal_stability_window`
  - `smoothness_window`
- `artifacts`
  - `enabled`
  - `reprojection_heatmaps`
  - `temporal_flicker`
  - `point_cloud_slices`
  - `road_model_overlay`
  - `confidence_overlay`
  - `max_frames`
  - `colormap`
  - `slice_thickness_m`
  - `flicker_neighbor_frames`

## Geometry-Fusion Settings

`providers.geometry_fusion.settings` controls the batch geometry-fusion path.

Important keys:

- trajectory handling
  - `preserve_metric_trajectory`
  - `joint_consistency_*`
- road rectification
  - `affine_mode`
  - `lambda_s`
  - `lambda_b`
- factor graph
  - `factor_graph_enabled`
  - `fg_env_name`
  - `fg_env_manager`
  - `fg_window_size`
  - `fg_overlap`
  - `fg_*` noise and continuity settings
- road surface / quality gating
  - `quadratic_enabled`
  - match-graph and fallback settings for DPVO scale handling

Geometry fusion is the current batch path for profiles that bind `geometry_fusion`.

Current geometry-fusion behavior now uses one shared metric-consistency framework across GT and non-GT profiles:

- GT camera height above the persisted road support is the authoritative metric constraint.
- DPVO / estimated trajectories keep DPVO pose shape and orientation, but geometry fusion may change one global translation scale factor.
- Estimated depth remains the least trusted signal and is corrected through constrained per-frame affine rectification so it can agree with the selected global trajectory scale and GT camera height.
- GT depth + GT trajectory inputs run through the same consistency framework, but they are expected to remain near-zero correction; small shared-solver drift is warned loudly and larger GT corrections fail fast.
- `joint_consistency_*` settings control the sampled-frame budget, hard reprojection bounds for candidate global scales, the coupling weight between corrected-depth road consistency and DPVO reprojection, and GT warn/fail thresholds for non-trivial scale corrections.
- `preserve_metric_trajectory` remains available as an explicit escape hatch for profiles that intentionally preserve an existing metric trajectory scale inside geometry fusion without the shared correction path.

## Lighting Provider Settings

Current profiles use multiple lighting providers:

- `unity_gt` / `unity_dpvo`: `UnityGTLightingProvider`
- `carla_gt`: `CarlaGTLightingProvider`
- `carla_dpvo` / `nuscenes_gt` / `nuscenes_dpvo`: `DiffusionLightTurboLightingProvider`

`DiffusionLightTurboLightingProvider` runs DiffusionLight-Turbo from env `diffusionlight-turbo`, estimates one clip-level lighting package, and publishes the standardized result to `standard/lighting/`.

Common `providers.lighting.settings` keys:

- execution
  - `repo_root`
  - `conda_env`
- frame selection and preprocessing
  - `input_size`
  - `num_keyframes` (legacy compatibility input; provider defaults to a larger primary candidate set)
- DiffusionLight-Turbo inference
  - `algorithm`
  - `offload`
  - `no_controlnet`
  - `allow_online_model_fetch` (defaults to `false`; current profiles are offline-first)
  - `hf_home`
  - `sdxl_model`
  - `sdxl_vae_model`
  - `sdxl_controlnet_model`
  - `depth_estimator_model`
- fusion / extraction
  - `sun_sigma_deg`
  - `max_fill_lights`
  - `diffuse_demote_enabled`
  - `diffuse_demote_aggressiveness`
  - `max_direct_to_fill_ratio_for_diffuse`
  - `fill_heavy_min_fill_count`
  - `fill_heavy_direct_scale`
  - `diffuse_softness_bias`
  - `fill_heavy_dark_side_target_ratio`
  - `fill_heavy_transport_gain`
  - `wrap_geometry_min_azimuth_separation_deg`
  - `wrap_geometry_counter_opposition_deg`
  - `wrap_geometry_sky_min_elevation_deg`
  - `wrap_geometry_candidate_count_per_role`

Current provider behavior is HDR-native and fail-fast:

- primary fusion uses the tool-produced EXRs, not a standardized fused LDR round-trip
- preprocessing avoids black letterbox padding
- sun extraction is consensus-based in camera space, then conservatively gated in world space using only the configured trajectory provider rotations
- final rig planning also uses hybrid diffuse-scene cues from the fused envmap plus selected-frame appearance metrics
- one deterministic recovery attempt is allowed inside the provider
- cross-run reuse is split between DLT inference artifacts and the final standardized lighting package
- Hugging Face-backed model loads are offline-first and use a stable cache root via
  `HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE`
- if direct-light consensus is weak, the provider may still publish a validated analytic rig built from broad diffuse fills
- if direct-light consensus is geometrically coherent but the scene still reads diffuse overall, the provider may aggressively demote the direct sun and rebalance the rig toward broad fills
- if the canonical envmap is valid but analytic decomposition is too weak to trust, the provider degrades to `rig_mode=envmap_only`
- if the canonical envmap is valid but analytic fill transport or dynamic-range checks are only weakly degraded, the provider may publish a validated degraded `envmap_only` fallback instead of aborting
- if the final ambient/envmap package fails plausibility validation, standardized lighting is not published

Provider-native staging inputs, per-keyframe outputs, recovery diagnostics, and validation reports stay under `raw/lighting/`. Downstream stages should read only the standardized JSON + EXR package described in `data-contract.md`.

## Road-Plane Provider Settings

`providers.road_plane.settings` controls the robust road-plane estimator.

Main groups:

- support selection and ROI
  - `include_sidewalk_in_support`
  - `support_source`
  - `support_pixel_stride`
  - `support_min_confidence`, `support_min_points`
  - `forward_min_m`, `forward_max_m`, `lateral_max_m`, `vertical_max_m`
- adaptive support fallback
  - `support_adaptive_forward_min_enabled`
  - `support_forward_min_floor_m`
  - `support_forward_min_step_m`
  - `support_target_min_points`
  - `support_adaptive_forward_max_iters`
- temporal estimation
  - `temporal_mode`
  - `adaptive_window_enabled`
  - `window_half_width`, `window_min_half_width`, `window_max_half_width`
  - `window_causal_only`
- state filtering and gating
  - `state_process_noise_*`
  - `state_meas_noise_*`
  - `state_innovation_gate`
  - `gating_*`
  - `saved_point_*`
  - `recovery_fit_*`
- robust fitting
  - `huber_delta`
  - `lambda_height`
  - `lambda_temp`
  - `trim_ratio`
  - `irls_iters`
- optional fallback/aggregation
  - `multi_hypothesis_*`
  - `metric_grid_*`
- visualization
  - `overlay_extent_m`
  - `overlay_max_points`
  - `viz_point_stride_m`
  - `viz_confidence_threshold`
  - `viz_residual_clamp`
  - `viz_video_codec`

## Point-Cloud Settings

`providers.point_cloud_3d.settings` controls standardized dense world-space fusion.

Common keys:

- `pixel_stride`
- `voxel_size_m`
- `min_depth_m`, `max_depth_m`
- `min_observations`
- `min_confidence`
- `max_points`
- `export_glb`, `glb_max_points`
- consistency filters such as `max_position_std_m`, `max_depth_std_m`, and `min_view_diversity`

Current point-cloud export writes the standardized fused cloud to `standard/point_cloud_3d/cloud.npz`, writes debug GLBs to `artifacts/geometry/point_cloud/rgb_pointcloud.glb` and `artifacts/geometry/point_cloud/semantic_pointcloud.glb`, and copies those two files to the run root as `rgb_pointcloud.glb` and `semantic_pointcloud.glb`.

When cross-run cache is enabled, the point-cloud provider may also reuse the standardized cloud and point-cloud GLB artifacts for equivalent depth + trajectory + semantics + frame inputs instead of rebuilding the fused cloud.

## Camera Height Settings

`providers.camera_height.settings` accepts exactly one of:

- `height`
- `heights`

Standardized camera-height outputs should include metadata identifying the world axis and coordinate system.

## Blender Scene And Mixamo

`runtime.settings.blender_scene` controls scene export, inserted pedestrian placement, trajectory-first pedestrian grounding, local support search, optional scene lighting, fast raster shadow/pedestrian rendering through `runtime.settings.blender_scene.render` and `runtime.settings.blender_scene.shadow`, clip-level raw pedestrian exposure calibration through `runtime.settings.blender_scene.render.raw_subject_exposure`, and contact-aware overlay occlusion through `runtime.settings.blender_scene.occlusion`.

Current Blender rendering uses one fast raster backend for the pedestrian pipeline. PEMOIN renders `artifacts/blender/pedestrian_frames/` and `artifacts/blender/pedestrian_depth_frames/` from the grounded scene, renders a single shadow-catcher sequence, materializes `artifacts/blender/shadow_frames/`, applies a bounded clip-level raw-subject exposure correction to the pedestrian renders, and then composites the shadow onto the plate before compositing the visible pedestrian on top. When the planned actor visibility is already off-camera for some frames, runtime may skip the raster backend for those frames and synthesize empty pedestrian/depth/shadow outputs instead.
Current Blender runs also emit a reusable single-clip FBX under `artifacts/blender/fbx_exports/character_root_motion.fbx` plus `character_root_motion.export.json` and a run-root convenience copy `character_root_motion.fbx`. That export uses the chosen Mixamo character + animation for the run, keeps one authored clip span, preserves root motion, normalizes the actor back to origin instead of PEMOIN scene placement, and is intended for later Blender/Unity import rather than downstream PEMOIN stages.

When `runtime.settings.cross_run_cache.blender_scene.enabled=true`, runtime may reuse previously published Blender scene/render/composition bundles for an equivalent run instead of invoking Blender again. Diagnostics for lookup/materialization/publish are written to `standard/providers/blender_scene.json`.

Current occlusion composition also applies default-on boundary-only edge treatment on the final visible pedestrian silhouette. This stays in the compositor stage, not the harmonisation stage. Current occlusion additionally applies temporal hysteresis for borderline small-actor depth ordering and can bypass destructive edge operations entirely when the visible actor is too small or too boundary-dominated.

Local support hold can now be configured either as:

- `local_support_temporal_hold_frames`
- `local_support_temporal_hold_seconds`

When `local_support_temporal_hold_seconds` is set, Blender grounding converts it to an effective frame budget using the authoritative runtime sampling FPS.

`max_plane_center_xy_distance_m` remains the strict base locality threshold for persisted support bootstrap. Current grounding may expand that budget internally, bounded by `global_plane_range_m`, when the inserted pedestrian is intentionally placed farther from the trajectory corridor than the fixed base threshold. Grounding diagnostics record the nearest persisted-plane center distance, the effective locality limit, and whether bootstrap-relaxed locality was used.

Trajectory-first vertical smoothing is controlled by:

- `trajectory_grounding_transition_frames`
- `trajectory_grounding_max_step_m`
- `trajectory_grounding_max_vertical_velocity_mps`
- `trajectory_grounding_max_vertical_accel_mps2`

Current grounding is trajectory-first: PEMOIN keeps the authored root `x,y`, resolves which persisted road planes the visible trajectory traverses, evaluates support height at the current path location, and smooths only the resulting root `z` signal across plane transitions. Contact-segment locks no longer drive actor root motion. Diagnostics now include authored vs grounded root poses, traversed support-plane segments, vertical profile metrics, and trajectory support segment debug outputs.

When Blender scene export is enabled, profiles must also provide the relevant `mixamo.*` paths/settings.

Current `mixamo` settings for Blender scene export include:

- `character_fbx_path`
- `animation_fbx_path`
  - the imported animation still provides pose timing and `mixamo_phase` contact timing
  - current Blender insertion no longer requires that animation clip to carry usable horizontal locomotion; PEMOIN resolves heading from a calibrated default Mixamo FBX forward axis
- both FBX paths are validated during profile load and must already exist
- optional `asset_root`
  - when omitted, PEMOIN uses the character FBX parent directory as the canonical Mixamo asset root
  - when present, it is validated during profile load and must already exist as a directory
  - imported character materials are relinked against this local package root when they are file-backed
  - embedded Mixamo FBX textures are accepted when Blender imports them as packed images, even if the stored source path is dead
  - the run fails fast only when a required texture is neither packed nor resolvable from the asset root

Current `runtime.settings.blender_scene` pedestrian-placement keys also include:

- `pedestrian_motion_policy`
  - supported values: `auto`, `stationary_at_spawn`, `animation_root_motion`, `camera_trajectory_relative`
  - current default: `auto`
  - current `auto` behavior resolves from the Mixamo animation path, not inferred clip motion
  - clips under `assets/mixamo/animations/idle/` resolve to `stationary_at_spawn`
  - clips under `assets/mixamo/animations/moving/` resolve to `animation_root_motion`
  - `animation_root_motion` uses the clip's extracted forward progress for world translation after spawn placement, treats `pedestrian_trajectory_t` only as the initial trajectory anchor/spawn selector, locks both facing and walking to the spawn-resolved `pedestrian_heading_deg`, keeps only non-forward hips pose motion such as lateral sway/vertical body motion, tracks unwrapped authored cycle counts explicitly so loop seams cannot silently reset forward progress, and now solves the hips/pelvis correction from the evaluated pelvis world transform before converting the corrected result back into pose space
  - `camera_trajectory_relative` preserves the old camera-coupled behavior and is legacy/debug only
  - paths outside those directories fail fast instead of guessing

Current `runtime.settings.blender_scene.render` keys include:

- `engine`
  - current supported value: `raster`
- `resolution_scale`
- `samples`
- `material_policy`
  - current supported values: `preserve_most_maps`, `preserve_base_alpha_normal`, `preserve_base_alpha`
  - current default: `preserve_base_alpha_normal`
- `dynamic_light_binding`
  - current supported values: `copy_location_constraint`, `sparse_keyframes`, `spawn_only_static`
  - current default: `copy_location_constraint`
- `salience_adaptive.enabled`
  - current default: `true`
- `salience_adaptive.low_salience_resolution_scale`
  - current default: `0.85`
  - must be `<= render.resolution_scale`
- `salience_adaptive.protect_below_visible_pixels`
  - current default: `10000`
- `salience_adaptive.protect_below_bbox_short_side_px`
  - current default: `56`
- `salience_adaptive.protect_when_center_distance_ratio_below`
  - current default: `0.30`
- `salience_adaptive.reduce_only_when_boundary_fraction_above`
  - current default: `0.24`
- `salience_adaptive.reduce_only_near_visibility_transition`
  - current default: `true`
- `salience_adaptive.shadow_quality_reduction_enabled`
- `salience_adaptive.fill_light_reduction_enabled`
- `performance.persistent_data`
- `performance.fast_png_compression`
- `performance.disable_raytracing`
- `performance.disable_volumetric_shadows`
- `performance.disable_volumetric_lighting`
- `performance.disable_bloom`
- `performance.disable_screen_space_reflections`
- `performance.disable_gtao`
- `performance.disable_motion_blur`
- `performance.disable_high_quality_normals`
- `raw_subject_exposure.enabled`
  - current default: `true`
- `raw_subject_exposure.target_match_strength`
  - current default: `0.75`
- `raw_subject_exposure.max_gain`
  - current default: `2.5`
  - note: runtime now treats this as an upper safety ceiling; the exposure stage itself only applies conservative fine-trim gains and may no-op when predicted residuals stay high
- `raw_subject_exposure.validation_tolerance`
  - current default: `0.18`
- `raw_subject_exposure.pedestrian_reference_weight`
- `raw_subject_exposure.min_pedestrian_reference_pixels`

Current `runtime.settings.blender_scene.lighting.wrap_subject_fill` keys include:

- `global_strength_scale`
  - Blender-side multiplier for all standardized wrap-fill `POINT` lights
- `wrap_key_role_scale`
- `counter_wrap_role_scale`
- `sky_fill_role_scale`
- `counter_side_lift_bias`
- `sky_softness_bias`
- `direct_preservation_bias`
- `raw_exposure_trim`
  - bounded multiplier applied after automatic raw-subject exposure calibration

See `blender-scene.md`.

## Video Export And Harmonisation

`runtime.settings.video_export` controls video generation. Export FPS comes from the runtime-resolved frame-provider cadence.

Current video-export keys:

- `enabled`
- `codec`
- `min_frames`
- `ground_grid.num_workers`
  - optional worker count for harmonized ground-grid frame preparation
  - `0` or omitted means auto-select based on clip length and host CPU count

`runtime.settings.harmonisation` is optional. When enabled, PEMOIN expects the Blender composition outputs that harmonisation consumes.

Current profile defaults point harmonisation I/O at the artifact tree:

- `overlay_dir=artifacts/blender/overlayed_frames`
- `occlusion_mask_dir=artifacts/blender/occlusion_masks`
- `output_dir=artifacts/harmonisation/harmonized_overlays`

When `runtime.settings.cross_run_cache.harmonisation.enabled=true`, runtime may reuse previously published Harmonizer outputs for an equivalent run. Diagnostics are written to `standard/providers/harmonisation.json`.

When `runtime.settings.cross_run_cache.ground_grid.enabled=true`, runtime may also reuse the generated `standard/videos/harmonized_overlays_ground_grid.mp4` for an equivalent harmonized-overlay + trajectory + road-plane + semantics state instead of regenerating the full overlay video.

Current harmonisation keys:

- execution and I/O
  - `enabled`
  - `conda_env`
  - `pretrained_path`
  - `overlay_dir`
  - `occlusion_mask_dir`
  - `output_dir`
  - `device`
- crop behavior
  - `mode`
    - current supported value: `local_crop`
  - `bbox_expansion_scale`
    - current default: `2.5`
  - `min_crop_size_ratio`
    - current default: `0.30`
  - `max_frame_coverage_ratio`
    - current default: `0.85`
  - `containment_margin_px`
    - current default: `8`
  - `reject_when_actor_exceeds_crop`
    - current default: `true`
  - `oversized_actor_behavior`
    - current supported value: `full_mask_affine_or_copy`
  - `full_frame_affine_min_mask_pixels`
    - current default: `512`
  - `mask_source`
    - current supported value: `visible_occlusion`
  - `empty_mask_behavior`
    - current supported value: `copy_through`
- eligibility gating
  - `eligibility.min_visible_mask_pixels_for_model`
  - `eligibility.min_visible_bbox_short_side_px_for_model`
  - `eligibility.max_crop_coverage_ratio_for_model`
  - `eligibility.max_crop_coverage_mask_pixels_threshold`
- diagnostics
  - `write_crop_diagnostics`
  - `write_crop_debug_overlays`
- color matching
  - `color_matching.enabled`
  - `color_matching.color_space`
    - current supported value: `lab`
  - `color_matching.ring_inner_px`
  - `color_matching.ring_outer_px`
  - `color_matching.exclude_top_band`
  - `color_matching.top_band_reference`
    - current supported value: `mask_top`
  - `color_matching.top_band_px`
  - `color_matching.use_semantics_for_sky_filter`
  - `color_matching.outlier_rejection`
    - current supported value: `robust_percentile`
  - `color_matching.luminance_match`
    - current supported value: `mean_std`
  - `color_matching.luminance_strength`
  - `color_matching.chroma_match`
    - current supported value: `mean_only`
  - `color_matching.chroma_strength`
  - `color_matching.prefer_pedestrian_reference`
  - `color_matching.pedestrian_reference_weight`
  - `color_matching.fallback_scene_reference_weight`
  - `color_matching.saturation_attenuation_strength`
  - `color_matching.contrast_attenuation_strength`
  - `color_matching.min_pedestrian_reference_pixels`
  - `color_matching.min_ring_pixels`
  - `color_matching.fallback_behavior`
    - current supported value: `skip_and_continue`
  - `color_matching.write_diagnostics`
- bounded correction clamps
  - `correction_clamps.min_foreground_pixels_for_luminance_scale`
  - `correction_clamps.min_foreground_luminance_std_for_scale`
  - `correction_clamps.luminance_delta_clamp_small_mask`
  - `correction_clamps.luminance_delta_clamp_model`
  - `correction_clamps.luminance_std_ratio_clamp`
  - `correction_clamps.chroma_shift_clamp`
- temporal smoothing
  - `temporal_smoothing.enabled`
    - current default: `true`
  - `temporal_smoothing.mode`
    - current supported value: `parameter_ema`
  - `temporal_smoothing.appearance_alpha`
    - current default: `0.85`
  - `temporal_smoothing.tonal_alpha`
    - current default: `0.92`
  - `temporal_smoothing.color_match_alpha`
    - current default: `0.85`
  - `temporal_smoothing.warmup_mode`
    - current supported value: `seed_from_first_valid`
- `temporal_smoothing.reset_on_empty_mask`
- `temporal_smoothing.reset_on_copy_through`
  - `temporal_smoothing.reset_on_harmonizer_failure`
  - `temporal_smoothing.reset_on_crop_iou_below`
  - `temporal_smoothing.reset_on_mask_area_ratio_outside`
  - `temporal_smoothing.reset_on_centroid_jump_fraction`
  - `temporal_smoothing.fallback_mode`
    - current supported value: `affine_rgb_gain_bias`
  - `temporal_smoothing.write_diagnostics`
- learned-result validation
  - `postcheck.max_ring_overshoot_luma`
  - `postcheck.max_small_mask_brighten_luma`

Current behavior is fail-fast and local-context aware:

- Blender overlay composition resolves depth-aware visibility first, then applies a narrow boundary-only edge treatment around the final visible silhouette; current defaults use a `4 px` band, `2 px` feather radius, slight boundary blur, slight halo suppression, and low-strength boundary-only regrain
- harmonisation thresholds the visible occlusion mask, computes a capped local expanded crop around that mask, optionally applies bounded local Lab-based foreground color matching against a filtered background ring, estimates learned parameters only on stable reference frames, and then fits/applies one track-level parameter curve so the same pedestrian track does not flicker between raw and harmonized frames
- close-up safety is fail-safe now: PEMOIN refuses to apply crop-limited harmonisation when the visible pedestrian extends beyond the local crop, so it will not recolor only a rectangular subset of the actor
- current defaults increase the harmonisation crop cap to `85%` of frame width/height, but if the actor still exceeds that crop PEMOIN falls back to a seam-safe full-visible-mask affine correction from track EMA when available and otherwise copies the pre-harmonised overlay through unchanged
- the color-matching ring is sampled outside the visible mask; current defaults use a `10-40 px` ring, partial luminance/chroma correction, heuristic top-band rejection, semantic `sky` exclusion when standardized semantics exist for the frame, robust median/IQR statistics, and clamped luminance/chroma transfer so tiny masks cannot produce extreme brightness jumps
- current temporal smoothing is parameter-level rather than pixel-level; PEMOIN smooths the bundled Harmonizer filter arguments `temperature`, `brightness`, `contrast`, `saturation`, `highlight`, and `shadow` with EMA defaults tuned separately for appearance vs tonal controls
- learned outputs are post-validated against local masked/ring luminance and are rejected when they overshoot the configured brightness guardrails; learned tracked frames now attenuate those rejected candidates into a bounded temporal-safe recovery instead of dropping a single frame back to the conservative pre-harmonized appearance, while non-track local failures still keep the bounded masked correction path
- temporal state resets conservatively on empty-mask frames, harmoniser failures, and large crop/mask discontinuities measured from crop IoU, visible-mask area ratio, and centroid jump fraction; tiny-object conservative copy-through no longer forces a reset by default
- frames with no visible pedestrian pixels are copied through unchanged instead of falling back to whole-frame harmonisation, and tiny/sparse visible masks either receive backfilled or interpolated learned parameters on a learned actor track or take one conservative track-wide harmonisation path when the track never becomes stably learnable
- current tiny-object harmonisation defaults are more conservative: `tiny_object.max_mask_pixels_for_conservative_path=256`, `tiny_object.max_bbox_short_side_px_for_conservative_path=20`, and `tiny_object.skip_color_match_below_mask_pixels=256`

## Cross-Run Cache Settings

`runtime.settings.cross_run_cache` controls shared content-addressed reuse across equivalent runs.

Common keys:

- `enabled`
- `root`
- `lighting.enabled`
- `geometry_fusion.enabled`
- `blender_scene.enabled`
- `harmonisation.enabled`

Provider-style cache publication currently applies to DPVO, UniDepth, CAVIS semantics, DiffusionLight-Turbo lighting, and GeometryFusion. Runtime-managed render-artifact reuse currently applies to Blender scene/render/composition artifacts and Harmonizer outputs.
- For standardized `.npz` inputs used in those cache keys, PEMOIN fingerprints canonical NPZ member content instead of raw archive bytes. Equivalent reruns therefore keep stable cache signatures across rewritten `poses.npz`, `intrinsics.npz`, depth, semantics, camera-height, and road-plane NPZ files.
- optional crop diagnostics are written next to the harmonized overlay output and are not standardized cross-stage resources; current diagnostics include eligibility decisions, crop-containment results, visible-mask-outside-crop counts, oversized-actor fallback decisions, raw vs smoothed parameter traces, post-check rejection reasons, rejected-candidate vs applied luma, recovery mode/strength, reset reasons, and fallback-transform metadata when used

## Top-Level Automation Blocks

Top-level profile blocks currently used by runtime/CLI include:

- `megasam`
  - commonly used for `conda_env` and bundle/cache settings
- `panst3r`
  - commonly used for `conda_env` and settings-file selection
- `unity_import`
  - controls pre-runtime Unity dataset import
- `mixamo`
  - provides Mixamo FBX paths and related scene settings

## Environment And CLI Overrides

Environment overrides:

- `PEMOIN_PROFILES_CONFIG`
- `PEMOIN_ACTIVE_PROFILE`

Common CLI overrides:

- `--config`
- `--profile`
- `--frames`
- `--output-root`
- `--max-frames`
- `--frame-rate`

Automation-related CLI flags:

- MegaSAM: `--megasam-*`
- PanSt3R: `--panst3r-*`

## Fail-Fast Policy

Profiles are intentionally strict. Invalid shapes, unsupported combinations, or missing required settings should raise clear errors rather than silently degrade behavior.
