# Blender Scene

## Purpose

This file documents PEMOIN's current Blender scene export and Mixamo integration behavior.

It is intentionally PEMOIN-specific. Generic Blender or Mixamo tutorials do not belong here.

## When It Runs

Blender scene generation is controlled by `runtime.settings.blender_scene`.

When enabled, PEMOIN can:

- export a trajectory/debug scene
- export a reusable single-clip character FBX artifact
- insert a Mixamo pedestrian
- render pedestrian frames and overlay composites
- generate local-support overlay visuals

## Required Inputs

When Blender scene export is enabled, profiles must provide:

- `mixamo.character_fbx_path`
- `mixamo.animation_fbx_path`
- a valid Mixamo asset root, either explicitly via `mixamo.asset_root` or implicitly as the character FBX parent directory
- a runtime-resolved frame cadence suitable for scene rendering

PEMOIN assumes imported character assets are already correctly metric-sized. Blender scene export preserves imported character scale as-is and never rescales characters automatically.

## Current Behavior

PEMOIN uses the already-standardized trajectory directly. The Blender script no longer applies a separate Blender-only recentering step.

Current scene behavior includes:

- optional PEMOIN-managed lighting rig
- standardized clip-level lighting consumption from `standard/lighting/` when present
- fast raster pedestrian rendering with a local animated shadow-catcher proxy derived from the resolved support surface
- pedestrian placement from `pedestrian_trajectory_t` plus forward/left/up offsets
- separate actor-root motion policy controlled by `pedestrian_motion_policy`
  - current default `auto` resolves from the animation path category, not inferred clip motion
  - `assets/mixamo/animations/idle/` keeps clips fixed at the resolved spawn point
  - `assets/mixamo/animations/moving/` uses heading-aligned animation-root-motion world translation
  - `pedestrian_trajectory_t` only selects the initial trajectory anchor/spawn for moving clips; after frame 0 the actor travels in scene world according to the resolved animation root motion, not the camera path
  - for moving clips, `pedestrian_heading_deg` resolves one constant post-spawn world heading and PEMOIN uses that heading for both body-facing and walking while preserving the clip's forward progress over time
  - forward locomotion is transferred onto the actor root; the baked hips/pelvis pose keeps lateral sway and vertical body motion but should not keep looping forward translation
  - moving-clip pelvis stabilization now works from the evaluated pelvis world transform, removes only heading-aligned world forward drift, and converts the corrected transform back into pose space instead of relying on rig-local channel heuristics
  - camera trajectory now selects spawn/heading only unless the explicit legacy `camera_trajectory_relative` policy is requested
  - paths outside those directories fail fast
- optional camera-relative placement optimization before Blender insertion; current optimized mode searches nearby trajectory anchors and offset/heading candidates, simulates projected support-point motion from standardized trajectory + intrinsics, rejects candidates that become too near-camera or too bottom-clipped, and fails fast if no valid candidate remains
- motion-relative heading control via `pedestrian_heading_deg`
- measured imported-body-facing resolution with fail-fast fallback diagnostics, so PEMOIN does not rely on clip locomotion to resolve heading and can prove that rendered body-facing matches the configured heading
- support resolution against persisted road geometry
- foot-contact planning with `mixamo_phase` or legacy `nearest_plane`
  - `mixamo_phase` is automatic: PEMOIN derives gait phase from the imported Mixamo cycle timing and uses the configured stance windows to build contact segments; users do not label support frames manually
  - imported animation clips still provide pose timing and gait/contact timing, but they no longer need usable horizontal locomotion to define actor heading
  - source animation sampling now stays bound to the imported animation rig while PEMOIN bakes a separate target action onto the inserted character rig
- automatic reusable FBX export for later Blender/Unity import
  - PEMOIN duplicates the baked character hierarchy into an isolated export family before grounding/render-only state is applied
  - the export keeps one authored clip span, preserves root motion, normalizes the actor back to origin instead of the PEMOIN world spawn, and writes `artifacts/blender/fbx_exports/character_root_motion.fbx`
  - PEMOIN also writes `artifacts/blender/fbx_exports/character_root_motion.export.json` with clip span, exporter settings, and texture-embedding diagnostics, plus a convenience copy `character_root_motion.fbx` at the run root
  - current exporter settings are Unity-oriented Blender FBX defaults: selected objects only, `EMPTY` + `ARMATURE` + `MESH`, no leaf bones, one active action, no NLA/all-actions export, `path_mode='COPY'`, and `embed_textures=True`
- true per-pixel pedestrian depth rendered from Blender's Z pass
- separate `shadow_frames/` output so Python can composite shadow and pedestrian independently
- traversable-ground-aware overlay occlusion that never lets resolved road, sidewalk, street, pavement, walkway, crosswalk, ground, or floor semantics occlude the inserted pedestrian while keeping strict depth occlusion for non-ground scene pixels
- default-on boundary-only edge treatment on the final visible silhouette, including slight feathering, halo suppression, narrow boundary blur, and low-strength boundary-only regrain
- backend-oriented render fast paths for pipeline-internal artifacts, including persistent render data when available, fast intermediate PNG encoding, default disabling of several costly EEVEE screen-space effects for the internal pedestrian/shadow renders, a default subject-material fast path that preserves base color/alpha/normal response while flattening secondary shading maps, dynamic subject-relative fill-light binding without per-frame light-location keys, a conservative protection-first adaptive render path that keeps tiny or central visible pedestrians on the baseline path and only allows a milder reduced-cost pass for disposable boundary-transition frames before upsampling back to the standard internal render shape, no-op skipping for raw-subject exposure when the configured correction would be identity, and visibility-culled frame skipping with synthesized empty pedestrian/depth/shadow outputs
- render/composition outputs such as `overlayed_frames` and `overlayed_frames_support_local_grid`

## Key Settings

Important `runtime.settings.blender_scene` groups:

- scene/lighting
  - `enabled`
  - `cube_size`
  - `collection_name`
  - `lighting`
- global support search
  - `global_plane_range_m`
  - `global_plane_min_range_m`
  - `global_plane_frame_window`
  - `global_plane_max_points_per_frame`
  - `global_plane_confidence_threshold`
  - `global_plane_trim_ratio`
- local support search
  - `local_support_radius_m`
  - `local_support_frame_window`
  - `local_support_min_points`
  - `local_support_plane_size_m`
  - `local_support_max_radius_m`
  - `local_support_radius_step_m`
  - `local_support_snap_to_nearest_road`
  - `local_support_snap_radius_m`
  - `local_support_temporal_hold_frames`
  - `local_support_temporal_hold_seconds`
- pedestrian placement
  - `pedestrian_trajectory_t`
  - `pedestrian_forward_offset_m`
  - `pedestrian_left_offset_m`
  - `pedestrian_up_offset_m`
  - `pedestrian_heading_deg`
- trajectory-first grounding
  - `max_plane_center_xy_distance_m`
  - `trajectory_grounding_transition_frames`
  - `trajectory_grounding_max_step_m`
  - `trajectory_grounding_max_vertical_velocity_mps`
  - `trajectory_grounding_max_vertical_accel_mps2`
- shadow catcher
  - `shadow.enabled`
  - `shadow.receiver_patch_size_m`
  - `shadow.map_resolution`
  - `shadow.softness`
  - `shadow.opacity`
  - `shadow.tint_rgb`
- render backend
  - `render.engine`
  - `render.resolution_scale`
  - `render.samples`
  - `render.material_policy`
  - `render.dynamic_light_binding`
  - `render.salience_adaptive.enabled`
  - `render.salience_adaptive.low_salience_resolution_scale`
  - `render.salience_adaptive.protect_below_visible_pixels`
  - `render.salience_adaptive.protect_below_bbox_short_side_px`
  - `render.salience_adaptive.protect_when_center_distance_ratio_below`
  - `render.salience_adaptive.reduce_only_when_boundary_fraction_above`
  - `render.salience_adaptive.reduce_only_near_visibility_transition`
  - `render.salience_adaptive.shadow_quality_reduction_enabled`
  - `render.salience_adaptive.fill_light_reduction_enabled`
- overlay occlusion
  - `occlusion.depth_source`
  - `occlusion.contact_ground_roles`
  - `occlusion.contact_ground_labels`
  - `occlusion.default_front_margin_m`
  - `occlusion.relative_margin`
  - `occlusion.contact_plane_band_m`
  - `occlusion.contact_patch_radius_m`
  - `occlusion.contact_coplanar_tolerance_m`
  - `occlusion.write_debug`
  - `occlusion.edge_treatment.enabled`
  - `occlusion.edge_treatment.boundary_band_px`
  - `occlusion.edge_treatment.feather_radius_px`
  - `occlusion.edge_treatment.feather_strength`
  - `occlusion.edge_treatment.blur_enabled`
  - `occlusion.edge_treatment.blur_radius_px`
  - `occlusion.edge_treatment.blur_strength`
  - `occlusion.edge_treatment.despill_enabled`
  - `occlusion.edge_treatment.despill_strength`
  - `occlusion.edge_treatment.regrain_enabled`
  - `occlusion.edge_treatment.regrain_strength`

## Overlay Occlusion

Current overlay compositing uses the rendered pedestrian RGBA, a per-pixel pedestrian depth map, and a separate shadow pass. PEMOIN now treats Blender depth export as a true render artifact, not a best-effort fallback:

- on Blender 5.x builds that expose `scene.compositing_node_group`, PEMOIN configures compositor output through that API
- on that Blender 5.x node-group path, PEMOIN creates depth EXR outputs through `CompositorNodeOutputFile.file_output_items`
- on older Blender builds that still expose `scene.node_tree`, PEMOIN uses the legacy compositor path
- on the legacy scene-tree path, PEMOIN configures the file-output node through `file_slots`
- Blender writes raw pedestrian depth EXRs from the Z pass into `_pedestrian_depth_exr/`
- PEMOIN now writes depth EXRs from the Z pass, renders one dedicated shadow-catcher PNG sequence from the same grounded scene with the raster backend, and synthesizes final `shadow_frames/*.png` host-side from that single pass; if Blender provides no usable shadow signal for a frame, PEMOIN writes a transparent shadow PNG instead of aborting the run
- PEMOIN then decodes those EXRs with the host Python interpreter, not Blender's embedded Python, and materializes `pedestrian_depth_frames/*.npz`

Pure Blender-side visualization imports are intentionally tolerant of missing `imageio` in Blender's embedded Python so scene assembly can still import. Host-side PNG processing and standardized frame or mask persistence still require `imageio` in the host PEMOIN environment and fail fast at the first PNG-backed operation.

PEMOIN no longer relies on Blender-side EXR reads for this step because Blender 5.0.1 on this project setup can write valid depth EXRs while `bpy.data.images.load(...)` does not reliably decode them in background mode.

The compositor applies:

- shadow compositing onto the real video frame before pedestrian compositing, with configurable opacity, blur, and tint
- strict depth occlusion for normal scene pixels
- full-silhouette visibility preservation for pixels whose scene semantics resolve to traversable ground
- a narrower contact-aware support-plane override for diagnostics and planted-foot continuity near the resolved support anchor
- fail-fast validation when traversable-ground semantics are unavailable or cannot be resolved for an overlay frame
- nearest-neighbor resampling of the resolved traversable-ground mask into overlay space when `render.resolution_scale` produces lower-resolution Blender render artifacts than the standardized frame/semantics resolution
- a boundary-only edge-treatment pass on the final visible silhouette, not the whole pedestrian; current defaults use a `4 px` band with slight feathering, slight halo suppression, slight boundary blur, and low-strength boundary-only regrain
- post-composite support validation now distinguishes trusted grounding failures from edge-clipped or otherwise unverifiable frames; PEMOIN still aborts on trustworthy support/contact mismatches, but records accepted degraded diagnostics instead of aborting when the selected contact foot is off-screen or validation had to fall back to weaker evidence

The goal is to keep inserted pedestrians from being cut by road-like scene labels while still honoring non-ground depth occlusion and keeping the grounding shadow independently controllable in Python.

## Fail-Fast Behavior

Current Blender/Mixamo flow fails fast when required assets, Mixamo material textures, FBX-export texture payloads, foot bones, support inputs, host-side depth decode, render visibility parity, or body-facing heading parity cannot be resolved reliably. Heading now comes from measured imported-rig body-facing rather than inferred clip locomotion, so stationary clips such as idles are supported and moving clips must keep rendered body-facing aligned with the configured `pedestrian_heading_deg`. PEMOIN also aborts when projected-visible actor frames render with zero visible pedestrian alpha, instead of silently continuing with empty pedestrian renders.

Current grounding diagnostics record authored root pose, grounded root pose, traversed support-plane ids, support height under the authored path, vertical velocity/acceleration, trajectory support segments, and a render-vs-grounding visibility contract so support-surface failures can be debugged directly from the planned root trajectory.

Current stable-ground smoothing is trajectory-first rather than contact-segment-lock based. PEMOIN keeps the authored root `x,y`, determines which persisted road planes the authored path traverses across the full track, evaluates support height at the current path location, and smooths only the resulting root `z` profile across plane transitions. Root motion is no longer driven by planted-foot locks or per-frame support-anchor blending, and temporary projected off-camera states no longer zero the grounded root height.

Persisted support bootstrap is also trajectory-aware rather than using only one fixed XY cutoff. `max_plane_center_xy_distance_m` remains the strict base locality threshold, but when the inserted pedestrian is validly offset farther from the camera trajectory corridor PEMOIN may relax that locality budget internally up to the configured global road-plane range. Diagnostics now record the nearest persisted-plane center distance, the effective locality limit used, and whether the accepted candidate required bootstrap-relaxed locality.

That behavior is intentional because downstream composition/harmonisation depends on the scene outputs being geometrically valid.

Lighting follows the same pattern. If a standardized lighting package exists, Blender requires a validated lighting package and uses `standard/lighting/lighting.json` plus `standard/lighting/envmap.exr` directly. The EXR remains the canonical world envmap. Blender consumes the standardized analytic `light_rig` from `lighting.json` and instantiates each light according to its published transport semantics. Validated direct lights remain `SUN` lights. Current diffuse subject fills use explicit wrap roles such as `wrap_key_fill`, `counter_wrap_fill`, and `sky_fill`, and Blender realizes those as shadowless `POINT` lights rather than literal one-sided area panels. For those wrap fills, standardized `strength` is treated as abstract transport strength and Blender converts it to effective `POINT` power using the published actor-relative offset plus the profile-backed `runtime.settings.blender_scene.lighting.wrap_subject_fill` realization controls. Those controls are now intent-oriented as well as numeric: they can bias counter-side lift, sky softness, and direct-light preservation rather than only scaling each role blindly. Upstream, the lighting provider now uses a hybrid-adaptive wrap-geometry planner instead of one fixed offset template. It exposes `diffuse_softness_bias`, `fill_heavy_dark_side_target_ratio`, `fill_heavy_transport_gain`, and the `wrap_geometry_*` controls so diffuse-scene planning can search for a wrap layout with stronger counter-side and sky support before the standardized rig is accepted. `placement_mode=subject_anchor_relative` means the stored fill offset is subject-relative instead of absolute world-space, and `placement_target=subject_root_dynamic` means Blender must keep that offset attached to the final grounded pedestrian root over time rather than only resolving it once at the initial spawn. Current fast rendering keeps that dynamic-anchor behavior but now defaults to a location-only Blender binding path instead of per-frame authored light-location keys. The provider also publishes clip-level view-aware fill diagnostics and dark-side brightness checks so at least one wrap fill lifts the non-sun side of the visible subject instead of only balancing a camera-agnostic subject proxy. That makes spawn resolution and post-grounding actor motion a hard dependency for standardized dynamic anchor-relative fills. Only validated direct lights cast shadows; wrap fills are shading-only. If the analytic rig is too weak but the canonical envmap is still plausible, the provider may now publish a validated degraded `envmap_only` fallback instead of aborting. If the provider failed to validate the canonical envmap itself, runtime aborts before Blender rendering. If no standardized lighting package exists because lighting is not configured for the run, Blender falls back to the built-in neutral rig.

Mixamo assets now follow the same discipline. Blender no longer trusts imported FBX texture paths blindly. It resolves a canonical Mixamo asset root, relinks imported `TEX_IMAGE` nodes by basename against that package when they are file-backed, accepts embedded FBX textures when Blender reports `packed_file`, normalizes the imported Mixamo legacy FBX material graph into a stable Principled interpretation, and then applies the selected fast-render material policy before rendering. The default fast policy keeps base-color, alpha, and normal-map response while flattening lower-value roughness/specular/gloss branches to cheaper constants. PEMOIN writes the chosen material policy and normalization counts to `standard/visualizations/blender_scene/mixamo_asset_diagnostics.json` and aborts only if a required image is neither packed nor resolvable from the asset root. The reusable FBX export reuses that same validated image set and aborts before export if Blender cannot embed or resolve the texture payloads needed for the selected character materials. That keeps saved `.blend` files, rendered pedestrians, and exported interchange FBXs tied to usable local or embedded texture payloads instead of transient exporter paths such as `/home/app/.../*.fbm/...`.

Raw pedestrian rendering also now includes a clip-level subject-exposure calibration pass before overlay composition. After `pedestrian_frames/` are rendered, PEMOIN first normalizes the RGBA sequence to a straight-alpha contract when the render backend emitted premultiplied-looking PNGs, records that result in `standard/visualizations/blender_scene/pedestrian_rgba_diagnostics.json`, and then runs a conservative fine-trim stage over the visible subject only. That exposure stage uses robust luminance anchors instead of full-body means, prefers nearby real-pedestrian references when semantics provide them, falls back to a local scene ring when they do not, caps the clip-wide gain to a low-amplitude correction band, and aborts the correction entirely when predicted residuals or gain dispersion indicate the calibration would be unstable. When the configured target-match strength and trim already imply an identity correction, PEMOIN records a no-op diagnostic and skips the calibration scan entirely. The applied result is written back to `pedestrian_frames/` and recorded in `standard/visualizations/blender_scene/raw_subject_exposure_diagnostics.json`. Shadow extraction now follows the fixed single-pass path: PEMOIN exports the receiver-luma shadow render from the main pedestrian pass, materializes `shadow_frames/`, and records metadata with `mode=single_pass_receiver_luma` so the downstream compositor receives localized cast-shadow alpha without extra baseline rerenders. Pedestrian rendering still uses `render.resolution_scale` as the baseline internal budget, but PEMOIN now treats tiny or central visible pedestrians as quality-protected and only allows a reduced-cost pass for boundary-transition frames that are safe to degrade. Those reduced frames render at `render.salience_adaptive.low_salience_resolution_scale` with milder fill-light and shadow reductions before upsampling back to the baseline internal shape. Frames that the grounding/visibility plan already marks as fully off-camera may still skip the Blender raster pass entirely while PEMOIN synthesizes empty pedestrian/depth/shadow outputs for downstream consumers. `standard/visualizations/blender_scene/render_backend_diagnostics.json` now records the adaptive frame split, per-frame reason codes, and the reduced-cost policy actually used. Harmonisation remains downstream, but it now starts from a better-calibrated raw subject render, with pedestrian-aware saturation and contrast softening when valid scene references exist.

## Related Outputs

Current Blender-related outputs commonly include:

- provider diagnostics under `raw/`
- standardized lighting inputs under `standard/lighting/`
- `artifacts/blender/fbx_exports/character_root_motion.fbx` and `artifacts/blender/fbx_exports/character_root_motion.export.json`
- `pedestrian_frames/`, `pedestrian_depth_frames/`, and `shadow_frames/`
- overlay image sequences
- optional harmonisation inputs
- videos exported from the overlay directories

See `data-contract.md` and `system-overview.md` for the standardized output rule.
