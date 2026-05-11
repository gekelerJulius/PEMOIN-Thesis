# System Overview

## Purpose

This is the canonical overview of how PEMOIN is structured and how a run executes.

Read this when you need to understand runtime behavior, change an entrypoint, or decide where code should live.

## Core Modules

- `src/pemoin/cli.py`
  - Parses arguments, loads profiles, resolves frame sources, runs optional import/automation helpers, and launches the runtime.
- `src/pemoin/runtime/runtime.py`
  - Main orchestration loop, provider lifecycle, post-processing, video export, and diagnostics.
- `src/pemoin/runtime/orchestration/`
  - Frame providers and frame-provider binding resolution.
- `src/pemoin/runtime/profiles/config.py`
  - Strict profile parsing and validation.
- `src/pemoin/providers/`
  - Provider contract, factory registration, adapters, and batch geometry providers.
- `src/pemoin/data/contracts.py`
  - Standardized resource kinds, layouts, and `ResourceStore`.
- `src/pemoin/coordinate_systems/`
  - Pose conversion, comparison-frame canonicalization, and consistency helpers.
- `src/pemoin/visualization/`
  - Canonical visualization and video generation modules.

## Provider Model

- Profiles bind logical stages such as `intrinsics`, `depth`, `trajectory`, `semantics`, and `lighting` to concrete provider tool strings.
- Providers declare `required_resources` and `produced_resources`.
- Frame-oriented providers run in the per-frame loop via `process(...)`.
- Batch-oriented providers run later via `run(resources, context)`.
- Some providers are `deferred_batch` and consume the full persisted frame sequence after the loop rather than the in-memory frame.

## Frame Sources

Current frame-provider tools:

- `DirectoryFrameProvider`
- `VideoFrameProvider`
- `UnityFrameProvider`
- `CarlaFrameProvider`
- `VirtualKitty2FrameProvider`
- `NuScenesFrameProvider`

`NuScenesFrameProvider` can emit either keyframes only or the full per-camera frame chain (sweeps + keyframes). For current NuScenes profiles, PEMOIN uses the full camera stream and derives the emitted FPS from timestamps after sampling.

## Runtime Data Flow

High-level flow:

1. CLI parses args and loads the selected profile.
2. Frame source is resolved from CLI overrides or `frame_provider`.
3. Optional helpers may run before runtime:
   - Unity import
   - MegaSAM bundle preparation
   - PanSt3R bundle preparation
4. Runtime builds providers and a `ResourceStore`.
5. Providers run over frames in normalized working resolution.
6. Batch post-processing, validation, lighting export, visualization, and video export run after the frame loop.

Console logging is stage-aware by default. PEMOIN shows a tree-style trace of the active runtime phase together with its own progress bars for long-running loops and Blender render passes, while raw Blender subprocess stdout/stderr is still captured to `standard/logs/` instead of streaming every per-frame save line to the terminal. `--verbose` keeps the same phase model but adds more detailed substep logging. `--quiet` reduces console output to warnings/errors and suppresses progress bars.

Runtime also writes a hierarchical timing report to `standard/runtime/timeline.json`. That report records phase start/end times, durations, statuses, aggregate per-provider frame-loop timing, and cache outcomes for late stages such as Blender, harmonisation, and the harmonized ground-grid video.

## Per-Frame Order

For each frame, runtime persists the frame first and then runs providers in this order when configured:

1. intrinsics
2. depth
3. camera height
4. trajectory
5. semantics

Runtime also updates internal scene state caches during this loop.

Runtime now records aggregate per-provider timing across the whole frame loop rather than persisting one timing record per frame. Those provider totals are written under the frame-loop branch in `standard/runtime/timeline.json`.

## Post-Frame Order

After frame processing completes, runtime may perform:

1. trajectory consolidation and persistence
2. batch semantics execution
3. geometry fusion, when configured
   - all active profiles now use one joint metric-consistency path
   - GT camera height above road plane is treated as authoritative
   - DPVO/non-metric trajectories may change only by one global translation scale factor
   - estimated depth absorbs the remaining correction burden through constrained per-frame affine rectification
   - GT inputs are expected to remain near-zero correction and warn/fail when the shared solver would need materially larger changes
   - runs fail fast when corrected depth, trajectory, road support, and GT camera height cannot be brought into one common metric scene
4. pre-fusion geometry consistency validation with `ok` / `degraded` / `failed` outcomes
5. point-cloud generation
6. road-plane generation when geometry fusion did not already produce planes
7. comparison-frame canonicalization
9. post-processing geometry validation
10. clip-level lighting estimation when `providers.lighting` is bound
   - current `carla_gt` runs use `CarlaGTLightingProvider`, which consumes additive `lighting_gt/` metadata exported with the CARLA dataset and synthesizes the standardized lighting package from simulator weather plus optional non-vehicle scene-light fills
   - that CARLA GT path is daylight-first: CARLA weather drives one analytic `SUN` light for direct sunlight and one sky-only envmap for ambient illumination; vehicle lights are intentionally ignored
   - current `unity_gt` and `unity_dpvo` runs use `UnityGTLightingProvider`, which consumes additive `lighting_gt/` metadata exported with the Unity dataset and synthesizes the standardized lighting package from one directional sun light plus reflection-probe env lighting and ambient-probe-derived ambient strength
   - `carla_dpvo` and the NuScenes profiles continue to use DiffusionLight-Turbo clip-level estimation
11. dense point-cloud debug export from aligned/scaled depth + trajectory
12. Blender scene export, reusable character FBX export, harmonisation, visualizations, quality metrics, and videos

Before optional harmonisation, Blender now renders the pedestrian with one fast raster backend plus a local animated shadow-catcher proxy derived from grounding support. Runtime renders the pedestrian/depth outputs once at the configured fixed internal `resolution_scale`, exports single-pass shadow catcher frames, materializes `artifacts/blender/shadow_frames/` host-side, composites that localized cast-shadow alpha onto the real plate first, and then resolves visible pedestrian pixels with depth-aware occlusion. Traversable-ground semantics are required for that overlay pass, and any pixel labeled as traversable ground is prevented from occluding the inserted pedestrian. Standardized semantics stay in working-resolution frame space; when Blender render scaling produces smaller overlay artifacts, PEMOIN resamples the traversable-ground mask into the overlay/composite target shape with nearest-neighbor semantics-safe resizing. All downstream overlay compositing and support/contact validation run in final overlay/background resolution only. Current occlusion also applies temporal hysteresis for borderline small-actor visibility changes so sub-centimeter depth jitter does not produce one-frame pops near strong occluders or image edges. PEMOIN then applies default-on boundary-only edge treatment on the final visible silhouette. That treatment uses a narrow boundary strip rather than the whole pedestrian and combines slight feathering, boundary blur, halo suppression, and low-strength regrain, but tiny visible silhouettes automatically clamp or bypass those destructive operations entirely when the visible actor is too small or too boundary-dominated.

Current Blender backend orchestration also trims avoidable work around the core raster pass. PEMOIN enables persistent render data when the Blender build supports it, encodes intermediate pipeline PNGs with fast settings, disables several expensive EEVEE screen-space features for these pipeline-internal pedestrian/shadow renders by default, realizes Mixamo subject materials through a faster policy that preserves base-color/alpha/normal response while flattening lower-value secondary shading maps by default, binds dynamic subject-relative fill lights without per-frame location keyframes, can route only disposable boundary-transition visible frames through a reduced-cost render path with slightly lower internal resolution and milder fill-light/shadow reductions before upsampling those artifacts back to the baseline internal render shape, skips raw-subject exposure entirely when the configured correction is a no-op, and uses grounding/visibility diagnostics to avoid rasterizing frames that are already known to be fully off-camera. Tiny visible pedestrians are treated as quality-protected rather than low-salience. Those visibility-culled frames still receive synthesized empty pedestrian/depth/shadow artifacts so downstream composition keeps a complete frame sequence and the visibility contract remains explicit in diagnostics.

Blender scene export now also emits a reusable single-clip character FBX artifact before grounding/render-only scene state is applied. PEMOIN duplicates the baked character hierarchy into an isolated export family, trims that duplicate to one authored clip span, normalizes the root transform so the exported actor starts at origin instead of the PEMOIN scene spawn, exports the mesh/armature/root hierarchy through Blender's built-in FBX exporter with Unity-oriented defaults, writes `artifacts/blender/fbx_exports/character_root_motion.fbx` plus `character_root_motion.export.json`, and copies the FBX to the run root as `character_root_motion.fbx`. That asset is a late-stage Blender artifact for interchange, not a standardized cross-stage pipeline input.

Overlay support validation remains fail-fast for trustworthy grounding evidence, but it no longer treats every suspicious frame as equally authoritative. PEMOIN now separates verified support/contact mismatches from degraded or unverifiable cases such as edge-clipped actors whose selected contact foot has already left the frame or trajectory-path frames whose support anchor cannot be trusted after projection. Those accepted-but-unverifiable frames are still written to diagnostics loudly, but run-level aborts are driven only by hard failures and trusted aggregate support/contact statistics.

Pre-fusion geometry consistency validation no longer aborts solely because isolated pairwise failures require a few frame replacements. PEMOIN now computes a minimal replacement set for recoverable catastrophic pairs, logs a degraded warning, persists degraded diagnostics, and continues. It still fails fast when contiguous catastrophic runs exceed the configured hard-fail limit or when no valid replacement anchors remain.

When `runtime.settings.validation_policy.enabled=true`, selected geometry-quality gates also use adaptive low-FPS severity. PEMOIN treats `10 FPS` as the reference cadence, keeps existing thresholds at or above that cadence, and gradually relaxes selected quality thresholds and frame-count budgets below it. Those soft-threshold overruns continue with degraded warnings and persisted diagnostics, while hard-threshold overruns and integrity failures still abort.

When `runtime.settings.harmonisation.enabled=true`, PEMOIN thresholds the visible pedestrian occlusion mask, derives a capped local crop, optionally applies bounded local Lab-based color matching inside that crop, and runs harmonisation as an offline two-pass track process. PEMOIN first estimates safe learned parameters only on stable reference frames, then fits one track-level parameter curve and applies it to every visible frame on that actor track, including early tiny frames that appear before the first stable learned reference. That backfilled application removes the old “not harmonized at the beginning, then harmonized later” transition. If a track never becomes stable enough for learned estimation, PEMOIN uses one conservative track-wide harmonisation path instead of mixing learned and raw frames. Shadow pixels are not part of the harmonisation mask. Close-up safety is fail-safe now: when the visible actor exceeds the local crop, PEMOIN does not apply crop-limited harmonisation to only that subset of the actor. Current defaults first try a seam-safe full-visible-mask affine fallback from track EMA and otherwise copy the pre-harmonized overlay through unchanged.

Current temporal smoothing operates on the bundled Harmonizer's explicit filter arguments (`temperature`, `brightness`, `contrast`, `saturation`, `highlight`, `shadow`) plus PEMOIN's derived local color-match parameters. Conservative resets clear temporal state on empty-mask frames, harmoniser failures, and large crop/mask discontinuities; tiny-object conservative copy-through frames no longer force a reset by default. Learned outputs are also post-validated against local masked/ring luminance. When a tracked frame fails that post-check, PEMOIN now attenuates the rejected result into a bounded temporal-safe recovery instead of snapping that one frame back to the pre-harmonized crop. Diagnostics record track ids, span ids, track policy, applied-parameter source, whether a frame was used for reference estimation, eligibility decisions, crop-containment decisions, oversized-actor fallback decisions, raw vs smoothed parameters, post-check rejections, recovery mode/strength, and reset reasons next to harmonized overlay outputs.

Grounding is trajectory-first now. Blender preserves the authored pedestrian root `x,y` path after spawn placement and solves only for root `z`. PEMOIN treats imported character assets as already metric-sized, preserves imported character scale as-is, derives only the asset-native root-to-support relation needed for grounding, queries the road-plane set under the authored path across the full authored track, builds contiguous traversed-plane segments, and smooths only the vertical profile across plane handoffs. Root motion is no longer driven by contact ownership, planted-foot locks, or per-frame support-anchor blending. For current Mixamo insertion, the authored `x,y` path is selected from the animation asset path category: clips under `assets/mixamo/animations/idle/` remain stationary at spawn, while clips under `assets/mixamo/animations/moving/` use animation-root-motion world translation extracted from the imported clip. `pedestrian_trajectory_t` chooses only the initial trajectory anchor/spawn; it does not make moving clips follow the camera trajectory after frame 0. Current extraction samples wrapped source pose time but tracks unwrapped authored cycle progress explicitly, so repeated walk loops preserve continuous forward motion and abort on backward seam regressions instead of repairing them later with clipped deltas. Moving clips also stabilize the pelvis from the evaluated world transform each frame, remove only the heading-aligned locomotion component in world space, and convert that corrected pelvis transform back into pose space before keyframing, which avoids rig-specific local-channel seam resets. Heading now resolves from measured imported-rig body-facing instead of a fixed FBX forward guess, and Blender validates sampled baked body-facing against the resolved world heading before render export. Camera trajectory selects only the insertion spawn/heading basis unless the explicit legacy `camera_trajectory_relative` policy is requested. Paths outside those directories fail fast. Grounding diagnostics now focus on authored root pose, grounded root pose, traversed support-plane ids, per-frame plane height, vertical velocity/acceleration, trajectory support segments, motion-direction parity, body-facing parity, and a render-vs-grounding visibility contract written alongside the usual road-surface summary.

## Caching And Reuse

PEMOIN does not treat old `standard/` outputs as an unconditional source of truth for arbitrary runtime reuse, but several providers do support cross-run cache materialization and publishing through runtime cache settings.

Current code paths include cache-aware behavior for providers such as:

- DPVO
- UniDepth
- CAVIS semantics
- DiffusionLight-Turbo lighting
- GeometryFusion

For cache-aware providers, runtime now publishes cross-run cache entries as soon as that provider's raw outputs and standardized artifacts are both durable. This avoids losing reusable cache artifacts when a later stage such as lighting, point-cloud export, Blender export, harmonisation, or video generation fails after the expensive provider already completed.

For standardized `.npz` resources such as trajectory, intrinsics, depth, semantics, camera-height, and road-plane outputs, cache signatures are computed from canonical NPZ member content rather than raw ZIP container bytes. Equivalent reruns therefore keep the same cache key even when those standardized files are rewritten.

Runtime also manages a separate render-artifact reuse layer for non-provider late stages. Current render bundles cover:

- Blender scene export artifacts such as `scene.blend`
- Blender reusable character FBX artifacts such as `artifacts/blender/fbx_exports/character_root_motion.fbx`
- Blender render/composition outputs under `artifacts/blender/` such as `pedestrian_frames/`, `shadow_frames/`, `overlayed_frames/`, and `occlusion_masks/`
- Harmonizer outputs under `artifacts/harmonisation/` such as `harmonized_overlays/` and `harmonized_overlays_diagnostics/`

These render bundles are still non-standardized run artifacts under `outputs/<run>/artifacts/`, not standardized resources. They are reused through content-addressed manifests under the shared cross-run cache root, and runtime records lookup/publish diagnostics for `blender_scene` and `harmonisation` in `standard/providers/`. The harmonized ground-grid MP4 under `standard/videos/` may also be reused through the same cache layer. Cache hit, materialization, and publish outcomes are recorded in `standard/runtime/timeline.json` next to the affected late-stage timing nodes.

Current Blender artifact reuse is now recorded with narrower bundle boundaries: scene export, pedestrian/depth render outputs, shadow outputs, composition outputs, and harmonisation outputs. Runtime still executes Blender through the same entrypoint, but cache metadata and published artifact bundles are split so rerun invalidation and diagnostics can distinguish render-vs-shadow-vs-composition churn.

The run root keeps `scene.blend` and `character_root_motion.fbx` as convenience files and also copies the selected final pedestrian video to `output.mp4`. Standardized videos remain under `standard/videos/`.
The run root also keeps point-cloud convenience GLBs as `rgb_pointcloud.glb` and `semantic_pointcloud.glb`, copied from `artifacts/geometry/point_cloud/`.

Overlay and harmonisation raster artifacts are persisted in normal top-origin image space. Late-stage video export preserves that orientation and pads odd frame dimensions to even codec-safe sizes instead of silently cropping rows or columns.

Blender subprocess timing diagnostics under `standard/visualizations/blender_scene/render_backend_diagnostics.json` now also record the effective backend fast-path settings plus the frame-plan split between baseline/protected visible frames, reduced-cost disposable-transition frames, and visibility-culled frames so benchmark runs can separate backend savings from scene-content differences.

When documenting caching, describe the current provider-specific cross-run cache behavior instead of saying output reuse is globally disabled.

## Output Discipline

- Standardized cross-stage resources live under `outputs/<run>/standard/`.
- Provider-native outputs, caches, and diagnostics live under `outputs/<run>/raw/<provider>/`.
- Late-stage Blender and harmonisation artifacts live under `outputs/<run>/artifacts/`.
- End-of-geometry point-cloud debug GLBs live under `outputs/<run>/artifacts/geometry/point_cloud/`.
- Downstream stages must consume standardized resources only.
- Lighting follows the same rule: the cross-stage contract is `standard/lighting/lighting.json` plus `standard/lighting/envmap.exr`, while DiffusionLight-Turbo intermediates stay under `raw/lighting/`. The EXR is the canonical fused HDR envmap. The JSON package now adds a versioned analytic `light_rig` decomposition with `rig_mode` (`analytic_rig`, `sun_plus_fill`, or `envmap_only`), per-light shadow policy, and decomposition diagnostics derived from provider-native HDR sun consensus plus diffuse-lobe extraction. Invalid envmaps still abort publish; decomposition may degrade to a simpler rig instead of failing the run. Cross-run reuse for lighting is now split between reusable DLT inference artifacts and the final standardized lighting package so planner-only rig changes can still reuse expensive DLT outputs.

See `data-contract.md` for the canonical layout.

## Current Profiles

Profiles currently defined in `config/profiles.json`:

- `unity_gt`
- `unity_dpvo`
- `carla_dpvo`
- `carla_gt`
- `nuscenes_gt`
- `nuscenes_dpvo`

Current lighting is profile-dependent: `carla_gt` binds `providers.lighting` to `CarlaGTLightingProvider`, the Unity profiles bind `providers.lighting` to `UnityGTLightingProvider`, and `carla_dpvo` plus the NuScenes profiles bind `providers.lighting` to `DiffusionLightTurboLightingProvider`.

## Where To Change What

- CLI flags or launch behavior: `src/pemoin/cli.py`
- Frame-source parsing: `src/pemoin/runtime/orchestration/frame_provider_builder.py`
- Runtime execution order: `src/pemoin/runtime/runtime.py`
- Profile schema validation: `src/pemoin/runtime/profiles/config.py`
- Provider registration: `src/pemoin/providers/factory.py`
- Persisted resource paths or payload formats: `src/pemoin/data/contracts.py`
- Alignment/canonicalization/grounding: `src/pemoin/coordinate_systems/`
- Visualizations and videos: `src/pemoin/visualization/`

## Documentation Rule

When runtime behavior changes, update this file together with:

- `profile-reference.md`
- `data-contract.md`
- `validation.md`

depending on what changed.
