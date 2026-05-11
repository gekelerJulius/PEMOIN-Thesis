# Integrations

## Purpose

This file documents PEMOIN-specific integration behavior for external tools and dataset-backed adapters.

It intentionally focuses on how PEMOIN uses them, not on reproducing upstream README content.

## MegaSAM

PEMOIN supports MegaSAM both as provider bindings and as CLI-assisted bundle preparation.

Relevant pieces:

- providers
  - `MegaSAMIntrinsicsProvider`
  - `MegaSAMDepthProvider`
  - `MegaSAMTrajectoryProvider`
- CLI automation
  - `--megasam-auto`
  - `--megasam-command`
  - `--megasam-preset`
  - `--megasam-output`
  - `--megasam-repo`
  - `--megasam-conda-env`

Current PEMOIN replay contract expects standardized bundle-style data and converts MegaSAM pose convention into PEMOIN/Blender convention before persistence.

## PanSt3R

PEMOIN supports PanSt3R as providers and through CLI bundle preparation.

Relevant CLI flags:

- `--panst3r-auto`
- `--panst3r-command`
- `--panst3r-preset`
- `--panst3r-output`
- `--panst3r-repo`
- `--panst3r-settings`
- `--panst3r-conda-env`

PanSt3R automation prepares a bundle and then applies it back into the selected profile/runtime context.

## DPVO

`DPVOTrajectoryProvider` is the current monocular VO provider used in DPVO-style profiles.

Notable behavior:

- uses deferred batch execution
- persists a canonical trajectory match graph for later scale handling
- supports provider-level cross-run cache behavior

## UniDepth

`UniDepthIntrinsicsProvider` and `UniDepthDepthProvider` supply learned depth/intrinsics.

Notable behavior:

- supports precision/device/batch settings
- can reuse provider exports via cross-run cache
- pins Hugging Face cache lookup to `HF_HOME`/`HF_HUB_CACHE` when PEMOIN repoints
  `XDG_CACHE_HOME` for mamba-managed subprocesses, so cached UniDepth weights remain
  visible across runs
- loads model weights offline-first from a local directory or existing Hugging Face
  cache before attempting any network-backed Hub resolution
- commonly paired with DPVO or GT-style trajectory sources

## DiffusionLight-Turbo

`DiffusionLightTurboLightingProvider` supplies clip-level lighting estimation.

Notable behavior:

- can reuse provider exports via cross-run cache
- pins `HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE` for the subprocess so
  Hugging Face-backed weights resolve from a stable cache root
- treats Hugging Face model sources as offline-first by default and fails fast
  before inference when required weights are not available locally
- accepts either Hugging Face model IDs or local directory paths for the SDXL
  base, VAE, ControlNet, and depth-estimator sources

## DepthAnything3

DepthAnything3 adapters are registered in the provider factory and can be used through profile bindings.

Current repo documentation should describe PEMOIN-specific use only. Upstream model internals or copied external API notes should not live here unless PEMOIN depends on them directly.

## Unity Import And Unity GT

Unity workflows in PEMOIN have two pieces:

- Unity GT providers for intrinsics/depth/trajectory/semantics
- optional pre-runtime Unity import controlled by the top-level `unity_import` profile block

When `unity_import.enabled=true`, CLI imports the selected dataset into the run directory before runtime begins and then points the frame source at the imported frames.

## CARLA

CARLA profiles use `CarlaFrameProvider` together with CARLA GT or learned providers.

Current profiles include both GT-style and DPVO/geometry-fusion style CARLA runs.

## Virtual KITTI 2

PEMOIN includes:

- `VirtualKitty2FrameProvider`
- Virtual KITTI 2 GT providers for intrinsics, depth, trajectory, and semantics

Adapter behavior is repo-specific and should stay summarized here rather than carrying upstream dataset documentation verbatim.

## NuScenes

PEMOIN includes:

- `NuScenesFrameProvider`
- `NuScenesIntrinsicsProvider`
- `NuScenesTrajectoryProvider`
- `NuScenesCameraHeightProvider`

NuScenes integration supports both keyframe-only and full per-camera stream sampling. Current NuScenes profiles use the full camera stream (`sampling_mode=all_camera_frames`); `nuscenes_gt` currently requests `sampling_fps=30`, while other profiles may choose a lower positive sampling rate. Runtime resolves the actual emitted FPS from timestamps.

For NuScenes GT providers:

- trajectory uses the exact selected `sample_data` record, including sweep-time `ego_pose`
- intrinsics are validated to remain constant within the selected camera stream
- camera height is derived from calibrated sensor translation and validated to remain constant within the selected camera stream

## Semantics Providers

Current semantics integrations include:

- `CAVISSemanticsProvider`
- `TemporalFusionSemanticsProvider`
- lazy-import providers such as Mask2Former, TwinLite SegFormer, and VideoKMaX

Some semantics providers also produce standardized `semantics_aux` sidecars or dynamic masks consumed by later geometry stages.

## Cross-Run Cache Note

Where provider-level cross-run cache exists, document it here and in `system-overview.md` as provider-specific behavior, not as a global runtime guarantee. Runtime now publishes those provider cache entries once the provider's own standardized outputs are durable, rather than waiting for the full pipeline to finish.
