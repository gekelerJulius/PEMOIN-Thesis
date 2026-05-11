# Validation

## Purpose

This file covers the implemented validation and evaluation stages in PEMOIN.

There are three distinct layers:

1. pre-fusion geometry consistency validation
2. post-processing geometry validation
3. optional quality metrics and human-readable artifacts

## Pre-Fusion Geometry Consistency Validation

Configured under `runtime.settings.geometry_consistency_validation`.

This stage runs before point-cloud fusion and checks depth/pose/intrinsics consistency on consecutive frame pairs.

Current outcomes:

- `ok`: no catastrophic consecutive-frame pairs were detected
- `degraded`: isolated catastrophic pairs were detected, PEMOIN selected a minimal replacement set, logged a strong warning, and continued
- `failed`: geometry collapse was considered definitive and PEMOIN aborted before downstream stages

Main settings:

- `enabled`
- `pixel_stride`
- `min_overlap_points`
- `min_static_overlap_points`
- `exclude_dynamic_pixels`
- `dynamic_mask_source`
- `reprojection_error_px`
- `max_reprojection_rmse_px`
- `max_reprojection_p90_px`
- `max_reprojection_p95_px`
- `reprojection_catastrophic_mode`
- `min_inlier_ratio`
- `max_depth_scale_drift`
- `max_consecutive_catastrophic`
- `max_skipped_frames`

Current semantics:

- dynamic/mobile pixels are excluded when possible by preferring `standard/dynamic_mask/` and falling back to mobile semantic-role masking from `standard/semantics_2d/`
- robust reprojection percentiles (`p90` / `p95`) are the primary reprojection failure signal; raw RMSE remains a supporting diagnostic for long tails and summaries
- pairwise outcomes are classified as `ok`, recoverable catastrophic, or severe catastrophic
- `max_consecutive_catastrophic` is the hard-fail limit for contiguous severe-catastrophic runs
- `max_skipped_frames` is the recoverable replacement warning budget, not an automatic abort threshold

Diagnostics are written under:

- `standard/visualizations/geometry_consistency/summary.json`
- `standard/visualizations/geometry_consistency/pairwise_reprojection_rmse.png`
- `standard/visualizations/geometry_consistency/pairwise_reprojection_p90.png`
- `standard/visualizations/geometry_consistency/pairwise_reprojection_p95.png`
- `standard/visualizations/geometry_consistency/pairwise_inlier_ratio.png`
- `standard/visualizations/geometry_consistency/pairwise_static_overlap.png`
- `standard/visualizations/geometry_consistency/depth_scale_drift.png`
- `standard/visualizations/geometry_consistency/catastrophic_frames.json`

Current summaries record the requested masking mode, the masking source actually used, robust reprojection medians, recoverable-vs-severe catastrophic counts, both catastrophic pair candidates and the actual replaced frames chosen by the validator. When replacement is accepted, runtime injects that replacement information into provider context so later stages can reuse nearby valid frames.

## Geometry Validation

Configured under `runtime.settings.geometry_validation`.

This stage runs after geometry providers and post-processing finish.

It validates standardized outputs in `standard/` and fails fast on the first critical violation.

Current checks include:

- resource presence and expected frame counts
- intrinsics matrix validity
- pose matrix structure, inversion, and motion plausibility
- depth validity and reprojection consistency
- camera-height consistency
- road-plane and point-cloud plausibility when those resources exist
- optional canonical camera orientation checks

NuScenes-specific fail-fast validation now also happens at provider/frame-source setup time for sweep-capable runs:

- strictly increasing source timestamps after filtering
- valid resolved emitted FPS
- constant intrinsics within the selected camera stream
- constant calibrated-sensor translation within the selected camera stream

Diagnostics are written under:

- `standard/visualizations/geometry_validation/summary.json`
- `standard/visualizations/geometry_validation/pose_consistency.png`
- `standard/visualizations/geometry_validation/depth_reprojection_metrics.png`
- `standard/visualizations/geometry_validation/relative_motion.png`
- `standard/visualizations/geometry_validation/view_motion_alignment.png`
- `standard/visualizations/geometry_validation/reprojection/{frame:06d}.png`
- `standard/visualizations/geometry_validation/trajectory_points.ply`

Road-plane specific debug artifacts are written separately by the provider.

## Quality Metrics

Configured under `runtime.settings.quality_metrics`.

This module is optional and runs near the end of the pipeline.

Current metric groups:

- trajectory metrics
  - ATE
  - RPE
  - scale drift
- road metrics
  - residual percentile summaries
  - normal stability
  - smoothness windows
- artifact generation
  - reprojection heatmaps
  - temporal flicker views
  - point-cloud slices
  - road-model overlays
  - confidence overlays

Current output directory:

- `standard/visualizations/quality_metrics/`

If GT trajectory data is absent, trajectory metrics are skipped rather than fabricated.

## Fail-Fast Policy

PEMOIN prefers clear failures for invalid standardized geometry.

Best-effort behavior is limited to optional artifact-generation stages such as some videos or quality-metric visuals. Core geometry validation should not silently pass bad outputs.

Current low-FPS policy adds one explicit middle state for selected quality gates: degraded continuation. When `runtime.settings.validation_policy.enabled=true`, PEMOIN treats `10 FPS` as the reference cadence, relaxes selected geometry-quality thresholds below that cadence, records effective soft and hard thresholds in diagnostics, and continues on soft overruns with loud warnings. It still fails fast on invariant violations and on hard-threshold quality overruns.

Harmonisation diagnostics remain outside the standardized validation contract. Temporal harmonisation smoothing is evaluated through run-root diagnostics and visual comparison rather than a fail-fast standardized validator.

## Related Settings

Also relevant:

- `runtime.settings.comparison_frame`
- `runtime.settings.geometry_validation`
- `providers.geometry_fusion.settings`
