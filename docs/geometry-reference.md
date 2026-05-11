# Geometry Reference

## Purpose

This file documents the coordinate conventions and geometry rules PEMOIN expects.

Read this when changing adapters, comparison-frame canonicalization, road-plane logic, or validation.

## Image Coordinates

- `u`: pixel x, increasing to the right
- `v`: pixel y, increasing downward

## Camera Conventions

Current code centralizes camera projection/backprojection in `src/pemoin/geometry/camera_model.py`.

Key conventions:

- OpenCV camera convention
  - `x`: right
  - `y`: down
  - `z`: forward
- Blender camera convention
  - `x`: right
  - `y`: up
  - `z`: backward

Providers/adapters are responsible for converting native outputs into PEMOIN's standardized convention before persistence.

## Canonical Backprojection

For depth `d` and intrinsics `(fx, fy, cx, cy)`:

- `x = (u - cx) / fx * d`
- `y = s_y * (v - cy) / fy * d`
- `z = s_z * d`

Where:

- OpenCV uses `s_y = +1`, `s_z = +1`
- Blender uses `s_y = -1`, `s_z = -1`

Do not duplicate these equations across providers; use the shared geometry helpers.

## Plane Convention

PEMOIN uses the canonical plane equation:

- `n^T x + d = 0`
- `||n|| = 1`

Naming:

- `normal`: plane normal `n`
- `offset`: scalar `d`

Camera-to-plane anchor checks use:

- `n^T c + d`

where `c` is the camera center in world coordinates.

## Coordinate-System Conversion

Current conversion helpers live in `src/pemoin/coordinate_systems/conversions.py`, including:

- OpenCV to Blender
- CARLA to Blender
- Unity to Blender

Adapters should convert native pose conventions at the boundary instead of leaking native conventions into downstream stages.

## Runtime Canonicalization

PEMOIN uses a two-stage approach:

1. provider/adapters normalize native outputs into PEMOIN-compatible conventions
2. runtime applies scene-level comparison-frame canonicalization after geometry fusion

For active profiles, runtime now uses exactly two maintained comparison-frame modes:

- `gt`: trust the GT metric trajectory, align world up from support geometry, derive motion yaw fail-fast, and ground the support surface to `z=0`
- `estimated`: start from geometry-fusion metric outputs, derive a stable comparison frame, derive motion yaw fail-fast, and ground the support surface to `z=0`

## Camera Height And Scale

Monocular reconstruction is scale-ambiguous unless a metric cue is introduced.

In PEMOIN, camera height remains an important metric plausibility signal for:

- GT support-anchor validation after comparison-frame canonicalization
- estimated-path plausibility checks after geometry fusion

The important constraint is the distance between the camera and the local support surface, not a fixed global road elevation.

## Grounding

Grounding is part of the maintained comparison-frame stage. Active profiles control it through
`runtime.settings.comparison_frame`.

## Validation Expectations

Validation expects standardized geometry to be internally consistent after conversion and post-processing, including:

- valid pose matrices
- correct depth sign/orientation
- camera-height metadata consistency
- road-plane and point-cloud plausibility

See `validation.md`.

## Code References

- camera model: `src/pemoin/geometry/camera_model.py`
- plane helpers: `src/pemoin/geometry/plane.py`
- conversions: `src/pemoin/coordinate_systems/conversions.py`
- comparison-frame canonicalization: `src/pemoin/coordinate_systems/alignment.py`
