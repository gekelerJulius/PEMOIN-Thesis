# Design Roadmap

## Status

This file is intentionally non-canonical.

It records future ideas that are still worth discussing, but it does not override current code behavior, profile settings, or data contracts.

## Keep: Local Road Support Estimator With Feedback

Current state:

- the road-plane provider produces global road-plane outputs
- Blender pedestrian insertion already performs local support fitting for grounding
- there is no standardized local-support resource or provider-level local-to-global feedback loop

Potential future change:

- promote local support from a Blender-only behavior into an optional pipeline concept
- estimate local support patches near insertion/contact regions
- optionally feed high-confidence local support back into the global road estimate

Potential gain:

- better foot grounding on local irregularities
- better handling of places where one global plane is too coarse

Main risks:

- more contracts/resources to maintain
- more gating and validation complexity
- more ways for sparse/noisy local evidence to destabilize global behavior

## Keep: Expanded Quality-Metrics Roadmap

Current state:

- PEMOIN already computes trajectory metrics, road metrics, and artifact outputs through `runtime.settings.quality_metrics`
- validation and metric coverage are useful, but still narrower than the older exploratory specs proposed

Potential future change:

- expand the evaluation surface in a controlled way
- add additional GT-backed or self-consistency metrics only when they justify ongoing maintenance
- grow the artifact set deliberately instead of scattering one-off diagnostics across the repo

Potential gain:

- clearer confidence in geometry quality
- easier comparison between profiles/providers/runs

Main risks:

- metrics can become noise if GT or stable baselines are unavailable
- artifact generation can bloat runtime and docs if not kept curated

## Explicitly Dropped Ideas

The following ideas were intentionally not retained in repo docs:

- road-aligned metric grid surface as a documented future direction
- targeted DepthAnything3 refinement on hard windows

If either becomes an active workstream later, reintroduce it here only after there is a concrete implementation direction.
