"""Pure helpers for motion-relative pedestrian placement."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_MIXAMO_FORWARD_WORLD_XY = np.array([0.0, -1.0], dtype=np.float32)


def _validate_camera_to_world(camera_to_world: np.ndarray) -> np.ndarray:
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(
            "camera_to_world must have shape (N, 4, 4), "
            f"got {c2w.shape}."
        )
    if c2w.shape[0] == 0:
        raise ValueError("camera_to_world must contain at least one pose.")
    if not np.isfinite(c2w).all():
        raise ValueError("camera_to_world contains non-finite values.")
    return c2w


def _xy_positions(camera_to_world: np.ndarray) -> np.ndarray:
    c2w = _validate_camera_to_world(camera_to_world)
    return np.asarray(c2w[:, :2, 3], dtype=np.float32)


def _cumulative_xy_path_lengths(xy_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xy_positions.ndim != 2 or xy_positions.shape[1] != 2:
        raise ValueError(
            "xy_positions must have shape (N, 2), "
            f"got {xy_positions.shape}."
        )
    if xy_positions.shape[0] < 2:
        raise ValueError("Trajectory must contain at least 2 poses for motion-relative placement.")
    deltas = np.diff(xy_positions, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1).astype(np.float32)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(seg_lengths, dtype=np.float32)],
        axis=0,
    )
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 1e-3:
        raise ValueError(
            "Trajectory XY motion is too small to define a stable pedestrian placement anchor."
        )
    return seg_lengths, cumulative


def _validate_fraction(trajectory_t: float) -> float:
    t = float(trajectory_t)
    if not np.isfinite(t) or t < 0.0 or t > 1.0:
        raise ValueError(
            f"trajectory_t must be a finite float in [0, 1], got {trajectory_t!r}."
        )
    return t


def _normalize_xy_direction(direction: np.ndarray, *, context: str) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-3:
        raise ValueError(
            f"Trajectory motion is too weak or ambiguous to define a stable forward direction ({context})."
        )
    return (vec / norm).astype(np.float32)


def infer_horizontal_direction_xy(
    direction: Sequence[float],
    *,
    context: str,
) -> np.ndarray:
    vec = np.asarray(tuple(float(v) for v in direction), dtype=np.float32).reshape(-1)
    if vec.shape[0] < 2:
        raise ValueError(
            f"{context} must contain at least 2 values to infer a horizontal direction."
        )
    return _normalize_xy_direction(vec[:2], context=context)


def infer_locomotion_forward_local_xy(cycle_offset_local: Sequence[float]) -> np.ndarray:
    return infer_horizontal_direction_xy(
        cycle_offset_local,
        context="local locomotion forward",
    )


def standard_mixamo_forward_world_xy() -> np.ndarray:
    """Return PEMOIN's calibrated default Mixamo forward axis in world XY."""
    return np.array(DEFAULT_MIXAMO_FORWARD_WORLD_XY, dtype=np.float32, copy=True)


def detect_mixamo_animation_motion_category(
    animation_path: str | Path,
) -> str | None:
    parts = [part.lower() for part in Path(animation_path).parts]
    for idx in range(len(parts) - 3):
        if parts[idx : idx + 3] != ["assets", "mixamo", "animations"]:
            continue
        category = parts[idx + 3]
        if category in {"idle", "moving"}:
            return category
    return None


def resolve_mixamo_animation_motion_category(animation_path: str | Path) -> str:
    category = detect_mixamo_animation_motion_category(animation_path)
    if category is None:
        raise ValueError(
            "Mixamo animation path must live under "
            "'assets/mixamo/animations/idle/' or "
            "'assets/mixamo/animations/moving/': "
            f"{Path(animation_path)}"
        )
    return category


def resolve_mixamo_motion_policy_from_animation_path(
    animation_path: str | Path,
) -> str:
    category = resolve_mixamo_animation_motion_category(animation_path)
    return "stationary_at_spawn" if category == "idle" else "animation_root_motion"


def classify_locomotion_from_world_deltas(
    primary_direction_xy: Sequence[float] | None,
    sample_directions_xy: Sequence[Sequence[float]],
    *,
    displacement_threshold: float = 1e-3,
    delta_threshold: float = 1e-3,
) -> tuple[bool, dict[str, float | int]]:
    primary_norm = 0.0
    if primary_direction_xy is not None:
        raw_primary = np.asarray(
            tuple(float(v) for v in primary_direction_xy),
            dtype=np.float32,
        ).reshape(-1)
        if raw_primary.shape[0] >= 2 and np.isfinite(raw_primary[:2]).all():
            primary_norm = float(np.linalg.norm(np.asarray(raw_primary[:2], dtype=np.float32)))

    usable_delta_count = 0
    delta_path_length = 0.0
    for direction in sample_directions_xy:
        raw = np.asarray(tuple(float(v) for v in direction), dtype=np.float32).reshape(-1)
        if raw.shape[0] < 2 or not np.isfinite(raw[:2]).all():
            continue
        norm = float(np.linalg.norm(np.asarray(raw[:2], dtype=np.float32)))
        delta_path_length += norm
        if norm > float(delta_threshold):
            usable_delta_count += 1

    has_locomotion = bool(
        primary_norm > float(displacement_threshold)
        or usable_delta_count > 0
    )
    return has_locomotion, {
        "cycle_horizontal_norm": float(primary_norm),
        "usable_delta_count": int(usable_delta_count),
        "delta_path_length": float(delta_path_length),
    }


def resolve_dominant_horizontal_direction(
    primary_direction_xy: Sequence[float] | None,
    sample_directions_xy: Sequence[Sequence[float]],
    *,
    min_confidence: float = 0.5,
) -> tuple[np.ndarray, str, float]:
    primary_vec = None
    primary_norm = 0.0
    if primary_direction_xy is not None:
        raw_primary = np.asarray(
            tuple(float(v) for v in primary_direction_xy),
            dtype=np.float32,
        ).reshape(-1)
        if raw_primary.shape[0] >= 2 and np.isfinite(raw_primary[:2]).all():
            primary_vec = np.asarray(raw_primary[:2], dtype=np.float32)
            primary_norm = float(np.linalg.norm(primary_vec))
            if primary_norm > 1e-3:
                return (
                    _normalize_xy_direction(
                        primary_vec,
                        context="world cycle displacement",
                    ),
                    "world_cycle_displacement",
                    1.0,
                )

    rows: list[np.ndarray] = []
    for direction in sample_directions_xy:
        raw = np.asarray(tuple(float(v) for v in direction), dtype=np.float32).reshape(-1)
        if raw.shape[0] < 2 or not np.isfinite(raw[:2]).all():
            continue
        vec = np.asarray(raw[:2], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-3:
            continue
        rows.append(np.concatenate([vec, np.asarray([norm], dtype=np.float32)]))

    if not rows:
        if primary_vec is not None:
            raise ValueError(
                "Trajectory motion is too weak or ambiguous to define a stable forward "
                f"direction (primary horizontal norm={primary_norm:.6f}, usable_deltas=0)."
            )
        raise ValueError(
            "Trajectory motion is too weak or ambiguous to define a stable forward "
            "direction (no usable horizontal samples)."
        )

    stacked = np.stack(rows, axis=0)
    vecs = stacked[:, :2]
    norms = stacked[:, 2]
    units = vecs / norms[:, None]
    weighted_sum = np.sum(units * norms[:, None], axis=0)
    weight_total = float(np.sum(norms))
    resultant_norm = float(np.linalg.norm(weighted_sum))
    confidence = 0.0 if weight_total <= 1e-6 else resultant_norm / weight_total
    if confidence < float(min_confidence) or resultant_norm <= 1e-3:
        raise ValueError(
            "Trajectory motion is too weak or ambiguous to define a stable forward "
            f"direction (primary horizontal norm={primary_norm:.6f}, usable_deltas={len(rows)}, "
            f"fallback_confidence={confidence:.3f})."
        )
    return (
        _normalize_xy_direction(weighted_sum, context="world delta aggregate"),
        "world_delta_weighted_mean",
        float(confidence),
    )


def rotate_xy_direction(direction_xy: Sequence[float], yaw_deg: float) -> np.ndarray:
    direction = _normalize_xy_direction(
        np.asarray(tuple(float(v) for v in direction_xy), dtype=np.float32).reshape(2),
        context="direction rotation",
    )
    yaw_rad = math.radians(float(yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    rotated = np.array(
        [
            c * float(direction[0]) - s * float(direction[1]),
            s * float(direction[0]) + c * float(direction[1]),
        ],
        dtype=np.float32,
    )
    return _normalize_xy_direction(rotated, context="rotated direction")


def project_motion_progress_onto_axis(
    samples_xy: np.ndarray,
    axis_xy: Sequence[float],
    *,
    enforce_monotonic: bool = False,
) -> np.ndarray:
    """Project horizontal samples onto one locomotion axis.

    Returns progress relative to the first sample so callers can rebuild a stable
    heading-aligned world path from clip timing without preserving sideways drift.
    """
    samples = np.asarray(samples_xy, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] != 2:
        raise ValueError(
            "samples_xy must have shape (N, 2) with at least one sample, "
            f"got {samples.shape}."
        )
    axis = infer_horizontal_direction_xy(axis_xy, context="motion projection axis")
    origin = np.asarray(samples[0], dtype=np.float32)
    progress = np.sum((samples - origin.reshape(1, 2)) * axis.reshape(1, 2), axis=1)
    progress = np.asarray(progress, dtype=np.float32)
    if enforce_monotonic:
        progress = np.maximum.accumulate(progress).astype(np.float32)
    return progress


def build_heading_aligned_root_motion_path_world(
    progress_m: np.ndarray,
    spawn_world: Sequence[float],
    heading_world: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a constant-heading world path from scalar progress samples."""
    progress = np.asarray(progress_m, dtype=np.float32).reshape(-1)
    if progress.ndim != 1 or progress.shape[0] < 1:
        raise ValueError("progress_m must contain at least one sample.")
    if not np.isfinite(progress).all():
        raise ValueError("progress_m must contain only finite values.")
    spawn = np.asarray(tuple(float(v) for v in spawn_world), dtype=np.float32).reshape(3)
    heading = np.asarray(tuple(float(v) for v in heading_world), dtype=np.float32).reshape(-1)
    if heading.shape[0] < 2:
        raise ValueError("heading_world must contain at least 2 values.")
    heading_xy = _normalize_xy_direction(heading[:2], context="heading-aligned root motion")

    path_world = np.repeat(spawn.reshape(1, 3), progress.shape[0], axis=0)
    path_world[:, :2] = spawn[:2].reshape(1, 2) + progress.reshape(-1, 1) * heading_xy.reshape(1, 2)
    forward_world = np.repeat(
        np.asarray([heading_xy[0], heading_xy[1], 0.0], dtype=np.float32).reshape(1, 3),
        progress.shape[0],
        axis=0,
    )
    heading_deg = np.full(
        (progress.shape[0],),
        float(math.degrees(math.atan2(float(heading_xy[1]), float(heading_xy[0])))),
        dtype=np.float32,
    )
    return np.asarray(path_world, dtype=np.float32), forward_world, heading_deg


def resolve_actor_yaw_to_world_heading_deg(
    asset_facing_world_xy: Sequence[float],
    desired_heading_world: Sequence[float],
) -> float:
    """Resolve the root yaw needed to align an asset-facing basis to a world heading."""
    asset_xy = infer_horizontal_direction_xy(
        asset_facing_world_xy,
        context="asset world facing",
    )
    desired = np.asarray(tuple(float(v) for v in desired_heading_world), dtype=np.float32).reshape(-1)
    if desired.shape[0] < 2:
        raise ValueError(
            "desired_heading_world must contain at least 2 values to resolve actor yaw."
        )
    desired_xy = _normalize_xy_direction(desired[:2], context="desired world heading")
    asset_yaw = math.atan2(float(asset_xy[1]), float(asset_xy[0]))
    desired_yaw = math.atan2(float(desired_xy[1]), float(desired_xy[0]))
    return float(math.degrees(desired_yaw - asset_yaw))


def resolve_motion_aligned_actor_yaw_deg(
    asset_forward_world_xy: Sequence[float],
    intended_forward_world: Sequence[float],
    heading_offset_deg: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    asset_xy = infer_horizontal_direction_xy(
        asset_forward_world_xy,
        context="asset world forward",
    )
    intended = np.asarray(tuple(float(v) for v in intended_forward_world), dtype=np.float32).reshape(-1)
    if intended.shape[0] < 2:
        raise ValueError(
            "intended_forward_world must contain at least 2 values to resolve actor yaw."
        )
    intended_xy = _normalize_xy_direction(intended[:2], context="intended world forward")
    expected_heading_world_xy = rotate_xy_direction(intended_xy, float(heading_offset_deg))
    resolved_yaw_deg = resolve_actor_yaw_to_world_heading_deg(
        asset_xy,
        expected_heading_world_xy,
    )
    return float(resolved_yaw_deg), expected_heading_world_xy, asset_xy


def resolve_pedestrian_anchor_along_trajectory(
    camera_to_world: np.ndarray,
    trajectory_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a grounded anchor point and motion-forward vector along the trajectory."""
    xy = _xy_positions(camera_to_world)
    seg_lengths, cumulative = _cumulative_xy_path_lengths(xy)
    t = _validate_fraction(trajectory_t)
    target = float(cumulative[-1]) * t

    if target <= 0.0:
        anchor_xy = xy[0].astype(np.float32)
        forward_xy = _normalize_xy_direction(xy[1] - xy[0], context="start")
    elif target >= float(cumulative[-1]):
        anchor_xy = xy[-1].astype(np.float32)
        forward_xy = _normalize_xy_direction(xy[-1] - xy[-2], context="end")
    else:
        seg_idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        seg_idx = max(0, min(seg_idx, seg_lengths.shape[0] - 1))
        seg_len = float(seg_lengths[seg_idx])
        if seg_len <= 1e-6:
            raise ValueError(
                "Trajectory motion is too weak or ambiguous to define a stable forward direction "
                f"at trajectory_t={t:.3f}."
            )
        alpha = float((target - float(cumulative[seg_idx])) / seg_len)
        alpha = min(max(alpha, 0.0), 1.0)
        anchor_xy = (
            (1.0 - alpha) * xy[seg_idx] + alpha * xy[seg_idx + 1]
        ).astype(np.float32)
        if seg_idx > 0 and (seg_idx + 2) < xy.shape[0]:
            tangent = xy[seg_idx + 2] - xy[seg_idx]
            forward_xy = _normalize_xy_direction(tangent, context="interior")
        else:
            forward_xy = _normalize_xy_direction(
                xy[seg_idx + 1] - xy[seg_idx],
                context="boundary_segment",
            )

    anchor_world = np.array([float(anchor_xy[0]), float(anchor_xy[1]), 0.0], dtype=np.float32)
    forward_world = np.array([float(forward_xy[0]), float(forward_xy[1]), 0.0], dtype=np.float32)
    return anchor_world, forward_world


def resolve_pedestrian_spawn_world(
    camera_to_world: np.ndarray,
    trajectory_t: float,
    forward_offset_m: float,
    left_offset_m: float,
    up_offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Resolve a motion-relative spawn point from trajectory position and local offsets."""
    anchor_world, forward_world = resolve_pedestrian_anchor_along_trajectory(
        camera_to_world,
        trajectory_t,
    )
    offsets = np.array(
        [float(forward_offset_m), float(left_offset_m), float(up_offset_m)],
        dtype=np.float32,
    )
    if not np.isfinite(offsets).all():
        raise ValueError("Pedestrian offsets must be finite values.")
    left_world = np.array([-forward_world[1], forward_world[0], 0.0], dtype=np.float32)
    spawn_world = (
        anchor_world
        + offsets[0] * forward_world
        + offsets[1] * left_world
        + np.array([0.0, 0.0, offsets[2]], dtype=np.float32)
    ).astype(np.float32)
    base_yaw_deg = float(math.degrees(math.atan2(float(forward_world[1]), float(forward_world[0]))))
    return spawn_world, anchor_world, forward_world, base_yaw_deg


def resolve_unity_world_horizontal_placement(
    authoring_to_canonical_transform: Sequence[Sequence[float]],
    *,
    position_x_m: float,
    position_z_m: float,
    heading_yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    """Resolve Unity-authored horizontal placement into PEMOIN's canonical world.

    Unity-facing authored inputs use world ``X`` / ``Z`` on the ground plane and a
    yaw around Unity ``+Y`` (with ``0 deg`` meaning facing Unity ``+Z``).
    """
    transform = np.asarray(authoring_to_canonical_transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(
            "authoring_to_canonical_transform must have shape (4, 4), "
            f"got {transform.shape}."
        )
    authored = np.asarray(
        [float(position_x_m), 0.0, float(position_z_m), 1.0],
        dtype=np.float32,
    )
    if not np.isfinite(authored).all():
        raise ValueError("Unity-authored pedestrian position must be finite.")
    resolved_h = transform @ authored
    if not np.isfinite(resolved_h).all():
        raise ValueError("Resolved canonical pedestrian position is non-finite.")
    spawn_world = np.asarray(resolved_h[:3], dtype=np.float32)

    yaw_rad = math.radians(float(heading_yaw_deg))
    authored_forward = np.asarray(
        [math.sin(yaw_rad), 0.0, math.cos(yaw_rad), 0.0],
        dtype=np.float32,
    )
    transformed_forward = (transform @ authored_forward)[:3]
    forward_xy = _normalize_xy_direction(
        np.asarray(transformed_forward[:2], dtype=np.float32),
        context="unity-authored canonical forward",
    )
    forward_world = np.asarray(
        [float(forward_xy[0]), float(forward_xy[1]), 0.0],
        dtype=np.float32,
    )
    heading_world_deg = float(
        math.degrees(math.atan2(float(forward_world[1]), float(forward_world[0])))
    )
    diagnostics: dict[str, object] = {
        "authoring_mode": "unity_world_horizontal",
        "authored_position_unity": [float(position_x_m), 0.0, float(position_z_m)],
        "authored_heading_yaw_deg": float(heading_yaw_deg),
        "resolved_spawn_world": spawn_world.astype(float).tolist(),
        "resolved_forward_world": forward_world.astype(float).tolist(),
        "resolved_heading_world_deg": float(heading_world_deg),
    }
    return spawn_world, forward_world, heading_world_deg, diagnostics


def sample_pedestrian_spawn_path_world(
    camera_to_world: np.ndarray,
    trajectory_t: float,
    forward_offset_m: float,
    left_offset_m: float,
    up_offset_m: float,
    *,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a trajectory-relative pedestrian path from the insertion anchor to clip end.

    The path starts at ``trajectory_t`` and linearly advances to the end of the
    standardized trajectory over ``sample_count`` samples, preserving the configured
    forward/left/up offsets in the local trajectory frame at each sample.
    """
    count = int(sample_count)
    if count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count!r}.")
    start_t = _validate_fraction(trajectory_t)
    if count == 1:
        sample_ts = np.asarray([start_t], dtype=np.float32)
    else:
        sample_ts = np.linspace(start_t, 1.0, num=count, dtype=np.float32)

    spawn_world_samples: list[np.ndarray] = []
    forward_world_samples: list[np.ndarray] = []
    base_heading_deg_samples: list[float] = []
    for sample_t in sample_ts.tolist():
        spawn_world, _anchor_world, forward_world, base_heading_deg = (
            resolve_pedestrian_spawn_world(
                camera_to_world,
                float(sample_t),
                forward_offset_m,
                left_offset_m,
                up_offset_m,
            )
        )
        spawn_world_samples.append(np.asarray(spawn_world, dtype=np.float32))
        forward_world_samples.append(np.asarray(forward_world, dtype=np.float32))
        base_heading_deg_samples.append(float(base_heading_deg))

    return (
        np.stack(spawn_world_samples, axis=0),
        np.stack(forward_world_samples, axis=0),
        np.asarray(base_heading_deg_samples, dtype=np.float32),
    )


def build_animation_root_motion_path_world(
    local_root_motion: np.ndarray,
    spawn_world: Sequence[float],
    *,
    root_yaw_deg: float,
) -> np.ndarray:
    """Transform local animation root motion into a world-space actor path.

    The returned path preserves the resolved spawn height. Grounding owns world
    Z later, so this helper transfers only horizontal authored locomotion.
    """
    local_motion = np.asarray(local_root_motion, dtype=np.float32)
    if local_motion.ndim != 2 or local_motion.shape[0] < 1 or local_motion.shape[1] < 2:
        raise ValueError(
            "local_root_motion must have shape (N, 2+) with at least one sample, "
            f"got {local_motion.shape}."
        )
    if not np.isfinite(local_motion).all():
        raise ValueError("local_root_motion must contain only finite values.")

    spawn = np.asarray(tuple(float(v) for v in spawn_world), dtype=np.float32).reshape(3)
    yaw_rad = math.radians(float(root_yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    world_xy = local_motion[:, :2] @ rot.T

    path_world = np.repeat(spawn.reshape(1, 3), local_motion.shape[0], axis=0)
    path_world[:, :2] = spawn[:2].reshape(1, 2) + world_xy
    return np.asarray(path_world, dtype=np.float32)


def derive_motion_path_heading_world(
    path_world: np.ndarray,
    fallback_forward_world: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive per-frame forward vectors and headings from a world-space path."""
    path = np.asarray(path_world, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != 3:
        raise ValueError(
            "path_world must have shape (N, 3) with at least one sample, "
            f"got {path.shape}."
        )
    fallback = np.asarray(
        tuple(float(v) for v in fallback_forward_world),
        dtype=np.float32,
    ).reshape(-1)
    if fallback.shape[0] < 2:
        raise ValueError(
            "fallback_forward_world must contain at least 2 values, "
            f"got shape {fallback.shape}."
        )
    fallback_xy = _normalize_xy_direction(fallback[:2], context="fallback world forward")

    count = path.shape[0]
    forward_world = np.zeros((count, 3), dtype=np.float32)
    heading_deg = np.zeros((count,), dtype=np.float32)
    deltas = np.diff(path[:, :2], axis=0) if count > 1 else np.zeros((0, 2), dtype=np.float32)

    for idx in range(count):
        candidate = None
        if idx < deltas.shape[0]:
            candidate = deltas[idx]
        if candidate is None or float(np.linalg.norm(candidate)) <= 1e-6:
            if idx > 0:
                candidate = deltas[idx - 1]
        if candidate is None or float(np.linalg.norm(candidate)) <= 1e-6:
            candidate_xy = fallback_xy
        else:
            candidate_xy = _normalize_xy_direction(candidate, context="motion path delta")
        forward_world[idx, :2] = candidate_xy
        heading_deg[idx] = float(
            math.degrees(math.atan2(float(candidate_xy[1]), float(candidate_xy[0])))
        )

    return forward_world, heading_deg


def stationary_pedestrian_spawn_path_world(
    spawn_world: Sequence[float],
    forward_world: Sequence[float],
    *,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(sample_count)
    if count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count!r}.")
    spawn = np.asarray(tuple(float(v) for v in spawn_world), dtype=np.float32).reshape(3)
    forward = np.asarray(tuple(float(v) for v in forward_world), dtype=np.float32).reshape(3)
    base_yaw_deg = float(math.degrees(math.atan2(float(forward[1]), float(forward[0]))))
    return (
        np.repeat(spawn.reshape(1, 3), count, axis=0),
        np.repeat(forward.reshape(1, 3), count, axis=0),
        np.full((count,), base_yaw_deg, dtype=np.float32),
    )


def minimum_xy_distance_to_trajectory(
    camera_to_world: np.ndarray,
    world_point: Sequence[float],
) -> float:
    """Return the minimum horizontal distance from a world point to the trajectory."""
    xy = _xy_positions(camera_to_world)
    point = np.asarray(tuple(float(v) for v in world_point), dtype=np.float32)
    if point.shape != (3,):
        raise ValueError(
            "world_point must contain exactly 3 values, "
            f"got {point.shape}."
        )
    deltas = xy - point[:2].reshape(1, 2)
    return float(np.min(np.linalg.norm(deltas, axis=1)))


def validate_pedestrian_spawn_near_trajectory(
    camera_to_world: np.ndarray,
    world_point: Sequence[float],
    *,
    max_distance_m: float,
) -> float:
    """Fail fast if the resolved pedestrian spawn is implausibly far away."""
    threshold = float(max_distance_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"max_distance_m must be finite and > 0, got {max_distance_m!r}.")
    point = np.asarray(tuple(float(v) for v in world_point), dtype=np.float32)
    min_distance = minimum_xy_distance_to_trajectory(camera_to_world, point)
    if min_distance > threshold:
        raise ValueError(
            "Resolved pedestrian spawn is too far from the standardized trajectory corridor: "
            f"spawn_world=({float(point[0]):.3f}, {float(point[1]):.3f}, {float(point[2]):.3f}) "
            f"min_xy_distance={min_distance:.3f}m > threshold={threshold:.3f}m."
        )
    return min_distance
