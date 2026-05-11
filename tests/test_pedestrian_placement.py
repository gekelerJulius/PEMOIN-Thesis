from __future__ import annotations

import numpy as np
import pytest

from pemoin.utils.animation_timing import compute_cycle_duration_seconds, resolve_looped_source_timing
from pemoin.visualization.pedestrian_placement import (
    build_animation_root_motion_path_world,
    build_heading_aligned_root_motion_path_world,
    classify_locomotion_from_world_deltas,
    derive_motion_path_heading_world,
    detect_mixamo_animation_motion_category,
    infer_locomotion_forward_local_xy,
    minimum_xy_distance_to_trajectory,
    project_motion_progress_onto_axis,
    resolve_actor_yaw_to_world_heading_deg,
    resolve_dominant_horizontal_direction,
    resolve_mixamo_animation_motion_category,
    resolve_mixamo_motion_policy_from_animation_path,
    resolve_motion_aligned_actor_yaw_deg,
    resolve_pedestrian_anchor_along_trajectory,
    resolve_pedestrian_spawn_world,
    resolve_unity_world_horizontal_placement,
    sample_pedestrian_spawn_path_world,
    stationary_pedestrian_spawn_path_world,
    standard_mixamo_forward_world_xy,
    rotate_xy_direction,
    validate_pedestrian_spawn_near_trajectory,
)


def _trajectory(*positions: tuple[float, float, float]) -> np.ndarray:
    mats = []
    for pos in positions:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.asarray(pos, dtype=np.float32)
        mats.append(c2w)
    return np.stack(mats, axis=0)


def test_resolve_pedestrian_anchor_along_trajectory_uses_arc_length() -> None:
    c2w = _trajectory((0.0, 0.0, 1.5), (1.0, 0.0, 1.5), (5.0, 0.0, 1.5))

    anchor_world, forward_world = resolve_pedestrian_anchor_along_trajectory(c2w, 0.5)

    np.testing.assert_allclose(anchor_world, np.array([2.5, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(forward_world, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_resolve_pedestrian_spawn_world_uses_motion_relative_offsets() -> None:
    c2w = _trajectory((0.0, 0.0, 1.5), (0.0, 4.0, 1.5), (0.0, 8.0, 1.5))

    spawn_world, anchor_world, forward_world, base_yaw_deg = resolve_pedestrian_spawn_world(
        c2w,
        0.0,
        5.0,
        2.0,
        0.5,
    )

    np.testing.assert_allclose(anchor_world, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(forward_world, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(spawn_world, np.array([-2.0, 5.0, 0.5], dtype=np.float32))
    assert abs(base_yaw_deg - 90.0) < 1e-4


def test_resolve_pedestrian_spawn_world_respects_nonzero_trajectory_fraction() -> None:
    c2w = _trajectory(
        (0.0, 0.0, 1.5),
        (10.0, 0.0, 1.5),
        (20.0, 0.0, 1.5),
        (30.0, 0.0, 1.5),
    )

    spawn_world, anchor_world, forward_world, base_yaw_deg = resolve_pedestrian_spawn_world(
        c2w,
        0.5,
        0.0,
        2.0,
        0.0,
    )

    np.testing.assert_allclose(anchor_world, np.array([15.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(forward_world, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(spawn_world, np.array([15.0, 2.0, 0.0], dtype=np.float32))
    assert abs(base_yaw_deg - 0.0) < 1e-4


def test_resolve_unity_world_horizontal_placement_maps_unity_axes_into_canonical_world() -> None:
    transform = np.array(
        [
            [0.0, 0.0, -1.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    spawn_world, forward_world, heading_world_deg, diagnostics = (
        resolve_unity_world_horizontal_placement(
            transform,
            position_x_m=3.0,
            position_z_m=4.0,
            heading_yaw_deg=90.0,
        )
    )

    np.testing.assert_allclose(
        spawn_world,
        np.array([6.0, 23.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forward_world,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    assert heading_world_deg == pytest.approx(90.0)
    assert diagnostics["authoring_mode"] == "unity_world_horizontal"


def test_standard_mixamo_forward_world_xy_matches_calibrated_default() -> None:
    np.testing.assert_allclose(
        standard_mixamo_forward_world_xy(),
        np.array([0.0, -1.0], dtype=np.float32),
    )


def test_detect_mixamo_animation_motion_category_from_idle_path() -> None:
    assert (
        detect_mixamo_animation_motion_category(
            "/repo/assets/mixamo/animations/idle/waving.fbx"
        )
        == "idle"
    )


def test_resolve_mixamo_motion_policy_from_animation_path_uses_category() -> None:
    assert (
        resolve_mixamo_motion_policy_from_animation_path(
            "/repo/assets/mixamo/animations/moving/walk.fbx"
        )
        == "animation_root_motion"
    )
    assert (
        resolve_mixamo_motion_policy_from_animation_path(
            "/repo/assets/mixamo/animations/idle/waving.fbx"
        )
        == "stationary_at_spawn"
    )


def test_resolve_mixamo_animation_motion_category_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="assets/mixamo/animations/idle/"):
        resolve_mixamo_animation_motion_category("/repo/assets/mixamo/waving.fbx")


def test_sample_pedestrian_spawn_path_world_advances_from_insertion_fraction_to_end() -> None:
    c2w = _trajectory(
        (0.0, 0.0, 1.5),
        (0.0, 4.0, 1.5),
        (0.0, 8.0, 1.5),
        (0.0, 12.0, 1.5),
        (0.0, 16.0, 1.5),
    )

    spawn_world, forward_world, heading_deg = sample_pedestrian_spawn_path_world(
        c2w,
        0.5,
        5.0,
        2.0,
        0.5,
        sample_count=3,
    )

    np.testing.assert_allclose(
        spawn_world,
        np.array(
            [
                [-2.0, 13.0, 0.5],
                [-2.0, 17.0, 0.5],
                [-2.0, 21.0, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        forward_world,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(heading_deg, np.array([90.0, 90.0, 90.0], dtype=np.float32))


def test_stationary_pedestrian_spawn_path_world_repeats_spawn_and_heading() -> None:
    spawn_world, forward_world, heading_deg = stationary_pedestrian_spawn_path_world(
        (1.0, 2.0, 0.5),
        (0.0, 1.0, 0.0),
        sample_count=3,
    )

    np.testing.assert_allclose(
        spawn_world,
        np.array([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forward_world,
        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(heading_deg, np.array([90.0, 90.0, 90.0], dtype=np.float32))


def test_heading_aligned_root_motion_path_stays_monotonic_across_loop_boundaries() -> None:
    cycle_seconds = compute_cycle_duration_seconds(35, 30.0)
    timing = [
        resolve_looped_source_timing(
            t_seconds=idx * 0.5,
            cycle_duration_seconds=cycle_seconds,
            source_cycle_frames=35,
        )
        for idx in range(5)
    ]
    cycle_distance_m = 1.75
    in_cycle_progress_m = np.array([0.0, 0.75, 1.5, 0.5, 1.25], dtype=np.float32)
    progress_m = np.array(
        [
            sample.completed_cycles * cycle_distance_m + float(in_cycle_progress_m[idx])
            for idx, sample in enumerate(timing)
        ],
        dtype=np.float32,
    )

    path_world, _, _ = build_heading_aligned_root_motion_path_world(
        progress_m,
        (10.0, 20.0, 0.5),
        (0.0, 1.0, 0.0),
    )

    np.testing.assert_allclose(progress_m, np.array([0.0, 0.75, 1.5, 2.25, 3.0], dtype=np.float32))
    assert np.all(np.diff(progress_m) > 0.0)
    np.testing.assert_allclose(path_world[:, 1], np.array([20.0, 20.75, 21.5, 22.25, 23.0], dtype=np.float32))


def test_build_animation_root_motion_path_world_rotates_local_xy_from_spawn() -> None:
    path_world = build_animation_root_motion_path_world(
        np.array(
            [
                [0.0, 0.0],
                [0.0, -1.0],
                [0.0, -2.0],
            ],
            dtype=np.float32,
        ),
        (10.0, 20.0, 0.5),
        root_yaw_deg=90.0,
    )

    np.testing.assert_allclose(
        path_world,
        np.array(
            [
                [10.0, 20.0, 0.5],
                [11.0, 20.0, 0.5],
                [12.0, 20.0, 0.5],
            ],
            dtype=np.float32,
        ),
    )


def test_derive_motion_path_heading_world_uses_fallback_for_stationary_segments() -> None:
    forward_world, heading_deg = derive_motion_path_heading_world(
        np.array(
            [
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5],
                [1.0, 5.0, 0.5],
            ],
            dtype=np.float32,
        ),
        (0.0, 1.0, 0.0),
    )

    np.testing.assert_allclose(
        forward_world,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(heading_deg, np.array([90.0, 90.0, 90.0], dtype=np.float32))


def test_classify_locomotion_from_world_deltas_detects_idle() -> None:
    has_locomotion, diagnostics = classify_locomotion_from_world_deltas(
        (0.0, 0.0),
        [(0.0, 0.0), (0.0, 0.0)],
    )

    assert has_locomotion is False
    assert diagnostics["usable_delta_count"] == 0


def test_classify_locomotion_from_world_deltas_detects_motion() -> None:
    has_locomotion, diagnostics = classify_locomotion_from_world_deltas(
        (0.0, -2.0),
        [(0.0, -1.0), (0.0, -1.0)],
    )

    assert has_locomotion is True
    assert diagnostics["cycle_horizontal_norm"] == pytest.approx(2.0)


def test_validate_pedestrian_spawn_near_trajectory_rejects_distant_spawn() -> None:
    c2w = _trajectory((0.0, 0.0, 1.5), (0.0, 4.0, 1.5), (0.0, 8.0, 1.5))
    spawn = np.array([30.0, 30.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="too far from the standardized trajectory corridor"):
        validate_pedestrian_spawn_near_trajectory(
            c2w,
            spawn,
            max_distance_m=10.0,
        )

    assert minimum_xy_distance_to_trajectory(c2w, spawn) > 10.0


def test_resolve_motion_aligned_actor_yaw_aligns_asset_forward_to_motion() -> None:
    resolved_yaw_deg, expected_heading_world_xy, asset_forward_world_xy = (
        resolve_motion_aligned_actor_yaw_deg(
            asset_forward_world_xy=(0.0, 1.0),
            intended_forward_world=(1.0, 0.0, 0.0),
            heading_offset_deg=0.0,
        )
    )

    assert resolved_yaw_deg == pytest.approx(-90.0)
    np.testing.assert_allclose(asset_forward_world_xy, np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(
        expected_heading_world_xy,
        np.array([1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_resolve_motion_aligned_actor_yaw_applies_heading_offset_after_alignment() -> None:
    resolved_yaw_deg, expected_heading_world_xy, _ = resolve_motion_aligned_actor_yaw_deg(
        asset_forward_world_xy=(0.0, 1.0),
        intended_forward_world=(1.0, 0.0, 0.0),
        heading_offset_deg=90.0,
    )

    assert resolved_yaw_deg == pytest.approx(0.0)
    np.testing.assert_allclose(
        expected_heading_world_xy,
        np.array([0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_resolve_actor_yaw_to_world_heading_deg_aligns_measured_body_facing() -> None:
    resolved_yaw_deg = resolve_actor_yaw_to_world_heading_deg(
        asset_facing_world_xy=(0.0, -1.0),
        desired_heading_world=(-1.0, 0.0, 0.0),
    )

    assert resolved_yaw_deg == pytest.approx(270.0)


def test_resolve_actor_yaw_to_world_heading_deg_rejects_degenerate_vectors() -> None:
    with pytest.raises(ValueError, match="asset world facing"):
        resolve_actor_yaw_to_world_heading_deg(
            asset_facing_world_xy=(0.0, 0.0),
            desired_heading_world=(1.0, 0.0, 0.0),
        )


def test_infer_locomotion_forward_local_xy_rejects_ambiguous_motion() -> None:
    with pytest.raises(ValueError, match="too weak or ambiguous"):
        infer_locomotion_forward_local_xy((0.0, 0.0, 0.0))


def test_resolve_dominant_horizontal_direction_uses_primary_world_displacement() -> None:
    direction_xy, method, confidence = resolve_dominant_horizontal_direction(
        primary_direction_xy=(0.0, -2.0),
        sample_directions_xy=[(0.0, -1.0), (0.0, -1.0)],
    )

    np.testing.assert_allclose(direction_xy, np.array([0.0, -1.0], dtype=np.float32))
    assert method == "world_cycle_displacement"
    assert confidence == pytest.approx(1.0)


def test_resolve_dominant_horizontal_direction_falls_back_to_weighted_mean() -> None:
    direction_xy, method, confidence = resolve_dominant_horizontal_direction(
        primary_direction_xy=(0.0, 0.0),
        sample_directions_xy=[(0.0, -0.5), (0.0, -0.7), (0.0, -0.4)],
    )

    np.testing.assert_allclose(direction_xy, np.array([0.0, -1.0], dtype=np.float32))
    assert method == "world_delta_weighted_mean"
    assert confidence == pytest.approx(1.0)


def test_resolve_dominant_horizontal_direction_rejects_conflicting_fallback() -> None:
    with pytest.raises(ValueError, match="fallback_confidence"):
        resolve_dominant_horizontal_direction(
            primary_direction_xy=(0.0, 0.0),
            sample_directions_xy=[(1.0, 0.0), (-1.0, 0.0)],
        )


def test_rotate_xy_direction_rotates_counterclockwise() -> None:
    rotated = rotate_xy_direction((1.0, 0.0), 90.0)
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)


def test_project_motion_progress_onto_axis_returns_zero_based_monotonic_progress() -> None:
    progress = project_motion_progress_onto_axis(
        np.array(
            [
                [4.0, 2.0],
                [5.0, 3.0],
                [6.0, 2.0],
                [8.0, 1.0],
            ],
            dtype=np.float32,
        ),
        axis_xy=(1.0, 0.0),
        enforce_monotonic=True,
    )

    np.testing.assert_allclose(
        progress,
        np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
    )


def test_build_heading_aligned_root_motion_path_world_locks_walk_to_heading() -> None:
    path_world, forward_world, heading_deg = build_heading_aligned_root_motion_path_world(
        np.array([0.0, 1.5, 3.0], dtype=np.float32),
        spawn_world=(10.0, 20.0, 0.5),
        heading_world=(0.0, 1.0, 0.0),
    )

    np.testing.assert_allclose(
        path_world,
        np.array(
            [
                [10.0, 20.0, 0.5],
                [10.0, 21.5, 0.5],
                [10.0, 23.0, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        forward_world,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(heading_deg, np.array([90.0, 90.0, 90.0], dtype=np.float32))


def test_resolve_pedestrian_anchor_along_trajectory_rejects_weak_motion() -> None:
    c2w = _trajectory((0.0, 0.0, 1.5), (0.0, 0.0, 1.5))

    with pytest.raises(ValueError, match="too small"):
        resolve_pedestrian_anchor_along_trajectory(c2w, 0.5)
