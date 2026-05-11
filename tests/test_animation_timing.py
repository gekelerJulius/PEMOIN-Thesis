from __future__ import annotations

import pytest

from pemoin.utils.animation_timing import (
    compute_clip_duration_seconds,
    compute_cycle_duration_seconds,
    compute_export_frame_count,
    compute_forward_speed_mps,
    compute_scene_cycle_length,
    map_output_time_to_source_frame_float,
    map_scene_frame_to_source_cycle_index,
    resolve_looped_source_timing,
)


def test_compute_scene_cycle_length_retimes_30fps_source_to_12fps_scene():
    assert compute_scene_cycle_length(30, 30.0, 12.0) == 12


def test_compute_scene_cycle_length_retimes_24fps_source_to_12fps_scene():
    assert compute_scene_cycle_length(24, 24.0, 12.0) == 12


def test_compute_scene_cycle_length_keeps_equal_fps_cycle():
    assert compute_scene_cycle_length(24, 12.0, 12.0) == 24


def test_compute_cycle_duration_seconds_uses_authored_rate():
    assert compute_cycle_duration_seconds(35, 30.0) == pytest.approx(35.0 / 30.0)


def test_compute_clip_duration_seconds_uses_frame_delta_at_scene_rate():
    assert compute_clip_duration_seconds(1, 13, 12.0) == pytest.approx(1.0)


def test_compute_export_frame_count_preserves_duration_at_export_rate():
    assert compute_export_frame_count(1.0, 30.0) == 31


def test_compute_scene_cycle_length_rejects_invalid_source_fps():
    with pytest.raises(ValueError, match="source_fps"):
        compute_scene_cycle_length(30, 0.0, 12.0)


def test_map_output_time_to_source_frame_float_preserves_continuous_phase():
    cycle_seconds = compute_cycle_duration_seconds(35, 30.0)
    mapped = [
        map_output_time_to_source_frame_float(
            t_seconds=idx * 0.5,
            cycle_duration_seconds=cycle_seconds,
            source_cycle_frames=35,
            source_start_frame=1.0,
        )
        for idx in range(5)
    ]
    assert mapped[0] == pytest.approx(1.0)
    assert mapped[1] == pytest.approx(16.0)
    assert mapped[2] == pytest.approx(31.0)
    assert mapped[3] == pytest.approx(11.0)
    assert mapped[4] == pytest.approx(26.0)


def test_resolve_looped_source_timing_tracks_unwrapped_cycle_state():
    cycle_seconds = compute_cycle_duration_seconds(35, 30.0)

    timing = [
        resolve_looped_source_timing(
            t_seconds=idx * 0.5,
            cycle_duration_seconds=cycle_seconds,
            source_cycle_frames=35,
            source_start_frame=1.0,
        )
        for idx in range(5)
    ]

    assert [sample.wrapped_source_frame_float for sample in timing] == pytest.approx(
        [1.0, 16.0, 31.0, 11.0, 26.0]
    )
    assert [sample.completed_cycles for sample in timing] == [0, 0, 0, 1, 1]
    assert [sample.absolute_source_progress_frames for sample in timing] == pytest.approx(
        [0.0, 15.0, 30.0, 45.0, 60.0]
    )


def test_resolve_looped_source_timing_keeps_exact_boundary_in_next_cycle():
    cycle_seconds = compute_cycle_duration_seconds(35, 30.0)
    timing = resolve_looped_source_timing(
        t_seconds=cycle_seconds,
        cycle_duration_seconds=cycle_seconds,
        source_cycle_frames=35,
        source_start_frame=1.0,
    )

    assert timing.wrapped_source_frame_float == pytest.approx(1.0)
    assert timing.completed_cycles == 1
    assert timing.absolute_source_progress_frames == pytest.approx(35.0)
    assert timing.cycle_phase == pytest.approx(0.0)


def test_compute_forward_speed_mps_uses_cycle_duration():
    speed = compute_forward_speed_mps(1.75, compute_cycle_duration_seconds(35, 30.0))
    assert speed == pytest.approx(1.5)


def test_map_scene_frame_to_source_cycle_index_wraps_scene_cycle():
    mapped = [map_scene_frame_to_source_cycle_index(t, 12, 30) for t in range(12)]
    assert mapped[0] == 0
    assert mapped[-1] == 28
    assert map_scene_frame_to_source_cycle_index(12, 12, 30) == 0
