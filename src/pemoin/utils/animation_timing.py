"""Pure timing helpers for authored animation timing and retiming."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopedSourceTiming:
    wrapped_source_frame_float: float
    absolute_source_progress_frames: float
    completed_cycles: int
    cycle_phase: float


def compute_cycle_duration_seconds(
    source_cycle_frames: int,
    source_fps: float,
) -> float:
    if int(source_cycle_frames) < 2:
        raise ValueError("source_cycle_frames must be >= 2.")
    if not math.isfinite(float(source_fps)) or float(source_fps) <= 0.0:
        raise ValueError("source_fps must be finite and > 0.")
    return float(source_cycle_frames) / float(source_fps)


def compute_scene_cycle_length(
    source_cycle_frames: int,
    source_fps: float,
    scene_fps: float,
) -> int:
    if int(source_cycle_frames) < 2:
        raise ValueError("source_cycle_frames must be >= 2.")
    if not math.isfinite(float(source_fps)) or float(source_fps) <= 0.0:
        raise ValueError("source_fps must be finite and > 0.")
    if not math.isfinite(float(scene_fps)) or float(scene_fps) <= 0.0:
        raise ValueError("scene_fps must be finite and > 0.")
    scene_cycle = int(round(float(source_cycle_frames) * float(scene_fps) / float(source_fps)))
    if scene_cycle < 2:
        raise ValueError(
            "Computed scene cycle length is invalid; check source_fps/scene_fps/source_cycle_frames."
        )
    return scene_cycle


def compute_clip_duration_seconds(
    frame_start: float,
    frame_end: float,
    fps: float,
) -> float:
    if not math.isfinite(float(frame_start)) or not math.isfinite(float(frame_end)):
        raise ValueError("frame_start/frame_end must be finite.")
    if float(frame_end) <= float(frame_start):
        raise ValueError("frame_end must be greater than frame_start.")
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps must be finite and > 0.")
    return (float(frame_end) - float(frame_start)) / float(fps)


def compute_export_frame_count(
    clip_duration_seconds: float,
    export_fps: float,
) -> int:
    if (
        not math.isfinite(float(clip_duration_seconds))
        or float(clip_duration_seconds) <= 0.0
    ):
        raise ValueError("clip_duration_seconds must be finite and > 0.")
    if not math.isfinite(float(export_fps)) or float(export_fps) <= 0.0:
        raise ValueError("export_fps must be finite and > 0.")
    return max(2, int(round(float(clip_duration_seconds) * float(export_fps))) + 1)


def map_output_time_to_source_frame_float(
    t_seconds: float,
    cycle_duration_seconds: float,
    source_cycle_frames: int,
    *,
    source_start_frame: float = 0.0,
) -> float:
    return resolve_looped_source_timing(
        t_seconds,
        cycle_duration_seconds,
        source_cycle_frames,
        source_start_frame=source_start_frame,
    ).wrapped_source_frame_float


def resolve_looped_source_timing(
    t_seconds: float,
    cycle_duration_seconds: float,
    source_cycle_frames: int,
    *,
    source_start_frame: float = 0.0,
) -> LoopedSourceTiming:
    if not math.isfinite(float(t_seconds)) or float(t_seconds) < 0.0:
        raise ValueError("t_seconds must be finite and >= 0.")
    if not math.isfinite(float(cycle_duration_seconds)) or float(cycle_duration_seconds) <= 0.0:
        raise ValueError("cycle_duration_seconds must be finite and > 0.")
    if int(source_cycle_frames) < 2:
        raise ValueError("source_cycle_frames must be >= 2.")
    cycle_count_float = float(t_seconds) / float(cycle_duration_seconds)
    completed_cycles = int(math.floor(cycle_count_float + 1e-9))
    absolute_source_progress_frames = cycle_count_float * float(source_cycle_frames)
    cycle_phase = cycle_count_float - float(completed_cycles)
    if cycle_phase < 0.0:
        cycle_phase += 1.0
    if cycle_phase >= 1.0:
        cycle_phase -= 1.0
    wrapped_source_frame_float = (
        float(source_start_frame) + cycle_phase * float(source_cycle_frames)
    )
    return LoopedSourceTiming(
        wrapped_source_frame_float=float(wrapped_source_frame_float),
        absolute_source_progress_frames=float(absolute_source_progress_frames),
        completed_cycles=int(completed_cycles),
        cycle_phase=float(cycle_phase),
    )


def compute_forward_speed_mps(
    cycle_distance_m: float,
    cycle_duration_seconds: float,
) -> float:
    if not math.isfinite(float(cycle_distance_m)) or float(cycle_distance_m) < 0.0:
        raise ValueError("cycle_distance_m must be finite and >= 0.")
    if not math.isfinite(float(cycle_duration_seconds)) or float(cycle_duration_seconds) <= 0.0:
        raise ValueError("cycle_duration_seconds must be finite and > 0.")
    return float(cycle_distance_m) / float(cycle_duration_seconds)


def map_scene_frame_to_source_cycle_index(
    t_scene: int,
    cycle_len_scene: int,
    cycle_len_source: int,
) -> int:
    if int(cycle_len_scene) < 2:
        raise ValueError("cycle_len_scene must be >= 2.")
    if int(cycle_len_source) < 2:
        raise ValueError("cycle_len_source must be >= 2.")
    t_scene_i = int(t_scene)
    phase = (t_scene_i % int(cycle_len_scene)) / float(cycle_len_scene)
    source_idx = int(round(phase * float(cycle_len_source)))
    return max(0, min(int(cycle_len_source) - 1, source_idx))
