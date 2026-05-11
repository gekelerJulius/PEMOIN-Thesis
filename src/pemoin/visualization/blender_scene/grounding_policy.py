from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportRelockMetrics:
    normal_jump_deg: float | None
    support_height_jump_m: float | None
    anchor_shift_m: float | None
    current_signed_distance_m: float | None
    previous_signed_distance_m: float | None


def resolve_effective_hold_frames(
    *,
    hold_frames: int,
    hold_seconds: float | None,
    sampling_fps: float,
) -> tuple[int, float | None]:
    if hold_seconds is None:
        return int(hold_frames), None
    seconds = float(hold_seconds)
    if not np.isfinite(seconds) or seconds < 0.0:
        raise ValueError(
            "local_support_temporal_hold_seconds must be a finite value >= 0."
        )
    fps = float(sampling_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("sampling_fps must be finite and > 0 for time-based hold.")
    effective_frames = int(math.ceil(seconds * fps))
    return effective_frames, seconds


def compute_support_relock_metrics(
    *,
    current_normal: np.ndarray,
    current_offset: float,
    previous_normal: np.ndarray | None,
    previous_offset: float | None,
    comparison_anchor: np.ndarray,
    current_anchor: np.ndarray,
    previous_anchor: np.ndarray | None,
) -> SupportRelockMetrics:
    if previous_normal is None or previous_offset is None or previous_anchor is None:
        return SupportRelockMetrics(
            normal_jump_deg=None,
            support_height_jump_m=None,
            anchor_shift_m=None,
            current_signed_distance_m=None,
            previous_signed_distance_m=None,
        )
    current_n = np.asarray(current_normal, dtype=np.float32)
    previous_n = np.asarray(previous_normal, dtype=np.float32)
    compare_anchor = np.asarray(comparison_anchor, dtype=np.float32)
    dot = float(np.clip(np.dot(current_n, previous_n), -1.0, 1.0))
    normal_jump = float(np.degrees(math.acos(dot)))
    current_signed = float(np.dot(current_n, compare_anchor) + float(current_offset))
    previous_signed = float(np.dot(previous_n, compare_anchor) + float(previous_offset))
    anchor_shift = float(
        np.linalg.norm(
            np.asarray(current_anchor[:2], dtype=np.float32)
            - np.asarray(previous_anchor[:2], dtype=np.float32)
        )
    )
    return SupportRelockMetrics(
        normal_jump_deg=normal_jump,
        support_height_jump_m=float(abs(current_signed - previous_signed)),
        anchor_shift_m=anchor_shift,
        current_signed_distance_m=current_signed,
        previous_signed_distance_m=previous_signed,
    )
