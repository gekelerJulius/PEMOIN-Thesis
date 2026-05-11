"""
Trajectory cleanup and pose conditioning utilities.

Provides two APIs:

1. **PoseConditioningSettings / condition_poses** — full 4-stage pose
   conditioning pipeline (translation outlier removal, acceleration smoothing,
   quaternion SLERP rotation smoothing, driving-prior roll/pitch
   regularization).  Used by the runtime after trajectory consolidation.

2. **TrajectoryCleanupOptions / cleanup_camera_to_world** — legacy lightweight
   API (outlier removal + optional moving-average translation smoothing).
   Used by PanSt3R and DepthAnything3 adapters.  Preserved for backward
   compatibility; signatures and behaviour are identical to the previous
   version.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# New pose-conditioning API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoseConditioningSettings:
    """Settings for the 4-stage pose conditioning pipeline."""

    enabled: bool = False
    outlier_speed_factor: float = 5.0
    acceleration_window: int = 5
    acceleration_sigma: float = 1.0
    rotation_window: int = 5
    driving_prior_lambda: float = 0.5
    driving_prior_window: int = 11

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any] | None
    ) -> "PoseConditioningSettings":
        if not mapping:
            return cls()

        def _float(key: str, default: float) -> float:
            try:
                v = float(mapping.get(key, default))
            except (TypeError, ValueError):
                v = default
            return v if math.isfinite(v) else default

        def _int(key: str, default: int) -> int:
            try:
                return max(0, int(mapping.get(key, default)))
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=bool(mapping.get("enabled", False)),
            outlier_speed_factor=_float("outlier_speed_factor", 5.0),
            acceleration_window=_int("acceleration_window", 5),
            acceleration_sigma=_float("acceleration_sigma", 1.0),
            rotation_window=_int("rotation_window", 5),
            driving_prior_lambda=_float("driving_prior_lambda", 0.5),
            driving_prior_window=_int("driving_prior_window", 11),
        )


def condition_poses(
    c2w: np.ndarray,
    settings: PoseConditioningSettings,
) -> Tuple[np.ndarray, MutableMapping[str, Any]]:
    """Apply 4-stage pose conditioning to camera-to-world matrices.

    Parameters
    ----------
    c2w : np.ndarray, shape (N, 4, 4)
        Camera-to-world matrices.
    settings : PoseConditioningSettings

    Returns
    -------
    c2w_out : np.ndarray, shape (N, 4, 4)
        Conditioned camera-to-world matrices (float32).
    metadata : dict
        Information about what was applied.
    """
    metadata: MutableMapping[str, Any] = {}
    if not settings.enabled:
        return c2w, metadata

    arr = np.asarray(c2w, dtype=np.float64).copy()
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected c2w shape (N,4,4), got {arr.shape}"
        )
    n = arr.shape[0]
    if n < 3:
        return c2w, metadata

    LOG.info("Running pose conditioning on raw trajectory (%d frames).", n)

    translations = arr[:, :3, 3].copy()
    rotations = arr[:, :3, :3].copy()

    # Stage 1: translation outlier removal
    median_speed = _median_speed(translations)
    if settings.outlier_speed_factor > 1.0 and median_speed > 0:
        cleaned, outlier_info = _remove_translation_outliers(
            translations, median_speed, settings.outlier_speed_factor
        )
        if outlier_info.get("outliers_interpolated", 0) > 0:
            metadata.update(outlier_info)
            translations = cleaned

    # Stage 2: acceleration smoothing
    if settings.acceleration_window >= 3 and n >= 4:
        translations = _smooth_accelerations(
            translations,
            window=settings.acceleration_window,
            sigma=settings.acceleration_sigma,
        )
        metadata["acceleration_smoothed"] = True

    # Stage 3: quaternion SLERP rotation smoothing
    if settings.rotation_window >= 3:
        rotations = _smooth_rotations_slerp(
            rotations, window=settings.rotation_window
        )
        metadata["rotation_smoothed"] = True

    # Stage 4: driving prior (soft roll/pitch regularization)
    if settings.driving_prior_lambda > 0 and settings.driving_prior_window >= 3:
        rotations = _apply_driving_prior(
            rotations,
            lam=settings.driving_prior_lambda,
            window=settings.driving_prior_window,
        )
        metadata["driving_prior_applied"] = True

    # Reassemble
    out = arr.copy()
    out[:, :3, 3] = translations
    out[:, :3, :3] = rotations
    metadata["pose_conditioning_applied"] = True
    return out.astype(np.float32), metadata


# ---------------------------------------------------------------------------
# Stage 2 helper: acceleration smoothing
# ---------------------------------------------------------------------------


def _smooth_accelerations(
    translations: np.ndarray, *, window: int, sigma: float
) -> np.ndarray:
    """Smooth accelerations with a Gaussian kernel, then integrate back."""
    from scipy.ndimage import convolve1d

    velocities = np.diff(translations, axis=0)  # (N-1, 3)
    accelerations = np.diff(velocities, axis=0)  # (N-2, 3)

    if accelerations.shape[0] == 0:
        return translations

    # Gaussian kernel
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()

    # Smooth each axis independently
    acc_smooth = np.empty_like(accelerations)
    for axis in range(3):
        acc_smooth[:, axis] = convolve1d(
            accelerations[:, axis], kernel, mode="nearest"
        )

    # Integrate back: accelerations → velocities → translations
    v0 = velocities[0]
    vel_smooth = np.empty_like(velocities)
    vel_smooth[0] = v0
    for i in range(acc_smooth.shape[0]):
        vel_smooth[i + 1] = vel_smooth[i] + acc_smooth[i]

    t0 = translations[0]
    t_smooth = np.empty_like(translations)
    t_smooth[0] = t0
    for i in range(vel_smooth.shape[0]):
        t_smooth[i + 1] = t_smooth[i] + vel_smooth[i]

    return t_smooth


# ---------------------------------------------------------------------------
# Stage 3 helper: quaternion SLERP rotation smoothing
# ---------------------------------------------------------------------------


def _smooth_rotations_slerp(
    rotations: np.ndarray, *, window: int
) -> np.ndarray:
    """Smooth rotations using quaternion averaging in a sliding window."""
    from scipy.spatial.transform import Rotation

    n = rotations.shape[0]
    quats = Rotation.from_matrix(rotations).as_quat()  # (N, 4) xyzw

    # Ensure hemisphere consistency
    for i in range(1, n):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    half = window // 2
    smoothed = np.empty_like(quats)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        avg = quats[start:end].mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm < 1e-10:
            smoothed[i] = quats[i]
        else:
            smoothed[i] = avg / norm

    return Rotation.from_quat(smoothed).as_matrix()


# ---------------------------------------------------------------------------
# Stage 4 helper: driving prior (soft roll/pitch regularization)
# ---------------------------------------------------------------------------


def _apply_driving_prior(
    rotations: np.ndarray, *, lam: float, window: int
) -> np.ndarray:
    """Soft exponential decay of roll/pitch toward a local running average.

    Uses YXZ Euler convention (Blender Y-up):
    - Component 0 = yaw   → untouched
    - Component 1 = pitch → regularized
    - Component 2 = roll  → regularized
    """
    from scipy.spatial.transform import Rotation

    n = rotations.shape[0]
    euler = Rotation.from_matrix(rotations).as_euler("YXZ")  # (N, 3)

    half = window // 2
    for comp in (1, 2):  # pitch, roll
        angles = euler[:, comp].copy()
        # Compute local running average
        avg = np.empty_like(angles)
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            avg[i] = angles[start:end].mean()
        # Apply soft decay toward average
        deviation = angles - avg
        euler[:, comp] = avg + deviation * np.exp(-lam * np.abs(deviation))

    return Rotation.from_euler("YXZ", euler).as_matrix()


# ---------------------------------------------------------------------------
# Legacy API — backward-compatible with existing callers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectoryCleanupOptions:
    enabled: bool = False
    translation_window: int = 0
    outlier_speed_factor: float = 5.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrajectoryCleanupOptions":
        enabled = bool(mapping.get("trajectory_cleanup_enabled", False))
        window_raw = mapping.get("trajectory_cleanup_translation_window", 0)
        try:
            translation_window = int(window_raw or 0)
        except (TypeError, ValueError):
            translation_window = 0
        translation_window = max(0, translation_window)

        factor_raw = mapping.get("trajectory_cleanup_outlier_speed_factor", 5.0)
        try:
            outlier_speed_factor = float(factor_raw)
        except (TypeError, ValueError):
            outlier_speed_factor = 5.0
        if not math.isfinite(outlier_speed_factor):
            outlier_speed_factor = 5.0

        return cls(
            enabled=enabled,
            translation_window=translation_window,
            outlier_speed_factor=outlier_speed_factor,
        )

    def signature(self) -> str:
        payload = {
            "enabled": self.enabled,
            "translation_window": self.translation_window,
            "outlier_speed_factor": self.outlier_speed_factor,
        }
        return json.dumps(payload, sort_keys=True)


def cleanup_camera_to_world(
    cam_c2w: np.ndarray,
    options: TrajectoryCleanupOptions,
) -> Tuple[np.ndarray, MutableMapping[str, Any]]:
    """
    Return a cleaned copy of camera-to-world matrices and metadata describing applied steps.
    """
    metadata: MutableMapping[str, Any] = {}
    if not options.enabled:
        return cam_c2w, metadata

    arr = np.asarray(cam_c2w, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] not in ((4, 4), (3, 4)):
        raise ValueError(f"Unsupported cam_c2w shape {arr.shape}; expected (N,4,4) or (N,3,4).")
    if arr.shape[1:] == (3, 4):
        padded = np.broadcast_to(np.array([0, 0, 0, 1], dtype=arr.dtype), (arr.shape[0], 1, 4))
        arr = np.concatenate([arr, padded], axis=1)

    translations = arr[:, :3, 3].copy()
    n = translations.shape[0]
    if n < 2:
        return cam_c2w, metadata

    applied = False

    cleaned = translations
    median_speed = _median_speed(translations)
    if options.outlier_speed_factor > 1.0 and median_speed > 0:
        cleaned_outliers, outlier_info = _remove_translation_outliers(
            cleaned, median_speed, options.outlier_speed_factor
        )
        if outlier_info.get("outliers_interpolated", 0) > 0:
            metadata.update(outlier_info)
            cleaned = cleaned_outliers
            applied = True

    if options.translation_window >= 3 and n >= 3:
        smoothed = _smooth_translations(cleaned, options.translation_window)
        if np.linalg.norm(smoothed - cleaned) / max(1e-6, np.linalg.norm(cleaned)) > 1e-3:
            cleaned = smoothed
            metadata["translation_smoothed"] = True
            metadata["translation_smoothing_window"] = options.translation_window
            applied = True

    if not applied:
        return cam_c2w, metadata

    cleaned_c2w = arr.astype(np.float32).copy()
    cleaned_c2w[:, :3, 3] = cleaned.astype(np.float32)
    metadata["trajectory_cleanup_applied"] = True
    return cleaned_c2w, metadata


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _median_speed(translations: np.ndarray) -> float:
    deltas = np.diff(translations, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    finite = speeds[np.isfinite(speeds)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def _remove_translation_outliers(
    translations: np.ndarray,
    median_speed: float,
    factor: float,
) -> Tuple[np.ndarray, MutableMapping[str, Any]]:
    deltas = np.diff(translations, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    threshold = factor * median_speed
    valid_steps = speeds <= threshold
    valid_frames = np.ones(translations.shape[0], dtype=bool)
    valid_frames[1:] = valid_steps
    if valid_frames.mean() < 0.5:
        return translations, {"outliers_interpolated": 0, "outlier_threshold": threshold}
    cleaned = translations.copy()
    outlier_indices = np.where(~valid_frames)[0]
    for idx in outlier_indices:
        prev_idx = idx - 1
        next_idx = idx + 1
        while next_idx < cleaned.shape[0] and not valid_frames[next_idx]:
            next_idx += 1
        if next_idx >= cleaned.shape[0]:
            cleaned[idx:] = cleaned[prev_idx]
            break
        t = (idx - prev_idx) / max(1, (next_idx - prev_idx))
        cleaned[idx] = (1.0 - t) * cleaned[prev_idx] + t * cleaned[next_idx]
        valid_frames[idx] = True
    return cleaned, {"outliers_interpolated": int(outlier_indices.size), "outlier_threshold": threshold}


def _smooth_translations(translations: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    n = translations.shape[0]
    smoothed = np.zeros_like(translations)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed[i] = translations[start:end].mean(axis=0)
    return smoothed
