"""Shared helpers for enforcing the standardized trajectory origin contract."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pemoin.data.contracts import PoseData, PoseSample, ResourceStore


def compute_origin_anchor_translation(
    camera_to_world: np.ndarray,
    *,
    anchor_height_m: float,
) -> np.ndarray:
    """Return the rigid translation that anchors frame 0 at (0, 0, anchor_height_m)."""
    c2w = np.asarray(camera_to_world, dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(
            "camera_to_world must have shape (N, 4, 4), "
            f"got {c2w.shape}."
        )
    if c2w.shape[0] == 0:
        raise ValueError("At least one pose is required to anchor trajectory origin.")
    if not np.isfinite(c2w).all():
        raise ValueError("camera_to_world contains non-finite values.")
    anchor_height = float(anchor_height_m)
    if not np.isfinite(anchor_height) or anchor_height <= 0.0:
        raise ValueError(
            f"anchor_height_m must be a finite positive float, got {anchor_height_m!r}."
        )
    target = np.array([0.0, 0.0, anchor_height], dtype=np.float32)
    return target - np.asarray(c2w[0, :3, 3], dtype=np.float32)


def resolve_anchor_height_from_store(resources: ResourceStore, frame_index: int) -> float:
    """Load the camera height that defines the standardized trajectory anchor."""
    height = resources.load_camera_height(int(frame_index))
    value = float(height.height_m)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Camera height for frame {frame_index} must be finite and > 0, got {value!r}."
        )
    return value


def anchor_pose_data_to_origin(
    pose_data: PoseData,
    *,
    anchor_height_m: float,
    metadata_label: str,
) -> tuple[PoseData, np.ndarray]:
    """Translate a trajectory so the first frame camera is at (0, 0, anchor_height_m)."""
    if not pose_data.samples:
        raise ValueError("PoseData must contain at least one sample to anchor origin.")
    samples = sorted(pose_data.samples, key=lambda sample: int(sample.frame_index))
    c2w_stack = np.stack(
        [np.asarray(sample.camera_to_world, dtype=np.float32) for sample in samples],
        axis=0,
    )
    delta = compute_origin_anchor_translation(c2w_stack, anchor_height_m=anchor_height_m)

    base_meta = dict(pose_data.metadata or {})
    base_meta.update(
        {
            "origin_anchor_enabled": True,
            "origin_anchor_mode": "first_frame_camera_height",
            "origin_anchor_target": [0.0, 0.0, float(anchor_height_m)],
            "origin_anchor_translation": delta.astype(float).tolist(),
            "origin_anchor_frame_index": int(samples[0].frame_index),
            "origin_anchor_metadata_source": str(metadata_label),
        }
    )

    anchored_samples: list[PoseSample] = []
    for sample in samples:
        c2w = np.asarray(sample.camera_to_world, dtype=np.float32).copy()
        c2w[:3, 3] = np.asarray(c2w[:3, 3], dtype=np.float32) + delta
        sample_meta = dict(sample.metadata or {})
        sample_meta.update(base_meta)
        anchored_samples.append(
            PoseSample(
                frame_index=int(sample.frame_index),
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w.astype(np.float64)).astype(np.float32),
                confidence=sample.confidence,
                metadata=sample_meta,
            )
        )

    return PoseData(samples=anchored_samples, metadata=base_meta), delta


def save_origin_anchored_trajectory(
    resources: ResourceStore,
    pose_data: PoseData,
    *,
    metadata_label: str,
) -> Tuple[PoseData, np.ndarray]:
    """Anchor a trajectory using the first frame camera height, then persist it."""
    if not pose_data.samples:
        raise ValueError("PoseData must contain at least one sample before saving.")
    first_frame = min(int(sample.frame_index) for sample in pose_data.samples)
    anchor_height_m = resolve_anchor_height_from_store(resources, first_frame)
    anchored_pose_data, delta = anchor_pose_data_to_origin(
        pose_data,
        anchor_height_m=anchor_height_m,
        metadata_label=metadata_label,
    )
    resources.save_trajectory(anchored_pose_data)
    return anchored_pose_data, delta
