"""
Utility helpers for writing geometry outputs to a common, comparable format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib as mpl
import numpy as np
from pemoin.visualization.debug_artifacts import save_rgb_image
from pemoin.utils.resolution import _resize_array_numpy


def save_standard_geometry(
    output_dir: Path,
    *,
    source: str,
    depths: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    frame_ids: Optional[Sequence[int]] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    source_camera_convention: str = "opencv",
) -> Path:
    """
    Persist geometry predictions to a unified, human-readable layout:

    - JSON metadata (`geometry.json`) for quick inspection.
    - Intrinsics/extrinsics as JSON files.
    - `depths.npz` storing all frames (and optional confidence).
    - Per-frame depth visualisations under `depth_viz/` using the Turbo colormap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "depth_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    depths_arr = np.asarray(depths)
    extr_arr = _ensure_rank(extrinsics_w2c, 3)
    intr_arr = _ensure_rank(intrinsics, 3)
    if target_shape is not None and len(target_shape) == 2:
        depths_arr, confidence, intr_arr = _align_geometry(depths_arr, confidence, intr_arr, target_shape)
    frame_ids_arr = (
        np.arange(depths_arr.shape[0], dtype=np.int32)
        if frame_ids is None
        else np.asarray(list(frame_ids))
    )

    # Convert extrinsics to Blender convention when needed.
    if str(source_camera_convention).lower() in {"opencv", "cv"}:
        t4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        extr_arr = t4 @ extr_arr @ t4

    # Save intrinsics/extrinsics (JSON).
    intrinsics_list = intr_arr.tolist()
    extrinsics_list = extr_arr.tolist()
    (output_dir / "intrinsics.json").write_text(
        json.dumps({"intrinsics": intrinsics_list}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "extrinsics_w2c.json").write_text(
        json.dumps({"extrinsics_w2c": extrinsics_list}, indent=2),
        encoding="utf-8",
    )

    # Save depth & optional confidence as a single NPZ for downstream consumption.
    depth_payload = {"depths": depths_arr.astype(np.float32)}
    conf_arr = np.asarray(confidence) if confidence is not None else None
    if conf_arr is not None:
        depth_payload["confidence"] = conf_arr.astype(np.float32)
    np.savez_compressed(output_dir / "depths.npz", **depth_payload)

    # Write per-frame Turbo visualisations.
    for idx, fid in enumerate(frame_ids_arr):
        viz = _depth_to_turbo(depths_arr[idx])
        save_rgb_image(viz_dir / f"{int(fid):06d}.png", viz)

    metadata = {
        "source": source,
        "frame_count": int(depths_arr.shape[0]),
        "frame_ids": frame_ids_arr.tolist(),
        "depth_shape": list(depths_arr.shape),
        "intrinsics_shape": list(intr_arr.shape),
        "extrinsics_shape": list(extr_arr.shape),
        "has_confidence": confidence is not None,
        "format": "geometry-v2",
        "camera_convention": "blender",
        "source_camera_convention": source_camera_convention,
        "files": {
            "intrinsics": "intrinsics.json",
            "extrinsics_w2c": "extrinsics_w2c.json",
            "depths_npz": "depths.npz",
            "depth_visualizations": "depth_viz/",
        },
        "colormap": "turbo_reversed",  # near = warm/bright, far = cool/dark
    }
    (output_dir / "geometry.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return output_dir


def _ensure_rank(arr: np.ndarray, rank: int) -> np.ndarray:
    array = np.asarray(arr)
    if array.ndim == rank - 1:
        return np.broadcast_to(array, (1,) + array.shape).copy()
    return array


def _depth_to_turbo(depth: np.ndarray) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(depth_arr) & (depth_arr > 0)
    if not np.any(mask):
        return np.zeros((*depth_arr.shape, 3), dtype=np.uint8)
    valid = depth_arr[mask]
    d_min = float(valid.min())
    # Clamp visualization to the 80th percentile to amplify nearby detail.
    d_max = float(np.percentile(valid, 80))
    if d_max <= d_min:
        d_max = float(valid.max())
    norm = np.zeros_like(depth_arr, dtype=np.float32)
    if d_max > d_min:
        norm[mask] = np.clip((depth_arr[mask] - d_min) / (d_max - d_min), 0.0, 1.0)
    else:
        norm[mask] = 0.5
    turbo = mpl.colormaps["turbo"]
    rgba = turbo(1.0 - norm)  # invert so nearer = warmer/brighter, farther = cooler/darker
    viz = (rgba[..., :3] * 255.0).astype(np.uint8)
    viz[~mask] = 0
    return viz


def _align_geometry(
    depths: np.ndarray,
    confidence: Optional[np.ndarray],
    intrinsics: np.ndarray,
    target_shape: Tuple[int, int],
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if depths.shape[1:3] == (target_h, target_w):
        return depths, confidence, intrinsics
    resized_depths = _resize_stack(depths, target_w, target_h, interpolation="bilinear")
    resized_conf = None
    if confidence is not None:
        resized_conf = _resize_stack(confidence, target_w, target_h, interpolation="bilinear")
    scaled_intrinsics = _scale_intrinsics_stack(intrinsics, source_shape=depths.shape[1:3], target_shape=(target_h, target_w))
    return resized_depths, resized_conf, scaled_intrinsics


def _resize_stack(arr: np.ndarray, target_w: int, target_h: int, *, interpolation: str = "bilinear") -> np.ndarray:
    layers = []
    for layer in np.asarray(arr):
        layers.append(_resize_array(layer, target_w, target_h, interpolation=interpolation))
    return np.stack(layers, axis=0)


def _resize_array(arr: np.ndarray, target_w: int, target_h: int, *, interpolation: str = "bilinear") -> np.ndarray:
    try:
        import cv2  # type: ignore

        interp = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
        return cv2.resize(arr, (target_w, target_h), interpolation=interp)
    except Exception:
        try:
            from PIL import Image

            mode = "F" if arr.dtype not in (np.uint8, np.uint16) else None
            img = Image.fromarray(arr.astype(np.float32) if mode else arr)
            pil_interp = Image.BILINEAR if interpolation == "bilinear" else Image.NEAREST
            resized = img.resize((target_w, target_h), resample=pil_interp)
            return np.asarray(resized)
        except Exception:
            return _resize_array_numpy(arr, (target_h, target_w), interpolation=interpolation)


def _scale_intrinsics_stack(
    intrinsics: np.ndarray,
    *,
    source_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    intr = _ensure_rank(intrinsics, 3).astype(np.float32).copy()
    src_h, src_w = float(source_shape[0]), float(source_shape[1])
    tgt_h, tgt_w = float(target_shape[0]), float(target_shape[1])
    if src_h <= 0 or src_w <= 0:
        return intr
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h
    intr[..., 0, 0] *= scale_x
    intr[..., 1, 1] *= scale_y
    intr[..., 0, 2] *= scale_x
    intr[..., 1, 2] *= scale_y
    return intr
