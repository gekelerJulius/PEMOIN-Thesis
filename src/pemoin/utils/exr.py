"""Helpers for reading EXR images with OpenEXR."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_exr_image(path: Path) -> np.ndarray:
    """Load an EXR image with OpenEXR.

    Depth EXRs in this project must be decoded through OpenEXR rather than
    OpenCV because some Unity exports are silently read back as all-zero arrays
    by OpenCV.
    """
    if not path.exists():
        raise FileNotFoundError(f"EXR file missing: {path}")

    depth = _load_exr_with_openexr(path)
    if depth is None:
        raise RuntimeError(
            f"Failed to decode EXR with OpenEXR: {path}. "
            "Install compatible OpenEXR/Imath bindings or inspect the source file."
        )
    return np.asarray(depth, dtype=np.float32)


def select_depth_channel(image: np.ndarray) -> np.ndarray:
    """Select the most plausible metric-depth channel from a loaded EXR image.

    Some Unity EXRs store metric depth in a non-first channel while leaving
    other channels constant or empty. Prefer the populated channel with the
    strongest finite-value variation rather than blindly taking channel 0.
    """
    depth = np.asarray(image, dtype=np.float32)
    if depth.ndim == 2:
        return depth
    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Depth EXR must be 2D or 3D, got shape {depth.shape}")

    best_idx = 0
    best_score = (-1, -1.0, -1.0)
    for idx in range(depth.shape[2]):
        channel = depth[..., idx]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            score = (0, -1.0, -1.0)
        else:
            nonzero_count = int(np.count_nonzero(finite))
            score = (
                nonzero_count,
                float(np.std(finite)),
                float(np.max(finite) - np.min(finite)),
            )
        if score > best_score:
            best_idx = idx
            best_score = score
    return depth[..., best_idx]


def _load_exr_with_openexr(path: Path) -> np.ndarray | None:
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore
    except Exception:
        return None

    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        channels = header.get("channels", {})
        if not channels:
            return None
        data_window = header["dataWindow"]
        width = int(data_window.max.x - data_window.min.x + 1)
        height = int(data_window.max.y - data_window.min.y + 1)
        arrays: list[np.ndarray] = []
        for channel_name in channels.keys():
            raw = exr.channel(str(channel_name), Imath.PixelType(Imath.PixelType.FLOAT))
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            if arr.size != width * height:
                continue
            arrays.append(arr.reshape((height, width)))
    finally:
        close = getattr(exr, "close", None)
        if callable(close):
            close()
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]
    return np.stack(arrays, axis=-1)
