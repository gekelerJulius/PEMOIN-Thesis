"""Utilities for normalising image/depth data to a common working resolution."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from pemoin.data.models import DepthData, FrameData, IntrinsicsData, SemanticSegment, SemanticsData
from pemoin.utils.camera_calibration import (
    validate_and_normalize_intrinsics,
    resolve_intrinsics_resolution,
)


def normalize_frame_resolution(frame: FrameData, target_shape: Optional[Sequence[int]]) -> FrameData:
    """Resize frame image so the longest side matches target max while preserving aspect ratio."""
    target_max = _resolve_target_max(target_shape)
    if target_max is None or frame.image is None:
        return frame
    src_h, src_w = frame.image.shape[:2]
    if max(src_h, src_w) == target_max:
        frame.metadata.setdefault("working_resolution", [src_h, src_w])
        return frame
    resized, scale = _scale_image_to_max(frame.image, target_max)
    meta = dict(frame.metadata)
    meta["working_resolution"] = list(resized.shape[:2])
    meta["input_resolution"] = [src_h, src_w]
    meta["scale"] = scale
    return FrameData(
        frame_id=frame.frame_id,
        index=frame.index,
        timestamp=frame.timestamp,
        image=resized,
        metadata=meta,
    )


def scale_intrinsics(intrinsics: IntrinsicsData, target_shape: Optional[Sequence[int]]) -> IntrinsicsData:
    """Scale intrinsics to match a max-side resized resolution."""
    target_dims = _resolve_target_dims(target_shape)
    if target_dims is None:
        return intrinsics
    matrix = np.asarray(intrinsics.matrix, dtype=np.float32)
    source = resolve_intrinsics_resolution(getattr(intrinsics, "metadata", {}) or {}, matrix)
    src_h = float(source.height)
    src_w = float(source.width)
    meta = dict(intrinsics.metadata)
    target_h, target_w = target_dims
    scale_y = float(target_h) / float(src_h)
    scale_x = float(target_w) / float(src_w)
    meta["input_resolution"] = [float(src_h), float(src_w)]
    meta["reference_resolution"] = [int(target_h), int(target_w)]
    meta["resolution"] = [int(target_h), int(target_w)]
    meta["working_resolution"] = [int(target_h), int(target_w)]
    meta["width"] = int(target_w)
    meta["height"] = int(target_h)
    meta["scale"] = (scale_y + scale_x) * 0.5 if abs(scale_y - scale_x) < 1e-6 else None
    meta["scale_y"] = scale_y
    meta["scale_x"] = scale_x
    if abs(scale_y - 1.0) < 1e-6 and abs(scale_x - 1.0) < 1e-6:
        _, normalized_meta, _ = validate_and_normalize_intrinsics(
            matrix,
            meta,
            frame_shape=(int(target_h), int(target_w)),
            allow_principal_point_fallback=False,
        )
        if normalized_meta == intrinsics.metadata:
            return intrinsics
        return IntrinsicsData(matrix=matrix, distortion=intrinsics.distortion, metadata=normalized_meta)
    scaled = matrix.copy()
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    _, normalized_meta, _ = validate_and_normalize_intrinsics(
        scaled,
        meta,
        frame_shape=(int(target_h), int(target_w)),
        allow_principal_point_fallback=False,
    )
    return IntrinsicsData(matrix=scaled, distortion=intrinsics.distortion, metadata=normalized_meta)


def resize_depth(depth: DepthData, target_shape: Optional[Sequence[int]]) -> DepthData:
    """Resize depth (and confidence map) with max-side scaling."""
    target_dims = _resolve_target_dims(target_shape)
    if target_dims is None:
        return depth
    target_h, target_w = target_dims
    src_h, src_w = depth.depth.shape[:2]
    if src_h == target_h and src_w == target_w:
        meta = dict(depth.metadata)
        if meta.get("reference_resolution") != [int(target_h), int(target_w)] or meta.get("resolution") != [int(target_h), int(target_w)]:
            meta.setdefault("input_resolution", [int(src_h), int(src_w)])
            meta["reference_resolution"] = [int(target_h), int(target_w)]
            meta["resolution"] = [int(target_h), int(target_w)]
            return DepthData(
                frame_index=depth.frame_index,
                depth=depth.depth,
                confidence=depth.confidence,
                metadata=meta,
            )
        return depth
    resized_depth = _resize_array(depth.depth, (target_h, target_w), interpolation="bilinear")
    resized_conf = None
    if depth.confidence is not None:
        resized_conf = _resize_array(depth.confidence, (target_h, target_w), interpolation="nearest")
    meta = dict(depth.metadata)
    meta.setdefault("input_resolution", [int(src_h), int(src_w)])
    meta["reference_resolution"] = [int(target_h), int(target_w)]
    meta["resolution"] = [int(target_h), int(target_w)]
    return DepthData(
        frame_index=depth.frame_index,
        depth=resized_depth,
        confidence=resized_conf,
        metadata=meta,
    )


def resize_semantics(semantics: SemanticsData, target_shape: Optional[Sequence[int]]) -> SemanticsData:
    """Resize semantic masks/labels to a target (H, W) using nearest neighbor."""
    target_dims = _resolve_target_dims(target_shape)
    if target_dims is None:
        return semantics
    target_h, target_w = target_dims
    label_ids = semantics.label_ids
    segment_ids = semantics.segment_ids
    src_shape = None
    if label_ids is not None and label_ids.ndim >= 2:
        src_shape = label_ids.shape[:2]
    elif segment_ids is not None and segment_ids.ndim >= 2:
        src_shape = segment_ids.shape[:2]
    elif semantics.segments:
        src_shape = np.asarray(semantics.segments[0].mask).shape[:2]
    if label_ids is not None and label_ids.shape[:2] == (target_h, target_w):
        if segment_ids is None or segment_ids.shape[:2] == (target_h, target_w):
            meta = dict(semantics.metadata)
            if meta.get("reference_resolution") != [int(target_h), int(target_w)] or meta.get("resolution") != [int(target_h), int(target_w)]:
                if src_shape is not None:
                    meta.setdefault("input_resolution", [int(src_shape[0]), int(src_shape[1])])
                meta["reference_resolution"] = [int(target_h), int(target_w)]
                meta["resolution"] = [int(target_h), int(target_w)]
                return SemanticsData(
                    frame_index=semantics.frame_index,
                    segments=semantics.segments,
                    frame_id=semantics.frame_id,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                    metadata=meta,
                )
            return semantics
    resized_label_ids = _resize_array(label_ids, (target_h, target_w), interpolation="nearest") if label_ids is not None else None
    resized_segment_ids = _resize_array(segment_ids, (target_h, target_w), interpolation="nearest") if segment_ids is not None else None
    resized_segments: list[SemanticSegment] = []
    for seg in semantics.segments:
        mask = np.asarray(seg.mask, dtype=bool)
        if mask.shape[:2] != (target_h, target_w):
            mask = _resize_array(mask.astype(np.uint8), (target_h, target_w), interpolation="nearest").astype(bool)
        resized_segments.append(
            SemanticSegment(
                segment_id=seg.segment_id,
                label=seg.label,
                score=seg.score,
                mask=mask,
                label_id=seg.label_id,
                area=int(mask.sum()),
                bbox=seg.bbox,
                metadata=dict(seg.metadata),
            )
        )
    meta = dict(semantics.metadata)
    if src_shape is not None:
        meta.setdefault("input_resolution", [int(src_shape[0]), int(src_shape[1])])
    meta["reference_resolution"] = [int(target_h), int(target_w)]
    meta["resolution"] = [int(target_h), int(target_w)]
    return SemanticsData(
        frame_index=semantics.frame_index,
        segments=resized_segments,
        frame_id=semantics.frame_id,
        segment_ids=resized_segment_ids,
        label_ids=resized_label_ids,
        metadata=meta,
    )


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _resolve_target_max(target_shape: Optional[Sequence[int] | int | float]) -> Optional[int]:
    if target_shape is None:
        return None
    if isinstance(target_shape, (int, float)):
        value = int(target_shape)
        return value if value > 0 else None
    if not isinstance(target_shape, (list, tuple)) or len(target_shape) == 0:
        return None
    value = int(max(target_shape))
    return value if value > 0 else None


def _resolve_target_dims(target_shape: Optional[Sequence[int] | int | float]) -> Optional[tuple[int, int]]:
    if target_shape is None:
        return None
    if isinstance(target_shape, (int, float)):
        size = int(target_shape)
        return (size, size) if size > 0 else None
    if not isinstance(target_shape, (list, tuple)) or len(target_shape) < 2:
        return None
    target_h = int(target_shape[0])
    target_w = int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        return None
    return (target_h, target_w)


def _scale_params(src_h: float, src_w: float, target_max: float) -> tuple[int, int, float]:
    scale = float(target_max) / float(max(src_h, src_w))
    scaled_h = int(max(1, round(src_h * scale)))
    scaled_w = int(max(1, round(src_w * scale)))
    return scaled_h, scaled_w, scale


def _scale_image_to_max(image: np.ndarray, target_max: int) -> tuple[np.ndarray, float]:
    src_h, src_w = image.shape[:2]
    scaled_h, scaled_w, scale = _scale_params(float(src_h), float(src_w), float(target_max))
    resized = _resize_image(image, (scaled_h, scaled_w))
    return resized, scale


def _scale_array_to_max(
    arr: np.ndarray, target_max: int, *, interpolation: str = "bilinear"
) -> tuple[np.ndarray, float]:
    src_h, src_w = arr.shape[:2]
    scaled_h, scaled_w, scale = _scale_params(float(src_h), float(src_w), float(target_max))
    resized = _resize_array(arr, (scaled_h, scaled_w), interpolation=interpolation)
    return resized, scale


def _resize_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    try:
        import cv2  # type: ignore

        return cv2.resize(image, (int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_LINEAR)
    except Exception:
        try:
            from PIL import Image

            if image.dtype != np.uint8:
                img = Image.fromarray(image.astype(np.float32), mode="F")
            else:
                img = Image.fromarray(image)
            resized = img.resize((int(target_shape[1]), int(target_shape[0])), resample=Image.BILINEAR)
            return np.asarray(resized)
        except Exception:
            return _resize_array_numpy(image, target_shape, interpolation="bilinear")


def _resize_array(arr: np.ndarray, target_shape: Tuple[int, int], *, interpolation: str = "bilinear") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[:2] == target_shape:
        return arr
    try:
        import cv2  # type: ignore

        interp = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
        return cv2.resize(arr, (int(target_shape[1]), int(target_shape[0])), interpolation=interp)
    except Exception:
        try:
            from PIL import Image

            pil_interp = Image.BILINEAR if interpolation == "bilinear" else Image.NEAREST
            if arr.dtype != np.uint8:
                img = Image.fromarray(arr.astype(np.float32), mode="F")
            else:
                img = Image.fromarray(arr)
            resized = img.resize((int(target_shape[1]), int(target_shape[0])), resample=pil_interp)
            return np.asarray(resized)
        except Exception:
            return _resize_array_numpy(arr, target_shape, interpolation=interpolation)


def _resize_array_numpy(
    arr: np.ndarray, target_shape: Tuple[int, int], *, interpolation: str = "bilinear"
) -> np.ndarray:
    """Resize 2D or channel-last arrays without optional imaging dependencies."""
    array = np.asarray(arr)
    if array.ndim < 2:
        raise ValueError(f"Expected array with at least 2 dims for resize, got shape {array.shape}.")
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    src_h, src_w = int(array.shape[0]), int(array.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Target shape must be positive, got {target_shape}.")
    if (src_h, src_w) == (target_h, target_w):
        return array

    if interpolation not in {"bilinear", "nearest"}:
        raise ValueError(f"Unsupported interpolation mode: {interpolation!r}")

    if interpolation == "nearest":
        y_idx = _nearest_indices(src_h, target_h)
        x_idx = _nearest_indices(src_w, target_w)
        return array[y_idx[:, None], x_idx[None, :], ...]

    work = np.asarray(array, dtype=np.float32)
    squeezed = work.ndim == 2
    if squeezed:
        work = work[:, :, None]

    y = _resample_positions(src_h, target_h)
    x = _resample_positions(src_w, target_w)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (y - y0.astype(np.float32))[:, None, None]
    wx = (x - x0.astype(np.float32))[None, :, None]

    top_left = work[y0[:, None], x0[None, :], :]
    top_right = work[y0[:, None], x1[None, :], :]
    bottom_left = work[y1[:, None], x0[None, :], :]
    bottom_right = work[y1[:, None], x1[None, :], :]

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    resized = top * (1.0 - wy) + bottom * wy
    if squeezed:
        resized = resized[:, :, 0]
    return _cast_resized_like_input(resized, array.dtype)


def _resample_positions(source_size: int, target_size: int) -> np.ndarray:
    coords = (np.arange(target_size, dtype=np.float32) + 0.5) * (float(source_size) / float(target_size)) - 0.5
    return np.clip(coords, 0.0, float(max(source_size - 1, 0)))


def _nearest_indices(source_size: int, target_size: int) -> np.ndarray:
    return np.rint(_resample_positions(source_size, target_size)).astype(np.int32)


def _cast_resized_like_input(resized: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return np.asarray(resized, dtype=dtype)
    if np.issubdtype(dtype, np.bool_):
        return np.asarray(resized > 0.5, dtype=bool)
    if np.issubdtype(dtype, np.integer):
        bounds = np.iinfo(dtype)
        return np.clip(np.rint(resized), bounds.min, bounds.max).astype(dtype)
    return np.asarray(resized, dtype=dtype)
