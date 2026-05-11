"""Road pixel selection and backprojection for geometry fusion.

Extracts confident road pixels from depth+semantics, backprojects to 3D camera
coordinates, and returns points with weights for downstream plane fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pemoin.data.contracts import ResourceStore, SemanticsAuxData, SemanticsData
from pemoin.geometry.camera_model import backproject_uv_depth_to_camera


@dataclass
class RoadPixelSelection:
    """Result of road pixel selection and backprojection."""

    points_cam: np.ndarray  # (N, 3) in camera coordinates
    weights: np.ndarray  # (N,) confidence weights
    pixel_uv: np.ndarray  # (N, 2) pixel coordinates


class RoadPixelSelectionError(RuntimeError):
    """Raised when geometry-fusion road selection cannot proceed."""

    def __init__(self, message: str, *, diagnostic_payload: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostic_payload = dict(diagnostic_payload or {})


def _semantic_label_diagnostic(
    semantics: SemanticsData,
    road_labels: tuple[str, ...],
    semantics_aux: SemanticsAuxData | None,
) -> dict[str, Any]:
    label_map: dict[int, str] = {}
    for seg in semantics.segments:
        if seg.label_id is not None:
            label_map[int(seg.label_id)] = str(seg.label).strip().lower()

    available_ids: list[int] = []
    if semantics.label_ids is not None:
        unique_ids = np.unique(np.asarray(semantics.label_ids, dtype=np.int32))
        available_ids = [int(label_id) for label_id in unique_ids.tolist() if int(label_id) >= 0]
    elif semantics.segment_ids is not None:
        unique_ids = np.unique(np.asarray(semantics.segment_ids, dtype=np.int32))
        available_ids = [int(label_id) for label_id in unique_ids.tolist() if int(label_id) >= 0]
    else:
        available_ids = sorted(int(label_id) for label_id in label_map)

    available_label_entries = []
    for label_id in available_ids:
        label_name = label_map.get(label_id)
        available_label_entries.append(
            {
                "label_id": int(label_id),
                "label": label_name,
            }
        )
    matched_road_ids = [
        int(label_id)
        for label_id, name in label_map.items()
        if name in road_labels
    ]
    matched_road_labels = sorted(
        {label_map[label_id] for label_id in matched_road_ids if label_id in label_map}
    )
    return {
        "frame_index": int(semantics.frame_index),
        "configured_road_labels": list(road_labels),
        "available_semantic_labels": available_label_entries,
        "available_semantic_label_names": sorted(
            {entry["label"] for entry in available_label_entries if entry["label"]}
        ),
        "matched_road_label_ids": matched_road_ids,
        "matched_road_label_names": matched_road_labels,
        "semantics_aux_available": semantics_aux is not None,
        "semantics_aux_has_class_probabilities": bool(
            semantics_aux is not None and semantics_aux.class_probabilities is not None
        ),
        "semantics_aux_has_confidence": bool(
            semantics_aux is not None and semantics_aux.confidence is not None
        ),
        "semantics_aux_has_road_confidence": bool(
            semantics_aux is not None and semantics_aux.road_confidence is not None
        ),
    }


def _road_confidence(
    resources: ResourceStore,
    semantics: SemanticsData,
    road_labels: tuple[str, ...],
) -> np.ndarray:
    """Extract per-pixel road confidence from semantics data.

    Returns an HxW float32 array with road confidence values.
    """
    try:
        semantics_aux = resources.load_semantics_aux(int(semantics.frame_index))
    except Exception:
        semantics_aux = None
    diagnostic = _semantic_label_diagnostic(semantics, road_labels, semantics_aux)
    label_map = {
        int(entry["label_id"]): str(entry["label"]).strip().lower()
        for entry in diagnostic["available_semantic_labels"]
        if entry["label"] is not None
    }
    road_ids = [
        int(label_id)
        for label_id, name in label_map.items()
        if name in road_labels
    ]
    if not road_ids and semantics.label_ids is not None:
        unique_ids = np.unique(np.asarray(semantics.label_ids, dtype=np.int32))
        for label_id in unique_ids.tolist():
            if label_id < 0:
                continue
            name = label_map.get(int(label_id), "").strip().lower()
            if name in road_labels:
                road_ids.append(int(label_id))
    if not road_ids:
        raise RoadPixelSelectionError(
            "Geometry fusion frame "
            f"{semantics.frame_index}: no road labels resolved from semantics. "
            f"configured road_labels={list(road_labels)}; "
            f"available_labels={diagnostic['available_semantic_label_names'] or ['<unresolved>']}.",
            diagnostic_payload=diagnostic,
        )

    if semantics_aux is not None:
        if (
            semantics_aux.class_probabilities is not None
            and semantics_aux.class_probabilities.ndim == 3
        ):
            probs = np.asarray(semantics_aux.class_probabilities, dtype=np.float32)
            class_ids = (
                np.asarray(semantics_aux.class_ids, dtype=np.int32)
                if semantics_aux.class_ids is not None
                else np.arange(probs.shape[0], dtype=np.int32)
            )
            if probs.ndim != 3:
                raise RoadPixelSelectionError(
                    "Geometry fusion frame "
                    f"{semantics.frame_index}: probability tensor must be CxHxW, got {probs.shape}.",
                    diagnostic_payload=diagnostic,
                )
            road_channels = [i for i, cid in enumerate(class_ids.tolist()) if int(cid) in set(road_ids)]
            if not road_channels:
                raise RoadPixelSelectionError(
                    "Geometry fusion frame "
                    f"{semantics.frame_index}: road class ids {road_ids} not found in probability tensor channels "
                    f"class_ids={class_ids.tolist()}.",
                    diagnostic_payload=diagnostic,
                )
            road_conf = np.max(probs[np.asarray(road_channels, dtype=np.int32)], axis=0)
            return np.clip(road_conf.astype(np.float32), 0.0, 1.0)
        if semantics_aux.road_confidence is not None:
            return np.clip(np.asarray(semantics_aux.road_confidence, dtype=np.float32), 0.0, 1.0)
        if semantics_aux.confidence is not None:
            ids = semantics.label_ids if semantics.label_ids is not None else semantics.segment_ids
            if ids is None:
                raise RoadPixelSelectionError(
                    f"Geometry fusion frame {semantics.frame_index}: semantics has no label_ids/segment_ids.",
                    diagnostic_payload=diagnostic,
                )
            ids = np.asarray(ids, dtype=np.int32)
            mask = np.isin(ids, np.asarray(road_ids, dtype=np.int32)).astype(np.float32)
            return np.clip(np.asarray(semantics_aux.confidence, dtype=np.float32), 0.0, 1.0) * mask

    ids = semantics.label_ids if semantics.label_ids is not None else semantics.segment_ids
    if ids is None:
        raise RoadPixelSelectionError(
            f"Geometry fusion frame {semantics.frame_index}: semantics has no label_ids/segment_ids.",
            diagnostic_payload=diagnostic,
        )
    ids = np.asarray(ids, dtype=np.int32)
    mask = np.isin(ids, np.asarray(road_ids, dtype=np.int32))
    return mask.astype(np.float32)


def select_road_pixels(
    resources: ResourceStore,
    depth: np.ndarray,
    semantics: SemanticsData,
    K: np.ndarray,
    road_labels: tuple[str, ...],
    *,
    conf_thresh: float = 0.6,
    roi_bottom_frac: float = 0.45,
    z_max_m: float = 30.0,
    min_points: int = 500,
) -> RoadPixelSelection:
    """Select road pixels and backproject to camera-frame 3D coordinates.

    Args:
        depth: HxW depth map in meters.
        semantics: Semantics data for the same frame.
        K: 3x3 intrinsics matrix.
        road_labels: Road label names.
        conf_thresh: Minimum road confidence.
        roi_bottom_frac: Use only bottom fraction of image.
        z_max_m: Maximum depth in meters.
        min_points: Minimum required points.

    Returns:
        RoadPixelSelection with backprojected points, weights, pixel coords.
    """
    z = np.asarray(depth, dtype=np.float32)
    if z.ndim != 2:
        raise RuntimeError(f"Geometry fusion: depth must be HxW, got {z.shape}.")
    h, w = z.shape

    conf = _road_confidence(resources, semantics, road_labels)
    if conf.shape != z.shape:
        raise RuntimeError(
            f"Geometry fusion frame {semantics.frame_index}: confidence/depth shape mismatch "
            f"{conf.shape} vs {z.shape}."
        )

    yy, xx = np.indices((h, w), dtype=np.float32)
    valid = (
        np.isfinite(z)
        & (z > 0.1)
        & (z <= z_max_m)
        & (conf >= conf_thresh)
        & (yy >= float(h) * (1.0 - roi_bottom_frac))
    )
    n_valid = int(np.count_nonzero(valid))
    if n_valid < min_points:
        raise RuntimeError(
            f"Geometry fusion frame {semantics.frame_index}: insufficient road pixels "
            f"({n_valid} < {min_points})."
        )

    u = xx[valid]
    v = yy[valid]
    z_sel = z[valid]
    conf_sel = conf[valid]

    # Backproject using PEMOIN's standardized Blender camera convention.
    uv = np.stack([u, v], axis=1).astype(np.float32)
    points = backproject_uv_depth_to_camera(
        uv,
        z_sel.astype(np.float32),
        np.asarray(K, dtype=np.float32),
        camera_convention="blender",
    )

    # Weights: road confidence * near-field preference
    dist_w = 1.0 / np.maximum(1.0 + z_sel, 1.0)
    weights = np.clip(conf_sel, 0.0, 1.0) * dist_w
    weights = np.maximum(weights, 1e-4).astype(np.float32)

    pixel_uv = np.stack([u, v], axis=1).astype(np.float32)

    return RoadPixelSelection(points_cam=points, weights=weights, pixel_uv=pixel_uv)
