"""
Provider-agnostic semantics debug visualizations.

Uses standardized SemanticsData + FrameData to generate debug assets across
all providers and profiles.

Canonical import path for new code: ``pemoin.visualization.semantics_debug``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from pemoin.data.contracts import (
    ResourceKind,
    ResourceStore,
    SemanticSegment,
    SemanticsData,
)
from pemoin.visualization.debug_artifacts import save_rgb_image
from pemoin.visualization.semantic_palette import (
    colorize_semantic_raster,
    semantic_palette_entries_from_id_map,
    semantic_palette_key,
    update_palette_manifest,
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SemanticsDebugSettings:
    """Configuration for provider-agnostic semantics debug outputs."""

    enabled: bool = False
    max_frames: Optional[int] = 5
    min_segment_area: int = 50
    road_label_tokens: Tuple[str, ...] = ("road",)
    overlay_alpha: float = 0.6
    output_subdir: str = "semantics_debug"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SemanticsDebugSettings:
        road_tokens = mapping.get("road_label_tokens", ("road",))
        if isinstance(road_tokens, str):
            road_tokens = [road_tokens]
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            max_frames=_optional_int(mapping.get("max_frames", 5)),
            min_segment_area=int(mapping.get("min_segment_area", 50)),
            road_label_tokens=tuple(str(tok).lower() for tok in road_tokens),
            overlay_alpha=float(mapping.get("overlay_alpha", 0.6)),
            output_subdir=str(mapping.get("output_subdir", "semantics_debug")),
        )


@dataclass(frozen=True, slots=True)
class _RoadPriorDebugSpec:
    name: str


@dataclass(frozen=True, slots=True)
class _TemporalFusionDebugSpec:
    model_names: Tuple[str, ...]
    road_priors: Tuple[_RoadPriorDebugSpec, ...]


@dataclass(frozen=True, slots=True)
class _ModelDebugOutput:
    label_ids: np.ndarray
    confidence: Optional[np.ndarray]
    road_confidence: Optional[np.ndarray]
    validity_mask: Optional[np.ndarray]


def generate_semantics_debug_visualizations(
    store: ResourceStore,
    settings: Optional[SemanticsDebugSettings] = None,
    frame_indices: Optional[Sequence[int]] = None,
) -> List[Path]:
    """Generate provider-agnostic semantics debug outputs."""
    if settings is None:
        settings = SemanticsDebugSettings()
    if not settings.enabled:
        LOG.debug("Semantics debug visualization is disabled")
        return []

    if not store.has(ResourceKind.SEMANTICS_2D):
        LOG.warning("Cannot generate semantics debug: no semantics data found")
        return []

    if not store.has(ResourceKind.FRAMES):
        LOG.warning("Cannot generate semantics debug: no frames found")
        return []

    use_store_indices = frame_indices is None
    if use_store_indices:
        frame_indices = store.frame_indices(ResourceKind.SEMANTICS_2D)
    frame_indices = list(frame_indices or [])
    if not frame_indices:
        LOG.warning("No semantics frames found to debug")
        return []

    output_dir = store.visualizations_dir() / settings.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_store_indices and settings.max_frames is not None and settings.max_frames > 0:
        frame_indices = frame_indices[: settings.max_frames]

    generated_paths: List[Path] = []
    palette_entries: Dict[str, Dict[str, Any]] = {}
    LOG.info(
        "Generating semantics debug visualizations for %d frames in %s",
        len(frame_indices),
        output_dir,
    )

    for frame_idx in frame_indices:
        try:
            frame_data = store.load_frame(frame_idx)
            semantics = store.load_semantics2d(frame_idx)

            label_ids, label_map, id_to_palette_key = _resolve_label_ids(
                semantics, frame_data.image, settings.min_segment_area
            )
            if label_ids is None:
                LOG.warning(
                    "No label ids available for frame %d; skipping debug", frame_idx
                )
                continue

            palette_entries.update(
                semantic_palette_entries_from_id_map(
                    id_to_palette_key=id_to_palette_key,
                    label_map=label_map,
                )
            )

            temporal_spec = _resolve_temporal_fusion_debug_spec(store, semantics)
            if temporal_spec:
                generated_paths.extend(
                    _write_temporal_fusion_debug(
                        store=store,
                        frame_idx=frame_idx,
                        frame_data=frame_data,
                        semantics=semantics,
                        fused_label_ids=label_ids,
                        label_map=label_map,
                        id_to_palette_key=id_to_palette_key,
                        base_output_dir=output_dir,
                        settings=settings,
                        spec=temporal_spec,
                    )
                )
            else:
                generated_paths.extend(
                    _write_label_debug(
                        frame_idx=frame_idx,
                        frame_data=frame_data,
                        label_ids=label_ids,
                        label_map=label_map,
                        id_to_palette_key=id_to_palette_key,
                        settings=settings,
                        output_dir=output_dir,
                    )
                )

        except Exception as exc:
            LOG.warning(
                "Failed semantics debug for frame %d: %s", frame_idx, exc, exc_info=True
            )

    if palette_entries:
        generated_paths.append(
            update_palette_manifest(
                store.visualizations_dir() / "semantics_palette.json",
                palette_entries,
            )
        )

    return generated_paths


def _write_temporal_fusion_debug(
    *,
    store: ResourceStore,
    frame_idx: int,
    frame_data,
    semantics: SemanticsData,
    fused_label_ids: np.ndarray,
    label_map: Mapping[int, str],
    id_to_palette_key: Mapping[int, str],
    base_output_dir: Path,
    settings: SemanticsDebugSettings,
    spec: _TemporalFusionDebugSpec,
) -> List[Path]:
    outputs: List[Path] = []
    fusion_root = base_output_dir / "temporal_fusion"

    fused_dir = fusion_root / "fused"
    outputs.extend(
        _write_label_debug(
            frame_idx=frame_idx,
            frame_data=frame_data,
            label_ids=fused_label_ids,
            label_map=label_map,
            id_to_palette_key=id_to_palette_key,
            settings=settings,
            output_dir=fused_dir,
        )
    )
    fused_road_conf = _load_fused_road_confidence(store, semantics, frame_idx)
    outputs.extend(
        _write_confidence_debug(
            frame_idx=frame_idx,
            frame_data=frame_data,
            confidence=fused_road_conf,
            output_dir=fused_dir,
            overlay_alpha=settings.overlay_alpha,
            name="road_confidence",
        )
    )
    fused_debug_maps = _load_fused_confidence_debug_maps(store, semantics, frame_idx)
    debug_layers = (
        "road_consensus_prob",
        "road_semantic_prob",
        "road_prior_prob",
        "road_agreement",
        "road_disagreement",
        "road_jsd",
        "road_logop_prob",
    )
    for layer_name in debug_layers:
        outputs.extend(
            _write_confidence_debug(
                frame_idx=frame_idx,
                frame_data=frame_data,
                confidence=fused_debug_maps.get(layer_name),
                output_dir=fused_dir,
                overlay_alpha=settings.overlay_alpha,
                name=layer_name,
            )
        )

    model_outputs = _load_model_debug_outputs(store, frame_idx)
    for model_name in spec.model_names:
        model_output = model_outputs.get(model_name)
        if model_output is None:
            LOG.warning(
                "Temporal fusion debug missing standardized model output for '%s'",
                model_name,
            )
            continue
        model_dir = fusion_root / "models" / _safe_path_component(model_name)
        outputs.extend(
            _write_label_debug(
                frame_idx=frame_idx,
                frame_data=frame_data,
                label_ids=model_output.label_ids,
                label_map=label_map,
                id_to_palette_key=_palette_keys_from_label_map(label_map),
                settings=settings,
                output_dir=model_dir,
            )
        )
        outputs.extend(
            _write_confidence_debug(
                frame_idx=frame_idx,
                frame_data=frame_data,
                confidence=(
                    model_output.road_confidence
                    if model_output.road_confidence is not None
                    else model_output.confidence
                ),
                output_dir=model_dir,
                overlay_alpha=settings.overlay_alpha,
                name=(
                    "road_confidence"
                    if model_output.road_confidence is not None
                    else "confidence"
                ),
            )
        )

    for road_prior in spec.road_priors:
        road_conf = _load_road_prior_output(store, frame_idx, road_prior.name)
        if road_conf is None:
            LOG.warning(
                "Temporal fusion debug missing standardized road prior for %s",
                road_prior.name,
            )
            continue
        prior_dir = fusion_root / "road_priors" / _safe_path_component(road_prior.name)
        outputs.extend(
            _write_road_prior_debug(
                frame_idx=frame_idx,
                frame_data=frame_data,
                road_conf=road_conf,
                output_dir=prior_dir,
                overlay_alpha=settings.overlay_alpha,
            )
        )

    return outputs


def _write_label_debug(
    *,
    frame_idx: int,
    frame_data,
    label_ids: np.ndarray,
    label_map: Mapping[int, str],
    id_to_palette_key: Mapping[int, str],
    settings: SemanticsDebugSettings,
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    stats = _semantic_label_stats(label_ids, label_map, settings.road_label_tokens)
    candidate = _road_candidate_from_region(label_ids)
    if candidate is not None:
        stats["road_candidate_id"] = int(candidate)
        stats["road_candidate_name"] = label_map.get(
            int(candidate), f"class_{int(candidate)}"
        )

    stats_path = output_dir / f"{frame_idx:06d}_labels.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    generated.append(stats_path)

    colorized = _colorize_labels(label_ids, id_to_palette_key=id_to_palette_key)
    colorized = _annotate_label_names(
        colorized, label_ids, label_map, settings.min_segment_area
    )
    label_path = output_dir / f"{frame_idx:06d}_labels.png"
    save_rgb_image(label_path, colorized)
    generated.append(label_path)

    road_mask = _road_mask(label_ids, label_map, settings.road_label_tokens)
    if road_mask is not None and road_mask.any():
        road_path = output_dir / f"{frame_idx:06d}_road_mask.png"
        road_rgb = np.stack([road_mask.astype(np.uint8) * 255] * 3, axis=-1)
        save_rgb_image(road_path, road_rgb)
        generated.append(road_path)

        if frame_data.image is not None:
            overlay = _overlay_mask(frame_data.image, road_mask, settings.overlay_alpha)
            if overlay is not None:
                overlay_path = output_dir / f"{frame_idx:06d}_road_overlay.png"
                save_rgb_image(overlay_path, overlay)
                generated.append(overlay_path)

    if candidate is not None:
        cand_mask = label_ids == int(candidate)
        cand_rgb = np.stack([cand_mask.astype(np.uint8) * 255] * 3, axis=-1)
        cand_path = output_dir / f"{frame_idx:06d}_road_candidate.png"
        save_rgb_image(cand_path, cand_rgb)
        generated.append(cand_path)

    return generated


def _write_confidence_debug(
    *,
    frame_idx: int,
    frame_data,
    confidence: Optional[np.ndarray],
    output_dir: Path,
    overlay_alpha: float,
    name: str = "confidence",
) -> List[Path]:
    if confidence is None:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    heatmap = _probability_heatmap_rgb(confidence)
    heatmap_path = output_dir / f"{frame_idx:06d}_{name}.png"
    save_rgb_image(heatmap_path, heatmap)
    generated.append(heatmap_path)

    if frame_data.image is not None:
        overlay = _blend_images(frame_data.image, heatmap, overlay_alpha)
        if overlay is not None:
            overlay_path = output_dir / f"{frame_idx:06d}_{name}_overlay.png"
            save_rgb_image(overlay_path, overlay)
            generated.append(overlay_path)

    return generated


def _write_road_prior_debug(
    *,
    frame_idx: int,
    frame_data,
    road_conf: np.ndarray,
    output_dir: Path,
    overlay_alpha: float,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    heatmap = _probability_heatmap_rgb(road_conf)
    heatmap_path = output_dir / f"{frame_idx:06d}_road_prior.png"
    save_rgb_image(heatmap_path, heatmap)
    generated.append(heatmap_path)

    if frame_data.image is not None:
        overlay = _blend_images(frame_data.image, heatmap, overlay_alpha)
        if overlay is not None:
            overlay_path = output_dir / f"{frame_idx:06d}_road_prior_overlay.png"
            save_rgb_image(overlay_path, overlay)
            generated.append(overlay_path)

    return generated


def _resolve_label_ids(
    semantics: SemanticsData,
    frame_image: Optional[np.ndarray],
    min_segment_area: int,
) -> Tuple[Optional[np.ndarray], Dict[int, str], Dict[int, str]]:
    label_map: Dict[int, str] = {}
    id_to_palette_key: Dict[int, str] = {}

    if _has_raster_map(semantics.label_ids):
        label_ids = np.asarray(semantics.label_ids, dtype=np.int32)
        _populate_label_map_from_segments(label_map, semantics.segments)
        for raster_id, label in label_map.items():
            id_to_palette_key[int(raster_id)] = semantic_palette_key(
                label_id=int(raster_id),
                label=label,
                segment_id=None,
            )
        return label_ids, label_map, id_to_palette_key

    if _has_raster_map(semantics.segment_ids):
        segment_ids = np.asarray(semantics.segment_ids, dtype=np.int32)
        _populate_label_map_from_segments(label_map, semantics.segments, use_label_id=False)
        for segment in semantics.segments:
            if segment.is_empty:
                continue
            id_to_palette_key[int(segment.segment_id)] = semantic_palette_key(
                label_id=segment.label_id,
                label=segment.label,
                segment_id=segment.segment_id,
            )
        return segment_ids, label_map, id_to_palette_key

    shape = _infer_shape(frame_image, semantics.segments)
    if shape is None:
        return None, {}, {}

    label_ids = np.full(shape, fill_value=-1, dtype=np.int32)
    segments = _sorted_segments(semantics.segments, min_segment_area)
    for segment in segments:
        label_id = segment.label_id if segment.label_id is not None else segment.segment_id
        label_map.setdefault(int(label_id), str(segment.label))
        id_to_palette_key.setdefault(
            int(label_id),
            semantic_palette_key(
                label_id=segment.label_id,
                label=segment.label,
                segment_id=segment.segment_id,
            ),
        )
        mask = np.asarray(segment.mask, dtype=bool)
        if mask.shape != label_ids.shape:
            continue
        label_ids[mask & (label_ids == -1)] = int(label_id)

    return label_ids, label_map, id_to_palette_key


def _infer_shape(
    frame_image: Optional[np.ndarray],
    segments: Sequence[SemanticSegment],
) -> Optional[Tuple[int, int]]:
    if frame_image is not None:
        return int(frame_image.shape[0]), int(frame_image.shape[1])
    for segment in segments:
        if segment.mask is not None and segment.mask.size > 0:
            return int(segment.mask.shape[0]), int(segment.mask.shape[1])
    return None


def _sorted_segments(
    segments: Sequence[SemanticSegment],
    min_segment_area: int,
) -> List[SemanticSegment]:
    filtered = []
    for segment in segments:
        if segment.is_empty:
            continue
        area = segment.area if segment.area is not None else int(np.sum(segment.mask))
        if area < min_segment_area:
            continue
        filtered.append(segment)
    return sorted(filtered, key=lambda seg: (seg.score, seg.segment_id), reverse=True)


def _populate_label_map_from_segments(
    label_map: Dict[int, str],
    segments: Iterable[SemanticSegment],
    use_label_id: bool = True,
) -> None:
    for segment in segments:
        if segment.is_empty:
            continue
        label_id = segment.label_id if use_label_id and segment.label_id is not None else segment.segment_id
        label_map.setdefault(int(label_id), str(segment.label))



def _semantic_label_stats(
    label_ids: np.ndarray,
    label_map: Mapping[int, str],
    road_tokens: Sequence[str],
) -> Dict[str, Any]:
    label_ids = np.asarray(label_ids, dtype=np.int32)
    unique, counts = np.unique(label_ids, return_counts=True)
    total = int(label_ids.size)
    labels: Dict[str, Any] = {}
    for lid, count in zip(unique.tolist(), counts.tolist()):
        if int(lid) < 0:
            continue
        name = label_map.get(int(lid), f"class_{int(lid)}")
        labels[str(lid)] = {"name": name, "count": int(count), "fraction": float(count / max(1, total))}
    road_ids = _road_label_ids_from_map(label_map, road_tokens)
    road_pixels = int(np.isin(label_ids, road_ids).sum())
    return {
        "total_pixels": total,
        "road_pixels": road_pixels,
        "road_fraction": float(road_pixels / max(1, total)),
        "labels": labels,
    }


def _road_label_ids_from_map(
    label_map: Mapping[int, str],
    road_tokens: Sequence[str],
) -> List[int]:
    road_ids = []
    for label_id, label in label_map.items():
        token = str(label).lower()
        if any(tok in token for tok in road_tokens):
            road_ids.append(int(label_id))
    return road_ids


def _road_mask(
    label_ids: np.ndarray,
    label_map: Mapping[int, str],
    road_tokens: Sequence[str],
) -> Optional[np.ndarray]:
    road_ids = _road_label_ids_from_map(label_map, road_tokens)
    if not road_ids:
        return None
    return np.isin(label_ids, road_ids)


def _road_candidate_from_region(label_ids: np.ndarray) -> Optional[int]:
    label_ids = np.asarray(label_ids, dtype=np.int32)
    if label_ids.ndim != 2:
        return None
    h, w = label_ids.shape
    y0 = int(h * 0.7)
    x0 = int(w * 0.3)
    x1 = int(w * 0.7)
    region = label_ids[y0:, x0:x1]
    if region.size == 0:
        return None
    region = region[region >= 0]
    if region.size == 0:
        return None
    unique, counts = np.unique(region, return_counts=True)
    if unique.size == 0:
        return None
    return int(unique[int(np.argmax(counts))])


def _colorize_labels(
    label_ids: np.ndarray,
    *,
    id_to_palette_key: Mapping[int, str],
) -> np.ndarray:
    return colorize_semantic_raster(
        label_ids,
        id_to_palette_key=id_to_palette_key,
    )


def _annotate_label_names(
    image: np.ndarray,
    label_ids: np.ndarray,
    label_map: Mapping[int, str],
    min_segment_area: int,
) -> np.ndarray:
    annotated = np.asarray(image).copy()
    label_ids = np.asarray(label_ids, dtype=np.int32)
    unique_ids = np.unique(label_ids)
    for lid in unique_ids:
        lid_int = int(lid)
        if lid_int < 0:
            continue
        mask = label_ids == lid_int
        area = int(np.sum(mask))
        if area < min_segment_area:
            continue
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        label = label_map.get(lid_int, f"class_{lid_int}")
        _draw_text_with_outline(annotated, label, (cx, cy))
    return annotated


def _draw_text_with_outline(image: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    outline = 3
    cv2.putText(
        image,
        text,
        origin,
        font,
        scale,
        (0, 0, 0),
        outline,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    if image is None:
        return None
    img = _ensure_rgb(image)
    if img is None:
        return None
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape[:2] != img.shape[:2]:
        return None
    overlay = img.copy()
    overlay[mask_arr] = overlay[mask_arr] * (1 - alpha) + np.array([255, 0, 0], dtype=np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _ensure_rgb(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None:
        return None
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] >= 3:
        return img[:, :, :3].astype(np.float32)
    return None


def _blend_images(image: np.ndarray, overlay: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    base = _ensure_rgb(image)
    if base is None:
        return None
    overlay_arr = np.asarray(overlay)
    if overlay_arr.ndim == 2:
        overlay_arr = np.stack([overlay_arr] * 3, axis=-1)
    if overlay_arr.shape[:2] != base.shape[:2]:
        return None
    if overlay_arr.shape[2] >= 3:
        overlay_rgb = overlay_arr[:, :, :3].astype(np.float32)
    else:
        return None
    blended = base * (1 - alpha) + overlay_rgb * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _probability_heatmap_rgb(probabilities: np.ndarray) -> np.ndarray:
    heat = np.clip(probabilities, 0.0, 1.0)
    heat = (heat * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _load_model_debug_outputs(
    store: ResourceStore,
    frame_idx: int,
) -> Dict[str, _ModelDebugOutput]:
    try:
        aux = store.load_semantics_aux(frame_idx)
    except Exception:
        return {}
    outputs: Dict[str, _ModelDebugOutput] = {}
    for name, payload in aux.model_outputs.items():
        if not isinstance(payload, Mapping) or "label_ids" not in payload:
            continue
        outputs[str(name)] = _ModelDebugOutput(
            label_ids=np.asarray(payload["label_ids"], dtype=np.int32),
            confidence=(
                np.asarray(payload["confidence"], dtype=np.float32)
                if "confidence" in payload
                else None
            ),
            road_confidence=(
                np.asarray(payload["road_confidence"], dtype=np.float32)
                if "road_confidence" in payload
                else None
            ),
            validity_mask=(
                np.asarray(payload["validity_mask"], dtype=bool)
                if "validity_mask" in payload
                else None
            ),
        )
    return outputs


def _load_road_prior_output(
    store: ResourceStore,
    frame_idx: int,
    name: str,
) -> Optional[np.ndarray]:
    try:
        aux = store.load_semantics_aux(frame_idx)
    except Exception:
        return None
    if name not in aux.road_prior_outputs:
        return None
    return np.clip(np.asarray(aux.road_prior_outputs[name], dtype=np.float32), 0.0, 1.0)


def _load_fused_road_confidence(
    store: ResourceStore,
    semantics: SemanticsData,
    frame_idx: int,
) -> Optional[np.ndarray]:
    try:
        aux = store.load_semantics_aux(frame_idx)
    except Exception:
        aux = None
    if aux is not None and aux.road_confidence is not None:
        return np.clip(np.asarray(aux.road_confidence, dtype=np.float32), 0.0, 1.0)
    debug_maps = _load_fused_confidence_debug_maps(store, semantics, frame_idx)
    conf = debug_maps.get("road_confidence")
    if conf is not None:
        return conf
    return debug_maps.get("road_consensus_prob")


def _load_fused_confidence_debug_maps(
    store: ResourceStore,
    semantics: SemanticsData,
    frame_idx: int,
) -> Dict[str, np.ndarray]:
    try:
        aux = store.load_semantics_aux(frame_idx)
    except Exception:
        return {}
    return {
        str(key): np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
        for key, value in aux.debug_maps.items()
        if np.asarray(value).ndim == 2
    }


def _resolve_temporal_fusion_debug_spec(
    store: ResourceStore, semantics: SemanticsData
) -> Optional[_TemporalFusionDebugSpec]:
    if not _is_temporal_fusion_semantics(semantics):
        return None
    metadata = semantics.metadata or {}
    model_names = _coerce_str_list(metadata.get("models"))
    road_prior_names = _coerce_str_list(metadata.get("road_prior_models"))
    if not model_names or not road_prior_names:
        try:
            aux = store.load_semantics_aux(int(semantics.frame_index))
        except Exception:
            aux = None
        if aux is not None:
            if not model_names:
                model_names = sorted(str(name) for name in aux.model_outputs.keys())
            if not road_prior_names:
                road_prior_names = sorted(str(name) for name in aux.road_prior_outputs.keys())

    return _TemporalFusionDebugSpec(
        model_names=tuple(model_names),
        road_priors=tuple(_RoadPriorDebugSpec(name=name) for name in road_prior_names),
    )


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _is_temporal_fusion_semantics(semantics: SemanticsData) -> bool:
    if not semantics.metadata:
        return False
    return str(semantics.metadata.get("source", "")).strip().lower() == "temporal_fusion"


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned or "model"


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_raster_map(value: Optional[np.ndarray]) -> bool:
    if value is None:
        return False
    arr = np.asarray(value)
    if arr.shape == () and arr.dtype == object and arr.item() is None:
        return False
    return True


def _palette_keys_from_label_map(label_map: Mapping[int, str]) -> Dict[int, str]:
    return {
        int(raster_id): semantic_palette_key(
            label_id=int(raster_id),
            label=label,
            segment_id=None,
        )
        for raster_id, label in label_map.items()
        if int(raster_id) >= 0
    }
