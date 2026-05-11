"""Shared semantic palette helpers for visualization outputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import cv2
import numpy as np

from pemoin.data.models import SemanticSegment, SemanticsData

PALETTE_VERSION = 1
UNKNOWN_SEMANTIC_KEY = "unknown"
UNKNOWN_SEMANTIC_LABEL = "unknown"
_SPACE_RE = re.compile(r"\s+")


def normalize_semantic_label(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return UNKNOWN_SEMANTIC_LABEL
    return _SPACE_RE.sub(" ", text)


def semantic_palette_key(
    *,
    label_id: Optional[int],
    label: Any,
    segment_id: Optional[int],
) -> str:
    if label_id is not None:
        return f"label_id:{int(label_id)}"
    normalized_label = normalize_semantic_label(label)
    if normalized_label != UNKNOWN_SEMANTIC_LABEL:
        return f"label:{normalized_label}"
    if segment_id is not None:
        return f"segment_id:{int(segment_id)}"
    return UNKNOWN_SEMANTIC_KEY


def semantic_color_for_key(key: str) -> np.ndarray:
    hash_bytes = hashlib.sha256(str(key).encode("utf-8")).digest()
    hue = int.from_bytes(hash_bytes[0:4], byteorder="big") % 180
    saturation = 178 + (int.from_bytes(hash_bytes[4:8], byteorder="big") % 78)
    value = 178 + (int.from_bytes(hash_bytes[8:12], byteorder="big") % 78)
    hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
    return rgb_color[0, 0, :].astype(np.uint8)


def semantic_color_for_segment(segment: SemanticSegment) -> np.ndarray:
    return semantic_color_for_key(
        semantic_palette_key(
            label_id=segment.label_id,
            label=segment.label,
            segment_id=segment.segment_id,
        )
    )


def semantic_palette_entries_from_semantics(
    semantics: SemanticsData,
) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for segment in semantics.segments:
        if segment.is_empty:
            continue
        key = semantic_palette_key(
            label_id=segment.label_id,
            label=segment.label,
            segment_id=segment.segment_id,
        )
        _add_palette_entry(entries, key=key, display_label=str(segment.label))
    return entries


def semantic_palette_entries_from_id_map(
    *,
    id_to_palette_key: Mapping[int, str],
    label_map: Mapping[int, str],
) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for raster_id, key in id_to_palette_key.items():
        if int(raster_id) < 0:
            continue
        _add_palette_entry(
            entries,
            key=key,
            display_label=str(label_map.get(int(raster_id), f"class_{int(raster_id)}")),
        )
    return entries


def colorize_semantic_raster(
    raster_ids: np.ndarray,
    *,
    id_to_palette_key: Mapping[int, str],
    unknown_color: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    raster = np.asarray(raster_ids, dtype=np.int32)
    h, w = raster.shape[:2]
    colors = np.zeros((h, w, 3), dtype=np.uint8)
    colors[:] = np.asarray(unknown_color, dtype=np.uint8)
    for raster_id in np.unique(raster):
        raster_id_int = int(raster_id)
        if raster_id_int < 0:
            continue
        key = id_to_palette_key.get(raster_id_int, f"label_id:{raster_id_int}")
        colors[raster == raster_id_int] = semantic_color_for_key(key)
    return colors


def update_palette_manifest(
    path: Path,
    entries: Mapping[str, Mapping[str, Any]],
) -> Path:
    manifest: MutableMapping[str, Any] = {
        "palette_version": PALETTE_VERSION,
        "entries": {},
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, Mapping):
            manifest["palette_version"] = int(existing.get("palette_version", PALETTE_VERSION))
            raw_entries = existing.get("entries", {})
            if isinstance(raw_entries, Mapping):
                manifest["entries"] = {
                    str(key): dict(value)
                    for key, value in raw_entries.items()
                    if isinstance(value, Mapping)
                }

    merged_entries = dict(manifest["entries"])
    for key, value in entries.items():
        entry = dict(value)
        entry.setdefault("color_rgb", semantic_color_for_key(str(key)).tolist())
        entry.setdefault("display_label", UNKNOWN_SEMANTIC_LABEL)
        merged_entries[str(key)] = entry

    manifest["entries"] = dict(sorted(merged_entries.items(), key=lambda item: item[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _add_palette_entry(
    entries: MutableMapping[str, Dict[str, Any]],
    *,
    key: str,
    display_label: str,
) -> None:
    entries[str(key)] = {
        "display_label": normalize_semantic_label(display_label),
        "color_rgb": semantic_color_for_key(str(key)).tolist(),
    }
