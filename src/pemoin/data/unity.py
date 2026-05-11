"""Unity export dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import json


_UNITY_REFLECTION_FACE_KEYS = (
    "PositiveX",
    "NegativeX",
    "PositiveY",
    "NegativeY",
    "PositiveZ",
    "NegativeZ",
)


@dataclass(frozen=True)
class UnityLightingFrameRecord:
    frame_index: int
    timestamp_sec: Optional[float]
    payload: Mapping[str, object]


class UnityLightingDataset:
    def __init__(self, root: Path) -> None:
        self.root = _normalize_unity_root(root)
        self.lighting_root = _resolve_lighting_root(self.root)
        self._run_lighting = _load_optional_json(self.lighting_root / "run_lighting.json")
        self._scene_lights = _load_optional_json(self.lighting_root / "scene_lights.json")
        self._frame_lighting = _load_jsonl_records(self.lighting_root / "frame_lighting.jsonl")
        self._reflection_faces = _load_reflection_faces(self.lighting_root / "reflection_probe_faces")

    def has_lighting_gt(self) -> bool:
        return bool(self._run_lighting) and bool(self._scene_lights) and bool(self._reflection_faces)

    def run_lighting(self) -> Mapping[str, object]:
        return dict(self._run_lighting)

    def scene_lights(self) -> Mapping[str, object]:
        return dict(self._scene_lights)

    def frame_indices(self) -> List[int]:
        return sorted(self._frame_lighting.keys())

    def frame_lighting(self, frame_index: int) -> Mapping[str, object]:
        try:
            return dict(self._frame_lighting[int(frame_index)].payload)
        except KeyError as exc:
            raise KeyError(f"Unity frame lighting {frame_index} is missing.") from exc

    def has_frame_lighting(self) -> bool:
        return bool(self._frame_lighting)

    def reflection_faces(self) -> Dict[str, Path]:
        return dict(self._reflection_faces)


def resolve_unity_lighting_dataset(
    settings: Mapping[str, object],
    context: Dict[str, object],
) -> UnityLightingDataset:
    root = settings.get("path") or settings.get("root")
    if not root:
        root = context.get("frame_source")
    if not root:
        raise ValueError(
            "Unity lighting providers require a 'path' setting or a frame_source in the provider context."
        )
    root_path = Path(str(root)).expanduser()
    cache_key = f"unity_lighting_dataset::{root_path}"
    cached = context.get(cache_key)
    if isinstance(cached, UnityLightingDataset):
        return cached
    dataset = UnityLightingDataset(root_path)
    context[cache_key] = dataset
    return dataset


def _normalize_unity_root(root: Path) -> Path:
    root = root.expanduser()
    if root.is_file():
        root = root.parent
    if not root.exists():
        raise FileNotFoundError(f"Unity export directory '{root}' does not exist.")
    if not root.is_dir():
        raise NotADirectoryError(f"Unity export root '{root}' must be a directory.")
    return root


def _resolve_lighting_root(root: Path) -> Path:
    if root.name == "lighting_gt" and root.is_dir():
        return root
    candidates = [root / "lighting_gt"]
    if root.parent.is_dir():
        candidates.append(root.parent / "lighting_gt")
    if root.name.startswith("sequence.") and root.parent.is_dir() and root.parent.parent.is_dir():
        candidates.append(root.parent.parent / "lighting_gt")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Unity lighting_gt directory not found under '{root}'.")


def _load_optional_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unity JSON file is invalid: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Unity JSON file must contain an object: {path}")
    return raw


def _load_jsonl_records(path: Path) -> Dict[int, UnityLightingFrameRecord]:
    if not path.exists():
        return {}
    records: Dict[int, UnityLightingFrameRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Unity JSONL record must be an object in {path}.")
        frame_index = int(payload.get("frameIndex", -1))
        if frame_index < 0:
            continue
        records[frame_index] = UnityLightingFrameRecord(
            frame_index=frame_index,
            timestamp_sec=_as_float(payload.get("timestampSec")),
            payload=payload,
        )
    return records


def _load_reflection_faces(root: Path) -> Dict[str, Path]:
    faces: Dict[str, Path] = {}
    if not root.exists():
        return faces
    for path in sorted(root.glob("*.exr")):
        for face in _UNITY_REFLECTION_FACE_KEYS:
            if path.stem.endswith(face):
                faces[face] = path
                break
    return faces


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
