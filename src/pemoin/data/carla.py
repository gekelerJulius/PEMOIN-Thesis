"""CARLA export dataset helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import json
import numpy as np


@dataclass(frozen=True)
class CarlaFrameRecord:
    frame: int
    timestamp: Optional[float]
    rgb_path: Path
    depth_path: Path
    semseg_path: Optional[Path]
    instseg_path: Optional[Path]
    world_from_camera: np.ndarray


@dataclass(frozen=True)
class CarlaLightingRecord:
    frame: int
    timestamp: Optional[float]
    payload: Mapping[str, object]


class CarlaDataset:
    def __init__(self, root: Path) -> None:
        self.root = _normalize_root(root)
        self._frames: Dict[int, CarlaFrameRecord] = _load_frames_jsonl(self.root)
        if not self._frames:
            raise FileNotFoundError(f"CARLA export contains no frames under {self.root}.")
        self._intrinsics = _load_intrinsics(self.root)
        self._run_config = _load_run_config(self.root)
        self._run_lighting = _load_optional_json(self.root / "lighting_gt" / "run_lighting.json")
        self._scene_lights = _load_optional_json(self.root / "lighting_gt" / "scene_lights.json")
        self._frame_lighting = _load_jsonl_records(self.root / "lighting_gt" / "frame_lighting.jsonl")

    def frame_indices(self) -> List[int]:
        return sorted(self._frames.keys())

    def frame(self, frame_index: int) -> CarlaFrameRecord:
        try:
            return self._frames[int(frame_index)]
        except KeyError as exc:
            raise KeyError(f"CARLA frame {frame_index} is missing.") from exc

    def intrinsics(self) -> Mapping[str, float]:
        return dict(self._intrinsics)

    def run_config(self) -> Mapping[str, object]:
        return dict(self._run_config)

    def frame_rate(self) -> Optional[float]:
        fps = self._run_config.get("fps")
        if fps is not None:
            try:
                return float(fps)
            except (TypeError, ValueError):
                return None
        return _estimate_frame_rate(list(self._frames.values()))

    def run_lighting(self) -> Mapping[str, object]:
        return dict(self._run_lighting)

    def scene_lights(self) -> Mapping[str, object]:
        return dict(self._scene_lights)

    def frame_lighting(self, frame_index: int) -> Mapping[str, object]:
        try:
            return dict(self._frame_lighting[int(frame_index)].payload)
        except KeyError as exc:
            raise KeyError(f"CARLA frame lighting {frame_index} is missing.") from exc

    def has_lighting_gt(self) -> bool:
        return bool(self._run_lighting) and bool(self._scene_lights) and bool(self._frame_lighting)


def resolve_carla_dataset(settings: Mapping[str, object], context: Dict[str, object]) -> CarlaDataset:
    root = settings.get("path") or settings.get("root")
    if not root:
        root = context.get("frame_source")
    if not root:
        raise ValueError(
            "CARLA providers require a 'path' setting or a frame_source in the provider context."
        )
    root_path = Path(str(root)).expanduser()
    cache_key = f"carla_dataset::{root_path}"
    cached = context.get(cache_key)
    if isinstance(cached, CarlaDataset):
        return cached
    dataset = CarlaDataset(root_path)
    context[cache_key] = dataset
    return dataset


def _normalize_root(root: Path) -> Path:
    root = root.expanduser()
    if root.is_file():
        root = root.parent
    if not root.exists():
        raise FileNotFoundError(f"CARLA export directory '{root}' does not exist.")
    if not root.is_dir():
        raise NotADirectoryError(f"CARLA export root '{root}' must be a directory.")
    return root


def _load_intrinsics(root: Path) -> Mapping[str, float]:
    path = root / "camera_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(f"CARLA intrinsics missing: {path}")
    raw = json.loads(path.read_text())
    required = ["fx", "fy", "cx", "cy", "width", "height"]
    for key in required:
        if key not in raw:
            raise ValueError(f"CARLA intrinsics missing '{key}'.")
    return {key: float(raw[key]) for key in required}


def _load_run_config(root: Path) -> Mapping[str, object]:
    path = root / "run_config.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"CARLA run_config.json is invalid: {path}") from exc
    return raw if isinstance(raw, dict) else {}


def _load_optional_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"CARLA JSON file is invalid: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"CARLA JSON file must contain an object: {path}")
    return raw


def _load_frames_jsonl(root: Path) -> Dict[int, CarlaFrameRecord]:
    frames_path = root / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"CARLA frames.jsonl missing: {frames_path}")
    frames: Dict[int, CarlaFrameRecord] = {}
    for line in frames_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        frame_index = int(record.get("frame", -1))
        if frame_index < 0:
            continue
        rgb = record.get("rgb")
        depth = record.get("depth_m")
        if not rgb or not depth:
            continue
        world_from_camera = np.asarray(record.get("T_world_from_camera"), dtype=np.float32)
        if world_from_camera.shape != (4, 4):
            raise ValueError(f"CARLA T_world_from_camera has unexpected shape {world_from_camera.shape}.")
        frames[frame_index] = CarlaFrameRecord(
            frame=frame_index,
            timestamp=_as_float(record.get("timestamp")),
            rgb_path=root / str(rgb),
            depth_path=root / str(depth),
            semseg_path=(root / str(record["semseg_id"])) if record.get("semseg_id") else None,
            instseg_path=(root / str(record["instseg_id"])) if record.get("instseg_id") else None,
            world_from_camera=world_from_camera,
        )
    return frames


def _load_jsonl_records(path: Path) -> Dict[int, CarlaLightingRecord]:
    if not path.exists():
        return {}
    records: Dict[int, CarlaLightingRecord] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"CARLA JSONL record must be an object in {path}.")
        frame_index = int(payload.get("frame", -1))
        if frame_index < 0:
            continue
        records[frame_index] = CarlaLightingRecord(
            frame=frame_index,
            timestamp=_as_float(payload.get("timestamp")),
            payload=payload,
        )
    return records


def _estimate_frame_rate(records: Sequence[CarlaFrameRecord]) -> Optional[float]:
    if len(records) < 2:
        return None
    timestamps = [rec.timestamp for rec in records if rec.timestamp is not None]
    if len(timestamps) < 2:
        return None
    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b is not None and a is not None and b > a]
    if not deltas:
        return None
    median_delta = float(np.median(np.asarray(deltas, dtype=np.float32)))
    if median_delta <= 0:
        return None
    return 1.0 / median_delta


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
