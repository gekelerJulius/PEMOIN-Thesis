"""
Frame provider for Unity Perception dataset exports.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

import imageio.v3 as iio
import numpy as np

from pemoin.data.contracts import FrameData
from .frame_provider import FrameProvider


@dataclass(frozen=True)
class _UnityFrameEntry:
    index: int
    timestamp: float
    image_path: Optional[Path]
    frame_id: str


class UnityFrameProvider(FrameProvider):
    """
    Streams frames from a Unity Perception sequence directory.

    Uses frame_data.json to preserve frame indices and timestamps.
    """

    def __init__(self, settings: Mapping[str, object], *, load_images: bool = True) -> None:
        self._settings = dict(settings)
        self._entries: List[_UnityFrameEntry] = []
        self._cursor = 0
        self._opened = False
        self._root: Optional[Path] = None
        self._load_images = bool(load_images)
        self._frame_rate: Optional[float] = None
        self._stride = 1

    def open(self, source) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Unity data directory '{path}' does not exist.")
        if not path.is_dir():
            raise NotADirectoryError(f"Unity data source '{path}' must be a directory.")

        sequence_dir = self._resolve_sequence_dir(path)
        entries = []
        for json_path in sorted(sequence_dir.glob("step*.frame_data.json")):
            payload = _load_json(json_path)
            step = int(payload.get("step", -1))
            if step < 0:
                continue
            timestamp = float(payload.get("timestamp", 0.0))
            capture = _pick_camera_capture(payload.get("captures", []))
            if capture is None:
                continue
            filename = capture.get("filename")
            img_path = sequence_dir / str(filename) if filename else None
            frame_id = str(filename) if filename else str(step).zfill(6)
            if self._load_images and img_path is None:
                raise ValueError(f"Unity frame {step} is missing an image filename.")
            entries.append(
                _UnityFrameEntry(
                    index=step,
                    timestamp=timestamp,
                    image_path=img_path,
                    frame_id=frame_id,
                )
            )
        if not entries:
            raise FileNotFoundError(f"No Unity frame data found under '{sequence_dir}'.")
        entries = sorted(entries, key=lambda entry: entry.index)
        start_frame = self._settings.get("start_frame")
        end_frame = self._settings.get("end_frame")
        if start_frame is not None:
            start_frame = int(start_frame)
        if end_frame is not None:
            end_frame = int(end_frame)
        if start_frame is not None and start_frame < 0:
            raise ValueError("start_frame must be >= 0.")
        if end_frame is not None and end_frame < 0:
            raise ValueError("end_frame must be >= 0.")
        if start_frame is not None and end_frame is not None and end_frame < start_frame:
            raise ValueError("end_frame must be >= start_frame.")
        if start_frame is not None:
            entries = [entry for entry in entries if entry.index >= start_frame]
        if end_frame is not None:
            entries = [entry for entry in entries if entry.index <= end_frame]
        if not entries:
            raise FileNotFoundError("No Unity frames found after applying start_frame/end_frame.")
        frame_rate = self._settings.get("frame_rate") or self._settings.get("fps")
        self._frame_rate = float(frame_rate) if frame_rate is not None else _estimate_frame_rate(entries)
        sampling_fps = self._settings.get("sampling_fps")
        if sampling_fps is not None:
            if self._frame_rate is None or self._frame_rate <= 0:
                raise ValueError("sampling_fps requires a valid frame_rate for UnityFrameProvider.")
            sampling_fps = float(sampling_fps)
            if sampling_fps <= 0:
                raise ValueError("sampling_fps must be > 0.")
            self._stride = max(1, int(round(self._frame_rate / sampling_fps)))
            entries = entries[:: self._stride]
        self._entries = entries
        self._cursor = 0
        self._opened = True
        self._root = sequence_dir

    def read(self) -> Optional[FrameData]:
        if not self._opened:
            raise RuntimeError("UnityFrameProvider.open must be called before reading frames.")
        if self._cursor >= len(self._entries):
            return None
        entry = self._entries[self._cursor]
        self._cursor += 1
        image = None
        if self._load_images:
            if entry.image_path is None:
                raise FileNotFoundError(f"Unity RGB frame missing for index {entry.index}.")
            image = np.asarray(iio.imread(entry.image_path))
        frame = FrameData(
            frame_id=entry.frame_id,
            index=int(entry.index),
            timestamp=float(entry.timestamp),
            image=image,
            metadata={
                "source_path": str(entry.image_path) if entry.image_path is not None else None,
                "frame_stride": self._stride,
                "sampling_fps": self._settings.get("sampling_fps"),
            },
        )
        return frame

    def __len__(self) -> int:
        return len(self._entries)

    def close(self) -> None:
        self._entries = []
        self._cursor = 0
        self._opened = False
        self._root = None

    def runtime_settings(self) -> Mapping[str, object]:
        resolved_sampling_fps = None
        if self._frame_rate is not None and self._frame_rate > 0:
            resolved_sampling_fps = float(self._frame_rate) / float(max(1, self._stride))
        return {
            "frame_rate": float(self._frame_rate) if self._frame_rate is not None else None,
            "sampling_fps": self._settings.get("sampling_fps"),
            "resolved_sampling_fps": resolved_sampling_fps,
            "frame_stride": int(self._stride),
        }

    def _resolve_sequence_dir(self, root: Path) -> Path:
        if root.name.startswith("sequence.") and root.is_dir():
            return root
        candidates = sorted(root.glob("sequence.*"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Unity sequence directory not found under {root}.")


def _load_json(path: Path):
    import json

    return json.loads(path.read_text())


def _pick_camera_capture(captures):
    for cap in captures:
        if cap.get("id") == "camera":
            return cap
    return None


def _estimate_frame_rate(entries: List[_UnityFrameEntry]) -> Optional[float]:
    if len(entries) < 2:
        return None
    timestamps = [entry.timestamp for entry in entries if entry.timestamp is not None]
    if len(timestamps) < 2:
        return None
    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    if not deltas:
        return None
    median_delta = float(np.median(np.asarray(deltas, dtype=np.float32)))
    if median_delta <= 0:
        return None
    return 1.0 / median_delta
