"""
Frame provider for CARLA export datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

import imageio.v3 as iio
import numpy as np

from pemoin.data.carla import CarlaDataset
from pemoin.data.contracts import FrameData
from .frame_provider import FrameProvider


@dataclass(frozen=True)
class _CarlaFrameEntry:
    index: int
    timestamp: Optional[float]
    image_path: Path


class CarlaFrameProvider(FrameProvider):
    """
    Streams frames from a CARLA export directory containing frames.jsonl.
    """

    def __init__(self, settings: Mapping[str, object], *, load_images: bool = True) -> None:
        self._settings = dict(settings)
        self._entries: List[_CarlaFrameEntry] = []
        self._cursor = 0
        self._opened = False
        self._load_images = bool(load_images)
        self._frame_rate: Optional[float] = None
        self._stride = 1

    def open(self, source) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CARLA data directory '{path}' does not exist.")
        if not path.is_dir():
            raise NotADirectoryError(f"CARLA data source '{path}' must be a directory.")

        dataset = CarlaDataset(path)
        required = _parse_required_resources(self._settings.get("required_resources"))
        entries = []
        for frame_index in dataset.frame_indices():
            record = dataset.frame(frame_index)
            if "depth" in required and not record.depth_path.exists():
                continue
            if "semseg" in required and (record.semseg_path is None or not record.semseg_path.exists()):
                continue
            if "instseg" in required and (record.instseg_path is None or not record.instseg_path.exists()):
                continue
            entries.append(
                _CarlaFrameEntry(
                    index=record.frame,
                    timestamp=record.timestamp,
                    image_path=record.rgb_path,
                )
            )
        if not entries:
            raise FileNotFoundError(f"No CARLA frames found under '{path}'.")
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
            raise FileNotFoundError("No CARLA frames found after applying start_frame/end_frame.")
        frame_rate = self._settings.get("frame_rate") or self._settings.get("fps")
        self._frame_rate = float(frame_rate) if frame_rate is not None else dataset.frame_rate()
        sampling_fps = self._settings.get("sampling_fps")
        if sampling_fps is not None:
            if self._frame_rate is None or self._frame_rate <= 0:
                raise ValueError("sampling_fps requires a valid frame_rate for CarlaFrameProvider.")
            sampling_fps = float(sampling_fps)
            if sampling_fps <= 0:
                raise ValueError("sampling_fps must be > 0.")
            self._stride = max(1, int(round(self._frame_rate / sampling_fps)))
            entries = entries[:: self._stride]
        frame_stride = self._settings.get("frame_stride")
        if frame_stride is not None:
            frame_stride = int(frame_stride)
            if frame_stride < 1:
                raise ValueError("frame_stride must be >= 1.")
            if frame_stride > 1:
                entries = entries[:: frame_stride]
        if not entries:
            raise FileNotFoundError("No CARLA frames found after applying sampling/stride.")
        self._entries = entries
        self._cursor = 0
        self._opened = True

    def read(self) -> Optional[FrameData]:
        if not self._opened:
            raise RuntimeError("CarlaFrameProvider.open must be called before reading frames.")
        if self._cursor >= len(self._entries):
            return None
        entry = self._entries[self._cursor]
        self._cursor += 1
        sequential_index = self._cursor - 1
        image = None
        if self._load_images:
            image = np.asarray(iio.imread(entry.image_path))
        frame = FrameData(
            frame_id=f"{sequential_index:06d}",
            index=int(sequential_index),
            timestamp=entry.timestamp,
            image=image,
            metadata={
                "source_path": str(entry.image_path),
                "source_frame_index": int(entry.index),
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


def _parse_required_resources(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [str(key) for key, value in raw.items() if bool(value)]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw]
    raise ValueError("required_resources must be a list or object of booleans.")
