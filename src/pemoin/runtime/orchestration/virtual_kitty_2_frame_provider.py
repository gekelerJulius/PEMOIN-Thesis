"""
Frame provider for the Virtual KITTI 2 dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

import imageio.v3 as iio
import numpy as np

from pemoin.data.contracts import FrameData
from pemoin.data.virtual_kitty_2 import resolve_vkitti2_dataset
from .frame_provider import FrameProvider


@dataclass(frozen=True)
class _VK2FrameEntry:
    index: int
    image_path: Path


class VirtualKitty2FrameProvider(FrameProvider):
    """
    Streams frames from a Virtual KITTI 2 scene/variation/camera selection.
    """

    def __init__(self, settings: Mapping[str, object], *, load_images: bool = True) -> None:
        self._settings = dict(settings)
        self._entries: List[_VK2FrameEntry] = []
        self._cursor = 0
        self._opened = False
        self._load_images = bool(load_images)
        self._dataset_root: Optional[Path] = None
        self._scene: Optional[str] = None
        self._variation: Optional[str] = None
        self._camera: Optional[int] = None
        self._frame_rate: Optional[float] = None

    def open(self, source) -> None:
        dataset_settings = dict(self._settings)
        dataset_settings["path"] = source
        dataset = resolve_vkitti2_dataset(dataset_settings)
        required_resources = _parse_required_resources(dataset_settings.get("required_resources"))
        indices = dataset.available_indices(required_resources)
        if not indices:
            raise FileNotFoundError("Virtual KITTI 2 selection contains no frames.")
        start_frame = dataset_settings.get("start_frame")
        end_frame = dataset_settings.get("end_frame")
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
            indices = [idx for idx in indices if idx >= start_frame]
        if end_frame is not None:
            indices = [idx for idx in indices if idx <= end_frame]
        if not indices:
            raise FileNotFoundError("Virtual KITTI 2 selection contains no frames after cropping.")
        frame_rate = dataset_settings.get("frame_rate") or dataset_settings.get("fps")
        self._frame_rate = float(frame_rate) if frame_rate is not None else 10.0
        sampling_fps = dataset_settings.get("sampling_fps")
        if sampling_fps is not None:
            if self._frame_rate is None:
                raise ValueError("sampling_fps requires frame_rate or fps to be set for Virtual KITTI 2.")
            sampling_fps = float(sampling_fps)
            if sampling_fps <= 0:
                raise ValueError("sampling_fps must be > 0.")
            stride = max(1, int(round(self._frame_rate / sampling_fps)))
            indices = indices[::stride]
        frame_stride = int(dataset_settings.get("frame_stride", 1))
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1.")
        if frame_stride > 1:
            indices = indices[::frame_stride]
        entries = [_VK2FrameEntry(index=idx, image_path=dataset.frame_path(idx)) for idx in indices]
        self._entries = entries
        self._cursor = 0
        self._opened = True
        self._dataset_root = dataset.root
        self._scene = dataset.selection.scene
        self._variation = dataset.selection.variation
        self._camera = dataset.selection.camera

    def read(self) -> Optional[FrameData]:
        if not self._opened:
            raise RuntimeError("VirtualKitty2FrameProvider.open must be called before reading frames.")
        if self._cursor >= len(self._entries):
            return None
        entry = self._entries[self._cursor]
        self._cursor += 1
        image = None
        if self._load_images:
            image = np.asarray(iio.imread(entry.image_path))
        timestamp = None
        if self._frame_rate is not None and self._frame_rate > 0:
            timestamp = float(entry.index) / float(self._frame_rate)
        frame = FrameData(
            frame_id=f"{entry.index:06d}",
            index=int(entry.index),
            timestamp=timestamp,
            image=image,
            metadata={
                "source_path": str(entry.image_path),
                "dataset_root": str(self._dataset_root) if self._dataset_root else None,
                "scene": self._scene,
                "variation": self._variation,
                "camera": self._camera,
            },
        )
        return frame

    def __len__(self) -> int:
        return len(self._entries)

    def close(self) -> None:
        self._entries = []
        self._cursor = 0
        self._opened = False
        self._dataset_root = None
        self._scene = None
        self._variation = None
        self._camera = None
        self._frame_rate = None


def _parse_required_resources(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [str(key) for key, value in raw.items() if bool(value)]
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw]
    raise ValueError("required_resources must be a list or object of booleans.")
