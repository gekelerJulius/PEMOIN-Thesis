"""
Frame provider for nuScenes datasets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from PIL import Image

from pemoin.data.contracts import FrameData
from .frame_provider import FrameProvider


@dataclass(frozen=True)
class _NuScenesFrameEntry:
    index: int
    timestamp: float
    image_path: str
    sample_token: str | None
    cam_sd_token: str
    is_key_frame: bool


_SAMPLING_MODE_KEYFRAMES = "keyframes_only"
_SAMPLING_MODE_ALL = "all_camera_frames"
_VALID_SAMPLING_MODES = frozenset({_SAMPLING_MODE_KEYFRAMES, _SAMPLING_MODE_ALL})


class NuScenesFrameProvider(FrameProvider):
    """Streams NuScenes camera frames with configurable keyframe/sweep sampling."""

    def __init__(
        self, settings: Mapping[str, object], *, load_images: bool = True
    ) -> None:
        self._settings = dict(settings)
        self._entries: List[_NuScenesFrameEntry] = []
        self._cursor = 0
        self._opened = False
        self._load_images = bool(load_images)
        self._source_fps: float | None = None
        self._effective_fps: float | None = None
        self._sampling_mode: str = _SAMPLING_MODE_KEYFRAMES

    def open(self, source) -> None:
        from nuscenes.nuscenes import NuScenes

        dataroot = str(source)
        version = str(self._settings.get("version", "v1.0-mini"))
        camera = str(self._settings.get("camera", "CAM_FRONT"))
        sampling_mode = str(
            self._settings.get("sampling_mode", _SAMPLING_MODE_KEYFRAMES)
        ).strip().lower()
        if sampling_mode not in _VALID_SAMPLING_MODES:
            raise ValueError(
                "NuScenesFrameProvider sampling_mode must be one of: "
                f"{sorted(_VALID_SAMPLING_MODES)!r}."
            )
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Resolve scene
        scene = self._resolve_scene(nusc)

        entries = self._build_entries(nusc, scene=scene, camera=camera, dataroot=dataroot, sampling_mode=sampling_mode)

        if not entries:
            raise FileNotFoundError(
                f"No nuScenes camera frames found in scene '{scene['name']}'."
            )

        # Apply start_frame / end_frame filtering
        start_frame = self._settings.get("start_frame")
        end_frame = self._settings.get("end_frame")
        if start_frame is not None:
            start_frame = int(start_frame)
            entries = [e for e in entries if e.index >= start_frame]
        if end_frame is not None:
            end_frame = int(end_frame)
            entries = [e for e in entries if e.index <= end_frame]

        # Apply frame_stride
        frame_stride = self._settings.get("frame_stride")
        if frame_stride is not None:
            frame_stride = int(frame_stride)
            if frame_stride > 1:
                entries = entries[::frame_stride]

        if not entries:
            raise FileNotFoundError(
                "No nuScenes frames found after applying start_frame/end_frame/sampling/stride."
            )

        source_fps = _derive_effective_fps(entries, label="source")
        combined_stride = 1
        sampling_fps = self._settings.get("sampling_fps")
        if sampling_fps is not None:
            sampling_fps = float(sampling_fps)
            if sampling_fps <= 0.0:
                raise ValueError("NuScenesFrameProvider sampling_fps must be > 0.")
            stride = max(1, int(round(source_fps / sampling_fps)))
            combined_stride *= stride
            entries = entries[::stride]

        frame_stride = self._settings.get("frame_stride")
        if frame_stride is not None:
            frame_stride = int(frame_stride)
            if frame_stride < 1:
                raise ValueError("frame_stride must be >= 1.")
            if frame_stride > 1:
                combined_stride *= frame_stride
                entries = entries[::frame_stride]

        if not entries:
            raise FileNotFoundError(
                "No nuScenes frames found after applying start_frame/end_frame/sampling/stride."
            )

        self._entries = entries
        self._source_fps = source_fps
        if len(entries) >= 2:
            self._effective_fps = _derive_effective_fps(entries, label="resolved")
        else:
            self._effective_fps = source_fps / float(combined_stride)
        self._sampling_mode = sampling_mode
        self._cursor = 0
        self._opened = True

    def read(self) -> Optional[FrameData]:
        if not self._opened:
            raise RuntimeError(
                "NuScenesFrameProvider.open must be called before reading frames."
            )
        if self._cursor >= len(self._entries):
            return None

        entry = self._entries[self._cursor]
        self._cursor += 1
        sequential_index = self._cursor - 1

        image = None
        if self._load_images:
            pil_img = Image.open(entry.image_path)
            image = np.array(pil_img)

        return FrameData(
            frame_id=f"{sequential_index:06d}",
            index=int(sequential_index),
            timestamp=entry.timestamp,
            image=image,
            metadata={
                "source_path": entry.image_path,
                "source_frame_index": int(entry.index),
                "sample_token": entry.sample_token,
                "cam_sd_token": entry.cam_sd_token,
                "source_timestamp": float(entry.timestamp),
                "source_is_key_frame": bool(entry.is_key_frame),
                "sampling_mode": self._sampling_mode,
            },
        )

    def __len__(self) -> int:
        return len(self._entries)

    def close(self) -> None:
        self._entries = []
        self._cursor = 0
        self._opened = False
        self._source_fps = None
        self._effective_fps = None

    def runtime_settings(self) -> Mapping[str, object]:
        if self._effective_fps is None or self._source_fps is None:
            return {}
        return {
            "source_sampling_fps": float(self._source_fps),
            "sampling_fps": float(self._effective_fps),
            "resolved_sampling_fps": float(self._effective_fps),
            "timing_source": "derived_from_timestamps",
            "sampling_mode": self._sampling_mode,
        }

    def _resolve_scene(self, nusc) -> dict:
        scene_name = self._settings.get("scene_name")
        if scene_name is not None:
            for sc in nusc.scene:
                if sc["name"] == str(scene_name):
                    return sc
            raise ValueError(f"nuScenes scene '{scene_name}' not found.")

        scene_index = int(self._settings.get("scene_index", 0))
        if scene_index < 0 or scene_index >= len(nusc.scene):
            raise IndexError(
                f"scene_index {scene_index} out of range (dataset has {len(nusc.scene)} scenes)."
            )
        return nusc.scene[scene_index]

    def _build_entries(
        self,
        nusc,
        *,
        scene: Mapping[str, Any],
        camera: str,
        dataroot: str,
        sampling_mode: str,
    ) -> List[_NuScenesFrameEntry]:
        sample_to_token: Dict[str, str] = {}
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            sample_to_token[str(sample["data"][camera])] = str(token)
            token = sample["next"] if sample["next"] else None

        first_key = str(nusc.get("sample", scene["first_sample_token"])["data"][camera])
        entries: List[_NuScenesFrameEntry] = []
        cam_sd_token = first_key
        idx = 0
        while cam_sd_token:
            cam_sd = nusc.get("sample_data", cam_sd_token)
            image_path = os.path.join(dataroot, cam_sd["filename"])
            entries.append(
                _NuScenesFrameEntry(
                    index=idx,
                    timestamp=float(cam_sd["timestamp"]) / 1e6,
                    image_path=image_path,
                    sample_token=sample_to_token.get(str(cam_sd_token)),
                    cam_sd_token=str(cam_sd_token),
                    is_key_frame=bool(cam_sd.get("is_key_frame", False)),
                )
            )
            idx += 1
            next_token = cam_sd["next"] if cam_sd["next"] else None
            if next_token is None:
                break
            cam_sd_token = str(next_token)

        if sampling_mode == _SAMPLING_MODE_KEYFRAMES:
            entries = [entry for entry in entries if entry.is_key_frame]
        return entries


def _derive_effective_fps(entries: List[_NuScenesFrameEntry], *, label: str = "effective") -> float:
    """Estimate effective cadence from the filtered scene timestamps."""
    if len(entries) < 2:
        raise ValueError(
            f"NuScenesFrameProvider requires at least 2 frames to derive {label} FPS."
        )
    timestamps = np.array([entry.timestamp for entry in entries], dtype=np.float64)
    deltas = np.diff(timestamps)
    valid_deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if valid_deltas.size == 0:
        raise ValueError(
            f"NuScenesFrameProvider could not derive {label} FPS from scene timestamps."
        )
    median_delta = float(np.median(valid_deltas))
    fps = 1.0 / median_delta
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"NuScenesFrameProvider derived an invalid {label} FPS.")
    return fps
