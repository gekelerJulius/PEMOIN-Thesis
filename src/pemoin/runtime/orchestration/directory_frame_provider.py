"""
Directory-backed frame provider that streams RGB images from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pemoin.data.contracts import FrameData
from .frame_provider import FrameProvider


def _natural_key(path: Path) -> List[object]:
    """Return a sorting key that preserves numeric ordering within filenames."""
    parts: List[object] = []
    token = ""
    for char in path.stem:
        if char.isdigit():
            token += char
        else:
            if token:
                parts.append(int(token))
                token = ""
            parts.append(char.lower())
    if token:
        parts.append(int(token))
    parts.append(path.suffix.lower())
    return parts


def _default_image_loader(path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "DirectoryFrameProvider requires imageio>=2.31. "
            "Install PEMOIN with `pip install pemoin[offline]` to enable image ingestion."
        ) from exc
    return np.asarray(iio.imread(path))


class DirectoryFrameProvider(FrameProvider):
    """
    Streams frames from a directory containing image files.

    The provider keeps the implementation lightweight so that tests can inject
    temporary folders without pulling on heavy video decoding dependencies.
    """

    SUPPORTED_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        *,
        frame_rate: Optional[float] = 24.0,
        recursive: bool = False,
        extensions: Optional[Sequence[str]] = None,
        image_loader: Optional[Callable[[Path], np.ndarray]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        sampling_fps: Optional[float] = None,
    ) -> None:
        if sampling_fps is not None and sampling_fps <= 0:
            raise ValueError("sampling_fps must be positive when provided.")
        self.frame_rate = frame_rate
        self.recursive = recursive
        self.extensions = tuple(ext.lower() for ext in (extensions or self.SUPPORTED_EXTENSIONS))
        self._frames: List[Tuple[int, Path]] = []
        self._cursor = 0
        self._opened = False
        self._root: Optional[Path] = None
        self._image_loader = image_loader or _default_image_loader
        self._start_frame = int(start_frame) if start_frame is not None else None
        self._end_frame = int(end_frame) if end_frame is not None else None
        self._sampling_fps = float(sampling_fps) if sampling_fps is not None else None
        self._stride = 1

    def open(self, source) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Frame directory '{path}' does not exist.")
        if not path.is_dir():
            raise NotADirectoryError(f"Frame source '{path}' must be a directory.")

        self._root = path
        paths = self._discover_frames(path)
        if not paths:
            raise FileNotFoundError(
                f"No frames found under '{path}'. "
                f"Supported extensions: {', '.join(self.extensions)}"
            )
        indexed = list(enumerate(paths))
        if self._start_frame is not None:
            if self._start_frame < 0:
                raise ValueError("start_frame must be >= 0.")
            indexed = [item for item in indexed if item[0] >= self._start_frame]
        if self._end_frame is not None:
            if self._end_frame < 0:
                raise ValueError("end_frame must be >= 0.")
            indexed = [item for item in indexed if item[0] <= self._end_frame]
        if not indexed:
            raise FileNotFoundError("No frames found after applying start_frame/end_frame.")
        if self._sampling_fps is not None:
            if self.frame_rate is None or self.frame_rate <= 0:
                raise ValueError("sampling_fps requires a valid frame_rate for DirectoryFrameProvider.")
            self._stride = max(1, int(round(self.frame_rate / self._sampling_fps)))
            indexed = indexed[:: self._stride]
        self._frames = indexed
        self._cursor = 0
        self._opened = True

    def read(self) -> Optional[FrameData]:
        if not self._opened:
            raise RuntimeError("DirectoryFrameProvider.open must be called before reading frames.")
        if self._cursor >= len(self._frames):
            return None

        source_index, file_path = self._frames[self._cursor]
        index = self._cursor
        self._cursor += 1
        image = self._load_image(file_path)
        timestamp = None
        if self.frame_rate is not None and self.frame_rate > 0.0:
            timestamp = source_index / self.frame_rate

        frame = FrameData(
            frame_id=file_path.name,
            index=index,
            timestamp=timestamp,
            image=image,
            metadata={
                "source_path": str(file_path),
                "source_frame_index": source_index,
                "frame_stride": self._stride,
                "sampling_fps": self._sampling_fps,
            },
        )
        return frame

    def close(self) -> None:
        self._frames = []
        self._cursor = 0
        self._opened = False
        self._root = None

    def __len__(self) -> int:
        return len(self._frames)

    def runtime_settings(self) -> dict[str, object]:
        resolved_sampling_fps = None
        if self.frame_rate is not None and self.frame_rate > 0:
            resolved_sampling_fps = float(self.frame_rate) / float(max(1, self._stride))
        return {
            "frame_rate": float(self.frame_rate) if self.frame_rate is not None else None,
            "sampling_fps": self._sampling_fps,
            "resolved_sampling_fps": resolved_sampling_fps,
            "frame_stride": int(self._stride),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _discover_frames(self, root: Path) -> List[Path]:
        pattern_iter: Iterable[Path]
        if self.recursive:
            pattern_iter = root.rglob("*")
        else:
            pattern_iter = root.iterdir()

        paths = [
            path
            for path in pattern_iter
            if path.is_file() and path.suffix.lower() in self.extensions
        ]
        paths.sort(key=_natural_key)
        return paths

    def _load_image(self, path: Path) -> np.ndarray:
        image = self._image_loader(path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image
