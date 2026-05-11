"""
Frames provider that ingests a folder or video and materialises frames under
outputs/<pipeline_name>/standard/frames using the standard resource layout.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, MutableMapping, Sequence

import imageio.v2 as imageio
import numpy as np

from pemoin.data.contracts import FrameData, ResourceKind, ResourceMissingError, ResourceStore
from pemoin.providers.base import Provider

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v")


class FramesProvider(Provider):
    """Validates input media and exports standardised frame images."""

    required_resources = frozenset()
    produced_resources = frozenset({ResourceKind.FRAMES})

    def __init__(self, source: str | Path, *, max_frames: int | None = None):
        self.source = Path(source)
        self.max_frames = max_frames
        self._frames_dir: Path | None = None

    def setup(self, context: MutableMapping[str, object]) -> None:
        self.source = self.source.expanduser()
        if not self.source.exists():
            raise FileNotFoundError(f"Frame source '{self.source}' does not exist.")

    def process(self, frame):
        raise NotImplementedError("FramesProvider is batch-oriented; use run().")

    def run(self, resources: ResourceStore, context: MutableMapping[str, object] | None = None) -> None:
        self.validate_requirements(resources)
        target_dir = resources.base_dir(ResourceKind.FRAMES)
        target_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir = target_dir
        if context is not None:
            # Expose the persisted frames directory so downstream consumers can reference it.
            context.setdefault("frames_dir", target_dir)
        frames = self._iter_frames()
        for idx, image in enumerate(frames):
            if self.max_frames is not None and idx >= self.max_frames:
                break
            frame = FrameData(frame_id=str(idx).zfill(6), index=idx, image=image)
            resources.save_frame(frame)

    def teardown(self) -> None:
        """No-op teardown for frame ingestion."""
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _iter_frames(self) -> Iterable[np.ndarray]:
        if self.source.is_dir():
            yield from self._iter_images(self.source.iterdir())
            return
        if self.source.is_file():
            if self.source.suffix.lower() in _VIDEO_EXTS:
                yield from self._iter_video(self.source)
                return
            if self.source.suffix.lower() in _IMAGE_EXTS:
                yield from self._iter_images([self.source])
                return
        raise ResourceMissingError(
            f"Unsupported frame source '{self.source}'. Provide a folder of images or a video file."
        )

    def _iter_images(self, entries: Iterable[Path]) -> Iterable[np.ndarray]:
        sorted_entries: Sequence[Path] = sorted(
            (p for p in entries if p.suffix.lower() in _IMAGE_EXTS and p.is_file()),
            key=lambda p: p.name,
        )
        if not sorted_entries:
            raise ResourceMissingError(f"No images found under '{self.source}'.")
        for path in sorted_entries:
            yield imageio.imread(path)

    def _iter_video(self, path: Path) -> Iterable[np.ndarray]:
        reader = imageio.get_reader(path)
        with reader:
            for frame in itertools.islice(reader, 0, self.max_frames):
                yield np.asarray(frame)
