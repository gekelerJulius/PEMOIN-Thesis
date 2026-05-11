"""
Abstractions for delivering frames to the runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Mapping, Optional

from pemoin.data.contracts import FrameData


class FrameProvider(ABC):
    """Supplies frames, timestamps, and frame identifiers to the runtime."""

    @abstractmethod
    def open(self, source) -> None:
        """Prepare the frame source (camera, video file, etc.)."""

    @abstractmethod
    def read(self) -> Optional[FrameData]:
        """Return the next frame bundle expected by the runtime."""

    @abstractmethod
    def close(self) -> None:
        """Release resources associated with the frame source."""

    def runtime_settings(self) -> Mapping[str, object]:
        """Return runtime-resolved settings to merge into frame_provider_info."""
        return {}

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate through frames until the provider signals completion."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
