"""
Short-window scene cache shared across runtime modules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional

from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    FrameData,
    IntrinsicsData,
    PoseSample,
    SemanticsData,
)


@dataclass(slots=True)
class SceneFrameState:
    """Aggregated state for a processed frame."""

    frame: FrameData
    depth: Optional[DepthData] = None
    pose: Optional[PoseSample] = None
    intrinsics: Optional[IntrinsicsData] = None
    camera_height: Optional[CameraHeightData] = None
    semantics: Optional[SemanticsData] = None


class SceneStateCache:
    """Maintains temporal buffers of depth, poses, and intrinsics."""

    def __init__(self, window_size: int):
        """
        Args:
            window_size: Number of frames to keep in the cache.
        """
        self.window_size = max(1, int(window_size))
        self._buffer: Deque[SceneFrameState] = deque(maxlen=self.window_size)

    def update(self, state: SceneFrameState) -> None:
        """Insert new frame data into the cache."""
        self._buffer.append(state)

    def get_recent(self) -> List[SceneFrameState]:
        """Return the buffered state used for short-horizon decisions."""
        return list(self._buffer)

    def latest(self) -> Optional[SceneFrameState]:
        """Return the most recent frame state, if available."""
        return self._buffer[-1] if self._buffer else None

    def __iter__(self) -> Iterable[SceneFrameState]:
        return iter(self._buffer)
