"""
Camera intrinsics provider.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pemoin.data.contracts import IntrinsicsData, ResourceKind
from pemoin.utils.resolution import scale_intrinsics
from .base import Provider


class IntrinsicsProvider(Provider):
    """Resolves the camera calibration parameters for each frame sequence."""

    produced_resources = frozenset({ResourceKind.INTRINSICS})

    def setup(self, context: Mapping[str, Any]) -> None:
        self._working_resolution = context.get("working_resolution")

    def _scale_intrinsics(self, intrinsics: IntrinsicsData, frame: Any) -> IntrinsicsData:
        target_shape = _resolve_intrinsics_target(frame, getattr(self, "_working_resolution", None))
        if target_shape is None:
            return intrinsics
        return scale_intrinsics(intrinsics, target_shape)

    def process(self, frame):
        """Return intrinsic parameters such as K matrix and distortion."""
        raise NotImplementedError("Intrinsics estimation will be implemented later.")


def _resolve_intrinsics_target(
    frame: Any, working_resolution: Optional[Sequence[int] | int | float]
) -> Optional[Sequence[int]]:
    image = getattr(frame, "image", None)
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        return image.shape[:2]
    metadata = getattr(frame, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        res = metadata.get("working_resolution")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            return (int(res[0]), int(res[1]))
    return working_resolution
