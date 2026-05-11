"""Config-driven camera-height provider."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

from pemoin.data.contracts import CameraHeightData, ResourceKind
from pemoin.providers.base import Provider


class CameraHeightProvider(Provider):
    """
    Provides camera height directly from profile settings.

    Supported settings:
    - ``height``: single float used for every processed frame.
    - ``heights``: array of floats, consumed in per-frame processing order.
    """

    produced_resources = frozenset({ResourceKind.CAMERA_HEIGHT})

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self._height: float | None = None
        self._heights: list[float] | None = None
        self._cursor = 0

    def setup(self, context: MutableMapping[str, Any]) -> None:
        _ = context
        has_height = "height" in self.settings
        has_heights = "heights" in self.settings
        if has_height and has_heights:
            raise ValueError("CameraHeightProvider settings must define exactly one of: 'height' or 'heights'.")
        if not has_height and not has_heights:
            raise ValueError("CameraHeightProvider requires either 'height' or 'heights' in settings.")

        self._cursor = 0
        if has_height:
            self._height = _as_height(self.settings["height"], key="height")
            self._heights = None
            return

        raw_heights = self.settings["heights"]
        if not isinstance(raw_heights, Sequence) or isinstance(raw_heights, (str, bytes)):
            raise ValueError("CameraHeightProvider setting 'heights' must be an array of numbers.")
        parsed = [_as_height(value, key=f"heights[{idx}]") for idx, value in enumerate(raw_heights)]
        if not parsed:
            raise ValueError("CameraHeightProvider setting 'heights' must not be empty.")
        self._height = None
        self._heights = parsed

    def process(self, frame) -> CameraHeightData:
        frame_idx = int(getattr(frame, "index"))
        if self._heights is None:
            if self._height is None:
                raise RuntimeError("CameraHeightProvider is not initialized. Did setup() run?")
            height_m = self._height
            source = "constant"
        else:
            if self._cursor >= len(self._heights):
                raise ValueError(
                    "CameraHeightProvider exhausted configured 'heights' array "
                    f"(processed={self._cursor}, configured={len(self._heights)})."
                )
            height_m = self._heights[self._cursor]
            source = "per_frame_array"

        metadata = {
            "source": "profile_config",
            "axis": "z",
            "world_coordinate_system": "blender",
            "height_source": source,
            "sequence_index": int(self._cursor),
        }
        self._cursor += 1
        return CameraHeightData(frame_index=frame_idx, height_m=float(height_m), metadata=metadata)


def register_camera_height_provider_builders(factory) -> None:
    factory.register("CameraHeightProvider", lambda binding, context: CameraHeightProvider(binding.settings))


def _as_height(value: Any, *, key: str) -> float:
    try:
        height = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CameraHeightProvider setting '{key}' must be a number.") from exc
    if not (height > 0.0):
        raise ValueError(f"CameraHeightProvider setting '{key}' must be > 0.")
    return height
