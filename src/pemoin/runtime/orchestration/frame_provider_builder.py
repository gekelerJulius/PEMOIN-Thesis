"""
Helpers for instantiating frame providers from profile bindings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from pemoin.runtime.profiles.config import ModuleBinding

from .frame_provider import FrameProvider
from .directory_frame_provider import DirectoryFrameProvider
from .video_frame_provider import VideoFrameProvider
from .unity_frame_provider import UnityFrameProvider
from .virtual_kitty_2_frame_provider import VirtualKitty2FrameProvider
from .carla_frame_provider import CarlaFrameProvider


def create_frame_provider_from_binding(
    binding: ModuleBinding,
    *,
    config_base: Path,
    override_path: Optional[Path] = None,
    frame_rate_override: Optional[float] = None,
) -> Tuple[FrameProvider, Path, dict]:
    """
    Instantiate and open a frame provider based on the profile binding.

    Returns:
        Tuple of (provider instance, resolved source path).
    """
    tool = binding.tool
    settings = dict(binding.settings)
    if "resize" in settings and settings.get("resize") is not None:
        raise ValueError(
            "frame_provider.settings.resize has been removed; use profile.working_resolution instead."
        )
    source_path = Path(override_path) if override_path else _resolve_path(settings.get("path"), config_base)
    if source_path is None:
        raise ValueError(
            f"Frame provider '{tool}' requires a 'path' setting or CLI override."
        )

    if tool == "DirectoryFrameProvider":
        provider = DirectoryFrameProvider(
            frame_rate=frame_rate_override or settings.get("frame_rate"),
            recursive=bool(settings.get("recursive", False)),
            extensions=_normalize_extensions(settings.get("extensions")),
            start_frame=settings.get("start_frame"),
            end_frame=settings.get("end_frame"),
            sampling_fps=settings.get("sampling_fps"),
        )
    elif tool == "VideoFrameProvider":
        end_seconds = settings.get("end_seconds")
        if end_seconds is not None:
            end_seconds = float(end_seconds)
        sampling_fps = settings.get("sampling_fps")
        sampling_fps = float(sampling_fps) if sampling_fps is not None else None
        legacy_stride = settings.pop("stride", None)
        frame_stride = max(1, int(legacy_stride)) if legacy_stride is not None else None
        settings["sampling_fps"] = sampling_fps
        if frame_stride is not None:
            settings["frame_stride"] = frame_stride
        start_frame = settings.get("start_frame")
        end_frame = settings.get("end_frame")
        if start_frame is not None and "start_seconds" in settings:
            raise ValueError("VideoFrameProvider does not allow both start_frame and start_seconds.")
        if end_frame is not None and "end_seconds" in settings:
            raise ValueError("VideoFrameProvider does not allow both end_frame and end_seconds.")
        provider = VideoFrameProvider(
            sampling_fps=sampling_fps,
            frame_stride=frame_stride,
            start_seconds=float(settings.get("start_seconds", 0.0)),
            end_seconds=end_seconds,
            frame_rate_hint=frame_rate_override or settings.get("frame_rate_hint"),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    elif tool == "UnityFrameProvider":
        provider = UnityFrameProvider(
            settings=settings,
            load_images=bool(settings.get("load_images", True)),
        )
    elif tool == "VirtualKitty2FrameProvider":
        provider = VirtualKitty2FrameProvider(
            settings=settings,
            load_images=bool(settings.get("load_images", True)),
        )
    elif tool == "CarlaFrameProvider":
        provider = CarlaFrameProvider(
            settings=settings,
            load_images=bool(settings.get("load_images", True)),
        )
    elif tool == "NuScenesFrameProvider":
        from .nuscenes_frame_provider import NuScenesFrameProvider
        provider = NuScenesFrameProvider(
            settings=settings,
            load_images=bool(settings.get("load_images", True)),
        )
    else:
        raise ValueError(f"Unknown frame provider tool '{tool}'.")

    provider.open(source_path)
    runtime_settings = dict(provider.runtime_settings())
    if runtime_settings:
        settings.update(runtime_settings)
    return provider, source_path.resolve(), {"tool": tool, "settings": settings}


def _resolve_path(value: Optional[str], base: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _normalize_extensions(raw: Optional[Sequence[str]]) -> Optional[Sequence[str]]:
    if raw is None:
        return None
    return tuple(str(ext).lower() for ext in raw)
