from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import sys

_PROGRESS_PREFIX = "PEMOIN_PROGRESS "


@dataclass
class BlenderSceneLogger:
    scope: str = "Scene"
    show_character_logs: bool = False
    show_render_logs: bool = False
    show_overlay_logs: bool = False
    show_road_plane_logs: bool = True
    show_road_plane_frame_logs: bool = True

    def should_emit_info(self, message: str) -> bool:
        if self.scope == "Character":
            return self.show_character_logs
        if self.scope == "Render":
            return self.show_render_logs
        if self.scope == "Overlay":
            return self.show_overlay_logs
        if message.startswith("[road-plane][global][frame ") or message.startswith(
            "[road-plane][local][frame "
        ):
            return self.show_road_plane_frame_logs
        if message.startswith("[road-plane]"):
            return self.show_road_plane_logs
        return True


LOGGER = BlenderSceneLogger()


@contextmanager
def log_scope(scope: str):
    previous = LOGGER.scope
    LOGGER.scope = scope
    try:
        yield
    finally:
        LOGGER.scope = previous


def log_info(message: str) -> None:
    if not LOGGER.should_emit_info(message):
        return
    print(f"[{LOGGER.scope}] {message}", flush=True)


def log_warning(message: str) -> None:
    print(f"[{LOGGER.scope}][warning] {message}", flush=True)


def log_warning_big(message: str) -> None:
    red = "\033[1;31m"
    yellow = "\033[1;33m"
    reset = "\033[0m"
    banner = "!" * 96
    print(f"{red}{banner}{reset}", flush=True)
    print(f"{yellow}[{LOGGER.scope}][WARNING] {message}{reset}", flush=True)
    print(f"{red}{banner}{reset}", flush=True)


def log_error(message: str) -> None:
    print(f"[{LOGGER.scope}][error] {message}", file=sys.stderr, flush=True)


def _emit_progress_event(event: str, **payload: object) -> None:
    progress_payload = {"event": str(event), **payload}
    print(f"{_PROGRESS_PREFIX}{json.dumps(progress_payload, sort_keys=True)}", flush=True)


def progress_begin(
    *,
    progress_id: str,
    label: str,
    total: int | None,
    unit: str = "frame",
    scope: str = "blender_render",
    resolution_scale: float | None = None,
    rerender_index: int | None = None,
) -> None:
    _emit_progress_event(
        "begin",
        id=progress_id,
        scope=scope,
        label=label,
        total=total,
        unit=unit,
        resolution_scale=resolution_scale,
        rerender_index=rerender_index,
    )


def progress_step(
    *,
    progress_id: str,
    current: int,
    total: int | None,
    unit: str = "frame",
    scope: str = "blender_render",
) -> None:
    _emit_progress_event(
        "step",
        id=progress_id,
        scope=scope,
        current=current,
        total=total,
        unit=unit,
    )


def progress_message(
    *,
    progress_id: str,
    message: str,
    scope: str = "blender_render",
) -> None:
    _emit_progress_event(
        "message",
        id=progress_id,
        scope=scope,
        message=message,
    )


def progress_end(
    *,
    progress_id: str,
    current: int | None = None,
    total: int | None = None,
    scope: str = "blender_render",
) -> None:
    _emit_progress_event(
        "end",
        id=progress_id,
        scope=scope,
        current=current,
        total=total,
    )
