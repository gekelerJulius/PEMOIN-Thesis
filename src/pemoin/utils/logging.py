"""
Console logging helpers with ANSI colour formatting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import time
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import IO, Any, Iterable, Iterator, Mapping, Optional, Sequence

from tqdm import tqdm

_LEVEL_COLOURS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
}
_RESET = "\033[0m"
_PROGRESS_PREFIX = "PEMOIN_PROGRESS "


@dataclass(frozen=True)
class ConsoleLoggingConfig:
    level: int
    summary_only: bool
    stream_blender_subprocess_output: bool
    show_progress_bars: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "level": int(self.level),
            "summary_only": bool(self.summary_only),
            "stream_blender_subprocess_output": bool(self.stream_blender_subprocess_output),
            "show_progress_bars": bool(self.show_progress_bars),
        }


@dataclass(frozen=True)
class LoggedSubprocessResult:
    args: tuple[str, ...]
    returncode: int
    stdout_log_path: Path
    stderr_log_path: Path
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]
    streamed_output: bool

    def format_failure(self, *, label: str, max_lines: int = 20) -> str:
        details = [
            f"{label} failed (exit code {self.returncode}).",
            f"stdout log: {self.stdout_log_path}",
            f"stderr log: {self.stderr_log_path}",
        ]
        tail_lines: list[str] = []
        half = max(1, int(max_lines) // 2)
        if self.stderr_tail:
            tail_lines.append("stderr tail:")
            tail_lines.extend(self.stderr_tail[-half:])
        if self.stdout_tail:
            tail_lines.append("stdout tail:")
            tail_lines.extend(self.stdout_tail[-half:])
        if tail_lines:
            details.append("captured output:")
            details.extend(tail_lines)
        return "\n".join(details)


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - standard override
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        message = super().format(record)
        if colour:
            message = f"{colour}{message}{_RESET}"
        return message


class _SummaryFilter(logging.Filter):
    def __init__(self, summary_only: bool = True) -> None:
        super().__init__()
        self.summary_only = summary_only

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - standard override
        if not self.summary_only:
            return True
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, "summary", False))


def _apply_summary_filter(handler: logging.Handler, summary_only: bool) -> None:
    for flt in handler.filters:
        if isinstance(flt, _SummaryFilter):
            flt.summary_only = summary_only
            return
    handler.addFilter(_SummaryFilter(summary_only=summary_only))


def resolve_console_logging_config(
    *,
    quiet: bool = False,
    verbose: bool = False,
) -> ConsoleLoggingConfig:
    if verbose:
        return ConsoleLoggingConfig(
            level=logging.DEBUG,
            summary_only=False,
            stream_blender_subprocess_output=False,
            show_progress_bars=True,
        )
    if quiet:
        return ConsoleLoggingConfig(
            level=logging.WARNING,
            summary_only=True,
            stream_blender_subprocess_output=False,
            show_progress_bars=False,
        )
    return ConsoleLoggingConfig(
        level=logging.INFO,
        summary_only=True,
        stream_blender_subprocess_output=False,
        show_progress_bars=True,
    )


def setup_console_logging(level: int = logging.INFO, *, summary_only: bool = True) -> logging.Logger:
    """
    Configure logging so that:
      - pemoin.* logs show at `level` with colour formatting
      - third-party libraries stay quiet (WARNING+), even if root was configured elsewhere
    """
    # 1) Quarantine third-party logs at the root
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    # If something already configured root (common in notebooks/CLIs), remove those handlers
    # to prevent library DEBUG/INFO spam.
    if root.handlers:
        root.handlers.clear()

    root_handler = logging.StreamHandler(stream=sys.stdout)
    root_handler.setLevel(logging.WARNING)
    root_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s", datefmt="%H:%M:%S")
    root_handler.setFormatter(root_formatter)
    root.addHandler(root_handler)

    for name in ("PIL", "matplotlib", "imageio", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 2) Configure pemoin logger (your logs)
    logger = logging.getLogger("pemoin")

    # If already configured, just update levels/filters
    if logger.handlers:
        logger.setLevel(level)
        for handler in logger.handlers:
            _apply_summary_filter(handler, summary_only)
        logger.propagate = False
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    formatter = _ColourFormatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    _apply_summary_filter(handler, summary_only)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # stop at pemoin; don't bubble to root
    return logger


def _write_stream_lines(
    stream: IO[str] | None,
    handle: IO[str],
    *,
    mirror_stream: IO[str] | None,
    tail: deque[str],
    progress_handler: "_SubprocessProgressHandler | None",
) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            handle.write(line)
            handle.flush()
            stripped = line.rstrip("\n")
            if progress_handler is not None and progress_handler.handle(stripped):
                continue
            tail.append(stripped)
            if mirror_stream is not None:
                mirror_stream.write(line)
                mirror_stream.flush()
    finally:
        stream.close()


class _SubprocessProgressHandler:
    def __init__(self, *, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._bars: dict[str, tqdm] = {}

    def handle(self, line: str) -> bool:
        if not line.startswith(_PROGRESS_PREFIX):
            return False
        payload_text = line[len(_PROGRESS_PREFIX) :].strip()
        try:
            payload = json.loads(payload_text)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        event = str(payload.get("event", "")).strip().lower()
        progress_id = str(payload.get("id", "")).strip()
        if not event or not progress_id:
            return True
        if not self._enabled:
            return True
        with self._lock:
            if event == "begin":
                self._begin(progress_id, payload)
            elif event == "step":
                self._step(progress_id, payload)
            elif event == "message":
                self._message(progress_id, payload)
            elif event == "end":
                self._end(progress_id, payload)
        return True

    def close(self) -> None:
        with self._lock:
            for bar in list(self._bars.values()):
                bar.close()
            self._bars.clear()

    def _begin(self, progress_id: str, payload: dict[str, object]) -> None:
        existing = self._bars.pop(progress_id, None)
        if existing is not None:
            existing.close()
        total_raw = payload.get("total")
        total = None if total_raw is None else int(total_raw)
        label = str(payload.get("label", progress_id))
        unit = str(payload.get("unit", "step"))
        resolution_scale = payload.get("resolution_scale")
        rerender_index = payload.get("rerender_index")
        postfix_parts: list[str] = []
        if resolution_scale is not None:
            postfix_parts.append(f"scale={float(resolution_scale):.2f}")
        if rerender_index is not None:
            postfix_parts.append(f"rerender={int(rerender_index)}")
        bar = tqdm(
            total=total,
            desc=label,
            unit=unit,
            dynamic_ncols=True,
            leave=False,
        )
        if postfix_parts:
            bar.set_postfix_str(" ".join(postfix_parts), refresh=False)
        self._bars[progress_id] = bar

    def _step(self, progress_id: str, payload: dict[str, object]) -> None:
        bar = self._bars.get(progress_id)
        if bar is None:
            self._begin(progress_id, payload)
            bar = self._bars.get(progress_id)
            if bar is None:
                return
        current_raw = payload.get("current")
        if current_raw is None:
            bar.update(1)
            return
        current = int(current_raw)
        delta = current - int(bar.n)
        if delta > 0:
            bar.update(delta)

    def _message(self, progress_id: str, payload: dict[str, object]) -> None:
        bar = self._bars.get(progress_id)
        if bar is None:
            return
        message = str(payload.get("message", "")).strip()
        if message:
            bar.set_postfix_str(message, refresh=True)

    def _end(self, progress_id: str, payload: dict[str, object]) -> None:
        bar = self._bars.pop(progress_id, None)
        if bar is None:
            return
        current_raw = payload.get("current")
        if current_raw is not None:
            current = int(current_raw)
            delta = current - int(bar.n)
            if delta > 0:
                bar.update(delta)
        elif bar.total is not None and int(bar.n) < int(bar.total):
            bar.update(int(bar.total) - int(bar.n))
        bar.close()


def run_logged_subprocess(
    cmd: Sequence[str],
    *,
    stdout_log_path: Path,
    stderr_log_path: Path,
    stream_output: bool = False,
    show_progress: bool = True,
    cwd: Path | None = None,
) -> LoggedSubprocessResult:
    stdout_log_path = Path(stdout_log_path).expanduser().resolve()
    stderr_log_path = Path(stderr_log_path).expanduser().resolve()
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_tail: deque[str] = deque(maxlen=20)
    stderr_tail: deque[str] = deque(maxlen=20)
    progress_handler = _SubprocessProgressHandler(enabled=show_progress)
    with stdout_log_path.open("w", encoding="utf-8") as stdout_handle, stderr_log_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            list(cmd),
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=_write_stream_lines,
            args=(process.stdout, stdout_handle),
            kwargs={
                "mirror_stream": sys.stdout if stream_output else None,
                "tail": stdout_tail,
                "progress_handler": progress_handler,
            },
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_write_stream_lines,
            args=(process.stderr, stderr_handle),
            kwargs={
                "mirror_stream": sys.stderr if stream_output else None,
                "tail": stderr_tail,
                "progress_handler": progress_handler,
            },
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = int(process.wait())
        stdout_thread.join()
        stderr_thread.join()
        progress_handler.close()
    return LoggedSubprocessResult(
        args=tuple(str(part) for part in cmd),
        returncode=returncode,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        stdout_tail=tuple(stdout_tail),
        stderr_tail=tuple(stderr_tail),
        streamed_output=bool(stream_output),
    )


def iter_with_progress(
    iterable: Iterable,
    *,
    enabled: bool,
    total: int | None,
    desc: str,
    unit: str,
) -> Iterator:
    if not enabled:
        yield from iterable
        return
    with tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=False) as bar:
        for item in iterable:
            yield item
            bar.update(1)



def get_logger() -> logging.Logger:
    """Return the shared PEMOIN logger."""
    return logging.getLogger("pemoin")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_symbol(status: str) -> str:
    normalized = str(status).strip().lower()
    if normalized == "completed":
        return "[done]"
    if normalized in {"cache_hit", "cache_materialized"}:
        return "[cache]"
    if normalized in {"skipped", "disabled"}:
        return "[skip]"
    if normalized == "failed":
        return "[fail]"
    return "[run]"


def _timeline_console_message(
    *,
    event: str,
    depth: int,
    display_name: str,
    status: str | None = None,
    duration_s: float | None = None,
) -> str:
    indent = "  " * max(0, int(depth))
    branch = "|- "
    if event == "start":
        return f"{indent}{branch}[run] {display_name}"
    suffix = ""
    if duration_s is not None:
        suffix = f" ({duration_s:.2f}s)"
    return f"{indent}{branch}{_status_symbol(status or 'completed')} {display_name}{suffix}"


@dataclass
class RuntimeStageRecord:
    name: str
    display_name: str
    started_at: str
    started_perf_counter: float
    status: str = "running"
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list["RuntimeStageRecord"] = field(default_factory=list)
    ended_at: str | None = None
    duration_s: float | None = None

    def finish(
        self,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
        ended_at: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        self.status = str(status)
        if metadata:
            self.metadata.update(dict(metadata))
        self.ended_at = ended_at or _utc_now_iso()
        self.duration_s = float(duration_s) if duration_s is not None else (
            time.perf_counter() - float(self.started_perf_counter)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "metadata": dict(self.metadata),
            "children": [child.to_mapping() for child in self.children],
        }


class RuntimeStageScope:
    def __init__(
        self,
        timeline: "RuntimeTimeline",
        record: RuntimeStageRecord,
        *,
        log_to_console: bool,
    ) -> None:
        self.timeline = timeline
        self.record = record
        self.log_to_console = bool(log_to_console)

    def finish(self, *, status: str = "completed", metadata: Mapping[str, Any] | None = None) -> None:
        self.timeline.finish_stage(
            self.record,
            status=status,
            metadata=metadata,
            log_to_console=self.log_to_console,
        )

    def __enter__(self) -> "RuntimeStageScope":
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc is not None:
            self.finish(
                status="failed",
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
            return False
        self.finish()
        return False


class RuntimeTimeline:
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        console_enabled: bool = True,
    ) -> None:
        self.logger = logger or get_logger()
        self.console_enabled = bool(console_enabled)
        self.started_at = _utc_now_iso()
        self.started_perf_counter = time.perf_counter()
        self.ended_at: str | None = None
        self.duration_s: float | None = None
        self.status = "running"
        self.metadata: dict[str, Any] = {}
        self.stages: list[RuntimeStageRecord] = []
        self._stack: list[RuntimeStageRecord] = []

    def set_metadata(self, **values: Any) -> None:
        self.metadata.update({str(key): value for key, value in values.items()})

    def stage(
        self,
        name: str,
        *,
        display_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        log_to_console: bool = True,
    ) -> RuntimeStageScope:
        record = self.begin_stage(
            name,
            display_name=display_name,
            metadata=metadata,
            log_to_console=log_to_console,
        )
        return RuntimeStageScope(self, record, log_to_console=log_to_console)

    def begin_stage(
        self,
        name: str,
        *,
        display_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        log_to_console: bool = True,
    ) -> RuntimeStageRecord:
        record = RuntimeStageRecord(
            name=str(name),
            display_name=str(display_name or name),
            started_at=_utc_now_iso(),
            started_perf_counter=time.perf_counter(),
            metadata=dict(metadata or {}),
        )
        if self._stack:
            self._stack[-1].children.append(record)
        else:
            self.stages.append(record)
        self._stack.append(record)
        if self.console_enabled and log_to_console:
            self.logger.info(
                _timeline_console_message(
                    event="start",
                    depth=len(self._stack) - 1,
                    display_name=record.display_name,
                ),
                extra={"summary": True},
            )
        return record

    def finish_stage(
        self,
        record: RuntimeStageRecord,
        *,
        status: str = "completed",
        metadata: Mapping[str, Any] | None = None,
        log_to_console: bool = True,
    ) -> None:
        if record.duration_s is not None:
            return
        if self._stack and self._stack[-1] is record:
            self._stack.pop()
        elif record in self._stack:
            self._stack.remove(record)
        record.finish(status=status, metadata=dict(metadata or {}))
        if self.console_enabled and log_to_console:
            self.logger.info(
                _timeline_console_message(
                    event="end",
                    depth=len(self._stack),
                    display_name=record.display_name,
                    status=record.status,
                    duration_s=record.duration_s,
                ),
                extra={"summary": True},
            )

    def add_completed_stage(
        self,
        name: str,
        *,
        display_name: str | None = None,
        status: str = "completed",
        duration_s: float | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        parent: RuntimeStageRecord | None = None,
    ) -> RuntimeStageRecord:
        start_perf = time.perf_counter()
        record = RuntimeStageRecord(
            name=str(name),
            display_name=str(display_name or name),
            started_at=started_at or _utc_now_iso(),
            started_perf_counter=start_perf,
            metadata=dict(metadata or {}),
        )
        record.finish(
            status=status,
            metadata=None,
            ended_at=ended_at or _utc_now_iso(),
            duration_s=duration_s if duration_s is not None else 0.0,
        )
        if parent is not None:
            parent.children.append(record)
        else:
            self.stages.append(record)
        return record

    def finalize(self, *, status: str = "completed", metadata: Mapping[str, Any] | None = None) -> None:
        if self.duration_s is not None:
            return
        self.status = str(status)
        if metadata:
            self.metadata.update(dict(metadata))
        self.ended_at = _utc_now_iso()
        self.duration_s = time.perf_counter() - float(self.started_perf_counter)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "metadata": dict(self.metadata),
            "stages": [stage.to_mapping() for stage in self.stages],
        }


def setup_step_debug_logging(
    run_dir: Path,
    step: str,
    *,
    logger_name: str = "pemoin",
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> Path:
    """
    Attach a per-step debug log file handler under outputs/<run>/standard/logs.
    """
    logger = logging.getLogger(logger_name)
    log_dir = Path(run_dir).expanduser().resolve() / "standard" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / f"{step}.debug.log").resolve()

    existing = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path:
            existing = handler
            break
    if existing is None:
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        existing.setLevel(level)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(console_level)

    if logger.level > level:
        logger.setLevel(level)
    return log_path


def write_error_report(
    log_dir: Path,
    step: str,
    exc: BaseException,
    *,
    context: Optional[dict[str, object]] = None,
) -> Path:
    """
    Write a detailed error report to <log_dir>/<step>.error.log.
    """
    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / f"{step}.error.log").resolve()
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(f"step: {step}\n")
        handle.write(f"error: {type(exc).__name__}: {exc}\n")
        if context:
            handle.write("context:\n")
            for key, value in context.items():
                handle.write(f"  - {key}: {value}\n")
        handle.write("traceback:\n")
        handle.write(traceback.format_exc())
    return log_path
