from __future__ import annotations

import io
from pathlib import Path

from pemoin.utils.logging import (
    RuntimeTimeline,
    resolve_console_logging_config,
    run_logged_subprocess,
)


class _FakePopen:
    def __init__(
        self,
        cmd,
        *,
        stdout_text: str = "",
        stderr_text: str = "",
        returncode: int = 0,
        **_: object,
    ) -> None:
        self.cmd = list(cmd)
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)
        self._returncode = int(returncode)

    def wait(self) -> int:
        return self._returncode


def test_resolve_console_logging_config_defaults_to_summary_mode() -> None:
    config = resolve_console_logging_config()

    assert config.summary_only is True
    assert config.stream_blender_subprocess_output is False
    assert config.show_progress_bars is True


def test_resolve_console_logging_config_verbose_streams_blender_output() -> None:
    config = resolve_console_logging_config(verbose=True)

    assert config.summary_only is False
    assert config.stream_blender_subprocess_output is False
    assert config.show_progress_bars is True


def test_run_logged_subprocess_writes_logs_and_captures_failure_tail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, **kwargs: _FakePopen(
            cmd,
            stdout_text="line 1\nline 2\n",
            stderr_text="err 1\nerr 2\n",
            returncode=7,
            **kwargs,
        ),
    )

    result = run_logged_subprocess(
        ["blender", "--background"],
        stdout_log_path=tmp_path / "stdout.log",
        stderr_log_path=tmp_path / "stderr.log",
        stream_output=False,
    )

    assert result.returncode == 7
    assert result.stdout_log_path.read_text(encoding="utf-8") == "line 1\nline 2\n"
    assert result.stderr_log_path.read_text(encoding="utf-8") == "err 1\nerr 2\n"
    assert result.stdout_tail == ("line 1", "line 2")
    assert result.stderr_tail == ("err 1", "err 2")
    failure = result.format_failure(label="Blender run")
    assert "Blender run failed (exit code 7)." in failure
    assert "stderr tail:" in failure
    assert "stdout tail:" in failure


def test_run_logged_subprocess_renders_progress_events_without_leaking_to_failure_tail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    progress_events: list[tuple[str, object]] = []

    class _FakeTqdm:
        def __init__(self, total=None, desc=None, unit=None, dynamic_ncols=None, leave=None):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0
            progress_events.append(("begin", desc))

        def update(self, amount: int) -> None:
            self.n += int(amount)
            progress_events.append(("update", self.n))

        def set_postfix_str(self, value: str, refresh: bool = True) -> None:
            _ = refresh
            progress_events.append(("message", value))

        def close(self) -> None:
            progress_events.append(("close", self.n))

    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, **kwargs: _FakePopen(
            cmd,
            stdout_text=(
                'PEMOIN_PROGRESS {"event":"begin","id":"ped","label":"Pedestrian render","total":3,"unit":"frame"}\n'
                'PEMOIN_PROGRESS {"event":"step","id":"ped","current":1,"total":3,"unit":"frame"}\n'
                'PEMOIN_PROGRESS {"event":"step","id":"ped","current":3,"total":3,"unit":"frame"}\n'
                'PEMOIN_PROGRESS {"event":"end","id":"ped","current":3,"total":3}\n'
                "regular child log\n"
            ),
            stderr_text="",
            returncode=0,
            **kwargs,
        ),
    )
    monkeypatch.setattr("pemoin.utils.logging.tqdm", _FakeTqdm)

    result = run_logged_subprocess(
        ["blender", "--background"],
        stdout_log_path=tmp_path / "stdout.log",
        stderr_log_path=tmp_path / "stderr.log",
        stream_output=False,
        show_progress=True,
    )

    assert result.returncode == 0
    assert ("begin", "Pedestrian render") in progress_events
    assert ("update", 1) in progress_events
    assert ("update", 3) in progress_events
    assert ("close", 3) in progress_events
    assert result.stdout_tail == ("regular child log",)


def test_runtime_timeline_serializes_nested_stage_tree() -> None:
    timeline = RuntimeTimeline(console_enabled=False)

    with timeline.stage("runtime.setup", display_name="Runtime setup"):
        pass
    with timeline.stage("runtime.post.blender_scene", display_name="Blender scene") as scope:
        scope.record.metadata["cache_enabled"] = True
        timeline.add_completed_stage(
            "blender_scene.cache_lookup",
            display_name="Blender cache lookup",
            status="cache_hit",
            duration_s=0.0,
            metadata={"reason": "test-hit"},
            parent=scope.record,
        )
    timeline.finalize(status="completed", metadata={"processed_frames": 3})

    payload = timeline.to_mapping()

    assert payload["status"] == "completed"
    assert payload["metadata"]["processed_frames"] == 3
    assert payload["stages"][0]["name"] == "runtime.setup"
    assert payload["stages"][1]["name"] == "runtime.post.blender_scene"
    assert payload["stages"][1]["children"][0]["status"] == "cache_hit"
