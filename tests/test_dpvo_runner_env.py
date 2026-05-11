from __future__ import annotations

from pathlib import Path

import pytest

from pemoin.providers.adapters import dpvo_adapter


def test_dpvo_runner_sets_xdg_cache_for_mamba_like_launchers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    dpvo_dir = repo_root / "tools" / "DPVO"
    dpvo_dir.mkdir(parents=True)
    (dpvo_dir / "pemoin_bridge.py").write_text("print('ok')\n", encoding="utf-8")

    image_dir = tmp_path / "frames"
    image_dir.mkdir(parents=True)
    calib_file = tmp_path / "calib.txt"
    calib_file.write_text("1 1 1 1\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    captured: dict[str, object] = {}

    def _fake_run(cmd, check, cwd, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(dpvo_adapter.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        dpvo_adapter,
        "_resolve_env_launcher",
        lambda env_name, env_manager: ("micromamba", "run", "-n", env_name),
    )

    runner = dpvo_adapter.DPVORunner(repo_root=repo_root)
    settings = dpvo_adapter.DPVOSettings(
        mamba_env="dpvo",
        env_manager="micromamba",
        repo_root=repo_root,
    )

    runner.run(settings, image_dir=image_dir, calib_file=calib_file, output_dir=output_dir)

    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("XDG_CACHE_HOME") == str(output_dir / ".mamba_cache")
    assert captured["check"] is True
    assert captured["cwd"] == dpvo_dir
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--memory-diag-sample-every" in cmd
    assert "--memory-diag-warn-ratio" in cmd
