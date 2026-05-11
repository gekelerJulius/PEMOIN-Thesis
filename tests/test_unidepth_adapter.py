from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pemoin.providers.adapters import unidepth_adapter


def test_unidepth_runner_builds_bridge_command_with_new_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    unidepth_dir = repo_root / "tools" / "UniDepth"
    unidepth_dir.mkdir(parents=True)
    (unidepth_dir / "pemoin_bridge.py").write_text("print('ok')\n", encoding="utf-8")

    image_dir = tmp_path / "frames"
    image_dir.mkdir(parents=True)
    output_dir = tmp_path / "out"
    intrinsics_path = tmp_path / "intrinsics.npz"
    np.savez(intrinsics_path, matrix=np.eye(3, dtype=np.float32))

    captured: dict[str, object] = {}

    def _fake_run(cmd, check, cwd, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(unidepth_adapter.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        unidepth_adapter,
        "_resolve_env_launcher",
        lambda env_name, env_manager: ("mamba", "run", "-n", env_name),
    )

    runner = unidepth_adapter.UniDepthRunner(repo_root=repo_root)
    settings = unidepth_adapter.UniDepthSettings(
        mamba_env="unidepth-cu121",
        model_name="lpiccinelli/unidepth-v2-vitl14",
        device="cuda",
        batch_size=4,
        amp=False,
        save_confidence=False,
        env_manager="mamba",
        repo_root=repo_root,
    )

    runner.run(settings, image_dir=image_dir, output_dir=output_dir, intrinsics_path=intrinsics_path)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:4] == ["mamba", "run", "-n", "unidepth-cu121"]
    assert "--image-dir" in cmd
    assert "--output-dir" in cmd
    assert "--model" in cmd
    assert "--batch-size" in cmd
    assert "--device" in cmd
    assert "--amp" in cmd
    assert "--save-confidence" in cmd
    assert "--intrinsics-path" in cmd

    amp_index = cmd.index("--amp")
    save_conf_index = cmd.index("--save-confidence")
    assert cmd[amp_index + 1] == "false"
    assert cmd[save_conf_index + 1] == "false"

    assert captured["check"] is True
    assert captured["cwd"] == unidepth_dir
    env = captured["env"]
    assert isinstance(env, dict)
    assert str(unidepth_dir) in str(env.get("PYTHONPATH", ""))
    assert str(output_dir / ".mamba_cache") == env.get("XDG_CACHE_HOME")
    assert env.get("HF_HOME", "").endswith(".cache/huggingface")
    assert env.get("HF_HUB_CACHE", "").endswith(".cache/huggingface/hub")
    assert env.get("TRANSFORMERS_CACHE") == env.get("HF_HUB_CACHE")
    assert env.get("HF_HUB_OFFLINE") == "1"


def test_unidepth_client_loads_depth_with_optional_confidence(tmp_path: Path):
    output_dir = tmp_path / "unidepth"
    output_dir.mkdir(parents=True)

    intrinsics = np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    np.savez_compressed(
        output_dir / "000000.npz",
        depth=np.ones((4, 5), dtype=np.float32),
        intrinsics=intrinsics,
    )
    np.savez_compressed(
        output_dir / "000001.npz",
        depth=np.full((4, 5), 2.0, dtype=np.float32),
        confidence=np.full((4, 5), 0.75, dtype=np.float32),
        intrinsics=intrinsics,
    )

    settings = unidepth_adapter.UniDepthSettings(repo_root=tmp_path)
    client = unidepth_adapter.UniDepthClient(settings=settings, output_dir=output_dir)
    client.load()

    assert client.num_frames == 2

    depth0 = client.get_depth(0)
    depth1 = client.get_depth(1)
    assert depth0.confidence is None
    assert depth1.confidence is not None
    np.testing.assert_allclose(depth1.confidence, np.full((4, 5), 0.75, dtype=np.float32))

    intrinsics_data = client.get_intrinsics()
    np.testing.assert_allclose(intrinsics_data.matrix, intrinsics)


def test_unidepth_settings_reject_non_positive_batch_size(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_size"):
        unidepth_adapter._coerce_unidepth_settings({"batch_size": 0}, repo_root=tmp_path)


def test_unidepth_resolves_rgb_subdir_when_root_has_no_images(tmp_path: Path):
    source = tmp_path / "carla_export"
    rgb_dir = source / "rgb"
    rgb_dir.mkdir(parents=True)
    (rgb_dir / "000001.jpg").write_bytes(b"")

    resolved = unidepth_adapter._resolve_image_dir(source)
    assert resolved == rgb_dir.resolve()
