from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pemoin.providers.adapters import dpvo_adapter


REPO_ROOT = Path(__file__).resolve().parents[1]
DPVO_BRIDGE_PATH = REPO_ROOT / "tools" / "DPVO" / "pemoin_bridge.py"


def _load_dpvo_bridge(module_name: str):
    import importlib.util
    import sys

    if not DPVO_BRIDGE_PATH.exists():
        pytest.skip(f"DPVO bridge not available at {DPVO_BRIDGE_PATH}")
    sys.path.insert(0, str(DPVO_BRIDGE_PATH.parent))
    spec = importlib.util.spec_from_file_location(module_name, DPVO_BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


def test_dpvo_resolves_rgb_subdir_when_root_has_no_images(tmp_path: Path):
    source = tmp_path / "carla_export"
    rgb_dir = source / "rgb"
    rgb_dir.mkdir(parents=True)
    (rgb_dir / "000001.jpg").write_bytes(b"")

    resolved = dpvo_adapter._resolve_image_dir(source)
    assert resolved == rgb_dir.resolve()


def test_dpvo_resolve_image_dir_fails_when_no_images(tmp_path: Path):
    source = tmp_path / "empty_export"
    source.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        dpvo_adapter._resolve_image_dir(source)


def test_dpvo_cfg_opts_must_be_even_pairs(tmp_path: Path):
    with pytest.raises(ValueError, match="KEY VALUE pairs"):
        dpvo_adapter._coerce_dpvo_settings(
            {"cfg_opts": ["BUFFER_SIZE"]},
            repo_root=tmp_path,
        )


def test_dpvo_memory_diag_settings_are_validated(tmp_path: Path):
    with pytest.raises(ValueError, match="sample_every_n_frames"):
        dpvo_adapter._coerce_dpvo_settings(
            {"memory_diagnostics": {"sample_every_n_frames": 0}},
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="warn_vram_used_ratio"):
        dpvo_adapter._coerce_dpvo_settings(
            {"memory_diagnostics": {"warn_vram_used_ratio": 1.5}},
            repo_root=tmp_path,
        )


def test_dpvo_allocator_defaults_to_native(tmp_path: Path):
    settings = dpvo_adapter._coerce_dpvo_settings({}, repo_root=tmp_path)
    assert settings.allocator_mode == "native"
    assert settings.allocator_max_split_size_mb is None


def test_dpvo_allocator_settings_are_validated(tmp_path: Path):
    with pytest.raises(ValueError, match="allocator_mode"):
        dpvo_adapter._coerce_dpvo_settings(
            {"allocator_mode": "broken"},
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="allocator_max_split_size_mb"):
        dpvo_adapter._coerce_dpvo_settings(
            {"allocator_max_split_size_mb": 0},
            repo_root=tmp_path,
        )


def test_dpvo_memory_guard_settings_are_validated(tmp_path: Path):
    with pytest.raises(ValueError, match="memory_guard.warmup_frames"):
        dpvo_adapter._coerce_dpvo_settings(
            {"memory_guard": {"warmup_frames": -1}},
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="abort_reserved_vram_ratio"):
        dpvo_adapter._coerce_dpvo_settings(
            {"memory_guard": {"abort_reserved_vram_ratio": 1.5}},
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="abort_reserved_to_allocated_ratio"):
        dpvo_adapter._coerce_dpvo_settings(
            {"memory_guard": {"abort_reserved_to_allocated_ratio": 0.0}},
            repo_root=tmp_path,
        )


def test_dpvo_runner_native_allocator_unsets_alloc_conf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runner = dpvo_adapter.DPVORunner(repo_root=tmp_path)
    bridge_script = tmp_path / "tools" / "DPVO" / "pemoin_bridge.py"
    bridge_script.parent.mkdir(parents=True)
    bridge_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_resolve_env_launcher(mamba_env: str, env_manager: str | None):
        assert mamba_env == "dpvo"
        assert env_manager is None
        return ["micromamba", "run", "-n", "dpvo"]

    def _fake_run(cmd, check, cwd, env):
        captured["cmd"] = list(cmd)
        captured["cwd"] = str(cwd)
        captured["env"] = dict(env)
        return None

    monkeypatch.setattr(dpvo_adapter, "_resolve_env_launcher", _fake_resolve_env_launcher)
    monkeypatch.setattr(dpvo_adapter.subprocess, "run", _fake_run)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    settings = dpvo_adapter.DPVOSettings(repo_root=tmp_path, allocator_mode="native")
    runner.run(
        settings=settings,
        image_dir=tmp_path,
        calib_file=tmp_path / "calib.txt",
        output_dir=tmp_path / "out",
    )

    env = captured["env"]
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env


def test_dpvo_runner_sets_requested_allocator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runner = dpvo_adapter.DPVORunner(repo_root=tmp_path)
    bridge_script = tmp_path / "tools" / "DPVO" / "pemoin_bridge.py"
    bridge_script.parent.mkdir(parents=True)
    bridge_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_resolve_env_launcher(mamba_env: str, env_manager: str | None):
        return ["micromamba", "run", "-n", mamba_env]

    def _fake_run(cmd, check, cwd, env):
        captured["env"] = dict(env)
        return None

    monkeypatch.setattr(dpvo_adapter, "_resolve_env_launcher", _fake_resolve_env_launcher)
    monkeypatch.setattr(dpvo_adapter.subprocess, "run", _fake_run)

    settings = dpvo_adapter.DPVOSettings(
        repo_root=tmp_path,
        allocator_mode="expandable_segments",
        allocator_max_split_size_mb=256,
    )
    runner.run(
        settings=settings,
        image_dir=tmp_path,
        calib_file=tmp_path / "calib.txt",
        output_dir=tmp_path / "out",
    )

    env = captured["env"]
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True,max_split_size_mb:256"


def test_dpvo_bridge_wraps_motionless_match_graph_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import argparse
    import sys
    import types

    bridge = _load_dpvo_bridge("dpvo_pemoin_bridge_test")

    class _BaseCfg:
        MIXED_PRECISION = True
        BUFFER_SIZE = 8
        PATCHES_PER_FRAME = 96
        OPTIMIZATION_WINDOW = 12
        PATCH_LIFETIME = 12
        REMOVAL_WINDOW = 6
        KEYFRAME_INDEX = 4

        def merge_from_file(self, path):
            return None

        def merge_from_list(self, opts):
            return None

    class _CudaProps:
        total_memory = 1024

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            reset_peak_memory_stats=lambda: None,
            current_device=lambda: 0,
            get_device_properties=lambda idx: _CudaProps(),
            get_device_name=lambda idx: "Fake GPU",
            memory_allocated=lambda idx: 0,
            memory_reserved=lambda idx: 0,
            max_memory_allocated=lambda idx: 0,
            max_memory_reserved=lambda idx: 0,
        ),
        Tensor=type("FakeTensor", (), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class _MatchGraphError(RuntimeError):
        pass

    def _demo_run(*args, **kwargs):
        raise _MatchGraphError(
            "Scene has not enough motion for DPVO initialization; no trajectory match graph could be produced."
        )

    demo_module = types.SimpleNamespace(
        DPVOMatchGraphError=_MatchGraphError,
        run=_demo_run,
    )
    monkeypatch.setitem(sys.modules, "demo", demo_module)
    cfg_module = types.SimpleNamespace(cfg=_BaseCfg())
    monkeypatch.setitem(sys.modules, "dpvo.config", cfg_module)

    args = argparse.Namespace(
        imagedir="frames",
        calib="calib.txt",
        output_dir=str(tmp_path),
        network="dpvo.pth",
        config="config.yaml",
        stride=1,
        skip=0,
        device="cuda",
        precision_mode="amp_fp16",
        opts=[],
        mask_dir=None,
        memory_diag_sample_every=1,
        memory_diag_warn_ratio=0.92,
        memory_guard_enabled=1,
        memory_guard_warmup_frames=1,
        memory_guard_abort_reserved_vram_ratio=0.70,
        memory_guard_abort_reserved_to_allocated_ratio=12.0,
    )

    with pytest.raises(RuntimeError, match="not enough motion"):
        bridge.run_dpvo(args)

    diagnostics_path = tmp_path / "dpvo_memory_diagnostics.json"
    assert diagnostics_path.exists()
    payload = diagnostics_path.read_text(encoding="utf-8")
    assert "not enough motion" in payload.lower()


def test_dpvo_bridge_classifies_allocator_assert_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import argparse
    import json
    import sys
    import types

    bridge = _load_dpvo_bridge("dpvo_pemoin_bridge_allocator_test")

    class _BaseCfg:
        MIXED_PRECISION = True
        BUFFER_SIZE = 8
        PATCHES_PER_FRAME = 96
        OPTIMIZATION_WINDOW = 12
        PATCH_LIFETIME = 12
        REMOVAL_WINDOW = 6
        KEYFRAME_INDEX = 4

        def merge_from_file(self, path):
            return None

        def merge_from_list(self, opts):
            return None

    class _CudaProps:
        total_memory = 1024

    fake_torch = types.SimpleNamespace(
        __version__="2.3.1",
        version=types.SimpleNamespace(cuda="12.1"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            reset_peak_memory_stats=lambda: None,
            current_device=lambda: 0,
            get_device_properties=lambda idx: _CudaProps(),
            get_device_name=lambda idx: "Fake GPU",
            memory_allocated=lambda idx: 64,
            memory_reserved=lambda idx: 128,
            max_memory_allocated=lambda idx: 64,
            max_memory_reserved=lambda idx: 128,
        ),
        Tensor=type("FakeTensor", (), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_scatter", types.SimpleNamespace(__version__="2.1.2"))

    class _MatchGraphError(RuntimeError):
        pass

    def _demo_run(*args, **kwargs):
        raise RuntimeError(
            '!block->expandable_segment_ INTERNAL ASSERT FAILED at "/tmp/CUDACachingAllocator.cpp":2549'
        )

    demo_module = types.SimpleNamespace(
        DPVOMatchGraphError=_MatchGraphError,
        run=_demo_run,
    )
    monkeypatch.setitem(sys.modules, "demo", demo_module)
    cfg_module = types.SimpleNamespace(cfg=_BaseCfg())
    monkeypatch.setitem(sys.modules, "dpvo.config", cfg_module)

    args = argparse.Namespace(
        imagedir="frames",
        calib="calib.txt",
        output_dir=str(tmp_path),
        network="dpvo.pth",
        config="config.yaml",
        stride=1,
        skip=0,
        device="cuda",
        precision_mode="amp_fp16",
        opts=[],
        mask_dir=None,
        memory_diag_sample_every=1,
        memory_diag_warn_ratio=0.92,
        memory_guard_enabled=1,
        memory_guard_warmup_frames=1,
        memory_guard_abort_reserved_vram_ratio=0.70,
        memory_guard_abort_reserved_to_allocated_ratio=12.0,
    )

    with pytest.raises(RuntimeError, match="expandable_segment_"):
        bridge.run_dpvo(args)

    payload = json.loads((tmp_path / "dpvo_memory_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["failure"]["classification"] == "allocator_internal_assert"


def test_dpvo_bridge_guard_classifies_runaway_reserved_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import argparse
    import json
    import sys
    import types

    bridge = _load_dpvo_bridge("dpvo_pemoin_bridge_guard_test")

    class _BaseCfg:
        MIXED_PRECISION = True
        BUFFER_SIZE = 8
        PATCHES_PER_FRAME = 96
        OPTIMIZATION_WINDOW = 12
        PATCH_LIFETIME = 12
        REMOVAL_WINDOW = 6
        KEYFRAME_INDEX = 4

        def merge_from_file(self, path):
            return None

        def merge_from_list(self, opts):
            return None

    class _CudaProps:
        total_memory = 1000

    state = {"allocated": 50, "reserved": 900}

    fake_torch = types.SimpleNamespace(
        __version__="2.3.1",
        version=types.SimpleNamespace(cuda="12.1"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            reset_peak_memory_stats=lambda: None,
            current_device=lambda: 0,
            get_device_properties=lambda idx: _CudaProps(),
            get_device_name=lambda idx: "Fake GPU",
            memory_allocated=lambda idx: state["allocated"],
            memory_reserved=lambda idx: state["reserved"],
            max_memory_allocated=lambda idx: state["allocated"],
            max_memory_reserved=lambda idx: state["reserved"],
        ),
        Tensor=type("FakeTensor", (), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_scatter", types.SimpleNamespace(__version__="2.1.2"))

    class _MatchGraphError(RuntimeError):
        pass

    def _demo_run(*args, **kwargs):
        kwargs["frame_observer"](24)
        return (
            (np.zeros((1, 7), dtype=np.float32), np.zeros((1,), dtype=np.int64)),
            (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8), None),
            {
                "schema_version": np.int32(2),
                "coord_space": np.array("full_res_pixels"),
                "res_factor": np.int32(4),
                "edge_src_frame_id": np.zeros((1,), dtype=np.int32),
                "edge_tgt_frame_id": np.zeros((1,), dtype=np.int32),
                "edge_src_node_idx": np.zeros((1,), dtype=np.int32),
                "edge_tgt_node_idx": np.zeros((1,), dtype=np.int32),
                "edge_patch_idx": np.zeros((1,), dtype=np.int32),
                "src_uv": np.zeros((1, 2), dtype=np.float32),
                "tgt_uv": np.zeros((1, 2), dtype=np.float32),
                "edge_weight": np.zeros((1,), dtype=np.float32),
                "edge_timestamp_src": np.zeros((1,), dtype=np.int64),
                "edge_timestamp_tgt": np.zeros((1,), dtype=np.int64),
            },
        )

    demo_module = types.SimpleNamespace(
        DPVOMatchGraphError=_MatchGraphError,
        run=_demo_run,
    )
    monkeypatch.setitem(sys.modules, "demo", demo_module)
    cfg_module = types.SimpleNamespace(cfg=_BaseCfg())
    monkeypatch.setitem(sys.modules, "dpvo.config", cfg_module)

    args = argparse.Namespace(
        imagedir="frames",
        calib="calib.txt",
        output_dir=str(tmp_path),
        network="dpvo.pth",
        config="config.yaml",
        stride=1,
        skip=0,
        device="cuda",
        precision_mode="amp_fp16",
        opts=[],
        mask_dir=None,
        memory_diag_sample_every=1,
        memory_diag_warn_ratio=0.92,
        memory_guard_enabled=1,
        memory_guard_warmup_frames=1,
        memory_guard_abort_reserved_vram_ratio=0.70,
        memory_guard_abort_reserved_to_allocated_ratio=12.0,
    )

    with pytest.raises(RuntimeError, match="runaway CUDA reserved memory"):
        bridge.run_dpvo(args)

    payload = json.loads((tmp_path / "dpvo_memory_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["failure"]["classification"] == "allocator_runaway_reserved_memory"
