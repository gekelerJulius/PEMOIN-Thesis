from __future__ import annotations

import pytest

from pemoin.runtime.orchestration.frame_provider_builder import (
    create_frame_provider_from_binding,
)
from pemoin.runtime.orchestration.nuscenes_frame_provider import (
    NuScenesFrameProvider,
    _NuScenesFrameEntry,
    _derive_effective_fps,
)
from pemoin.runtime.profiles.config import ModuleBinding


def test_derive_effective_fps_uses_median_timestamp_delta():
    entries = [
        _NuScenesFrameEntry(0, 0.00, "a.png", "s0", "c0", True),
        _NuScenesFrameEntry(1, 0.50, "b.png", "s1", "c1", True),
        _NuScenesFrameEntry(2, 1.01, "c.png", "s2", "c2", True),
        _NuScenesFrameEntry(3, 1.50, "d.png", "s3", "c3", True),
    ]

    assert _derive_effective_fps(entries) == pytest.approx(2.0, rel=1e-6)


def test_derive_effective_fps_rejects_non_increasing_timestamps():
    entries = [
        _NuScenesFrameEntry(0, 5.0, "a.png", "s0", "c0", True),
        _NuScenesFrameEntry(1, 5.0, "b.png", "s1", "c1", True),
    ]

    with pytest.raises(ValueError, match="could not derive effective FPS"):
        _derive_effective_fps(entries)


def test_builder_merges_runtime_settings_for_nuscenes(monkeypatch, tmp_path):
    source_dir = tmp_path / "nusc"
    source_dir.mkdir()

    def fake_open(self, source) -> None:
        self._source_fps = 2.0
        self._effective_fps = 2.0
        self._opened = True

    monkeypatch.setattr(NuScenesFrameProvider, "open", fake_open)

    binding = ModuleBinding(
        tool="NuScenesFrameProvider",
        settings={
            "path": str(source_dir),
            "version": "v1.0-mini",
            "scene_index": 0,
            "camera": "CAM_FRONT",
        },
    )

    _, resolved_source, provider_info = create_frame_provider_from_binding(
        binding,
        config_base=tmp_path,
    )

    assert resolved_source == source_dir.resolve()
    assert "sampling_fps" not in binding.settings
    assert provider_info["settings"]["source_sampling_fps"] == pytest.approx(2.0)
    assert provider_info["settings"]["sampling_fps"] == pytest.approx(2.0)
    assert provider_info["settings"]["resolved_sampling_fps"] == pytest.approx(2.0)
    assert provider_info["settings"]["timing_source"] == "derived_from_timestamps"


class _DummyNuScenes:
    def __init__(self) -> None:
        self.scene = [
            {
                "name": "scene-0001",
                "first_sample_token": "sample-0",
            }
        ]
        self._sample = {
            "sample-0": {"data": {"CAM_FRONT": "sd-0"}, "next": "sample-1"},
            "sample-1": {"data": {"CAM_FRONT": "sd-2"}, "next": ""},
        }
        self._sample_data = {
            "sd-0": {
                "token": "sd-0",
                "timestamp": 0,
                "filename": "samples/CAM_FRONT/000.png",
                "is_key_frame": True,
                "next": "sd-1",
                "prev": "",
            },
            "sd-1": {
                "token": "sd-1",
                "timestamp": 100_000,
                "filename": "sweeps/CAM_FRONT/001.png",
                "is_key_frame": False,
                "next": "sd-2",
                "prev": "sd-0",
            },
            "sd-2": {
                "token": "sd-2",
                "timestamp": 500_000,
                "filename": "samples/CAM_FRONT/002.png",
                "is_key_frame": True,
                "next": "sd-3",
                "prev": "sd-1",
            },
            "sd-3": {
                "token": "sd-3",
                "timestamp": 600_000,
                "filename": "sweeps/CAM_FRONT/003.png",
                "is_key_frame": False,
                "next": "",
                "prev": "sd-2",
            },
        }

    def get(self, table: str, token: str) -> dict:
        if table == "sample":
            return self._sample[token]
        if table == "sample_data":
            return self._sample_data[token]
        raise AssertionError(f"unexpected table {table}")


def test_nuscenes_frame_provider_all_camera_frames_keeps_sweeps(monkeypatch, tmp_path):
    dummy_module = type("DummyModule", (), {"NuScenes": lambda **_: _DummyNuScenes()})
    monkeypatch.setitem(__import__("sys").modules, "nuscenes.nuscenes", dummy_module)

    provider = NuScenesFrameProvider(
        {
            "version": "v1.0-mini",
            "scene_index": 0,
            "camera": "CAM_FRONT",
            "sampling_mode": "all_camera_frames",
        },
        load_images=False,
    )
    provider.open(tmp_path)

    assert len(provider) == 4
    frame0 = provider.read()
    frame1 = provider.read()
    assert frame0 is not None and frame1 is not None
    assert frame0.metadata["source_is_key_frame"] is True
    assert frame1.metadata["source_is_key_frame"] is False
    assert frame1.metadata["cam_sd_token"] == "sd-1"
    assert frame1.metadata["sample_token"] is None


def test_nuscenes_frame_provider_sampling_fps_caps_all_camera_frames(monkeypatch, tmp_path):
    dummy_module = type("DummyModule", (), {"NuScenes": lambda **_: _DummyNuScenes()})
    monkeypatch.setitem(__import__("sys").modules, "nuscenes.nuscenes", dummy_module)

    provider = NuScenesFrameProvider(
        {
                "version": "v1.0-mini",
                "scene_index": 0,
                "camera": "CAM_FRONT",
                "sampling_mode": "all_camera_frames",
                "sampling_fps": 5.0,
            },
            load_images=False,
        )
    provider.open(tmp_path)

    assert len(provider) == 2
    runtime = provider.runtime_settings()
    assert runtime["source_sampling_fps"] == pytest.approx(10.0)
    assert runtime["sampling_fps"] == pytest.approx(2.0)
    assert runtime["resolved_sampling_fps"] == pytest.approx(2.0)
