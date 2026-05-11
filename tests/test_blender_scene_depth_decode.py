from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from pemoin.visualization.blender_scene.depth_decode import (
    _read_depth_exr,
    materialize_depth_npz_from_exr_sequence,
)


class _FakeInputFile:
    def __init__(self, path: str, channels: dict[str, bytes], width: int, height: int):
        self.path = path
        self._channels = channels
        self._width = width
        self._height = height

    def header(self):
        return {
            "channels": {name: object() for name in self._channels},
            "dataWindow": SimpleNamespace(
                min=SimpleNamespace(x=0, y=0),
                max=SimpleNamespace(x=self._width - 1, y=self._height - 1),
            ),
        }

    def channel(self, name: str, _pixel_type):
        return self._channels[name]

    def close(self):
        return None


def _install_fake_openexr(monkeypatch, *, arrays_by_name: dict[str, np.ndarray]) -> None:
    channels = {
        name: np.asarray(arr, dtype=np.float32).tobytes()
        for name, arr in arrays_by_name.items()
    }
    sample = next(iter(arrays_by_name.values()))
    height, width = map(int, sample.shape)
    openexr = ModuleType("OpenEXR")
    openexr.InputFile = lambda path: _FakeInputFile(path, channels, width, height)
    imath = ModuleType("Imath")

    class _PixelType:
        FLOAT = "FLOAT"

        def __init__(self, value):
            self.value = value

    imath.PixelType = _PixelType
    monkeypatch.setitem(sys.modules, "OpenEXR", openexr)
    monkeypatch.setitem(sys.modules, "Imath", imath)


def test_read_depth_exr_decodes_float_channel(monkeypatch) -> None:
    expected = np.array([[1.0, 2.5], [0.0, 4.0]], dtype=np.float32)
    _install_fake_openexr(monkeypatch, arrays_by_name={"Depth.V": expected})

    depth = _read_depth_exr(Path("/tmp/fake.exr"))

    assert depth.shape == (2, 2)
    np.testing.assert_allclose(depth, expected)


def test_materialize_depth_npz_from_exr_sequence_writes_npz_and_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    depth_exr_dir = tmp_path / "depth_exr"
    depth_exr_dir.mkdir()
    (depth_exr_dir / "depth_000001.exr").write_bytes(b"fake")
    expected = np.array([[3.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    _install_fake_openexr(monkeypatch, arrays_by_name={"Depth.V": expected})

    depth_output_dir = tmp_path / "depth_npz"
    materialize_depth_npz_from_exr_sequence(
        depth_exr_dir=depth_exr_dir,
        depth_output_dir=depth_output_dir,
        blender_version="5.0.1",
        export_api="compositing_node_group",
    )

    with np.load(depth_output_dir / "000001.npz", allow_pickle=True) as data:
        np.testing.assert_allclose(np.asarray(data["depth"], dtype=np.float32), expected)
    metadata = json.loads((depth_output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "z_pass_exr"
    assert metadata["blender_version"] == "5.0.1"
    assert metadata["export_api"] == "compositing_node_group"
    assert metadata["channel_name"] == "Depth.V"
