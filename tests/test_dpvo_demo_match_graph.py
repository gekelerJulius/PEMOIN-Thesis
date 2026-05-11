from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DPVO_DEMO_PATH = REPO_ROOT / "tools" / "DPVO" / "demo.py"


def _load_demo_module():
    if not DPVO_DEMO_PATH.exists():
        pytest.skip(f"DPVO demo module not available at {DPVO_DEMO_PATH}")
    fake_torch = types.SimpleNamespace(
        no_grad=lambda: (lambda fn: fn),
    )
    fake_cv2 = types.SimpleNamespace()
    fake_file_interface = types.SimpleNamespace()
    fake_cfg = types.SimpleNamespace()
    fake_dpvo = types.SimpleNamespace(DPVO=object)
    fake_plot_utils = types.SimpleNamespace(
        plot_trajectory=lambda *args, **kwargs: None,
        save_output_for_COLMAP=lambda *args, **kwargs: None,
        save_ply=lambda *args, **kwargs: None,
    )
    fake_stream = types.SimpleNamespace(
        image_stream=lambda *args, **kwargs: None,
        video_stream=lambda *args, **kwargs: None,
    )
    modules = {
        "torch": fake_torch,
        "cv2": fake_cv2,
        "evo.core.trajectory": types.SimpleNamespace(PoseTrajectory3D=object),
        "evo.tools": types.SimpleNamespace(file_interface=fake_file_interface),
        "dpvo.config": types.SimpleNamespace(cfg=fake_cfg),
        "dpvo.dpvo": fake_dpvo,
        "dpvo.plot_utils": fake_plot_utils,
        "dpvo.stream": fake_stream,
        "dpvo.utils": types.SimpleNamespace(Timer=object),
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)

    spec = importlib.util.spec_from_file_location("dpvo_demo_match_graph_test", DPVO_DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._array)

    def numel(self):
        return int(np.asarray(self._array).size)

    @property
    def shape(self):
        return np.asarray(self._array).shape

    def __getitem__(self, item):
        return _FakeTensor(np.asarray(self._array)[item])


def _fake_cat(tensors, dim=0):
    arrays = [np.asarray(t._array) for t in tensors]
    return _FakeTensor(np.concatenate(arrays, axis=dim))


class _FakeTorch(types.SimpleNamespace):
    def cat(self, tensors, dim=0):
        return _fake_cat(tensors, dim=dim)


def test_extract_match_graph_raises_motion_error_when_no_edges(monkeypatch: pytest.MonkeyPatch):
    demo = _load_demo_module()
    monkeypatch.setattr(demo, "torch", _FakeTorch(no_grad=lambda: (lambda fn: fn)))

    pg = types.SimpleNamespace(
        ii_inac=_FakeTensor(np.zeros((0,), dtype=np.int64)),
        jj_inac=_FakeTensor(np.zeros((0,), dtype=np.int64)),
        kk_inac=_FakeTensor(np.zeros((0,), dtype=np.int64)),
        ii=_FakeTensor(np.zeros((0,), dtype=np.int64)),
        jj=_FakeTensor(np.zeros((0,), dtype=np.int64)),
        kk=_FakeTensor(np.zeros((0,), dtype=np.int64)),
        target_inac=_FakeTensor(np.zeros((1, 0, 2), dtype=np.float32)),
        weight_inac=_FakeTensor(np.zeros((1, 0, 2), dtype=np.float32)),
        target=_FakeTensor(np.zeros((1, 0, 2), dtype=np.float32)),
        weight=_FakeTensor(np.zeros((1, 0, 2), dtype=np.float32)),
        tstamps_=np.zeros((0,), dtype=np.int64),
    )
    slam = types.SimpleNamespace(pg=pg, RES=4, n=3, is_initialized=False)

    with pytest.raises(demo.DPVOMatchGraphError, match="not enough motion"):
        demo._extract_match_graph(slam)


def test_extract_match_graph_combines_active_and_inactive_edges(monkeypatch: pytest.MonkeyPatch):
    demo = _load_demo_module()
    monkeypatch.setattr(demo, "torch", _FakeTorch(no_grad=lambda: (lambda fn: fn)))

    pg = types.SimpleNamespace(
        ii_inac=_FakeTensor(np.array([0], dtype=np.int64)),
        jj_inac=_FakeTensor(np.array([1], dtype=np.int64)),
        kk_inac=_FakeTensor(np.array([0], dtype=np.int64)),
        ii=_FakeTensor(np.array([1], dtype=np.int64)),
        jj=_FakeTensor(np.array([2], dtype=np.int64)),
        kk=_FakeTensor(np.array([1], dtype=np.int64)),
        target_inac=_FakeTensor(np.array([[[10.0, 20.0]]], dtype=np.float32)),
        weight_inac=_FakeTensor(np.array([[[0.2, 0.4]]], dtype=np.float32)),
        target=_FakeTensor(np.array([[[30.0, 40.0]]], dtype=np.float32)),
        weight=_FakeTensor(np.array([[[0.6, 0.8]]], dtype=np.float32)),
        tstamps_=np.array([100, 101, 102], dtype=np.int64),
    )
    patches = np.zeros((1, 2, 3, 3, 3), dtype=np.float32)
    patches[0, 0, :2, 1, 1] = np.array([1.0, 2.0], dtype=np.float32)
    patches[0, 1, :2, 1, 1] = np.array([3.0, 4.0], dtype=np.float32)
    slam = types.SimpleNamespace(
        pg=pg,
        RES=4,
        n=4,
        is_initialized=True,
        patches=_FakeTensor(patches),
    )

    payload = demo._extract_match_graph(slam)
    assert payload["edge_src_frame_id"].tolist() == [100, 101]
    assert payload["edge_tgt_frame_id"].tolist() == [101, 102]
    assert payload["edge_patch_idx"].tolist() == [0, 1]
    np.testing.assert_allclose(payload["edge_weight"], np.array([0.3, 0.7], dtype=np.float32))
