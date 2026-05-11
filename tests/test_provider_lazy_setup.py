from __future__ import annotations


from pemoin.providers.adapters.dpvo_adapter import DPVOTrajectoryProvider
from pemoin.providers.adapters.unidepth_adapter import UniDepthDepthProvider


class _DummyFrame:
    def __init__(self, index: int = 0):
        self.index = index


class _DummyClient:
    def get_depth(self, frame_index: int):
        return {"frame_index": frame_index}

    def get_pose(self, frame_index: int):
        return {"frame_index": frame_index}


class _DummyAdapter:
    def __init__(self):
        self.calls = 0
        self.client = _DummyClient()

    def _ensure_ready(self):
        self.calls += 1
        return self.client


def test_unidepth_depth_provider_is_lazy() -> None:
    adapter = _DummyAdapter()
    provider = UniDepthDepthProvider(adapter=adapter)

    provider.setup(context={})
    assert adapter.calls == 0

    out = provider.process(_DummyFrame(3))
    assert adapter.calls == 1
    assert out["frame_index"] == 3

    provider.process(_DummyFrame(4))
    assert adapter.calls == 1


def test_dpvo_trajectory_provider_is_lazy() -> None:
    adapter = _DummyAdapter()
    provider = DPVOTrajectoryProvider(adapter=adapter)

    provider.setup(context={})
    assert adapter.calls == 0

    out = provider.process(_DummyFrame(7))
    assert adapter.calls == 1
    assert out["frame_index"] == 7

    provider.process(_DummyFrame(8))
    assert adapter.calls == 1
