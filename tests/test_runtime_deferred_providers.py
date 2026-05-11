from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    FrameData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceKind,
    SemanticSegment,
    SemanticsData,
)
from pemoin.providers.depth import DepthProvider
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.providers.camera_height import CameraHeightProvider
from pemoin.runtime.cache import CrossRunCacheManager
from pemoin.runtime.profiles.config import ProfileConfig, RuntimeBindings
from pemoin.runtime.runtime import Runtime


class _DummyFrameProvider:
    def __init__(self, count: int) -> None:
        self._frames = [
            FrameData(
                frame_id=f"{idx:06d}",
                index=idx,
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )
            for idx in range(count)
        ]

    def __iter__(self):
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def close(self) -> None:
        pass


class _DummyIntrinsicsProvider(IntrinsicsProvider):
    def process(self, frame: Any) -> IntrinsicsData:
        return IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "test"},
        )

    def teardown(self) -> None:
        pass


class _DeferredDepthProvider(DepthProvider):
    deferred_batch = True

    def __init__(self) -> None:
        self.frame_file_counts: list[int] = []
        self._frames_dir: Path | None = None

    def setup(self, context) -> None:
        self._frames_dir = Path(str(context["frames_dir"]))

    def process(self, frame: Any) -> DepthData:
        assert self._frames_dir is not None
        self.frame_file_counts.append(len(list(self._frames_dir.glob("*.png"))))
        return DepthData(
            frame_index=int(frame.index),
            depth=np.ones((8, 8), dtype=np.float32),
            metadata={"source": "test"},
        )

    def teardown(self) -> None:
        pass


class _MaterializingDeferredDepthProvider(_DeferredDepthProvider):
    def __init__(self) -> None:
        super().__init__()
        self.materialize_calls = 0

    def try_materialize_standardized_outputs(self, resource_store) -> bool:
        self.materialize_calls += 1
        for frame_idx in resource_store.frame_indices(ResourceKind.FRAMES):
            resource_store.save_depth(
                DepthData(
                    frame_index=int(frame_idx),
                    depth=np.ones((8, 8), dtype=np.float32),
                    metadata={"source": "materialized"},
                )
            )
        return True


class _DummyCameraHeightProvider(CameraHeightProvider):
    def __init__(self) -> None:
        super().__init__({})

    def setup(self, context: Any) -> None:
        _ = context

    def process(self, frame: Any) -> CameraHeightData:
        return CameraHeightData(
            frame_index=int(frame.index),
            height_m=1.5,
            metadata={"source": "test"},
        )

    def teardown(self) -> None:
        pass


def test_runtime_defers_depth_provider_until_all_frames_are_persisted(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = Runtime(
        ProfileConfig(
            name="test_profile",
            runtime=RuntimeBindings(
                state_window=1,
                degradation_policy="OfflineDegradationPolicy",
                settings={
                    "comparison_frame": {"enabled": False},
                    "geometry_validation": {"enabled": False},
                    "video_export": {"enabled": False},
                },
            ),
            providers={},
            effects={},
            working_resolution=(8, 8),
        )
    )
    intrinsics_provider = _DummyIntrinsicsProvider()
    depth_provider = _DeferredDepthProvider()

    monkeypatch.setattr(
        runtime,
        "build_providers",
        lambda factory, context: {
            "intrinsics": intrinsics_provider,
            "depth": depth_provider,
        },
    )

    result = runtime.run(
        _DummyFrameProvider(2),
        context={"run_dir": str(tmp_path / "run")},
    )

    assert depth_provider.frame_file_counts == [2, 2]
    assert result.processed_frames == 2
    assert result.expected_frames == 2


def test_runtime_skips_deferred_depth_replay_when_standardized_outputs_materialize(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = Runtime(
        ProfileConfig(
            name="test_profile",
            runtime=RuntimeBindings(
                state_window=1,
                degradation_policy="OfflineDegradationPolicy",
                settings={
                    "comparison_frame": {"enabled": False},
                    "geometry_validation": {"enabled": False},
                    "video_export": {"enabled": False},
                },
            ),
            providers={},
            effects={},
            working_resolution=(8, 8),
        )
    )
    intrinsics_provider = _DummyIntrinsicsProvider()
    depth_provider = _MaterializingDeferredDepthProvider()

    monkeypatch.setattr(
        runtime,
        "build_providers",
        lambda factory, context: {
            "intrinsics": intrinsics_provider,
            "depth": depth_provider,
        },
    )

    run_dir = tmp_path / "run"
    runtime.run(_DummyFrameProvider(2), context={"run_dir": str(run_dir)})

    assert depth_provider.materialize_calls == 1
    assert depth_provider.frame_file_counts == []
    assert len(list((run_dir / "standard" / "depth").glob("*.npz"))) == 2


def test_runtime_persists_timeline_with_provider_aggregate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = Runtime(
        ProfileConfig(
            name="timeline_profile",
            runtime=RuntimeBindings(
                state_window=1,
                degradation_policy="OfflineDegradationPolicy",
                settings={
                    "comparison_frame": {"enabled": False},
                    "geometry_validation": {"enabled": False},
                    "video_export": {"enabled": False},
                },
            ),
            providers={},
            effects={},
            working_resolution=(8, 8),
        )
    )
    intrinsics_provider = _DummyIntrinsicsProvider()
    monkeypatch.setattr(
        runtime,
        "build_providers",
        lambda factory, context: {
            "intrinsics": intrinsics_provider,
        },
    )

    run_dir = tmp_path / "run"
    runtime.run(_DummyFrameProvider(2), context={"run_dir": str(run_dir)})

    timeline_path = run_dir / "standard" / "runtime" / "timeline.json"
    assert timeline_path.exists()
    payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["metadata"]["processed_frames"] == 2
    frame_loop = next(stage for stage in payload["stages"] if stage["name"] == "runtime.frame_loop")
    provider_totals = next(child for child in frame_loop["children"] if child["name"] == "runtime.frame_loop.providers")
    intrinsics_stage = next(
        child for child in provider_totals["children"] if child["name"] == "runtime.frame_loop.providers.intrinsics"
    )
    assert intrinsics_stage["metadata"]["calls"] == 2


def test_runtime_skips_batch_semantics_run_when_standardized_outputs_materialize(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = Runtime(
        ProfileConfig(
            name="batch_semantics_profile",
            runtime=RuntimeBindings(
                state_window=1,
                degradation_policy="OfflineDegradationPolicy",
                settings={
                    "comparison_frame": {"enabled": False},
                    "geometry_validation": {"enabled": False},
                    "video_export": {"enabled": False},
                },
            ),
            providers={},
            effects={},
            working_resolution=(8, 8),
        )
    )
    semantics_provider = _MaterializingBatchSemanticsProvider()
    monkeypatch.setattr(
        runtime,
        "build_providers",
        lambda factory, context: {
            "intrinsics": _DummyIntrinsicsProvider(),
            "semantics": semantics_provider,
        },
    )

    run_dir = tmp_path / "run"
    runtime.run(_DummyFrameProvider(2), context={"run_dir": str(run_dir)})

    assert semantics_provider.materialize_calls == 1
    assert semantics_provider.run_count == 0
    payload = json.loads((run_dir / "standard" / "runtime" / "timeline.json").read_text(encoding="utf-8"))
    batch_semantics = next(
        stage for stage in payload["stages"] if stage["name"] == "runtime.post.batch_semantics"
    )
    assert batch_semantics["status"] == "cache_materialized"


class _CacheAwareBatchSemanticsProvider:
    batch_oriented = True
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})
    required_resources = frozenset({ResourceKind.FRAMES})

    def __init__(self) -> None:
        self.run_count = 0
        self._cache_manager: CrossRunCacheManager | None = None
        self._signature: str | None = None
        self._payload: dict[str, Any] | None = None
        self._status: dict[str, Any] = {}

    def setup(self, context: dict[str, Any]) -> None:
        cache = context.get("cross_run_cache")
        self._cache_manager = cache if isinstance(cache, CrossRunCacheManager) else None

    def teardown(self) -> None:
        pass

    def run(self, resources, context=None) -> None:
        assert self._cache_manager is not None
        frames_dir = resources.base_dir(ResourceKind.FRAMES)
        self._payload = {"frames_dir": self._cache_manager.directory_signature(frames_dir)}
        self._signature = self._cache_manager.signature("dummy-semantics", self._payload)
        lookup = self._cache_manager.lookup("dummy-semantics", self._signature)
        self._status = {
            "cross_run_cache_hit": lookup.hit,
            "cross_run_cache_validation": lookup.reason,
        }
        if lookup.hit:
            self._cache_manager.materialize("dummy-semantics", self._signature, run_root=resources.root)
            return
        self.run_count += 1
        raw_path = resources.provider_dir("dummy_semantics") / "000000.npz"
        np.savez_compressed(raw_path, marker=np.array([1], dtype=np.int32))
        label_ids = np.zeros((8, 8), dtype=np.int32)
        segment_ids = np.zeros((8, 8), dtype=np.int32)
        resources.save_semantics2d(
            SemanticsData(
                frame_index=0,
                frame_id="000000",
                segments=[
                    SemanticSegment(
                        segment_id=0,
                        label="road",
                        score=1.0,
                        mask=np.ones((8, 8), dtype=bool),
                        label_id=0,
                        area=64,
                        metadata={},
                    )
                ],
                segment_ids=segment_ids,
                label_ids=label_ids,
                metadata={"source": "dummy"},
            )
        )

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._status)

    def get_cross_run_cache_spec(self, resources) -> dict[str, Any] | None:
        if self._cache_manager is None or self._signature is None or self._payload is None:
            return None
        raw_dir = resources.raw_root / "dummy_semantics"
        sem_dir = resources.base_dir(ResourceKind.SEMANTICS_2D)
        ready = raw_dir.exists() and any(raw_dir.glob("*.npz")) and sem_dir.exists() and any(sem_dir.glob("*.npz"))
        spec = {
            "provider_id": "dummy-semantics",
            "signature": self._signature,
            "payload": self._payload,
            "artifacts": {},
            "ready": ready,
        }
        if raw_dir.exists():
            spec["artifacts"].update(self._cache_manager.collect_tree(raw_dir, rel_prefix="raw/dummy_semantics"))
        if sem_dir.exists():
            spec["artifacts"].update(self._cache_manager.collect_tree(sem_dir, rel_prefix="standard/semantics_2d"))
        if not ready:
            spec["not_ready_reason"] = "missing-semantics-artifacts"
        return spec


class _MaterializingBatchSemanticsProvider:
    batch_oriented = True
    produced_resources = frozenset({ResourceKind.SEMANTICS_2D})
    required_resources = frozenset({ResourceKind.FRAMES})

    def __init__(self) -> None:
        self.run_count = 0
        self.materialize_calls = 0

    def setup(self, context: dict[str, Any]) -> None:
        _ = context

    def teardown(self) -> None:
        pass

    def try_materialize_standardized_outputs(self, resources) -> bool:
        self.materialize_calls += 1
        for frame_idx in resources.frame_indices(ResourceKind.FRAMES):
            resources.save_semantics2d(
                SemanticsData(
                    frame_index=int(frame_idx),
                    frame_id=f"{int(frame_idx):06d}",
                    segments=[
                        SemanticSegment(
                            segment_id=0,
                            label="road",
                            score=1.0,
                            mask=np.ones((8, 8), dtype=bool),
                            label_id=0,
                            area=64,
                            metadata={},
                        )
                    ],
                    segment_ids=np.zeros((8, 8), dtype=np.int32),
                    label_ids=np.zeros((8, 8), dtype=np.int32),
                    metadata={"source": "materialized"},
                )
            )
        return True

    def run(self, resources, context=None) -> None:
        self.run_count += 1


class _CacheAwareDeferredTrajectoryProvider:
    deferred_batch = True
    produced_resources = frozenset({ResourceKind.TRAJECTORY})
    required_resources = frozenset({ResourceKind.FRAMES})

    def __init__(self) -> None:
        self.run_count = 0
        self._cache_manager: CrossRunCacheManager | None = None
        self._run_dir: Path | None = None
        self._signature: str | None = None
        self._payload: dict[str, Any] | None = None
        self._status: dict[str, Any] = {}

    def setup(self, context: dict[str, Any]) -> None:
        cache = context.get("cross_run_cache")
        self._cache_manager = cache if isinstance(cache, CrossRunCacheManager) else None
        self._run_dir = Path(str(context["run_dir"]))

    def teardown(self) -> None:
        pass

    def flush(self) -> PoseData:
        assert self._cache_manager is not None
        assert self._run_dir is not None
        frames_dir = self._run_dir / "standard" / "frames"
        self._payload = {"frames_dir": self._cache_manager.directory_signature(frames_dir)}
        self._signature = self._cache_manager.signature("dummy-trajectory", self._payload)
        lookup = self._cache_manager.lookup("dummy-trajectory", self._signature)
        self._status = {
            "cross_run_cache_hit": lookup.hit,
            "cross_run_cache_validation": lookup.reason,
        }
        if lookup.hit:
            self._cache_manager.materialize("dummy-trajectory", self._signature, run_root=self._run_dir)
        else:
            self.run_count += 1
            raw_path = self._run_dir / "raw" / "dummy_trajectory" / "result.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text("trajectory", encoding="utf-8")
        return PoseData(
            samples=[
                PoseSample(
                    frame_index=0,
                    camera_to_world=np.eye(4, dtype=np.float32),
                    world_to_camera=np.eye(4, dtype=np.float32),
                    confidence=1.0,
                    metadata={"source": "dummy"},
                ),
                PoseSample(
                    frame_index=1,
                    camera_to_world=np.eye(4, dtype=np.float32),
                    world_to_camera=np.eye(4, dtype=np.float32),
                    confidence=1.0,
                    metadata={"source": "dummy"},
                ),
            ],
            metadata={"source": "dummy"},
        )

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._status)

    def get_cross_run_cache_spec(self, resources) -> dict[str, Any] | None:
        if self._cache_manager is None or self._signature is None or self._payload is None:
            return None
        raw_dir = resources.raw_root / "dummy_trajectory"
        traj_path = resources.path_for(ResourceKind.TRAJECTORY)
        ready = raw_dir.exists() and any(raw_dir.iterdir()) and traj_path.exists()
        artifacts: dict[str, Path] = {}
        if raw_dir.exists():
            artifacts.update(self._cache_manager.collect_tree(raw_dir, rel_prefix="raw/dummy_trajectory"))
        if traj_path.exists():
            artifacts.update(self._cache_manager.collect_file(traj_path, relpath="standard/trajectory/poses.npz"))
        spec = {
            "provider_id": "dummy-trajectory",
            "signature": self._signature,
            "payload": self._payload,
            "artifacts": artifacts,
            "ready": ready,
        }
        if not ready:
            spec["not_ready_reason"] = "missing-trajectory-artifacts"
        return spec


class _CacheAwareDeferredDepthProvider(DepthProvider):
    deferred_batch = True
    produced_resources = frozenset({ResourceKind.DEPTH})
    required_resources = frozenset({ResourceKind.FRAMES})

    def __init__(self) -> None:
        self.run_count = 0
        self._cache_manager: CrossRunCacheManager | None = None
        self._run_dir: Path | None = None
        self._signature: str | None = None
        self._payload: dict[str, Any] | None = None
        self._status: dict[str, Any] = {}
        self._initialized = False

    def setup(self, context: dict[str, Any]) -> None:
        cache = context.get("cross_run_cache")
        self._cache_manager = cache if isinstance(cache, CrossRunCacheManager) else None
        self._run_dir = Path(str(context["run_dir"]))

    def teardown(self) -> None:
        pass

    def process(self, frame: Any) -> DepthData:
        assert self._cache_manager is not None
        assert self._run_dir is not None
        if not self._initialized:
            frames_dir = self._run_dir / "standard" / "frames"
            self._payload = {"frames_dir": self._cache_manager.directory_signature(frames_dir)}
            self._signature = self._cache_manager.signature("dummy-depth", self._payload)
            lookup = self._cache_manager.lookup("dummy-depth", self._signature)
            self._status = {
                "cross_run_cache_hit": lookup.hit,
                "cross_run_cache_validation": lookup.reason,
            }
            if lookup.hit:
                self._cache_manager.materialize("dummy-depth", self._signature, run_root=self._run_dir)
            else:
                self.run_count += 1
                raw_path = self._run_dir / "raw" / "dummy_depth" / "result.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text("depth", encoding="utf-8")
            self._initialized = True
        return DepthData(
            frame_index=int(frame.index),
            depth=np.ones((8, 8), dtype=np.float32),
            metadata={"source": "dummy"},
        )

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._status)

    def get_cross_run_cache_spec(self, resources) -> dict[str, Any] | None:
        if self._cache_manager is None or self._signature is None or self._payload is None:
            return None
        raw_dir = resources.raw_root / "dummy_depth"
        depth_dir = resources.base_dir(ResourceKind.DEPTH)
        ready = raw_dir.exists() and any(raw_dir.iterdir()) and depth_dir.exists() and any(depth_dir.glob("*.npz"))
        artifacts: dict[str, Path] = {}
        if raw_dir.exists():
            artifacts.update(self._cache_manager.collect_tree(raw_dir, rel_prefix="raw/dummy_depth"))
        if depth_dir.exists():
            artifacts.update(self._cache_manager.collect_tree(depth_dir, rel_prefix="standard/depth"))
        spec = {
            "provider_id": "dummy-depth",
            "signature": self._signature,
            "payload": self._payload,
            "artifacts": artifacts,
            "ready": ready,
        }
        if not ready:
            spec["not_ready_reason"] = "missing-depth-artifacts"
        return spec


class _LateFailLightingProvider:
    produced_resources = frozenset()
    required_resources = frozenset()

    def setup(self, context: dict[str, Any]) -> None:
        pass

    def teardown(self) -> None:
        pass

    def run(self, resources, context=None) -> None:
        raise RuntimeError("late-stage failure")


def test_runtime_publishes_cache_before_late_stage_failure_and_reuses_it(
    monkeypatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"

    def _make_runtime() -> Runtime:
        return Runtime(
            ProfileConfig(
                name="test_profile",
                runtime=RuntimeBindings(
                    state_window=1,
                    degradation_policy="OfflineDegradationPolicy",
                    settings={
                        "comparison_frame": {"enabled": False},
                        "geometry_validation": {"enabled": False},
                        "video_export": {"enabled": False},
                        "cross_run_cache": {"enabled": True, "root": str(cache_root)},
                    },
                ),
                providers={},
                effects={},
                working_resolution=(8, 8),
            )
        )

    monkeypatch.setattr("pemoin.runtime.runtime.validate_geometry_store", lambda *args, **kwargs: None)

    intrinsics_provider = _DummyIntrinsicsProvider()
    camera_height_provider = _DummyCameraHeightProvider()
    semantics_provider_first = _CacheAwareBatchSemanticsProvider()
    trajectory_provider_first = _CacheAwareDeferredTrajectoryProvider()
    depth_provider_first = _CacheAwareDeferredDepthProvider()
    runtime_first = _make_runtime()
    monkeypatch.setattr(
        runtime_first,
        "build_providers",
        lambda factory, context: {
            "intrinsics": intrinsics_provider,
            "camera_height": camera_height_provider,
            "semantics": semantics_provider_first,
            "trajectory": trajectory_provider_first,
            "depth": depth_provider_first,
            "lighting": _LateFailLightingProvider(),
        },
    )

    try:
        runtime_first.run(
            _DummyFrameProvider(2),
            context={"run_dir": str(tmp_path / "run_a")},
        )
    except RuntimeError as exc:
        assert "late-stage failure" in str(exc)
    else:
        raise AssertionError("Expected late-stage lighting failure")

    cache = CrossRunCacheManager(cache_root)
    assert cache.lookup("dummy-semantics", semantics_provider_first._signature or "").hit is True
    assert cache.lookup("dummy-trajectory", trajectory_provider_first._signature or "").hit is True
    assert cache.lookup("dummy-depth", depth_provider_first._signature or "").hit is True

    semantics_provider_second = _CacheAwareBatchSemanticsProvider()
    trajectory_provider_second = _CacheAwareDeferredTrajectoryProvider()
    depth_provider_second = _CacheAwareDeferredDepthProvider()
    runtime_second = _make_runtime()
    monkeypatch.setattr(
        runtime_second,
        "build_providers",
        lambda factory, context: {
            "intrinsics": _DummyIntrinsicsProvider(),
            "camera_height": _DummyCameraHeightProvider(),
            "semantics": semantics_provider_second,
            "trajectory": trajectory_provider_second,
            "depth": depth_provider_second,
        },
    )

    result = runtime_second.run(
        _DummyFrameProvider(2),
        context={"run_dir": str(tmp_path / "run_b")},
    )

    assert result.processed_frames == 2
    assert semantics_provider_first.run_count == 1
    assert trajectory_provider_first.run_count == 1
    assert depth_provider_first.run_count == 1
    assert semantics_provider_second.run_count == 0
    assert trajectory_provider_second.run_count == 0
    assert depth_provider_second.run_count == 0
