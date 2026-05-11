"""Adapter integrating PanSt3R for multi-view depth, trajectory, and intrinsics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Optional

from pemoin.coordinate_systems.trajectory_origin import save_origin_anchored_trajectory
from pemoin.data.contracts import DepthData, PoseData, ResourceKind, ResourceMissingError, ResourceStore
from pemoin.providers.base import Provider
from pemoin.providers.depth import DepthProvider
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.providers.trajectory import TrajectoryProvider
from pemoin.providers.factory import ProviderFactory
from pemoin.runtime.profiles.config import ModuleBinding
from pemoin.utils.resolution import scale_intrinsics

from .panst3r.client import PanSt3RClient, PanSt3RSettings


def _coerce_settings(raw: Mapping[str, Any]) -> PanSt3RSettings:
    bundle_path = Path(str(raw.get("bundle_path", ""))).expanduser()
    if not bundle_path:
        raise ValueError("PanSt3R adapter requires a 'bundle_path'.")
    return PanSt3RSettings(
        bundle_path=bundle_path,
        scene_name=str(raw.get("scene_name")) if raw.get("scene_name") else None,
        device=str(raw.get("device", "cuda:0")),
        precision=str(raw.get("precision", "float32")),
        working_resolution=tuple(raw["working_resolution"]) if isinstance(raw.get("working_resolution"), (list, tuple)) else None,
        standard_export_root=(
            Path(str(raw["standard_export_root"])).expanduser()
            if raw.get("standard_export_root")
            else None
        ),
    )


class PanSt3RAdapter:
    """Wraps PanSt3R batches and exposes them through PEMOIN providers."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        client_factory: Optional[Callable[[PanSt3RSettings], PanSt3RClient]] = None,
    ):
        self._settings = _coerce_settings(settings)
        self._client_factory = client_factory or PanSt3RClient
        self._client: Optional[PanSt3RClient] = None

    @property
    def client(self) -> PanSt3RClient:
        if self._client is None:
            self._client = self._client_factory(self._settings)
        self._client.initialise()
        return self._client

    def create_depth_provider(self, inference_options: Mapping[str, Any]) -> PanSt3RDepthProvider:
        return PanSt3RDepthProvider(client=self.client, inference_options=dict(inference_options))

    def create_trajectory_provider(self, inference_options: Mapping[str, Any]) -> PanSt3RTrajectoryProvider:
        return PanSt3RTrajectoryProvider(client=self.client, inference_options=dict(inference_options))

    def create_intrinsics_provider(self, inference_options: Mapping[str, Any]) -> PanSt3RIntrinsicsProvider:
        return PanSt3RIntrinsicsProvider(client=self.client, inference_options=dict(inference_options))

    def create_composite_provider(self, inference_options: Mapping[str, Any]) -> "PanSt3RCompositeProvider":
        return PanSt3RCompositeProvider(client=self.client, inference_options=dict(inference_options))

class _PanSt3RProviderBase(Provider):
    def __init__(self, client: PanSt3RClient, inference_options: Mapping[str, Any]):
        self._client = client
        self._inference_options = dict(inference_options)

    def setup(self, context: MutableMapping[str, Any]):
        self._client.initialise()
        self._working_resolution = context.get("working_resolution")

    def teardown(self):
        pass


class PanSt3RDepthProvider(_PanSt3RProviderBase, DepthProvider):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.DEPTH})

    def process(self, frame: Any):
        frames: Iterable[Any] = [frame]
        return self._client.estimate_depth(frames, self._inference_options)

    def run(self, resources: ResourceStore, context: MutableMapping[str, Any] | None = None) -> None:
        self.validate_requirements(resources)
        frames = list(resources.iter_frames())
        if not frames:
            raise ResourceMissingError("PanSt3RDepthProvider requires frames to determine frame indices.")
        depth_results = self._client.estimate_depth(frames, self._inference_options)
        self._persist_depth(resources, depth_results)

    def _persist_depth(self, resources: ResourceStore, depth_results: DepthData | List[DepthData]) -> None:
        results = depth_results if isinstance(depth_results, list) else [depth_results]
        for depth in results:
            resources.save_depth(depth)


class PanSt3RTrajectoryProvider(_PanSt3RProviderBase, TrajectoryProvider):
    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.TRAJECTORY})

    def process(self, frame: Any):
        frames: Iterable[Any] = [frame]
        metadata: MutableMapping[str, Any] = {}
        return self._client.estimate_trajectory(frames, self._inference_options, metadata)

    def run(self, resources: ResourceStore, context: MutableMapping[str, Any] | None = None) -> None:
        self.validate_requirements(resources)
        frames = list(resources.iter_frames())
        if not frames:
            raise ResourceMissingError("PanSt3RTrajectoryProvider requires frames to determine frame indices.")
        pose_results = self._client.estimate_trajectory(frames, self._inference_options, {})
        results = pose_results if isinstance(pose_results, list) else [pose_results]
        all_samples = [sample for pose in results for sample in pose.samples]
        if not all_samples:
            raise ResourceMissingError("No pose samples returned by PanSt3R trajectory estimation.")
        save_origin_anchored_trajectory(
            resources,
            PoseData(samples=all_samples, metadata=results[0].metadata),
            metadata_label="panst3r_batch_run",
        )


class PanSt3RIntrinsicsProvider(_PanSt3RProviderBase, IntrinsicsProvider):
    produced_resources = frozenset({ResourceKind.INTRINSICS})

    def process(self, frame: Any):
        intrinsics = self._client.fetch_intrinsics()
        return self._scale_intrinsics(intrinsics, frame)

    def run(self, resources: ResourceStore, context: MutableMapping[str, Any] | None = None) -> None:
        intrinsics = self._client.fetch_intrinsics()
        if context is not None:
            target = context.get("working_resolution")
            if target is not None:
                intrinsics = scale_intrinsics(intrinsics, target)
        resources.save_intrinsics(intrinsics)


class PanSt3RCompositeProvider(_PanSt3RProviderBase):
    """
    Batch provider that exports depth, trajectory, and intrinsics in one pass.
    """

    required_resources = frozenset({ResourceKind.FRAMES})
    produced_resources = frozenset({ResourceKind.DEPTH, ResourceKind.TRAJECTORY, ResourceKind.INTRINSICS})

    def run(self, resources: ResourceStore, context: MutableMapping[str, Any] | None = None) -> None:
        self.validate_requirements(resources)
        frames = list(resources.iter_frames())
        if not frames:
            raise ResourceMissingError("PanSt3RCompositeProvider requires frames to determine frame indices.")
        depth_results = self._client.estimate_depth(frames, self._inference_options)
        traj_results = self._client.estimate_trajectory(frames, self._inference_options, {})
        intrinsics = self._client.fetch_intrinsics()
        if context is not None:
            target = context.get("working_resolution")
            if target is not None:
                intrinsics = scale_intrinsics(intrinsics, target)

        depth_list = depth_results if isinstance(depth_results, list) else [depth_results]
        for depth in depth_list:
            resources.save_depth(depth)

        traj_list = traj_results if isinstance(traj_results, list) else [traj_results]
        samples = [sample for traj in traj_list for sample in traj.samples]
        if samples:
            save_origin_anchored_trajectory(
                resources,
                PoseData(samples=samples, metadata=traj_list[0].metadata),
                metadata_label="panst3r_composite_run",
            )
        resources.save_intrinsics(intrinsics)


def register_panst3r_provider_builders(factory: ProviderFactory) -> None:
    """Register PanSt3R-backed provider builders."""

    def _resolve_adapter(binding: ModuleBinding, context: MutableMapping[str, Any]) -> PanSt3RAdapter:
        adapter_settings = binding.settings.get("adapter")
        if not adapter_settings:
            raise ValueError(
                f"Provider '{binding.tool}' requires 'adapter' settings specifying PanSt3R resources."
            )
        working_res = context.get("working_resolution")
        if working_res and "working_resolution" not in adapter_settings:
            adapter_settings["working_resolution"] = list(working_res)
        store = context.get("resource_store")
        if isinstance(store, ResourceStore) and "standard_export_root" not in adapter_settings:
            adapter_settings["standard_export_root"] = str(store.standard_root / "geometry")
        cache = context.setdefault("panst3r_adapters", {})
        key = json.dumps(adapter_settings, sort_keys=True)
        if key not in cache:
            cache[key] = PanSt3RAdapter(adapter_settings)
        return cache[key]

    def build_depth(binding: ModuleBinding, context: MutableMapping[str, Any]) -> PanSt3RDepthProvider:
        adapter = _resolve_adapter(binding, context)
        inference_options = binding.settings.get("inference", {})
        return adapter.create_depth_provider(inference_options)

    def build_traj(binding: ModuleBinding, context: MutableMapping[str, Any]) -> PanSt3RTrajectoryProvider:
        adapter = _resolve_adapter(binding, context)
        inference_options = binding.settings.get("inference", {})
        return adapter.create_trajectory_provider(inference_options)

    def build_intr(binding: ModuleBinding, context: MutableMapping[str, Any]) -> PanSt3RIntrinsicsProvider:
        adapter = _resolve_adapter(binding, context)
        inference_options = binding.settings.get("inference", {})
        return adapter.create_intrinsics_provider(inference_options)

    def build_composite(binding: ModuleBinding, context: MutableMapping[str, Any]) -> PanSt3RCompositeProvider:
        adapter = _resolve_adapter(binding, context)
        inference_options = binding.settings.get("inference", {})
        return adapter.create_composite_provider(inference_options)

    factory.register("PanSt3RDepthProvider", build_depth)
    factory.register("PanSt3RTrajectoryProvider", build_traj)
    factory.register("PanSt3RIntrinsicsProvider", build_intr)
    factory.register("PanSt3RCompositeProvider", build_composite)
