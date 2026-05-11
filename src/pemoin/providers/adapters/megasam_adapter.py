"""MegaSAM adapter exposing depth, trajectory, and intrinsics providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

from pemoin.data.contracts import IntrinsicsData, ResourceKind, ResourceStore
from pemoin.providers.depth import DepthProvider
from pemoin.providers.factory import ProviderFactory
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.providers.trajectory import TrajectoryProvider
from pemoin.runtime.profiles.config import ModuleBinding

from .megasam.client import MegaSAMClient, MegaSAMSettings, ensure_megasam_log_handler


def _coerce_resolution(value: object) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        h = int(value[0])
        w = int(value[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"working_resolution must be positive, got {value!r}")
        return (h, w)
    if isinstance(value, (int, float)):
        v = int(value)
        if v <= 0:
            raise ValueError(f"working_resolution must be > 0, got {value!r}")
        # Scalar profile resolution is max-side in runtime; exact H/W comes from frame metadata.
        return None
    raise ValueError(f"Unsupported working_resolution value: {value!r}")


def _coerce_settings(raw: Mapping[str, Any]) -> MegaSAMSettings:
    if "checkpoint_path" not in raw:
        raise ValueError("MegaSAM adapter requires 'checkpoint_path'.")
    if "config_path" not in raw:
        raise ValueError("MegaSAM adapter requires 'config_path'.")
    if "cache_dir" not in raw:
        raise ValueError("MegaSAM adapter requires 'cache_dir'.")

    return MegaSAMSettings(
        checkpoint_path=Path(str(raw["checkpoint_path"])),
        config_path=Path(str(raw["config_path"])),
        cache_dir=Path(str(raw["cache_dir"])),
        device=str(raw.get("device", "cuda:0")),
        precision=str(raw.get("precision", "float16")),
        bundle_path=Path(str(raw["bundle_path"])) if raw.get("bundle_path") else None,
        scene_name=str(raw.get("scene_name")) if raw.get("scene_name") else None,
        repository_root=Path(str(raw["repository_root"])) if raw.get("repository_root") else None,
        working_resolution=_coerce_resolution(raw.get("working_resolution")),
        standard_export_root=Path(str(raw["standard_export_root"])).expanduser()
        if raw.get("standard_export_root")
        else None,
        frame_index_map_path=Path(str(raw["frame_index_map_path"])).expanduser()
        if raw.get("frame_index_map_path")
        else None,
        tracking_preprocess_path=Path(str(raw["tracking_preprocess_path"])).expanduser()
        if raw.get("tracking_preprocess_path")
        else None,
        gt_intrinsics_path=Path(str(raw["gt_intrinsics_path"])).expanduser()
        if raw.get("gt_intrinsics_path")
        else None,
        require_final_bundle=bool(raw.get("require_final_bundle", True)),
        enforce_gt_intrinsics=bool(raw.get("enforce_gt_intrinsics", True)),
        write_debug_artifacts=bool(raw.get("write_debug_artifacts", True)),
    )


class MegaSAMAdapter:
    """Wrap MegaSAM bundle data behind PEMOIN providers."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        client_factory: Optional[Callable[[MegaSAMSettings], MegaSAMClient]] = None,
    ):
        self._settings = _coerce_settings(settings)
        self._client_factory = client_factory or MegaSAMClient
        self._client: Optional[MegaSAMClient] = None

    @property
    def client(self) -> MegaSAMClient:
        if self._client is None:
            self._client = self._client_factory(self._settings)
        self._client.initialise()
        return self._client

    def create_depth_provider(self) -> "MegaSAMDepthProvider":
        return MegaSAMDepthProvider(client=self.client)

    def create_trajectory_provider(self) -> "MegaSAMTrajectoryProvider":
        return MegaSAMTrajectoryProvider(client=self.client)

    def create_intrinsics_provider(self) -> "MegaSAMIntrinsicsProvider":
        return MegaSAMIntrinsicsProvider(client=self.client)


class _MegaSAMProviderBase:
    def __init__(self, client: MegaSAMClient):
        self._client = client

    def setup(self, context: MutableMapping[str, Any]) -> None:
        self._client.initialise()

    def _apply_runtime_intrinsics(self, frame: Any) -> None:
        metadata = getattr(frame, "metadata", {}) or {}
        intrinsics = metadata.get("intrinsics") if isinstance(metadata, Mapping) else None
        if isinstance(intrinsics, IntrinsicsData):
            self._client.set_runtime_intrinsics_override(intrinsics)

    def teardown(self) -> None:
        # MegaSAM client is intentionally long-lived within the runtime context.
        return None


class MegaSAMDepthProvider(_MegaSAMProviderBase, DepthProvider):
    """Depth provider driven by MegaSAM final bundles."""

    def process(self, frame: Any):
        self._apply_runtime_intrinsics(frame)
        frames: Iterable[Any] = [frame]
        return self._client.estimate_depth(frames)


class MegaSAMTrajectoryProvider(_MegaSAMProviderBase, TrajectoryProvider):
    """Trajectory provider driven by MegaSAM final bundles."""

    def process(self, frame: Any):
        self._apply_runtime_intrinsics(frame)
        frames: Iterable[Any] = [frame]
        return self._client.estimate_trajectory(frames)


class MegaSAMIntrinsicsProvider(_MegaSAMProviderBase, IntrinsicsProvider):
    """Intrinsics provider sourced from MegaSAM bundle or GT override."""

    def process(self, frame: Any):
        self._apply_runtime_intrinsics(frame)
        intrinsics = self._client.fetch_intrinsics()
        return self._scale_intrinsics(intrinsics, frame)


def _inject_runtime_defaults(
    adapter_settings: MutableMapping[str, Any],
    context: MutableMapping[str, Any],
) -> None:
    working_res = context.get("working_resolution")
    if "working_resolution" not in adapter_settings and isinstance(working_res, Sequence) and not isinstance(
        working_res, (str, bytes)
    ):
        if len(working_res) >= 2:
            adapter_settings["working_resolution"] = [int(working_res[0]), int(working_res[1])]

    store = context.get("resource_store")
    if isinstance(store, ResourceStore):
        adapter_settings.setdefault("standard_export_root", str(store.standard_root / "geometry"))

        # Always prefer store intrinsics when available (for example CARLA GT intrinsics).
        adapter_settings.setdefault("gt_intrinsics_path", str(store.path_for(ResourceKind.INTRINSICS)))
        adapter_settings.setdefault("enforce_gt_intrinsics", True)
        adapter_settings.setdefault("require_final_bundle", True)

        ensure_megasam_log_handler(store.standard_root / "logs")


def register_megasam_provider_builders(factory: ProviderFactory) -> None:
    """Register MegaSAM-backed provider builders with provider factory."""

    def _resolve_adapter(binding: ModuleBinding, context: MutableMapping[str, Any]) -> MegaSAMAdapter:
        raw_adapter = binding.settings.get("adapter")
        if not isinstance(raw_adapter, Mapping):
            raise ValueError(
                f"Provider '{binding.tool}' requires an 'adapter' mapping in settings."
            )

        adapter_settings: MutableMapping[str, Any] = dict(raw_adapter)
        _inject_runtime_defaults(adapter_settings, context)

        cache = context.setdefault("megasam_adapters", {})
        key = json.dumps(adapter_settings, sort_keys=True)
        if key not in cache:
            cache[key] = MegaSAMAdapter(adapter_settings)
        return cache[key]

    def build_depth(binding: ModuleBinding, context: MutableMapping[str, Any]) -> MegaSAMDepthProvider:
        return _resolve_adapter(binding, context).create_depth_provider()

    def build_trajectory(binding: ModuleBinding, context: MutableMapping[str, Any]) -> MegaSAMTrajectoryProvider:
        return _resolve_adapter(binding, context).create_trajectory_provider()

    def build_intrinsics(binding: ModuleBinding, context: MutableMapping[str, Any]) -> MegaSAMIntrinsicsProvider:
        return _resolve_adapter(binding, context).create_intrinsics_provider()

    factory.register("MegaSAMDepthProvider", build_depth)
    factory.register("MegaSAMTrajectoryProvider", build_trajectory)
    factory.register("MegaSAMIntrinsicsProvider", build_intrinsics)
