"""Typed runtime context objects with mapping compatibility for providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Optional

from pemoin.data.contracts import ResourceKind, ResourceStore


@dataclass(frozen=True, slots=True)
class FrameProviderInfo:
    """Resolved frame-provider metadata for a run."""

    tool: str
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FrameProviderInfo | None":
        if value is None:
            return None
        tool = value.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError("frame_provider_info.tool must be a non-empty string.")
        settings = value.get("settings", {})
        if not isinstance(settings, Mapping):
            raise ValueError("frame_provider_info.settings must be a mapping.")
        return cls(tool=tool, settings={str(k): v for k, v in settings.items()})

    def to_mapping(self) -> dict[str, Any]:
        return {"tool": self.tool, "settings": dict(self.settings)}


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Resolved filesystem locations for a run."""

    run_dir: Path
    profiles_config_path: Optional[Path] = None
    run_key: Optional[str] = None


class RuntimeContext(MutableMapping[str, Any]):
    """
    Shared runtime context with typed accessors and mapping compatibility.

    Existing providers can continue using mapping-style access while runtime and
    CLI code gain explicit structured fields.
    """

    def __init__(
        self,
        initial: Mapping[str, Any] | None = None,
        *,
        run_paths: RunPaths | None = None,
        frame_source: Path | None = None,
        frame_provider_info: FrameProviderInfo | None = None,
        profile_name: str | None = None,
        run_timestamp: str | None = None,
    ) -> None:
        self._data: dict[str, Any] = dict(initial or {})
        if run_paths is not None:
            self.run_paths = run_paths
        if frame_source is not None:
            self.frame_source = frame_source
        if frame_provider_info is not None:
            self.frame_provider_info = frame_provider_info
        if profile_name is not None:
            self.profile_name = profile_name
        if run_timestamp is not None:
            self.run_timestamp = run_timestamp

    @classmethod
    def coerce(cls, value: MutableMapping[str, Any] | None) -> "RuntimeContext":
        if isinstance(value, cls):
            return value
        return cls(value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def run_paths(self) -> RunPaths | None:
        run_dir = self._data.get("run_dir")
        if run_dir is None:
            return None
        config = self._data.get("profiles_config_path")
        return RunPaths(
            run_dir=Path(run_dir),
            profiles_config_path=Path(config) if config is not None else None,
            run_key=str(self._data.get("run_key")) if self._data.get("run_key") is not None else None,
        )

    @run_paths.setter
    def run_paths(self, value: RunPaths) -> None:
        self._data["run_dir"] = value.run_dir
        if value.profiles_config_path is not None:
            self._data["profiles_config_path"] = value.profiles_config_path
        if value.run_key is not None:
            self._data["run_key"] = value.run_key

    @property
    def frame_source(self) -> Path | None:
        raw = self._data.get("frame_source")
        return Path(raw) if raw is not None else None

    @frame_source.setter
    def frame_source(self, value: Path) -> None:
        self._data["frame_source"] = value

    @property
    def frame_provider_info(self) -> FrameProviderInfo | None:
        raw = self._data.get("frame_provider_info")
        if isinstance(raw, FrameProviderInfo):
            return raw
        if isinstance(raw, Mapping):
            info = FrameProviderInfo.from_mapping(raw)
            if info is not None:
                self._data["frame_provider_info"] = info.to_mapping()
            return info
        return None

    @frame_provider_info.setter
    def frame_provider_info(self, value: FrameProviderInfo) -> None:
        self._data["frame_provider_info"] = value.to_mapping()

    @property
    def profile_name(self) -> str | None:
        raw = self._data.get("profile_name")
        return str(raw) if raw is not None else None

    @profile_name.setter
    def profile_name(self, value: str) -> None:
        self._data["profile_name"] = value

    @property
    def run_timestamp(self) -> str | None:
        raw = self._data.get("run_timestamp")
        return str(raw) if raw is not None else None

    @run_timestamp.setter
    def run_timestamp(self, value: str) -> None:
        self._data["run_timestamp"] = value

    @property
    def resource_store(self) -> ResourceStore | None:
        raw = self._data.get("resource_store")
        return raw if isinstance(raw, ResourceStore) else None

    @resource_store.setter
    def resource_store(self, value: ResourceStore) -> None:
        self._data["resource_store"] = value
        self._data.setdefault("frames_dir", value.base_dir(ResourceKind.FRAMES))
