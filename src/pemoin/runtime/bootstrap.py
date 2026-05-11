"""Runtime launch assembly helpers for CLI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pemoin.data.contracts import ResourceStore
from pemoin.providers import ProviderFactory, create_default_provider_factory
from pemoin.runtime.context import FrameProviderInfo, RunPaths, RuntimeContext
from pemoin.runtime.profiles.config import ProfileConfig
from pemoin.runtime.runtime import Runtime


@dataclass(frozen=True, slots=True)
class RuntimeLaunch:
    """Fully assembled runtime launch inputs."""

    runtime: Runtime
    provider_factory: ProviderFactory
    provider_context: RuntimeContext


def save_profile_snapshot(
    *,
    run_dir: Path,
    snapshot: Mapping[str, Any],
) -> ResourceStore:
    store = ResourceStore(run_dir.name, root=run_dir.parent)
    store.save_profile_snapshot(snapshot)
    return store


def create_runtime_launch(
    *,
    profile: ProfileConfig,
    run_dir: Path,
    frame_source: Path,
    frame_provider_info: Mapping[str, Any],
    run_timestamp: str,
    profiles_config_path: Path,
) -> RuntimeLaunch:
    return RuntimeLaunch(
        runtime=Runtime(profile=profile),
        provider_factory=create_default_provider_factory(),
        provider_context=RuntimeContext(
            run_paths=RunPaths(
                run_dir=run_dir,
                profiles_config_path=profiles_config_path.expanduser().resolve(),
                run_key=run_dir.name,
            ),
            frame_source=frame_source,
            frame_provider_info=FrameProviderInfo.from_mapping(frame_provider_info),
            run_timestamp=run_timestamp,
            profile_name=profile.name,
        ),
    )
