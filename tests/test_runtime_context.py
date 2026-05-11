from __future__ import annotations

from pathlib import Path

from pemoin.data.contracts import ResourceKind, ResourceStore
from pemoin.runtime.context import FrameProviderInfo, RunPaths, RuntimeContext


def test_runtime_context_exposes_typed_fields_and_mapping_compatibility(
    tmp_path: Path,
) -> None:
    context = RuntimeContext(
        run_paths=RunPaths(
            run_dir=tmp_path / "run",
            profiles_config_path=tmp_path / "profiles.json",
            run_key="run-key",
        ),
        frame_source=tmp_path / "frames",
        frame_provider_info=FrameProviderInfo(
            tool="DirectoryFrameProvider",
            settings={"sampling_fps": 12.5},
        ),
        profile_name="carla_gt",
        run_timestamp="20260312_120000",
    )

    assert context.run_paths is not None
    assert context.run_paths.run_dir == tmp_path / "run"
    assert context.frame_source == tmp_path / "frames"
    assert context.frame_provider_info is not None
    assert context.frame_provider_info.tool == "DirectoryFrameProvider"
    assert context["profile_name"] == "carla_gt"

    store = ResourceStore("run", root=tmp_path)
    context.resource_store = store

    assert context["resource_store"] is store
    assert context["frames_dir"] == store.base_dir(ResourceKind.FRAMES)
