from __future__ import annotations

from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import pytest

from pemoin.data.contracts import (
    CameraHeightData,
    DepthData,
    FrameData,
    IntrinsicsData,
    PointCloud3DData,
    ResourceKind,
    ResourceStore,
    PoseData,
    PoseSample,
    RoadPlaneData,
    SemanticsData,
)
from pemoin.runtime.cache import RenderArtifactCacheManager
from pemoin.runtime.profiles.config import ProfileConfig, RuntimeBindings
from pemoin.runtime.runtime import Runtime
from pemoin.utils.logging import LoggedSubprocessResult
from pemoin.visualization.video import copy_canonical_output_video


def test_render_trajectory_scene_uses_run_profile_snapshot(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    snapshot_path = run_dir / "standard" / "profile.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text('{"profile":"nuscenes_gt"}', encoding="utf-8")

    captured: list[list[str]] = []

    def fake_run_logged_subprocess(
        cmd: list[str],
        *,
        stdout_log_path: Path,
        stderr_log_path: Path,
        stream_output: bool,
        show_progress: bool,
    ) -> LoggedSubprocessResult:
        captured.append(list(cmd))
        ResourceStore.blender_artifact_dir_for(
            run_dir,
            "pedestrian_frames",
            create=True,
        )
        ResourceStore.blender_artifact_dir_for(
            run_dir,
            "shadow_frames",
            create=True,
        )
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text("", encoding="utf-8")
        stderr_log_path.write_text("", encoding="utf-8")
        assert stream_output is False
        assert show_progress is True
        return LoggedSubprocessResult(
            args=tuple(cmd),
            returncode=0,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            stdout_tail=(),
            stderr_tail=(),
            streamed_output=stream_output,
        )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")
    monkeypatch.setattr(Runtime, "_validate_blender_scene_inputs", staticmethod(lambda path: None))
    monkeypatch.setattr(
        "pemoin.visualization.blender_runner.validate_blender_scene_inputs",
        lambda path: None,
    )
    monkeypatch.setattr(
        "pemoin.visualization.blender_runner.run_logged_subprocess",
        fake_run_logged_subprocess,
    )

    Runtime._render_trajectory_scene(
        run_dir=run_dir,
        config_path=tmp_path / "profiles.json",
        profile_name="nuscenes_gt",
    )

    assert captured
    cmd = captured[0]
    config_index = cmd.index("--config")
    assert Path(cmd[config_index + 1]) == snapshot_path
    host_python_index = cmd.index("--host-python")
    assert Path(cmd[host_python_index + 1]) == Path(sys.executable).resolve()


def test_render_trajectory_scene_passes_stream_output_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    snapshot_path = run_dir / "standard" / "profile.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text('{"profile":"nuscenes_gt"}', encoding="utf-8")
    streamed: list[bool] = []

    def fake_run_logged_subprocess(
        cmd: list[str],
        *,
        stdout_log_path: Path,
        stderr_log_path: Path,
        stream_output: bool,
        show_progress: bool,
    ) -> LoggedSubprocessResult:
        streamed.append(stream_output)
        assert show_progress is True
        ResourceStore.blender_artifact_dir_for(run_dir, "pedestrian_frames", create=True)
        ResourceStore.blender_artifact_dir_for(run_dir, "shadow_frames", create=True)
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log_path.write_text("", encoding="utf-8")
        stderr_log_path.write_text("", encoding="utf-8")
        return LoggedSubprocessResult(
            args=tuple(cmd),
            returncode=0,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            stdout_tail=(),
            stderr_tail=(),
            streamed_output=stream_output,
        )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")
    monkeypatch.setattr(Runtime, "_validate_blender_scene_inputs", staticmethod(lambda path: None))
    monkeypatch.setattr(
        "pemoin.visualization.blender_runner.validate_blender_scene_inputs",
        lambda path: None,
    )
    monkeypatch.setattr(
        "pemoin.visualization.blender_runner.run_logged_subprocess",
        fake_run_logged_subprocess,
    )

    Runtime._render_trajectory_scene(
        run_dir=run_dir,
        config_path=tmp_path / "profiles.json",
        profile_name="nuscenes_gt",
        stream_output=True,
        show_progress=True,
    )

    assert streamed == [True]


def test_render_trajectory_scene_requires_run_profile_snapshot(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")
    monkeypatch.setattr(Runtime, "_validate_blender_scene_inputs", staticmethod(lambda path: None))

    with pytest.raises(FileNotFoundError, match="saved run profile snapshot"):
        Runtime._render_trajectory_scene(
            run_dir=run_dir,
            config_path=tmp_path / "profiles.json",
            profile_name="nuscenes_gt",
        )


def test_validate_blender_scene_inputs_rejects_unvalidated_lighting(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    intrinsics_dir = run_dir / "standard" / "intrinsics"
    frames_dir = run_dir / "standard" / "frames"
    lighting_dir = run_dir / "standard" / "lighting"
    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    lighting_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        intrinsics_dir / "intrinsics.npz",
        matrix=np.array(
            [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        metadata={"image_shape": (32, 32)},
    )

    imageio.imwrite(frames_dir / "000000.png", np.zeros((32, 32, 3), dtype=np.uint8))
    (lighting_dir / "lighting.json").write_text(
        '{"mode": "full_sun", "validation": {"passed": false}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not validated"):
        Runtime._validate_blender_scene_inputs(run_dir)


def test_blender_bundle_payload_is_stable_for_equivalent_rewritten_npz_inputs(
    tmp_path: Path,
) -> None:
    profile = ProfileConfig(
        name="cache_render_profile",
        runtime=RuntimeBindings(
            state_window=1,
            degradation_policy="OfflineDegradationPolicy",
            settings={
                "comparison_frame": {"enabled": False},
                "cross_run_cache": {"enabled": True, "root": str(tmp_path / "cache")},
            },
        ),
        providers={},
        effects={},
        working_resolution=(8, 8),
    )
    runtime = Runtime(profile)
    render_cache = RenderArtifactCacheManager.from_runtime_settings(
        profile.runtime.settings,
        base_root=tmp_path,
    )

    signatures: list[str] = []
    for run_name in ("run_a", "run_b"):
        store = ResourceStore(run_name, root=tmp_path)
        frame_index = 0
        store.save_frame(
            FrameData(
                frame_id="000000",
                index=frame_index,
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        )
        store.save_intrinsics(
            IntrinsicsData(
                matrix=np.array(
                    [[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                metadata={"source": "test"},
            )
        )
        store.save_depth(
            DepthData(
                frame_index=frame_index,
                depth=np.ones((8, 8), dtype=np.float32),
                metadata={"source": "test"},
            )
        )
        store.save_semantics2d(
            SemanticsData(
                frame_index=frame_index,
                segments=[],
                segment_ids=np.zeros((8, 8), dtype=np.int32),
                metadata={"source": "test"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_index,
                height_m=1.6,
                metadata={"source": "test"},
            )
        )
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_index,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                offset=-1.6,
                metadata={"source": "test"},
            )
        )
        c2w = np.eye(4, dtype=np.float32)
        store.save_trajectory(
            PoseData(
                samples=[
                    PoseSample(
                        frame_index=frame_index,
                        camera_to_world=c2w,
                        world_to_camera=np.linalg.inv(c2w),
                        metadata={"source": "test"},
                    )
                ],
                metadata={"source": "test"},
            )
        )
        lighting_dir = store.base_dir(ResourceKind.LIGHTING)
        lighting_dir.mkdir(parents=True, exist_ok=True)
        (lighting_dir / "lighting.json").write_text(
            '{"schema_version":2,"rig_mode":"envmap_only","light_rig":[],"mode":"ambient_only","quality":{},"sun_diagnostics":{},"validation":{"passed":true},"recovery":{},"decomposition":{}}',
            encoding="utf-8",
        )
        (lighting_dir / "envmap.exr").write_bytes(b"exr")
        profile_snapshot = store.root / "standard" / "profile.json"
        profile_snapshot.parent.mkdir(parents=True, exist_ok=True)
        profile_snapshot.write_text('{"profile":"cache_render_profile"}', encoding="utf-8")

        payload = runtime._blender_bundle_payload(
            run_dir=store.root,
            profile_name=profile.name,
            blender_settings={"enabled": True},
            render_cache=render_cache,
            bundle_id="blender_scene_export",
        )
        signatures.append(render_cache.signature("blender_scene_export", payload))

    assert signatures[0] == signatures[1]


def test_blender_bundle_artifacts_split_render_shadow_and_composition(
    tmp_path: Path,
) -> None:
    profile = ProfileConfig(
        name="cache_render_profile",
        runtime=RuntimeBindings(
            state_window=1,
            degradation_policy="OfflineDegradationPolicy",
            settings={
                "comparison_frame": {"enabled": False},
                "cross_run_cache": {"enabled": True, "root": str(tmp_path / "cache")},
            },
        ),
        providers={},
        effects={},
        working_resolution=(8, 8),
    )
    runtime = Runtime(profile)
    render_cache = RenderArtifactCacheManager.from_runtime_settings(
        profile.runtime.settings,
        base_root=tmp_path,
    )
    run_dir = tmp_path / "run"
    (run_dir / "scene.blend").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "scene.blend").write_bytes(b"blend")
    for relpath in (
        "artifacts/blender/fbx_exports/character_root_motion.fbx",
        "artifacts/blender/fbx_exports/character_root_motion.export.json",
        "artifacts/blender/pedestrian_frames/frame_0001.png",
        "artifacts/blender/pedestrian_depth_frames/000001.npz",
        "artifacts/blender/_pedestrian_depth_exr/depth_0001.exr",
        "artifacts/blender/shadow_frames/shadow_0001.png",
        "artifacts/blender/overlayed_frames/000001.png",
        "artifacts/blender/overlayed_frames_support_local_grid/000001.png",
        "artifacts/blender/occlusion_masks/000001.png",
        "artifacts/blender/occlusion_debug/000001.png",
    ):
        path = run_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (run_dir / "character_root_motion.fbx").write_bytes(b"fbx")

    fbx_artifacts = runtime._blender_bundle_artifacts(
        run_dir,
        "blender_fbx_export",
        render_cache,
    )
    render_artifacts = runtime._blender_bundle_artifacts(
        run_dir,
        "blender_pedestrian_render_outputs",
        render_cache,
    )
    shadow_artifacts = runtime._blender_bundle_artifacts(
        run_dir,
        "blender_shadow_outputs",
        render_cache,
    )
    composition_artifacts = runtime._blender_bundle_artifacts(
        run_dir,
        "blender_composition_outputs",
        render_cache,
    )

    assert "artifacts/blender/fbx_exports/character_root_motion.fbx" in fbx_artifacts
    assert (
        "artifacts/blender/fbx_exports/character_root_motion.export.json"
        in fbx_artifacts
    )
    assert "character_root_motion.fbx" in fbx_artifacts
    assert "artifacts/blender/pedestrian_frames/frame_0001.png" in render_artifacts
    assert "artifacts/blender/pedestrian_depth_frames/000001.npz" in render_artifacts
    assert "artifacts/blender/_pedestrian_depth_exr/depth_0001.exr" in render_artifacts
    assert "artifacts/blender/shadow_frames/shadow_0001.png" not in render_artifacts
    assert shadow_artifacts == {
        "artifacts/blender/shadow_frames/shadow_0001.png": (
            run_dir / "artifacts/blender/shadow_frames/shadow_0001.png"
        )
    }
    assert "artifacts/blender/overlayed_frames/000001.png" in composition_artifacts
    assert (
        "artifacts/blender/overlayed_frames_support_local_grid/000001.png"
        in composition_artifacts
    )
    assert "artifacts/blender/occlusion_masks/000001.png" in composition_artifacts
    assert "artifacts/blender/occlusion_debug/000001.png" in composition_artifacts


def test_blender_bundle_payload_signature_changes_with_mixamo_motion_category(
    tmp_path: Path,
) -> None:
    idle_dir = tmp_path / "assets" / "mixamo" / "animations" / "idle"
    moving_dir = tmp_path / "assets" / "mixamo" / "animations" / "moving"
    idle_dir.mkdir(parents=True, exist_ok=True)
    moving_dir.mkdir(parents=True, exist_ok=True)
    character_path = tmp_path / "assets" / "mixamo" / "character.fbx"
    character_path.parent.mkdir(parents=True, exist_ok=True)
    character_path.write_text("character", encoding="utf-8")
    idle_animation_path = idle_dir / "wave.fbx"
    moving_animation_path = moving_dir / "wave.fbx"
    idle_animation_path.write_text("same-bytes", encoding="utf-8")
    moving_animation_path.write_text("same-bytes", encoding="utf-8")

    render_cache = RenderArtifactCacheManager.from_runtime_settings(
        {"cross_run_cache": {"enabled": True, "root": str(tmp_path / "cache")}},
        base_root=tmp_path,
    )

    def _signature(animation_path: Path) -> str:
        profile = ProfileConfig(
            name="cache_render_profile",
            runtime=RuntimeBindings(
                state_window=1,
                degradation_policy="OfflineDegradationPolicy",
                settings={"comparison_frame": {"enabled": False}},
            ),
            providers={},
            effects={},
            working_resolution=(8, 8),
            mixamo={
                "character_fbx_path": character_path.as_posix(),
                "animation_fbx_path": animation_path.as_posix(),
            },
        )
        runtime = Runtime(profile)
        run_dir = tmp_path / f"run_{animation_path.parent.name}"
        store = ResourceStore(run_dir.name, root=run_dir.parent)
        for frame_index in range(1):
            store.save_frame(
                FrameData(
                    frame_id=f"{frame_index:06d}",
                    index=frame_index,
                    image=np.zeros((8, 8, 3), dtype=np.uint8),
                )
            )
            store.save_intrinsics(
                IntrinsicsData(
                    matrix=np.eye(3, dtype=np.float32),
                    metadata={"width": 8, "height": 8},
                )
            )
            store.save_depth(
                DepthData(
                    frame_index=frame_index,
                    depth=np.ones((8, 8), dtype=np.float32),
                    metadata={"source": "test"},
                )
            )
            store.save_semantics2d(
                SemanticsData(
                    frame_index=frame_index,
                    segments=[],
                    segment_ids=np.zeros((8, 8), dtype=np.int32),
                    metadata={"source": "test"},
                )
            )
            store.save_camera_height(
                CameraHeightData(
                    frame_index=frame_index,
                    height_m=1.6,
                    metadata={"source": "test"},
                )
            )
            store.save_road_plane(
                RoadPlaneData(
                    frame_index=frame_index,
                    normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    offset=-1.6,
                    metadata={"source": "test"},
                )
            )
            c2w = np.eye(4, dtype=np.float32)
            store.save_trajectory(
                PoseData(
                    samples=[
                        PoseSample(
                            frame_index=frame_index,
                            camera_to_world=c2w,
                            world_to_camera=np.linalg.inv(c2w),
                            metadata={"source": "test"},
                        )
                    ],
                    metadata={"source": "test"},
                )
            )
        lighting_dir = store.base_dir(ResourceKind.LIGHTING)
        lighting_dir.mkdir(parents=True, exist_ok=True)
        (lighting_dir / "lighting.json").write_text(
            '{"schema_version":2,"rig_mode":"envmap_only","light_rig":[],"mode":"ambient_only","quality":{},"sun_diagnostics":{},"validation":{"passed":true},"recovery":{},"decomposition":{}}',
            encoding="utf-8",
        )
        (lighting_dir / "envmap.exr").write_bytes(b"exr")
        (store.root / "standard" / "profile.json").write_text(
            '{"profile":"cache_render_profile"}',
            encoding="utf-8",
        )
        payload = runtime._blender_bundle_payload(
            run_dir=store.root,
            profile_name=profile.name,
            blender_settings={"enabled": True},
            render_cache=render_cache,
            bundle_id="blender_scene_export",
        )
        return render_cache.signature("blender_scene_export", payload)

    assert _signature(idle_animation_path) != _signature(moving_animation_path)


def test_blender_fbx_bundle_payload_ignores_render_only_standard_resources(
    tmp_path: Path,
) -> None:
    assets_dir = tmp_path / "assets" / "mixamo"
    animation_dir = assets_dir / "animations" / "moving"
    animation_dir.mkdir(parents=True, exist_ok=True)
    character_path = assets_dir / "character.fbx"
    animation_path = animation_dir / "walk.fbx"
    character_path.write_text("character", encoding="utf-8")
    animation_path.write_text("walk", encoding="utf-8")

    render_cache = RenderArtifactCacheManager.from_runtime_settings(
        {"cross_run_cache": {"enabled": True, "root": str(tmp_path / "cache")}},
        base_root=tmp_path,
    )

    def _signature(road_plane_offset: float, lighting_body: str) -> str:
        profile = ProfileConfig(
            name="cache_render_profile",
            runtime=RuntimeBindings(
                state_window=1,
                degradation_policy="OfflineDegradationPolicy",
                settings={"comparison_frame": {"enabled": False}},
            ),
            providers={},
            effects={},
            working_resolution=(8, 8),
            mixamo={
                "character_fbx_path": character_path.as_posix(),
                "animation_fbx_path": animation_path.as_posix(),
            },
        )
        runtime = Runtime(profile)
        store = ResourceStore(f"run_{road_plane_offset}".replace(".", "_"), root=tmp_path)
        store.save_frame(
            FrameData(
                frame_id="000000",
                index=0,
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        )
        store.save_depth(
            DepthData(
                frame_index=0,
                depth=np.full((8, 8), float(abs(road_plane_offset) + 1.0), dtype=np.float32),
                metadata={"source": "test"},
            )
        )
        store.save_semantics2d(
            SemanticsData(
                frame_index=0,
                segments=[],
                segment_ids=np.full((8, 8), int(abs(road_plane_offset) * 10), dtype=np.int32),
                metadata={"source": "test"},
            )
        )
        store.save_road_plane(
            RoadPlaneData(
                frame_index=0,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                offset=road_plane_offset,
                metadata={"source": "test"},
            )
        )
        profile_snapshot = store.root / "standard" / "profile.json"
        profile_snapshot.parent.mkdir(parents=True, exist_ok=True)
        profile_snapshot.write_text('{"profile":"cache_render_profile"}', encoding="utf-8")
        lighting_dir = store.base_dir(ResourceKind.LIGHTING)
        lighting_dir.mkdir(parents=True, exist_ok=True)
        (lighting_dir / "lighting.json").write_text(lighting_body, encoding="utf-8")
        payload = runtime._blender_bundle_payload(
            run_dir=store.root,
            profile_name=profile.name,
            blender_settings={"enabled": True},
            render_cache=render_cache,
            bundle_id="blender_fbx_export",
        )
        return render_cache.signature("blender_fbx_export", payload)

    assert _signature(
        -1.6,
        '{"schema_version":2,"validation":{"passed":true},"variant":"a"}',
    ) == _signature(
        -9.9,
        '{"schema_version":2,"validation":{"passed":true},"variant":"b"}',
    )


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


class _IntrinsicsProvider:
    produced_resources = frozenset()
    required_resources = frozenset()

    def setup(self, context) -> None:
        _ = context

    def process(self, frame) -> IntrinsicsData:
        _ = frame
        return IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "test"},
        )

    def teardown(self) -> None:
        pass


class _DepthProvider:
    produced_resources = frozenset()
    required_resources = frozenset()

    def setup(self, context) -> None:
        _ = context

    def process(self, frame) -> DepthData:
        return DepthData(
            frame_index=int(frame.index),
            depth=np.ones((8, 8), dtype=np.float32),
            metadata={"source": "test"},
        )

    def teardown(self) -> None:
        pass


class _TrajectoryProvider:
    produced_resources = frozenset()
    required_resources = frozenset()

    def setup(self, context) -> None:
        _ = context

    def process(self, frame) -> PoseData:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([float(frame.index), 1.5, 0.0], dtype=np.float32)
        return PoseData(
            samples=[
                PoseSample(
                    frame_index=int(frame.index),
                    camera_to_world=c2w,
                    world_to_camera=np.linalg.inv(c2w),
                    metadata={"source": "test"},
                )
            ],
            metadata={"source": "test"},
        )

    def teardown(self) -> None:
        pass


class _CameraHeightProvider:
    produced_resources = frozenset()
    required_resources = frozenset()

    def setup(self, context) -> None:
        _ = context

    def process(self, frame) -> CameraHeightData:
        return CameraHeightData(
            frame_index=int(frame.index),
            height_m=1.5,
            metadata={"source": "test"},
        )

    def teardown(self) -> None:
        pass


class _PointCloudProvider:
    produced_resources = frozenset()
    required_resources = frozenset()

    def setup(self, context) -> None:
        _ = context

    def run(self, resources: ResourceStore, context) -> None:
        _ = context
        resources.save_point_cloud_3d(
            PointCloud3DData(
                points_world=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
                labels=np.array([1, 2], dtype=np.int32),
                label_confidences=np.array([0.9, 0.8], dtype=np.float32),
                colors=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
                label_names={1: "road", 2: "sidewalk"},
                observation_counts=np.array([2, 3], dtype=np.int32),
                metadata={"source": "test"},
            )
        )
        resources.rgb_pointcloud_artifact_path().write_bytes(b"rgb-glb")
        resources.semantic_pointcloud_artifact_path().write_bytes(b"semantic-glb")
        resources.rgb_pointcloud_path().write_bytes(b"rgb-glb")
        resources.semantic_pointcloud_path().write_bytes(b"semantic-glb")

    def teardown(self) -> None:
        pass


class _RoadPlaneProvider:
    produced_resources = frozenset()
    required_resources = frozenset()

    def setup(self, context) -> None:
        _ = context

    def run(self, resources, context=None) -> None:
        _ = context
        resources.save_road_plane(
            RoadPlaneData(
                frame_index=0,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                offset=-1.5,
                metadata={"source": "test"},
            )
        )

    def teardown(self) -> None:
        pass


def test_runtime_reuses_blender_and_harmonisation_caches_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    profile = ProfileConfig(
        name="cache_render_profile",
        runtime=RuntimeBindings(
            state_window=1,
            degradation_policy="OfflineDegradationPolicy",
                settings={
                    "comparison_frame": {"enabled": False},
                    "geometry_validation": {"enabled": False},
                    "video_export": {"enabled": False},
                    "blender_scene": {"enabled": True},
                "harmonisation": {
                    "enabled": True,
                    "pretrained_path": str(tmp_path / "harmonizer.pth"),
                },
                "cross_run_cache": {
                    "enabled": True,
                    "root": str(cache_root),
                    "blender_scene": {"enabled": True},
                    "harmonisation": {"enabled": True},
                },
            },
        ),
        providers={},
        effects={},
        working_resolution=(8, 8),
    )
    Path(profile.runtime.settings["harmonisation"]["pretrained_path"]).write_bytes(b"weights")
    providers = {
        "intrinsics": _IntrinsicsProvider(),
        "depth": _DepthProvider(),
        "trajectory": _TrajectoryProvider(),
        "camera_height": _CameraHeightProvider(),
        "point_cloud_3d": _PointCloudProvider(),
        "road_plane": _RoadPlaneProvider(),
    }
    blender_calls: list[Path] = []
    harmonisation_calls: list[Path] = []

    def _fake_render(
        run_dir: Path,
        config_path: Path,
        profile_name: str,
        stream_output: bool = False,
        show_progress: bool = True,
    ) -> None:
        _ = (config_path, profile_name, stream_output, show_progress)
        blender_calls.append(run_dir)
        (run_dir / "scene.blend").write_bytes(b"blend")
        for dirname in (
            "pedestrian_frames",
            "pedestrian_depth_frames",
            "shadow_frames",
            "_pedestrian_depth_exr",
            "overlayed_frames",
            "overlayed_frames_support_local_grid",
            "occlusion_masks",
            "occlusion_debug",
        ):
            path = ResourceStore.blender_artifact_dir_for(
                run_dir,
                dirname,
                create=True,
            )
            path.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(2):
                (path / f"{frame_idx:06d}.png").write_bytes(b"frame")

    def _fake_harmonisation(run_dir: Path, settings) -> Path:
        harmonisation_calls.append(run_dir)
        out_dir = run_dir / settings.output_dir
        diag_dir = run_dir / f"{settings.output_dir}_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        diag_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(2):
            (out_dir / f"{frame_idx:06d}.png").write_bytes(b"harmonized")
            (diag_dir / f"{frame_idx:06d}.json").write_text("{}", encoding="utf-8")
        return out_dir

    monkeypatch.setattr(
        Runtime,
        "_render_trajectory_scene",
        staticmethod(_fake_render),
    )
    monkeypatch.setattr("pemoin.runtime.runtime.run_harmonisation", _fake_harmonisation)

    profiles_config_path = tmp_path / "profiles.json"
    profiles_config_path.write_text("{}", encoding="utf-8")

    for run_name in ("run_a", "run_b"):
        run_dir = tmp_path / run_name
        snapshot_path = run_dir / "standard" / "profile.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text('{"profile":"cache_render_profile"}', encoding="utf-8")
        runtime = Runtime(profile)
        monkeypatch.setattr(runtime, "build_providers", lambda factory, context: providers)
        runtime.run(
            _DummyFrameProvider(2),
            context={
                "run_dir": str(run_dir),
                "profiles_config_path": str(profiles_config_path),
                "profile_name": profile.name,
            },
        )

    assert len(blender_calls) == 1
    assert len(harmonisation_calls) == 1
    second_run_root = tmp_path / "run_b"
    assert (second_run_root / "scene.blend").exists()
    assert (
        second_run_root / "artifacts" / "blender" / "overlayed_frames" / "000000.png"
    ).exists()
    assert (
        second_run_root
        / "artifacts"
        / "harmonisation"
        / "harmonized_overlays"
        / "000000.png"
    ).exists()


def test_copy_canonical_output_video_writes_run_root_convenience_mp4(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "standard" / "videos" / "harmonized_overlays.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"video")

    destination_path = tmp_path / "output.mp4"
    copied_path = copy_canonical_output_video(source_path, destination_path)

    assert copied_path == destination_path
    assert destination_path.read_bytes() == b"video"


def test_require_point_cloud_debug_outputs_rejects_missing_glbs(tmp_path: Path) -> None:
    store = ResourceStore("run", root=tmp_path)
    store.save_point_cloud_3d(
        PointCloud3DData(
            points_world=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            label_confidences=np.array([1.0], dtype=np.float32),
            colors=np.array([[255, 255, 255]], dtype=np.uint8),
            label_names={1: "road"},
            observation_counts=np.array([2], dtype=np.int32),
            metadata={"source": "unit-test"},
        )
    )

    with pytest.raises(RuntimeError, match="Point-cloud debug outputs are required"):
        Runtime._require_point_cloud_debug_outputs(store)
