from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from pemoin.data.contracts import (
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceStore,
    RoadPlaneData,
    SemanticSegment,
    SemanticsData,
)
from pemoin.runtime.cache import CrossRunCacheManager, RenderArtifactCacheManager
from pemoin.runtime.profiles.config import ProfileConfig, RuntimeBindings
from pemoin.runtime.runtime import Runtime
from pemoin.visualization.video import write_video_from_paths


def _make_runtime(
    *,
    video_settings: dict[str, object] | None = None,
    blender_scene_settings: dict[str, object] | None = None,
    harmonisation_settings: dict[str, object] | None = None,
) -> Runtime:
    return Runtime(
        ProfileConfig(
            name="test_profile",
            runtime=RuntimeBindings(
                state_window=0,
                degradation_policy="fail_fast",
                settings={
                    "video_export": dict(video_settings or {}),
                    "blender_scene": dict(blender_scene_settings or {}),
                    "harmonisation": dict(harmonisation_settings or {}),
                    "comparison_frame": {"enabled": False},
                },
            ),
            providers={},
            effects={},
            working_resolution=(480, 640),
        )
    )


def test_export_videos_skips_optional_flat_exports_when_features_disabled(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    runtime = _make_runtime()
    resource_store = ResourceStore("run", root=tmp_path)
    generated_calls: list[tuple[Path, Path, str | None]] = []
    discovered_calls: list[tuple[Path, Path, float]] = []

    def fake_generate_visualization_videos(vis_root, output_dir, settings):
        discovered_calls.append((vis_root, output_dir, settings.fps))
        return {}

    def fake_generate_flat_video_from_dir(
        source_dir, output_dir, settings, *, name=None
    ):
        generated_calls.append((source_dir, output_dir, name))
        return output_dir / f"{name}.mp4"

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_visualization_videos",
        fake_generate_visualization_videos,
    )
    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_flat_video_from_dir",
        fake_generate_flat_video_from_dir,
    )

    with caplog.at_level(logging.DEBUG):
        runtime._export_videos(
            resource_store=resource_store,
            provider_context={"frame_provider_info": {"settings": {"sampling_fps": 12.5}}},
        )

    assert discovered_calls == [
        (
            resource_store.standard_root / "visualizations",
            resource_store.standard_root / "videos",
            12.5,
        )
    ]
    assert generated_calls == []
    assert True


def test_export_videos_uses_resolved_sampling_fps_when_sampling_fps_is_not_configured(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _make_runtime()
    resource_store = ResourceStore("run", root=tmp_path)
    discovered_calls: list[tuple[Path, Path, float]] = []

    def fake_generate_visualization_videos(vis_root, output_dir, settings):
        discovered_calls.append((vis_root, output_dir, settings.fps))
        return {}

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_visualization_videos",
        fake_generate_visualization_videos,
    )
    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_flat_video_from_dir",
        lambda *args, **kwargs: None,
    )

    runtime._export_videos(
        resource_store=resource_store,
        provider_context={
            "frame_provider_info": {
                "settings": {
                    "sampling_fps": None,
                    "resolved_sampling_fps": 23.976,
                }
            }
        },
    )

    assert discovered_calls == [
        (
            resource_store.standard_root / "visualizations",
            resource_store.standard_root / "videos",
            23.976,
        )
    ]


def test_export_videos_runs_blender_and_harmonisation_flat_exports_when_expected(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(
        blender_scene_settings={"enabled": True},
        harmonisation_settings={"enabled": True},
    )
    resource_store = ResourceStore("run", root=tmp_path)
    (resource_store.root / "overlayed_frames").mkdir(parents=True)
    (resource_store.root / "overlayed_frames_support_local_grid").mkdir(parents=True)
    (resource_store.root / "harmonized_overlays").mkdir(parents=True)
    generated_calls: list[tuple[Path, Path, str | None]] = []

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_visualization_videos",
        lambda *args, **kwargs: {},
    )

    def fake_generate_flat_video_from_dir(
        source_dir, output_dir, settings, *, name=None
    ):
        generated_calls.append((source_dir, output_dir, name))
        return output_dir / f"{name}.mp4"

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_flat_video_from_dir",
        fake_generate_flat_video_from_dir,
    )

    runtime._export_videos(
        resource_store=resource_store,
        provider_context={"frame_provider_info": {"settings": {"sampling_fps": 10.0}}},
    )

    assert generated_calls == [
        (
            resource_store.root / "overlayed_frames",
            resource_store.standard_root / "videos",
            "overlayed_frames",
        ),
        (
            resource_store.root / "overlayed_frames_support_local_grid",
            resource_store.standard_root / "videos",
            "overlayed_frames_support_local_grid",
        ),
        (
            resource_store.root / "harmonized_overlays",
            resource_store.standard_root / "videos",
            "harmonized_overlays",
        ),
    ]


def test_export_videos_warns_once_for_missing_expected_optional_sources(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    runtime = _make_runtime(
        blender_scene_settings={"enabled": True},
        harmonisation_settings={"enabled": True},
    )
    resource_store = ResourceStore("run", root=tmp_path)
    generated_calls: list[tuple[Path, Path, str | None]] = []

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_visualization_videos",
        lambda *args, **kwargs: {},
    )

    def fake_generate_flat_video_from_dir(
        source_dir, output_dir, settings, *, name=None
    ):
        generated_calls.append((source_dir, output_dir, name))
        return output_dir / f"{name}.mp4"

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_flat_video_from_dir",
        fake_generate_flat_video_from_dir,
    )

    with caplog.at_level(logging.WARNING):
        runtime._export_videos(
            resource_store=resource_store,
            provider_context={"frame_provider_info": {"settings": {"sampling_fps": 10.0}}},
        )

    assert generated_calls == []
    assert True


def test_export_videos_skips_all_generation_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _make_runtime(video_settings={"enabled": False})
    resource_store = ResourceStore("run", root=tmp_path)
    generate_calls: list[str] = []

    def fake_generate_visualization_videos(*args, **kwargs):
        generate_calls.append("discover")
        return {}

    def fake_generate_flat_video_from_dir(*args, **kwargs):
        generate_calls.append("flat")
        return None

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_visualization_videos",
        fake_generate_visualization_videos,
    )
    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_flat_video_from_dir",
        fake_generate_flat_video_from_dir,
    )

    runtime._export_videos(resource_store=resource_store, provider_context={})

    assert generate_calls == []


def test_write_video_from_paths_streams_frames_from_disk(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(3):
        path = frame_dir / f"{idx:06d}.png"
        image = np.full((12, 16, 3), idx * 40, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        paths.append(path)

    output_path = tmp_path / "out.mp4"
    written = write_video_from_paths(paths, output_path, fps=12.0)

    assert written == 3
    assert output_path.exists()


def test_write_video_from_paths_pads_odd_frame_height_without_flipping(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(2):
        path = frame_dir / f"{idx:06d}.png"
        image = np.zeros((281, 500, 3), dtype=np.uint8)
        image[0, :, :] = np.array([255, 0, 0], dtype=np.uint8)
        image[-1, :, :] = np.array([0, 0, 255], dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        paths.append(path)

    output_path = tmp_path / "out_odd.mp4"
    written = write_video_from_paths(paths, output_path, fps=12.0)

    assert written == 2
    cap = cv2.VideoCapture(str(output_path))
    ok, frame = cap.read()
    cap.release()

    assert ok is True
    assert frame is not None
    assert frame.shape[:2] == (282, 500)
    top_row = frame[0].mean(axis=0)
    bottom_row = frame[-1].mean(axis=0)
    assert int(np.argmax(top_row)) == 0
    assert int(np.argmax(bottom_row)) == 2


def test_export_videos_reuses_ground_grid_cache(monkeypatch, tmp_path: Path) -> None:
    runtime = _make_runtime(
        harmonisation_settings={"enabled": True},
        video_settings={"ground_grid": {"num_workers": 2}},
    )
    resource_store = ResourceStore("run", root=tmp_path)
    cache_root = tmp_path / "cache"
    render_cache = RenderArtifactCacheManager(
        CrossRunCacheManager(cache_root),
        stage_settings={"ground_grid": {"enabled": True}},
    )
    harmonized_dir = resource_store.root / "artifacts" / "harmonisation" / "harmonized_overlays"
    harmonized_dir.mkdir(parents=True, exist_ok=True)
    occlusion_dir = resource_store.blender_artifacts_dir("occlusion_masks")
    occlusion_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(2):
        frame_path = harmonized_dir / f"{frame_idx:06d}.png"
        assert cv2.imwrite(str(frame_path), np.full((12, 16, 3), 32 + frame_idx, dtype=np.uint8))
        assert cv2.imwrite(
            str(occlusion_dir / f"{frame_idx:06d}.png"),
            np.zeros((12, 16), dtype=np.uint8),
        )
        resource_store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                offset=0.0,
                metadata={"source": "test"},
            )
        )
        resource_store.save_semantics2d(
            SemanticsData(
                frame_index=frame_idx,
                frame_id=f"{frame_idx:06d}",
                segments=[
                    SemanticSegment(
                        segment_id=0,
                        label="road",
                        score=1.0,
                        mask=np.ones((12, 16), dtype=bool),
                        label_id=0,
                        area=12 * 16,
                        metadata={},
                    )
                ],
                segment_ids=np.zeros((12, 16), dtype=np.int32),
                label_ids=np.zeros((12, 16), dtype=np.int32),
                metadata={"source": "test"},
            )
        )
    resource_store.save_intrinsics(
        IntrinsicsData(
            matrix=np.array(
                [[100.0, 0.0, 8.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            metadata={"source": "test"},
        )
    )
    resource_store.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=frame_idx,
                    camera_to_world=np.eye(4, dtype=np.float32),
                    world_to_camera=np.eye(4, dtype=np.float32),
                    confidence=1.0,
                    metadata={"source": "test"},
                )
                for frame_idx in range(2)
            ],
            metadata={"source": "test"},
        )
    )

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_visualization_videos",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_flat_video_from_dir",
        lambda *args, **kwargs: None,
    )
    calls: list[Path] = []

    def fake_generate_ground_grid(*args, **kwargs):
        output_path = Path(kwargs.get("output_path") or args[2])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"cached-ground-grid")
        calls.append(output_path)
        return output_path

    monkeypatch.setattr(
        "pemoin.runtime.runtime.generate_harmonized_ground_grid_video",
        fake_generate_ground_grid,
    )

    provider_context = {
        "frame_provider_info": {"settings": {"sampling_fps": 10.0}},
        "render_artifact_cache": render_cache,
    }
    runtime._export_videos(
        resource_store=resource_store,
        provider_context=provider_context,
    )
    output_path = resource_store.videos_dir() / "harmonized_overlays_ground_grid.mp4"
    assert output_path.exists()
    output_path.unlink()

    runtime._export_videos(
        resource_store=resource_store,
        provider_context=provider_context,
    )

    assert calls == [resource_store.videos_dir() / "harmonized_overlays_ground_grid.mp4"]
    assert output_path.exists()
