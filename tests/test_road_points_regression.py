import math
from pathlib import Path

import numpy as np

from pemoin.data.contracts import (
    DepthData,
    FrameData,
    IntrinsicsData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    SemanticSegment,
    SemanticsData,
)
from pemoin.providers.point_cloud_3d.provider import DensePointCloud3DProvider
from pemoin.runtime.cache import CrossRunCacheManager


def _rotation_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c = math.cos(r)
    s = math.sin(r)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def test_point_cloud_lift_stays_below_camera_for_car_like_pose():
    h, w = 80, 120
    depth = np.full((h, w), 8.0, dtype=np.float32)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    intr = IntrinsicsData(
        matrix=np.array([[200.0, 0.0, w / 2.0], [0.0, 200.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        metadata={"camera_convention": "blender"},
    )
    label_ids = np.zeros((h, w), dtype=np.int32)
    label_ids[h // 2 :, :] = 1
    semantics = SemanticsData(
        frame_index=0,
        segments=[
            SemanticSegment(segment_id=1, label="road", score=1.0, label_id=1, mask=label_ids == 1),
        ],
        label_ids=label_ids,
        segment_ids=label_ids.copy(),
        metadata={},
    )

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _rotation_x(20.0)
    c2w[2, 3] = 1.6
    pose = PoseSample(
        frame_index=0,
        camera_to_world=c2w,
        metadata={"camera_convention": "blender"},
    )
    provider = DensePointCloud3DProvider(
        {
            "pixel_stride": 1,
            "min_depth_m": 0.1,
            "max_depth_m": 30.0,
            "min_confidence": 0.0,
            "min_total_points": 1,
        }
    )
    points, _colors, _labels, _conf, _depth, _view_dirs, diag = provider._lift_frame(
        frame_index=0,
        depth=depth,
        semantics=semantics,
        intrinsics=intr,
        pose=pose,
        image=image,
        class_index={1: 0},
    )
    assert points.shape[0] > 100
    assert float(np.median(points[:, 2])) < float(c2w[2, 3])
    assert float(diag["valid_ratio"]) > 0.2


def test_point_cloud_provider_subsamples_frames_for_heavy_runs():
    provider = DensePointCloud3DProvider(
        {
            "frame_subsample_target": 3,
        }
    )

    selected = provider._select_frame_indices([0, 1, 2, 3, 4, 5, 6, 7])

    assert selected == [0, 3, 6, 7]


def test_point_cloud_provider_adapts_pixel_stride_to_sample_budget():
    provider = DensePointCloud3DProvider(
        {
            "pixel_stride": 2,
            "max_sampled_pixels_per_frame": 10_000,
        }
    )

    stride = provider._effective_pixel_stride(480, 640)

    assert stride > 2


def test_point_cloud_provider_collapses_duplicate_replacement_sources():
    provider = DensePointCloud3DProvider({})

    selected_sources = provider._selected_source_indices(
        [0, 1, 2, 3],
        {1: 0, 2: 0, 3: 3},
    )

    assert selected_sources == [0, 3]


def test_point_cloud_provider_reuses_cross_run_cache(tmp_path: Path) -> None:
    run_a = ResourceStore("run_a", root=tmp_path)
    cache_root = tmp_path / "cache"
    provider_a = DensePointCloud3DProvider(
        {
            "pixel_stride": 2,
            "min_depth_m": 0.1,
            "max_depth_m": 30.0,
            "min_confidence": 0.0,
            "min_total_points": 1,
            "min_observations": 1,
            "export_glb": False,
        }
    )
    for frame_idx in range(2):
        run_a.save_frame(
            FrameData(
                frame_id=f"{frame_idx:06d}",
                index=frame_idx,
                image=np.full((16, 16, 3), 64 + frame_idx, dtype=np.uint8),
            )
        )
        run_a.save_depth(
            DepthData(
                frame_index=frame_idx,
                depth=np.full((16, 16), 8.0, dtype=np.float32),
                metadata={"source": "test"},
            )
        )
        label_ids = np.zeros((16, 16), dtype=np.int32)
        label_ids[8:, :] = 1
        run_a.save_semantics2d(
            SemanticsData(
                frame_index=frame_idx,
                frame_id=f"{frame_idx:06d}",
                segments=[
                    SemanticSegment(
                        segment_id=0,
                        label="sidewalk",
                        score=1.0,
                        label_id=0,
                        mask=label_ids == 0,
                    ),
                    SemanticSegment(
                        segment_id=1,
                        label="road",
                        score=1.0,
                        label_id=1,
                        mask=label_ids == 1,
                    ),
                ],
                label_ids=label_ids,
                segment_ids=label_ids.copy(),
                metadata={},
            )
        )
    run_a.save_intrinsics(
        IntrinsicsData(
            matrix=np.array([[200.0, 0.0, 8.0], [0.0, 200.0, 8.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            metadata={"camera_convention": "blender"},
        )
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _rotation_x(20.0)
    c2w[2, 3] = 1.6
    run_a.save_trajectory(
        PoseData(
            samples=[
                PoseSample(
                    frame_index=frame_idx,
                    camera_to_world=c2w,
                    world_to_camera=np.linalg.inv(c2w),
                    confidence=1.0,
                    metadata={},
                )
                for frame_idx in range(2)
            ],
            metadata={"source": "test", "camera_convention": "blender"},
        )
    )

    provider_a.setup(
        {
            "cross_run_cache": CrossRunCacheManager(cache_root),
            "profile_name": "test_profile",
        }
    )
    provider_a.run(run_a, {})
    spec = provider_a.get_cross_run_cache_spec(run_a)
    assert spec is not None and spec["ready"] is True
    CrossRunCacheManager(cache_root).publish(
        "point_cloud_3d",
        provider_a._cache_signature or "",
        payload=provider_a._cache_payload or {},
        artifacts=spec["artifacts"],
        source_summary=spec["source_summary"],
    )

    point_cloud_path = run_a.path_for(ResourceKind.POINT_CLOUD_3D)
    assert point_cloud_path.exists()
    point_cloud_path.unlink()

    provider_b = DensePointCloud3DProvider(
        {
            "pixel_stride": 2,
            "min_depth_m": 0.1,
            "max_depth_m": 30.0,
            "min_confidence": 0.0,
            "min_total_points": 1,
            "min_observations": 1,
            "export_glb": False,
        }
    )
    provider_b.setup(
        {
            "cross_run_cache": CrossRunCacheManager(cache_root),
            "profile_name": "test_profile",
        }
    )
    provider_b.run(run_a, {})

    status = provider_b.get_cross_run_cache_status()
    assert status["cross_run_cache_hit"] is True
    assert run_a.path_for(ResourceKind.POINT_CLOUD_3D).exists()
