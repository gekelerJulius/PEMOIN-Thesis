import numpy as np
import pytest

from pemoin.data.contracts import (
    PointCloud3DData,
    PoseData,
    ResourceKind,
    ResourceStore,
    RoadPlaneData,
    RoadPlaneSupportData,
    SemanticsData,
)
from pemoin.providers.point_cloud_3d.voxel_grid import VoxelGrid
from pemoin.providers.road_plane_internal.diagnostics import assert_plane_residual_metadata_consistency
from pemoin.providers.road_plane_internal.fit import solve_plane_weighted
from pemoin.providers.road_plane import FrameBundle, PlaneResult, RobustRoadPlaneProvider
from pemoin.data.contracts import PoseSample
from pemoin.visualization.point_cloud_glb import semantic_colors_from_labels
from pemoin.visualization.semantic_palette import semantic_color_for_key, semantic_palette_key


def test_road_plane_support_label_resolution_includes_sidewalk_when_enabled():
    provider = RobustRoadPlaneProvider({"include_sidewalk_in_support": True})
    provider.setup({"semantic_role_defaults": {"road": ["road"], "sidewalk": ["sidewalk"]}})
    ids = provider._support_label_ids_from_semantics(
        semantics=SemanticsData(
            frame_index=0,
            frame_id="0",
            segments=[],
            segment_ids=np.zeros((1, 1), dtype=np.int32),
            metadata={"semantic_roles": {"road": ["road"], "sidewalk": ["sidewalk"]}},
        ),
        label_map={0: "road", 1: "sidewalk", 2: "building"},
    )
    assert ids == [0, 1]


def test_resource_store_point_cloud_3d_roundtrip(tmp_path):
    store = ResourceStore("test_run", root=tmp_path)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    payload = PointCloud3DData(
        points_world=points,
        labels=np.array([1, 2], dtype=np.int32),
        label_confidences=np.array([0.6, 0.9], dtype=np.float32),
        colors=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        label_names={1: "road", 2: "car"},
        observation_counts=np.array([3, 4], dtype=np.int32),
        metadata={"source": "unit-test", "foo": 1},
    )

    path = store.save_point_cloud_3d(payload)
    assert path.exists()

    loaded = store.load_point_cloud_3d()
    np.testing.assert_allclose(loaded.points_world, points)
    np.testing.assert_allclose(loaded.label_confidences, payload.label_confidences)
    assert loaded.metadata["source"] == "unit-test"


def test_point_cloud_semantic_glb_colors_match_shared_semantic_palette():
    labels = np.array([1, 2], dtype=np.int32)
    label_names = {1: "road", 2: "sidewalk"}

    colors = semantic_colors_from_labels(labels, label_names=label_names)

    expected_road = semantic_color_for_key(
        semantic_palette_key(label_id=1, label="road", segment_id=None)
    )
    expected_sidewalk = semantic_color_for_key(
        semantic_palette_key(label_id=2, label="sidewalk", segment_id=None)
    )
    np.testing.assert_array_equal(colors[0, :3], expected_road)
    np.testing.assert_array_equal(colors[1, :3], expected_sidewalk)
    np.testing.assert_array_equal(colors[:, 3], np.array([255, 255], dtype=np.uint8))


def test_voxel_grid_merge_is_weighted_and_deterministic():
    grid = VoxelGrid(
        voxel_size_m=0.5,
        class_ids=[1, 2],
        label_names={1: "road", 2: "car"},
    )
    points = np.array(
        [
            [0.10, 0.10, 0.00],
            [0.20, 0.20, 0.00],
            [1.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    colors = np.array([[20, 20, 20], [30, 30, 30], [200, 10, 10]], dtype=np.uint8)
    labels = np.array([1, 1, 2], dtype=np.int32)
    conf = np.array([0.7, 0.9, 0.8], dtype=np.float32)
    weights = np.array([1.0, 3.0, 2.0], dtype=np.float32)

    grid.integrate_frame(
        points_world=points,
        colors=colors,
        label_ids=labels,
        confidences=conf,
        weights=weights,
    )
    c1 = grid.extract_cloud(
        min_observations=1,
        min_confidence=0.0,
        max_points=100,
        rng=np.random.default_rng(0),
    )
    c2 = grid.extract_cloud(
        min_observations=1,
        min_confidence=0.0,
        max_points=100,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(c1.points_world, c2.points_world)
    assert c1.points_world.shape[0] == 2

    # First voxel weighted centroid of [0.1,0.1,0] (w=1) and [0.2,0.2,0] (w=3).
    expected = np.array([(0.1 * 1.0 + 0.2 * 3.0) / 4.0, (0.1 * 1.0 + 0.2 * 3.0) / 4.0, 0.0])
    near = np.argmin(np.linalg.norm(c1.points_world - expected[None, :], axis=1))
    np.testing.assert_allclose(c1.points_world[near], expected, atol=1e-6)
    np.testing.assert_allclose(c1.observation_counts[near], 2, atol=1e-6)


def test_road_plane_gates_flag_one_sided_windows_and_requires_road_points():
    provider = RobustRoadPlaneProvider({})
    assert ResourceKind.POINT_CLOUD_3D not in provider.required_resources
    assert ResourceKind.DEPTH in provider.required_resources

    c2w = np.eye(4, dtype=np.float32)
    bundle = FrameBundle(
        frame_idx=0,
        pose=PoseSample(frame_index=0, camera_to_world=c2w),
        camera_center=np.zeros((3,), dtype=np.float32),
        camera_height=1.7,
    )

    # One-sided lateral support: all points on +X side.
    rng = np.random.default_rng(42)
    x = rng.uniform(1.0, 5.0, size=(800, 1))
    y = rng.uniform(-2.0, 2.0, size=(800, 1))
    z = rng.uniform(-12.0, -1.0, size=(800, 1))
    points = np.concatenate([x, y, z], axis=1).astype(np.float32)

    ok, meta, quality = provider._evaluate_pre_fit_gates(points=points, center_bundle=bundle)
    # Pre-fit gate is now soft for non-catastrophic geometry issues.
    assert ok
    assert meta["gate_reason"] == "ok"
    assert "left_right_balance" in tuple(meta.get("soft_gate_reasons", ()))
    assert 0.0 <= quality < 1.0

    bad_result = PlaneResult(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        offset=0.0,
        residuals=np.zeros((3,), dtype=np.float32),
        cov_diag=np.ones((4,), dtype=np.float32),
        quality={"residual_p90": 0.8, "inlier_ratio": 0.1},
    )
    ok_post, reason, post_quality = provider._evaluate_post_fit_gates(bad_result)
    # Post-fit gate is soft unless a catastrophic threshold is exceeded.
    assert ok_post
    assert reason in {"soft_max_residual_p90", "soft_min_inlier_ratio"}
    assert 0.0 <= post_quality < 1.0


def test_saved_point_gate_uses_startup_grace_then_enforces_strict_thresholds():
    provider = RobustRoadPlaneProvider(
        {
            "saved_point_max_residual_p90_m": 0.3,
            "saved_point_min_inlier_ratio": 0.5,
            "saved_point_startup_grace_frames": 3,
            "saved_point_startup_max_residual_p90_m": 0.6,
            "saved_point_startup_min_inlier_ratio": 0.2,
        }
    )
    ok, startup = provider._evaluate_saved_point_gate(
        residual_p90=0.5,
        inlier_ratio=0.3,
        frame_order_index=1,
    )
    assert ok
    assert startup

    ok_late, startup_late = provider._evaluate_saved_point_gate(
        residual_p90=0.5,
        inlier_ratio=0.3,
        frame_order_index=4,
    )
    assert not ok_late
    assert not startup_late


def test_saved_point_gate_validation_rejects_invalid_startup_settings():
    provider = RobustRoadPlaneProvider({"saved_point_startup_grace_frames": -1})
    with pytest.raises(ValueError):
        provider._validate_sampling_settings()


def test_layering_metrics_detect_duplicate_pixel_depth_layers():
    uv = np.array(
        [
            [10.1, 20.1],
            [10.2, 20.2],
            [30.0, 40.0],
            [30.0, 40.0],
            [50.0, 60.0],
        ],
        dtype=np.float32,
    )
    depth = np.array([5.0, 7.0, 3.0, 3.2, 4.0], dtype=np.float32)
    layering_ratio, spread_p90 = RobustRoadPlaneProvider._layering_metrics(
        uv=uv,
        depth=depth,
        image_width=100,
    )
    assert layering_ratio > 0.0
    assert spread_p90 > 0.0


def test_road_plane_support_validation_rejects_invalid_layering_threshold():
    provider = RobustRoadPlaneProvider({"support_max_layering_ratio": 1.5})
    with pytest.raises(ValueError):
        provider._validate_sampling_settings()


def test_support_quality_gate_combines_min_points_layering_and_spread():
    provider = RobustRoadPlaneProvider(
        {
            "support_min_points": 100,
            "support_max_layering_ratio": 0.4,
            "support_max_depth_spread_p90_m": 1.5,
        }
    )
    assert provider._evaluate_support_quality_gate(
        support_points=120,
        layering_ratio=0.3,
        depth_spread_p90_m=1.0,
    )
    assert not provider._evaluate_support_quality_gate(
        support_points=80,
        layering_ratio=0.3,
        depth_spread_p90_m=1.0,
    )
    assert not provider._evaluate_support_quality_gate(
        support_points=120,
        layering_ratio=0.5,
        depth_spread_p90_m=1.0,
    )
    assert not provider._evaluate_support_quality_gate(
        support_points=120,
        layering_ratio=0.3,
        depth_spread_p90_m=2.0,
    )


def test_gate_reason_priority_prefers_support_over_saved_point():
    provider = RobustRoadPlaneProvider({})
    current = provider._resolve_gate_reason(current="saved_point_quality", candidate="support_quality")
    assert current == "support_quality"
    current2 = provider._resolve_gate_reason(current="support_quality", candidate="saved_point_quality")
    assert current2 == "support_quality"


def test_road_plane_settings_expose_degraded_output_toggle():
    provider = RobustRoadPlaneProvider({})
    assert provider.settings.allow_degraded_output is True
    provider_strict = RobustRoadPlaneProvider({"allow_degraded_output": False})
    assert provider_strict.settings.allow_degraded_output is False


def test_collect_window_points_uses_causal_window_and_exclusions():
    provider = RobustRoadPlaneProvider(
        {
            "window_causal_only": True,
            "window_exclude_low_support_frames": True,
            "window_min_support_points_for_inclusion": 10,
            "window_min_frames_required": 1,
        }
    )
    frame_indices = [1, 2, 3, 4]
    sampled_points = {
        1: np.ones((5, 3), dtype=np.float32),
        2: np.ones((6, 3), dtype=np.float32) * 2.0,
        3: np.ones((7, 3), dtype=np.float32) * 3.0,
        4: np.ones((8, 3), dtype=np.float32) * 4.0,
    }
    sampled_weights = {k: np.ones((v.shape[0],), dtype=np.float32) for k, v in sampled_points.items()}
    bundles = {
        k: FrameBundle(
            frame_idx=k,
            pose=PoseSample(frame_index=k, camera_to_world=np.eye(4, dtype=np.float32)),
            camera_center=np.array([float(k), 0.0, 0.0], dtype=np.float32),
            camera_height=1.7,
        )
        for k in frame_indices
    }
    points, _weights, _centers, _h, _a, meta = provider._collect_window_points_from_cache(
        frame_indices=frame_indices,
        center_idx=3,  # frame 4
        half_width=2,
        sampled_points=sampled_points,
        sampled_weights=sampled_weights,
        bundles=bundles,
        excluded_frames=frozenset({3}),
        support_point_counts={1: 100, 2: 5, 3: 100, 4: 100},
    )
    # causal window for center frame 4 with half_width=2 => candidate frames [2,3,4]
    # frame 3 excluded as catastrophic, frame 2 excluded for low support, frame 4 remains.
    assert points.shape[0] == 8
    assert meta["window_included_frame_count"] == 1
    assert meta["window_excluded_catastrophic_count"] == 1
    assert meta["window_excluded_low_support_count"] == 1


def test_sampling_validation_rejects_invalid_recovery_fit_settings():
    provider = RobustRoadPlaneProvider({"recovery_fit_accept_min_inlier_ratio": 1.2})
    with pytest.raises(ValueError):
        provider._validate_sampling_settings()


def test_unanchored_plane_solve_handles_scale_mismatch_better_than_hard_anchor():
    rng = np.random.default_rng(0)
    # Plane around z=-4 with noise; camera at z=0 but reported height=1.6 (mismatched scale).
    x = rng.uniform(-3.0, 3.0, size=3000).astype(np.float32)
    y = rng.uniform(3.0, 7.0, size=3000).astype(np.float32)
    z = (-4.0 + 0.08 * rng.standard_normal(size=3000)).astype(np.float32)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    w = np.ones((pts.shape[0],), dtype=np.float32)
    cam = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    n_a, d_a, _ = solve_plane_weighted(
        points=pts,
        weights=w,
        anchor_camera_center=cam,
        camera_height_m=1.6,
        lambda_up=200.0,
        lambda_temp=0.0,
        prev_plane=None,
        up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        enforce_height_anchor=True,
    )
    r_a = np.abs(pts @ n_a + d_a)

    n_u, d_u, _ = solve_plane_weighted(
        points=pts,
        weights=w,
        anchor_camera_center=cam,
        camera_height_m=1.6,
        lambda_up=200.0,
        lambda_temp=0.0,
        prev_plane=None,
        up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        enforce_height_anchor=False,
    )
    r_u = np.abs(pts @ n_u + d_u)
    assert float(np.percentile(r_u, 90)) < float(np.percentile(r_a, 90))


def test_road_plane_residual_metadata_consistency_guard_raises_on_mismatch():
    residuals = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    with pytest.raises(RuntimeError):
        assert_plane_residual_metadata_consistency(
            residuals=residuals,
            metadata_residual_median=0.5,
            metadata_residual_p90=0.6,
            tolerance_m=0.01,
        )


def test_road_plane_refresh_debug_visualizations_uses_sampled_points(tmp_path):
    store = ResourceStore("road_plane_refresh", root=tmp_path)
    frames = [1, 2]
    samples = []
    for frame_idx in frames:
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = float(frame_idx)
        samples.append(PoseSample(frame_index=frame_idx, camera_to_world=c2w))
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                offset=0.0,
                metadata={"source": "unit-test"},
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata={"source": "unit-test"}))

    per_frame_points = {}
    for frame_idx in frames:
        pts = np.array(
            [[float(frame_idx), 0.0, 0.0], [float(frame_idx), 1.0, 0.0]],
            dtype=np.float32,
        )
        per_frame_points[frame_idx] = pts
        store.save_road_plane_support(
            RoadPlaneSupportData(
                frame_index=frame_idx,
                points_world=pts,
                weights=np.ones((pts.shape[0],), dtype=np.float32),
                diagnostics={"source": "unit-test"},
                metadata={"source": "unit-test"},
            )
        )

    provider = RobustRoadPlaneProvider({})
    calls: list[tuple[int, np.ndarray]] = []
    generated: dict[str, object] = {}

    provider._reset_visualization_artifacts = lambda _resources: None

    def _fake_write_debug(resources, frame_idx, points, normal, offset):
        calls.append((int(frame_idx), np.asarray(points, dtype=np.float32).copy()))
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return image, image

    provider._write_debug = _fake_write_debug

    def _fake_generate_videos(resources, residuals_frames, overlay_frames, *, fps):
        generated["fps"] = float(fps)
        generated["residual_count"] = len(residuals_frames)
        generated["overlay_count"] = len(overlay_frames)

    provider._generate_videos = _fake_generate_videos

    provider.refresh_debug_visualizations(
        store,
        {"frame_provider_info": {"settings": {"sampling_fps": 12}}},
    )

    assert [frame_idx for frame_idx, _ in calls] == frames
    for frame_idx, points in calls:
        np.testing.assert_allclose(points, per_frame_points[frame_idx])
    assert generated["fps"] == 12.0
    assert generated["residual_count"] == len(frames)
    assert generated["overlay_count"] == len(frames)
