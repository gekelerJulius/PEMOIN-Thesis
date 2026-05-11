"""Tests for the external GTSAM factor-graph bridge."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pemoin.data.contracts import PoseData, PoseSample
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.stages import factor_graph
from pemoin.providers.geometry_fusion.stages.road_rectification import FrameRectificationResult


def _make_pose(frame_idx: int, tx: float = 0.0, tz: float = 1.6) -> PoseSample:
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = tx
    c2w[2, 3] = tz
    return PoseSample(frame_index=frame_idx, camera_to_world=c2w)


def _make_rect(frame_idx: int) -> FrameRectificationResult:
    return FrameRectificationResult(
        frame_index=frame_idx,
        normal_cam=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        offset_cam=1.6,
        implied_height_m=1.6,
        scale=1.0,
        bias=0.0,
        inlier_ratio=0.9,
        residual_p90_m=0.05,
        support_count=500,
    )


class TestFactorGraph:
    def test_disabled_returns_input(self):
        settings = GeometryFusionSettings(factor_graph_enabled=False)
        poses = PoseData(samples=[_make_pose(0)])
        rect = [_make_rect(0)]
        result_poses, result_rect, _ = factor_graph.run_factor_graph_fusion(
            poses, rect, [], 1.6, np.eye(3, dtype=np.float32), None, [0], settings
        )
        assert result_poses is poses
        assert result_rect is rect

    def test_too_few_frames_returns_input(self):
        settings = GeometryFusionSettings(factor_graph_enabled=True)
        poses = PoseData(samples=[_make_pose(0), _make_pose(1, tx=1.0)])
        rect = [_make_rect(0), _make_rect(1)]
        result_poses, result_rect, _ = factor_graph.run_factor_graph_fusion(
            poses, rect, [], 1.6, np.eye(3, dtype=np.float32), None, [0, 1], settings
        )
        assert result_poses is poses
        assert result_rect is rect

    def test_successful_bridge_updates_poses_and_metadata(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            factor_graph,
            "_resolve_env_launcher",
            lambda env_name, env_manager: ("micromamba", "run", "-n", env_name),
        )

        def _fake_run(cmd, check, capture_output, text, timeout, cwd, env):
            del check, capture_output, text, timeout, cwd, env
            output_path = None
            for idx, token in enumerate(cmd):
                if token == "--output":
                    output_path = cmd[idx + 1]
                    break
            assert output_path is not None
            np.savez_compressed(
                output_path,
                optimized_pose_frame_indices=np.asarray([0, 1], dtype=np.int64),
                optimized_poses_c2w=np.asarray(
                    [
                        np.array(
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 1.6],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        ),
                        np.array(
                            [
                                [1.0, 0.0, 0.0, 1.5],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 1.6],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        ),
                    ],
                    dtype=np.float32,
                ),
                optimized_rect_frame_indices=np.asarray([0, 1], dtype=np.int64),
                optimized_rect_scales=np.asarray([1.0, 1.2], dtype=np.float32),
                optimized_rect_biases=np.asarray([0.0, 0.1], dtype=np.float32),
            )
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(factor_graph.subprocess, "run", _fake_run)

        poses = PoseData(samples=[_make_pose(0), _make_pose(1, tx=1.0)])
        rect = [_make_rect(0), _make_rect(1)]
        settings = GeometryFusionSettings(
            factor_graph_enabled=True,
            fg_window_size=3,
            fg_max_step_jump_m=2.0,
        )

        result_poses, result_rect, _ = factor_graph.run_factor_graph_fusion(
            poses, rect, [], 1.6, np.eye(3, dtype=np.float32), None, [0, 1, 2], settings
        )

        assert result_poses.metadata["factor_graph_optimized"] is True
        np.testing.assert_allclose(result_poses.samples[1].camera_to_world[0, 3], 1.5)
        np.testing.assert_allclose(result_rect[1].scale, 1.2)
        np.testing.assert_allclose(result_rect[1].bias, 0.1)

    def test_bridge_input_is_preconditioned_to_temp_z_up(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            factor_graph,
            "_resolve_env_launcher",
            lambda env_name, env_manager: ("conda", "run", "-n", env_name),
        )

        def _fake_run(cmd, check, capture_output, text, timeout, cwd, env):
            del check, capture_output, text, timeout, cwd, env
            input_path = cmd[cmd.index("--input") + 1]
            output_path = cmd[cmd.index("--output") + 1]
            with np.load(input_path, allow_pickle=False) as data:
                poses_c2w = np.asarray(data["poses_c2w"], dtype=np.float32)
            up_avg = np.mean(poses_c2w[:, :3, 1], axis=0)
            up_avg /= np.linalg.norm(up_avg)
            assert up_avg[2] > 0.99
            np.savez_compressed(
                output_path,
                optimized_pose_frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
                optimized_poses_c2w=poses_c2w,
                optimized_rect_frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
                optimized_rect_scales=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                optimized_rect_biases=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            )
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(factor_graph.subprocess, "run", _fake_run)

        samples = []
        for idx in range(3):
            c2w = np.eye(4, dtype=np.float32)
            c2w[0, 3] = float(idx)
            c2w[1, 3] = 1.6
            c2w[2, 3] = 0.1 * float(idx)
            samples.append(PoseSample(frame_index=idx, camera_to_world=c2w, world_to_camera=np.linalg.inv(c2w)))
        poses = PoseData(samples=samples, metadata={"metric_scale": True})
        rect = [_make_rect(0), _make_rect(1), _make_rect(2)]

        result_poses, _, _ = factor_graph.run_factor_graph_fusion(
            poses,
            rect,
            [],
            1.6,
            np.eye(3, dtype=np.float32),
            None,
            [0, 1, 2],
            GeometryFusionSettings(factor_graph_enabled=True, fg_window_size=3, fg_max_step_jump_m=5.0),
        )
        np.testing.assert_allclose(result_poses.samples[0].camera_to_world, poses.samples[0].camera_to_world)

    def test_discontinuous_output_falls_back_to_input(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            factor_graph,
            "_resolve_env_launcher",
            lambda env_name, env_manager: ("conda", "run", "-n", env_name),
        )

        def _fake_run(cmd, check, capture_output, text, timeout, cwd, env):
            del check, capture_output, text, timeout, cwd, env
            output_path = cmd[cmd.index("--output") + 1]
            np.savez_compressed(
                output_path,
                optimized_pose_frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
                optimized_poses_c2w=np.asarray(
                    [
                        np.eye(4, dtype=np.float32),
                        np.array(
                            [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.6], [0.0, 0.0, 0.0, 1.0]],
                            dtype=np.float32,
                        ),
                        np.array(
                            [[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.6], [0.0, 0.0, 0.0, 1.0]],
                            dtype=np.float32,
                        ),
                    ],
                    dtype=np.float32,
                ),
                optimized_rect_frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
                optimized_rect_scales=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                optimized_rect_biases=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            )
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(factor_graph.subprocess, "run", _fake_run)

        poses = PoseData(samples=[_make_pose(0, tx=0.0), _make_pose(1, tx=0.1), _make_pose(2, tx=0.2)], metadata={"metric_scale": True})
        rect = [_make_rect(0), _make_rect(1), _make_rect(2)]

        result_poses, result_rect, _ = factor_graph.run_factor_graph_fusion(
            poses,
            rect,
            [],
            1.6,
            np.eye(3, dtype=np.float32),
            None,
            [0, 1, 2],
            GeometryFusionSettings(factor_graph_enabled=True, fg_window_size=3),
        )

        assert result_rect is rect
        assert result_poses.metadata["factor_graph_rejected"] is True
        assert result_poses.metadata["factor_graph_optimized"] is False
        np.testing.assert_allclose(result_poses.samples[2].camera_to_world, poses.samples[2].camera_to_world)

    def test_bridge_failure_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            factor_graph,
            "_resolve_env_launcher",
            lambda env_name, env_manager: ("conda", "run", "-n", env_name),
        )
        monkeypatch.setattr(
            factor_graph.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="missing gtsam"),
        )

        poses = PoseData(samples=[_make_pose(0), _make_pose(1), _make_pose(2)])
        rect = [_make_rect(0), _make_rect(1), _make_rect(2)]

        with pytest.raises(RuntimeError, match="missing gtsam"):
            factor_graph.run_factor_graph_fusion(
                poses,
                rect,
                [],
                1.6,
                np.eye(3, dtype=np.float32),
                None,
                [0, 1, 2],
                GeometryFusionSettings(factor_graph_enabled=True, fg_window_size=3),
            )

    def test_malformed_output_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            factor_graph,
            "_resolve_env_launcher",
            lambda env_name, env_manager: ("conda", "run", "-n", env_name),
        )

        def _fake_run(cmd, **kwargs):
            del kwargs
            output_path = cmd[cmd.index("--output") + 1]
            np.savez_compressed(output_path, optimized_pose_frame_indices=np.asarray([0], dtype=np.int64))
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(factor_graph.subprocess, "run", _fake_run)

        poses = PoseData(samples=[_make_pose(0), _make_pose(1), _make_pose(2)])
        rect = [_make_rect(0), _make_rect(1), _make_rect(2)]

        with pytest.raises(RuntimeError, match="missing fields"):
            factor_graph.run_factor_graph_fusion(
                poses,
                rect,
                [],
                1.6,
                np.eye(3, dtype=np.float32),
                None,
                [0, 1, 2],
                GeometryFusionSettings(factor_graph_enabled=True, fg_window_size=3),
            )

    def test_signal_termination_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            factor_graph,
            "_resolve_env_launcher",
            lambda env_name, env_manager: ("conda", "run", "-n", env_name),
        )
        monkeypatch.setattr(
            factor_graph.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=-11, stderr=""),
        )

        poses = PoseData(samples=[_make_pose(0), _make_pose(1), _make_pose(2)])
        rect = [_make_rect(0), _make_rect(1), _make_rect(2)]

        with pytest.raises(RuntimeError, match="terminated by signal 11"):
            factor_graph.run_factor_graph_fusion(
                poses,
                rect,
                [],
                1.6,
                np.eye(3, dtype=np.float32),
                None,
                [0, 1, 2],
                GeometryFusionSettings(factor_graph_enabled=True, fg_window_size=3),
            )
