from __future__ import annotations

import numpy as np
import pytest

from pemoin.data.contracts import FrameData
from pemoin.providers.adapters.nuscenes_adapter import (
    NuScenesCameraHeightProvider,
    NuScenesTrajectoryProvider,
)
from pemoin.providers.camera_height import CameraHeightProvider


def test_camera_height_provider_uses_constant_height() -> None:
    provider = CameraHeightProvider({"height": 1.6})
    provider.setup({})

    first = provider.process(FrameData(frame_id="000001", index=1))
    second = provider.process(FrameData(frame_id="000002", index=2))

    assert first.height_m == pytest.approx(1.6)
    assert second.height_m == pytest.approx(1.6)
    assert first.metadata["height_source"] == "constant"
    assert first.metadata["axis"] == "z"
    assert first.metadata["world_coordinate_system"] == "blender"


def test_camera_height_provider_uses_per_frame_array() -> None:
    provider = CameraHeightProvider({"heights": [1.5, 1.7]})
    provider.setup({})

    first = provider.process(FrameData(frame_id="000010", index=10))
    second = provider.process(FrameData(frame_id="000011", index=11))

    assert first.height_m == pytest.approx(1.5)
    assert second.height_m == pytest.approx(1.7)
    assert first.metadata["height_source"] == "per_frame_array"
    assert second.metadata["sequence_index"] == 1


def test_camera_height_provider_requires_single_height_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        CameraHeightProvider({"height": 1.6, "heights": [1.6]}).setup({})

    with pytest.raises(ValueError, match="requires either 'height' or 'heights'"):
        CameraHeightProvider({}).setup({})


def test_camera_height_provider_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        CameraHeightProvider({"height": 0.0}).setup({})

    with pytest.raises(ValueError, match="must be an array of numbers"):
        CameraHeightProvider({"heights": "1.6"}).setup({})

    provider = CameraHeightProvider({"heights": [1.6]})
    provider.setup({})
    provider.process(FrameData(frame_id="000001", index=1))
    with pytest.raises(ValueError, match="exhausted configured 'heights' array"):
        provider.process(FrameData(frame_id="000002", index=2))


def test_nuscenes_camera_height_provider_emits_standardized_metadata() -> None:
    provider = NuScenesCameraHeightProvider({})

    class _DummyNusc:
        @staticmethod
        def get(name: str, token: str) -> dict:
            assert name == "calibrated_sensor"
            assert token == "calib-token"
            return {"translation": [0.0, 0.0, 1.511]}

    provider._nusc = _DummyNusc()
    provider._constant_camera_translation = np.array([0.0, 0.0, 1.511], dtype=np.float32)
    provider._sample_data_for_frame = lambda frame: {"calibrated_sensor_token": "calib-token"}  # type: ignore[method-assign]

    data = provider.process(FrameData(frame_id="000001", index=1))

    assert data.height_m == pytest.approx(1.511)
    assert data.metadata["axis"] == "z"
    assert data.metadata["world_coordinate_system"] == "blender"
    assert data.metadata["height_reference"] == "ego_vehicle_center"
    assert data.metadata["height_source"] == "nuscenes_calibrated_sensor_constant"


def test_nuscenes_trajectory_provider_marks_gt_output_as_metric() -> None:
    provider = NuScenesTrajectoryProvider({})

    class _DummyNusc:
        @staticmethod
        def get(name: str, token: str) -> dict:
            if name == "calibrated_sensor":
                assert token == "calib-token"
                return {
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 1.5],
                }
            if name == "ego_pose":
                assert token == "ego-token"
                return {
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [10.0, 0.0, 0.0],
                }
            raise AssertionError(f"unexpected table {name}")

    provider._nusc = _DummyNusc()
    provider._constant_intrinsics = np.array(
        [[1000.0, 0.0, 500.0], [0.0, 1000.0, 250.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    provider._sample_data_for_frame = lambda frame: {  # type: ignore[method-assign]
        "calibrated_sensor_token": "calib-token",
        "ego_pose_token": "ego-token",
    }

    frame = FrameData(
        frame_id="000001",
        index=1,
        metadata={},
    )
    pose = provider.process(frame)

    assert pose.metadata["metric_scale"] is True
    assert pose.samples[0].metadata["metric_scale"] is True
    np.testing.assert_allclose(
        pose.samples[0].camera_to_world[:3, 3],
        np.array([10.0, 0.0, 1.5], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pose.samples[0].camera_to_world[:3, :3],
        np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        atol=1e-6,
    )


def test_nuscenes_provider_uses_frame_cam_sd_token_before_index_lookup() -> None:
    provider = NuScenesCameraHeightProvider({})

    class _DummyNusc:
        @staticmethod
        def get(name: str, token: str) -> dict:
            if name == "sample_data":
                assert token == "cam-token-39"
                return {"calibrated_sensor_token": "calib-token"}
            if name == "calibrated_sensor":
                assert token == "calib-token"
                return {"translation": [0.0, 0.0, 1.23]}
            raise AssertionError(f"unexpected table {name}")

    provider._nusc = _DummyNusc()
    provider._cam_sd_tokens = ["wrong-scene-token"] * 3

    data = provider.process(
        FrameData(
            frame_id="000039",
            index=39,
            metadata={"cam_sd_token": "cam-token-39"},
        )
    )

    assert data.height_m == pytest.approx(1.23)


def test_nuscenes_provider_validates_constant_calibration_in_setup() -> None:
    provider = NuScenesCameraHeightProvider({"sampling_mode": "all_camera_frames"})

    class _DummyNusc:
        scene = [
            {
                "name": "scene-0001",
                "description": "demo",
                "first_sample_token": "sample-0",
            }
        ]

        @staticmethod
        def get(name: str, token: str) -> dict:
            if name == "sample":
                if token == "sample-0":
                    return {"data": {"CAM_FRONT": "sd-0"}, "next": ""}
            if name == "sample_data":
                if token == "sd-0":
                    return {
                        "token": "sd-0",
                        "calibrated_sensor_token": "calib-0",
                        "timestamp": 0,
                        "next": "sd-1",
                    }
                if token == "sd-1":
                    return {
                        "token": "sd-1",
                        "calibrated_sensor_token": "calib-1",
                        "timestamp": 100_000,
                        "next": "",
                    }
            if name == "calibrated_sensor":
                if token == "calib-0":
                    return {
                        "translation": [0.0, 0.0, 1.5],
                        "camera_intrinsic": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
                    }
                if token == "calib-1":
                    return {
                        "translation": [0.0, 0.0, 1.6],
                        "camera_intrinsic": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
                    }
            raise AssertionError(f"unexpected table {name} {token}")

    provider._nusc = _DummyNusc()
    provider._resolved_settings = {"scene_index": 0, "camera": "CAM_FRONT", "sampling_mode": "all_camera_frames"}
    provider._camera = "CAM_FRONT"

    with pytest.raises(RuntimeError, match="translation varies"):
        provider._resolve_scene()
        provider._sample_tokens = ["sample-0"]
        provider._cam_sd_tokens = ["sd-0"]
        provider._cam_sd_to_index = {"sd-0": 0}
        provider._validate_stream_consistency()
