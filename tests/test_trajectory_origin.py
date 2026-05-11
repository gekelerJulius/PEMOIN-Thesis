import numpy as np

from pemoin.coordinate_systems.trajectory_origin import anchor_pose_data_to_origin
from pemoin.data.contracts import PoseData, PoseSample


def _pose(frame_index: int, translation: tuple[float, float, float]) -> PoseSample:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.asarray(translation, dtype=np.float32)
    return PoseSample(
        frame_index=frame_index,
        camera_to_world=c2w,
        world_to_camera=np.linalg.inv(c2w),
    )


def test_anchor_pose_data_to_origin_applies_rigid_translation_only() -> None:
    pose_data = PoseData(
        samples=[
            _pose(0, (4.0, -2.0, 0.5)),
            _pose(1, (6.0, 1.0, 1.2)),
        ],
        metadata={"source": "unit-test"},
    )

    anchored, delta = anchor_pose_data_to_origin(
        pose_data,
        anchor_height_m=1.7,
        metadata_label="unit_test",
    )

    np.testing.assert_allclose(delta, np.array([-4.0, 2.0, 1.2], dtype=np.float32))
    np.testing.assert_allclose(
        anchored.samples[0].camera_to_world[:3, 3],
        np.array([0.0, 0.0, 1.7], dtype=np.float32),
    )
    np.testing.assert_allclose(
        anchored.samples[1].camera_to_world[:3, 3] - anchored.samples[0].camera_to_world[:3, 3],
        np.array([2.0, 3.0, 0.7], dtype=np.float32),
    )
    np.testing.assert_allclose(
        anchored.samples[1].camera_to_world[:3, :3],
        pose_data.samples[1].camera_to_world[:3, :3],
    )
    np.testing.assert_allclose(
        anchored.samples[1].world_to_camera,
        np.linalg.inv(anchored.samples[1].camera_to_world),
    )
    assert anchored.metadata["origin_anchor_enabled"] is True
    assert anchored.metadata["origin_anchor_frame_index"] == 0
