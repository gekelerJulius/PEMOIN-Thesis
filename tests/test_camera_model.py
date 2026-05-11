import numpy as np

from pemoin.geometry.camera_model import (
    backproject_uv_depth_to_camera,
    camera_to_world,
    project_world_to_image,
    world_to_camera,
)


def test_backproject_sign_conventions():
    k = np.array([[1000.0, 0.0, 100.0], [0.0, 1000.0, 100.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    uv = np.array([[100.0, 120.0]], dtype=np.float32)
    d = np.array([10.0], dtype=np.float32)

    p_bl = backproject_uv_depth_to_camera(uv, d, k, camera_convention="blender")[0]
    p_cv = backproject_uv_depth_to_camera(uv, d, k, camera_convention="opencv")[0]

    assert p_bl[0] == p_cv[0]
    assert np.isclose(p_bl[1], -p_cv[1])
    assert np.isclose(p_bl[2], -p_cv[2])


def test_project_backproject_roundtrip_blender_identity():
    k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    pts_cam = np.array(
        [
            [0.2, -0.1, -4.0],
            [-0.5, 0.3, -7.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    pts_world = camera_to_world(pts_cam, c2w)
    uv, valid = project_world_to_image(
        pts_world,
        k,
        camera_to_world_matrix=c2w,
        camera_convention="blender",
    )
    assert bool(np.all(valid))
    depth = -pts_cam[:, 2]
    pts_cam_recovered = backproject_uv_depth_to_camera(uv, depth, k, camera_convention="blender")
    np.testing.assert_allclose(pts_cam_recovered, pts_cam, atol=1e-4)
    np.testing.assert_allclose(world_to_camera(pts_world, camera_to_world_matrix=c2w), pts_cam, atol=1e-5)

