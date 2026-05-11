import numpy as np
import pytest

from pemoin.geometry.plane import Plane


def test_plane_height_anchor_and_signed_distance():
    camera = np.array([0.0, 0.0, 1.6], dtype=np.float32)
    plane = Plane.from_height_anchor(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        camera_center=camera,
        plane_height_at_camera_m=1.6,
    )
    assert np.isclose(plane.height_at_camera(camera), 1.6)
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=np.float32)
    signed = plane.signed_distance(pts)
    assert signed.shape == (2,)
    assert signed[0] < signed[1]


def test_plane_enforce_normal_orientation():
    camera = np.array([0.0, 0.0, 1.6], dtype=np.float32)
    flipped = Plane(normal=np.array([0.0, 0.0, -1.0], dtype=np.float32), offset=0.0)
    corrected = flipped.enforce_normal_orientation(camera_center=camera, target_height_m=1.6)
    assert corrected.height_at_camera(camera) >= 0.0


def test_plane_rejects_degenerate_normal():
    with pytest.raises(ValueError):
        Plane(normal=np.array([0.0, 0.0, 0.0], dtype=np.float32), offset=0.0)

