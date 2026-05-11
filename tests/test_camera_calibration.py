from __future__ import annotations

import numpy as np
import pytest

from pemoin.data.contracts import IntrinsicsData, ResourceKind, ResourceStore
from pemoin.utils.camera_calibration import (
    BlenderCameraParityError,
    IntrinsicsValidationError,
    solve_blender_camera_for_intrinsics,
    validate_and_normalize_intrinsics,
)
from pemoin.utils.resolution import scale_intrinsics


def test_scale_intrinsics_prefers_explicit_width_height_over_principal_point_fallback():
    source_matrix = np.array(
        [
            [1266.417236328125, 0.0, 816.2670288085938],
            [0.0, 1266.417236328125, 491.507080078125],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    intrinsics = IntrinsicsData(
        matrix=source_matrix,
        distortion=None,
        metadata={
            "source": "unit-test",
            "width": 1600,
            "height": 900,
        },
    )

    scaled = scale_intrinsics(intrinsics, (506, 900))

    np.testing.assert_allclose(
        scaled.matrix,
        np.array(
            [
                [712.3596802, 0.0, 459.1502075],
                [0.0, 712.0079346, 276.3362122],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-3,
    )
    assert scaled.metadata["input_resolution"] == [900.0, 1600.0]
    assert scaled.metadata["width"] == 900
    assert scaled.metadata["height"] == 506
    assert scaled.metadata["intrinsics_resolution_source"] == "metadata.width_height"
    assert scaled.metadata["intrinsics_resolution_was_heuristic"] is False


def test_validate_and_normalize_intrinsics_rejects_frame_shape_mismatch():
    matrix = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(IntrinsicsValidationError, match="frame shape"):
        validate_and_normalize_intrinsics(
            matrix,
            {"width": 640, "height": 480},
            frame_shape=(360, 640),
            allow_principal_point_fallback=False,
        )


def test_solve_blender_camera_for_intrinsics_matches_anisotropic_target():
    target = np.array(
        [
            [698.1633, 0.0, 449.99997],
            [0.0, 651.8798, 252.99998],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    solution = solve_blender_camera_for_intrinsics(
        target,
        width=900,
        height=506,
    )

    np.testing.assert_allclose(solution.effective_matrix, target, atol=1e-4)
    assert solution.focal_residual_px <= 1e-4
    assert solution.principal_point_residual_px <= 1e-4


def test_solve_blender_camera_for_intrinsics_rejects_invalid_matrix():
    with pytest.raises(BlenderCameraParityError, match="shape"):
        solve_blender_camera_for_intrinsics(np.eye(4, dtype=np.float32), width=900, height=506)


def test_resource_store_save_intrinsics_canonicalizes_metadata(tmp_path):
    store = ResourceStore("camera_calibration_store", root=tmp_path)
    matrix = np.array(
        [
            [25.0, 0.0, 16.0],
            [0.0, 25.0, 16.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    store.save_intrinsics(IntrinsicsData(matrix=matrix, distortion=None, metadata={"source": "unit-test"}))

    with np.load(store.path_for(ResourceKind.INTRINSICS), allow_pickle=True) as data:
        metadata = data["metadata"].item()
    assert metadata["width"] == 32
    assert metadata["height"] == 32
    assert metadata["intrinsics_resolution_was_heuristic"] is True
    assert metadata["intrinsics_resolution_source"] == "principal_point_fallback"
