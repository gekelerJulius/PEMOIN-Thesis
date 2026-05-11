from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pemoin.coordinate_systems.alignment import (
    ComparisonFrameSettings,
    _resolve_unity_authoring_to_canonical_transform,
    _resolve_unity_gt_gravity_prior,
    compute_up_direction_alignment,
)


def _write_unity_rotation_frame(
    sequence_dir: Path,
    frame_idx: int,
    rotation_xyzw: list[float],
    *,
    position_xyz: list[float] | None = None,
) -> None:
    payload = {
        "step": int(frame_idx),
        "timestamp": float(frame_idx) * 0.1,
        "captures": [
            {
                "id": "camera",
                "position": [0.0, 0.0, 0.0] if position_xyz is None else position_xyz,
                "rotation": rotation_xyzw,
                "filename": f"step{frame_idx}.camera.png",
            }
        ],
    }
    (sequence_dir / f"step{frame_idx}.frame_data.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(float(angle_deg))
    x, y, z = axis.tolist()
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float32,
    )


def test_resolve_unity_gt_gravity_prior_projects_gt_up_into_estimated_world(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence.0"
    sequence_dir.mkdir(parents=True)
    for frame_idx in range(5):
        _write_unity_rotation_frame(sequence_dir, frame_idx, [0.0, 0.0, 0.0, 1.0])
    target_up = np.array([0.0, 0.6, 0.8], dtype=np.float32)
    target_up = target_up / np.linalg.norm(target_up)
    r_est = compute_up_direction_alignment(np.array([0.0, 1.0, 0.0], dtype=np.float32), target_up)
    c2w = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    c2w[:, :3, :3] = r_est
    cfg = ComparisonFrameSettings.from_mapping(
        {
            "mode": "estimated",
            "up_direction_source": "gravity_prior",
            "gravity_prior": {"provider": "unity_gt", "min_valid_frames": 5},
        }
    )

    resolved, diagnostics = _resolve_unity_gt_gravity_prior(
        frame_indices=np.arange(5, dtype=np.int32),
        c2w=c2w,
        cfg=cfg,
        context={"frame_source": sequence_dir},
    )

    assert resolved == pytest.approx(target_up, abs=1e-5)
    assert diagnostics["provider"] == "unity_gt"
    assert diagnostics["inlier_frames"] == 5


def test_resolve_unity_gt_gravity_prior_rejects_large_outliers(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence.0"
    sequence_dir.mkdir(parents=True)
    for frame_idx in range(6):
        _write_unity_rotation_frame(sequence_dir, frame_idx, [0.0, 0.0, 0.0, 1.0])
    target_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    c2w = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
    outlier_rot = _rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float32), 90.0)
    c2w[5, :3, :3] = outlier_rot
    cfg = ComparisonFrameSettings.from_mapping(
        {
            "mode": "estimated",
            "up_direction_source": "gravity_prior",
            "gravity_prior": {
                "provider": "unity_gt",
                "min_valid_frames": 5,
                "max_outlier_angle_deg": 10.0,
            },
        }
    )

    resolved, diagnostics = _resolve_unity_gt_gravity_prior(
        frame_indices=np.arange(6, dtype=np.int32),
        c2w=c2w,
        cfg=cfg,
        context={"frame_source": sequence_dir},
    )

    assert resolved == pytest.approx(target_up, abs=1e-5)
    assert diagnostics["inlier_frames"] == 5
    assert diagnostics["rejected_frames"] == 1


def test_resolve_unity_authoring_to_canonical_transform_maps_unity_horizontal_basis(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence.0"
    sequence_dir.mkdir(parents=True)
    authored_positions = (
        [0.0, 1.5, 0.0],
        [1.0, 1.5, 0.0],
        [0.0, 1.5, 2.0],
    )
    for frame_idx, position in enumerate(authored_positions):
        _write_unity_rotation_frame(
            sequence_dir,
            frame_idx,
            [0.0, 0.0, 0.0, 1.0],
            position_xyz=position,
        )
    final_c2w = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    final_c2w[0, :3, 3] = np.array([10.0, 20.0, 1.5], dtype=np.float32)
    final_c2w[1, :3, 3] = np.array([10.0, 21.0, 1.5], dtype=np.float32)
    final_c2w[2, :3, 3] = np.array([8.0, 20.0, 1.5], dtype=np.float32)

    metadata = _resolve_unity_authoring_to_canonical_transform(
        frame_indices=np.arange(3, dtype=np.int32),
        final_c2w=final_c2w,
        context={"frame_source": sequence_dir},
    )

    assert metadata is not None
    transform = np.asarray(metadata["authoring_to_canonical_transform"], dtype=np.float32)
    authored_point = np.array([3.0, 0.0, 4.0, 1.0], dtype=np.float32)
    resolved = transform @ authored_point
    np.testing.assert_allclose(
        resolved[:3],
        np.array([6.0, 23.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )
