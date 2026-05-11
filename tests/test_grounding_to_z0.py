from __future__ import annotations

import json
import numpy as np

from pemoin.coordinate_systems.alignment import GroundingSettings, ground_scene_to_z0
from pemoin.data.contracts import CameraHeightData, PoseData, PoseSample, ResourceStore, RoadPlaneData


def test_grounding_to_z0_shifts_scene_and_writes_summary(tmp_path):
    store = ResourceStore("grounding_z0", root=tmp_path)
    samples = []
    for frame_idx in range(1, 6):
        c2w = np.eye(4, dtype=np.float32)
        c2w[1, 3] = float(frame_idx) * 0.2
        c2w[2, 3] = 2.0
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender", "alignment_transform_id": "base"},
            )
        )
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                offset=-1.0,
                metadata={"source": "unit-test", "alignment_transform_id": "base"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.0,
                metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata={"metric_scale": True, "alignment_transform_id": "base"}))

    ground_scene_to_z0(
        store,
        settings=GroundingSettings(enabled=True, source="road_plane", min_ground_samples=3),
        road_labels=("road",),
        sidewalk_labels=("sidewalk",),
    )

    summary_path = store.visualizations_dir("alignment") / "grounding_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert abs(float(summary["median_ground_z_after_m"])) < 1e-4

    plane = store.load_road_plane(1)
    # z=0 plane after grounding (n=[0,0,1], d~0)
    assert abs(float(plane.offset)) < 1e-4


def test_grounding_to_z0_rejects_planes_above_camera(tmp_path):
    store = ResourceStore("grounding_rejects_mirrored_planes", root=tmp_path)
    samples = []
    for frame_idx in range(1, 6):
        c2w = np.eye(4, dtype=np.float32)
        c2w[2, 3] = 1.5
        samples.append(
            PoseSample(
                frame_index=frame_idx,
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w),
                metadata={"camera_convention": "blender", "alignment_transform_id": "base"},
            )
        )
        # This plane yields a negative anchor and therefore lies on the wrong support side.
        store.save_road_plane(
            RoadPlaneData(
                frame_index=frame_idx,
                normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                offset=-2.0,
                metadata={"source": "unit-test", "alignment_transform_id": "base"},
            )
        )
        store.save_camera_height(
            CameraHeightData(
                frame_index=frame_idx,
                height_m=1.5,
                metadata={"source": "unit-test", "axis": "z", "world_coordinate_system": "blender"},
            )
        )
    store.save_trajectory(PoseData(samples=samples, metadata={"metric_scale": True, "alignment_transform_id": "base"}))

    raised = False
    try:
        ground_scene_to_z0(
            store,
            settings=GroundingSettings(enabled=True, source="road_plane", min_ground_samples=3),
            road_labels=("road",),
            sidewalk_labels=("sidewalk",),
        )
    except RuntimeError:
        raised = True
    assert raised
