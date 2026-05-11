from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from pemoin.data.carla import CarlaDataset


def test_carla_dataset_loads_optional_lighting_gt_metadata(tmp_path: Path) -> None:
    export_root = tmp_path / "carla_export"
    (export_root / "lighting_gt").mkdir(parents=True)
    (export_root / "rgb").mkdir()
    (export_root / "depth_m").mkdir()

    (export_root / "camera_intrinsics.json").write_text(
        json.dumps(
            {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2, "height": 2}
        ),
        encoding="utf-8",
    )
    (export_root / "run_config.json").write_text(
        json.dumps({"fps": 10.0}),
        encoding="utf-8",
    )
    (export_root / "frames.jsonl").write_text(
        json.dumps(
            {
                "frame": 1,
                "timestamp": 0.1,
                "T_world_from_camera": np.eye(4, dtype=float).tolist(),
                "rgb": "rgb/000001.jpg",
                "depth_m": "depth_m/000001.npy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (export_root / "lighting_gt" / "run_lighting.json").write_text(
        json.dumps({"weather": {"sun_altitude_angle": 35.0}}),
        encoding="utf-8",
    )
    (export_root / "lighting_gt" / "scene_lights.json").write_text(
        json.dumps({"lights": [{"id": 1, "is_on": True}]}),
        encoding="utf-8",
    )
    (export_root / "lighting_gt" / "frame_lighting.jsonl").write_text(
        json.dumps({"frame": 1, "timestamp": 0.1, "weather": {"cloudiness": 20.0}}) + "\n",
        encoding="utf-8",
    )
    (export_root / "rgb" / "000001.jpg").write_bytes(b"jpg")
    np.save(export_root / "depth_m" / "000001.npy", np.ones((2, 2), dtype=np.float32))

    dataset = CarlaDataset(export_root)

    assert dataset.has_lighting_gt() is True
    assert dataset.run_lighting()["weather"]["sun_altitude_angle"] == 35.0
    assert dataset.scene_lights()["lights"][0]["id"] == 1
    assert dataset.frame_lighting(1)["weather"]["cloudiness"] == 20.0


def test_start_carla_build_command_defaults_to_epic_quality() -> None:
    script_path = Path(__file__).resolve().parents[1] / "carla_scripts" / "start_carla.py"
    spec = importlib.util.spec_from_file_location("start_carla_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Args:
        quality_level = "Epic"
        resolution = (1920, 1080)
        port = 2000
        opengl = False
        offscreen = False

    command = module.build_command(Args())

    assert command[0].endswith("CarlaUE4.sh")
    assert "-quality-level=Epic" in command
    assert "-carla-rpc-port=2000" in command
    assert "-ResX=1920" in command
    assert "-ResY=1080" in command
    assert "-windowed" in command
