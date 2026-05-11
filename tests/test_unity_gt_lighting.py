from __future__ import annotations

import json
from pathlib import Path

from pemoin.data.unity import UnityLightingDataset
from pemoin.scripts.unity_import import ImportSelection, import_unity_dataset


def test_unity_lighting_dataset_resolves_sequence_parent_lighting_root(tmp_path: Path) -> None:
    export_root = tmp_path / "unity_export"
    sequence_dir = export_root / "sequence.0"
    lighting_root = export_root / "lighting_gt"
    sequence_dir.mkdir(parents=True)
    lighting_root.mkdir(parents=True)
    (lighting_root / "run_lighting.json").write_text(
        json.dumps({"pipeline": "HDRP"}),
        encoding="utf-8",
    )
    (lighting_root / "scene_lights.json").write_text(
        json.dumps({"mainDirectionalLight": {"enabled": True}}),
        encoding="utf-8",
    )
    (lighting_root / "frame_lighting.jsonl").write_text(
        json.dumps({"frameIndex": 0, "timestampSec": 0.0}) + "\n",
        encoding="utf-8",
    )
    faces = lighting_root / "reflection_probe_faces"
    faces.mkdir()
    for face in ("PositiveX", "NegativeX", "PositiveY", "NegativeY", "PositiveZ", "NegativeZ"):
        (faces / f"fallback_capture_{face}.exr").write_bytes(b"exr")

    dataset = UnityLightingDataset(sequence_dir)

    assert dataset.has_lighting_gt() is True
    assert dataset.lighting_root == lighting_root
    assert dataset.run_lighting()["pipeline"] == "HDRP"
    assert len(dataset.reflection_faces()) == 6


def test_unity_lighting_dataset_resolves_sibling_lighting_root(tmp_path: Path) -> None:
    root = tmp_path / "DefaultCompany" / "DT"
    export_root = root / "solo_37"
    lighting_root = root / "lighting_gt"
    export_root.mkdir(parents=True)
    lighting_root.mkdir(parents=True)
    (lighting_root / "run_lighting.json").write_text(
        json.dumps({"pipeline": "HDRP"}),
        encoding="utf-8",
    )
    (lighting_root / "scene_lights.json").write_text(
        json.dumps({"mainDirectionalLight": {"enabled": True}}),
        encoding="utf-8",
    )
    faces = lighting_root / "reflection_probe_faces"
    faces.mkdir()
    for face in ("PositiveX", "NegativeX", "PositiveY", "NegativeY", "PositiveZ", "NegativeZ"):
        (faces / f"fallback_capture_{face}.exr").write_bytes(b"exr")

    dataset = UnityLightingDataset(export_root)

    assert dataset.lighting_root == lighting_root


def test_unity_import_copies_lighting_gt_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    sequence_dir = source / "sequence.0"
    lighting_root = source / "lighting_gt"
    sequence_dir.mkdir(parents=True)
    lighting_root.mkdir(parents=True)
    (lighting_root / "run_lighting.json").write_text(
        json.dumps({"pipeline": "HDRP"}),
        encoding="utf-8",
    )
    (sequence_dir / "step0.frame_data.json").write_text(
        json.dumps(
            {
                "step": 0,
                "timestamp": 0.0,
                "captures": [
                    {
                        "id": "camera",
                        "filename": "step0.camera.png",
                        "dimension": [8.0, 8.0],
                        "annotations": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = import_unity_dataset(
        source,
        tmp_path / "dest",
        name="imported",
        stride=1,
        sampling_fps=None,
        resize_max_side=None,
        prune=True,
        write_videos=False,
        selection=ImportSelection(frames=False, depth=False, semantics=False),
    )

    assert (result.dest_dir / "lighting_gt" / "run_lighting.json").exists()


def test_unity_import_copies_sibling_lighting_gt_directory(tmp_path: Path) -> None:
    root = tmp_path / "DefaultCompany" / "DT"
    source = root / "solo_9"
    sequence_dir = source / "sequence.0"
    lighting_root = root / "lighting_gt"
    sequence_dir.mkdir(parents=True)
    lighting_root.mkdir(parents=True)
    (lighting_root / "run_lighting.json").write_text(
        json.dumps({"pipeline": "HDRP"}),
        encoding="utf-8",
    )
    (sequence_dir / "step0.frame_data.json").write_text(
        json.dumps(
            {
                "step": 0,
                "timestamp": 0.0,
                "captures": [
                    {
                        "id": "camera",
                        "filename": "step0.camera.png",
                        "dimension": [8.0, 8.0],
                        "annotations": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = import_unity_dataset(
        source,
        tmp_path / "dest",
        name="imported",
        stride=1,
        sampling_fps=None,
        resize_max_side=None,
        prune=True,
        write_videos=False,
        selection=ImportSelection(frames=False, depth=False, semantics=False),
    )

    assert (result.dest_dir / "lighting_gt" / "run_lighting.json").exists()
