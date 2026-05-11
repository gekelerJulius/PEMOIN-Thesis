"""Blender scene validation and execution helpers."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from pemoin.data.contracts import ResourceStore, lighting_from_payload
from pemoin.utils.camera_calibration import validate_and_normalize_intrinsics
from pemoin.utils.logging import run_logged_subprocess

LOG = logging.getLogger(__name__)


def validate_blender_scene_inputs(run_dir: Path) -> None:
    intrinsics_path = run_dir / "standard" / "intrinsics" / "intrinsics.npz"
    if not intrinsics_path.exists():
        raise FileNotFoundError(
            f"Cannot render Blender trajectory scene without intrinsics at {intrinsics_path}."
        )
    frames_dir = run_dir / "standard" / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Cannot render Blender trajectory scene without frames at {frames_dir}."
        )
    frame_candidates = sorted(frames_dir.glob("*.png"))
    if not frame_candidates:
        raise FileNotFoundError(
            f"Cannot render Blender trajectory scene because no frames were found in {frames_dir}."
        )
    import imageio.v2 as imageio

    sample_frame = imageio.imread(frame_candidates[0])
    with np.load(intrinsics_path, allow_pickle=True) as data:
        matrix = np.asarray(data["matrix"], dtype=np.float32)
        metadata = data["metadata"].item() if "metadata" in data.files else {}
    validate_and_normalize_intrinsics(
        matrix,
        metadata,
        frame_shape=sample_frame.shape[:2],
        allow_principal_point_fallback=False,
        fail_on_heuristic=True,
    )
    lighting_json = run_dir / "standard" / "lighting" / "lighting.json"
    if lighting_json.exists():
        payload = json.loads(lighting_json.read_text(encoding="utf-8"))
        validation = payload.get("validation", {})
        if not isinstance(validation, Mapping) or not bool(validation.get("passed", False)):
            raise ValueError(
                f"Lighting contract at {lighting_json} is not validated for Blender rendering."
            )
        envmap_rel = str(payload.get("envmap_path", "standard/lighting/envmap.exr"))
        envmap_path = run_dir / envmap_rel
        if not envmap_path.exists():
            raise FileNotFoundError(
                f"Lighting envmap declared by {lighting_json} was not found at {envmap_path}."
            )
        lighting_from_payload(payload, envmap_path=str(envmap_path), key=str(lighting_json))


def build_blender_trajectory_command(run_dir: Path, *, profile_name: str) -> list[str]:
    """Build the Blender subprocess command for trajectory scene generation."""
    if shutil.which("blender") is None:
        raise FileNotFoundError(
            "Blender executable not found in PATH. "
            "Install Blender to enable scene generation: https://www.blender.org/download/"
        )

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "src" / "pemoin" / "scripts" / "blender_trajectory_scene.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Blender trajectory script not found at {script_path}."
        )
    profile_snapshot_path = run_dir / "standard" / "profile.json"
    if not profile_snapshot_path.exists():
        raise FileNotFoundError(
            "Blender trajectory scene generation requires the saved run profile "
            f"snapshot at {profile_snapshot_path}."
        )
    output_path = run_dir / "scene.blend"
    cmd = [
        "blender",
        "--background",
        "--python",
        str(script_path),
        "--",
        "--run-dir",
        str(run_dir),
        "--output",
        str(output_path),
        "--host-python",
        str(Path(sys.executable).resolve()),
        "--config",
        str(profile_snapshot_path),
        "--profile",
        profile_name,
    ]
    return cmd


def render_trajectory_scene(
    run_dir: Path,
    *,
    profile_name: str,
    stream_output: bool = False,
    show_progress: bool = True,
):
    validate_blender_scene_inputs(run_dir)
    cmd = build_blender_trajectory_command(run_dir, profile_name=profile_name)
    logs_dir = run_dir / "standard" / "logs"
    result = run_logged_subprocess(
        cmd,
        stdout_log_path=logs_dir / "blender_scene.stdout.log",
        stderr_log_path=logs_dir / "blender_scene.stderr.log",
        stream_output=stream_output,
        show_progress=show_progress,
    )
    if result.returncode != 0:
        raise RuntimeError(result.format_failure(label="Blender trajectory scene generation"))
    pedestrian_frames_dir = ResourceStore.blender_artifact_dir_for(
        run_dir,
        "pedestrian_frames",
    )
    if not pedestrian_frames_dir.exists():
        raise RuntimeError(
            "Blender trajectory scene generation finished without pedestrian frames at "
            f"{pedestrian_frames_dir}."
        )
    shadow_frames_dir = ResourceStore.blender_artifact_dir_for(
        run_dir,
        "shadow_frames",
    )
    if not shadow_frames_dir.exists():
        raise RuntimeError(
            "Blender trajectory scene generation finished without shadow frames at "
            f"{shadow_frames_dir}."
        )
    LOG.info(
        "Blender subprocess logs written to %s and %s",
        result.stdout_log_path,
        result.stderr_log_path,
    )
    return result
