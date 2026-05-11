"""External GTSAM factor-graph bridge for geometry fusion."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from pemoin.coordinate_systems.alignment import compute_up_direction_alignment
from pemoin.data.contracts import PoseData, PoseSample, ResourceStore
from pemoin.providers.geometry_fusion.settings import GeometryFusionSettings
from pemoin.providers.geometry_fusion.stages.quadratic_surface import QuadraticSurfaceResult
from pemoin.providers.geometry_fusion.stages.road_rectification import FrameRectificationResult
from pemoin.utils.env_launcher import resolve_env_launcher as _resolve_env_launcher
from pemoin.utils.logging import get_logger

LOG = get_logger()

_PARENTS = Path(__file__).resolve().parents
_REPO_ROOT_FALLBACK = _PARENTS[5] if len(_PARENTS) > 5 else _PARENTS[-1]


def _repo_root() -> Path:
    env_root = os.environ.get("PEMOIN_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return _REPO_ROOT_FALLBACK


def _bridge_script_path() -> Path:
    return _repo_root() / "tools" / "gtsam" / "pemoin_bridge.py"


def _settings_payload(settings: GeometryFusionSettings, camera_height_m: float) -> str:
    return json.dumps(
        {
            "camera_height_m": float(camera_height_m),
            "fg_window_size": int(settings.fg_window_size),
            "fg_overlap": int(settings.fg_overlap),
            "fg_dpvo_noise_rot_deg": float(settings.fg_dpvo_noise_rot_deg),
            "fg_dpvo_noise_trans": float(settings.fg_dpvo_noise_trans),
            "fg_height_noise_m": float(settings.fg_height_noise_m),
            "fg_huber_k": float(settings.fg_huber_k),
            "fg_max_iterations": int(settings.fg_max_iterations),
            "gate_min_inlier": float(settings.gate_min_inlier),
        }
    )


def _write_bridge_input(
    input_path: Path,
    poses: PoseData,
    rect_results: Sequence[FrameRectificationResult],
    camera_height_m: float,
    frame_indices: Sequence[int],
    settings: GeometryFusionSettings,
) -> None:
    pose_frame_indices = np.asarray(
        [int(sample.frame_index) for sample in poses.samples], dtype=np.int64
    )
    poses_c2w = np.asarray(
        [np.asarray(sample.camera_to_world, dtype=np.float32) for sample in poses.samples],
        dtype=np.float32,
    )
    rect_frame_indices = np.asarray(
        [int(result.frame_index) for result in rect_results], dtype=np.int64
    )
    rect_scales = np.asarray([float(result.scale) for result in rect_results], dtype=np.float32)
    rect_biases = np.asarray([float(result.bias) for result in rect_results], dtype=np.float32)
    rect_inlier_ratios = np.asarray(
        [float(result.inlier_ratio) for result in rect_results], dtype=np.float32
    )
    np.savez_compressed(
        input_path,
        frame_indices=np.asarray(list(frame_indices), dtype=np.int64),
        pose_frame_indices=pose_frame_indices,
        poses_c2w=poses_c2w,
        rect_frame_indices=rect_frame_indices,
        rect_scales=rect_scales,
        rect_biases=rect_biases,
        rect_inlier_ratios=rect_inlier_ratios,
        settings_json=np.asarray(_settings_payload(settings, camera_height_m)),
    )


def _rotation_inverse(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=np.float32).T


def _compute_preconditioning_rotation(poses: PoseData) -> np.ndarray:
    if not poses.samples:
        return np.eye(3, dtype=np.float32)
    up_vectors = np.asarray(
        [np.asarray(sample.camera_to_world, dtype=np.float32)[:3, 1] for sample in poses.samples],
        dtype=np.float32,
    )
    up_avg = np.mean(up_vectors, axis=0)
    if not np.isfinite(up_avg).all() or float(np.linalg.norm(up_avg)) < 1e-6:
        return np.eye(3, dtype=np.float32)
    return compute_up_direction_alignment(
        up_avg,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ).astype(np.float32)


def _apply_global_rotation_to_pose_data(poses: PoseData, rotation: np.ndarray) -> PoseData:
    rot = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    rotated_samples: list[PoseSample] = []
    for sample in poses.samples:
        c2w = np.asarray(sample.camera_to_world, dtype=np.float32).copy()
        c2w[:3, :3] = rot @ c2w[:3, :3]
        c2w[:3, 3] = rot @ c2w[:3, 3]
        rotated_samples.append(
            PoseSample(
                frame_index=int(sample.frame_index),
                camera_to_world=c2w,
                world_to_camera=np.linalg.inv(c2w.astype(np.float64)).astype(np.float32),
                confidence=sample.confidence,
                metadata=dict(sample.metadata or {}),
            )
        )
    meta = dict(poses.metadata or {})
    return PoseData(samples=rotated_samples, metadata=meta)


def _step_metrics(poses: PoseData) -> dict[str, float]:
    if len(poses.samples) < 2:
        return {"step_count": 0.0, "median_step_m": 0.0, "p95_step_m": 0.0, "max_step_m": 0.0}
    c2w = np.asarray(
        [np.asarray(sample.camera_to_world, dtype=np.float32) for sample in poses.samples],
        dtype=np.float32,
    )
    steps = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    return {
        "step_count": float(steps.size),
        "median_step_m": float(np.median(steps)),
        "p95_step_m": float(np.percentile(steps, 95)),
        "max_step_m": float(np.max(steps)),
    }


def _maybe_reject_discontinuous_result(
    original: PoseData,
    optimized: PoseData,
    settings: GeometryFusionSettings,
) -> tuple[bool, dict[str, float]]:
    orig = _step_metrics(original)
    opt = _step_metrics(optimized)
    ratio = float(opt["max_step_m"] / max(orig["max_step_m"], 1e-6))
    metrics = {
        **{f"original_{k}": v for k, v in orig.items()},
        **{f"optimized_{k}": v for k, v in opt.items()},
        "max_step_inflation_ratio": ratio,
    }
    reject = (
        float(opt["max_step_m"]) > float(settings.fg_max_step_jump_m)
        or ratio > float(settings.fg_max_step_inflation_ratio)
    )
    return reject, metrics


def _run_bridge(bridge_dir: Path, settings: GeometryFusionSettings) -> tuple[Path, Path]:
    bridge_script = _bridge_script_path()
    if not bridge_script.exists():
        raise RuntimeError(f"GTSAM bridge script not found at '{bridge_script}'.")

    input_path = bridge_dir / "input.npz"
    output_path = bridge_dir / "output.npz"

    launcher = _resolve_env_launcher(settings.fg_env_name, settings.fg_env_manager)
    cmd = [*launcher, "python", str(bridge_script), "--input", str(input_path), "--output", str(output_path)]

    env = os.environ.copy()
    if launcher[0] in {"micromamba", "mamba"} and not env.get("XDG_CACHE_HOME"):
        cache_root = bridge_dir / ".mamba_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        env["XDG_CACHE_HOME"] = str(cache_root)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=bridge_script.parent,
        env=env,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if result.returncode < 0:
            detail = f"terminated by signal {-result.returncode}"
        elif stderr:
            detail = stderr
        else:
            detail = f"exit code {result.returncode}"
        raise RuntimeError(f"GTSAM bridge failed: {detail}")

    return input_path, output_path


def _load_output_arrays(output_path: Path) -> dict[str, np.ndarray]:
    if not output_path.exists():
        raise RuntimeError(f"GTSAM bridge did not produce output at '{output_path}'.")

    required = {
        "optimized_pose_frame_indices",
        "optimized_poses_c2w",
        "optimized_rect_frame_indices",
        "optimized_rect_scales",
        "optimized_rect_biases",
    }
    with np.load(output_path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(
                f"GTSAM bridge output '{output_path}' is missing fields: {', '.join(missing)}."
            )
        return {name: np.asarray(data[name]) for name in required}


def _validate_output(
    arrays: dict[str, np.ndarray],
    pose_frame_set: set[int],
    rect_frame_set: set[int],
) -> None:
    pose_indices = np.asarray(arrays["optimized_pose_frame_indices"], dtype=np.int64)
    poses_c2w = np.asarray(arrays["optimized_poses_c2w"], dtype=np.float32)
    rect_indices = np.asarray(arrays["optimized_rect_frame_indices"], dtype=np.int64)
    rect_scales = np.asarray(arrays["optimized_rect_scales"], dtype=np.float32)
    rect_biases = np.asarray(arrays["optimized_rect_biases"], dtype=np.float32)

    if pose_indices.ndim != 1:
        raise RuntimeError("GTSAM bridge output has invalid optimized_pose_frame_indices shape.")
    if poses_c2w.ndim != 3 or poses_c2w.shape[1:] != (4, 4):
        raise RuntimeError("GTSAM bridge output has invalid optimized_poses_c2w shape.")
    if poses_c2w.shape[0] != pose_indices.shape[0]:
        raise RuntimeError("GTSAM bridge output pose counts do not match frame indices.")
    if rect_indices.ndim != 1:
        raise RuntimeError("GTSAM bridge output has invalid optimized_rect_frame_indices shape.")
    if rect_scales.ndim != 1 or rect_biases.ndim != 1:
        raise RuntimeError("GTSAM bridge output rectification arrays must be 1D.")
    if rect_indices.shape[0] != rect_scales.shape[0] or rect_indices.shape[0] != rect_biases.shape[0]:
        raise RuntimeError("GTSAM bridge output rectification counts do not match frame indices.")
    if not set(int(v) for v in pose_indices.tolist()).issubset(pose_frame_set):
        raise RuntimeError("GTSAM bridge returned pose frames outside the input set.")
    if not set(int(v) for v in rect_indices.tolist()).issubset(rect_frame_set):
        raise RuntimeError("GTSAM bridge returned rectification frames outside the input set.")


def _merge_poses(poses: PoseData, arrays: dict[str, np.ndarray]) -> PoseData:
    pose_indices = np.asarray(arrays["optimized_pose_frame_indices"], dtype=np.int64)
    poses_c2w = np.asarray(arrays["optimized_poses_c2w"], dtype=np.float32)
    updated_by_frame = {
        int(frame_index): poses_c2w[idx]
        for idx, frame_index in enumerate(pose_indices.tolist())
    }

    new_samples: list[PoseSample] = []
    for sample in poses.samples:
        frame_index = int(sample.frame_index)
        if frame_index not in updated_by_frame:
            new_samples.append(sample)
            continue
        c2w = np.asarray(updated_by_frame[frame_index], dtype=np.float32)
        w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
        new_samples.append(
            PoseSample(
                frame_index=frame_index,
                camera_to_world=c2w,
                world_to_camera=w2c,
                confidence=sample.confidence,
                metadata=dict(sample.metadata or {}),
            )
        )

    return PoseData(samples=new_samples, metadata=dict(poses.metadata or {}))


def _merge_rectification(
    rect_results: Sequence[FrameRectificationResult],
    arrays: dict[str, np.ndarray],
) -> list[FrameRectificationResult]:
    rect_indices = np.asarray(arrays["optimized_rect_frame_indices"], dtype=np.int64)
    rect_scales = np.asarray(arrays["optimized_rect_scales"], dtype=np.float32)
    rect_biases = np.asarray(arrays["optimized_rect_biases"], dtype=np.float32)
    updates = {
        int(frame_index): (float(rect_scales[idx]), float(rect_biases[idx]))
        for idx, frame_index in enumerate(rect_indices.tolist())
    }

    merged: list[FrameRectificationResult] = []
    for result in rect_results:
        if result.frame_index not in updates:
            merged.append(result)
            continue
        scale, bias = updates[result.frame_index]
        merged.append(
            FrameRectificationResult(
                frame_index=result.frame_index,
                normal_cam=result.normal_cam,
                offset_cam=result.offset_cam,
                implied_height_m=result.implied_height_m,
                scale=scale,
                bias=bias,
                inlier_ratio=result.inlier_ratio,
                residual_p90_m=result.residual_p90_m,
                support_count=result.support_count,
            )
        )
    return merged


def run_factor_graph_fusion(
    poses: PoseData,
    rect_results: list[FrameRectificationResult],
    quad_surfaces: list[QuadraticSurfaceResult],
    camera_height_m: float,
    K: np.ndarray,
    resources: ResourceStore | None,
    frame_indices: list[int],
    settings: GeometryFusionSettings,
) -> tuple[PoseData, list[FrameRectificationResult], list[QuadraticSurfaceResult]]:
    """Run factor-graph fusion in a dedicated external GTSAM environment."""
    del K

    if not settings.factor_graph_enabled:
        LOG.info("Factor graph disabled in settings; skipping.")
        return poses, rect_results, quad_surfaces

    n = len(frame_indices)
    if n < 3:
        LOG.info("Too few frames (%d) for factor graph fusion; skipping.", n)
        return poses, rect_results, quad_surfaces

    precondition_rotation = _compute_preconditioning_rotation(poses)
    preconditioned = _apply_global_rotation_to_pose_data(poses, precondition_rotation)

    base_dir: Path | None = None
    if resources is not None:
        base_dir = resources.provider_dir("geometry_fusion") / "factor_graph_bridge"
        base_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_dir) as temp_dir:
        bridge_dir = Path(temp_dir)
        input_path = bridge_dir / "input.npz"
        _write_bridge_input(
            input_path=input_path,
            poses=preconditioned,
            rect_results=rect_results,
            camera_height_m=camera_height_m,
            frame_indices=frame_indices,
            settings=settings,
        )
        _, output_path = _run_bridge(bridge_dir, settings)
        arrays = _load_output_arrays(output_path)

    pose_frame_set = {int(sample.frame_index) for sample in poses.samples}
    rect_frame_set = {int(result.frame_index) for result in rect_results}
    _validate_output(arrays, pose_frame_set=pose_frame_set, rect_frame_set=rect_frame_set)

    merged_preconditioned = _merge_poses(preconditioned, arrays)
    optimized_poses = _apply_global_rotation_to_pose_data(
        merged_preconditioned,
        _rotation_inverse(precondition_rotation),
    )
    rejected, continuity_metrics = _maybe_reject_discontinuous_result(
        poses,
        optimized_poses,
        settings,
    )
    if rejected:
        detail = (
            "Factor-graph optimized trajectory is discontinuous: "
            f"max_step_m={float(continuity_metrics['optimized_max_step_m']):.4f} "
            f"ratio={float(continuity_metrics['max_step_inflation_ratio']):.4f}."
        )
        if settings.fg_reject_on_discontinuity:
            raise RuntimeError(detail)
        if settings.fg_fallback_on_discontinuity:
            LOG.warning("%s Falling back to pre-factor-graph trajectory.", detail)
            fallback_meta = {
                **dict(poses.metadata or {}),
                "factor_graph_optimized": False,
                "factor_graph_rejected": True,
                "factor_graph_rejection_reason": "trajectory_discontinuity",
                "factor_graph_discontinuity_metrics": continuity_metrics,
            }
            return (
                PoseData(samples=list(poses.samples), metadata=fallback_meta),
                rect_results,
                quad_surfaces,
            )
        LOG.warning("%s Keeping optimized output because fallback is disabled.", detail)

    pose_meta = {
        **dict(optimized_poses.metadata or {}),
        "factor_graph_optimized": True,
        "factor_graph_preconditioned_z_up": bool(
            not np.allclose(precondition_rotation, np.eye(3, dtype=np.float32), atol=1e-5)
        ),
        "factor_graph_discontinuity_metrics": continuity_metrics,
    }
    return (
        PoseData(samples=optimized_poses.samples, metadata=pose_meta),
        _merge_rectification(rect_results, arrays),
        quad_surfaces,
    )
