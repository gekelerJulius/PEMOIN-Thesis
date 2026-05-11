"""
Command-line entry point for running PEMOIN pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import sys
from datetime import datetime

import imageio.v3 as iio
import numpy as np
from PIL import Image
from pemoin.providers.adapters.megasam.job_runner import (
    MegaSAMJobCommand,
    MegaSAMJobConfig,
    MegaSAMJobRunner,
)
from pemoin.providers.adapters.megasam.client import ensure_megasam_log_handler
from pemoin.providers.adapters.panst3r.job_runner import (
    PanSt3RJobCommand,
    PanSt3RJobConfig,
    PanSt3RJobRunner,
)
from pemoin.runtime.orchestration.frame_provider import FrameProvider
from pemoin.runtime.orchestration.directory_frame_provider import DirectoryFrameProvider
from pemoin.runtime.orchestration.frame_provider_builder import (
    create_frame_provider_from_binding,
)
from pemoin.runtime.orchestration.unity_frame_provider import UnityFrameProvider
from pemoin.runtime.orchestration.video_frame_provider import VideoFrameProvider
from pemoin.runtime.bootstrap import create_runtime_launch, save_profile_snapshot
from pemoin.runtime.profiles.config import ProfileConfig, load_profiles_from_json
from pemoin.runtime.profiles.registry import ProfileRegistry
from pemoin.utils.logging import (
    ConsoleLoggingConfig,
    resolve_console_logging_config,
    setup_console_logging,
)

_DEFAULT_CONSOLE_LOGGING = resolve_console_logging_config()
LOGGER = setup_console_logging(
    level=_DEFAULT_CONSOLE_LOGGING.level,
    summary_only=_DEFAULT_CONSOLE_LOGGING.summary_only,
)


def _summary(message: str, *args: object) -> None:
    LOGGER.info(message, *args, extra={"summary": True})


def _configure_console_logging_from_args(args: argparse.Namespace) -> ConsoleLoggingConfig:
    config = resolve_console_logging_config(
        quiet=bool(getattr(args, "quiet", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )
    setup_console_logging(level=config.level, summary_only=config.summary_only)
    return config


def _load_env_settings(env_path: Path = Path(".env")) -> Dict[str, str]:
    """
    Load KEY=VALUE pairs from a .env file and seed os.environ where unset.

    This is intentionally lightweight to avoid extra dependencies.
    """
    settings: Dict[str, str] = {}
    if not env_path.exists():
        return settings
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        settings[key] = value
        os.environ.setdefault(key, value)
    return settings


def _build_registry(config_path: Path) -> ProfileRegistry:
    LOGGER.info("Loading profiles from %s", config_path)
    configs = load_profiles_from_json(config_path)
    registry = ProfileRegistry()
    for profile in configs.values():
        registry.register(profile)
    return registry


def _normalize_for_json(value: Any) -> Any:
    """Convert profile objects into JSON-friendly primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(v) for v in value]
    return value


def _normalize_for_signature(value: Any) -> Any:
    """Normalise values for stable hash signatures."""
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_signature(v) for v in value]
    return value


def _profile_snapshot(
    profile: ProfileConfig,
    *,
    config_path: Path,
    frame_source: Path,
    frame_provider_info: Mapping[str, object],
    run_timestamp: str,
) -> Dict[str, Any]:
    """Build a serialisable snapshot of the active profile configuration."""

    def _binding_dict(binding) -> Dict[str, Any]:
        return {"tool": binding.tool, "settings": _normalize_for_json(binding.settings)}

    config_hash = None
    try:
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except Exception:
        config_hash = None

    return {
        "profile": profile.name,
        "source_config": str(config_path),
        "source_config_sha256": config_hash,
        "run_timestamp": run_timestamp,
        "frame_source": str(frame_source),
        "frame_provider": _normalize_for_json(frame_provider_info),
        "working_resolution": _normalize_for_json(profile.working_resolution),
        "runtime": {
            "state_window": profile.runtime.state_window,
            "degradation_policy": profile.runtime.degradation_policy,
            "settings": _normalize_for_json(profile.runtime.settings),
        },
        "providers": {
            name: _binding_dict(binding) for name, binding in profile.providers.items()
        },
        "effects": {
            name: _binding_dict(binding) for name, binding in profile.effects.items()
        },
        "megasam": _normalize_for_json(profile.megasam),
        "depthanything3": _normalize_for_json(profile.depthanything3),
        "panst3r": _normalize_for_json(profile.panst3r),
        "unity_import": _normalize_for_json(profile.unity_import),
        "mixamo": _normalize_for_json(profile.mixamo),
    }


def _parse_megasam_commands(
    raw_commands: Optional[Sequence[str]],
) -> List[MegaSAMJobCommand]:
    commands: List[MegaSAMJobCommand] = []
    if not raw_commands:
        return commands
    for idx, raw in enumerate(raw_commands, start=1):
        if not raw.strip():
            continue
        commands.append(MegaSAMJobCommand(label=f"cmd-{idx}", args=shlex.split(raw)))
    return commands


def _build_standard_preset_command(
    repo_root: Path,
    *,
    checkpoint: Path,
    depth_checkpoint: Path,
    raft_checkpoint: Path,
    cuda_devices: str,
) -> List[MegaSAMJobCommand]:
    return [
        MegaSAMJobCommand(
            label="megasam-standard-pipeline",
            args=[
                "python",
                "pemoin_pipeline.py",
                "--frames",
                "{frames}",
                "--scene-name",
                "{scene}",
                "--bundle",
                "{bundle}",
                "--output-dir",
                "{output}",
                "--megasam-checkpoint",
                str(checkpoint),
                "--depth-anything-checkpoint",
                str(depth_checkpoint),
                "--raft-checkpoint",
                str(raft_checkpoint),
                "--cuda-devices",
                cuda_devices,
                "--tracking-resize-mode",
                "preserve_input",
                "--tracking-pad-mode",
                "pad",
                "--tracking-multiple-of",
                "8",
            ],
            env={"PEMOIN_GT_INTRINSICS_NPZ": "{intrinsics}"},
            skip_if_exists="{bundle}",
        )
    ]


def _resolve_megasam_commands(
    args: argparse.Namespace, repo_root: Path
) -> List[MegaSAMJobCommand]:
    if args.megasam_commands:
        return _parse_megasam_commands(args.megasam_commands)
    if not args.megasam_preset:
        return []

    preset = args.megasam_preset.lower()
    checkpoint = (
        Path(args.megasam_checkpoint).resolve()
        if args.megasam_checkpoint
        else (repo_root / "checkpoints/megasam_final.pth").resolve()
    )
    depth_ckpt = (
        Path(args.megasam_depth_checkpoint).resolve()
        if args.megasam_depth_checkpoint
        else (
            repo_root / "Depth-Anything/checkpoints/depth_anything_vitl14.pth"
        ).resolve()
    )
    raft_ckpt = (
        Path(args.megasam_raft_checkpoint).resolve()
        if args.megasam_raft_checkpoint
        else (repo_root / "cvd_opt/raft-things.pth").resolve()
    )

    if preset == "standard":
        LOGGER.info("Using MegaSAM preset '%s'", preset)
        return _build_standard_preset_command(
            repo_root,
            checkpoint=checkpoint,
            depth_checkpoint=depth_ckpt,
            raft_checkpoint=raft_ckpt,
            cuda_devices=args.megasam_cuda_devices,
        )
    raise SystemExit(f"Unknown MegaSAM preset '{args.megasam_preset}'.")


def _parse_panst3r_commands(
    raw_commands: Optional[Sequence[str]],
) -> List[PanSt3RJobCommand]:
    if not raw_commands:
        return []
    commands: List[PanSt3RJobCommand] = []
    for idx, raw in enumerate(raw_commands, start=1):
        if not raw.strip():
            continue
        commands.append(
            PanSt3RJobCommand(label=f"panst3r-cmd-{idx}", args=shlex.split(raw))
        )
    return commands


def _build_panst3r_preset_command(
    repo_root: Path, settings_path: Optional[Path]
) -> List[PanSt3RJobCommand]:
    script = repo_root / "tools" / "panst3r" / "panst3r_pipeline.py"
    if not script.exists():
        script = repo_root / "panst3r_pipeline.py"
    args = [
        "python",
        str(script),
        "--frames",
        "{frames}",
        "--scene-name",
        "{scene}",
        "--bundle",
        "{bundle}",
        "--output-dir",
        "{output}",
    ]
    if settings_path:
        args.extend(["--settings-file", str(settings_path)])
    return [
        PanSt3RJobCommand(
            label="panst3r-bundle",
            args=args,
            skip_if_exists="{bundle}",
        )
    ]


def _resolve_panst3r_commands(
    args: argparse.Namespace,
    repo_root: Path,
    settings_path: Optional[Path],
) -> List[PanSt3RJobCommand]:
    if args.panst3r_commands:
        return _parse_panst3r_commands(args.panst3r_commands)
    if not args.panst3r_preset:
        return []
    preset = args.panst3r_preset.lower()
    if preset in ("default", "stub"):
        LOGGER.info("Using PanSt3R preset '%s'", preset)
        return _build_panst3r_preset_command(repo_root, settings_path)
    raise SystemExit(f"Unknown PanSt3R preset '{args.panst3r_preset}'.")


def _supports_color() -> bool:
    stream = getattr(sys.stderr, "isatty", None)
    return bool(stream and stream())


def _profile_banner(profile_name: str) -> str:
    banner = f"=== ACTIVE PROFILE: {profile_name} ==="
    if _supports_color():
        return f"\033[95m{banner}\033[0m"
    return banner


def _pipeline_folder_name(
    profile_name: str, frame_source: Path, run_timestamp: str
) -> str:
    """
    Compose the pipeline folder name: <profile>_<timestamp>_<source>.
    """
    source_name = frame_source.stem if frame_source.is_file() else frame_source.name
    source_component = source_name.strip().replace(" ", "_") or "scene"
    timestamp_component = run_timestamp.strip().replace(" ", "_") or "run"
    profile_component = profile_name.strip().replace(" ", "_") or "pipeline"
    return f"{profile_component}_{timestamp_component}_{source_component}"


def _resolve_cache_path(value: object, base_root: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_root / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_megasam_bundle_cache_root(
    profile: ProfileConfig, base_root: Path
) -> Optional[Path]:
    raw_root = profile.megasam.get("bundle_cache_root") or profile.megasam.get(
        "cache_root"
    )
    if raw_root:
        return _resolve_cache_path(raw_root, base_root)
    for binding in profile.providers.values():
        if not binding.tool.startswith("MegaSAM"):
            continue
        adapter_settings = binding.settings.get("adapter", {})
        raw_root = adapter_settings.get("bundle_cache_root") or adapter_settings.get(
            "cache_root"
        )
        if raw_root:
            return _resolve_cache_path(raw_root, base_root)
        cache_dir = adapter_settings.get("cache_dir")
        if cache_dir:
            cache_root = _resolve_cache_path(cache_dir, base_root)
            if cache_root is not None:
                return cache_root / "bundles"
    return None


def _frame_dir_signature(frame_dir: Path) -> Dict[str, object]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    frames = sorted(
        path
        for path in frame_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    entries: List[str] = []
    max_mtime = 0.0
    for frame in frames:
        stat = frame.stat()
        rel = frame.relative_to(frame_dir)
        entries.append(f"{rel.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
        if stat.st_mtime > max_mtime:
            max_mtime = stat.st_mtime
    digest = (
        hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest() if entries else None
    )
    return {
        "frame_count": len(frames),
        "frame_digest": digest,
        "max_mtime": max_mtime,
    }


def _file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _npz_matrix_signature(path: Path, key: str = "matrix") -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise SystemExit(f"Expected key '{key}' in NPZ: {path}")
        matrix = np.asarray(data[key], dtype=np.float32)
    return {
        "key": key,
        "shape": tuple(int(v) for v in matrix.shape),
        "sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    }


def _megasam_signature_payload(
    args: argparse.Namespace,
    frame_source: Path,
    provider_info: Dict[str, Dict[str, object]],
    scene_name: str,
    megasam_repo_root: Path,
    gt_intrinsics_npz: Optional[Path] = None,
) -> Dict[str, object]:
    frame_payload: Dict[str, object] = {
        "path": str(frame_source.resolve()),
        "kind": "directory" if frame_source.is_dir() else "file",
        "provider": provider_info.get("tool"),
    }
    if frame_source.is_dir():
        frame_payload.update(_frame_dir_signature(frame_source))
    else:
        frame_payload.update(_file_signature(frame_source))
    provider_settings = provider_info.get("settings", {})
    if isinstance(provider_settings, dict):
        frame_payload["provider_settings"] = _normalize_for_signature(provider_settings)
    if provider_info.get("tool") == "VideoFrameProvider":
        frame_payload["video_settings"] = _normalize_for_signature(
            _video_settings_from_info(provider_settings if isinstance(provider_settings, dict) else {})
        )

    if args.megasam_commands:
        megasam_payload: Dict[str, object] = {
            "commands": list(args.megasam_commands),
        }
    else:
        checkpoint = (
            Path(args.megasam_checkpoint).resolve()
            if args.megasam_checkpoint
            else (megasam_repo_root / "checkpoints/megasam_final.pth").resolve()
        )
        depth_ckpt = (
            Path(args.megasam_depth_checkpoint).resolve()
            if args.megasam_depth_checkpoint
            else (
                megasam_repo_root
                / "Depth-Anything/checkpoints/depth_anything_vitl14.pth"
            ).resolve()
        )
        raft_ckpt = (
            Path(args.megasam_raft_checkpoint).resolve()
            if args.megasam_raft_checkpoint
            else (megasam_repo_root / "cvd_opt/raft-things.pth").resolve()
        )
        megasam_payload = {
            "preset": args.megasam_preset,
            "checkpoint_path": str(checkpoint),
            "depth_checkpoint": str(depth_ckpt),
            "raft_checkpoint": str(raft_ckpt),
            "cuda_devices": str(args.megasam_cuda_devices),
        }
    if gt_intrinsics_npz is not None and gt_intrinsics_npz.exists():
        megasam_payload["gt_intrinsics_signature"] = _npz_matrix_signature(
            gt_intrinsics_npz, key="matrix"
        )

    return {
        "scene_name": scene_name,
        "frame_source": frame_payload,
        "megasam": _normalize_for_signature(megasam_payload),
    }


def _megasam_bundle_signature(payload: Dict[str, object]) -> str:
    normalized = _normalize_for_signature(payload)
    raw = json.dumps(normalized, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _megasam_metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "metadata.json"


def _megasam_metadata_matches(meta_path: Path, signature: str) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("signature") == signature


def _write_megasam_metadata(
    meta_path: Path, signature: str, payload: Dict[str, object]
) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "signature": signature,
        "payload": _normalize_for_signature(payload),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _prepare_carla_gt_intrinsics_npz(
    profile: ProfileConfig,
    frame_source: Path,
    provider_info: Dict[str, Dict[str, object]],
    provider_root: Path,
) -> Optional[Path]:
    """
    Export CARLA GT intrinsics as an NPZ override consumable by MegaSAM automation.
    """
    intr_binding = profile.providers.get("intrinsics")
    if intr_binding is None or intr_binding.tool != "CarlaIntrinsicsProvider":
        return None
    if provider_info.get("tool") != "CarlaFrameProvider":
        return None

    from pemoin.data.carla import CarlaDataset

    dataset = CarlaDataset(frame_source if frame_source.is_dir() else frame_source.parent)
    intr = dataset.intrinsics()
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    src_w = int(intr.get("width", 0))
    src_h = int(intr.get("height", 0))
    if src_w <= 0 or src_h <= 0:
        raise SystemExit("CARLA intrinsics width/height must be positive for MegaSAM automation.")

    target_h, target_w = src_h, src_w
    if profile.working_resolution is not None:
        if isinstance(profile.working_resolution, (int, float)):
            target_max = int(profile.working_resolution)
        elif isinstance(profile.working_resolution, (list, tuple)) and len(profile.working_resolution) >= 1:
            target_max = int(max(profile.working_resolution))
        else:
            raise SystemExit(
                "working_resolution must be int/float or sequence with at least one value."
            )
        if target_max <= 0:
            raise SystemExit("working_resolution must be > 0 for MegaSAM CARLA intrinsics scaling.")
        if max(src_h, src_w) != target_max:
            scale = float(target_max) / float(max(src_h, src_w))
            target_h = int(max(1, round(src_h * scale)))
            target_w = int(max(1, round(src_w * scale)))

    scale_x = float(target_w) / float(src_w)
    scale_y = float(target_h) / float(src_h)
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    out = provider_root / "gt_intrinsics.npz"
    np.savez(
        out,
        matrix=matrix,
        metadata={
            "source": "carla_gt",
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "source_width": src_w,
            "source_height": src_h,
            "width": target_w,
            "height": target_h,
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
    )
    LOGGER.info("Prepared GT intrinsics override for MegaSAM automation: %s", out)
    return out


def _run_megasam_pointcloud_tools(
    *,
    bundle_path: Path,
    scene_name: str,
    repo_root: Path,
    provider_root: Path,
    conda_env: Optional[str],
    pointcloud_debug: bool,
) -> Path:
    export_script = (repo_root / "tools" / "export_pointcloud.py").resolve()
    if not export_script.exists():
        raise SystemExit(
            f"MegaSAM pointcloud export script not found at '{export_script}'."
        )

    view_script = (repo_root / "tools" / "view_pointcloud.py").resolve()
    if pointcloud_debug and not view_script.exists():
        raise SystemExit(
            f"MegaSAM pointcloud viewer script not found at '{view_script}'."
        )

    pointcloud_dir = (provider_root / "pointcloud").resolve()
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    pointcloud_path = (pointcloud_dir / f"{scene_name}.ply").resolve()
    commands = [
        MegaSAMJobCommand(
            label="megasam-export-pointcloud",
            args=[
                "python",
                str(export_script),
                "--input",
                "{bundle}",
                "--output",
                str(pointcloud_path),
            ],
        )
    ]
    if pointcloud_debug:
        commands.append(
            MegaSAMJobCommand(
                label="megasam-view-pointcloud",
                args=[
                    "python",
                    str(view_script),
                    "--input",
                    str(pointcloud_path),
                ],
            )
        )
    LOGGER.info("Generating MegaSAM pointcloud debug artifact at %s", pointcloud_path)
    if pointcloud_debug:
        LOGGER.info(
            "MegaSAM pointcloud viewer enabled; close the viewer (Ctrl+C) to continue the pipeline."
        )
    runner = MegaSAMJobRunner(logger=LOGGER.info)
    runner.run(
        MegaSAMJobConfig(
            frame_dir=bundle_path.parent,
            scene_name=scene_name,
            repo_root=repo_root,
            output_dir=bundle_path.parent,
            bundle_name=bundle_path.name,
            commands=commands,
            conda_env=conda_env,
            intrinsics_npz=None,
        )
    )
    return pointcloud_path


def _prepare_megasam_bundle(
    args: argparse.Namespace,
    profile: ProfileConfig,
    frame_source: Path,
    provider_info: Dict[str, Dict[str, object]],
    run_dir: Path,
    conda_env: Optional[str],
    base_root: Path,
) -> Tuple[Path, str, Optional[Path], Optional[Path], Optional[Path]]:
    repo_root = args.megasam_repo.resolve()
    scene_name = args.megasam_scene or (
        frame_source.stem if frame_source.is_file() else frame_source.name
    )
    provider_root = run_dir / "raw" / "mega-sam"
    provider_root.mkdir(parents=True, exist_ok=True)
    gt_intrinsics_npz = _prepare_carla_gt_intrinsics_npz(
        profile, frame_source, provider_info, provider_root
    )
    bundle_path: Optional[Path]
    cache_root = _resolve_megasam_bundle_cache_root(profile, base_root)
    cache_signature = None
    cache_payload: Optional[Dict[str, object]] = None
    cache_metadata_path: Optional[Path] = None
    if args.megasam_output:
        bundle_path = Path(args.megasam_output)
        if not bundle_path.is_absolute():
            bundle_path = (provider_root / bundle_path).resolve()
        else:
            bundle_path = bundle_path.resolve()
    elif cache_root is not None:
        cache_payload = _megasam_signature_payload(
            args,
            frame_source,
            provider_info,
            scene_name,
            repo_root,
            gt_intrinsics_npz=gt_intrinsics_npz,
        )
        cache_signature = _megasam_bundle_signature(cache_payload)
        cache_dir = cache_root / cache_signature
        bundle_path = (cache_dir / "bundle.npz").resolve()
        cache_metadata_path = _megasam_metadata_path(cache_dir)
        LOGGER.info("MegaSAM bundle cache target: %s", bundle_path)
    else:
        bundle_path = (provider_root / f"{scene_name}_sgd_cvd_hr.npz").resolve()

    cache_valid = True
    if cache_metadata_path is not None and cache_signature is not None:
        cache_valid = _megasam_metadata_matches(cache_metadata_path, cache_signature)
        if not cache_valid and bundle_path.exists() and args.megasam_auto:
            LOGGER.warning(
                "MegaSAM cache metadata mismatch at %s; clearing cached outputs.",
                bundle_path.parent,
            )
            shutil.rmtree(bundle_path.parent, ignore_errors=True)
        if not cache_valid and not args.megasam_auto:
            raise SystemExit(
                "MegaSAM cache metadata does not match the current settings. "
                "Enable --megasam-auto to regenerate or remove the cached bundle directory."
            )

    needs_automation = args.megasam_auto and (
        not bundle_path.exists() or not cache_valid
    )
    frame_map_path: Optional[Path] = None
    tool = (provider_info or {}).get("tool")
    frame_dir: Optional[Path] = None
    if tool == "CarlaFrameProvider":
        frames_dir = provider_root / "frames_extracted"
        if (
            needs_automation
            or not frames_dir.exists()
            or not _carla_frames_sequential(frames_dir)
        ):
            frame_dir = _resolve_automation_frame_directory(
                frame_source, provider_info, provider_root, profile=profile
            )
        else:
            frame_dir = frames_dir
        frame_map_path = _resolve_frame_index_map(frame_dir)
    if needs_automation and tool != "CarlaFrameProvider":
        frame_dir = _resolve_automation_frame_directory(
            frame_source, provider_info, provider_root, profile=profile
        )
        frame_map_path = _resolve_frame_index_map(frame_dir)
    if needs_automation:
        if frame_dir is None:
            frame_dir = _resolve_automation_frame_directory(
                frame_source, provider_info, provider_root, profile=profile
            )
            frame_map_path = _resolve_frame_index_map(frame_dir)
        commands = _resolve_megasam_commands(args, repo_root)
        if not commands:
            raise SystemExit(
                "MegaSAM automation requested but no commands were provided. "
                "Use --megasam-command or --megasam-preset to describe the pipeline stages."
            )
        LOGGER.info("MegaSAM automation enabled. Generating bundle from %s", frame_dir)
        LOGGER.info(
            "Invoking MegaSAM tools inside conda env '%s'",
            conda_env or "default shell",
        )
        config = MegaSAMJobConfig(
            frame_dir=frame_dir,
            scene_name=scene_name,
            repo_root=repo_root,
            output_dir=bundle_path.parent,
            bundle_name=bundle_path.name,
            commands=commands,
            conda_env=conda_env,
            intrinsics_npz=gt_intrinsics_npz,
        )
        runner = MegaSAMJobRunner()
        bundle_path = runner.run(config)
        if (
            cache_metadata_path is not None
            and cache_signature is not None
            and cache_payload is not None
        ):
            _write_megasam_metadata(cache_metadata_path, cache_signature, cache_payload)
    else:
        if args.megasam_auto:
            LOGGER.info("Reusing existing MegaSAM bundle: %s", bundle_path)
        else:
            LOGGER.warning(
                "MegaSAM automation disabled; expecting bundle at %s", bundle_path
            )

    if not bundle_path.exists():
        raise SystemExit(
            f"MegaSAM bundle not found at '{bundle_path}'. "
            "Run with --megasam-auto (default) or provide --megasam-output."
        )
    _run_megasam_pointcloud_tools(
        bundle_path=bundle_path,
        scene_name=scene_name,
        repo_root=repo_root,
        provider_root=provider_root,
        conda_env=conda_env,
        pointcloud_debug=bool(args.megasam_pointcloud_debug),
    )
    tracking_preprocess_path = bundle_path.with_suffix(".tracking.json")
    if not tracking_preprocess_path.exists():
        tracking_preprocess_path = None
    return bundle_path, scene_name, frame_map_path, gt_intrinsics_npz, tracking_preprocess_path


def _resolve_automation_frame_directory(
    frame_source: Path,
    provider_info: Dict[str, Dict[str, object]],
    provider_root: Path,
    profile: Optional[ProfileConfig] = None,
) -> Path:
    tool = (provider_info or {}).get("tool")
    if frame_source.is_dir():
        if tool == "CarlaFrameProvider":
            frames_dir = provider_root / "frames_extracted"
            settings = (
                provider_info.get("settings", {})
                if isinstance(provider_info, dict)
                else {}
            )
            settings = dict(settings)
            if profile is not None and profile.working_resolution is not None:
                wr = profile.working_resolution
                if isinstance(wr, (int, float)):
                    settings["resize_max_side"] = int(wr)
                elif isinstance(wr, (list, tuple)):
                    settings["resize_max_side"] = int(max(wr))
                else:
                    raise SystemExit(
                        "working_resolution must be int/float or sequence for CARLA MegaSAM automation."
                    )
            resize_ok = _carla_frames_match_resize(
                frames_dir, int(settings["resize_max_side"])
            ) if settings.get("resize_max_side") is not None else True
            if (
                not _directory_contains_frames(frames_dir)
                or not _carla_frames_sequential(frames_dir)
                or not resize_ok
            ):
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)
                _export_carla_frames(frame_source, frames_dir, settings)
            else:
                LOGGER.info(
                    "Reusing previously extracted CARLA frames at '%s'", frames_dir
                )
            return frames_dir
        LOGGER.info("Using frame directory '%s' for automation inputs", frame_source)
        return frame_source
    if tool != "VideoFrameProvider":
        raise SystemExit(
            "Automation cannot convert this source type automatically. "
            "Provide --frames pointing to an image directory or disable automation."
        )
    frames_dir = provider_root / "frames_extracted"
    if not _directory_contains_frames(frames_dir):
        settings = _video_settings_from_info(provider_info.get("settings", {}))
        _export_video_frames(frame_source, frames_dir, settings)
    else:
        LOGGER.info("Reusing previously extracted frames at '%s'", frames_dir)
    return frames_dir


def _directory_contains_frames(path: Path) -> bool:
    return path.exists() and any(path.glob("*.png"))


def _video_settings_from_info(settings: Dict[str, object]) -> Dict[str, object]:
    result = dict(settings)
    sampling_fps = result.get("sampling_fps")
    result["sampling_fps"] = float(sampling_fps) if sampling_fps is not None else None
    frame_stride = result.pop("frame_stride", None)
    if frame_stride is None:
        frame_stride = result.get("stride")
    result["frame_stride"] = int(frame_stride) if frame_stride is not None else None
    result["start_seconds"] = float(result.get("start_seconds", 0.0))
    end_value = result.get("end_seconds")
    result["end_seconds"] = float(end_value) if end_value is not None else None
    result["frame_rate_hint"] = result.get("frame_rate_hint")
    return result


def _export_carla_frames(
    source_dir: Path, frames_dir: Path, settings: Mapping[str, object]
) -> None:
    from pemoin.data.carla import CarlaDataset

    dataset = CarlaDataset(source_dir)
    entries = [dataset.frame(idx) for idx in dataset.frame_indices()]
    if not entries:
        raise SystemExit(f"No CARLA frames found under '{source_dir}'.")

    start_frame = settings.get("start_frame")
    end_frame = settings.get("end_frame")
    if start_frame is not None:
        start_frame = int(start_frame)
        entries = [entry for entry in entries if entry.frame >= start_frame]
    if end_frame is not None:
        end_frame = int(end_frame)
        entries = [entry for entry in entries if entry.frame <= end_frame]
    if not entries:
        raise SystemExit("No CARLA frames found after applying start_frame/end_frame.")

    sampling_fps = settings.get("sampling_fps")
    frame_rate = (
        settings.get("frame_rate") or settings.get("fps") or dataset.frame_rate()
    )
    stride = 1
    if sampling_fps is not None:
        if frame_rate is None or frame_rate <= 0:
            raise SystemExit(
                "sampling_fps requires a valid frame_rate for CARLA exports."
            )
        sampling_fps = float(sampling_fps)
        if sampling_fps <= 0:
            raise SystemExit("sampling_fps must be > 0.")
        stride = max(1, int(round(frame_rate / sampling_fps)))
        entries = entries[::stride]

    frame_stride = settings.get("frame_stride")
    if frame_stride is not None:
        frame_stride = int(frame_stride)
        if frame_stride < 1:
            raise SystemExit("frame_stride must be >= 1.")
        if frame_stride > 1:
            entries = entries[::frame_stride]
    if not entries:
        raise SystemExit("No CARLA frames found after applying sampling/stride.")

    resize_max_side = settings.get("resize_max_side")
    resize_max_side = int(resize_max_side) if resize_max_side is not None else None
    if resize_max_side is not None and resize_max_side <= 0:
        raise SystemExit("resize_max_side must be > 0 when set for CARLA frame export.")

    frames_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Exporting %s CARLA RGB frames to '%s' for MegaSAM (resize_max_side=%s).",
        len(entries),
        frames_dir,
        resize_max_side if resize_max_side is not None else "none",
    )
    exported_indices: List[int] = []
    for sequential_index, entry in enumerate(entries, start=1):
        if not entry.rgb_path.exists():
            raise SystemExit(f"CARLA RGB frame missing: {entry.rgb_path}")
        target = frames_dir / f"{sequential_index:06d}.png"
        if target.exists():
            exported_indices.append(sequential_index)
            continue
        if resize_max_side is None:
            shutil.copy2(entry.rgb_path, target)
        else:
            with Image.open(entry.rgb_path) as image:
                rgb = image.convert("RGB")
                src_w, src_h = rgb.size
                if max(src_h, src_w) == resize_max_side:
                    resized = rgb
                else:
                    scale = float(resize_max_side) / float(max(src_h, src_w))
                    tgt_h = int(max(1, round(src_h * scale)))
                    tgt_w = int(max(1, round(src_w * scale)))
                    resized = rgb.resize((tgt_w, tgt_h), Image.BICUBIC)
                resized.save(target)
        exported_indices.append(sequential_index)
    _write_frame_index_map(frames_dir, exported_indices)


def _write_frame_index_map(frames_dir: Path, frame_indices: List[int]) -> Path:
    path = frames_dir / "frame_index_map.json"
    payload = {"frame_indices": [int(idx) for idx in frame_indices]}
    path.write_text(json.dumps(payload, indent=2))
    return path


def _resolve_frame_index_map(frames_dir: Path) -> Optional[Path]:
    path = frames_dir / "frame_index_map.json"
    if path.exists():
        return path
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return None
    indices: List[int] = []
    for frame in frames:
        stem = frame.stem
        if stem.isdigit():
            indices.append(int(stem))
    if not indices:
        return None
    return _write_frame_index_map(frames_dir, indices)


def _carla_frames_sequential(frames_dir: Path) -> bool:
    map_path = frames_dir / "frame_index_map.json"
    if map_path.exists():
        try:
            raw = json.loads(map_path.read_text())
            frames = raw.get("frame_indices") if isinstance(raw, dict) else raw
            if isinstance(frames, list) and frames and frames[0] == 1:
                if all(isinstance(value, int) for value in frames):
                    return max(frames) == len(frames)
        except Exception:
            return False
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return False
    stems = [int(frame.stem) for frame in frames if frame.stem.isdigit()]
    if not stems:
        return False
    return min(stems) == 1 and max(stems) == len(stems)


def _carla_frames_match_resize(frames_dir: Path, resize_max_side: int) -> bool:
    if resize_max_side <= 0:
        return False
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return False
    first = frames[0]
    try:
        with Image.open(first) as img:
            w, h = img.size
    except Exception:
        return False
    return max(h, w) == int(resize_max_side)


def _export_video_frames(
    video_path: Path, target_dir: Path, settings: Dict[str, object]
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    sampling_fps = settings.get("sampling_fps")
    sampling_fps_value = float(sampling_fps) if sampling_fps is not None else None
    frame_stride = settings.get("frame_stride")
    frame_stride = int(frame_stride) if frame_stride is not None else None
    extractor = VideoFrameProvider(
        sampling_fps=sampling_fps_value,
        frame_stride=frame_stride,
        start_seconds=float(settings.get("start_seconds", 0.0)),
        end_seconds=settings.get("end_seconds"),
        frame_rate_hint=settings.get("frame_rate_hint"),
    )
    extractor.open(video_path)
    LOGGER.info(
        "Extracting frames from %s → %s (sampling_fps=%s, frame_stride=%s, effective_stride=%s, start=%.2fs, end=%s)",
        video_path,
        target_dir,
        sampling_fps_value if sampling_fps_value is not None else "full",
        frame_stride if frame_stride is not None else "auto",
        extractor.frame_stride,
        settings.get("start_seconds"),
        settings.get("end_seconds"),
    )
    count = 0
    try:
        while True:
            frame = extractor.read()
            if frame is None:
                break
            frame_path = target_dir / f"frame_{frame.index:06d}.png"
            iio.imwrite(frame_path, frame.image)
            count += 1
            if count % 100 == 0:
                LOGGER.info("Extracted %d frames...", count)
    finally:
        extractor.close()
    if count == 0:
        raise RuntimeError(
            f"Failed to extract frames from '{video_path}'. Ensure the video contains decodable frames."
        )
    LOGGER.info("Frame extraction complete (%d frames).", count)


def _list_image_frames(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    allowed = {".png", ".jpg", ".jpeg", ".bmp"}
    frames = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(frames, key=lambda path: path.name)


def _prepare_panst3r_frames(
    frame_dir: Path, provider_root: Path, target_size: int = 512
) -> Path:
    """
    Prepare frames for PanSt3R by square-padding and resizing to the target size.
    """
    processed_dir = provider_root / f"panst3r_frames_{target_size}"
    manifest_path = processed_dir / ".manifest.json"
    source_frames = _list_image_frames(frame_dir)
    if not source_frames:
        raise SystemExit(
            f"No frames found under '{frame_dir}' for PanSt3R preprocessing."
        )

    source_resolved = str(frame_dir.resolve())
    source_mtime = max((path.stat().st_mtime for path in source_frames), default=0.0)
    manifest = {
        "source": source_resolved,
        "count": len(source_frames),
        "max_mtime": source_mtime,
        "target_size": target_size,
    }

    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            cached = None
        if cached:
            same_source = cached.get("source") == manifest["source"]
            same_count = cached.get("count") == manifest["count"]
            same_target = cached.get("target_size") == manifest["target_size"]
            similar_mtime = (
                abs(float(cached.get("max_mtime", -1.0)) - source_mtime) < 1e-3
            )
            if (
                same_source
                and same_count
                and same_target
                and similar_mtime
                and processed_dir.exists()
            ):
                LOGGER.info(
                    "Reusing PanSt3R-ready frames from %s (already square-padded to %dx%d).",
                    processed_dir,
                    target_size,
                    target_size,
                )
                return processed_dir

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Preparing %d frame(s) for PanSt3R under %s (square pad → %dx%d).",
        len(source_frames),
        processed_dir,
        target_size,
        target_size,
    )

    for idx, source_path in enumerate(source_frames, start=1):
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            target_dim = max(width, height)
            if width != height:
                canvas = Image.new("RGB", (target_dim, target_dim))
                offset = ((target_dim - width) // 2, (target_dim - height) // 2)
                canvas.paste(rgb, offset)
            else:
                canvas = rgb
            if canvas.size != (target_size, target_size):
                canvas = canvas.resize((target_size, target_size), Image.BICUBIC)
            output_path = processed_dir / f"{source_path.stem}.png"
            canvas.save(output_path)
        if idx % 100 == 0 or idx == len(source_frames):
            LOGGER.info("PanSt3R frame prep progress: %d/%d", idx, len(source_frames))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return processed_dir


def _apply_megasam_bundle(
    profile: ProfileConfig,
    bundle_path: Path,
    scene_name: str,
    frame_index_map: Optional[Path] = None,
    gt_intrinsics_path: Optional[Path] = None,
    tracking_preprocess_path: Optional[Path] = None,
) -> None:
    for binding in profile.providers.values():
        if not binding.tool.startswith("MegaSAM"):
            continue
        adapter_settings = dict(binding.settings.get("adapter", {}))
        adapter_settings["bundle_path"] = str(bundle_path)
        adapter_settings.setdefault("scene_name", scene_name)
        if frame_index_map is not None:
            adapter_settings["frame_index_map_path"] = str(frame_index_map)
        if gt_intrinsics_path is not None:
            adapter_settings["gt_intrinsics_path"] = str(gt_intrinsics_path)
            adapter_settings["enforce_gt_intrinsics"] = True
        if tracking_preprocess_path is not None:
            adapter_settings["tracking_preprocess_path"] = str(tracking_preprocess_path)
        adapter_settings.setdefault("require_final_bundle", True)
        binding.settings["adapter"] = adapter_settings


def _prepare_panst3r_bundle(
    args: argparse.Namespace,
    frame_source: Path,
    provider_info: Dict[str, Dict[str, object]],
    run_dir: Path,
    conda_env: Optional[str],
    settings_path: Optional[Path],
) -> Tuple[Path, str]:
    repo_root = args.panst3r_repo.resolve()
    scene_name = args.panst3r_scene or (
        frame_source.stem if frame_source.is_file() else frame_source.name
    )
    provider_root = run_dir / "raw" / "panst3r"
    provider_root.mkdir(parents=True, exist_ok=True)
    raw_frame_dir = _resolve_automation_frame_directory(
        frame_source, provider_info, provider_root
    )
    frame_dir = _prepare_panst3r_frames(raw_frame_dir, provider_root, target_size=512)
    LOGGER.info("PanSt3R will read normalized frames from %s", frame_dir)
    if args.panst3r_output:
        bundle_path = Path(args.panst3r_output)
        if not bundle_path.is_absolute():
            bundle_path = (provider_root / bundle_path).resolve()
        else:
            bundle_path = bundle_path.resolve()
    else:
        bundle_path = (provider_root / f"{scene_name}_panst3r_bundle.npz").resolve()

    LOGGER.info("PanSt3R bundle target: %s", bundle_path)
    needs_automation = args.panst3r_auto and not bundle_path.exists()
    if needs_automation:
        commands = _resolve_panst3r_commands(args, repo_root, settings_path)
        if not commands:
            raise SystemExit(
                "PanSt3R automation requested but no commands were provided. "
                "Use --panst3r-command or --panst3r-preset to describe the pipeline stages."
            )
        LOGGER.info("PanSt3R automation enabled. Generating bundle from %s", frame_dir)
        LOGGER.info(
            "Invoking PanSt3R tools inside conda env '%s'",
            conda_env or "default shell",
        )
        if settings_path:
            LOGGER.info("Passing PanSt3R settings file: %s", settings_path)
        config = PanSt3RJobConfig(
            frame_dir=frame_dir,
            scene_name=scene_name,
            repo_root=repo_root,
            output_dir=bundle_path.parent,
            bundle_name=bundle_path.name,
            commands=commands,
            conda_env=conda_env,
        )
        runner = PanSt3RJobRunner(logger=LOGGER.info)
        bundle_path = runner.run(config)
    else:
        if args.panst3r_auto:
            LOGGER.info("Reusing existing PanSt3R bundle: %s", bundle_path)
        else:
            LOGGER.warning(
                "PanSt3R automation disabled; expecting bundle at %s", bundle_path
            )

    if not bundle_path.exists():
        raise SystemExit(
            f"PanSt3R bundle not found at '{bundle_path}'. "
            "Run with --panst3r-auto (default) or provide --panst3r-output."
        )
    return bundle_path, scene_name


def _apply_panst3r_bundle(
    profile: ProfileConfig, bundle_path: Path, scene_name: str
) -> None:
    for binding in profile.providers.values():
        if not binding.tool.startswith("PanSt3R"):
            continue
        adapter_settings = dict(binding.settings.get("adapter", {}))
        adapter_settings["bundle_path"] = str(bundle_path)
        adapter_settings.setdefault("scene_name", scene_name)
        binding.settings["adapter"] = adapter_settings


def _resolve_run_directory(
    root: Path,
    profile_name: str,
    frame_source: Path,
    run_timestamp: str,
) -> Path:
    run_key = _pipeline_folder_name(profile_name, frame_source, run_timestamp)
    run_dir = root / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run directory initialised at %s", run_dir)
    return run_dir


def _build_frame_provider(
    profile: ProfileConfig,
    args: argparse.Namespace,
    config_path: Path,
) -> Tuple[FrameProvider, Path, Dict[str, Dict[str, object]]]:
    if args.frames:
        override = Path(args.frames).resolve()
        if override.is_dir():
            if (
                profile.frame_provider is not None
                and profile.frame_provider.tool == "UnityFrameProvider"
            ):
                load_images = bool(
                    profile.frame_provider.settings.get("load_images", True)
                )
                provider = UnityFrameProvider(
                    settings=profile.frame_provider.settings,
                    load_images=load_images,
                )
                provider.open(override)
                LOGGER.info("Using Unity frame override directory: %s", override)
                settings = {
                    "path": str(override),
                    "load_images": load_images,
                }
                settings.update(dict(provider.runtime_settings()))
                return (
                    provider,
                    override,
                    {
                        "tool": "UnityFrameProvider",
                        "settings": settings,
                    },
                )
            provider = DirectoryFrameProvider(frame_rate=args.frame_rate)
            provider.open(override)
            LOGGER.info("Using frame override directory: %s", override)
            settings = {"path": str(override)}
            settings.update(dict(provider.runtime_settings()))
            return (
                provider,
                override,
                {"tool": "DirectoryFrameProvider", "settings": settings},
            )
        else:
            provider = VideoFrameProvider(
                frame_rate_hint=args.frame_rate,
                sampling_fps=None,
                start_seconds=0.0,
                end_seconds=None,
            )
            provider.open(override)
            LOGGER.info("Using video override: %s", override)
            settings = {"path": str(override)}
            settings.update(dict(provider.runtime_settings()))
            return (
                provider,
                override,
                {"tool": "VideoFrameProvider", "settings": settings},
            )

    if profile.frame_provider is None:
        raise SystemExit(
            "Profile does not define a frame provider and no --frames override was supplied."
        )
    provider, source, info = create_frame_provider_from_binding(
        profile.frame_provider,
        config_base=config_path,
        frame_rate_override=args.frame_rate,
    )
    LOGGER.info(
        "Using profile frame provider '%s' with source %s",
        info["tool"],
        source,
    )
    return provider, source, info


def _sync_vkitti2_selection(profile: ProfileConfig, config_base: Path) -> ProfileConfig:
    if (
        profile.frame_provider is None
        or profile.frame_provider.tool != "VirtualKitty2FrameProvider"
    ):
        return profile
    from dataclasses import replace

    from pemoin.data.virtual_kitty_2 import resolve_vkitti2_selection

    frame_settings = dict(profile.frame_provider.settings)
    path_raw = frame_settings.get("path") or frame_settings.get("root")
    if not path_raw:
        return profile
    path = Path(str(path_raw))
    if not path.is_absolute():
        path = (config_base / path).resolve()
    frame_settings["path"] = str(path)
    selection = resolve_vkitti2_selection(path, frame_settings)
    updated_providers = dict(profile.providers)
    for name, binding in updated_providers.items():
        if not binding.tool.startswith("VirtualKitty2"):
            continue
        updated_settings = dict(binding.settings)
        updated_settings["path"] = str(path)
        updated_settings["scene"] = selection.scene
        updated_settings["variation"] = selection.variation
        updated_settings["camera"] = selection.camera
        updated_providers[name] = replace(binding, settings=updated_settings)
    return replace(profile, providers=updated_providers)


def run_pipeline(args: argparse.Namespace) -> None:
    logging_config = _configure_console_logging_from_args(args)
    registry = _build_registry(args.config)
    profile = registry.get(args.profile)
    LOGGER.info(_profile_banner(profile.name))
    megasam_conda_env = args.megasam_conda_env or profile.megasam.get("conda_env")
    panst3r_conda_env = args.panst3r_conda_env or profile.panst3r.get("conda_env")
    profile_panst3r_settings = profile.panst3r.get("settings_file")

    uses_megasam = any(
        binding.tool.startswith("MegaSAM") for binding in profile.providers.values()
    )
    uses_panst3r = any(
        binding.tool.startswith("PanSt3R") for binding in profile.providers.values()
    )

    if uses_megasam and megasam_conda_env:
        LOGGER.info(
            "MegaSAM commands will run inside env '%s' (conda/mamba)", megasam_conda_env
        )
    if uses_panst3r and panst3r_conda_env:
        LOGGER.info(
            "PanSt3R commands will run inside env '%s' (conda/mamba)", panst3r_conda_env
        )
    if uses_panst3r:
        if args.panst3r_settings:
            LOGGER.info(
                "PanSt3R automation will read settings from %s", args.panst3r_settings
            )
        elif isinstance(profile_panst3r_settings, str) and profile_panst3r_settings:
            LOGGER.info(
                "PanSt3R automation will use profile settings file %s",
                profile_panst3r_settings,
            )

    config_file = args.config.resolve()
    repo_root = config_file.parent.parent
    output_root = (
        Path(args.output_root)
        if Path(args.output_root).is_absolute()
        else repo_root / args.output_root
    ).resolve()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir: Optional[Path] = None
    if profile.unity_import.get("enabled"):
        from pemoin.scripts.unity_import import (
            import_unity_dataset,
            parse_import_selection,
        )

        source_raw = profile.unity_import.get("source") or profile.unity_import.get(
            "path"
        )
        if not source_raw:
            raise SystemExit(
                "unity_import.enabled is true but no unity_import.source was provided."
            )
        source_path = Path(source_raw).expanduser().resolve()
        if run_dir is None:
            run_dir = _resolve_run_directory(
                output_root, profile.name, source_path, run_timestamp
            )
        dest_root_raw = profile.unity_import.get("dest_root")
        dest_root = None
        if isinstance(dest_root_raw, (str, Path)) and str(dest_root_raw).strip():
            dest_root = Path(dest_root_raw)
        if dest_root is None or str(dest_root) == "unity_data":
            dest_root = run_dir / "raw" / "unity_import"
        elif not dest_root.is_absolute():
            dest_root = (repo_root / dest_root).resolve()
        name = profile.unity_import.get("name")
        sampling_fps_raw = profile.unity_import.get("sampling_fps")
        sampling_fps = float(sampling_fps_raw) if sampling_fps_raw is not None else None
        stride_raw = profile.unity_import.get(
            "stride", profile.unity_import.get("frame_stride")
        )
        stride = int(stride_raw) if stride_raw is not None else None
        if profile.unity_import.get("resize") is not None:
            raise SystemExit(
                "unity_import.resize has been removed; use profile.working_resolution instead."
            )
        resize_max_side = None
        if profile.working_resolution is not None:
            resize_max_side = int(max(profile.working_resolution))
        prune = bool(profile.unity_import.get("prune_unused", True))
        write_videos = bool(profile.unity_import.get("write_videos", True))
        resources_raw = profile.unity_import.get("resources")
        selection = parse_import_selection(
            resources_raw if isinstance(resources_raw, Mapping) else None
        )
        result = import_unity_dataset(
            source_path,
            dest_root,
            name=name,
            stride=stride,
            sampling_fps=sampling_fps,
            resize_max_side=resize_max_side,
            prune=prune,
            write_videos=write_videos,
            selection=selection,
        )
        imported_root = result.dest_dir
        sequence_dirs = sorted(imported_root.glob("sequence.*"))
        if sequence_dirs:
            imported_frames = sequence_dirs[0]
        else:
            imported_frames = imported_root
        args.frames = imported_frames
        from dataclasses import replace

        updated_frame_provider = profile.frame_provider
        if profile.frame_provider is not None:
            frame_provider = profile.frame_provider
            updated_settings = dict(frame_provider.settings)
            updated_settings["sampling_fps"] = float(result.output_fps)
            updated_settings["path"] = str(imported_frames)
            updated_frame_provider = replace(frame_provider, settings=updated_settings)
        else:
            from pemoin.runtime.profiles.config import ModuleBinding

            updated_frame_provider = ModuleBinding(
                tool="UnityFrameProvider",
                settings={
                    "path": str(imported_frames),
                    "sampling_fps": float(result.output_fps),
                },
            )

        updated_providers = dict(profile.providers)
        for name, binding in updated_providers.items():
            if binding.tool.startswith("UnityGT"):
                updated_settings = dict(binding.settings)
                updated_settings["path"] = str(imported_frames)
                updated_providers[name] = replace(binding, settings=updated_settings)

        profile = replace(
            profile,
            frame_provider=updated_frame_provider,
            providers=updated_providers,
        )
    profile = _sync_vkitti2_selection(profile, config_file.parent)
    if args.panst3r_settings:
        panst3r_settings_path: Optional[Path] = (
            args.panst3r_settings.expanduser().resolve()
        )
    elif isinstance(profile_panst3r_settings, str) and profile_panst3r_settings:
        profile_path = Path(profile_panst3r_settings)
        panst3r_settings_path = (
            profile_path if profile_path.is_absolute() else (repo_root / profile_path)
        ).resolve()
    else:
        panst3r_settings_path = None
    frame_provider, frame_source, provider_info = _build_frame_provider(
        profile, args, repo_root
    )
    if run_dir is None:
        run_dir = _resolve_run_directory(
            output_root, profile.name, frame_source, run_timestamp
        )
    if uses_megasam:
        ensure_megasam_log_handler(run_dir / "standard" / "logs")
    _summary(
        "Step 1/4 Profiles: %s (%d providers)",
        args.profile,
        len(profile.providers),
    )
    _summary(
        "Step 2/4 Frames: %s via %s -> %s",
        frame_source,
        provider_info["tool"],
        run_dir,
    )

    megasam_summary = "skipped"
    if uses_megasam:
        (
            bundle_path,
            bundle_scene,
            frame_index_map,
            gt_intrinsics_npz,
            tracking_preprocess_path,
        ) = _prepare_megasam_bundle(
            args,
            profile,
            frame_source,
            provider_info,
            run_dir,
            megasam_conda_env,
            repo_root,
        )
        _apply_megasam_bundle(
            profile,
            bundle_path,
            bundle_scene,
            frame_index_map=frame_index_map,
            gt_intrinsics_path=gt_intrinsics_npz,
            tracking_preprocess_path=tracking_preprocess_path,
        )
        megasam_summary = f"bundle={bundle_path.name}"
    else:
        LOGGER.info("Skipping MegaSAM automation (no MegaSAM providers requested).")

    panst3r_summary = "skipped"
    if uses_panst3r:
        pan_bundle_path, pan_scene = _prepare_panst3r_bundle(
            args,
            frame_source,
            provider_info,
            run_dir,
            panst3r_conda_env,
            panst3r_settings_path,
        )
        _apply_panst3r_bundle(profile, pan_bundle_path, pan_scene)
        panst3r_summary = f"bundle={pan_bundle_path.name}"
    else:
        LOGGER.info("Skipping PanSt3R automation (no PanSt3R providers requested).")

    _summary(
        "Step 3/4 Automation: MegaSAM=%s, PanSt3R=%s",
        megasam_summary,
        panst3r_summary,
    )

    snapshot = _profile_snapshot(
        profile,
        config_path=config_file,
        frame_source=frame_source,
        frame_provider_info=provider_info,
        run_timestamp=run_timestamp,
    )
    save_profile_snapshot(run_dir=run_dir, snapshot=snapshot)
    launch = create_runtime_launch(
        profile=profile,
        run_dir=run_dir,
        frame_source=frame_source,
        frame_provider_info=provider_info,
        run_timestamp=run_timestamp,
        profiles_config_path=config_file,
    )
    launch.provider_context["logging"] = logging_config.to_mapping()

    if args.video_fps is not None:
        raise SystemExit(
            "--video-fps has been removed. Video FPS is resolved from the active frame provider settings."
        )

    # Apply video export overrides from CLI
    if args.generate_videos is not None:
        video_override = {"enabled": args.generate_videos}
        launch.provider_context["video_export_override"] = video_override

    result = launch.runtime.run(
        frame_provider,
        provider_factory=launch.provider_factory,
        context=launch.provider_context,
        max_frames=args.max_frames,
        on_frame=None,
    )

    processed = result.processed_frames
    _summary("Step 4/4 Runtime: processed %d frame(s)", processed)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    env_settings = _load_env_settings()
    default_profile = env_settings.get("PEMOIN_ACTIVE_PROFILE", "unity_gt_offline")
    default_config = Path(
        env_settings.get("PEMOIN_PROFILES_CONFIG", "config/profiles.json")
    )

    parser = argparse.ArgumentParser(
        description="Run the PEMOIN pipeline on a directory of frames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Profile JSON path (default: {default_config}).",
    )
    parser.add_argument(
        "--profile",
        default=default_profile,
        help=f"Profile name defined in the config file (default: {default_profile}).",
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity to warnings/errors.",
    )
    verbosity_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed PEMOIN logs while keeping PEMOIN-owned progress bars enabled.",
    )
    parser.add_argument(
        "--frames", type=Path, default=None, help="Override frame/video source path."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames to process.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Optional override for frame rate (used for timestamps or video FPS hint).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Directory where per-run artifacts are stored (default: outputs/<timestamp>_<source>/...).",
    )
    parser.add_argument(
        "--megasam-auto",
        action="store_true",
        default=True,
        help="Automatically run MegaSAM preprocessing (default: enabled).",
    )
    parser.add_argument(
        "--no-megasam-auto",
        action="store_false",
        dest="megasam_auto",
        help="Disable automatic MegaSAM preprocessing.",
    )
    parser.add_argument(
        "--megasam-command",
        dest="megasam_commands",
        action="append",
        default=None,
        help="Command template executed when --megasam-auto is set. "
        "Placeholders: {frames}, {scene}, {repo}, {output}, {bundle}, {intrinsics}.",
    )
    parser.add_argument(
        "--megasam-preset",
        choices=["standard"],
        default="standard",
        help="Built-in MegaSAM automation preset (default: standard).",
    )
    parser.add_argument(
        "--megasam-output",
        type=Path,
        default=None,
        help="Path to an existing MegaSAM bundle (NPZ). "
        "If omitted, defaults to <repo>/outputs_cvd/<scene>_sgd_cvd_hr.npz.",
    )
    parser.add_argument(
        "--megasam-repo",
        type=Path,
        default=Path("tools/mega-sam"),
        help="MegaSAM repository root used for automation.",
    )
    parser.add_argument(
        "--megasam-scene",
        type=str,
        default=None,
        help="Scene name used for MegaSAM diagnostics and generated bundles.",
    )
    parser.add_argument(
        "--megasam-checkpoint",
        type=Path,
        default=None,
        help="Override the MegaSAM checkpoint path used by presets.",
    )
    parser.add_argument(
        "--megasam-depth-checkpoint",
        type=Path,
        default=None,
        help="Override the Depth-Anything checkpoint path used by presets.",
    )
    parser.add_argument(
        "--megasam-raft-checkpoint",
        type=Path,
        default=None,
        help="Override the RAFT checkpoint path used by presets.",
    )
    parser.add_argument(
        "--megasam-cuda-devices",
        default="0",
        help="Value assigned to CUDA_VISIBLE_DEVICES for MegaSAM presets (default: 0).",
    )
    parser.add_argument(
        "--megasam-conda-env",
        default=None,
        help="Name of the conda/mamba environment used for MegaSAM commands (default: profile-specified).",
    )
    parser.add_argument(
        "--megasam-pointcloud-debug",
        action="store_true",
        default=False,
        help="Launch MegaSAM pointcloud viewer after exporting PLY debug artifact.",
    )
    parser.add_argument(
        "--panst3r-auto",
        action="store_true",
        default=True,
        help="Automatically run PanSt3R preprocessing (default: enabled).",
    )
    parser.add_argument(
        "--no-panst3r-auto",
        action="store_false",
        dest="panst3r_auto",
        help="Disable automatic PanSt3R preprocessing.",
    )
    parser.add_argument(
        "--panst3r-command",
        dest="panst3r_commands",
        action="append",
        help="Command template executed when --panst3r-auto is set. "
        "Use placeholders {frames}, {scene}, {bundle}, {output}.",
    )
    parser.add_argument(
        "--panst3r-preset",
        default="default",
        help="Named preset for PanSt3R automation (default: 'default').",
    )
    parser.add_argument(
        "--panst3r-output",
        type=Path,
        default=None,
        help="Override the PanSt3R bundle path (default: <run_dir>/<scene>_panst3r_bundle.npz).",
    )
    parser.add_argument(
        "--panst3r-repo",
        type=Path,
        default=Path("tools/panst3r"),
        help="PanSt3R repository root used for automation.",
    )
    parser.add_argument(
        "--panst3r-scene",
        type=str,
        default=None,
        help="Scene name used for PanSt3R diagnostics and generated bundles.",
    )
    parser.add_argument(
        "--panst3r-conda-env",
        default=None,
        help="Name of the conda/mamba environment used for PanSt3R commands (default: profile-specified).",
    )
    parser.add_argument(
        "--panst3r-settings",
        type=Path,
        default=None,
        help="Path to a PanSt3R settings JSON file passed to the automation script.",
    )
    parser.add_argument(
        "--generate-videos",
        action="store_true",
        default=None,
        dest="generate_videos",
        help="Generate video files from per-frame visualizations.",
    )
    parser.add_argument(
        "--no-generate-videos",
        action="store_false",
        dest="generate_videos",
        help="Skip video generation from visualizations.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Deprecated and unsupported. Video FPS is resolved from the active frame provider settings.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        args = parse_args(argv)
        run_pipeline(args)
    except Exception as e:
        LOGGER.exception("Pipeline execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
