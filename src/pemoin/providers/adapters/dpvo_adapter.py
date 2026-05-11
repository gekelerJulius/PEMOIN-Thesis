"""
Adapter integrating DPVO (Deep Patch Visual Odometry) as a trajectory provider.

DPVO runs inside a separate mamba environment and communicates via subprocess +
NPZ files, following the established DA3 adapter pattern.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from pemoin.data.contracts import (
    DynamicMaskData,
    PoseData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    TrajectoryMatchGraphData,
)
from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender
from pemoin.providers.base import ProviderExecutionMode
from pemoin.providers.factory import ProviderFactory
from pemoin.providers.trajectory import TrajectoryProvider
from pemoin.runtime.cache import CrossRunCacheManager
from pemoin.runtime.profiles.config import ModuleBinding
from pemoin.utils.env_launcher import resolve_env_launcher as _resolve_env_launcher
from pemoin.utils.logging import get_logger

LOG = get_logger()

_SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_PARENTS = Path(__file__).resolve().parents
_REPO_ROOT_FALLBACK = _PARENTS[5] if len(_PARENTS) > 5 else _PARENTS[-1]


def _resolve_repo_root(candidate: Optional[Path] = None) -> Path:
    bases: List[Path] = []
    if candidate is not None:
        bases.append(candidate)
    env_root = os.environ.get("PEMOIN_REPO_ROOT")
    if env_root:
        bases.append(Path(env_root))
    bases.append(Path.cwd())
    bases.append(_REPO_ROOT_FALLBACK)
    for base in bases:
        base = base.expanduser().resolve()
        dpvo_dir = base / "tools" / "DPVO"
        if dpvo_dir.exists():
            return base
    return bases[-1].expanduser().resolve()


def _discover_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for ext in _SUPPORTED_EXTENSIONS:
        images.extend(root.glob(f"*{ext}"))
        images.extend(root.glob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in images})


def _resolve_image_dir(frame_source: Path) -> Path:
    source = frame_source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Frame source path does not exist: '{source}'.")
    if not source.is_dir():
        raise ValueError(
            f"DPVO adapter expects a frame directory, got '{source}'."
        )

    if _discover_images(source):
        return source

    rgb_dir = source / "rgb"
    if rgb_dir.is_dir() and _discover_images(rgb_dir):
        return rgb_dir

    raise FileNotFoundError(
        f"No supported images found under '{source}' or '{rgb_dir}'."
    )


@dataclass(frozen=True, slots=True)
class DPVOSettings:
    """Settings controlling how DPVO is invoked."""

    mamba_env: str = "dpvo"
    network_path: Path = Path("tools/DPVO/dpvo.pth")
    config_path: Path = Path("tools/DPVO/config/default.yaml")
    stride: int = 1
    skip: int = 0
    reuse_exports: bool = True
    device: str = "cuda"
    precision_mode: str = "amp_fp16"
    memory_preset: str = "balanced"
    allocator_mode: str = "native"
    allocator_max_split_size_mb: Optional[int] = None
    cfg_opts: tuple[str, ...] = ()
    env_manager: Optional[str] = None
    mask_dir: Optional[Path] = None
    memory_diag_sample_every: int = 1
    memory_diag_warn_ratio: float = 0.92
    memory_guard_enabled: bool = True
    memory_guard_warmup_frames: int = 24
    memory_guard_abort_reserved_vram_ratio: float = 0.70
    memory_guard_abort_reserved_to_allocated_ratio: float = 12.0
    repo_root: Path = _REPO_ROOT_FALLBACK


class DPVORunner:
    """Executes the DPVO bridge script inside the specified mamba environment."""

    def __init__(self, repo_root: Optional[Path] = None):
        self._repo_root = _resolve_repo_root(repo_root)

    @staticmethod
    def _allocator_env_value(settings: DPVOSettings) -> Optional[str]:
        mode = settings.allocator_mode.strip().lower()
        parts: list[str] = []
        if mode == "native":
            return None
        if mode == "expandable_segments":
            parts.append("expandable_segments:True")
        elif mode == "cuda_malloc_async":
            parts.append("backend:cudaMallocAsync")
        else:
            raise ValueError(
                "DPVO allocator_mode must be one of: native, expandable_segments, cuda_malloc_async."
            )
        if settings.allocator_max_split_size_mb is not None:
            parts.append(f"max_split_size_mb:{int(settings.allocator_max_split_size_mb)}")
        return ",".join(parts)

    def run(
        self,
        settings: DPVOSettings,
        image_dir: Path,
        calib_file: Path,
        output_dir: Path,
    ) -> None:
        bridge_script = self._repo_root / "tools" / "DPVO" / "pemoin_bridge.py"
        if not bridge_script.exists():
            raise FileNotFoundError(
                f"DPVO bridge script not found at '{bridge_script}'. "
                "Ensure tools/DPVO/pemoin_bridge.py exists."
            )

        network_path = Path(settings.network_path)
        if not network_path.is_absolute():
            network_path = (self._repo_root / network_path).resolve()

        config_path = Path(settings.config_path)
        if not config_path.is_absolute():
            config_path = (self._repo_root / config_path).resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        launcher = _resolve_env_launcher(settings.mamba_env, settings.env_manager)
        cmd = [
            *launcher,
            "python", str(bridge_script),
            "--imagedir", str(image_dir),
            "--calib", str(calib_file),
            "--output-dir", str(output_dir),
            "--network", str(network_path),
            "--config", str(config_path),
            "--stride", str(settings.stride),
            "--skip", str(settings.skip),
            "--device", settings.device,
            "--precision-mode", settings.precision_mode,
        ]
        if settings.cfg_opts:
            cmd.extend(["--opts", *settings.cfg_opts])
        if settings.mask_dir is not None:
            cmd.extend(["--mask-dir", str(settings.mask_dir)])
        cmd.extend(
            [
                "--memory-diag-sample-every",
                str(int(settings.memory_diag_sample_every)),
                "--memory-diag-warn-ratio",
                str(float(settings.memory_diag_warn_ratio)),
                "--memory-guard-enabled",
                "1" if settings.memory_guard_enabled else "0",
                "--memory-guard-warmup-frames",
                str(int(settings.memory_guard_warmup_frames)),
                "--memory-guard-abort-reserved-vram-ratio",
                str(float(settings.memory_guard_abort_reserved_vram_ratio)),
                "--memory-guard-abort-reserved-to-allocated-ratio",
                str(float(settings.memory_guard_abort_reserved_to_allocated_ratio)),
            ]
        )

        env = os.environ.copy()
        dpvo_dir = self._repo_root / "tools" / "DPVO"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(dpvo_dir), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        if launcher[0] in {"micromamba", "mamba"} and not env.get("XDG_CACHE_HOME"):
            cache_root = output_dir / ".mamba_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            env["XDG_CACHE_HOME"] = str(cache_root)
        allocator_conf = self._allocator_env_value(settings)
        if allocator_conf is None:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            env["PYTORCH_CUDA_ALLOC_CONF"] = allocator_conf

        LOG.info(
            "Running DPVO inference: manager=%s, env=%s, images=%s allocator_mode=%s",
            launcher[0], settings.mamba_env, image_dir, settings.allocator_mode,
            extra={"summary": True},
        )
        subprocess.run(cmd, check=True, cwd=dpvo_dir, env=env)


class DPVOClient:
    """Loads DPVO results and converts to PEMOIN data contracts."""

    def __init__(self, settings: DPVOSettings, output_dir: Path):
        self.settings = settings
        self._output_dir = output_dir
        self._poses_c2w: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None
        self._index_by_timestamp: dict[int, int] = {}

    def load(self) -> None:
        results_path = self._output_dir / "dpvo_results.npz"
        if not results_path.exists():
            raise FileNotFoundError(
                f"DPVO results not found at '{results_path}'. "
                "Run DPVO inference first."
            )
        with np.load(results_path, allow_pickle=True) as data:
            self._poses_c2w = np.asarray(data["poses_c2w"], dtype=np.float64)
            self._timestamps = np.asarray(data["timestamps"], dtype=np.int64)
        self._index_by_timestamp = {}
        for idx, ts in enumerate(self._timestamps.tolist()):
            key = int(ts)
            if key not in self._index_by_timestamp:
                self._index_by_timestamp[key] = idx
        LOG.info(
            "Loaded DPVO results: %d poses from '%s'.",
            self._poses_c2w.shape[0], results_path,
            extra={"summary": True},
        )

    @property
    def num_poses(self) -> int:
        if self._poses_c2w is None:
            return 0
        return self._poses_c2w.shape[0]

    def get_pose(self, frame_index: int) -> PoseData:
        if self._poses_c2w is None:
            raise RuntimeError("DPVO results not loaded. Call load() first.")

        if frame_index in self._index_by_timestamp:
            idx = self._index_by_timestamp[frame_index]
        else:
            idx = min(max(frame_index, 0), self._poses_c2w.shape[0] - 1)
        c2w_opencv = self._poses_c2w[idx].astype(np.float64)
        w2c_opencv = np.linalg.inv(c2w_opencv).astype(np.float64)

        c2w_blender, w2c_blender = convert_pose_opencv_to_blender(c2w_opencv, w2c_opencv)

        metadata: MutableMapping[str, Any] = {
            "source": "DPVO",
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
            "pose_coordinate_system": "blender",
            "metric_scale": False,
            "frame_index": frame_index,
        }
        if frame_index != idx:
            metadata["index_clamped_from"] = frame_index

        sample = PoseSample(
            frame_index=frame_index,
            camera_to_world=c2w_blender.astype(np.float32),
            world_to_camera=w2c_blender.astype(np.float32),
            confidence=None,
            metadata=metadata,
        )
        return PoseData(
            samples=[sample],
            metadata={
                "source": "DPVO",
                "camera_convention": "blender",
                "source_camera_convention": "opencv",
                "metric_scale": False,
            },
        )

    def to_pose_data(self) -> PoseData:
        if self._poses_c2w is None:
            raise RuntimeError("DPVO results not loaded. Call load() first.")
        timestamps = (
            self._timestamps.tolist()
            if self._timestamps is not None and self._timestamps.size
            else list(range(self._poses_c2w.shape[0]))
        )
        samples: list[PoseSample] = []
        for idx, ts in enumerate(timestamps):
            sample = self.get_pose(int(ts)).samples[0]
            if sample.frame_index != int(ts):
                sample.frame_index = int(ts)
            samples.append(sample)
        return PoseData(
            samples=samples,
            metadata={
                "source": "DPVO",
                "camera_convention": "blender",
                "source_camera_convention": "opencv",
                "metric_scale": False,
            },
        )


def _generate_dynamic_masks_from_semantics(
    resource_store: ResourceStore,
    dynamic_classes: tuple[str, ...],
) -> Optional[Path]:
    """Read semantics_2d NPZs, generate binary masks, save to dynamic_mask dir.

    Returns the mask directory path, or None if no semantics available.
    """
    sem_indices = resource_store.frame_indices(ResourceKind.SEMANTICS_2D)
    if not sem_indices:
        return None

    dynamic_lower = {c.lower() for c in dynamic_classes}
    if not dynamic_lower:
        return None

    generated = 0
    for frame_idx in sem_indices:
        try:
            sem = resource_store.load_semantics2d(frame_idx)
        except Exception:
            LOG.debug("Could not load semantics for frame %d, skipping mask.", frame_idx)
            continue

        # Build a boolean mask: True=static, False=dynamic
        shape = None
        dynamic_pixel_mask = None
        for seg in sem.segments:
            if shape is None:
                shape = seg.mask.shape
                dynamic_pixel_mask = np.zeros(shape, dtype=bool)
            if seg.label.lower() in dynamic_lower:
                dynamic_pixel_mask |= seg.mask

        if shape is None:
            continue

        static_mask = ~dynamic_pixel_mask
        mask_data = DynamicMaskData(
            frame_index=frame_idx,
            mask=static_mask,
            dynamic_classes=dynamic_classes,
            metadata={"source": "semantics_2d"},
        )
        resource_store.save_dynamic_mask(mask_data)
        generated += 1

    if generated == 0:
        return None

    mask_dir = resource_store.base_dir(ResourceKind.DYNAMIC_MASK)
    LOG.info(
        "Generated %d dynamic masks from semantics_2d in '%s'.",
        generated, mask_dir,
        extra={"summary": True},
    )
    return mask_dir


class DPVOAdapter:
    """Bridges DPVO into PEMOIN providers."""

    def __init__(
        self,
        settings: DPVOSettings,
        image_dir: Path,
        output_dir: Path,
        intrinsics_path: Optional[Path] = None,
        expected_frame_count: Optional[int] = None,
        resource_store: Optional[ResourceStore] = None,
        mobile_labels: tuple[str, ...] = (),
        cache_manager: Optional[CrossRunCacheManager] = None,
        profile_name: Optional[str] = None,
    ):
        self._settings = settings
        self._image_dir = image_dir
        self._output_dir = output_dir
        self._intrinsics_path = intrinsics_path
        self._expected_frame_count = (
            int(expected_frame_count)
            if expected_frame_count is not None
            else None
        )
        self._resource_store = resource_store
        self._mobile_labels = mobile_labels
        self._cache_manager = cache_manager
        self._profile_name = profile_name
        self._client: Optional[DPVOClient] = None
        self._runner = DPVORunner(settings.repo_root)
        self._cache_status: dict[str, Any] = {
            "cross_run_cache_enabled": bool(cache_manager and cache_manager.enabled),
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled" if cache_manager is None or not cache_manager.enabled else "not-checked",
            "dpvo_memory_diagnostics": "not-loaded",
        }
        self._cache_signature: Optional[str] = None
        self._cache_payload: Optional[dict[str, Any]] = None
        self._memory_diagnostics: Optional[dict[str, Any]] = None

    def _run_root(self) -> Path:
        return self._output_dir.parents[1]

    def _settings_signature_payload(self) -> dict[str, Any]:
        payload = asdict(self._settings)
        for key in ("reuse_exports", "repo_root", "mask_dir"):
            payload.pop(key, None)
        return payload

    def _cross_run_payload(self) -> Optional[dict[str, Any]]:
        if self._cache_manager is None or not self._cache_manager.enabled:
            return None
        if self._intrinsics_path is None or not self._intrinsics_path.exists():
            return None
        payload: dict[str, Any] = {
            "settings": self._settings_signature_payload(),
            "image_dir": self._cache_manager.directory_signature(self._image_dir),
            "expected_frame_count": self._expected_frame_count,
            "intrinsics_matrix": self._cache_manager.npz_array_key_signature(
                self._intrinsics_path,
                key="matrix",
                logical_name="intrinsics_matrix",
            ),
            "adapter_script": self._cache_manager.script_key_signature(
                Path(__file__),
                repo_root=self._settings.repo_root,
            ),
            "bridge_script": self._cache_manager.script_key_signature(
                self._settings.repo_root / "tools" / "DPVO" / "pemoin_bridge.py",
                repo_root=self._settings.repo_root,
            ),
            "mobile_labels": list(self._mobile_labels),
        }
        if self._settings.mask_dir is not None and self._settings.mask_dir.exists():
            payload["mask_dir"] = self._cache_manager.directory_signature(self._settings.mask_dir)
        return payload

    def _maybe_materialize_cross_run_cache(self) -> bool:
        payload = self._cross_run_payload()
        self._cache_payload = payload
        if payload is None or self._cache_manager is None:
            return False
        signature = self._cache_manager.signature("dpvo", payload)
        self._cache_signature = signature
        lookup = self._cache_manager.lookup("dpvo", signature)
        self._cache_status.update(
            {
                "cross_run_cache_signature": signature,
                "cross_run_cache_hit": lookup.hit,
                "cross_run_cache_entry": str(lookup.entry_dir),
                "cross_run_cache_validation": lookup.reason,
            }
        )
        if not lookup.hit:
            self._cache_status["cross_run_cache_reason"] = lookup.reason
            return False
        materialized = self._cache_manager.materialize(
            "dpvo",
            signature,
            run_root=self._run_root(),
        )
        self._cache_status["cross_run_cache_materialized"] = materialized
        return materialized > 0

    def _memory_diagnostics_path(self) -> Path:
        return self._output_dir / "dpvo_memory_diagnostics.json"

    def _validate_memory_diagnostics(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        schema_version = int(payload.get("schema_version", -1))
        if schema_version != 1:
            raise RuntimeError(
                f"DPVO memory diagnostics schema_version must be 1, got {schema_version}."
            )
        status = str(payload.get("status", "")).strip().lower()
        if status not in {"success", "failed"}:
            raise RuntimeError(
                "DPVO memory diagnostics status must be one of: success, failed."
            )
        memory = payload.get("memory")
        if not isinstance(memory, Mapping):
            raise RuntimeError("DPVO memory diagnostics must include a 'memory' mapping.")
        if "peak_allocated_bytes" not in memory or "peak_reserved_bytes" not in memory:
            raise RuntimeError(
                "DPVO memory diagnostics missing required memory peak fields."
            )
        frames = payload.get("frames")
        if not isinstance(frames, Mapping):
            raise RuntimeError("DPVO memory diagnostics must include a 'frames' mapping.")
        processed = int(frames.get("processed", -1))
        if processed < 0:
            raise RuntimeError("DPVO memory diagnostics frames.processed must be >= 0.")
        normalized = dict(payload)
        normalized["status"] = status
        return normalized

    def _materialize_memory_visualization(self, diagnostics: Mapping[str, Any]) -> None:
        memory = diagnostics.get("memory")
        if not isinstance(memory, Mapping):
            return
        samples = memory.get("samples")
        if not isinstance(samples, Sequence) or len(samples) == 0:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            LOG.warning("DPVO diagnostics plot disabled: matplotlib unavailable (%s).", exc)
            return
        frame_idx: list[int] = []
        allocated_mb: list[float] = []
        reserved_mb: list[float] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            try:
                frame_idx.append(int(sample.get("frame_index", len(frame_idx))))
                allocated_mb.append(float(sample.get("allocated_bytes", 0)) / (1024.0 * 1024.0))
                reserved_mb.append(float(sample.get("reserved_bytes", 0)) / (1024.0 * 1024.0))
            except Exception:
                continue
        if not frame_idx:
            return
        vis_dir = self._run_root() / "standard" / "visualizations" / "dpvo"
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir / "memory_usage_mb.png"
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(frame_idx, allocated_mb, label="allocated_mb", linewidth=1.8)
            ax.plot(frame_idx, reserved_mb, label="reserved_mb", linewidth=1.8)
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Memory (MiB)")
            ax.set_title("DPVO CUDA Memory Usage")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_path, dpi=140)
            plt.close(fig)
            self._cache_status["dpvo_memory_plot"] = str(out_path)
        except Exception as exc:
            LOG.warning("Failed to render DPVO diagnostics plot: %s", exc)

    def _load_memory_diagnostics(self) -> dict[str, Any]:
        diagnostics_path = self._memory_diagnostics_path()
        if not diagnostics_path.exists():
            raise RuntimeError(
                f"DPVO diagnostics artifact missing: '{diagnostics_path}'. "
                "Bridge must emit diagnostics on every run."
            )
        try:
            payload_raw = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse DPVO diagnostics at '{diagnostics_path}': {exc}"
            ) from exc
        if not isinstance(payload_raw, Mapping):
            raise RuntimeError(
                f"DPVO diagnostics at '{diagnostics_path}' must be a JSON object."
            )
        payload = self._validate_memory_diagnostics(payload_raw)
        self._memory_diagnostics = payload
        memory = payload.get("memory")
        frames = payload.get("frames")
        if isinstance(memory, Mapping):
            self._cache_status["dpvo_peak_allocated_bytes"] = int(
                memory.get("peak_allocated_bytes", 0)
            )
            self._cache_status["dpvo_peak_reserved_bytes"] = int(
                memory.get("peak_reserved_bytes", 0)
            )
        if isinstance(frames, Mapping):
            self._cache_status["dpvo_frames_processed"] = int(frames.get("processed", 0))
        self._cache_status["dpvo_memory_diagnostics"] = str(diagnostics_path)
        LOG.info(
            "DPVO diagnostics: peak_allocated=%.1f MiB peak_reserved=%.1f MiB processed_frames=%d",
            float(self._cache_status.get("dpvo_peak_allocated_bytes", 0)) / (1024.0 * 1024.0),
            float(self._cache_status.get("dpvo_peak_reserved_bytes", 0)) / (1024.0 * 1024.0),
            int(self._cache_status.get("dpvo_frames_processed", 0)),
            extra={"summary": True},
        )
        self._materialize_memory_visualization(payload)
        return payload

    def _ensure_ready(self) -> DPVOClient:
        if self._client is not None:
            return self._client

        # Auto-generate dynamic masks from semantics if available
        if (
            self._settings.mask_dir is None
            and self._resource_store is not None
            and self._mobile_labels
        ):
            mask_dir = _generate_dynamic_masks_from_semantics(
                self._resource_store,
                self._mobile_labels,
            )
            if mask_dir is not None:
                # Replace settings with mask_dir set
                self._settings = DPVOSettings(
                    mamba_env=self._settings.mamba_env,
                    network_path=self._settings.network_path,
                    config_path=self._settings.config_path,
                    stride=self._settings.stride,
                    skip=self._settings.skip,
                    reuse_exports=self._settings.reuse_exports,
                    device=self._settings.device,
                    precision_mode=self._settings.precision_mode,
                    memory_preset=self._settings.memory_preset,
                    allocator_mode=self._settings.allocator_mode,
                    allocator_max_split_size_mb=self._settings.allocator_max_split_size_mb,
                    cfg_opts=self._settings.cfg_opts,
                    env_manager=self._settings.env_manager,
                    mask_dir=mask_dir,
                    memory_diag_sample_every=self._settings.memory_diag_sample_every,
                    memory_diag_warn_ratio=self._settings.memory_diag_warn_ratio,
                    memory_guard_enabled=self._settings.memory_guard_enabled,
                    memory_guard_warmup_frames=self._settings.memory_guard_warmup_frames,
                    memory_guard_abort_reserved_vram_ratio=self._settings.memory_guard_abort_reserved_vram_ratio,
                    memory_guard_abort_reserved_to_allocated_ratio=self._settings.memory_guard_abort_reserved_to_allocated_ratio,
                    repo_root=self._settings.repo_root,
                )

        if self._maybe_materialize_cross_run_cache():
            LOG.info(
                "Reused cross-run DPVO cache at '%s'.",
                self._cache_status.get("cross_run_cache_entry"),
                extra={"summary": True},
            )
        results_path = self._output_dir / "dpvo_results.npz"
        if self._settings.reuse_exports and results_path.exists():
            LOG.info(
                "Reusing existing DPVO exports at '%s'.",
                self._output_dir,
                extra={"summary": True},
            )
        else:
            calib_file = self._write_calib_file()
            try:
                self._runner.run(
                    self._settings,
                    self._image_dir,
                    calib_file,
                    self._output_dir,
                )
            except subprocess.CalledProcessError as exc:
                diagnostics_path = self._memory_diagnostics_path()
                message = (
                    "DPVO subprocess failed. "
                    f"Exit code={exc.returncode}. "
                    f"Diagnostics: '{diagnostics_path}'."
                )
                raise RuntimeError(message) from exc

        diagnostics = self._load_memory_diagnostics()
        if str(diagnostics.get("status", "")).lower() != "success":
            diagnostics_path = self._memory_diagnostics_path()
            error_type = diagnostics.get("error_type")
            error_message = diagnostics.get("error_message")
            raise RuntimeError(
                "DPVO diagnostics reported failed status. "
                f"error_type={error_type!r} error_message={error_message!r} "
                f"diagnostics='{diagnostics_path}'."
            )

        client = DPVOClient(self._settings, self._output_dir)
        client.load()
        self._publish_match_graph()
        self._client = client
        return client

    def _write_calib_file(self) -> Path:
        if self._intrinsics_path is None:
            raise RuntimeError(
                "DPVO requires camera intrinsics. Ensure an intrinsics provider "
                "runs before the trajectory provider in the profile."
            )
        intrinsics_data = np.load(self._intrinsics_path, allow_pickle=True)
        K = np.asarray(intrinsics_data["matrix"], dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        calib_path = self._output_dir / "calib.txt"
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        calib_path.write_text(f"{fx} {fy} {cx} {cy}\n", encoding="utf-8")
        return calib_path

    def create_trajectory_provider(self) -> "DPVOTrajectoryProvider":
        return DPVOTrajectoryProvider(
            adapter=self,
            expected_frame_count=self._expected_frame_count,
        )

    def _publish_match_graph(self) -> None:
        if self._resource_store is None:
            return
        raw_path = self._output_dir / "dpvo_match_graph.npz"
        if not raw_path.exists():
            return
        with np.load(raw_path, allow_pickle=True) as data:
            payload = {
                str(key): np.asarray(data[key])
                for key in data.files
            }
        self._resource_store.save_trajectory_match_graph(
            TrajectoryMatchGraphData(
                payload=payload,
                metadata={
                    "source": "dpvo",
                    "tool_output_path": str(raw_path),
                },
            )
        )

    def cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._cache_status)

    def cross_run_cache_spec(self, resource_store: Optional[ResourceStore]) -> Optional[dict[str, Any]]:
        if (
            self._cache_manager is None
            or not self._cache_manager.enabled
            or self._cache_signature is None
            or self._cache_payload is None
        ):
            return None
        artifacts = self._cache_manager.collect_tree(self._output_dir, rel_prefix="raw/dpvo")
        trajectory_path = self._run_root() / "standard" / "trajectory" / "poses.npz"
        ready = True
        not_ready_reason: Optional[str] = None
        if not artifacts:
            ready = False
            not_ready_reason = "raw-exports-missing"
        if not trajectory_path.exists():
            ready = False
            not_ready_reason = "standard-trajectory-missing"
        if trajectory_path.exists():
            artifacts.update(
                self._cache_manager.collect_file(
                    trajectory_path,
                    relpath="standard/trajectory/poses.npz",
                )
            )
        match_graph_path = (
            self._run_root()
            / "standard"
            / "trajectory_match_graph"
            / "dpvo_match_graph.npz"
        )
        raw_match_graph_path = self._output_dir / "dpvo_match_graph.npz"
        if raw_match_graph_path.exists() and not match_graph_path.exists():
            ready = False
            not_ready_reason = "standard-match-graph-missing"
        if match_graph_path.exists():
            artifacts.update(
                self._cache_manager.collect_file(
                    match_graph_path,
                    relpath="standard/trajectory_match_graph/dpvo_match_graph.npz",
                )
            )
        spec = {
            "provider_id": "dpvo",
            "signature": self._cache_signature,
            "payload": self._cache_payload,
            "artifacts": artifacts,
            "ready": ready,
            "source_summary": {
                "profile": self._profile_name,
                "run_root": str(self._run_root()),
            },
            "provenance": {
                "image_dir": str(self._image_dir),
                "intrinsics_path": (
                    str(self._intrinsics_path.resolve())
                    if self._intrinsics_path is not None and self._intrinsics_path.exists()
                    else None
                ),
                "mask_dir": (
                    str(self._settings.mask_dir.resolve())
                    if self._settings.mask_dir is not None and self._settings.mask_dir.exists()
                    else None
                ),
            },
        }
        if not_ready_reason is not None:
            spec["not_ready_reason"] = not_ready_reason
        return spec


class DPVOTrajectoryProvider(TrajectoryProvider):
    """Streaming trajectory provider backed by DPVO."""

    # When True, the runtime skips this provider in the per-frame loop and
    # calls flush() after batch-oriented providers (e.g. CAVIS) complete.
    deferred_batch: bool = True
    execution_mode = ProviderExecutionMode.DEFERRED_BATCH

    def __init__(self, adapter: DPVOAdapter, expected_frame_count: Optional[int] = None):
        self._adapter = adapter
        self._client: Optional[DPVOClient] = None
        self._expected_frame_count = expected_frame_count
        self._processed_frames = 0

    def setup(self, context: MutableMapping[str, Any]) -> None:
        self._client = None
        self._processed_frames = 0

    def process(self, frame: Any) -> PoseData:
        self._processed_frames += 1
        frame_index = int(getattr(frame, "index", 0))
        if (
            self._expected_frame_count is not None
            and self._processed_frames < self._expected_frame_count
        ):
            # DPVO requires the full sampled sequence. Defer execution until the
            # final frame has been persisted by runtime to standard/frames.
            return PoseData(samples=[], metadata={"source": "DPVO"})

        if self._client is None:
            self._client = self._adapter._ensure_ready()
        return self._client.get_pose(frame_index)

    def flush(self) -> PoseData:
        """Force DPVO to run immediately, bypassing the per-frame counter.

        Called by the runtime after batch-oriented providers complete so that
        DPVO can use any masks they produced (e.g. CAVIS semantics → dynamic masks).
        """
        if self._client is None:
            self._client = self._adapter._ensure_ready()
        return self._client.to_pose_data()

    def teardown(self) -> None:
        pass

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return self._adapter.cross_run_cache_status()

    def get_cross_run_cache_spec(self, _resource_store: Optional[ResourceStore]) -> Optional[dict[str, Any]]:
        return self._adapter.cross_run_cache_spec(_resource_store)


def _coerce_dpvo_settings(raw: Mapping[str, Any], repo_root: Path) -> DPVOSettings:
    network_path = Path(str(raw.get("network_path", "tools/DPVO/dpvo.pth")))
    if not network_path.is_absolute():
        network_path = repo_root / network_path

    memory_preset = str(raw.get("memory_preset", "balanced")).strip().lower()
    if memory_preset not in {"balanced", "low_vram"}:
        raise ValueError("DPVO adapter memory_preset must be one of: balanced, low_vram.")

    config_default = (
        "tools/DPVO/config/fast.yaml"
        if memory_preset == "low_vram"
        else "tools/DPVO/config/default.yaml"
    )
    config_path = Path(str(raw.get("config_path", config_default)))
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    precision_mode = str(raw.get("precision_mode", "amp_fp16")).strip().lower()
    if precision_mode not in {"amp_fp16", "fp32"}:
        raise ValueError("DPVO adapter precision_mode must be one of: amp_fp16, fp32.")
    allocator_mode = str(raw.get("allocator_mode", "native")).strip().lower()
    if allocator_mode not in {"native", "expandable_segments", "cuda_malloc_async"}:
        raise ValueError(
            "DPVO adapter allocator_mode must be one of: native, expandable_segments, cuda_malloc_async."
        )
    allocator_max_split_size_mb_raw = raw.get("allocator_max_split_size_mb")
    allocator_max_split_size_mb: Optional[int] = None
    if allocator_max_split_size_mb_raw is not None:
        allocator_max_split_size_mb = int(allocator_max_split_size_mb_raw)
        if allocator_max_split_size_mb < 1:
            raise ValueError("DPVO adapter allocator_max_split_size_mb must be >= 1.")

    cfg_opts_raw = raw.get("cfg_opts")
    cfg_opts: list[str] = []
    if cfg_opts_raw is not None:
        if not isinstance(cfg_opts_raw, Sequence) or isinstance(cfg_opts_raw, (str, bytes)):
            raise ValueError("DPVO adapter cfg_opts must be a list of strings.")
        for token in cfg_opts_raw:
            text = str(token).strip()
            if not text:
                continue
            cfg_opts.append(text)

    if memory_preset == "low_vram" and not cfg_opts:
        # Reduce DPVO's persistent buffer allocation for low-memory GPUs.
        cfg_opts = ["BUFFER_SIZE", "1024"]
    if cfg_opts and len(cfg_opts) % 2 != 0:
        raise ValueError(
            "DPVO adapter cfg_opts must contain KEY VALUE pairs (even number of tokens)."
        )
    cfg_pairs: dict[str, str] = {}
    for idx in range(0, len(cfg_opts), 2):
        key = cfg_opts[idx].strip()
        value = cfg_opts[idx + 1].strip()
        if not key:
            raise ValueError("DPVO adapter cfg_opts contains an empty key.")
        cfg_pairs[key] = value

    diagnostics_raw = raw.get("memory_diagnostics", {})
    if diagnostics_raw is None:
        diagnostics_raw = {}
    if not isinstance(diagnostics_raw, Mapping):
        raise ValueError("DPVO adapter memory_diagnostics must be a mapping when provided.")
    memory_diag_sample_every = int(diagnostics_raw.get("sample_every_n_frames", 1))
    if memory_diag_sample_every < 1:
        raise ValueError(
            "DPVO adapter memory_diagnostics.sample_every_n_frames must be >= 1."
        )
    memory_diag_warn_ratio = float(diagnostics_raw.get("warn_vram_used_ratio", 0.92))
    if not (0.0 < memory_diag_warn_ratio <= 1.0):
        raise ValueError(
            "DPVO adapter memory_diagnostics.warn_vram_used_ratio must be in (0, 1]."
        )
    memory_guard_raw = raw.get("memory_guard", {})
    if memory_guard_raw is None:
        memory_guard_raw = {}
    if not isinstance(memory_guard_raw, Mapping):
        raise ValueError("DPVO adapter memory_guard must be a mapping when provided.")
    memory_guard_enabled = bool(memory_guard_raw.get("enabled", True))
    memory_guard_warmup_frames = int(memory_guard_raw.get("warmup_frames", 24))
    if memory_guard_warmup_frames < 0:
        raise ValueError("DPVO adapter memory_guard.warmup_frames must be >= 0.")
    memory_guard_abort_reserved_vram_ratio = float(
        memory_guard_raw.get("abort_reserved_vram_ratio", 0.70)
    )
    if not (0.0 < memory_guard_abort_reserved_vram_ratio <= 1.0):
        raise ValueError(
            "DPVO adapter memory_guard.abort_reserved_vram_ratio must be in (0, 1]."
        )
    memory_guard_abort_reserved_to_allocated_ratio = float(
        memory_guard_raw.get("abort_reserved_to_allocated_ratio", 12.0)
    )
    if memory_guard_abort_reserved_to_allocated_ratio <= 0.0:
        raise ValueError(
            "DPVO adapter memory_guard.abort_reserved_to_allocated_ratio must be > 0."
        )
    if memory_preset == "low_vram":
        patches_value = int(cfg_pairs.get("PATCHES_PER_FRAME", "48"))
        buffer_value = int(cfg_pairs.get("BUFFER_SIZE", "1024"))
        if patches_value > 64 or buffer_value > 1024:
            LOG.warning(
                "DPVO low_vram preset configured with aggressive settings: "
                "PATCHES_PER_FRAME=%d BUFFER_SIZE=%d. "
                "This may increase CUDA OOM risk.",
                patches_value,
                buffer_value,
                extra={"summary": True},
            )

    mask_dir_raw = raw.get("mask_dir")
    mask_dir: Optional[Path] = None
    if mask_dir_raw is not None:
        mask_dir = Path(str(mask_dir_raw)).expanduser().resolve()

    return DPVOSettings(
        mamba_env=str(raw.get("mamba_env", "dpvo")),
        network_path=network_path,
        config_path=config_path,
        stride=int(raw.get("stride", 1)),
        skip=int(raw.get("skip", 0)),
        reuse_exports=bool(raw.get("reuse_exports", True)),
        device=str(raw.get("device", "cuda")),
        precision_mode=precision_mode,
        memory_preset=memory_preset,
        allocator_mode=allocator_mode,
        allocator_max_split_size_mb=allocator_max_split_size_mb,
        cfg_opts=tuple(cfg_opts),
        env_manager=(
            str(raw.get("env_manager")).strip()
            if raw.get("env_manager") is not None and str(raw.get("env_manager")).strip()
            else None
        ),
        mask_dir=mask_dir,
        memory_diag_sample_every=memory_diag_sample_every,
        memory_diag_warn_ratio=memory_diag_warn_ratio,
        memory_guard_enabled=memory_guard_enabled,
        memory_guard_warmup_frames=memory_guard_warmup_frames,
        memory_guard_abort_reserved_vram_ratio=memory_guard_abort_reserved_vram_ratio,
        memory_guard_abort_reserved_to_allocated_ratio=memory_guard_abort_reserved_to_allocated_ratio,
        repo_root=repo_root,
    )


def register_dpvo_provider_builders(factory: ProviderFactory) -> None:
    """Register DPVO-backed provider builders with the factory."""

    def _resolve_adapter(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> DPVOAdapter:
        adapter_settings_raw: dict[str, Any] = {}
        adapter_settings_raw.update(binding.settings.get("adapter", {}))

        repo_root = _resolve_repo_root(
            Path(adapter_settings_raw["repo_root"]) if "repo_root" in adapter_settings_raw else None
        )
        settings = _coerce_dpvo_settings(adapter_settings_raw, repo_root)

        # Allow runtime context to provide mask directory
        ctx_mask_dir = context.get("dpvo_mask_dir")
        if ctx_mask_dir is not None and settings.mask_dir is None:
            settings = DPVOSettings(
                mamba_env=settings.mamba_env,
                network_path=settings.network_path,
                config_path=settings.config_path,
                stride=settings.stride,
                skip=settings.skip,
                reuse_exports=settings.reuse_exports,
                device=settings.device,
                precision_mode=settings.precision_mode,
                memory_preset=settings.memory_preset,
                allocator_mode=settings.allocator_mode,
                allocator_max_split_size_mb=settings.allocator_max_split_size_mb,
                cfg_opts=settings.cfg_opts,
                env_manager=settings.env_manager,
                mask_dir=Path(str(ctx_mask_dir)).expanduser().resolve(),
                memory_diag_sample_every=settings.memory_diag_sample_every,
                memory_diag_warn_ratio=settings.memory_diag_warn_ratio,
                memory_guard_enabled=settings.memory_guard_enabled,
                memory_guard_warmup_frames=settings.memory_guard_warmup_frames,
                memory_guard_abort_reserved_vram_ratio=settings.memory_guard_abort_reserved_vram_ratio,
                memory_guard_abort_reserved_to_allocated_ratio=settings.memory_guard_abort_reserved_to_allocated_ratio,
                repo_root=settings.repo_root,
            )

        frames_dir = context.get("frames_dir")
        if frames_dir is None:
            raise ValueError(
                "DPVOTrajectoryProvider requires 'frames_dir' in context so DPVO uses "
                "FrameProvider-persisted frames (standard/frames)."
            )
        image_dir = Path(str(frames_dir)).expanduser().resolve()
        image_dir.mkdir(parents=True, exist_ok=True)

        store = context.get("resource_store")
        if isinstance(store, ResourceStore):
            output_dir = store.raw_root / "dpvo"
            intrinsics_path = store.path_for(
                __import__("pemoin.data.contracts", fromlist=["ResourceKind"]).ResourceKind.INTRINSICS
            )
        else:
            run_dir = context.get("run_dir")
            if run_dir is None:
                raise ValueError(
                    "DPVOTrajectoryProvider requires 'resource_store' or 'run_dir' in context."
                )
            output_dir = Path(run_dir) / "raw" / "dpvo"
            intrinsics_path = Path(run_dir) / "standard" / "intrinsics" / "intrinsics.npz"

        expected_frame_count_raw = context.get("expected_frame_count")
        expected_frame_count = (
            int(expected_frame_count_raw)
            if expected_frame_count_raw is not None
            else None
        )

        # Resolve mobile semantic role labels for dynamic mask generation.
        mobile_labels: tuple[str, ...] = ()
        role_defaults = context.get("semantic_role_defaults")
        if isinstance(role_defaults, Mapping):
            raw_mobile = role_defaults.get("mobile")
            if isinstance(raw_mobile, str):
                mobile_labels = tuple(part.strip().lower() for part in raw_mobile.split(",") if part.strip())
            elif isinstance(raw_mobile, (list, tuple)):
                mobile_labels = tuple(str(label).strip().lower() for label in raw_mobile if str(label).strip())

        cache = context.setdefault("dpvo_adapters", {})
        key = json.dumps({"settings": str(settings), "image_dir": str(image_dir)}, sort_keys=True)
        if key not in cache:
            cache[key] = DPVOAdapter(
                settings=settings,
                image_dir=image_dir,
                output_dir=output_dir,
                intrinsics_path=intrinsics_path,
                expected_frame_count=expected_frame_count,
                resource_store=store if isinstance(store, ResourceStore) else None,
                mobile_labels=mobile_labels,
                cache_manager=context.get("cross_run_cache")
                if isinstance(context.get("cross_run_cache"), CrossRunCacheManager)
                else None,
                profile_name=str(context.get("profile_name")) if context.get("profile_name") is not None else None,
            )
        return cache[key]

    def build_trajectory(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> DPVOTrajectoryProvider:
        adapter = _resolve_adapter(binding, context)
        return adapter.create_trajectory_provider()

    factory.register("DPVOTrajectoryProvider", build_trajectory)
