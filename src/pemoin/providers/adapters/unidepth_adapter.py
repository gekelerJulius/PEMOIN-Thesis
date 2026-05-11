"""
Adapter integrating UniDepth V2 as depth and intrinsics providers.

UniDepth runs inside a separate mamba environment and communicates via subprocess +
per-frame NPZ files, following the established DA3 adapter pattern.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional

import numpy as np

from pemoin.data.contracts import DepthData, IntrinsicsData, ResourceKind, ResourceStore
from pemoin.providers.base import ProviderExecutionMode
from pemoin.providers.depth import DepthProvider
from pemoin.providers.factory import ProviderFactory
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.runtime.cache import CrossRunCacheManager
from pemoin.runtime.profiles.config import ModuleBinding
from pemoin.utils.env_launcher import resolve_env_launcher as _resolve_env_launcher
from pemoin.utils.logging import get_logger
from pemoin.utils.model_cache import configure_hf_subprocess_env

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
        unidepth_dir = base / "tools" / "UniDepth"
        if unidepth_dir.exists():
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
            f"UniDepth adapters expect a frame directory, got '{source}'."
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
class UniDepthSettings:
    """Settings controlling how UniDepth is invoked."""

    mamba_env: str = "unidepth-cu121"
    model_name: str = "lpiccinelli/unidepth-v2-vitl14"
    reuse_exports: bool = True
    device: str = "cuda"
    batch_size: int = 1
    amp: Optional[bool] = None
    precision_mode: str = "amp_fp16"
    save_confidence: bool = True
    channels_last: bool = True
    env_manager: Optional[str] = None
    repo_root: Path = _REPO_ROOT_FALLBACK


class UniDepthRunner:
    """Executes the UniDepth bridge script inside the specified mamba environment."""

    def __init__(self, repo_root: Optional[Path] = None):
        self._repo_root = _resolve_repo_root(repo_root)

    def run(
        self,
        settings: UniDepthSettings,
        image_dir: Path,
        output_dir: Path,
        intrinsics_path: Optional[Path] = None,
    ) -> None:
        bridge_script = self._repo_root / "tools" / "UniDepth" / "pemoin_bridge.py"
        if not bridge_script.exists():
            raise FileNotFoundError(
                f"UniDepth bridge script not found at '{bridge_script}'. "
                "Ensure tools/UniDepth/pemoin_bridge.py exists."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        launcher = _resolve_env_launcher(settings.mamba_env, settings.env_manager)
        cmd = [
            *launcher,
            "python", str(bridge_script),
            "--image-dir", str(image_dir),
            "--output-dir", str(output_dir),
            "--model", settings.model_name,
            "--batch-size", str(settings.batch_size),
            "--device", settings.device,
            "--amp",
            "true" if _settings_amp_enabled(settings) else "false",
            "--precision-mode", settings.precision_mode,
            "--save-confidence", "true" if settings.save_confidence else "false",
            "--channels-last", "true" if settings.channels_last else "false",
        ]
        if intrinsics_path is not None and intrinsics_path.exists():
            cmd.extend(["--intrinsics-path", str(intrinsics_path)])

        env = os.environ.copy()
        unidepth_dir = self._repo_root / "tools" / "UniDepth"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(unidepth_dir), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        if launcher[0] in {"micromamba", "mamba"} and not env.get("XDG_CACHE_HOME"):
            cache_root = output_dir / ".mamba_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            env["XDG_CACHE_HOME"] = str(cache_root)
        env = configure_hf_subprocess_env(env)

        LOG.info(
            "Running UniDepth inference: manager=%s, env=%s, model=%s, images=%s",
            launcher[0], settings.mamba_env, settings.model_name, image_dir,
            extra={"summary": True},
        )
        subprocess.run(cmd, check=True, cwd=unidepth_dir, env=env)


class UniDepthClient:
    """Loads UniDepth per-frame results and converts to PEMOIN data contracts."""

    def __init__(self, settings: UniDepthSettings, output_dir: Path):
        self.settings = settings
        self._output_dir = output_dir
        self._intrinsics: Optional[np.ndarray] = None
        self._frame_count: int = 0

    def load(self) -> None:
        npz_files = sorted(self._output_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No UniDepth frame results found in '{self._output_dir}'. "
                "Run UniDepth inference first."
            )
        self._frame_count = len(npz_files)
        first = np.load(npz_files[0], allow_pickle=True)
        if "intrinsics" in first.files:
            self._intrinsics = np.asarray(first["intrinsics"], dtype=np.float32)
        first.close()
        LOG.info(
            "Loaded UniDepth results: %d frames from '%s'.",
            self._frame_count, self._output_dir,
            extra={"summary": True},
        )

    @property
    def num_frames(self) -> int:
        return self._frame_count

    def get_depth(self, frame_index: int) -> DepthData:
        npz_path = self._output_dir / f"{frame_index:06d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"UniDepth depth for frame {frame_index} not found at '{npz_path}'."
            )
        with np.load(npz_path, allow_pickle=True) as data:
            depth = np.asarray(data["depth"], dtype=np.float32)
            confidence = (
                np.asarray(data["confidence"], dtype=np.float32)
                if "confidence" in data.files
                else None
            )

        metadata: MutableMapping[str, Any] = {
            "source": "UniDepth",
            "model_name": self.settings.model_name,
            "metric_depth": True,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
            "frame_index": frame_index,
        }
        return DepthData(
            frame_index=frame_index,
            depth=depth,
            confidence=confidence,
            metadata=metadata,
        )

    def get_intrinsics(self) -> IntrinsicsData:
        if self._intrinsics is not None:
            K = self._intrinsics.copy()
        else:
            first_npz = self._output_dir / "000000.npz"
            if not first_npz.exists():
                raise FileNotFoundError(
                    f"UniDepth intrinsics not found at '{first_npz}'."
                )
            with np.load(first_npz, allow_pickle=True) as data:
                K = np.asarray(data["intrinsics"], dtype=np.float32)
            self._intrinsics = K.copy()

        metadata: MutableMapping[str, Any] = {
            "source": "UniDepth",
            "model_name": self.settings.model_name,
            "dynamic": False,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        return IntrinsicsData(matrix=K, metadata=metadata)


class UniDepthAdapter:
    """Bridges UniDepth into PEMOIN providers."""

    def __init__(
        self,
        settings: UniDepthSettings,
        image_dir: Path,
        output_dir: Path,
        intrinsics_path: Optional[Path] = None,
        expected_frame_count: Optional[int] = None,
        cache_manager: Optional[CrossRunCacheManager] = None,
        profile_name: Optional[str] = None,
    ):
        self._settings = settings
        self._image_dir = image_dir
        self._output_dir = output_dir
        self._intrinsics_path = intrinsics_path
        self._expected_frame_count = (
            int(expected_frame_count) if expected_frame_count is not None else None
        )
        self._cache_manager = cache_manager
        self._profile_name = profile_name
        self._client: Optional[UniDepthClient] = None
        self._runner = UniDepthRunner(settings.repo_root)
        self._cache_status: dict[str, Any] = {
            "cross_run_cache_enabled": bool(cache_manager and cache_manager.enabled),
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled" if cache_manager is None or not cache_manager.enabled else "not-checked",
        }
        self._cache_signature: Optional[str] = None
        self._cache_payload: Optional[dict[str, Any]] = None
        self._depth_provider_created = False
        self._intrinsics_provider_created = False

    def _run_root(self) -> Path:
        return self._output_dir.parents[1]

    def _settings_signature_payload(self) -> dict[str, Any]:
        payload = asdict(self._settings)
        for key in ("reuse_exports", "repo_root"):
            payload.pop(key, None)
        return payload

    def _cross_run_payload(self) -> Optional[dict[str, Any]]:
        if self._cache_manager is None or not self._cache_manager.enabled:
            return None
        images = _discover_images(self._image_dir)
        if self._expected_frame_count is not None and len(images) < self._expected_frame_count:
            self._cache_status.update(
                {
                    "cross_run_cache_hit": False,
                    "cross_run_cache_validation": "incomplete-input",
                    "cross_run_cache_reason": "frames-not-fully-persisted",
                }
            )
            return None
        payload: dict[str, Any] = {
            "settings": self._settings_signature_payload(),
            "image_dir": self._cache_manager.directory_signature(self._image_dir),
            "image_count": len(images),
            "expected_frame_count": self._expected_frame_count,
            "adapter_script": self._cache_manager.script_key_signature(
                Path(__file__),
                repo_root=self._settings.repo_root,
            ),
            "bridge_script": self._cache_manager.script_key_signature(
                self._settings.repo_root / "tools" / "UniDepth" / "pemoin_bridge.py",
                repo_root=self._settings.repo_root,
            ),
        }
        if self._intrinsics_path is not None and self._intrinsics_path.exists():
            payload["intrinsics_matrix"] = self._cache_manager.npz_array_key_signature(
                self._intrinsics_path,
                key="matrix",
                logical_name="intrinsics_matrix",
            )
        return payload

    def _maybe_materialize_cross_run_cache(self) -> bool:
        payload = self._cross_run_payload()
        self._cache_payload = payload
        if payload is None or self._cache_manager is None:
            return False
        signature = self._cache_manager.signature("unidepth", payload)
        self._cache_signature = signature
        lookup = self._cache_manager.lookup("unidepth", signature)
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
            "unidepth",
            signature,
            run_root=self._run_root(),
        )
        self._cache_status["cross_run_cache_materialized"] = materialized
        return materialized > 0

    def _ensure_ready(self) -> UniDepthClient:
        if self._client is not None:
            return self._client

        # Validate image_dir now that frames have been extracted
        if not self._image_dir.is_dir():
            raise FileNotFoundError(
                f"UniDepth image directory does not exist: '{self._image_dir}'. "
                "Ensure frames have been extracted before running UniDepth."
            )
        if not _discover_images(self._image_dir):
            raise FileNotFoundError(
                f"No supported images found in '{self._image_dir}'."
            )

        if self._maybe_materialize_cross_run_cache():
            LOG.info(
                "Reused cross-run UniDepth cache at '%s'.",
                self._cache_status.get("cross_run_cache_entry"),
                extra={"summary": True},
            )
        existing = list(self._output_dir.glob("*.npz"))
        if self._settings.reuse_exports and existing:
            LOG.info(
                "Reusing existing UniDepth exports at '%s' (%d frames).",
                self._output_dir, len(existing),
                extra={"summary": True},
            )
        else:
            self._runner.run(
                self._settings,
                self._image_dir,
                self._output_dir,
                intrinsics_path=self._intrinsics_path,
            )

        client = UniDepthClient(self._settings, self._output_dir)
        client.load()
        self._client = client
        return client

    def create_depth_provider(self) -> "UniDepthDepthProvider":
        self._depth_provider_created = True
        return UniDepthDepthProvider(adapter=self)

    def create_intrinsics_provider(self) -> "UniDepthIntrinsicsProvider":
        self._intrinsics_provider_created = True
        return UniDepthIntrinsicsProvider(adapter=self)

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
        artifacts = self._cache_manager.collect_tree(self._output_dir, rel_prefix="raw/unidepth")
        run_root = self._run_root()
        depth_dir = run_root / "standard" / "depth"
        intr_path = run_root / "standard" / "intrinsics" / "intrinsics.npz"
        ready = True
        not_ready_reason: Optional[str] = None
        if not artifacts:
            ready = False
            not_ready_reason = "raw-exports-missing"
        if self._depth_provider_created and not depth_dir.exists():
            ready = False
            not_ready_reason = "standard-depth-missing"
        if self._intrinsics_provider_created and not intr_path.exists():
            ready = False
            not_ready_reason = "standard-intrinsics-missing"
        if intr_path.exists():
            artifacts.update(self._cache_manager.collect_file(intr_path, relpath="standard/intrinsics/intrinsics.npz"))
        if depth_dir.exists():
            artifacts.update(self._cache_manager.collect_tree(depth_dir, rel_prefix="standard/depth"))
        spec = {
            "provider_id": "unidepth",
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
            },
        }
        if not_ready_reason is not None:
            spec["not_ready_reason"] = not_ready_reason
        return spec


class UniDepthDepthProvider(DepthProvider):
    """Depth provider backed by UniDepth."""

    # When True, the runtime skips this provider in the per-frame loop and
    # runs it after deferred trajectory providers complete.
    deferred_batch: bool = True
    execution_mode = ProviderExecutionMode.DEFERRED_BATCH

    def __init__(self, adapter: UniDepthAdapter):
        self._adapter = adapter
        self._client: Optional[UniDepthClient] = None

    def setup(self, context: MutableMapping[str, Any]) -> None:
        self._client = None

    def process(self, frame: Any) -> DepthData:
        if self._client is None:
            self._client = self._adapter._ensure_ready()
        frame_index = getattr(frame, "index", 0)
        return self._client.get_depth(frame_index)

    def teardown(self) -> None:
        pass

    def try_materialize_standardized_outputs(
        self,
        resource_store: Optional[ResourceStore],
    ) -> bool:
        if resource_store is None:
            return False
        client = self._adapter._ensure_ready()
        status = self._adapter.cross_run_cache_status()
        if not bool(status.get("cross_run_cache_hit")):
            return False
        depth_indices = resource_store.frame_indices(ResourceKind.DEPTH)
        if len(depth_indices) != int(client.num_frames):
            return False
        return True

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return self._adapter.cross_run_cache_status()

    def get_cross_run_cache_spec(self, _resource_store: Optional[ResourceStore]) -> Optional[dict[str, Any]]:
        return self._adapter.cross_run_cache_spec(_resource_store)


class UniDepthIntrinsicsProvider(IntrinsicsProvider):
    """Intrinsics provider backed by UniDepth (static intrinsics from first frame)."""

    def __init__(self, adapter: UniDepthAdapter):
        self._adapter = adapter
        self._client: Optional[UniDepthClient] = None

    def setup(self, context: MutableMapping[str, Any]) -> None:
        super().setup(context)
        self._client = None

    def process(self, frame: Any) -> IntrinsicsData:
        if self._client is None:
            self._client = self._adapter._ensure_ready()
        intrinsics = self._client.get_intrinsics()
        return self._scale_intrinsics(intrinsics, frame)

    def teardown(self) -> None:
        pass

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return self._adapter.cross_run_cache_status()

    def get_cross_run_cache_spec(self, _resource_store: Optional[ResourceStore]) -> Optional[dict[str, Any]]:
        return self._adapter.cross_run_cache_spec(_resource_store)


def _coerce_unidepth_settings(raw: Mapping[str, Any], repo_root: Path) -> UniDepthSettings:
    batch_size = int(raw.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("UniDepth adapter batch_size must be > 0.")

    precision_mode_raw = raw.get("precision_mode")
    if precision_mode_raw is None:
        if "amp" in raw:
            precision_mode = "amp_fp16" if bool(raw.get("amp", True)) else "fp32"
        else:
            precision_mode = "amp_fp16"
    else:
        precision_mode = str(precision_mode_raw).strip().lower()
    if precision_mode not in {"amp_fp16", "fp32", "bf16"}:
        raise ValueError(
            "UniDepth adapter precision_mode must be one of: amp_fp16, fp32, bf16."
        )

    return UniDepthSettings(
        mamba_env=str(raw.get("mamba_env", "unidepth-cu121")),
        model_name=str(raw.get("model_name", "lpiccinelli/unidepth-v2-vitl14")),
        reuse_exports=bool(raw.get("reuse_exports", True)),
        device=str(raw.get("device", "cuda")),
        batch_size=batch_size,
        amp=(
            bool(raw.get("amp"))
            if "amp" in raw
            else None
        ),
        precision_mode=precision_mode,
        save_confidence=bool(raw.get("save_confidence", True)),
        channels_last=bool(raw.get("channels_last", True)),
        env_manager=(
            str(raw.get("env_manager")).strip()
            if raw.get("env_manager") is not None and str(raw.get("env_manager")).strip()
            else None
        ),
        repo_root=repo_root,
    )


def _settings_amp_enabled(settings: UniDepthSettings) -> bool:
    if settings.amp is not None:
        return bool(settings.amp)
    return settings.precision_mode == "amp_fp16"


def register_unidepth_provider_builders(factory: ProviderFactory) -> None:
    """Register UniDepth-backed provider builders with the factory."""

    def _resolve_adapter(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> UniDepthAdapter:
        adapter_settings_raw: dict[str, Any] = {}
        adapter_settings_raw.update(binding.settings.get("adapter", {}))

        repo_root = _resolve_repo_root(
            Path(adapter_settings_raw["repo_root"]) if "repo_root" in adapter_settings_raw else None
        )
        settings = _coerce_unidepth_settings(adapter_settings_raw, repo_root)

        frames_dir = context.get("frames_dir")
        if frames_dir is None:
            raise ValueError(
                f"Provider '{binding.tool}' requires 'frames_dir' in context."
            )
        image_dir = Path(str(frames_dir)).expanduser().resolve()

        store = context.get("resource_store")
        if isinstance(store, ResourceStore):
            output_dir = store.raw_root / "unidepth"
            from pemoin.data.contracts import ResourceKind
            intrinsics_path = store.path_for(ResourceKind.INTRINSICS)
        else:
            run_dir = context.get("run_dir")
            if run_dir is None:
                raise ValueError(
                    f"Provider '{binding.tool}' requires 'resource_store' or 'run_dir' in context."
                )
            output_dir = Path(run_dir) / "raw" / "unidepth"
            intrinsics_path = Path(run_dir) / "standard" / "intrinsics" / "intrinsics.npz"

        # Only pass intrinsics_path if the adapter settings request it
        use_gt_intrinsics = bool(adapter_settings_raw.get("use_gt_intrinsics", False))
        resolved_intrinsics_path = intrinsics_path if use_gt_intrinsics else None
        expected_frame_count_raw = context.get("expected_frame_count")
        expected_frame_count = (
            int(expected_frame_count_raw)
            if expected_frame_count_raw is not None
            else None
        )
        cache_manager = context.get("cross_run_cache")
        cache_obj = cache_manager if isinstance(cache_manager, CrossRunCacheManager) else None

        cache = context.setdefault("unidepth_adapters", {})
        key = json.dumps({"settings": str(settings), "image_dir": str(image_dir)}, sort_keys=True)
        if key not in cache:
            cache[key] = UniDepthAdapter(
                settings=settings,
                image_dir=image_dir,
                output_dir=output_dir,
                intrinsics_path=resolved_intrinsics_path,
                expected_frame_count=expected_frame_count,
                cache_manager=cache_obj,
                profile_name=str(context.get("profile_name")) if context.get("profile_name") is not None else None,
            )
        return cache[key]

    def build_depth(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> UniDepthDepthProvider:
        adapter = _resolve_adapter(binding, context)
        return adapter.create_depth_provider()

    def build_intrinsics(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> UniDepthIntrinsicsProvider:
        adapter = _resolve_adapter(binding, context)
        return adapter.create_intrinsics_provider()

    factory.register("UniDepthDepthProvider", build_depth)
    factory.register("UniDepthIntrinsicsProvider", build_intrinsics)
