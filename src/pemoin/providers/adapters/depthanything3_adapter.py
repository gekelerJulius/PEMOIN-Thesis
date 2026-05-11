"""
Adapter integrating Depth Anything 3 (DA3) outputs (streaming or standard)
as depth, trajectory, and intrinsics providers.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import re
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from pemoin.data.contracts import DepthData, IntrinsicsData, PoseData, PoseSample, ResourceStore
from pemoin.providers.depth import DepthProvider
from pemoin.providers.factory import ProviderFactory
from pemoin.providers.intrinsics import IntrinsicsProvider
from pemoin.providers.trajectory import TrajectoryProvider
from pemoin.runtime.profiles.config import ModuleBinding
from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender
from pemoin.utils.geometry_export import save_standard_geometry
from pemoin.utils.logging import get_logger
from pemoin.utils.trajectory_cleanup import TrajectoryCleanupOptions, cleanup_camera_to_world

LOG = get_logger()

_DEFAULT_MODEL_DIR = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
_DA3_MODE_STREAMING = "streaming"
_DA3_MODE_STANDARD = "standard"
_SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v")
_STREAMING_EXTENSIONS = (".png", ".jpg")
_PARENTS = Path(__file__).resolve().parents
_REPO_ROOT_FALLBACK = (
    _PARENTS[5]
    if len(_PARENTS) > 5
    else _PARENTS[-1]
)

_DA3_LINALG_WRAPPER = (
    "import os, runpy, sys\n"
    "import torch\n"
    "pref = os.environ.get('PEMOIN_DA3_LINALG', '').strip().lower()\n"
    "if pref:\n"
    "    if pref == 'auto':\n"
    "        candidates = ['magma', 'cublas']\n"
    "    else:\n"
    "        candidates = [pref]\n"
    "    last_err = None\n"
    "    for name in candidates:\n"
    "        try:\n"
    "            torch.backends.cuda.preferred_linalg_library(name)\n"
    "            sys.stderr.write('[PEMOIN][DA3] Using linalg backend: %s\\n' % name)\n"
    "            last_err = None\n"
    "            break\n"
    "        except Exception as exc:\n"
    "            last_err = exc\n"
    "    if last_err is not None:\n"
    "        raise SystemExit('Failed to set preferred linalg backend: %s' % last_err)\n"
    "target = sys.argv[1]\n"
    "sys.argv = [target, *sys.argv[2:]]\n"
    "if target.endswith('.py') or os.path.exists(target):\n"
    "    runpy.run_path(target, run_name='__main__')\n"
    "else:\n"
    "    runpy.run_module(target, run_name='__main__')\n"
)


def _normalise_export_format(export_format: str) -> tuple[str, str]:
    """
    Legacy helper for DA3 export formats.
    """
    parts: list[str] = []
    for fmt in str(export_format).split("-"):
        if fmt and fmt not in parts:
            parts.append(fmt)

    has_npz = any(fmt in ("mini_npz", "npz") for fmt in parts)
    if not has_npz:
        parts.insert(0, "mini_npz")
    if "glb" not in parts:
        parts.append("glb")

    npz_format = "mini_npz" if "mini_npz" in parts else "npz"
    effective = "-".join(parts)
    return effective, npz_format


def _normalize_da3_mode(value: Any) -> str:
    if value is None:
        return _DA3_MODE_STREAMING
    token = str(value).strip().lower()
    if token in {"streaming", "stream", "da3_streaming"}:
        return _DA3_MODE_STREAMING
    if token in {"standard", "normal", "da3", "classic"}:
        return _DA3_MODE_STANDARD
    raise ValueError(
        "DepthAnything3 mode must be 'streaming' or 'standard' (alias: 'normal')."
    )


def _normalize_extrinsics_convention(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"w2c", "world_to_camera", "world-to-camera", "world2camera", "world2cam"}:
        return "w2c"
    if token in {"c2w", "camera_to_world", "camera-to-world", "camera2world", "cam2world"}:
        return "c2w"
    raise ValueError(
        "DepthAnything3 extrinsics_convention must be 'w2c' or 'c2w' (world_to_camera/camera_to_world)."
    )

def _resolve_npz_format(export_format: str) -> str:
    parts = [part for part in str(export_format).split("-") if part]
    if "mini_npz" in parts:
        return "mini_npz"
    if "npz" in parts:
        return "npz"
    raise ValueError(
        "DepthAnything3 standard mode requires export_format including 'mini_npz' or 'npz'."
    )


@dataclass(frozen=True, slots=True)
class DepthAnything3Settings:
    """Settings controlling how DA3 is invoked and where outputs are stored."""

    input_path: Path
    export_root: Path
    geometry_root: Optional[Path] = None
    cache_root: Optional[Path] = None
    mode: str = _DA3_MODE_STREAMING
    model_dir: str = _DEFAULT_MODEL_DIR
    conda_env: str = "DA3"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    input_max_side: Optional[int] = None
    input_center_crop: bool = False
    device: str = "cuda"
    preferred_linalg_library: Optional[str] = None
    cleanup_export_dir: bool = False
    reuse_exports: bool = True
    run_inference: bool = True
    repo_root: Path = _REPO_ROOT_FALLBACK
    streaming_config: Path = Path("da3_streaming/configs/base_config.yaml")
    max_images: Optional[int] = None
    export_format: str = "mini_npz-glb"
    standard_export_resolution: Optional[tuple[int, int]] = None
    use_ray_pose: bool = True
    ref_view_strategy: str = "middle"
    extrinsics_convention: Optional[str] = None


@dataclass(slots=True)
class DepthAnything3Prediction:
    """In-memory representation of DA3 outputs used by PEMOIN providers."""

    depth: np.ndarray
    conf: Optional[np.ndarray]
    extrinsics: Optional[np.ndarray]
    intrinsics: Optional[np.ndarray]

    @classmethod
    def from_npz(cls, path: Path) -> "DepthAnything3Prediction":
        if not path.exists():
            raise FileNotFoundError(
                f"Depth Anything 3 results not found at '{path}'. "
                "Run the DA3 exporter or enable auto-inference."
            )
        with np.load(path, allow_pickle=True) as data:
            depth = np.asarray(data["depth"])
            conf = np.asarray(data["conf"]) if "conf" in data.files else None
            extrinsics = (
                cls._normalize_extrinsics(np.asarray(data["extrinsics"]))
                if "extrinsics" in data.files
                else None
            )
            intrinsics = np.asarray(data["intrinsics"]) if "intrinsics" in data.files else None
        return cls(depth=depth, conf=conf, extrinsics=extrinsics, intrinsics=intrinsics)

    @staticmethod
    def _normalize_extrinsics(raw: np.ndarray) -> np.ndarray:
        """Ensure extrinsics are padded to 4x4 matrices (w2c or c2w as provided)."""
        arr = np.asarray(raw)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected extrinsics shape {arr.shape}; expected (N, 3/4, 4).")
        if arr.shape[1:] == (3, 4):
            padded = np.concatenate(
                [
                    arr,
                    np.broadcast_to(
                        np.array([0, 0, 0, 1], dtype=arr.dtype),
                        (arr.shape[0], 1, 4),
                    ),
                ],
                axis=1,
            )
            return padded.astype(np.float32)
        if arr.shape[1:] == (4, 4):
            return arr.astype(np.float32)
        raise ValueError(f"Unsupported extrinsics shape {arr.shape}; expected (N,3,4) or (N,4,4).")


class DepthAnything3Runner:
    """Executes DA3 (streaming or standard) inside the specified conda environment."""

    def __init__(self, repo_root: Optional[Path] = None):
        self._repo_root = _resolve_repo_root(repo_root)

    def run(self, settings: DepthAnything3Settings) -> None:
        mode = settings.mode
        if mode == _DA3_MODE_STREAMING:
            self._run_streaming(settings)
            return
        if mode == _DA3_MODE_STANDARD:
            self._run_standard(settings)
            return
        raise ValueError(
            f"Unsupported DepthAnything3 mode '{mode}'. "
            "Expected 'streaming' or 'standard'."
        )

    def _run_streaming(self, settings: DepthAnything3Settings) -> None:
        if not self._repo_root.exists():
            raise FileNotFoundError(
                f"Depth Anything 3 repository not found at '{self._repo_root}'. "
                "Set 'repo_root' in adapter settings or export PEMOIN_REPO_ROOT to the PEMOIN project root."
            )
        if not (self._repo_root / "src" / "depth_anything_3").exists():
            raise FileNotFoundError(
                f"Depth Anything 3 sources not found under '{self._repo_root}'. "
                "Point this to the Depth-Anything-3 repository (with src/depth_anything_3)."
            )
        streaming_dir = self._repo_root / "da3_streaming"
        if not streaming_dir.exists():
            raise FileNotFoundError(
                f"Depth Anything 3 streaming folder not found at '{streaming_dir}'. "
                "Update Depth-Anything-3 or adjust 'repo_root' in adapter settings."
            )
        salad_models = streaming_dir / "loop_utils" / "salad" / "models"
        if not salad_models.exists():
            raise FileNotFoundError(
                "DA3-Streaming submodule 'loop_utils/salad' is missing. "
                "Run 'git submodule update --init --recursive da3_streaming/loop_utils/salad' "
                "inside the Depth-Anything-3 repo."
            )

        config_path = Path(settings.streaming_config).expanduser()
        if not config_path.is_absolute():
            config_path = (self._repo_root / config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"DA3-Streaming config not found at '{config_path}'. "
                "Set 'streaming_config' in adapter settings."
            )

        export_dir = settings.export_root
        export_dir.mkdir(parents=True, exist_ok=True)
        input_path = self._prepare_input(settings)

        script_path = str(streaming_dir / "da3_streaming.py")
        da3_args = [
            "--image_dir",
            str(input_path),
            "--config",
            str(config_path),
            "--output_dir",
            str(export_dir),
        ]

        env = os.environ.copy()
        repo_src = self._repo_root / "src"
        streaming_src = streaming_dir
        alloc_conf = "expandable_segments:True"
        # Match DA3 Gradio app defaults to reduce fragmentation/OOM.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
        env.setdefault("PYTORCH_ALLOC_CONF", alloc_conf)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_src), str(streaming_src), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        preferred = None
        if settings.preferred_linalg_library:
            preferred = str(settings.preferred_linalg_library).strip().lower()
        if preferred and str(settings.device).lower().startswith("cuda"):
            env["PEMOIN_DA3_LINALG"] = preferred
            cmd = [
                "conda",
                "run",
                "-n",
                settings.conda_env,
                "python",
                "-c",
                _DA3_LINALG_WRAPPER,
                script_path,
                *da3_args,
            ]
        else:
            cmd = [
                "conda",
                "run",
                "-n",
                settings.conda_env,
                "python",
                script_path,
                *da3_args,
            ]
        subprocess.run(cmd, check=True, cwd=streaming_dir, env=env)

    def _run_standard(self, settings: DepthAnything3Settings) -> None:
        if not self._repo_root.exists():
            raise FileNotFoundError(
                f"Depth Anything 3 repository not found at '{self._repo_root}'. "
                "Set 'repo_root' in adapter settings or export PEMOIN_REPO_ROOT to the PEMOIN project root."
            )
        if not (self._repo_root / "src" / "depth_anything_3").exists():
            raise FileNotFoundError(
                f"Depth Anything 3 sources not found under '{self._repo_root}'. "
                "Point this to the Depth-Anything-3 repository (with src/depth_anything_3)."
            )

        _resolve_npz_format(settings.export_format)
        export_dir = settings.export_root
        export_dir.mkdir(parents=True, exist_ok=True)
        input_path = self._prepare_input(settings)

        da3_args = [
            "auto",
            str(input_path),
            "--model-dir",
            settings.model_dir,
            "--export-dir",
            str(export_dir),
            "--export-format",
            settings.export_format,
            "--device",
            settings.device,
            "--process-res",
            str(settings.process_res),
            "--process-res-method",
            settings.process_res_method,
            "--ref-view-strategy",
            settings.ref_view_strategy,
            "--auto-cleanup",
        ]
        if settings.use_ray_pose:
            da3_args.append("--use-ray-pose")

        env = os.environ.copy()
        repo_src = self._repo_root / "src"
        alloc_conf = "expandable_segments:True"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
        env.setdefault("PYTORCH_ALLOC_CONF", alloc_conf)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_src), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        preferred = None
        if settings.preferred_linalg_library:
            preferred = str(settings.preferred_linalg_library).strip().lower()
        if preferred and str(settings.device).lower().startswith("cuda"):
            env["PEMOIN_DA3_LINALG"] = preferred
            cmd = [
                "conda",
                "run",
                "-n",
                settings.conda_env,
                "python",
                "-c",
                _DA3_LINALG_WRAPPER,
                "depth_anything_3.cli",
                *da3_args,
            ]
        else:
            cmd = [
                "conda",
                "run",
                "-n",
                settings.conda_env,
                "python",
                "-m",
                "depth_anything_3.cli",
                *da3_args,
            ]
        subprocess.run(cmd, check=True, cwd=self._repo_root, env=env)

    def _prepare_input(self, settings: DepthAnything3Settings) -> Path:
        """
        Optionally stage a limited number of frames and/or preprocess inputs for DA3.
        """
        limit_images = settings.max_images
        preprocess = settings.input_max_side is not None or settings.input_center_crop
        source = settings.input_path
        if source.is_dir():
            images = _discover_images(source)
            if not images:
                raise FileNotFoundError(f"No images found under '{source}' for DA3 staging.")
            needs_format_stage = any(
                path.suffix.lower() not in _STREAMING_EXTENSIONS for path in images
            )
            if limit_images is None and not preprocess and not needs_format_stage:
                return source

        # Stage inputs outside the export root to avoid auto-cleanup wiping inputs.
        base_parent = settings.export_root.parent
        staging_dir = base_parent / f"da3_input_{settings.export_root.name}"
        _clear_directory(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            images = _discover_images(source)
            subset = images[:limit_images] if limit_images is not None else images
            if not subset:
                raise FileNotFoundError(f"No images found under '{source}' for DA3 staging.")
            for idx, path in enumerate(subset):
                target = staging_dir / f"{idx:06d}.png"
                _copy_with_preprocess(
                    path,
                    target,
                    max_side=settings.input_max_side,
                    center_crop=settings.input_center_crop,
                )
            return staging_dir

        if source.is_file():
            if source.suffix.lower() in _VIDEO_EXTENSIONS:
                self._extract_video_frames(
                    source,
                    staging_dir,
                    limit_images,
                    settings.input_max_side,
                    settings.input_center_crop,
                )
                return staging_dir
            if source.suffix.lower() in _SUPPORTED_EXTENSIONS:
                target = staging_dir / "000000.png"
                _copy_with_preprocess(
                    source,
                    target,
                    max_side=settings.input_max_side,
                    center_crop=settings.input_center_crop,
                )
                return staging_dir

        raise ValueError(
            f"Unsupported DA3 input '{source}'. Provide a directory of images, a video, or an image file."
        )

    @staticmethod
    def _extract_video_frames(
        video_path: Path,
        output_dir: Path,
        max_frames: Optional[int],
        max_side: Optional[int],
        center_crop: bool,
    ) -> None:
        cv2 = _ensure_cv2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video '{video_path}' for DA3 staging.")

        saved = 0
        index = 0
        while max_frames is None or saved < max_frames:
            success, frame = cap.read()
            if not success or frame is None:
                break
            frame = _preprocess_frame(frame, max_side=max_side, center_crop=center_crop)
            output_path = output_dir / f"{index:06d}.png"
            cv2.imwrite(str(output_path), frame)
            saved += 1
            index += 1

        cap.release()
        if saved == 0:
            raise RuntimeError(f"No frames extracted from '{video_path}' for DA3 staging.")


class DepthAnything3Client:
    """Loads DA3 outputs and exposes them through PEMOIN data contracts."""

    def __init__(
        self,
        settings: DepthAnything3Settings,
        *,
        runner_factory: Optional[Callable[[Optional[Path]], DepthAnything3Runner]] = None,
    ):
        self.settings = settings
        self._runner_factory = runner_factory or DepthAnything3Runner
        self._prediction: Optional[DepthAnything3Prediction] = None
        self._image_order: List[Path] = []
        self._index_by_path: dict[str, int] = {}
        self._frame_count: Optional[int] = None
        self._cached_signature: Optional[str] = None
        self._active_export_root: Optional[Path] = None
        self._cleaned_cam_c2w: Optional[np.ndarray] = None
        self._cleaned_cam_w2c: Optional[np.ndarray] = None
        self._cleanup_signature: Optional[str] = None
        self._cleanup_metadata: MutableMapping[str, Any] = {}

    def initialise(self) -> None:
        if self._prediction is not None:
            return
        self._image_order = self._discover_image_order(self.settings.input_path)
        self._index_by_path = self._build_index(self._image_order)
        self._frame_count = self._estimate_frame_count(self.settings.input_path, self._image_order)
        prediction = self._load_or_infer()
        self._prediction = self._standardize_prediction(prediction)
        # When inputs were staged (e.g., video extraction), align indices to prediction length.
        if self._prediction is not None and not self._image_order:
            self._image_order = [Path(f"frame_{i:04d}.png") for i in range(self._prediction.depth.shape[0])]
            self._index_by_path = self._build_index(self._image_order)

    def estimate_depth(
        self, frames: Iterable[Any], options: Mapping[str, Any]
    ) -> DepthData | List[DepthData]:
        self._ensure_ready()
        indices = self._map_frame_indices(frames)
        assert self._prediction is not None  # for type checkers
        results: List[DepthData] = []
        for clamped_index, output_index, requested in indices:
            depth = self._prediction.depth[clamped_index]
            conf = None
            if self._prediction.conf is not None and clamped_index < self._prediction.conf.shape[0]:
                conf = self._prediction.conf[clamped_index]
            metadata = self._settings_metadata()
            metadata.update(
                {
                    "export_root": str(self._export_root()),
                    "input_path": str(self.settings.input_path),
                    "options": dict(options),
                    "metric_depth": True,
                    "input_max_side": self.settings.input_max_side,
                    "input_center_crop": self.settings.input_center_crop,
                    "camera_convention": "blender",
                    "source_camera_convention": "opencv",
                }
            )
            if requested is not None and requested != clamped_index:
                metadata["index_clamped_from"] = requested
            if output_index is not None and requested is not None and requested != output_index:
                metadata["source_frame_index"] = requested

            frame_index = output_index if output_index is not None else requested
            results.append(
                DepthData(
                    frame_index=frame_index,
                    depth=np.asarray(depth, dtype=np.float32),
                    confidence=np.asarray(conf, dtype=np.float32) if conf is not None else None,
                    metadata=metadata,
                )
            )
        return results[0] if len(results) == 1 else results

    def estimate_trajectory(
        self,
        frames: Iterable[Any],
        options: Mapping[str, Any],
        metadata: MutableMapping[str, Any] | None = None,
    ) -> PoseData | List[PoseData]:
        self._ensure_ready()
        indices = self._map_frame_indices(frames)
        assert self._prediction is not None
        extrinsics = self._prediction.extrinsics
        if extrinsics is None:
            raise RuntimeError("Depth Anything 3 prediction does not contain extrinsics.")

        diagnostics = metadata if metadata is not None else {}
        for key, value in self._settings_metadata().items():
            diagnostics.setdefault(key, value)

        cam_c2w_all, cam_w2c_all, cleanup_md = self._clean_prediction_trajectory(extrinsics, options)
        if cleanup_md:
            diagnostics.update(cleanup_md)

        results: List[PoseData] = []
        for clamped_index, output_index, requested in indices:
            pose_metadata = self._pose_metadata(
                output_index if output_index is not None else requested, options
            )
            if requested is not None and requested != clamped_index:
                pose_metadata["index_clamped_from"] = requested
            if output_index is not None and requested is not None and requested != output_index:
                pose_metadata["source_frame_index"] = requested
            if cleanup_md:
                pose_metadata.update(cleanup_md)
            w2c = cam_w2c_all[clamped_index]
            c2w = cam_c2w_all[clamped_index]
            c2w, w2c = convert_pose_opencv_to_blender(c2w, w2c)
            pose_metadata["camera_convention"] = "blender"
            pose_metadata["pose_coordinate_system"] = "blender"
            pose_metadata["source_camera_convention"] = "opencv"
            sample = PoseSample(
                frame_index=output_index if output_index is not None else requested,
                camera_to_world=c2w.astype(np.float32),
                world_to_camera=w2c.astype(np.float32),
                confidence=None,
                metadata=pose_metadata,
            )
            results.append(PoseData(samples=[sample], metadata=dict(diagnostics)))
        return results[0] if len(results) == 1 else results

    def fetch_intrinsics(self) -> IntrinsicsData:
        self._ensure_ready()
        assert self._prediction is not None
        intrinsics = self._prediction.intrinsics
        if intrinsics is None:
            raise RuntimeError("Depth Anything 3 prediction does not contain intrinsics.")
        first = np.asarray(intrinsics[0], dtype=np.float32)
        dynamic = self._intrinsics_are_dynamic(intrinsics)
        metadata: MutableMapping[str, Any] = self._settings_metadata()
        reference_resolution = None
        try:
            reference_resolution = tuple(int(dim) for dim in intrinsics.shape[1:3])
        except Exception:
            reference_resolution = None
        metadata.update(
            {
                "dynamic": dynamic,
                "input_path": str(self.settings.input_path),
                "export_root": str(self._export_root()),
                "reference_resolution": reference_resolution,
                "camera_convention": "blender",
                "source_camera_convention": "opencv",
            }
        )
        return IntrinsicsData(matrix=first, metadata=metadata)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_ready(self) -> None:
        if self._prediction is None:
            self.initialise()
        if self._prediction is None:
            raise RuntimeError("Depth Anything 3 client failed to initialise.")

    def _export_root(self) -> Path:
        return self._active_export_root or self.settings.export_root

    def _settings_metadata(self) -> MutableMapping[str, Any]:
        metadata: MutableMapping[str, Any] = {
            "source": "DepthAnything3",
            "model_dir": self.settings.model_dir,
            "da3_mode": self.settings.mode,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        metadata["extrinsics_convention"] = "w2c" if self._extrinsics_are_w2c() else "c2w"
        if self.settings.mode == _DA3_MODE_STREAMING:
            metadata["streaming_config"] = str(self.settings.streaming_config)
        else:
            metadata["export_format"] = str(self.settings.export_format)
            metadata["process_res"] = self.settings.process_res
            metadata["process_res_method"] = self.settings.process_res_method
            metadata["use_ray_pose"] = self.settings.use_ray_pose
            metadata["ref_view_strategy"] = self.settings.ref_view_strategy
        return metadata

    def _extrinsics_are_w2c(self) -> bool:
        return self._prediction_extrinsics_convention() == "w2c"

    def _extrinsics_to_w2c(self, extrinsics: np.ndarray) -> np.ndarray:
        if self._extrinsics_are_w2c():
            return np.asarray(extrinsics, dtype=np.float32)
        cam_c2w = np.asarray(extrinsics, dtype=np.float64)
        cam_w2c = self._invert_extrinsics(cam_c2w, label="camera-to-world")
        return cam_w2c.astype(np.float32)

    def _prediction_extrinsics_convention(self) -> str:
        raw = _normalize_extrinsics_convention(self.settings.extrinsics_convention)
        if raw is not None:
            return raw
        if self.settings.mode == _DA3_MODE_STREAMING:
            return "w2c"
        if self.settings.use_ray_pose:
            return "c2w"
        return "w2c"

    def _clean_prediction_trajectory(
        self, extrinsics_w2c: np.ndarray, options: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, MutableMapping[str, Any]]:
        cleanup_options = TrajectoryCleanupOptions.from_mapping(options)
        if not cleanup_options.enabled:
            self._cleanup_signature = None
            self._cleaned_cam_c2w = None
            self._cleaned_cam_w2c = None
            self._cleanup_metadata = {}
            if self._extrinsics_are_w2c():
                cam_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
                cam_c2w = self._invert_extrinsics(cam_w2c, label="world-to-camera").astype(np.float32)
            else:
                cam_c2w = np.asarray(extrinsics_w2c, dtype=np.float32)
                cam_w2c = self._invert_extrinsics(cam_c2w, label="camera-to-world").astype(np.float32)
            return cam_c2w, cam_w2c, {}
        signature = cleanup_options.signature()
        if signature == self._cleanup_signature and self._cleaned_cam_c2w is not None and self._cleaned_cam_w2c is not None:
            return self._cleaned_cam_c2w, self._cleaned_cam_w2c, dict(self._cleanup_metadata)
        if self._extrinsics_are_w2c():
            cam_w2c = np.asarray(extrinsics_w2c, dtype=np.float64)
            cam_c2w = self._invert_extrinsics(cam_w2c, label="world-to-camera")
        else:
            cam_c2w = np.asarray(extrinsics_w2c, dtype=np.float64)
            cam_w2c = self._invert_extrinsics(cam_c2w, label="camera-to-world")
        cleaned_c2w, cleanup_md = cleanup_camera_to_world(cam_c2w, cleanup_options)
        cleaned_w2c = np.linalg.inv(cleaned_c2w).astype(np.float32)
        self._cleaned_cam_c2w = cleaned_c2w
        self._cleaned_cam_w2c = cleaned_w2c
        self._cleanup_signature = signature
        self._cleanup_metadata = dict(cleanup_md)
        return cleaned_c2w, cleaned_w2c, dict(cleanup_md)

    def _load_or_infer(self) -> DepthAnything3Prediction:
        signature = self._export_signature()
        self._cached_signature = signature
        export_root = self._resolve_export_root(signature)
        self._active_export_root = export_root
        if (
            self.settings.reuse_exports
            and self._outputs_ready(export_root)
            and self._metadata_matches(signature, export_root)
        ):
            LOG.info(
                "Reusing existing DepthAnything3 %s exports at '%s' (signature match).",
                self.settings.mode,
                export_root,
                extra={"summary": True},
            )
            prediction = self._load_prediction(export_root)
            self._log_extrinsics_convention(prediction)
            self._write_standard_export(prediction, export_root)
            return prediction
        if not self.settings.run_inference:
            raise FileNotFoundError(
                f"Depth Anything 3 exports missing for mode '{self.settings.mode}' "
                f"under '{export_root}' and auto-inference is disabled."
            )

        if export_root.exists():
            if self.settings.cleanup_export_dir or not self.settings.reuse_exports or any(export_root.iterdir()):
                _clear_directory(export_root)
        export_root.mkdir(parents=True, exist_ok=True)

        prediction = self._run_single_inference(export_root)
        self._write_export_metadata(signature, export_root)
        self._log_extrinsics_convention(prediction)
        self._write_standard_export(prediction, export_root)
        return prediction

    def _run_single_inference(self, export_root: Path) -> DepthAnything3Prediction:
        runner = self._runner_factory(self.settings.repo_root or None)
        run_settings = self.settings
        if export_root != self.settings.export_root:
            run_settings = replace(self.settings, export_root=export_root)
        runner.run(run_settings)
        return self._load_prediction(export_root)

    def _outputs_ready(self, export_root: Path) -> bool:
        if self.settings.mode == _DA3_MODE_STREAMING:
            return self._streaming_outputs_ready(export_root)
        if self.settings.mode == _DA3_MODE_STANDARD:
            return self._standard_outputs_ready(export_root)
        raise ValueError(f"Unsupported DepthAnything3 mode '{self.settings.mode}'.")

    def _load_prediction(self, export_root: Path) -> DepthAnything3Prediction:
        if self.settings.mode == _DA3_MODE_STREAMING:
            return self._load_streaming_prediction(export_root)
        if self.settings.mode == _DA3_MODE_STANDARD:
            return self._load_standard_prediction(export_root)
        raise ValueError(f"Unsupported DepthAnything3 mode '{self.settings.mode}'.")

    def _streaming_results_dir(self, export_root: Optional[Path] = None) -> Path:
        base = export_root if export_root is not None else self._export_root()
        return base / "results_output"

    def _streaming_outputs_ready(self, export_root: Path) -> bool:
        results_dir = self._streaming_results_dir(export_root)
        if not results_dir.exists():
            return False
        if not any(results_dir.glob("frame_*.npz")):
            return False
        return (export_root / "camera_poses.txt").exists() and (export_root / "intrinsic.txt").exists()

    def _standard_npz_path(self, export_root: Optional[Path] = None) -> Path:
        root = export_root if export_root is not None else self._export_root()
        fmt = _resolve_npz_format(self.settings.export_format)
        return root / "exports" / fmt / "results.npz"

    def _standard_outputs_ready(self, export_root: Path) -> bool:
        return self._standard_npz_path(export_root).exists()

    def _load_standard_prediction(self, export_root: Path) -> DepthAnything3Prediction:
        npz_path = self._standard_npz_path(export_root)
        return DepthAnything3Prediction.from_npz(npz_path)

    def _load_streaming_prediction(self, export_root: Path) -> DepthAnything3Prediction:
        results_dir = self._streaming_results_dir(export_root)
        if not results_dir.exists():
            raise FileNotFoundError(
                f"DA3-Streaming outputs not found under '{results_dir}'. "
                "Ensure Model.save_depth_conf_result is enabled in the streaming config."
            )

        npz_files = _discover_streaming_npz(results_dir)
        if not npz_files:
            raise FileNotFoundError(
                f"No frame_*.npz files found under '{results_dir}'. "
                "Ensure Model.save_depth_conf_result is enabled in the streaming config."
            )

        depths: List[np.ndarray] = []
        conf_layers: List[np.ndarray] = []
        intr_layers: List[np.ndarray] = []
        missing_conf = False
        missing_intr = False

        for path in npz_files:
            with np.load(path, allow_pickle=True) as data:
                if "depth" not in data.files:
                    raise KeyError(f"Missing 'depth' in streaming output '{path}'.")
                depths.append(np.asarray(data["depth"], dtype=np.float32))
                if "conf" in data.files:
                    conf_layers.append(np.asarray(data["conf"], dtype=np.float32))
                else:
                    missing_conf = True
                if "intrinsics" in data.files:
                    intr_layers.append(np.asarray(data["intrinsics"], dtype=np.float32))
                else:
                    missing_intr = True

        if missing_conf and conf_layers:
            raise ValueError("Streaming outputs have inconsistent confidence data; rerun with save_depth_conf_result.")
        if missing_intr and intr_layers:
            raise ValueError("Streaming outputs have inconsistent intrinsics data; rerun with save_depth_conf_result.")

        depth_stack = np.stack(depths, axis=0)
        conf_stack = None if missing_conf else np.stack(conf_layers, axis=0)
        intr_stack = None if missing_intr else np.stack(intr_layers, axis=0)

        pose_path = export_root / "camera_poses.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"DA3-Streaming camera poses not found at '{pose_path}'.")
        c2w = _load_camera_poses(pose_path)
        if c2w.shape[0] != depth_stack.shape[0]:
            raise ValueError(
                "DA3-Streaming outputs have mismatched frame counts between depth and camera poses."
            )
        w2c = np.linalg.inv(c2w).astype(np.float32)

        if intr_stack is None:
            intr_path = export_root / "intrinsic.txt"
            if not intr_path.exists():
                raise FileNotFoundError(f"DA3-Streaming intrinsics not found at '{intr_path}'.")
            intr_stack = _load_intrinsics_txt(intr_path)
        if intr_stack.shape[0] != depth_stack.shape[0]:
            raise ValueError(
                "DA3-Streaming outputs have mismatched frame counts between depth and intrinsics."
            )

        return DepthAnything3Prediction(
            depth=depth_stack,
            conf=conf_stack,
            extrinsics=w2c,
            intrinsics=intr_stack,
        )

    def _export_npz_path(self, export_root: Optional[Path] = None) -> Path:
        _, fmt = _normalise_export_format(self.settings.export_format or "mini_npz")
        root = export_root if export_root is not None else self._export_root()
        return root / "exports" / fmt / "results.npz"

    def _persist_prediction(self, prediction: DepthAnything3Prediction, npz_path: Path) -> None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "depth": np.asarray(prediction.depth, dtype=np.float32),
        }
        if prediction.conf is not None:
            payload["conf"] = np.asarray(prediction.conf, dtype=np.float32)
        if prediction.extrinsics is not None:
            payload["extrinsics"] = np.asarray(prediction.extrinsics, dtype=np.float32)
        if prediction.intrinsics is not None:
            payload["intrinsics"] = np.asarray(prediction.intrinsics, dtype=np.float32)
        np.savez_compressed(npz_path, **payload)

    def _standardize_prediction(self, prediction: DepthAnything3Prediction) -> DepthAnything3Prediction:
        """
        Optionally resize depth/confidence/intrinsics to a standard resolution for downstream consumers.
        """
        target_shape = self.settings.standard_export_resolution
        if target_shape is None or len(target_shape) != 2:
            return prediction
        depths, conf, intrinsics = _resize_to_target_resolution(
            prediction.depth,
            prediction.conf,
            prediction.intrinsics,
            target_shape=tuple(int(dim) for dim in target_shape),
        )
        return DepthAnything3Prediction(
            depth=depths,
            conf=conf,
            extrinsics=prediction.extrinsics,
            intrinsics=intrinsics,
        )

    def _normalize_for_signature(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, Mapping):
            return {str(k): self._normalize_for_signature(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._normalize_for_signature(v) for v in value]
        return value

    def _signature_payload(self) -> dict[str, Any]:
        payload = asdict(self.settings)
        for key in (
            "export_root",
            "geometry_root",
            "cache_root",
            "cleanup_export_dir",
            "reuse_exports",
            "run_inference",
            "repo_root",
        ):
            payload.pop(key, None)
        return payload

    def _resolve_export_root(self, signature: str) -> Path:
        if self.settings.cache_root is None:
            return self.settings.export_root
        return self.settings.cache_root / signature

    def _export_metadata_path(self, root: Optional[Path] = None) -> Path:
        base = root if root is not None else self._export_root()
        return base / "exports" / "metadata.json"

    def _input_digest(self) -> str:
        names = [p.name for p in self._image_order] if self._image_order else []
        payload = "\n".join(names).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _export_signature(self) -> str:
        payload = self._signature_payload()
        payload = self._normalize_for_signature(payload)
        payload["input_digest"] = self._input_digest()
        payload["frame_count"] = self._frame_count
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _metadata_matches(self, signature: str, root: Optional[Path] = None) -> bool:
        meta_path = self._export_metadata_path(root)
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if meta.get("signature") != signature:
            return False
        if "frame_count" in meta and self._frame_count is not None:
            try:
                if int(meta["frame_count"]) != int(self._frame_count):
                    return False
            except Exception:
                return False
        return True

    def _write_export_metadata(self, signature: str, root: Optional[Path] = None) -> None:
        meta_path = self._export_metadata_path(root)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": signature,
            "frame_count": self._frame_count,
            "input_digest": self._input_digest(),
            "settings": self._normalize_for_signature(self._signature_payload()),
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _estimate_frame_count(self, source: Path, discovered: Sequence[Path]) -> Optional[int]:
        """
        Infer the total number of frames available in the input.
        """
        if source.is_dir():
            return len(discovered)
        if source.is_file():
            suffix = source.suffix.lower()
            if suffix in _VIDEO_EXTENSIONS:
                count = self._probe_video_frame_count(source)
                return count if count is not None else len(discovered)
            if suffix in _SUPPORTED_EXTENSIONS:
                return max(1, len(discovered))
        return len(discovered) if discovered else None

    def _probe_video_frame_count(self, video_path: Path) -> Optional[int]:
        try:
            cv2 = _ensure_cv2()
        except ImportError:
            return None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return count if count > 0 else None

    def _discover_image_order(self, root: Path) -> List[Path]:
        if not root.exists():
            raise FileNotFoundError(f"Depth Anything 3 input path '{root}' does not exist.")
        if root.is_file():
            return [root]
        candidates: List[Path] = []
        for ext in _SUPPORTED_EXTENSIONS:
            candidates.extend(root.glob(f"*{ext}"))
            candidates.extend(root.glob(f"*{ext.upper()}"))
        images = sorted({path.resolve() for path in candidates})
        if not images:
            raise FileNotFoundError(
                f"No images with supported extensions {', '.join(_SUPPORTED_EXTENSIONS)} found in '{root}'."
            )
        return list(images)

    @staticmethod
    def _build_index(images: Sequence[Path]) -> dict[str, int]:
        index: dict[str, int] = {}
        for idx, path in enumerate(images):
            index[str(path)] = idx
            index[str(path.resolve())] = idx
            index[path.name] = idx
        return index

    def _map_frame_indices(self, frames: Iterable[Any]) -> List[tuple[int, Optional[int], Optional[int]]]:
        """
        Resolve runtime frames to DA3 prediction indices.

        If the requested index is outside the available range, clamp to the nearest valid index
        and record the original index for diagnostics.
        """
        if self._prediction is None:
            raise RuntimeError("DA3 prediction is not available.")

        max_idx = int(self._prediction.depth.shape[0] - 1)
        indices: List[tuple[int, Optional[int], Optional[int]]] = []
        for frame in frames:
            output_index: Optional[int] = None
            if hasattr(frame, "index"):
                output_index = int(getattr(frame, "index"))
            requested = self._resolve_frame_index(frame)
            clamped = min(max(requested, 0), max_idx)
            indices.append((clamped, output_index, requested))
        if not indices:
            raise ValueError("No frame references supplied to Depth Anything 3 provider.")
        return indices

    def _resolve_frame_index(self, frame: Any) -> int:
        path_hint = None
        if hasattr(frame, "metadata"):
            metadata = getattr(frame, "metadata", {}) or {}
            source_index = metadata.get("source_frame_index")
            if source_index is not None:
                try:
                    return int(source_index)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"source_frame_index must be an integer, got {source_index!r}.") from exc
            path_hint = metadata.get("source_path")
        if path_hint:
            hit = self._index_by_path.get(str(Path(path_hint).resolve()))
            if hit is None:
                hit = self._index_by_path.get(Path(path_hint).name)
            if hit is not None:
                return hit
        if hasattr(frame, "frame_id"):
            hit = self._index_by_path.get(str(getattr(frame, "frame_id")))
            if hit is not None:
                return hit
        if hasattr(frame, "index"):
            return int(getattr(frame, "index"))
        raise TypeError("Frame must expose 'source_frame_index', 'index', or a resolvable source path.")

    @staticmethod
    def _intrinsics_are_dynamic(intrinsics: np.ndarray) -> bool:
        arr = np.asarray(intrinsics)
        if arr.ndim != 3 or arr.shape[0] <= 1:
            return False
        first = arr[0]
        return not np.allclose(first, arr[1:])

    def _pose_metadata(self, frame_index: int, options: Mapping[str, Any]) -> MutableMapping[str, Any]:
        metadata = self._settings_metadata()
        metadata.update(
            {
                "frame_index": frame_index,
                "options": dict(options),
                "input_max_side": self.settings.input_max_side,
                "input_center_crop": self.settings.input_center_crop,
            }
        )
        return metadata

    def _write_standard_export(
        self, prediction: DepthAnything3Prediction, export_root: Optional[Path] = None
    ) -> None:
        raw_root = export_root if export_root is not None else self._export_root()
        target_dir = self.settings.geometry_root or (raw_root / "geometry")
        pcd_source = raw_root / "pcd" / "combined_pcd.ply"
        if pcd_source.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pcd_source, target_dir / "combined_pcd.ply")
        depths = prediction.depth
        conf = prediction.conf if prediction.conf is not None else None
        intrinsics = prediction.intrinsics
        extrinsics = prediction.extrinsics
        target_shape = self.settings.standard_export_resolution
        if target_shape is not None and len(target_shape) == 2:
            depths, conf, intrinsics = _resize_to_target_resolution(
                depths,
                conf,
                intrinsics,
                target_shape=tuple(int(dim) for dim in target_shape),
            )
        if intrinsics is None or extrinsics is None:
            return
        extrinsics_w2c = self._extrinsics_to_w2c(extrinsics)
        save_standard_geometry(
            target_dir,
            source="DepthAnything3",
            depths=depths,
            confidence=conf,
            intrinsics=intrinsics,
            extrinsics_w2c=extrinsics_w2c,
            frame_ids=range(depths.shape[0]),
            target_shape=self.settings.standard_export_resolution,
            source_camera_convention="opencv",
        )

    def _log_extrinsics_convention(self, prediction: DepthAnything3Prediction) -> None:
        if prediction.extrinsics is None:
            LOG.debug("[DA3] Extrinsics missing; skipping convention log.")
            return
        raw = _normalize_extrinsics_convention(self.settings.extrinsics_convention)
        if raw is not None:
            reason = "settings override"
        elif self.settings.mode == _DA3_MODE_STREAMING:
            reason = "DA3-Streaming outputs w2c; camera_poses.txt is c2w and inverted"
        elif self.settings.use_ray_pose:
            reason = "DA3 ray pose outputs c2w (model converts w2c -> c2w)"
        else:
            reason = "DA3 camera decoder outputs w2c"
        convention = self._prediction_extrinsics_convention()
        LOG.debug(
            "[DA3] Extrinsics convention: %s (mode=%s use_ray_pose=%s reason=%s).",
            convention,
            self.settings.mode,
            self.settings.use_ray_pose,
            reason,
        )

    @staticmethod
    def _invert_extrinsics(extrinsics: np.ndarray, *, label: str) -> np.ndarray:
        try:
            return np.linalg.inv(extrinsics)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"DepthAnything3 {label} extrinsics are non-invertible; check the pose convention."
            ) from exc


def _scale_intrinsics(intrinsics: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Scale intrinsics to account for image resizing.
    """
    intr = np.asarray(intrinsics, dtype=np.float32).copy()
    if intr.ndim == 2:
        intr = np.broadcast_to(intr, (1, *intr.shape)).copy()
    intr[..., 0, 0] *= scale_x
    intr[..., 1, 1] *= scale_y
    intr[..., 0, 2] *= scale_x
    intr[..., 1, 2] *= scale_y
    return intr


def _resize_to_target_resolution(
    depths: np.ndarray,
    confidence: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    *,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Resize depth/confidence to a target (height, width) while scaling intrinsics accordingly.
    """
    depths_arr = np.asarray(depths)
    if depths_arr.ndim != 3:
        return depths_arr, confidence, intrinsics
    target_h, target_w = target_shape
    src_h, src_w = depths_arr.shape[1:3]
    if (src_h, src_w) == (target_h, target_w):
        return depths_arr, confidence, intrinsics

    cv2 = _ensure_cv2()
    resized_depths = np.stack(
        [cv2.resize(layer, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for layer in depths_arr]
    )
    # Depth interpolation can introduce small negative values; clamp to keep geometry valid.
    resized_depths = np.clip(resized_depths, a_min=1e-6, a_max=None)
    resized_conf = None
    if confidence is not None:
        conf_arr = np.asarray(confidence)
        resized_conf = np.stack(
            [cv2.resize(layer, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for layer in conf_arr]
        )
    scaled_intrinsics = None
    if intrinsics is not None:
        scale_x = float(target_w) / float(src_w)
        scale_y = float(target_h) / float(src_h)
        scaled_intrinsics = _scale_intrinsics(np.asarray(intrinsics), scale_x, scale_y)
    return resized_depths, resized_conf, scaled_intrinsics


def _preprocess_frame(frame: np.ndarray, *, max_side: Optional[int], center_crop: bool) -> np.ndarray:
    """
    Downscale and optionally center-crop a frame prior to DA3 ingestion.
    """
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return arr

    if center_crop:
        h, w = arr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        arr = arr[y0 : y0 + side, x0 : x0 + side]

    if max_side is not None and max_side > 0:
        h, w = arr.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = float(max_side) / float(longest)
            target_w = max(1, int(round(w * scale)))
            target_h = max(1, int(round(h * scale)))
            cv2 = _ensure_cv2()
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return arr


def _copy_with_preprocess(
    source: Path, target: Path, *, max_side: Optional[int], center_crop: bool
) -> None:
    """
    Copy an image to target with optional pre-resize/center-crop applied.
    """
    if max_side is None and not center_crop and source.suffix.lower() == target.suffix.lower():
        shutil.copy2(source, target)
        return

    cv2 = _ensure_cv2()
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image '{source}' for DA3 preprocessing.")
    processed = _preprocess_frame(image, max_side=max_side, center_crop=center_crop)
    cv2.imwrite(str(target), processed)


class DepthAnything3Adapter:
    """Bridges DA3 outputs into PEMOIN providers."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        client_factory: Optional[Callable[[DepthAnything3Settings], DepthAnything3Client]] = None,
    ):
        self._settings = _coerce_settings(settings)
        self._client_factory = client_factory or DepthAnything3Client
        self._client: Optional[DepthAnything3Client] = None

    @property
    def client(self) -> DepthAnything3Client:
        if self._client is None:
            self._client = self._client_factory(self._settings)
        self._client.initialise()
        return self._client

    def create_depth_provider(self, inference_options: Mapping[str, Any]) -> "DepthAnything3DepthProvider":
        return DepthAnything3DepthProvider(client=self.client, inference_options=dict(inference_options))

    def create_trajectory_provider(
        self, inference_options: Mapping[str, Any]
    ) -> "DepthAnything3TrajectoryProvider":
        return DepthAnything3TrajectoryProvider(client=self.client, inference_options=dict(inference_options))

    def create_intrinsics_provider(
        self, inference_options: Mapping[str, Any]
    ) -> "DepthAnything3IntrinsicsProvider":
        return DepthAnything3IntrinsicsProvider(client=self.client, inference_options=dict(inference_options))


class _DepthAnything3ProviderBase:
    def __init__(self, client: DepthAnything3Client, inference_options: Mapping[str, Any]):
        self._client = client
        self._inference_options = dict(inference_options)

    def setup(self, context: MutableMapping[str, Any]):
        self._client.initialise()
        self._working_resolution = context.get("working_resolution")

    def teardown(self):
        pass


class DepthAnything3DepthProvider(_DepthAnything3ProviderBase, DepthProvider):
    def process(self, frame: Any):
        frames: Iterable[Any] = [frame]
        return self._client.estimate_depth(frames, self._inference_options)


class DepthAnything3TrajectoryProvider(_DepthAnything3ProviderBase, TrajectoryProvider):
    def process(self, frame: Any):
        frames: Iterable[Any] = [frame]
        metadata: MutableMapping[str, Any] = {}
        return self._client.estimate_trajectory(frames, self._inference_options, metadata)


class DepthAnything3IntrinsicsProvider(_DepthAnything3ProviderBase, IntrinsicsProvider):
    def process(self, frame: Any):
        intrinsics = self._client.fetch_intrinsics()
        return self._scale_intrinsics(intrinsics, frame)


def _coerce_settings(raw: Mapping[str, Any]) -> DepthAnything3Settings:
    base_root = Path(os.environ.get("PEMOIN_REPO_ROOT", ".")).expanduser().resolve()
    if "max_frame_count" in raw:
        raise ValueError(
            "DepthAnything3 batching is no longer supported; remove 'max_frame_count' from settings."
        )
    try:
        input_path_raw = Path(str(raw["input_path"])).expanduser()
        export_root_raw = Path(str(raw["export_root"])).expanduser()
    except KeyError as exc:
        raise ValueError("DepthAnything3 adapter requires 'input_path' and 'export_root'.") from exc
    input_path = input_path_raw if input_path_raw.is_absolute() else (base_root / input_path_raw)
    export_root = export_root_raw if export_root_raw.is_absolute() else (base_root / export_root_raw)
    input_path = input_path.resolve()
    export_root = export_root.resolve()
    geometry_root_raw = Path(str(raw["geometry_root"])).expanduser() if "geometry_root" in raw else None
    if geometry_root_raw is not None and not geometry_root_raw.is_absolute():
        geometry_root_raw = (base_root / geometry_root_raw).resolve()
    repo_root_raw = Path(str(raw["repo_root"])).expanduser() if "repo_root" in raw else None
    if repo_root_raw is not None and not repo_root_raw.is_absolute():
        repo_root_raw = (base_root / repo_root_raw).resolve()
    repo_root = _resolve_repo_root(repo_root_raw)
    cache_root_raw = raw.get("cache_root")
    if cache_root_raw is None:
        cache_root_raw = raw.get("cache_dir")
    cache_root = None
    if cache_root_raw:
        cache_root = Path(str(cache_root_raw)).expanduser()
        if not cache_root.is_absolute():
            cache_root = (base_root / cache_root).resolve()
    mode = _normalize_da3_mode(raw.get("mode") or raw.get("da3_mode"))

    streaming_config_raw = raw.get("streaming_config")
    if streaming_config_raw is None:
        streaming_config_raw = raw.get("da3_streaming_config")
    if streaming_config_raw is None:
        streaming_config = repo_root / "da3_streaming" / "configs" / "base_config.yaml"
    else:
        streaming_config = Path(str(streaming_config_raw)).expanduser()
        if not streaming_config.is_absolute():
            streaming_config = (base_root / streaming_config).resolve()

    target_resolution = _parse_resolution(raw.get("standard_export_resolution"))
    preferred_linalg = raw.get("preferred_linalg_library")
    preferred_linalg = str(preferred_linalg).strip() if preferred_linalg is not None else None
    extrinsics_convention = _normalize_extrinsics_convention(raw.get("extrinsics_convention"))

    return DepthAnything3Settings(
        input_path=input_path,
        export_root=export_root,
        geometry_root=geometry_root_raw,
        cache_root=cache_root,
        mode=mode,
        model_dir=str(raw.get("model_dir", _DEFAULT_MODEL_DIR)),
        conda_env=str(raw.get("conda_env", "DA3")),
        process_res=int(raw.get("process_res", 504)),
        process_res_method=str(raw.get("process_res_method", "upper_bound_resize")),
        input_max_side=_coerce_positive_int(raw.get("input_max_side")),
        input_center_crop=bool(raw.get("input_center_crop", False)),
        device=str(raw.get("device", "cuda")),
        preferred_linalg_library=preferred_linalg or None,
        cleanup_export_dir=bool(raw.get("cleanup_export_dir", False)),
        reuse_exports=bool(raw.get("reuse_exports", True)),
        run_inference=bool(raw.get("run_inference", True)),
        repo_root=repo_root,
        streaming_config=streaming_config,
        standard_export_resolution=target_resolution,
        export_format=str(raw.get("export_format", "mini_npz-glb")),
        max_images=_coerce_positive_int(raw.get("max_images")),
        use_ray_pose=bool(raw.get("use_ray_pose", True)),
        ref_view_strategy=str(raw.get("ref_view_strategy", "middle")),
        extrinsics_convention=extrinsics_convention,
    )


def _resolve_repo_root(candidate: Optional[Path]) -> Path:
    """
    Resolve the Depth-Anything-3 repository root.

    Preference order:
    1. Explicit candidate (adapter settings)
    2. PEMOIN_REPO_ROOT environment variable
    3. Current working directory
    4. Source tree fallback (relative to this file)
    """
    bases: List[Path] = []
    if candidate is not None:
        bases.append(candidate)
    env_root = os.environ.get("PEMOIN_REPO_ROOT")
    if env_root:
        bases.append(Path(env_root))
    bases.append(Path.cwd())
    bases.append(_REPO_ROOT_FALLBACK)

    def _as_da3_root(base: Path) -> Path:
        base = base.expanduser().resolve()
        if base.name == "Depth-Anything-3":
            return base
        nested = base / "tools" / "Depth-Anything-3"
        if nested.exists():
            return nested
        return base

    for base in bases:
        root = _as_da3_root(base)
        if root.exists() and (root / "src" / "depth_anything_3").exists():
            return root
    return _as_da3_root(bases[-1])


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_resolution(value: Any) -> Optional[tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _discover_images(root: Path) -> List[Path]:
    images = []
    for ext in _SUPPORTED_EXTENSIONS:
        images.extend(root.glob(f"*{ext}"))
        images.extend(root.glob(f"*{ext.upper()}"))
    images = sorted({p.resolve() for p in images})
    return images


_STREAMING_FRAME_RE = re.compile(r"^frame_(\d+)\.npz$")


def _discover_streaming_npz(results_dir: Path) -> List[Path]:
    candidates = [p for p in results_dir.glob("frame_*.npz") if _STREAMING_FRAME_RE.match(p.name)]
    return sorted(candidates, key=_streaming_frame_index)


def _streaming_frame_index(path: Path) -> int:
    match = _STREAMING_FRAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected streaming output filename '{path.name}'.")
    return int(match.group(1))


def _load_camera_poses(path: Path) -> np.ndarray:
    poses: List[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(token) for token in line.strip().split()]
        if len(values) != 16:
            raise ValueError(f"Expected 16 values per pose in '{path}', got {len(values)}.")
        pose = np.asarray(values, dtype=np.float32).reshape(4, 4)
        poses.append(pose)
    if not poses:
        raise ValueError(f"No camera poses found in '{path}'.")
    return np.stack(poses, axis=0)


def _load_intrinsics_txt(path: Path) -> np.ndarray:
    intrinsics: List[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(token) for token in line.strip().split()]
        if len(values) != 4:
            raise ValueError(f"Expected 4 values per intrinsics row in '{path}', got {len(values)}.")
        fx, fy, cx, cy = values
        intrinsics.append(
            np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        )
    if not intrinsics:
        raise ValueError(f"No intrinsics found in '{path}'.")
    return np.stack(intrinsics, axis=0)


def _ensure_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "DepthAnything3 video input requires opencv-python in the DA3 environment."
        ) from exc
    return cv2


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_file():
            entry.unlink()
        else:
            shutil.rmtree(entry)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def register_depthanything3_provider_builders(factory: ProviderFactory) -> None:
    """Register DA3-backed provider builders with the factory."""

    def _resolve_adapter(binding: ModuleBinding, context: MutableMapping[str, Any]) -> DepthAnything3Adapter:
        adapter_settings: dict[str, Any] = {}
        profile_defaults = context.get("depthanything3_settings", {})
        if isinstance(profile_defaults, Mapping):
            adapter_settings.update(profile_defaults)
        adapter_settings.update(binding.settings.get("adapter", {}))
        working_res = context.get("working_resolution")
        if working_res and "standard_export_resolution" not in adapter_settings:
            adapter_settings["standard_export_resolution"] = list(working_res)
        # Fill in input/export paths from context when not provided in config.
        if "input_path" not in adapter_settings:
            frames_dir = context.get("frames_dir")
            if frames_dir is None:
                raise ValueError(
                    f"Provider '{binding.tool}' requires 'frames_dir' in context "
                    "or an explicit 'input_path' in adapter settings."
                )
            adapter_settings["input_path"] = str(frames_dir)
        if "export_root" not in adapter_settings:
            store = context.get("resource_store")
            if isinstance(store, ResourceStore):
                adapter_settings["export_root"] = str(store.raw_root / "depthanything3")
            else:
                run_dir = context.get("run_dir")
                if run_dir is None:
                    raise ValueError(
                        f"Provider '{binding.tool}' requires 'export_root' or a 'run_dir' in context."
                    )
                adapter_settings["export_root"] = str(Path(run_dir) / "raw" / "depthanything3")
        if "geometry_root" not in adapter_settings:
            store = context.get("resource_store")
            if isinstance(store, ResourceStore):
                adapter_settings["geometry_root"] = str(store.standard_root / "geometry" / "depthanything3")
            else:
                run_dir = context.get("run_dir")
                if run_dir is None:
                    raise ValueError(
                        f"Provider '{binding.tool}' requires 'geometry_root' or a 'run_dir' in context."
                    )
                adapter_settings["geometry_root"] = str(Path(run_dir) / "standard" / "geometry" / "depthanything3")

        cache = context.setdefault("da3_adapters", {})
        key = json.dumps(adapter_settings, sort_keys=True)
        if key not in cache:
            cache[key] = DepthAnything3Adapter(adapter_settings)
        return cache[key]

    def build_depth(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> DepthAnything3DepthProvider:
        adapter = _resolve_adapter(binding, context)
        inference_options = binding.settings.get("inference", {})
        return adapter.create_depth_provider(inference_options)

    def build_traj(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> DepthAnything3TrajectoryProvider:
        adapter = _resolve_adapter(binding, context)
        inference_options = binding.settings.get("inference", {})
        return adapter.create_trajectory_provider(inference_options)

    def build_intr(
        binding: ModuleBinding, context: MutableMapping[str, Any]
    ) -> DepthAnything3IntrinsicsProvider:
        adapter = _resolve_adapter(binding, context)
        inference_options = binding.settings.get("inference", {})
        return adapter.create_intrinsics_provider(inference_options)

    factory.register("DepthAnything3DepthProvider", build_depth)
    factory.register("DepthAnything3TrajectoryProvider", build_traj)
    factory.register("DepthAnything3IntrinsicsProvider", build_intr)
