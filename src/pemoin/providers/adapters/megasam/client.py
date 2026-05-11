"""Robust MegaSAM bundle loader and contract adapter for PEMOIN."""

from __future__ import annotations

import json
import logging
import filecmp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender
from pemoin.data.contracts import DepthData, IntrinsicsData, PoseData, PoseSample
from pemoin.utils.geometry_export import save_standard_geometry
from pemoin.utils.logging import get_logger
from pemoin.visualization.debug_artifacts import (
    write_depth_preview,
    write_intrinsics_summary,
    write_trajectory_path_plots,
)

LOG = get_logger()


def ensure_megasam_log_handler(log_dir: Path) -> Path:
    """Attach a dedicated MegaSAM file logger once and return its path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / "megasam.log").resolve()
    for handler in LOG.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path:
            return log_path
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    if LOG.level > logging.DEBUG:
        LOG.setLevel(logging.DEBUG)
    return log_path


@dataclass(frozen=True, slots=True)
class MegaSAMSettings:
    """Settings used to initialize the MegaSAM client."""

    checkpoint_path: Path
    config_path: Path
    cache_dir: Path
    device: str = "cuda:0"
    precision: str = "float16"
    bundle_path: Optional[Path] = None
    scene_name: Optional[str] = None
    repository_root: Optional[Path] = None
    working_resolution: Optional[tuple[int, int]] = None
    standard_export_root: Optional[Path] = None
    frame_index_map_path: Optional[Path] = None
    tracking_preprocess_path: Optional[Path] = None
    gt_intrinsics_path: Optional[Path] = None
    require_final_bundle: bool = True
    enforce_gt_intrinsics: bool = True
    write_debug_artifacts: bool = True


@dataclass(slots=True)
class MegaSAMBundle:
    """In-memory representation of a MegaSAM final bundle."""

    path: Path
    scene_name: str
    images_rgb: Optional[np.ndarray]
    depths: np.ndarray
    intrinsic_cv: np.ndarray
    cam_c2w_cv: np.ndarray
    cam_w2c_cv: np.ndarray
    cam_c2w_bl: np.ndarray
    cam_w2c_bl: np.ndarray

    @classmethod
    def load(
        cls,
        path: Path,
        scene_name: Optional[str],
        *,
        require_final_bundle: bool,
    ) -> "MegaSAMBundle":
        if not path.exists():
            raise FileNotFoundError(
                f"MegaSAM bundle not found at '{path}'. Run MegaSAM automation first."
            )

        if require_final_bundle:
            name = path.name.lower()
            if "droid" in name:
                raise ValueError(
                    "Refusing non-final MegaSAM bundle: filename contains 'droid'. "
                    "Use the CVD-optimized final bundle (for example '*_sgd_cvd_hr.npz')."
                )
            if "cvd" not in name and not cls._matches_sibling_cvd_bundle(path):
                raise ValueError(
                    "Refusing bundle without a verifiable CVD marker. Expected a final "
                    "optimized bundle (for example '*_sgd_cvd_hr.npz') or a byte-identical "
                    "copy of one in the same directory."
                )

        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in ("depths", "intrinsic", "cam_c2w") if k not in data.files]
            if missing:
                raise ValueError(
                    f"MegaSAM bundle missing required keys {missing}. Found keys: {list(data.files)}"
                )

            depths = cls._ensure_depth_stack(data["depths"])
            frame_count = int(depths.shape[0])
            cam_c2w_cv = cls._ensure_pose_stack(data["cam_c2w"], frame_count=frame_count)
            intrinsic_cv = cls._ensure_intrinsic_matrix(data["intrinsic"], frame_count=frame_count)
            images_rgb = cls._optional_images_rgb(data, frame_count=frame_count)

        try:
            cam_w2c_cv = np.linalg.inv(cam_c2w_cv).astype(np.float32)
        except np.linalg.LinAlgError as exc:
            raise ValueError("MegaSAM cam_c2w matrices are non-invertible.") from exc

        cam_c2w_bl, cam_w2c_bl = cls._convert_pose_stack_to_blender(cam_c2w_cv, cam_w2c_cv)

        return cls(
            path=path,
            scene_name=scene_name or path.stem,
            images_rgb=images_rgb,
            depths=depths,
            intrinsic_cv=intrinsic_cv,
            cam_c2w_cv=cam_c2w_cv,
            cam_w2c_cv=cam_w2c_cv,
            cam_c2w_bl=cam_c2w_bl,
            cam_w2c_bl=cam_w2c_bl,
        )

    @staticmethod
    def _matches_sibling_cvd_bundle(path: Path) -> bool:
        try:
            candidates = sorted(path.parent.glob("*_sgd_cvd*.npz"))
        except Exception:
            return False
        if not candidates:
            return False
        for candidate in candidates:
            if candidate.resolve() == path.resolve():
                return True
            try:
                if filecmp.cmp(str(path), str(candidate), shallow=False):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _optional_images_rgb(data: Any, *, frame_count: int) -> Optional[np.ndarray]:
        if "images" not in data.files:
            return None
        arr = np.asarray(data["images"])
        if arr.ndim != 4:
            raise ValueError(f"MegaSAM images must be rank-4, got {arr.shape}.")
        if arr.shape[0] != frame_count:
            raise ValueError(
                f"MegaSAM images frame count {arr.shape[0]} does not match depths {frame_count}."
            )
        if arr.shape[-1] == 3:
            rgb = arr
        elif arr.shape[1] == 3:
            rgb = np.transpose(arr, (0, 2, 3, 1))
        else:
            raise ValueError(
                f"MegaSAM images must be [F,H,W,3] or [F,3,H,W], got {arr.shape}."
            )
        return np.asarray(rgb, dtype=np.uint8)

    @staticmethod
    def _ensure_depth_stack(depths: np.ndarray) -> np.ndarray:
        arr = np.asarray(depths, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"MegaSAM depths must be [F,H,W], got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("MegaSAM depths contain non-finite values.")
        if float(np.max(arr)) <= 0.0:
            raise ValueError("MegaSAM depths contain no positive values.")
        return arr

    @staticmethod
    def _ensure_pose_stack(cam_c2w: np.ndarray, *, frame_count: int) -> np.ndarray:
        arr = np.asarray(cam_c2w, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"MegaSAM cam_c2w must be [F,4,4] or [F,3,4], got {arr.shape}.")
        if arr.shape[0] != frame_count:
            raise ValueError(
                f"MegaSAM cam_c2w frame count {arr.shape[0]} does not match depths {frame_count}."
            )
        if arr.shape[1:] == (3, 4):
            bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0], dtype=arr.dtype), (arr.shape[0], 1, 4))
            arr = np.concatenate([arr, bottom], axis=1)
        if arr.shape[1:] != (4, 4):
            raise ValueError(f"MegaSAM cam_c2w must resolve to [F,4,4], got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("MegaSAM cam_c2w contains non-finite values.")
        if not np.allclose(arr[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
            raise ValueError("MegaSAM cam_c2w has invalid homogeneous last row.")
        return arr

    @staticmethod
    def _ensure_intrinsic_matrix(intrinsic: np.ndarray, *, frame_count: int) -> np.ndarray:
        arr = np.asarray(intrinsic, dtype=np.float32)
        if arr.shape == (3, 3):
            return arr
        if arr.ndim == 2 and arr.shape[1] == 4 and arr.shape[0] == frame_count:
            fx, fy, cx, cy = [arr[0, i] for i in range(4)]
            k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            if not np.allclose(arr, arr[0], atol=1e-6):
                raise ValueError("MegaSAM per-frame intrinsics (fx,fy,cx,cy) vary by frame.")
            return k
        if arr.ndim == 3 and arr.shape[1:] == (3, 3):
            if arr.shape[0] != frame_count:
                raise ValueError(
                    f"MegaSAM intrinsics frame count {arr.shape[0]} does not match depths {frame_count}."
                )
            if not np.allclose(arr, arr[0], atol=1e-6):
                raise ValueError("MegaSAM per-frame 3x3 intrinsics vary by frame.")
            return np.asarray(arr[0], dtype=np.float32)
        raise ValueError(
            "MegaSAM intrinsic must be [3,3], [F,3,3], or [F,4](fx,fy,cx,cy); "
            f"got {arr.shape}."
        )

    @staticmethod
    def _convert_pose_stack_to_blender(
        cam_c2w_cv: np.ndarray,
        cam_w2c_cv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        c2w_list: List[np.ndarray] = []
        w2c_list: List[np.ndarray] = []
        for c2w_cv, w2c_cv in zip(cam_c2w_cv, cam_w2c_cv):
            c2w_bl, w2c_bl = convert_pose_opencv_to_blender(c2w_cv, w2c_cv)
            if w2c_bl is None:
                raise RuntimeError("Pose conversion returned no world_to_camera matrix.")
            c2w_list.append(np.asarray(c2w_bl, dtype=np.float32))
            w2c_list.append(np.asarray(w2c_bl, dtype=np.float32))
        return np.stack(c2w_list, axis=0), np.stack(w2c_list, axis=0)

    @property
    def frame_count(self) -> int:
        return int(self.depths.shape[0])


class MegaSAMClient:
    """Expose MegaSAM bundle artifacts via PEMOIN contracts."""

    def __init__(self, settings: MegaSAMSettings):
        self.settings = settings
        self._initialised = False
        self._bundle: Optional[MegaSAMBundle] = None
        self._frame_index_map: Optional[Dict[int, int]] = None
        self._runtime_intrinsics_override: Optional[np.ndarray] = None
        self._tracking_preprocess_meta: Optional[Dict[str, Any]] = None

    def initialise(self) -> None:
        if self._initialised:
            return
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self._initialised = True

    def set_runtime_intrinsics_override(self, intrinsics: Optional[IntrinsicsData | np.ndarray]) -> None:
        """Set per-run GT intrinsics (for example from CARLA) when available."""
        if intrinsics is None:
            return
        matrix: Optional[np.ndarray]
        if isinstance(intrinsics, IntrinsicsData):
            matrix = np.asarray(intrinsics.matrix, dtype=np.float32)
        else:
            matrix = np.asarray(intrinsics, dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"Runtime intrinsics override must be 3x3, got {matrix.shape}.")
        self._runtime_intrinsics_override = matrix

    def estimate_depth(self, frames: Iterable[Any]) -> DepthData | List[DepthData]:
        bundle = self._ensure_bundle_loaded()
        refs = self._normalise_frames(frames)
        results: List[DepthData] = []
        for frame_index, bundle_index, target_shape in refs:
            depth = np.asarray(bundle.depths[bundle_index], dtype=np.float32)
            if target_shape is not None and depth.shape[:2] != tuple(target_shape):
                depth = _resize_2d(depth, target_shape, interpolation="bilinear")
            metadata = self._build_metadata("depth", frame_index)
            metadata.update(
                {
                    "units": "depth_along_camera_z",
                    "frame_count": bundle.frame_count,
                    "bundle_index": bundle_index,
                    "camera_convention": "blender",
                    "source_camera_convention": "opencv",
                    "camera_axes": "x-right,y-up,z-backward",
                    "source_camera_axes": "x-right,y-down,z-forward",
                    "reference_resolution": [int(depth.shape[0]), int(depth.shape[1])],
                }
            )
            results.append(
                DepthData(
                    frame_index=frame_index,
                    depth=depth,
                    metadata=metadata,
                )
            )
        return results[0] if len(results) == 1 else results

    def estimate_trajectory(
        self, frames: Iterable[Any], metadata: Optional[MutableMapping[str, Any]] = None
    ) -> PoseData | List[PoseData]:
        bundle = self._ensure_bundle_loaded()
        refs = self._normalise_frames(frames)
        diagnostics = metadata if metadata is not None else {}
        diagnostics.setdefault("source", "MegaSAM")
        diagnostics.setdefault("scene", bundle.scene_name)
        diagnostics.setdefault("metric_scale", False)

        results: List[PoseData] = []
        for frame_index, bundle_index, _target_shape in refs:
            pose_meta = self._build_metadata("trajectory", frame_index)
            pose_meta.update(
                {
                    "bundle_index": bundle_index,
                    "camera_convention": "blender",
                    "pose_coordinate_system": "blender",
                    "source_camera_convention": "opencv",
                    "camera_axes": "x-right,y-up,z-backward",
                    "source_camera_axes": "x-right,y-down,z-forward",
                    "pose_representation": "4x4_matrix",
                }
            )
            results.append(
                PoseData(
                    samples=[
                        PoseSample(
                            frame_index=frame_index,
                            camera_to_world=np.asarray(bundle.cam_c2w_bl[bundle_index], dtype=np.float32),
                            world_to_camera=np.asarray(bundle.cam_w2c_bl[bundle_index], dtype=np.float32),
                            metadata=pose_meta,
                        )
                    ],
                    metadata=dict(diagnostics),
                )
            )
        return results[0] if len(results) == 1 else results

    def fetch_intrinsics(self) -> IntrinsicsData:
        bundle = self._ensure_bundle_loaded()
        matrix, source_label, source_path = self._resolve_active_intrinsics(bundle)
        tracking_meta = self._load_tracking_preprocess_metadata(bundle)
        intrinsics_applied = False
        if tracking_meta is not None and source_label in {"runtime_gt", "gt_intrinsics_path"}:
            matrix = _map_intrinsics_to_tracking_geometry(
                np.asarray(matrix, dtype=np.float32),
                tracking_meta,
            )
            intrinsics_applied = True

        if tracking_meta is not None:
            reference_resolution = (
                int(tracking_meta["tracking_height"]),
                int(tracking_meta["tracking_width"]),
            )
            width = int(tracking_meta["tracking_width"])
            height = int(tracking_meta["tracking_height"])
        else:
            reference_resolution = tuple(int(dim) for dim in bundle.depths.shape[1:3])
            width = int(reference_resolution[1])
            height = int(reference_resolution[0])

        return IntrinsicsData(
            matrix=np.asarray(matrix, dtype=np.float32),
            metadata={
                "source": "MegaSAM",
                "intrinsics_source": source_label,
                "intrinsics_source_path": str(source_path) if source_path is not None else None,
                "scene": bundle.scene_name,
                "bundle_path": str(bundle.path),
                "device": self.settings.device,
                "precision": self.settings.precision,
                "dynamic": False,
                "reference_resolution": reference_resolution,
                "width": width,
                "height": height,
                "camera_convention": "blender",
                "source_camera_convention": "opencv",
                "tracking_preprocess_applied": intrinsics_applied,
                "tracking_preprocess": tracking_meta,
            },
        )

    def _ensure_bundle_loaded(self) -> MegaSAMBundle:
        if self._bundle is not None:
            return self._bundle
        bundle_path = self.settings.bundle_path
        if bundle_path is None:
            raise RuntimeError(
                "MegaSAM adapter requires 'bundle_path'. Provide a final MegaSAM CVD bundle."
            )
        bundle = MegaSAMBundle.load(
            bundle_path,
            self.settings.scene_name,
            require_final_bundle=bool(self.settings.require_final_bundle),
        )
        self._bundle = bundle

        export_dir = self._write_standard_export(bundle)
        self._log_bundle_summary(bundle, export_dir=export_dir)
        return bundle

    def _normalise_frames(self, frames: Iterable[Any]) -> List[tuple[int, int, Optional[tuple[int, int]]]]:
        refs: List[tuple[int, int, Optional[tuple[int, int]]]] = []
        for frame in frames:
            frame_index = self._extract_frame_index(frame)
            bundle_index = self._resolve_bundle_index(frame_index)
            refs.append((frame_index, bundle_index, self._target_shape_from_frame(frame)))
        if not refs:
            raise ValueError("MegaSAMClient requires at least one frame reference.")
        return refs

    def _resolve_bundle_index(self, frame_index: int) -> int:
        bundle = self._ensure_bundle_loaded()
        mapping = self._load_frame_index_map()
        if not mapping:
            if frame_index < 0 or frame_index >= bundle.frame_count:
                raise IndexError(
                    f"Frame index {frame_index} is out of bundle bounds [0, {bundle.frame_count - 1}]."
                )
            return frame_index

        mapped = mapping.get(int(frame_index))
        if mapped is None:
            raise IndexError(
                f"MegaSAM frame_index_map does not contain frame {frame_index}. "
                f"Available examples: {sorted(mapping.keys())[:10]}"
            )
        if mapped < 0 or mapped >= bundle.frame_count:
            raise IndexError(
                f"Frame map resolved {mapped} for frame {frame_index}, "
                f"but bundle has {bundle.frame_count} frames."
            )
        return int(mapped)

    def _load_frame_index_map(self) -> Dict[int, int]:
        if self._frame_index_map is not None:
            return dict(self._frame_index_map)

        path = self.settings.frame_index_map_path
        if path is None:
            self._frame_index_map = {}
            return {}

        resolved = _resolve_path(path)
        if not resolved.exists():
            self._frame_index_map = {}
            return {}

        raw = json.loads(resolved.read_text())
        frames = raw.get("frame_indices") if isinstance(raw, dict) else raw
        if not isinstance(frames, list):
            raise ValueError(f"MegaSAM frame_index_map must be a list, got {type(frames)}")

        mapping: Dict[int, int] = {}
        for bundle_idx, frame_idx in enumerate(frames):
            mapping[int(frame_idx)] = int(bundle_idx)
        self._frame_index_map = mapping
        return dict(mapping)

    @staticmethod
    def _extract_frame_index(frame: Any) -> int:
        if hasattr(frame, "index"):
            return int(getattr(frame, "index"))
        if isinstance(frame, Mapping) and "index" in frame:
            return int(frame["index"])
        if isinstance(frame, int):
            return int(frame)
        raise TypeError("Frames must expose an 'index' attribute or be integer indices.")

    def _target_shape_from_frame(self, frame: Any) -> Optional[tuple[int, int]]:
        image = getattr(frame, "image", None)
        if isinstance(image, np.ndarray) and image.ndim >= 2:
            return int(image.shape[0]), int(image.shape[1])

        metadata = getattr(frame, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            wr = metadata.get("working_resolution")
            if isinstance(wr, (list, tuple)) and len(wr) >= 2:
                return int(wr[0]), int(wr[1])

        return self.settings.working_resolution

    def _build_metadata(self, kind: str, frame_index: int) -> Dict[str, Any]:
        bundle = self._ensure_bundle_loaded()
        tracking_meta = self._load_tracking_preprocess_metadata(bundle)
        return {
            "source": "MegaSAM",
            "scene": bundle.scene_name,
            "frame_index": int(frame_index),
            "device": self.settings.device,
            "precision": self.settings.precision,
            "bundle_path": str(bundle.path),
            "kind": kind,
            "metric_scale": False,
            "extrinsics_convention": "c2w",
            "pose_representation": "4x4_matrix",
            "tracking_preprocess": tracking_meta,
        }

    def _load_intrinsics_from_npz(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        resolved = _resolve_path(path)
        if not resolved.exists():
            return None
        with np.load(resolved, allow_pickle=True) as data:
            if "matrix" not in data.files:
                raise ValueError(f"Intrinsics NPZ missing 'matrix': {resolved}")
            matrix = np.asarray(data["matrix"], dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"Intrinsics matrix must be 3x3, got {matrix.shape} from {resolved}")
        return matrix

    def _resolve_active_intrinsics(
        self,
        bundle: MegaSAMBundle,
    ) -> tuple[np.ndarray, str, Optional[Path]]:
        if self._runtime_intrinsics_override is not None:
            return np.asarray(self._runtime_intrinsics_override, dtype=np.float32), "runtime_gt", None

        gt_matrix = self._load_intrinsics_from_npz(self.settings.gt_intrinsics_path)
        if gt_matrix is not None:
            return gt_matrix, "gt_intrinsics_path", _resolve_path(self.settings.gt_intrinsics_path)

        if self.settings.enforce_gt_intrinsics:
            gt_path = self.settings.gt_intrinsics_path
            if gt_path is None:
                raise RuntimeError(
                    "MegaSAM GT intrinsics enforcement is enabled but no gt_intrinsics_path "
                    "was provided and no runtime intrinsics override is available."
                )
            raise RuntimeError(
                "MegaSAM GT intrinsics enforcement is enabled but GT intrinsics could not be "
                f"loaded from '{_resolve_path(gt_path)}' and no runtime intrinsics override is available."
            )

        return np.asarray(bundle.intrinsic_cv, dtype=np.float32), "bundle", bundle.path

    def _load_tracking_preprocess_metadata(self, bundle: MegaSAMBundle) -> Optional[Dict[str, Any]]:
        if self._tracking_preprocess_meta is not None:
            return dict(self._tracking_preprocess_meta)

        explicit = self.settings.tracking_preprocess_path
        candidates: List[Path] = []
        if explicit is not None:
            candidates.append(_resolve_path(explicit))
        candidates.append(bundle.path.with_suffix(".tracking.json"))
        candidates.append(bundle.path.parent / f"{bundle.path.stem}_tracking_preprocess.json")

        selected: Optional[Path] = None
        for candidate in candidates:
            if candidate.exists():
                selected = candidate
                break

        if explicit is not None and selected is None:
            raise FileNotFoundError(
                f"MegaSAM tracking_preprocess_path does not exist: {explicit}"
            )
        if selected is None:
            self._tracking_preprocess_meta = {}
            return None

        raw = json.loads(selected.read_text(encoding="utf-8"))
        meta = _validate_tracking_preprocess_metadata(raw, path=selected)
        meta["path"] = str(selected)
        self._tracking_preprocess_meta = dict(meta)
        LOG.info("[MegaSAM] Loaded tracking preprocess metadata: %s", selected)
        return dict(meta)

    def _write_standard_export(self, bundle: MegaSAMBundle) -> Path:
        source_name = bundle.path.parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.settings.standard_export_root or bundle.path.parent
        target_dir = base_dir / f"megasam_{source_name}_{timestamp}"

        intrinsics, intr_source, _intr_path = self._resolve_active_intrinsics(bundle)
        target_shape = self.settings.working_resolution

        save_standard_geometry(
            target_dir,
            source="MegaSAM",
            depths=bundle.depths,
            confidence=None,
            intrinsics=intrinsics,
            extrinsics_w2c=bundle.cam_w2c_bl,
            frame_ids=range(bundle.frame_count),
            target_shape=target_shape,
            source_camera_convention="blender",
        )

        if self.settings.write_debug_artifacts:
            debug_dir = target_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            try:
                write_intrinsics_summary(
                    debug_dir / "intrinsics_summary.png",
                    intrinsics,
                    {
                        "reference_resolution": list(bundle.depths.shape[1:3]),
                        "source": "MegaSAM",
                        "intrinsics_source": intr_source,
                    },
                )
            except Exception as exc:
                LOG.debug("[MegaSAM] Failed to render intrinsics summary: %s", exc)

            try:
                write_trajectory_path_plots(debug_dir / "trajectory", bundle.cam_c2w_bl)
            except Exception as exc:
                LOG.debug("[MegaSAM] Failed to render trajectory plots: %s", exc)

            try:
                indices = [0, bundle.frame_count // 2, bundle.frame_count - 1]
                for idx in sorted(set(indices)):
                    write_depth_preview(debug_dir / f"depth_preview_{idx:06d}.png", bundle.depths[idx])
            except Exception as exc:
                LOG.debug("[MegaSAM] Failed to render depth previews: %s", exc)

        LOG.info("[MegaSAM] Standardized geometry export written to %s", target_dir)
        return target_dir

    def _log_bundle_summary(self, bundle: MegaSAMBundle, *, export_dir: Optional[Path]) -> None:
        log_dir: Optional[Path] = None
        if self.settings.standard_export_root is not None:
            log_dir = Path(self.settings.standard_export_root).parent / "logs"
        if log_dir is not None:
            ensure_megasam_log_handler(log_dir)

        LOG.info(
            "[MegaSAM] Bundle loaded: scene=%s path=%s frames=%d",
            bundle.scene_name,
            bundle.path,
            bundle.frame_count,
        )
        if export_dir is not None:
            LOG.info("[MegaSAM] Geometry export directory: %s", export_dir)

        depth = bundle.depths
        LOG.debug(
            "[MegaSAM] Depth stats: shape=%s dtype=%s min=%.4f med=%.4f mean=%.4f p95=%.4f max=%.4f",
            depth.shape,
            depth.dtype,
            float(np.min(depth)),
            float(np.median(depth)),
            float(np.mean(depth)),
            float(np.percentile(depth, 95)),
            float(np.max(depth)),
        )

        c2w = bundle.cam_c2w_cv
        rot = c2w[:, :3, :3]
        det = np.linalg.det(rot)
        ortho = np.linalg.norm(rot @ np.transpose(rot, (0, 2, 1)) - np.eye(3), axis=(1, 2))
        steps = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1) if c2w.shape[0] > 1 else np.array([0.0])

        LOG.debug(
            "[MegaSAM] Pose stats (OpenCV source): det[min/med/max]=%.6f/%.6f/%.6f "
            "ortho_err[med/max]=%.6e/%.6e step_m[med/max]=%.4f/%.4f",
            float(np.min(det)),
            float(np.median(det)),
            float(np.max(det)),
            float(np.median(ortho)),
            float(np.max(ortho)),
            float(np.median(steps)),
            float(np.max(steps)),
        )

        # Data-backed convention notes from artifact inspection heuristics.
        up = c2w[:, :3, 1]
        fwd = c2w[:, :3, 2]
        motion = np.diff(c2w[:, :3, 3], axis=0)
        motion_norm = np.linalg.norm(motion, axis=1)
        valid = motion_norm > 1e-8
        if np.any(valid):
            v = motion[valid] / motion_norm[valid, None]
            f = fwd[:-1][valid]
            cos_forward = np.sum(v * f, axis=1)
            LOG.debug(
                "[MegaSAM] Convention heuristics: median(up)=%s median(forward)=%s median(cos(motion,+Zcam))=%.6f",
                np.array2string(np.median(up, axis=0), precision=5, suppress_small=True),
                np.array2string(np.median(fwd, axis=0), precision=5, suppress_small=True),
                float(np.median(cos_forward)),
            )


# ------------------------------------------------------------------ #
# Local helpers
# ------------------------------------------------------------------ #


def _resolve_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except Exception:
        return Path(path)


def _resize_2d(arr: np.ndarray, shape_hw: Sequence[int], *, interpolation: str) -> np.ndarray:
    target_h, target_w = int(shape_hw[0]), int(shape_hw[1])
    src = np.asarray(arr)
    if src.shape[:2] == (target_h, target_w):
        return src.astype(np.float32, copy=False)
    try:
        import cv2  # type: ignore

        interp = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
        out = cv2.resize(src, (target_w, target_h), interpolation=interp)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        from PIL import Image

        img = Image.fromarray(src.astype(np.float32), mode="F")
        resample = Image.BILINEAR if interpolation == "bilinear" else Image.NEAREST
        out = img.resize((target_w, target_h), resample=resample)
        return np.asarray(out, dtype=np.float32)


def _validate_tracking_preprocess_metadata(raw: Any, *, path: Path) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Invalid tracking preprocess metadata at {path}: expected object, got {type(raw)}"
        )
    required = (
        "source_width",
        "source_height",
        "resized_width",
        "resized_height",
        "tracking_width",
        "tracking_height",
        "scale_x",
        "scale_y",
        "transform",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            f"Tracking preprocess metadata at {path} missing keys: {missing}"
        )
    transform = raw["transform"]
    if not isinstance(transform, Mapping):
        raise ValueError(f"Tracking preprocess transform must be an object in {path}")
    if transform.get("mode") not in {"pad", "crop"}:
        raise ValueError(
            f"Tracking preprocess transform.mode must be 'pad' or 'crop' in {path}"
        )
    for key in ("left", "top"):
        if key not in transform:
            raise ValueError(f"Tracking preprocess transform missing '{key}' in {path}")
    return dict(raw)


def _map_intrinsics_to_tracking_geometry(
    matrix: np.ndarray,
    tracking_meta: Mapping[str, Any],
) -> np.ndarray:
    k = np.asarray(matrix, dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"Intrinsics matrix must be 3x3, got {k.shape}")
    scale_x = float(tracking_meta["scale_x"])
    scale_y = float(tracking_meta["scale_y"])
    t = tracking_meta["transform"]
    if not isinstance(t, Mapping):
        raise ValueError("tracking_preprocess.transform must be a mapping")
    mode = str(t.get("mode"))
    left = float(t.get("left", 0.0))
    top = float(t.get("top", 0.0))

    out = k.copy()
    out[0, 0] *= scale_x
    out[1, 1] *= scale_y
    out[0, 2] *= scale_x
    out[1, 2] *= scale_y
    if mode == "pad":
        out[0, 2] += left
        out[1, 2] += top
    elif mode == "crop":
        out[0, 2] -= left
        out[1, 2] -= top
    else:
        raise ValueError(f"Unsupported tracking transform mode: {mode}")
    return out
