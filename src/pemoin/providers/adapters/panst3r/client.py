"""
Thin client responsible for loading PanSt3R bundle outputs and exposing them through PEMOIN contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from pemoin.coordinate_systems.conversions import convert_pose_opencv_to_blender
from pemoin.data.contracts import DepthData, IntrinsicsData, PoseData, PoseSample
from pemoin.utils.geometry_export import save_standard_geometry
from pemoin.utils.trajectory_cleanup import TrajectoryCleanupOptions, cleanup_camera_to_world


@dataclass(frozen=True, slots=True)
class PanSt3RSettings:
    """Settings describing how to access a PanSt3R bundle."""

    bundle_path: Path
    scene_name: Optional[str] = None
    device: str = "cuda:0"
    precision: str = "float32"
    working_resolution: Optional[tuple[int, int]] = None
    standard_export_root: Optional[Path] = None


@dataclass(slots=True)
class PanSt3RBundle:
    """In-memory representation of a PanSt3R NPZ output."""

    path: Path
    scene_name: str
    depths: np.ndarray
    position_confidence: Optional[np.ndarray]
    intrinsic: np.ndarray
    cam_c2w: np.ndarray
    cam_w2c: np.ndarray

    @classmethod
    def load(cls, path: Path, scene_name: Optional[str] = None) -> PanSt3RBundle:
        if not path.exists():
            raise FileNotFoundError(
                f"PanSt3R bundle not found at '{path}'. Run the PanSt3R pipeline first."
            )
        with np.load(path, allow_pickle=True) as data:
            depths = np.asarray(data["depths"])
            position_confidence = (
                np.asarray(data["position_confidence"])
                if "position_confidence" in data.files
                else None
            )
            cam_c2w = np.asarray(data["cam_c2w"])
            intrinsic = np.asarray(data["intrinsic"])

        scene = scene_name or path.stem
        cam_w2c = np.linalg.inv(cam_c2w)
        return cls(
            path=path,
            scene_name=scene,
            depths=depths,
            position_confidence=position_confidence,
            intrinsic=intrinsic,
            cam_c2w=cam_c2w,
            cam_w2c=cam_w2c,
        )

    @property
    def frame_count(self) -> int:
        return int(self.depths.shape[0])


class PanSt3RClient:
    """Exposes PanSt3R bundle data as PEMOIN data contracts."""

    def __init__(self, settings: PanSt3RSettings):
        self.settings = settings
        self._bundle: Optional[PanSt3RBundle] = None
        self._initialised = False
        self._world_points_cache: Dict[int, np.ndarray] = {}
        self._ray_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._intrinsic_inv: Optional[np.ndarray] = None
        self._cleaned_cam_c2w: Optional[np.ndarray] = None
        self._cleaned_cam_w2c: Optional[np.ndarray] = None
        self._cleanup_signature: Optional[str] = None
        self._cleanup_metadata: MutableMapping[str, Any] = {}

    def initialise(self) -> None:
        if self._initialised:
            return
        self._initialised = True

    # ------------------------------------------------------------------ #
    # Bundle-backed estimation
    # ------------------------------------------------------------------ #

    def estimate_depth(self, frames: Iterable[Any], options: Mapping[str, Any]) -> DepthData | List[DepthData]:
        bundle = self._ensure_bundle_loaded()
        indices = self._normalise_frames(frames)
        results: List[DepthData] = []
        for index in indices:
            metadata = self._common_metadata("depth", index, options)
            metadata["camera_convention"] = "blender"
            metadata["source_camera_convention"] = "opencv"
            confidence_map = None
            if bundle.position_confidence is not None:
                confidence_map = np.asarray(bundle.position_confidence[index])
            results.append(
                DepthData(
                    frame_index=index,
                    depth=np.asarray(bundle.depths[index]),
                    confidence=confidence_map,
                    metadata=metadata,
                )
            )
        return results[0] if len(results) == 1 else results

    def estimate_trajectory(
        self, frames: Iterable[Any], options: Mapping[str, Any], metadata: MutableMapping[str, Any] | None = None
    ) -> PoseData | List[PoseData]:
        bundle = self._ensure_bundle_loaded()
        indices = self._normalise_frames(frames)
        diagnostics = metadata if metadata is not None else {}
        diagnostics.setdefault("source", "PanSt3R")
        diagnostics.setdefault("scene", bundle.scene_name)
        cam_c2w_all, cam_w2c_all, cleanup_md = self._clean_bundle_trajectory(bundle, options)
        if cleanup_md:
            diagnostics.update(cleanup_md)
        results: List[PoseData] = []
        for index in indices:
            pose_metadata = self._common_metadata("trajectory", index, options)
            if cleanup_md:
                pose_metadata.update(cleanup_md)
            pose_metadata["camera_convention"] = "blender"
            pose_metadata["pose_coordinate_system"] = "blender"
            pose_metadata["source_camera_convention"] = "opencv"
            cam_c2w = np.asarray(cam_c2w_all[index])
            cam_w2c = np.asarray(cam_w2c_all[index])
            cam_c2w, cam_w2c = convert_pose_opencv_to_blender(cam_c2w, cam_w2c)
            results.append(
                PoseData(
                    samples=[
                        PoseSample(
                            frame_index=index,
                            camera_to_world=cam_c2w,
                            world_to_camera=cam_w2c,
                            metadata=pose_metadata,
                        )
                    ],
                    metadata=dict(diagnostics),
                )
            )
        return results[0] if len(results) == 1 else results

    def fetch_intrinsics(self) -> IntrinsicsData:
        bundle = self._ensure_bundle_loaded()
        reference_resolution = tuple(int(dim) for dim in bundle.depths.shape[1:3])
        metadata = {
            "source": "PanSt3R",
            "scene": bundle.scene_name,
            "bundle_path": str(bundle.path),
            "device": self.settings.device,
            "precision": self.settings.precision,
            "dynamic": False,
            "reference_resolution": reference_resolution,
            "camera_convention": "blender",
            "source_camera_convention": "opencv",
        }
        return IntrinsicsData(
            matrix=bundle.intrinsic,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_bundle_loaded(self) -> PanSt3RBundle:
        if self._bundle is not None:
            return self._bundle
        bundle_path = self.settings.bundle_path
        scene_name = self.settings.scene_name
        self._bundle = PanSt3RBundle.load(bundle_path, scene_name)
        self._intrinsic_inv = np.linalg.inv(self._bundle.intrinsic)
        self._world_points_cache.clear()
        self._ray_cache.clear()
        self._write_standard_export(self._bundle)
        return self._bundle

    def _clean_bundle_trajectory(
        self, bundle: PanSt3RBundle, options: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, MutableMapping[str, Any]]:
        cleanup_options = TrajectoryCleanupOptions.from_mapping(options)
        if not cleanup_options.enabled:
            self._cleanup_signature = None
            self._cleaned_cam_c2w = None
            self._cleaned_cam_w2c = None
            self._cleanup_metadata = {}
            return bundle.cam_c2w, bundle.cam_w2c, {}
        signature = cleanup_options.signature()
        if signature == self._cleanup_signature and self._cleaned_cam_c2w is not None and self._cleaned_cam_w2c is not None:
            return self._cleaned_cam_c2w, self._cleaned_cam_w2c, dict(self._cleanup_metadata)
        cleaned_c2w, cleanup_md = cleanup_camera_to_world(bundle.cam_c2w, cleanup_options)
        cleaned_w2c = np.linalg.inv(cleaned_c2w).astype(np.float32)
        self._cleaned_cam_c2w = cleaned_c2w
        self._cleaned_cam_w2c = cleaned_w2c
        self._cleanup_signature = signature
        self._cleanup_metadata = dict(cleanup_md)
        return cleaned_c2w, cleaned_w2c, dict(cleanup_md)

    def _normalise_frames(self, frames: Iterable[Any]) -> List[int]:
        indices: List[int] = []
        for frame in frames:
            index = self._extract_frame_index(frame)
            indices.append(index)
        if not indices:
            raise ValueError("PanSt3RClient requires at least one frame reference.")
        return indices

    @staticmethod
    def _extract_frame_index(frame: Any) -> int:
        if hasattr(frame, "index"):
            return int(getattr(frame, "index"))
        if isinstance(frame, Mapping) and "index" in frame:
            return int(frame["index"])
        if isinstance(frame, int):
            return int(frame)
        raise TypeError("Frames must expose an 'index' attribute or be integer indices.")

    def _common_metadata(self, kind: str, frame_index: int, options: Mapping[str, Any]) -> Dict[str, Any]:
        bundle = self._ensure_bundle_loaded()
        return {
            "source": "PanSt3R",
            "scene": bundle.scene_name,
            "frame_index": frame_index,
            "device": self.settings.device,
            "precision": self.settings.precision,
            "kind": kind,
            "options": dict(options),
            "bundle_path": str(bundle.path),
        }

    def _pixel_rays(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        if self._intrinsic_inv is None:
            raise RuntimeError("PanSt3RClient intrinsic matrix is not initialised.")
        u = (np.arange(width, dtype=np.float32) + 0.5)
        v = (np.arange(height, dtype=np.float32) + 0.5)
        uu, vv = np.meshgrid(u, v)
        ones = np.ones_like(uu, dtype=np.float32)
        pixels = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3)
        rays = (pixels @ self._intrinsic_inv.T).reshape(height, width, 3).astype(np.float32)
        self._ray_cache[key] = rays
        return rays

    def _world_points_for_frame(self, bundle: PanSt3RBundle, frame_index: int) -> np.ndarray:
        cached = self._world_points_cache.get(frame_index)
        if cached is not None:
            return cached
        depth = np.asarray(bundle.depths[frame_index], dtype=np.float32)
        height, width = depth.shape
        rays = self._pixel_rays(height, width)
        cam_points = rays * depth[..., None]
        cam_points = cam_points.reshape(-1, 3)
        ones = np.ones((cam_points.shape[0], 1), dtype=np.float32)
        homo = np.concatenate((cam_points, ones), axis=1)
        world = (bundle.cam_c2w[frame_index] @ homo.T).T[:, :3]
        world = world.reshape(height, width, 3).astype(np.float32)
        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        if np.any(invalid):
            world[invalid] = np.nan
        self._world_points_cache[frame_index] = world
        return world

    def _write_standard_export(self, bundle: PanSt3RBundle) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = bundle.scene_name
        base_dir = self.settings.standard_export_root or bundle.path.parent
        target_dir = base_dir / f"panst3r_{source_name}_{timestamp}"
        conf = bundle.position_confidence if bundle.position_confidence is not None else None
        save_standard_geometry(
            target_dir,
            source="PanSt3R",
            depths=bundle.depths,
            confidence=conf,
            intrinsics=bundle.intrinsic,
            extrinsics_w2c=bundle.cam_w2c,
            frame_ids=range(bundle.frame_count),
            target_shape=self.settings.working_resolution,
            source_camera_convention="opencv",
        )
