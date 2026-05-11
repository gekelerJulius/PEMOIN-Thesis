"""Dense global point cloud provider with Bayesian semantic fusion."""

from __future__ import annotations

import math
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np

from pemoin.data.contracts import (
    CAMERA_CONVENTION_KEY,
    CAMERA_CONVENTION_VALUES,
    IntrinsicsData,
    PointCloud3DData,
    PoseSample,
    ResourceKind,
    ResourceStore,
    SemanticsData,
)
from pemoin.geometry.camera_model import backproject_uv_depth_to_camera, camera_to_world
from pemoin.geometry.conventions import normalize_camera_convention
from pemoin.providers.base import Provider
from pemoin.providers.point_cloud_3d.voxel_grid import VoxelGrid
from pemoin.providers.semantic_roles import resolve_semantic_role_labels
from pemoin.runtime.cache import CrossRunCacheManager
from pemoin.utils.logging import get_logger
from pemoin.visualization.point_cloud_glb import (
    semantic_colors_from_labels,
    write_point_cloud_glb,
)

LOG = get_logger()


@dataclass(frozen=True)
class DensePointCloud3DSettings:
    pixel_stride: int = 2
    voxel_size_m: float = 0.05
    min_depth_m: float = 0.5
    max_depth_m: float = 80.0
    min_observations: int = 2
    min_confidence: float = 0.3
    min_total_points: int = 10000
    max_points: int = 5_000_000
    export_glb: bool = True
    glb_max_points: int = 500_000
    seed: int = 42
    max_position_std_m: float = 0.20
    max_depth_std_m: float = 1.50
    min_view_diversity: float = 0.0
    frame_subsample_target: int = 120
    max_sampled_pixels_per_frame: int = 50_000
    num_workers: int = 0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DensePointCloud3DSettings":
        return cls(
            pixel_stride=int(mapping.get("pixel_stride", cls.pixel_stride)),
            voxel_size_m=float(mapping.get("voxel_size_m", cls.voxel_size_m)),
            min_depth_m=float(mapping.get("min_depth_m", cls.min_depth_m)),
            max_depth_m=float(mapping.get("max_depth_m", cls.max_depth_m)),
            min_observations=int(mapping.get("min_observations", cls.min_observations)),
            min_confidence=float(mapping.get("min_confidence", cls.min_confidence)),
            min_total_points=int(mapping.get("min_total_points", cls.min_total_points)),
            max_points=int(mapping.get("max_points", cls.max_points)),
            export_glb=bool(mapping.get("export_glb", cls.export_glb)),
            glb_max_points=int(mapping.get("glb_max_points", cls.glb_max_points)),
            seed=int(mapping.get("seed", cls.seed)),
            max_position_std_m=float(mapping.get("max_position_std_m", cls.max_position_std_m)),
            max_depth_std_m=float(mapping.get("max_depth_std_m", cls.max_depth_std_m)),
            min_view_diversity=float(mapping.get("min_view_diversity", cls.min_view_diversity)),
            frame_subsample_target=int(mapping.get("frame_subsample_target", cls.frame_subsample_target)),
            max_sampled_pixels_per_frame=int(mapping.get("max_sampled_pixels_per_frame", cls.max_sampled_pixels_per_frame)),
            num_workers=int(mapping.get("num_workers", cls.num_workers)),
        )


class DensePointCloud3DProvider(Provider):
    required_resources = frozenset(
        {
            ResourceKind.DEPTH,
            ResourceKind.INTRINSICS,
            ResourceKind.TRAJECTORY,
            ResourceKind.SEMANTICS_2D,
            ResourceKind.FRAMES,
        }
    )
    produced_resources = frozenset({ResourceKind.POINT_CLOUD_3D})

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = DensePointCloud3DSettings.from_mapping(settings)
        self._cache_manager: CrossRunCacheManager | None = None
        self._profile_name: str | None = None
        self._cache_signature: str | None = None
        self._cache_payload: dict[str, Any] | None = None
        self._cache_status: dict[str, Any] = {
            "cross_run_cache_enabled": False,
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": "disabled",
        }

    def setup(self, context: MutableMapping[str, Any]):
        cache_manager = context.get("cross_run_cache")
        self._cache_manager = (
            cache_manager if isinstance(cache_manager, CrossRunCacheManager) else None
        )
        self._profile_name = (
            str(context.get("profile_name"))
            if context.get("profile_name") is not None
            else None
        )
        self._cache_status = {
            "cross_run_cache_enabled": bool(self._cache_manager and self._cache_manager.enabled),
            "cross_run_cache_hit": False,
            "cross_run_cache_validation": (
                "disabled"
                if self._cache_manager is None or not self._cache_manager.enabled
                else "not-checked"
            ),
        }
        return None

    def process(self, frame: Any):
        raise NotImplementedError("DensePointCloud3DProvider runs in batch mode.")

    def teardown(self):
        return None

    def run(
        self,
        resources: ResourceStore,
        context: MutableMapping[str, object] | None = None,
    ) -> None:
        self.validate_requirements(resources)
        self._validate_settings()

        intrinsics = resources.load_intrinsics()
        self._validate_intrinsics(intrinsics)

        trajectory = resources.load_trajectory()
        frame_indices = [int(sample.frame_index) for sample in trajectory.samples]
        if len(frame_indices) < 2:
            raise RuntimeError("DensePointCloud3DProvider requires trajectory with at least 2 frames.")
        if self._try_materialize_cached_outputs(resources):
            return
        mobile_labels = self._mobile_labels_from_resources(resources, frame_indices, context)
        pose_by_frame = {
            int(sample.frame_index): sample
            for sample in trajectory.samples
        }

        replacement_map: Dict[int, int] = {}
        skipped_frames: tuple[int, ...] = tuple()
        if isinstance(context, Mapping):
            raw_rep = context.get("geometry_consistency_replacement_map")
            if isinstance(raw_rep, Mapping):
                replacement_map = {int(k): int(v) for k, v in raw_rep.items()}
            raw_skip = context.get("geometry_consistency_skipped_frames")
            if isinstance(raw_skip, (list, tuple)):
                skipped_frames = tuple(int(v) for v in raw_skip)
        if skipped_frames:
            LOG.warning(
                "[PointCloud3D] Reusing nearby frame data for skipped frames: %s",
                skipped_frames,
            )

        selected_frame_indices = self._select_frame_indices(frame_indices)
        selected_source_indices = self._selected_source_indices(
            selected_frame_indices,
            replacement_map,
        )
        label_vocabulary = self._build_label_vocabulary(resources, selected_source_indices)
        if not label_vocabulary:
            raise RuntimeError("Could not derive any valid semantic labels for point cloud fusion.")
        label_vocabulary = self._exclude_mobile_labels(label_vocabulary, mobile_labels)
        if not label_vocabulary:
            raise RuntimeError(
                "All semantic labels were excluded by canonical mobile semantic roles; "
                "cannot build point cloud."
            )

        class_ids = sorted(label_vocabulary.keys())
        grid = VoxelGrid(
            voxel_size_m=self.settings.voxel_size_m,
            class_ids=class_ids,
            label_names=label_vocabulary,
        )
        class_index = {int(label_id): idx for idx, label_id in enumerate(class_ids)}
        total_lifted = 0
        total_valid = 0
        worker_count = self._resolve_worker_count(len(selected_frame_indices))
        if len(selected_frame_indices) != len(frame_indices):
            LOG.info(
                "[PointCloud3D] using %d/%d frame(s) for debug-cloud generation.",
                int(len(selected_frame_indices)),
                int(len(frame_indices)),
            )

        source_frame_cache = self._load_source_frame_inputs(
            resources,
            selected_source_indices,
            pose_by_frame=pose_by_frame,
        )
        if worker_count <= 1:
            lifted_results = [
                self._lift_cached_source_frame(
                    source_idx=int(source_idx),
                    source_inputs=source_frame_cache[int(source_idx)],
                    intrinsics=intrinsics,
                    class_index=class_index,
                )
                for source_idx in selected_source_indices
            ]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                lifted_results = list(
                    executor.map(
                        lambda source_idx: self._lift_cached_source_frame(
                            source_idx=int(source_idx),
                            source_inputs=source_frame_cache[int(source_idx)],
                            intrinsics=intrinsics,
                            class_index=class_index,
                        ),
                        selected_source_indices,
                    )
                )

        source_results = {
            int(source_idx): (
                points_world,
                colors,
                labels,
                confidences,
                depth_values,
                view_dirs,
                stats,
            )
            for source_idx, points_world, colors, labels, confidences, depth_values, view_dirs, stats in lifted_results
        }

        for frame_idx in selected_frame_indices:
            source_idx = int(replacement_map.get(int(frame_idx), int(frame_idx)))
            (
                points_world,
                colors,
                labels,
                confidences,
                depth_values,
                view_dirs,
                stats,
            ) = source_results[int(source_idx)]
            total_lifted += int(stats["lifted"])
            total_valid += int(stats["valid"])
            if points_world.shape[0] > 0:
                weights = np.clip(confidences, 1e-3, 1.0)
                grid.integrate_frame(
                    points_world=points_world,
                    colors=colors,
                    label_ids=labels,
                    confidences=confidences,
                    weights=weights,
                    depth_values=depth_values,
                    view_dirs=view_dirs,
                )

            LOG.info(
                "[PointCloud3D] frame=%d source_frame=%d lifted=%d valid=%d valid_ratio=%.3f labels=%s",
                int(frame_idx),
                int(source_idx),
                int(stats["lifted"]),
                int(stats["valid"]),
                float(stats["valid_ratio"]),
                json.dumps(stats["label_distribution"], sort_keys=True),
            )

        cloud = grid.extract_cloud(
            min_observations=self.settings.min_observations,
            min_confidence=self.settings.min_confidence,
            max_points=self.settings.max_points,
            rng=np.random.default_rng(self.settings.seed),
            max_position_std_m=self.settings.max_position_std_m,
            max_depth_std_m=self.settings.max_depth_std_m,
            min_view_diversity=self.settings.min_view_diversity,
        )
        self._validate_cloud(cloud)
        cloud.metadata.update(
            {
                "source": "dense_point_cloud_3d_provider",
                "frame_count": int(len(frame_indices)),
                "selected_frame_count": int(len(selected_frame_indices)),
                "total_lifted_points": int(total_lifted),
                "total_valid_points": int(total_valid),
                "mobile_labels": list(mobile_labels),
                "settings": {
                    "pixel_stride": int(self.settings.pixel_stride),
                    "voxel_size_m": float(self.settings.voxel_size_m),
                    "min_depth_m": float(self.settings.min_depth_m),
                    "max_depth_m": float(self.settings.max_depth_m),
                    "min_observations": int(self.settings.min_observations),
                    "min_confidence": float(self.settings.min_confidence),
                    "max_points": int(self.settings.max_points),
                    "max_position_std_m": float(self.settings.max_position_std_m),
                    "max_depth_std_m": float(self.settings.max_depth_std_m),
                    "min_view_diversity": float(self.settings.min_view_diversity),
                    "frame_subsample_target": int(self.settings.frame_subsample_target),
                    "max_sampled_pixels_per_frame": int(self.settings.max_sampled_pixels_per_frame),
                    "num_workers": int(worker_count),
                },
                "consistency_skipped_frames": list(skipped_frames),
                "consistency_replacement_map": {
                    str(k): int(v) for k, v in replacement_map.items()
                },
            }
        )
        resources.save_point_cloud_3d(cloud)

        if self.settings.export_glb:
            self._export_glbs(resources, cloud)

        LOG.info(
            "[PointCloud3D] completed: points=%d labels=%d median_obs=%.2f",
            int(cloud.points_world.shape[0]),
            int(np.unique(cloud.labels).size if cloud.labels.size else 0),
            float(np.median(cloud.observation_counts)) if cloud.observation_counts.size else 0.0,
        )

    @staticmethod
    def _mobile_labels_from_resources(
        resources: ResourceStore,
        frame_indices: Sequence[int],
        context: Mapping[str, object] | None,
    ) -> tuple[str, ...]:
        semantics_tool = None
        defaults = None
        if isinstance(context, Mapping):
            if context.get("semantics_tool") is not None:
                semantics_tool = str(context.get("semantics_tool"))
            if isinstance(context.get("semantic_role_defaults"), Mapping):
                defaults = context.get("semantic_role_defaults")
        metadata = None
        for frame_idx in frame_indices:
            semantics = resources.load_semantics2d(int(frame_idx))
            if semantics.metadata:
                metadata = semantics.metadata
                break
        return resolve_semantic_role_labels(
            "mobile",
            metadata=metadata,
            tool=semantics_tool,
            defaults=defaults,
            required=True,
            source_name="DensePointCloud3DProvider",
        )

    def _validate_settings(self) -> None:
        if self.settings.pixel_stride <= 0:
            raise ValueError("point_cloud_3d.pixel_stride must be > 0.")
        if self.settings.voxel_size_m <= 0.0:
            raise ValueError("point_cloud_3d.voxel_size_m must be > 0.")
        if self.settings.min_depth_m <= 0.0 or self.settings.max_depth_m <= self.settings.min_depth_m:
            raise ValueError("point_cloud_3d depth range is invalid.")
        if self.settings.min_observations <= 0:
            raise ValueError("point_cloud_3d.min_observations must be > 0.")
        if not (0.0 <= self.settings.min_confidence <= 1.0):
            raise ValueError("point_cloud_3d.min_confidence must be in [0, 1].")
        if self.settings.max_points <= 0:
            raise ValueError("point_cloud_3d.max_points must be > 0.")
        if self.settings.max_position_std_m <= 0.0:
            raise ValueError("point_cloud_3d.max_position_std_m must be > 0.")
        if self.settings.max_depth_std_m <= 0.0:
            raise ValueError("point_cloud_3d.max_depth_std_m must be > 0.")
        if self.settings.min_view_diversity < 0.0:
            raise ValueError("point_cloud_3d.min_view_diversity must be >= 0.")
        if self.settings.frame_subsample_target < 0:
            raise ValueError("point_cloud_3d.frame_subsample_target must be >= 0.")
        if self.settings.max_sampled_pixels_per_frame < 0:
            raise ValueError("point_cloud_3d.max_sampled_pixels_per_frame must be >= 0.")
        if self.settings.num_workers < 0:
            raise ValueError("point_cloud_3d.num_workers must be >= 0.")

    def _resolve_worker_count(self, frame_count: int) -> int:
        if frame_count <= 1:
            return 1
        if self.settings.num_workers > 0:
            return int(self.settings.num_workers)
        cpu_count = os.cpu_count() or 1
        return max(1, min(8, cpu_count, frame_count))

    def _select_frame_indices(self, frame_indices: Sequence[int]) -> list[int]:
        indices = [int(v) for v in frame_indices]
        target = int(self.settings.frame_subsample_target)
        if target <= 0 or len(indices) <= target:
            return indices
        stride = max(1, int(math.ceil(len(indices) / float(target))))
        selected = indices[::stride]
        if selected[-1] != indices[-1]:
            selected.append(indices[-1])
        return selected

    def _effective_pixel_stride(self, height: int, width: int) -> int:
        stride = max(1, int(self.settings.pixel_stride))
        budget = int(self.settings.max_sampled_pixels_per_frame)
        if budget <= 0:
            return stride
        while ((height + stride - 1) // stride) * ((width + stride - 1) // stride) > budget:
            stride += 1
        return stride

    def _selected_source_indices(
        self,
        frame_indices: Sequence[int],
        replacement_map: Mapping[int, int] | None,
    ) -> list[int]:
        replacement_map = replacement_map or {}
        selected: list[int] = []
        seen: set[int] = set()
        for frame_idx in frame_indices:
            source_idx = int(replacement_map.get(int(frame_idx), int(frame_idx)))
            if source_idx in seen:
                continue
            seen.add(source_idx)
            selected.append(source_idx)
        return selected

    def _load_source_frame_inputs(
        self,
        resources: ResourceStore,
        frame_indices: Sequence[int],
        *,
        pose_by_frame: Mapping[int, PoseSample],
    ) -> dict[int, tuple[np.ndarray, SemanticsData, PoseSample, np.ndarray]]:
        loaded: dict[int, tuple[np.ndarray, SemanticsData, PoseSample, np.ndarray]] = {}
        for frame_idx in frame_indices:
            pose = pose_by_frame.get(int(frame_idx))
            if pose is None:
                raise RuntimeError(
                    f"Pose for frame {int(frame_idx)} is missing for dense point cloud generation."
                )
            depth = resources.load_depth(int(frame_idx))
            semantics = resources.load_semantics2d(int(frame_idx))
            frame = resources.load_frame(int(frame_idx))
            if frame.image is None:
                raise RuntimeError(
                    f"Frame {int(frame_idx)} image missing for dense point cloud provider."
                )
            loaded[int(frame_idx)] = (
                np.asarray(depth.depth),
                semantics,
                pose,
                np.asarray(frame.image),
            )
        return loaded

    def _lift_cached_source_frame(
        self,
        *,
        source_idx: int,
        source_inputs: tuple[np.ndarray, SemanticsData, PoseSample, np.ndarray],
        intrinsics: IntrinsicsData,
        class_index: Mapping[int, int],
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        depth, semantics, pose, image = source_inputs
        lifted = self._lift_frame(
            frame_index=int(source_idx),
            depth=depth,
            semantics=semantics,
            resources=None,
            pose=pose,
            intrinsics=intrinsics,
            image=image,
            class_index=class_index,
        )
        return (int(source_idx), *lifted)

    @staticmethod
    def _validate_intrinsics(intrinsics: IntrinsicsData) -> None:
        k = np.asarray(intrinsics.matrix, dtype=np.float32)
        if k.shape != (3, 3):
            raise ValueError(f"Intrinsics matrix must be 3x3, got {k.shape}.")
        if float(k[0, 0]) <= 0.0 or float(k[1, 1]) <= 0.0:
            raise ValueError("Intrinsics focal lengths must be positive.")

    @staticmethod
    def _label_map(semantics: SemanticsData) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for seg in semantics.segments:
            if seg.label_id is None:
                mapping[int(seg.segment_id)] = str(seg.label).lower()
            else:
                mapping[int(seg.label_id)] = str(seg.label).lower()
        return mapping

    def _build_label_vocabulary(
        self,
        resources: ResourceStore,
        frame_indices: Sequence[int],
    ) -> Dict[int, str]:
        label_map: Dict[int, str] = {}
        for frame_idx in frame_indices:
            semantics = resources.load_semantics2d(int(frame_idx))
            ids = semantics.label_ids if semantics.label_ids is not None else semantics.segment_ids
            if ids is None:
                continue
            ids_arr = np.asarray(ids, dtype=np.int32)
            unique_ids = np.unique(ids_arr[ids_arr >= 0])
            frame_map = self._label_map(semantics)
            for label_id in unique_ids.tolist():
                label_map[int(label_id)] = frame_map.get(int(label_id), f"class_{int(label_id)}")
        return label_map

    def _cross_run_payload(
        self,
        resources: ResourceStore,
    ) -> dict[str, Any] | None:
        if self._cache_manager is None or not self._cache_manager.enabled:
            return None
        repo_root = Path(__file__).resolve().parents[4]
        return {
            "settings": {
                "pixel_stride": int(self.settings.pixel_stride),
                "voxel_size_m": float(self.settings.voxel_size_m),
                "min_depth_m": float(self.settings.min_depth_m),
                "max_depth_m": float(self.settings.max_depth_m),
                "min_observations": int(self.settings.min_observations),
                "min_confidence": float(self.settings.min_confidence),
                "min_total_points": int(self.settings.min_total_points),
                "max_points": int(self.settings.max_points),
                "export_glb": bool(self.settings.export_glb),
                "glb_max_points": int(self.settings.glb_max_points),
                "seed": int(self.settings.seed),
                "max_position_std_m": float(self.settings.max_position_std_m),
                "max_depth_std_m": float(self.settings.max_depth_std_m),
                "min_view_diversity": float(self.settings.min_view_diversity),
                "frame_subsample_target": int(self.settings.frame_subsample_target),
                "max_sampled_pixels_per_frame": int(self.settings.max_sampled_pixels_per_frame),
            },
            "depth_dir": self._cache_manager.directory_signature(resources.base_dir(ResourceKind.DEPTH)),
            "frames_dir": self._cache_manager.directory_signature(resources.base_dir(ResourceKind.FRAMES)),
            "trajectory": self._cache_manager.resource_file_key_signature(
                resources.path_for(ResourceKind.TRAJECTORY),
                logical_name="standard/trajectory/poses.npz",
            ),
            "intrinsics": self._cache_manager.resource_file_key_signature(
                resources.path_for(ResourceKind.INTRINSICS),
                logical_name="standard/intrinsics/intrinsics.npz",
            ),
            "semantics_dir": self._cache_manager.directory_signature(
                resources.base_dir(ResourceKind.SEMANTICS_2D)
            ),
            "provider_script": self._cache_manager.script_key_signature(
                Path(__file__),
                repo_root=repo_root,
            ),
            "voxel_grid_script": self._cache_manager.script_key_signature(
                Path(__file__).with_name("voxel_grid.py"),
                repo_root=repo_root,
            ),
            "glb_writer_script": self._cache_manager.script_key_signature(
                repo_root / "src" / "pemoin" / "visualization" / "point_cloud_glb.py",
                repo_root=repo_root,
            ),
        }

    def _try_materialize_cached_outputs(self, resources: ResourceStore) -> bool:
        self._cache_payload = self._cross_run_payload(resources)
        if self._cache_payload is None or self._cache_manager is None:
            return False
        self._cache_signature = self._cache_manager.signature(
            "point_cloud_3d",
            self._cache_payload,
        )
        lookup = self._cache_manager.lookup("point_cloud_3d", self._cache_signature)
        self._cache_status.update(
            {
                "cross_run_cache_signature": self._cache_signature,
                "cross_run_cache_hit": lookup.hit,
                "cross_run_cache_entry": str(lookup.entry_dir),
                "cross_run_cache_validation": lookup.reason,
            }
        )
        if not lookup.hit:
            self._cache_status["cross_run_cache_reason"] = lookup.reason
            return False
        materialized = self._cache_manager.materialize(
            "point_cloud_3d",
            self._cache_signature,
            run_root=resources.root,
        )
        self._cache_status["cross_run_cache_materialized"] = materialized
        LOG.info("Reused cross-run point_cloud_3d cache at '%s'.", lookup.entry_dir)
        return True

    def get_cross_run_cache_status(self) -> dict[str, Any]:
        return dict(self._cache_status)

    def get_cross_run_cache_spec(self, resources: ResourceStore | None) -> dict[str, Any] | None:
        if (
            resources is None
            or self._cache_manager is None
            or not self._cache_manager.enabled
            or self._cache_signature is None
            or self._cache_payload is None
        ):
            return None
        cloud_path = resources.path_for(ResourceKind.POINT_CLOUD_3D)
        artifacts: dict[str, Path] = {}
        if cloud_path.exists():
            artifacts.update(
                self._cache_manager.collect_file(
                    cloud_path,
                    relpath="standard/point_cloud_3d/cloud.npz",
                )
            )
        for path, relpath in (
            (
                resources.rgb_pointcloud_artifact_path(),
                "artifacts/geometry/point_cloud/rgb_pointcloud.glb",
            ),
            (
                resources.semantic_pointcloud_artifact_path(),
                "artifacts/geometry/point_cloud/semantic_pointcloud.glb",
            ),
            (resources.rgb_pointcloud_path(), "rgb_pointcloud.glb"),
            (resources.semantic_pointcloud_path(), "semantic_pointcloud.glb"),
        ):
            if path.exists():
                artifacts.update(self._cache_manager.collect_file(path, relpath=relpath))
        ready = cloud_path.exists()
        spec = {
            "provider_id": "point_cloud_3d",
            "signature": self._cache_signature,
            "payload": self._cache_payload,
            "artifacts": artifacts,
            "ready": ready,
            "source_summary": {
                "profile": self._profile_name,
                "run_root": str(resources.root),
            },
        }
        if not ready:
            spec["not_ready_reason"] = "point-cloud-artifacts-missing"
        return spec

    @staticmethod
    def _exclude_mobile_labels(
        label_vocabulary: Mapping[int, str],
        mobile_labels: Sequence[str],
    ) -> Dict[int, str]:
        mobile = {str(name).strip().lower() for name in mobile_labels if str(name).strip()}
        if not mobile:
            return {int(k): str(v) for k, v in label_vocabulary.items()}
        filtered = {
            int(label_id): str(label_name)
            for label_id, label_name in label_vocabulary.items()
            if str(label_name).strip().lower() not in mobile
        }
        excluded = sorted(
            {
                str(label_name).strip().lower()
                for label_id, label_name in label_vocabulary.items()
                if int(label_id) not in filtered
            }
        )
        if excluded:
            LOG.info(
                "[PointCloud3D] excluded mobile classes from fusion: %s",
                json.dumps(excluded),
            )
        return filtered

    def _lift_frame(
        self,
        *,
        frame_index: int,
        depth: np.ndarray,
        semantics: SemanticsData,
        resources: ResourceStore | None = None,
        pose: PoseSample,
        intrinsics: IntrinsicsData,
        image: np.ndarray,
        class_index: Mapping[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        depth_map = np.asarray(depth, dtype=np.float32)
        label_ids, confidence_map = self._resolve_label_and_confidence_maps(
            resources,
            semantics,
        )
        label_ids = np.asarray(label_ids, dtype=np.int32)
        confidence_map = np.asarray(confidence_map, dtype=np.float32)

        if depth_map.shape[:2] != label_ids.shape[:2]:
            raise RuntimeError(
                f"Frame {frame_index}: depth and semantics resolution mismatch "
                f"({depth_map.shape[:2]} vs {label_ids.shape[:2]})."
            )
        if image.shape[:2] != depth_map.shape[:2]:
            raise RuntimeError(
                f"Frame {frame_index}: frame and depth resolution mismatch "
                f"({image.shape[:2]} vs {depth_map.shape[:2]})."
            )

        stride = self._effective_pixel_stride(depth_map.shape[0], depth_map.shape[1])
        ys = np.arange(0, depth_map.shape[0], stride, dtype=np.int32)
        xs = np.arange(0, depth_map.shape[1], stride, dtype=np.int32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        y_flat = yy.reshape(-1)
        x_flat = xx.reshape(-1)

        sampled_depth = depth_map[y_flat, x_flat]
        sampled_labels = label_ids[y_flat, x_flat]
        sampled_conf = np.clip(confidence_map[y_flat, x_flat], 0.0, 1.0)
        sampled_colors = image[y_flat, x_flat, :3].astype(np.uint8)

        valid_depth = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= float(self.settings.min_depth_m))
            & (sampled_depth <= float(self.settings.max_depth_m))
        )
        if float(np.mean(valid_depth)) < 0.10:
            raise RuntimeError(
                f"Frame {frame_index}: fewer than 10% sampled pixels have valid depth "
                f"({float(np.mean(valid_depth)):.3f})."
            )

        valid = valid_depth & (sampled_labels >= 0)
        sampled_depth = sampled_depth[valid]
        sampled_labels = sampled_labels[valid]
        sampled_conf = sampled_conf[valid]
        sampled_colors = sampled_colors[valid]
        x_valid = x_flat[valid].astype(np.float32)
        y_valid = y_flat[valid].astype(np.float32)

        if sampled_depth.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                {
                    "lifted": int(valid_depth.sum()),
                    "valid": 0,
                    "valid_ratio": 0.0,
                    "label_distribution": {},
                },
            )

        pose_meta = dict(pose.metadata or {})
        if CAMERA_CONVENTION_KEY not in pose_meta:
            raise ValueError(
                f"Frame {frame_index}: missing pose metadata key {CAMERA_CONVENTION_KEY!r}."
            )
        convention_raw = str(pose_meta.get(CAMERA_CONVENTION_KEY, "blender")).strip().lower()
        if convention_raw and convention_raw not in CAMERA_CONVENTION_VALUES:
            raise ValueError(
                f"Frame {frame_index}: unsupported camera convention {convention_raw!r}."
            )
        camera_convention = normalize_camera_convention(convention_raw or "blender")

        uv = np.stack([x_valid, y_valid], axis=1)
        cam_points = backproject_uv_depth_to_camera(
            uv,
            sampled_depth.astype(np.float32),
            intrinsics.matrix,
            camera_convention=camera_convention,
        )

        z_cam = cam_points[:, 2]
        if camera_convention == "blender":
            front_mask = z_cam < 0.0
        else:
            front_mask = z_cam > 0.0
        front_ratio = float(np.mean(front_mask)) if front_mask.size else 0.0
        if front_ratio < 0.95:
            raise RuntimeError(
                f"Frame {frame_index}: backprojected front-facing ratio too low ({front_ratio:.3f})."
            )

        cam_points = cam_points[front_mask]
        sampled_labels = sampled_labels[front_mask]
        sampled_conf = sampled_conf[front_mask]
        sampled_colors = sampled_colors[front_mask]
        sampled_depth = sampled_depth[front_mask]

        if cam_points.shape[0] == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                {
                    "lifted": int(valid_depth.sum()),
                    "valid": 0,
                    "valid_ratio": 0.0,
                    "label_distribution": {},
                },
            )

        world_points = camera_to_world(cam_points, pose.camera_to_world).astype(np.float32)

        # Keep only labels present in the class vocabulary.
        keep = np.array([int(label) in class_index for label in sampled_labels.tolist()], dtype=bool)
        world_points = world_points[keep]
        sampled_labels = sampled_labels[keep]
        sampled_conf = sampled_conf[keep]
        sampled_colors = sampled_colors[keep]
        sampled_depth = sampled_depth[keep]
        cam_points = cam_points[keep]

        label_distribution: Dict[str, int] = {}
        if sampled_labels.size > 0:
            unique, counts = np.unique(sampled_labels, return_counts=True)
            for label_id, count in zip(unique.tolist(), counts.tolist()):
                label_distribution[str(int(label_id))] = int(count)

        return (
            world_points,
            sampled_colors,
            sampled_labels.astype(np.int32),
            sampled_conf.astype(np.float32),
            sampled_depth.astype(np.float32),
            (-cam_points / np.maximum(np.linalg.norm(cam_points, axis=1, keepdims=True), 1e-6)).astype(np.float32),
            {
                "lifted": int(valid_depth.sum()),
                "valid": int(world_points.shape[0]),
                "valid_ratio": float(world_points.shape[0]) / max(1, int(valid_depth.sum())),
                "label_distribution": label_distribution,
            },
        )

    def _resolve_label_and_confidence_maps(
        self,
        resources: ResourceStore | None,
        semantics: SemanticsData,
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = semantics.label_ids if semantics.label_ids is not None else semantics.segment_ids
        if ids is None:
            raise RuntimeError(f"Frame {semantics.frame_index}: semantics has no label_ids/segment_ids.")

        ids_arr = np.asarray(ids, dtype=np.int32)
        confidence = np.ones(ids_arr.shape, dtype=np.float32)

        aux = None
        if resources is not None:
            try:
                aux = resources.load_semantics_aux(int(semantics.frame_index))
            except Exception:
                aux = None
        if aux is not None:
            probs = aux.class_probabilities
            if probs is not None and probs.ndim == 3 and probs.shape[1:] == ids_arr.shape:
                class_ids = (
                    np.asarray(aux.class_ids, dtype=np.int32)
                    if aux.class_ids is not None
                    else np.arange(probs.shape[0], dtype=np.int32)
                )
                class_lookup = {int(label_id): idx for idx, label_id in enumerate(class_ids.tolist())}
                valid = np.isin(ids_arr, class_ids)
                confidence = np.zeros(ids_arr.shape, dtype=np.float32)
                rows, cols = np.nonzero(valid)
                for row, col in zip(rows.tolist(), cols.tolist()):
                    confidence[row, col] = probs[class_lookup[int(ids_arr[row, col])], row, col]
                return ids_arr, np.clip(confidence, 0.0, 1.0)
            if aux.confidence is not None and aux.confidence.shape == ids_arr.shape:
                return ids_arr, np.clip(np.asarray(aux.confidence, dtype=np.float32), 0.0, 1.0)

        if semantics.segment_ids is not None and semantics.segments:
            segment_ids = np.asarray(semantics.segment_ids, dtype=np.int32)
            if segment_ids.shape == ids_arr.shape:
                score_map = np.zeros(ids_arr.shape, dtype=np.float32)
                for seg in semantics.segments:
                    sid = int(seg.segment_id)
                    score = float(np.clip(seg.score, 0.0, 1.0))
                    score_map[segment_ids == sid] = score
                confidence = np.maximum(confidence, score_map)

        return ids_arr, np.clip(confidence, 0.0, 1.0)

    def _validate_cloud(self, cloud: PointCloud3DData) -> None:
        count = int(cloud.points_world.shape[0])
        if count < int(self.settings.min_total_points):
            raise RuntimeError(
                f"Dense point cloud too sparse: {count} < min_total_points={self.settings.min_total_points}."
            )
        unique_labels = np.unique(cloud.labels)
        if unique_labels.size < 2:
            raise RuntimeError(
                "Dense point cloud semantic distribution is degenerate: fewer than 2 distinct labels present."
            )
        if cloud.observation_counts.size == 0 or float(np.median(cloud.observation_counts)) <= 1.0:
            raise RuntimeError(
                "Dense point cloud fusion quality check failed: median observation_count must be > 1."
            )

    def _export_glbs(self, resources: ResourceStore, cloud: PointCloud3DData) -> None:
        if cloud.points_world.shape[0] == 0:
            return
        rng = np.random.default_rng(self.settings.seed)
        semantic_artifact_path = resources.semantic_pointcloud_artifact_path()
        rgb_artifact_path = resources.rgb_pointcloud_artifact_path()

        semantic_colors = semantic_colors_from_labels(
            cloud.labels,
            label_names=cloud.label_names,
        )
        semantic_path = write_point_cloud_glb(
            semantic_artifact_path,
            points=cloud.points_world,
            colors=semantic_colors,
            max_points=self.settings.glb_max_points,
            rng=rng,
        )
        rgb_path = write_point_cloud_glb(
            rgb_artifact_path,
            points=cloud.points_world,
            colors=cloud.colors,
            max_points=self.settings.glb_max_points,
            rng=rng,
        )
        shutil.copy2(semantic_path, resources.semantic_pointcloud_path())
        shutil.copy2(rgb_path, resources.rgb_pointcloud_path())


def register_point_cloud_3d_provider_builders(factory) -> None:
    factory.register(
        "DensePointCloud3DProvider",
        lambda binding, context: DensePointCloud3DProvider(binding.settings),
    )
