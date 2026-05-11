"""Consistency-aware voxel fusion for dense multi-view point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from pemoin.data.contracts import PointCloud3DData


@dataclass(slots=True)
class VoxelState:
    position_sum: np.ndarray
    position_sq_sum: np.ndarray
    color_sum: np.ndarray
    weight_sum: float
    log_odds: np.ndarray
    observation_count: int
    depth_sum: float
    depth_sq_sum: float
    view_dir_sum: np.ndarray
    view_outer_sum: np.ndarray


class VoxelGrid:
    """Sparse hash-grid with uncertainty-aware semantic and geometric fusion."""

    def __init__(
        self,
        *,
        voxel_size_m: float,
        class_ids: Sequence[int],
        label_names: Mapping[int, str],
    ):
        if voxel_size_m <= 0.0:
            raise ValueError(f"voxel_size_m must be > 0, got {voxel_size_m}.")
        class_id_array = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        if class_id_array.size == 0:
            raise ValueError("VoxelGrid requires at least one semantic class id.")
        self.voxel_size_m = float(voxel_size_m)
        self.class_ids = class_id_array
        self.label_names = {int(k): str(v) for k, v in label_names.items()}
        self._class_index: Dict[int, int] = {
            int(class_id): int(idx) for idx, class_id in enumerate(self.class_ids.tolist())
        }
        self._grid: Dict[Tuple[int, int, int], VoxelState] = {}

    def integrate_frame(
        self,
        *,
        points_world: np.ndarray,
        colors: np.ndarray,
        label_ids: np.ndarray,
        confidences: np.ndarray,
        weights: np.ndarray,
        depth_values: np.ndarray | None = None,
        view_dirs: np.ndarray | None = None,
    ) -> None:
        points = np.asarray(points_world, dtype=np.float32)
        rgb = np.asarray(colors, dtype=np.uint8)
        labels = np.asarray(label_ids, dtype=np.int32).reshape(-1)
        conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        count = points.shape[0]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_world must have shape (N, 3), got {points.shape}.")
        if rgb.shape != (count, 3):
            raise ValueError(f"colors must have shape ({count}, 3), got {rgb.shape}.")
        if labels.shape[0] != count or conf.shape[0] != count or w.shape[0] != count:
            raise ValueError("integrate_frame input lengths must match.")
        if count == 0:
            return

        if depth_values is None:
            depth_values = np.linalg.norm(points, axis=1).astype(np.float32)
        depth_arr = np.asarray(depth_values, dtype=np.float32).reshape(-1)
        if depth_arr.shape[0] != count:
            raise ValueError("depth_values length must match points.")

        if view_dirs is None:
            dirs = points.astype(np.float32)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms = np.where(norms <= 1e-6, 1.0, norms)
            view_arr = (dirs / norms).astype(np.float32)
        else:
            view_arr = np.asarray(view_dirs, dtype=np.float32)
            if view_arr.shape != (count, 3):
                raise ValueError(f"view_dirs must have shape ({count}, 3), got {view_arr.shape}.")
            norms = np.linalg.norm(view_arr, axis=1, keepdims=True)
            norms = np.where(norms <= 1e-6, 1.0, norms)
            view_arr = (view_arr / norms).astype(np.float32)

        conf = np.clip(conf, 1e-4, 1.0 - 1e-4)
        w = np.clip(w, 1e-6, np.inf)
        voxel_keys = np.floor(points / self.voxel_size_m).astype(np.int64)

        for idx in range(count):
            label_id = int(labels[idx])
            class_idx = self._class_index.get(label_id)
            if class_idx is None:
                continue
            key = (int(voxel_keys[idx, 0]), int(voxel_keys[idx, 1]), int(voxel_keys[idx, 2]))
            state = self._grid.get(key)
            if state is None:
                state = VoxelState(
                    position_sum=np.zeros((3,), dtype=np.float64),
                    position_sq_sum=np.zeros((3,), dtype=np.float64),
                    color_sum=np.zeros((3,), dtype=np.float64),
                    weight_sum=0.0,
                    log_odds=np.zeros((self.class_ids.size,), dtype=np.float32),
                    observation_count=0,
                    depth_sum=0.0,
                    depth_sq_sum=0.0,
                    view_dir_sum=np.zeros((3,), dtype=np.float64),
                    view_outer_sum=np.zeros((3, 3), dtype=np.float64),
                )
                self._grid[key] = state

            weight = float(w[idx])
            point = points[idx].astype(np.float64)
            color = rgb[idx].astype(np.float64)
            depth = float(depth_arr[idx])
            view_dir = view_arr[idx].astype(np.float64)
            evidence = float(np.log(conf[idx] / (1.0 - conf[idx])))

            state.position_sum += point * weight
            state.position_sq_sum += (point * point) * weight
            state.color_sum += color * weight
            state.weight_sum += weight
            state.log_odds[class_idx] += evidence
            state.observation_count += 1
            state.depth_sum += depth
            state.depth_sq_sum += depth * depth
            state.view_dir_sum += view_dir
            state.view_outer_sum += np.outer(view_dir, view_dir)

    def extract_cloud(
        self,
        *,
        min_observations: int,
        min_confidence: float,
        max_points: int,
        rng: np.random.Generator,
        max_position_std_m: float = 0.20,
        max_depth_std_m: float = 1.50,
        min_view_diversity: float = 0.0,
    ) -> PointCloud3DData:
        min_obs = max(1, int(min_observations))
        min_conf = float(np.clip(min_confidence, 0.0, 1.0))
        max_keep = max(1, int(max_points))

        points: list[np.ndarray] = []
        labels: list[int] = []
        confidences: list[float] = []
        colors: list[np.ndarray] = []
        obs_counts: list[int] = []

        rejected_low_obs = 0
        rejected_conf = 0
        rejected_position_std = 0
        rejected_depth_std = 0
        rejected_view_div = 0

        for state in self._grid.values():
            if state.observation_count < min_obs or state.weight_sum <= 0.0:
                rejected_low_obs += 1
                continue

            map_class = int(np.argmax(state.log_odds))
            map_label = int(self.class_ids[map_class])
            map_conf = float(1.0 / (1.0 + np.exp(-float(state.log_odds[map_class]))))
            if map_conf < min_conf:
                rejected_conf += 1
                continue

            mean_pos = (state.position_sum / state.weight_sum).astype(np.float32)
            pos_var = np.maximum(state.position_sq_sum / state.weight_sum - (mean_pos.astype(np.float64) ** 2), 0.0)
            pos_std = float(np.sqrt(float(np.max(pos_var))))
            if pos_std > float(max_position_std_m):
                rejected_position_std += 1
                continue

            mean_depth = state.depth_sum / max(float(state.observation_count), 1.0)
            depth_var = max(state.depth_sq_sum / max(float(state.observation_count), 1.0) - mean_depth * mean_depth, 0.0)
            depth_std = float(np.sqrt(depth_var))
            if depth_std > float(max_depth_std_m):
                rejected_depth_std += 1
                continue

            mean_view = state.view_dir_sum / max(float(state.observation_count), 1.0)
            mean_outer = state.view_outer_sum / max(float(state.observation_count), 1.0)
            cov = mean_outer - np.outer(mean_view, mean_view)
            diversity = float(max(0.0, np.trace(cov)))
            if diversity < float(min_view_diversity):
                rejected_view_div += 1
                continue

            points.append(mean_pos)
            labels.append(map_label)
            confidences.append(map_conf)
            colors.append(np.clip(state.color_sum / state.weight_sum, 0.0, 255.0).astype(np.uint8))
            obs_counts.append(int(state.observation_count))

        if not points:
            return PointCloud3DData(
                points_world=np.zeros((0, 3), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.int32),
                label_confidences=np.zeros((0,), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                label_names=dict(self.label_names),
                observation_counts=np.zeros((0,), dtype=np.int32),
                metadata={
                    "fusion_mode": "consistency_aware_voxel",
                    "voxel_size_m": float(self.voxel_size_m),
                    "num_voxels_total": int(len(self._grid)),
                    "rejection_stats": {
                        "low_observations": int(rejected_low_obs),
                        "low_confidence": int(rejected_conf),
                        "position_std": int(rejected_position_std),
                        "depth_std": int(rejected_depth_std),
                        "view_diversity": int(rejected_view_div),
                    },
                },
            )

        points_arr = np.stack(points, axis=0)
        labels_arr = np.asarray(labels, dtype=np.int32)
        conf_arr = np.asarray(confidences, dtype=np.float32)
        colors_arr = np.stack(colors, axis=0).astype(np.uint8)
        obs_arr = np.asarray(obs_counts, dtype=np.int32)

        if points_arr.shape[0] > max_keep:
            choice = rng.choice(points_arr.shape[0], size=max_keep, replace=False)
            points_arr = points_arr[choice]
            labels_arr = labels_arr[choice]
            conf_arr = conf_arr[choice]
            colors_arr = colors_arr[choice]
            obs_arr = obs_arr[choice]

        return PointCloud3DData(
            points_world=points_arr,
            labels=labels_arr,
            label_confidences=conf_arr,
            colors=colors_arr,
            label_names=dict(self.label_names),
            observation_counts=obs_arr,
            metadata={
                "fusion_mode": "consistency_aware_voxel",
                "voxel_size_m": float(self.voxel_size_m),
                "num_voxels_total": int(len(self._grid)),
                "num_points_exported": int(points_arr.shape[0]),
                "rejection_stats": {
                    "low_observations": int(rejected_low_obs),
                    "low_confidence": int(rejected_conf),
                    "position_std": int(rejected_position_std),
                    "depth_std": int(rejected_depth_std),
                    "view_diversity": int(rejected_view_div),
                },
                "consistency_filters": {
                    "max_position_std_m": float(max_position_std_m),
                    "max_depth_std_m": float(max_depth_std_m),
                    "min_view_diversity": float(min_view_diversity),
                },
            },
        )

    def filter_by_labels(self, label_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        wanted = {str(name).strip().lower() for name in label_names if str(name).strip()}
        if not wanted:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        accepted_label_ids = {
            int(label_id)
            for label_id, label_name in self.label_names.items()
            if str(label_name).strip().lower() in wanted
        }
        if not accepted_label_ids:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        points: list[np.ndarray] = []
        confidences: list[float] = []
        for state in self._grid.values():
            if state.weight_sum <= 0.0:
                continue
            map_class = int(np.argmax(state.log_odds))
            map_label = int(self.class_ids[map_class])
            if map_label not in accepted_label_ids:
                continue
            points.append((state.position_sum / state.weight_sum).astype(np.float32))
            confidences.append(float(1.0 / (1.0 + np.exp(-float(state.log_odds[map_class])))))

        if not points:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.stack(points, axis=0), np.asarray(confidences, dtype=np.float32)
