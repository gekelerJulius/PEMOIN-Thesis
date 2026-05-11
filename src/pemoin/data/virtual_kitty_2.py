"""Virtual KITTI 2 dataset helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

from pemoin.utils.logging import get_logger

LOG = get_logger()

DATASET_SUBDIRS = {
    "rgb": "vkitti_2.0.3_rgb",
    "depth": "vkitti_2.0.3_depth",
    "class_segmentation": "vkitti_2.0.3_classSegmentation",
    "instance_segmentation": "vkitti_2.0.3_instanceSegmentation",
    "textgt": "vkitti_2.0.3_textgt",
}

_SELECTION_CACHE: Dict[Tuple, "VirtualKitty2Selection"] = {}
_DATASET_CACHE: Dict[Tuple, "VirtualKitty2Dataset"] = {}


@dataclass(frozen=True)
class VirtualKitty2Selection:
    scene: str
    variation: str
    camera: int

    @property
    def camera_dir(self) -> str:
        return f"Camera_{self.camera}"


class VirtualKitty2Dataset:
    def __init__(self, root: Path, selection: VirtualKitty2Selection) -> None:
        self.root = _normalize_vkitti_root(root)
        self.selection = selection
        self.rgb_root = _subdir_if_exists(self.root, DATASET_SUBDIRS["rgb"])
        if self.rgb_root is None:
            raise FileNotFoundError(f"Virtual KITTI 2 rgb folder not found under {self.root}.")
        self.depth_root = _subdir_if_exists(self.root, DATASET_SUBDIRS["depth"])
        self.class_root = _subdir_if_exists(self.root, DATASET_SUBDIRS["class_segmentation"])
        self.instance_root = _subdir_if_exists(self.root, DATASET_SUBDIRS["instance_segmentation"])
        self.textgt_root = _subdir_if_exists(self.root, DATASET_SUBDIRS["textgt"])
        self._frame_paths = _collect_frame_paths(
            self.rgb_root, selection.scene, selection.variation, selection.camera_dir
        )
        if not self._frame_paths:
            raise FileNotFoundError(
                f"No rgb frames found for {selection.scene}/{selection.variation}/{selection.camera_dir} under "
                f"{self.rgb_root}."
            )
        self._intrinsics_cache: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        self._extrinsics_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._colors_cache: Optional[List[Tuple[str, Tuple[int, int, int]]]] = None
        self._info_cache: Optional[Dict[int, Mapping[str, str]]] = None

    def frame_indices(self) -> List[int]:
        return sorted(self._frame_paths.keys())

    def available_indices(self, required: Iterable[str]) -> List[int]:
        required_set = _normalize_required_resources(required)
        indices = self.frame_indices()
        if not required_set:
            return indices
        missing = []
        if "depth" in required_set and self.depth_root is None:
            missing.append("depth")
        if "class_segmentation" in required_set and self.class_root is None:
            missing.append("class_segmentation")
        if "instance_segmentation" in required_set and self.instance_root is None:
            missing.append("instance_segmentation")
        if missing:
            raise FileNotFoundError(f"Virtual KITTI 2 missing required folders: {', '.join(missing)}")
        filtered: List[int] = []
        for index in indices:
            if "depth" in required_set:
                if not self.depth_path(index).exists():
                    continue
            if "class_segmentation" in required_set:
                if not self.class_segmentation_path(index).exists():
                    continue
            if "instance_segmentation" in required_set:
                if not self.instance_segmentation_path(index).exists():
                    continue
            filtered.append(index)
        return filtered

    def frame_path(self, frame_index: int) -> Path:
        return self._frame_paths[int(frame_index)]

    def depth_path(self, frame_index: int) -> Path:
        if self.depth_root is None:
            raise FileNotFoundError("Virtual KITTI 2 depth folder is missing.")
        return _frame_path(
            self.depth_root,
            self.selection.scene,
            self.selection.variation,
            "depth",
            self.selection.camera_dir,
            "depth",
            frame_index,
            ".png",
        )

    def class_segmentation_path(self, frame_index: int) -> Path:
        if self.class_root is None:
            raise FileNotFoundError("Virtual KITTI 2 class segmentation folder is missing.")
        return _frame_path(
            self.class_root,
            self.selection.scene,
            self.selection.variation,
            "classSegmentation",
            self.selection.camera_dir,
            "classgt",
            frame_index,
            ".png",
        )

    def instance_segmentation_path(self, frame_index: int) -> Path:
        if self.instance_root is None:
            raise FileNotFoundError("Virtual KITTI 2 instance segmentation folder is missing.")
        return _frame_path(
            self.instance_root,
            self.selection.scene,
            self.selection.variation,
            "instanceSegmentation",
            self.selection.camera_dir,
            "instancegt",
            frame_index,
            ".png",
        )

    def intrinsics(self, frame_index: int) -> Tuple[float, float, float, float]:
        key = (int(frame_index), self.selection.camera)
        if key in self._intrinsics_cache:
            return self._intrinsics_cache[key]
        if self.textgt_root is None:
            raise FileNotFoundError("Virtual KITTI 2 textgt folder is missing.")
        intrinsic_path = self.textgt_root / self.selection.scene / self.selection.variation / "intrinsic.txt"
        table = _read_table(intrinsic_path)
        for row in table:
            frame = int(row["frame"])
            cam = int(row["cameraID"])
            self._intrinsics_cache[(frame, cam)] = (
                float(row["K[0,0]"]),
                float(row["K[1,1]"]),
                float(row["K[0,2]"]),
                float(row["K[1,2]"]),
            )
        try:
            return self._intrinsics_cache[key]
        except KeyError as exc:
            raise KeyError(f"Intrinsics missing for frame {frame_index} camera {self.selection.camera}.") from exc

    def extrinsic(self, frame_index: int) -> np.ndarray:
        key = (int(frame_index), self.selection.camera)
        if key in self._extrinsics_cache:
            return self._extrinsics_cache[key]
        if self.textgt_root is None:
            raise FileNotFoundError("Virtual KITTI 2 textgt folder is missing.")
        extrinsic_path = self.textgt_root / self.selection.scene / self.selection.variation / "extrinsic.txt"
        for row in _read_extrinsics(extrinsic_path):
            self._extrinsics_cache[(row[0], row[1])] = row[2]
        try:
            return self._extrinsics_cache[key]
        except KeyError as exc:
            raise KeyError(f"Extrinsics missing for frame {frame_index} camera {self.selection.camera}.") from exc

    def colors(self) -> List[Tuple[str, Tuple[int, int, int]]]:
        if self._colors_cache is not None:
            return self._colors_cache
        if self.textgt_root is None:
            raise FileNotFoundError("Virtual KITTI 2 textgt folder is missing.")
        colors_path = self.textgt_root / self.selection.scene / self.selection.variation / "colors.txt"
        entries = []
        for row in _read_table(colors_path):
            entries.append(
                (
                    str(row["Category"]),
                    (int(row["r"]), int(row["g"]), int(row["b"])),
                )
            )
        if not entries:
            raise ValueError(f"No entries found in {colors_path}.")
        self._colors_cache = entries
        return entries

    def info(self) -> Dict[int, Mapping[str, str]]:
        if self._info_cache is not None:
            return self._info_cache
        if self.textgt_root is None:
            raise FileNotFoundError("Virtual KITTI 2 textgt folder is missing.")
        info_path = self.textgt_root / self.selection.scene / self.selection.variation / "info.txt"
        entries: Dict[int, Mapping[str, str]] = {}
        for row in _read_table(info_path):
            track_id = int(row["trackID"])
            entries[track_id] = {
                "label": str(row.get("label", "")),
                "model": str(row.get("model", "")),
                "color": str(row.get("color", "")),
            }
        self._info_cache = entries
        return entries

    def load_instance_indices(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        return np.asarray(image, dtype=np.int32)


def resolve_vkitti2_dataset(settings: Mapping[str, object]) -> VirtualKitty2Dataset:
    root = _resolve_root_from_settings(settings)
    selection = resolve_vkitti2_selection(root, settings)
    cache_key = (str(root), selection.scene, selection.variation, selection.camera)
    cached = _DATASET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dataset = VirtualKitty2Dataset(root, selection)
    _DATASET_CACHE[cache_key] = dataset
    return dataset


def resolve_vkitti2_selection(root: Path, settings: Mapping[str, object]) -> VirtualKitty2Selection:
    root = _normalize_vkitti_root(root)
    random_selection = bool(settings.get("random_selection", False))
    random_seed = settings.get("random_seed")
    scene_raw = settings.get("scene")
    variation_raw = settings.get("variation")
    camera_raw = settings.get("camera")
    cache_key = (str(root), random_selection, random_seed, scene_raw, variation_raw, camera_raw)
    cached = _SELECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rgb_root = _subdir_if_exists(root, DATASET_SUBDIRS["rgb"])
    if rgb_root is None:
        raise FileNotFoundError(f"Virtual KITTI 2 rgb folder not found under {root}.")
    scene, variation = _split_scene_variation(scene_raw, variation_raw)
    if random_selection:
        selection = _random_selection(rgb_root, seed=random_seed)
        LOG.info(
            "Virtual KITTI 2 random selection: scene=%s variation=%s camera=%s",
            selection.scene,
            selection.variation,
            selection.camera,
        )
    else:
        if scene is None or variation is None:
            raise ValueError("Virtual KITTI 2 selection requires 'scene' and 'variation' when random_selection is false.")
        camera = _normalize_camera(camera_raw)
        selection = VirtualKitty2Selection(scene=scene, variation=variation, camera=camera)
        _validate_selection(rgb_root, selection)
    _SELECTION_CACHE[cache_key] = selection
    return selection


def _random_selection(rgb_root: Path, seed: Optional[object]) -> VirtualKitty2Selection:
    import random

    rng = random.Random(seed)
    scenes = _list_scenes(rgb_root)
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {rgb_root}.")
    scene = rng.choice(scenes)
    variations = _list_variations(rgb_root / scene)
    if not variations:
        raise FileNotFoundError(f"No variations found under {rgb_root / scene}.")
    variation = rng.choice(variations)
    cameras = _list_cameras(rgb_root / scene / variation)
    if not cameras:
        raise FileNotFoundError(f"No camera folders found under {rgb_root / scene / variation}.")
    camera = rng.choice(cameras)
    return VirtualKitty2Selection(scene=scene, variation=variation, camera=camera)


def _resolve_root_from_settings(settings: Mapping[str, object]) -> Path:
    root_raw = settings.get("path") or settings.get("root")
    if not root_raw:
        raise ValueError("Virtual KITTI 2 settings require a 'path' pointing at the dataset root.")
    return Path(str(root_raw)).expanduser()


def _normalize_vkitti_root(root: Path) -> Path:
    root = root.expanduser()
    if root.name in DATASET_SUBDIRS.values():
        return root.parent
    return root


def _subdir_if_exists(root: Path, name: str) -> Optional[Path]:
    path = root / name
    return path if path.exists() else None


def _split_scene_variation(scene_raw: object, variation_raw: object) -> Tuple[Optional[str], Optional[str]]:
    scene = str(scene_raw) if scene_raw is not None else None
    variation = str(variation_raw) if variation_raw is not None else None
    if scene and ("/" in scene or "\\" in scene) and variation is None:
        parts = Path(scene).parts
        if len(parts) >= 2:
            scene = parts[-2]
            variation = parts[-1]
    return scene, variation


def _normalize_camera(camera_raw: object) -> int:
    if camera_raw is None:
        return 0
    if isinstance(camera_raw, str):
        lowered = camera_raw.strip().lower()
        if lowered in {"left", "camera_0", "0"}:
            return 0
        if lowered in {"right", "camera_1", "1"}:
            return 1
        if lowered == "random":
            raise ValueError("Camera cannot be 'random' unless random_selection is enabled.")
    try:
        camera = int(camera_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Virtual KITTI 2 camera must be 0 or 1.") from exc
    if camera not in (0, 1):
        raise ValueError("Virtual KITTI 2 camera must be 0 or 1.")
    return camera


def _normalize_required_resources(required: Iterable[str]) -> List[str]:
    mapping = {
        "depth": "depth",
        "class": "class_segmentation",
        "class_segmentation": "class_segmentation",
        "instance": "instance_segmentation",
        "instance_segmentation": "instance_segmentation",
        "semantics": "class_segmentation",
        "semantics_2d": "class_segmentation",
    }
    out: List[str] = []
    for item in required:
        key = mapping.get(str(item).lower().strip())
        if key and key not in out:
            out.append(key)
    return out


def _validate_selection(rgb_root: Path, selection: VirtualKitty2Selection) -> None:
    scene_dir = rgb_root / selection.scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Virtual KITTI 2 scene '{selection.scene}' not found under {rgb_root}.")
    variation_dir = scene_dir / selection.variation
    if not variation_dir.is_dir():
        raise FileNotFoundError(
            f"Virtual KITTI 2 variation '{selection.variation}' not found under {scene_dir}."
        )
    camera_dir = variation_dir / "frames" / "rgb" / selection.camera_dir
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Virtual KITTI 2 camera folder '{camera_dir}' not found.")


def _collect_frame_paths(rgb_root: Path, scene: str, variation: str, camera_dir: str) -> Dict[int, Path]:
    base = rgb_root / scene / variation / "frames" / "rgb" / camera_dir
    if not base.is_dir():
        return {}
    paths: Dict[int, Path] = {}
    for path in sorted(base.glob("rgb_*.jpg")):
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        paths[index] = path
    return paths


def _frame_path(
    root: Path,
    scene: str,
    variation: str,
    subdir: str,
    camera_dir: str,
    prefix: str,
    frame_index: int,
    suffix: str,
) -> Path:
    filename = f"{prefix}_{int(frame_index):05d}{suffix}"
    return root / scene / variation / "frames" / subdir / camera_dir / filename


def _list_scenes(rgb_root: Path) -> List[str]:
    return [path.name for path in sorted(rgb_root.glob("Scene*")) if path.is_dir()]


def _list_variations(scene_dir: Path) -> List[str]:
    return [path.name for path in sorted(scene_dir.iterdir()) if path.is_dir()]


def _list_cameras(variation_dir: Path) -> List[int]:
    camera_base = variation_dir / "frames" / "rgb"
    cameras: List[int] = []
    if not camera_base.is_dir():
        return cameras
    for path in sorted(camera_base.glob("Camera_*")):
        try:
            cam_id = int(path.name.split("_")[-1])
        except ValueError:
            continue
        cameras.append(cam_id)
    return cameras


def _read_table(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Virtual KITTI 2 metadata file not found: {path}")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < len(header):
            continue
        rows.append({header[i]: parts[i] for i in range(len(header))})
    return rows


def _read_extrinsics(path: Path) -> Iterable[Tuple[int, int, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Virtual KITTI 2 metadata file not found: {path}")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    rows: List[Tuple[int, int, np.ndarray]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 18:
            continue
        frame = int(parts[0])
        camera = int(parts[1])
        values = [float(v) for v in parts[2:18]]
        mat = np.array(values, dtype=np.float32).reshape(4, 4)
        rows.append((frame, camera, mat))
    return rows
