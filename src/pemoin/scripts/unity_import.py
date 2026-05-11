#!/usr/bin/env python3
"""
Import Unity Perception exports into a PEMOIN destination folder.

Features:
- Frame subsampling (every Nth frame)
- Resize RGB/segmentation/depth consistently
- Prune unused frames in destination
- Generate RGB/instance/depth (turbo) videos
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pemoin.visualization.video import write_video
from pemoin.utils.exr import load_exr_image, select_depth_channel


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("opencv-python is required for unity_import.py") from exc


TOP_LEVEL_FILES = (
    "annotation_definitions.json",
    "metadata.json",
    "metric_definitions.json",
    "sensor_definitions.json",
)


@dataclass(frozen=True)
class FramePaths:
    step: int
    frame_json: Path
    rgb: Optional[Path]
    depth: Optional[Path]
    instance: Optional[Path]


@dataclass(frozen=True)
class ImportResult:
    dest_dir: Path
    output_fps: float


@dataclass(frozen=True)
class ImportSelection:
    frames: bool = True
    depth: bool = True
    semantics: bool = True


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def parse_import_selection(selection: Optional[Mapping[str, object]]) -> ImportSelection:
    if selection is None:
        return ImportSelection()
    if not isinstance(selection, dict):
        raise ValueError("unity_import.resources must be an object with boolean flags.")
    def _flag(key: str, default: bool = True) -> bool:
        value = selection.get(key, default)
        return bool(value)
    return ImportSelection(
        frames=_flag("frames", True),
        depth=_flag("depth", True),
        semantics=_flag("semantics", True),
    )


def _resolve_sequences(root: Path) -> List[Path]:
    if root.name.startswith("sequence.") and root.is_dir():
        return [root]
    sequences = sorted(root.glob("sequence.*"))
    if not sequences:
        raise FileNotFoundError(f"No sequence.* directory under {root}")
    return sequences


def _collect_frames(seq_dir: Path) -> List[FramePaths]:
    frames = []
    for json_path in sorted(seq_dir.glob("step*.frame_data.json")):
        payload = _load_json(json_path)
        step = int(payload.get("step", -1))
        if step < 0:
            continue
        capture = _pick_camera_capture(payload.get("captures", []))
        if capture is None:
            continue
        rgb_name = capture.get("filename")
        depth_ann = _find_annotation(capture.get("annotations", []), "Depth")
        inst_ann = _find_annotation(capture.get("annotations", []), "instance segmentation")
        rgb_path = seq_dir / str(rgb_name) if rgb_name else None
        frames.append(
            FramePaths(
                step=step,
                frame_json=json_path,
                rgb=rgb_path,
                depth=seq_dir / str(depth_ann.get("filename")) if depth_ann is not None else None,
                instance=seq_dir / str(inst_ann.get("filename")) if inst_ann is not None else None,
            )
        )
    return sorted(frames, key=lambda f: f.step)


def _pick_camera_capture(captures: Iterable[dict]) -> Optional[dict]:
    for cap in captures:
        if cap.get("id") == "camera":
            return cap
    return None


def _find_annotation(annotations: Iterable[dict], key: str) -> Optional[dict]:
    for ann in annotations:
        if ann.get("id") == key:
            return ann
    return None


def _parse_resize_max_side(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    value = value.strip().lower()
    if "x" in value:
        raise ValueError("resize now expects a single max-side value (e.g. 640)")
    return int(value)


def _compute_resize_from_max_side(width: int, height: int, max_side: Optional[int]) -> Tuple[int, int]:
    if max_side is None:
        return width, height
    if width <= 0 or height <= 0:
        return width, height
    if max(width, height) <= max_side:
        return width, height
    scale = float(max_side) / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height


def _load_exr_depth(path: Path) -> np.ndarray:
    depth = select_depth_channel(load_exr_image(path))
    if depth.ndim != 2:
        raise ValueError(f"Depth EXR must be 2D, got {depth.shape}")
    return depth.astype(np.float32)


def _resize_depth(depth: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_image(image: np.ndarray, size: Tuple[int, int], *, interp: int) -> np.ndarray:
    width, height = size
    return cv2.resize(image, (width, height), interpolation=interp)


def _depth_turbo_frames(depths: Sequence[np.ndarray], *, vmin: float, vmax: float) -> List[np.ndarray]:
    frames = []
    denom = max(1e-6, float(vmax - vmin))
    for depth in depths:
        norm = np.clip((depth - vmin) / denom, 0.0, 1.0)
        img = (norm * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        frames.append(color)
    return frames


def _write_video(frames: Sequence[np.ndarray], output: Path, fps: int) -> None:
    if not frames:
        return
    # Unity import frames are loaded in OpenCV BGR; canonical writer expects RGB.
    rgb_frames: list[np.ndarray] = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb_frames.append(cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB))
        else:
            rgb_frames.append(arr)
    write_video(rgb_frames, output, float(fps), "mp4v")


def _infer_fps(root: Path, default: float = 24.0) -> float:
    ts_fps = _infer_fps_from_timestamps(root)
    meta_fps = _infer_fps_from_metadata(root)
    sensor_fps = _infer_fps_from_sensor_definitions(root)
    # Prefer timestamps (most directly tied to frames), then metadata, then sensor defaults.
    if ts_fps is not None:
        return max(1.0, float(ts_fps))
    if meta_fps is not None:
        return max(1.0, float(meta_fps))
    if sensor_fps is not None:
        return max(1.0, float(sensor_fps))
    return float(default)


def _resolve_stride(
    source_fps: float,
    *,
    stride: Optional[int],
    sampling_fps: Optional[float],
) -> int:
    if sampling_fps is not None:
        if stride is not None:
            raise ValueError("Provide either sampling_fps or stride, not both.")
        try:
            target_fps = float(sampling_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError("sampling_fps must be a positive float.") from exc
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("sampling_fps must be a positive float.")
        if source_fps <= 0:
            return 1
        return max(1, int(round(source_fps / target_fps)))
    if stride is None:
        return 1
    return max(1, int(stride))


def _infer_fps_from_timestamps(root: Path) -> Optional[float]:
    sequences = _resolve_sequences(root)
    if not sequences:
        return None
    seq_dir = sequences[0]
    timestamps = []
    for json_path in sorted(seq_dir.glob("step*.frame_data.json"))[:200]:
        payload = _load_json(json_path)
        ts = payload.get("timestamp")
        if ts is None:
            continue
        try:
            timestamps.append(float(ts))
        except (TypeError, ValueError):
            continue
    if len(timestamps) < 2:
        return None
    timestamps = sorted(set(timestamps))
    deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b > a]
    if not deltas:
        return None
    median_delta = float(np.median(deltas))
    if median_delta <= 1e-6:
        return None
    return 1.0 / median_delta


def _infer_fps_from_metadata(root: Path) -> Optional[float]:
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        return None
    payload = _load_json(meta_path)
    total_frames = payload.get("totalFrames")
    start = payload.get("simulationStartTime")
    end = payload.get("simulationEndTime")
    if total_frames is None or start is None or end is None:
        return None
    try:
        total_frames = int(total_frames)
        start_dt = datetime.strptime(start, "%m/%d/%Y %I:%M:%S %p")
        end_dt = datetime.strptime(end, "%m/%d/%Y %I:%M:%S %p")
    except (ValueError, TypeError):
        return None
    duration = (end_dt - start_dt).total_seconds()
    if duration <= 0 or total_frames <= 1:
        return None
    return (total_frames - 1) / duration


def _infer_fps_from_sensor_definitions(root: Path) -> Optional[float]:
    sensor_path = root / "sensor_definitions.json"
    if not sensor_path.exists():
        return None
    payload = _load_json(sensor_path)
    sensors = payload.get("sensorDefinitions", [])
    if not sensors:
        return None
    sensor = sensors[0]
    delta = sensor.get("simulationDeltaTime")
    frames_between = sensor.get("framesBetweenCaptures", 0)
    if delta is None:
        return None
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        return None
    if delta <= 0:
        return None
    try:
        frames_between = int(frames_between)
    except (TypeError, ValueError):
        frames_between = 0
    interval = delta * (frames_between + 1)
    if interval <= 0:
        return None
    return 1.0 / interval


def import_unity_dataset(
    source: Path,
    dest_root: Path,
    *,
    name: Optional[str],
    stride: Optional[int],
    sampling_fps: Optional[float] = None,
    resize_max_side: Optional[int],
    prune: bool,
    write_videos: bool,
    selection: Optional[ImportSelection | dict] = None,
) -> ImportResult:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    dest_root = dest_root.expanduser().resolve()
    dataset_name = name or source.name
    dest_dir = dest_root / dataset_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    if selection is None:
        selection = ImportSelection()
    elif not isinstance(selection, ImportSelection):
        selection = parse_import_selection(selection)
    source_fps = float(_infer_fps(source, default=24.0))
    stride = _resolve_stride(source_fps, stride=stride, sampling_fps=sampling_fps)
    output_fps = max(1.0, source_fps / float(stride))
    if not selection.frames:
        write_videos = False

    for fname in TOP_LEVEL_FILES:
        src = source / fname
        if src.exists():
            shutil.copy2(src, dest_dir / fname)
    lighting_src = source / "lighting_gt"
    if not lighting_src.is_dir() and source.parent.is_dir():
        sibling_lighting = source.parent / "lighting_gt"
        if sibling_lighting.is_dir():
            lighting_src = sibling_lighting
    lighting_dest = dest_dir / "lighting_gt"
    if lighting_src.is_dir():
        if prune and lighting_dest.exists():
            shutil.rmtree(lighting_dest)
        shutil.copytree(lighting_src, lighting_dest, dirs_exist_ok=True)

    sequences = _resolve_sequences(source)
    for seq_dir in sequences:
        rel = seq_dir.name
        dest_seq = dest_dir / rel
        dest_seq.mkdir(parents=True, exist_ok=True)
        frames = _collect_frames(seq_dir)
        kept = [frame for frame in frames if frame.step % stride == 0]
        if prune:
            for path in dest_seq.glob("step*.*"):
                path.unlink(missing_ok=True)

        depths_for_video = []
        rgb_frames = []
        seg_frames = []
        for new_step, frame in enumerate(kept):
            payload = _load_json(frame.frame_json)
            payload["step"] = int(new_step)
            capture = _pick_camera_capture(payload.get("captures", []))
            if capture is None:
                continue
            dims = capture.get("dimension") or [0, 0]
            width = int(round(float(dims[0])))
            height = int(round(float(dims[1])))
            if resize_max_side is not None:
                width, height = _compute_resize_from_max_side(width, height, resize_max_side)
                capture["dimension"] = [float(width), float(height)]
                for ann in capture.get("annotations", []):
                    if "dimension" in ann:
                        ann["dimension"] = [float(width), float(height)]
            capture["filename"] = f"step{new_step}.camera.png" if selection.frames else None
            annotations = []
            for ann in capture.get("annotations", []):
                ann_id = ann.get("id")
                if ann_id == "Depth":
                    if selection.depth:
                        ann["filename"] = f"step{new_step}.camera.Depth.exr"
                        annotations.append(ann)
                elif ann_id == "instance segmentation":
                    if selection.semantics:
                        ann["filename"] = f"step{new_step}.camera.instance segmentation.png"
                        annotations.append(ann)
                else:
                    annotations.append(ann)
            capture["annotations"] = annotations

            if selection.frames:
                if frame.rgb is None:
                    raise FileNotFoundError(f"RGB frame missing for step {frame.step}.")
                rgb = cv2.imread(str(frame.rgb), cv2.IMREAD_UNCHANGED)
                if rgb is None:
                    raise FileNotFoundError(f"RGB frame missing: {frame.rgb}")
                if resize_max_side is not None:
                    rgb = _resize_image(rgb, (width, height), interp=cv2.INTER_AREA)
                cv2.imwrite(str(dest_seq / f"step{new_step}.camera.png"), rgb)
                rgb_frames.append(rgb)

            if selection.semantics:
                if frame.instance is None:
                    raise FileNotFoundError(f"Instance segmentation missing for step {frame.step}.")
                seg = cv2.imread(str(frame.instance), cv2.IMREAD_UNCHANGED)
                if seg is None:
                    raise FileNotFoundError(f"Instance segmentation missing: {frame.instance}")
                if resize_max_side is not None:
                    seg = _resize_image(seg, (width, height), interp=cv2.INTER_NEAREST)
                cv2.imwrite(str(dest_seq / f"step{new_step}.camera.instance segmentation.png"), seg)
                seg_frames.append(seg)

            if selection.depth:
                if frame.depth is None:
                    raise FileNotFoundError(f"Depth EXR missing for step {frame.step}.")
                depth = _load_exr_depth(frame.depth)
                if resize_max_side is not None:
                    depth = _resize_depth(depth, (width, height))
                cv2.imwrite(str(dest_seq / f"step{new_step}.camera.Depth.exr"), depth)
                depths_for_video.append(depth)

            _write_json(dest_seq / f"step{new_step}.frame_data.json", payload)

        if write_videos and kept and selection.frames:
            print(
                f"[unity_import] video fps: source={source_fps:.3f}, stride={stride}, output={output_fps:.3f}"
            )
            _write_video(rgb_frames, dest_seq / "rgb.mp4", output_fps)
            if selection.semantics:
                _write_video(seg_frames, dest_seq / "instance_segmentation.mp4", output_fps)
            if selection.depth and depths_for_video:
                vals = np.concatenate([d[np.isfinite(d) & (d > 1e-6)] for d in depths_for_video])
                if vals.size:
                    vmin = float(np.quantile(vals, 0.01))
                    vmax = float(np.quantile(vals, 0.99))
                else:
                    vmin, vmax = 0.0, 1.0
                depth_frames = _depth_turbo_frames(depths_for_video, vmin=vmin, vmax=vmax)
                _write_video(depth_frames, dest_seq / "depth_turbo.mp4", output_fps)

    return ImportResult(dest_dir=dest_dir, output_fps=output_fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Unity Perception data into a PEMOIN destination folder."
    )
    parser.add_argument("source", type=Path, help="Unity export root (e.g. .../solo_7)")
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("outputs/unity_import"),
        help="Destination root for imported Unity exports.",
    )
    parser.add_argument("--name", type=str, default=None, help="Destination dataset name.")
    parser.add_argument("--stride", type=int, default=None, help="Keep every Nth frame.")
    parser.add_argument(
        "--sampling-fps",
        type=float,
        default=None,
        help="Target output FPS (overrides stride when provided).",
    )
    parser.add_argument("--resize", type=str, default=None, help="Resize so max side <= N (e.g. 640).")
    parser.add_argument("--prune", action="store_true", help="Delete unused frames in destination.")
    parser.add_argument("--no-videos", action="store_true", help="Skip mp4 generation.")
    args = parser.parse_args()

    resize_max_side = _parse_resize_max_side(args.resize)
    result = import_unity_dataset(
        args.source,
        args.dest_root,
        name=args.name,
        stride=int(args.stride) if args.stride is not None else None,
        sampling_fps=args.sampling_fps,
        resize_max_side=resize_max_side,
        prune=bool(args.prune),
        write_videos=not bool(args.no_videos),
    )
    print(f"[unity_import] imported {result.dest_dir} (output_fps={result.output_fps:.3f})")


if __name__ == "__main__":
    main()
