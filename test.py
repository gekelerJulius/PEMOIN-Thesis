from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

# =========================
# EDIT THESE VARIABLES
# =========================
DATAROOT = "/home/juli/Datasets/nuScenes-mini/v1.0-mini"  # adjust to your layout
VERSION = "v1.0-mini"
SEED = 0  # set to None for true random
# =========================


def _collect_scene_cam_front_sweeps(
    nusc: NuScenes, scene: dict
) -> list[dict]:
    first_sample = nusc.get("sample", scene["first_sample_token"])
    last_sample = nusc.get("sample", scene["last_sample_token"])

    current_token = first_sample["data"]["CAM_FRONT"]
    last_token = last_sample["data"]["CAM_FRONT"]
    frames: list[dict] = []

    while current_token:
        sample_data = nusc.get("sample_data", current_token)
        frames.append(sample_data)
        if current_token == last_token:
            break
        current_token = sample_data["next"]

    if not frames:
        raise RuntimeError(
            f"Scene '{scene['name']}' does not contain any CAM_FRONT sample_data frames."
        )
    if frames[-1]["token"] != last_token:
        raise RuntimeError(
            f"Failed to reach the last CAM_FRONT frame for scene '{scene['name']}'."
        )
    return frames


def _infer_video_fps(sample_data_frames: list[dict], default: float = 12.0) -> float:
    if len(sample_data_frames) < 2:
        return float(default)

    timestamps = np.array(
        [int(frame["timestamp"]) for frame in sample_data_frames], dtype=np.float64
    )
    deltas_sec = np.diff(timestamps) / 1_000_000.0
    deltas_sec = deltas_sec[np.isfinite(deltas_sec) & (deltas_sec > 1e-6)]
    if deltas_sec.size == 0:
        return float(default)
    return float(1.0 / np.median(deltas_sec))


def _write_scene_video(
    sample_data_frames: list[dict], *, dataroot: str, output_path: Path, fps: float
) -> None:
    first_frame_path = os.path.join(dataroot, sample_data_frames[0]["filename"])
    first_frame = np.asarray(Image.open(first_frame_path).convert("RGB"), dtype=np.uint8)
    height, width = first_frame.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for '{output_path}'.")

    try:
        for sample_data in sample_data_frames:
            frame_path = os.path.join(dataroot, sample_data["filename"])
            frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _infer_label_dtype(file_path: str, n_points: int) -> np.dtype:
    n_bytes = os.path.getsize(file_path)
    if n_bytes == n_points:
        return np.uint8
    if n_bytes == 2 * n_points:
        return np.uint16
    if n_bytes == 4 * n_points:
        return np.uint32
    # Fallback: try uint16 (common) and let reshape/size checks fail later if wrong.
    return np.uint16


def _find_by_sample_data_token(
    nusc: NuScenes, table_name: str, sample_data_token: str
) -> Optional[dict]:
    # Only exists if you downloaded the corresponding annotations (lidarseg/panoptic).
    table = getattr(nusc, table_name, None)
    if not table:
        return None
    for rec in table:
        if rec.get("sample_data_token") == sample_data_token:
            return rec
    return None


def _project_lidar_to_cam_depth(
    nusc: NuScenes,
    lidar_sd_token: str,
    cam_sd_token: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      uv: (2, M) float pixel coords
      depth: (M,) float depth in camera frame (meters)
    """
    lidar_sd = nusc.get("sample_data", lidar_sd_token)
    cam_sd = nusc.get("sample_data", cam_sd_token)

    lidar_path = os.path.join(nusc.dataroot, lidar_sd["filename"])
    pc = LidarPointCloud.from_file(lidar_path)  # (4, N): x,y,z, intensity

    # --- Lidar sensor -> ego (at lidar timestamp)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(lidar_cs["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_cs["translation"]))

    # --- Ego (lidar timestamp) -> global
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    pc.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_pose["translation"]))

    # --- Global -> ego (camera timestamp)
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    pc.translate(-np.array(cam_pose["translation"]))
    pc.rotate(Quaternion(cam_pose["rotation"]).rotation_matrix.T)

    # --- Ego -> camera sensor
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    pc.translate(-np.array(cam_cs["translation"]))
    pc.rotate(Quaternion(cam_cs["rotation"]).rotation_matrix.T)

    # Points now in camera frame
    pts_cam = pc.points[:3, :]  # (3, N)
    depth = pts_cam[2, :]

    # Keep points in front of camera
    keep = depth > 0.1
    pts_cam = pts_cam[:, keep]
    depth = depth[keep]

    K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)  # (3, 3)
    uv = view_points(pts_cam, K, normalize=True)[:2, :]  # (2, M)

    # Filter to image bounds
    w, h = cam_sd["width"], cam_sd["height"]
    in_img = (uv[0, :] >= 0) & (uv[0, :] < w) & (uv[1, :] >= 0) & (uv[1, :] < h)
    return uv[:, in_img], depth[in_img]


def _process_scene(nusc: NuScenes, scene_index: int) -> None:
    scene = nusc.scene[scene_index]
    scene_token = scene["token"]

    # Pick first sample of scene (or change to random sample if you want)
    first_sample = nusc.get("sample", scene["first_sample_token"])
    cam_front_sweeps = _collect_scene_cam_front_sweeps(nusc, scene)
    scene_video_path = Path(__file__).resolve().with_name(
        f"testvid_{scene_index}.mp4"
    )
    scene_video_fps = _infer_video_fps(cam_front_sweeps)
    _write_scene_video(
        cam_front_sweeps,
        dataroot=nusc.dataroot,
        output_path=scene_video_path,
        fps=scene_video_fps,
    )

    cam_sd_token = first_sample["data"]["CAM_FRONT"]
    lidar_sd_token = first_sample["data"]["LIDAR_TOP"]

    cam_sd = nusc.get("sample_data", cam_sd_token)
    cam_path = os.path.join(nusc.dataroot, cam_sd["filename"])

    # -------------------------
    # FRONT CAM RGB
    # -------------------------
    img = Image.open(cam_path)
    img_np = np.array(img)
    print("\n=== SCENE ===")
    print(f"scene.index       : {scene_index}")
    print(f"scene.name        : {scene['name']}")
    print(f"scene.token       : {scene_token}")
    print(f"scene.description : {scene.get('description', '')}")
    print(f"sample.token      : {first_sample['token']}")
    print("\n=== SCENE VIDEO ===")
    print(f"path              : {scene_video_path}")
    print(f"frames            : {len(cam_front_sweeps)}")
    print(f"fps               : {scene_video_fps:.3f}")
    print("\n=== CAM_FRONT RGB ===")
    print(f"path              : {cam_path}")
    print(f"size (W,H)        : {img.size}")
    print(f"array shape       : {img_np.shape}  dtype={img_np.dtype}")

    # -------------------------
    # CAMERA INTRINSICS
    # -------------------------
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
    print("\n=== CAM_FRONT INTRINSICS ===")
    print(f"K:\n{K}")
    print(f"fx, fy            : {K[0,0]:.3f}, {K[1,1]:.3f}")
    print(f"cx, cy            : {K[0,2]:.3f}, {K[1,2]:.3f}")

    # -------------------------
    # DEPTH (sparse) via LiDAR->camera projection
    # -------------------------
    uv, depth = _project_lidar_to_cam_depth(nusc, lidar_sd_token, cam_sd_token)
    print("\n=== DEPTH (SPARSE, from LiDAR projection) ===")
    print(f"projected points  : {depth.size}")
    if depth.size:
        print(
            f"depth min/mean/max: {depth.min():.3f} / {depth.mean():.3f} / {depth.max():.3f} meters"
        )
        print(f"uv range x        : [{uv[0].min():.1f}, {uv[0].max():.1f}]")
        print(f"uv range y        : [{uv[1].min():.1f}, {uv[1].max():.1f}]")

    # -------------------------
    # PANOPTIC LABELS (if available)
    # Note: nuScenes "panoptic" is for LiDAR points (LIDAR_TOP), not camera pixels.
    # Requires nuScenes-panoptic download + corresponding metadata.
    # -------------------------
    panoptic_rec = _find_by_sample_data_token(nusc, "panoptic", lidar_sd_token)
    print("\n=== PANOPTIC LABELS (LiDAR points) ===")
    if panoptic_rec is None:
        print(
            "panoptic          : NOT AVAILABLE in this dataset install (likely not downloaded)."
        )
        print(
            "note              : nuScenes panoptic labels are for LiDAR points, not CAM_FRONT pixels."
        )
    else:
        pan_path = os.path.join(nusc.dataroot, panoptic_rec["filename"])
        # Need point count to infer dtype
        lidar_path = os.path.join(
            nusc.dataroot, nusc.get("sample_data", lidar_sd_token)["filename"]
        )
        n_points = LidarPointCloud.from_file(lidar_path).points.shape[1]
        dtype = _infer_label_dtype(pan_path, n_points)
        labels = np.fromfile(pan_path, dtype=dtype)
        print(f"path              : {pan_path}")
        print(f"dtype             : {labels.dtype}")
        print(f"num labels        : {labels.size} (lidar points: {n_points})")
        if labels.size:
            uniq = np.unique(labels)
            print(f"unique ids        : {uniq.size}")
            print(f"min/max id        : {int(uniq.min())} / {int(uniq.max())}")

    # -------------------------
    # (Optional) LIDARSEG (semantic) labels (if you meant semantic instead of panoptic)
    # -------------------------
    lidarseg_rec = _find_by_sample_data_token(nusc, "lidarseg", lidar_sd_token)
    print("\n=== LIDARSEG LABELS (semantic, LiDAR points) ===")
    if lidarseg_rec is None:
        print(
            "lidarseg          : NOT AVAILABLE in this dataset install (likely not downloaded)."
        )
    else:
        ls_path = os.path.join(nusc.dataroot, lidarseg_rec["filename"])
        lidar_path = os.path.join(
            nusc.dataroot, nusc.get("sample_data", lidar_sd_token)["filename"]
        )
        n_points = LidarPointCloud.from_file(lidar_path).points.shape[1]
        dtype = _infer_label_dtype(ls_path, n_points)
        labels = np.fromfile(ls_path, dtype=dtype)
        print(f"path              : {ls_path}")
        print(f"dtype             : {labels.dtype}")
        print(f"num labels        : {labels.size} (lidar points: {n_points})")
        if labels.size:
            uniq = np.unique(labels)
            print(f"unique classes    : {uniq.size}")
            print(f"min/max class id  : {int(uniq.min())} / {int(uniq.max())}")


def main() -> None:
    if SEED is not None:
        np.random.seed(SEED)

    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    total_scenes = len(nusc.scene)
    print(f"processing {total_scenes} scene(s)")
    for scene_index in range(total_scenes):
        _process_scene(nusc, scene_index)


if __name__ == "__main__":
    main()
