#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports resolve when running from repo root without installation.
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pemoin.visualization.video import VideoExportSettings, generate_flat_video_from_dir

# Edit these two values.
frames_dir = "outputs/carla_gt_20260220_181441_20260209_180917_Town01_fps10_1280x720_fov90_sem1_inst1_n50/standard/visualizations/depth"
fps = 5


def main() -> None:
    frame_dir = Path(frames_dir).expanduser().resolve()
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frame_dir}")

    output_path = frame_dir.parent / f"{frame_dir.name}.mp4"
    settings = VideoExportSettings(
        fps=float(fps), codec="mp4v", enabled=True, min_frames=1
    )
    generated = generate_flat_video_from_dir(
        frame_dir,
        frame_dir.parent,
        settings,
        name=frame_dir.name,
    )
    if generated is None:
        raise RuntimeError(
            "Video was not generated. Ensure frames are PNG files named like 000001.png."
        )

    print(f"Created video: {output_path}")


if __name__ == "__main__":
    main()
