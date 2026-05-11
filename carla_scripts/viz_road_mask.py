#!/usr/bin/env python3
from __future__ import annotations
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
from PIL import Image

import json
from pathlib import Path
from typing import Dict

# Same dump file you used in the exporter
CARLA_LABEL_DUMP_PATH = Path("/home/juli/PycharmProjects/PEMOIN/carla_scripts/carla_label_map_dump.json")


def _load_semseg_label_map(run_dir: Path) -> Dict[str, int]:
    # 1) Prefer per-run label map (written by exporter)
    p = run_dir / "semseg_label_map.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("name_to_id", {})
        if isinstance(m, dict) and all(isinstance(k, str) and isinstance(v, int) for k, v in m.items()):
            return m

    # 2) Fallback: load mapping from your CARLA label dump file
    with open(CARLA_LABEL_DUMP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = data.get("CityObjectLabel_name_to_id", {})
    if not isinstance(m, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in m.items()):
        raise ValueError(
            f"Invalid label dump format in {CARLA_LABEL_DUMP_PATH} (missing CityObjectLabel_name_to_id)."
        )

    # Optional normalization: allow singular keys commonly used in scripts
    aliases = {
        "Unlabeled": "NONE",
        "Road": "Roads",
        "Sidewalk": "Sidewalks",
        "Building": "Buildings",
        "Wall": "Walls",
        "Fence": "Fences",
        "Pole": "Poles",
        "TrafficSign": "TrafficSigns",
        "Pedestrian": "Pedestrians",
        "RoadLine": "RoadLines",
    }
    for alias, target in aliases.items():
        if target in m and alias not in m:
            m[alias] = m[target]

    return m



def _infer_frame_stem(path: Path) -> str:
    return path.stem  # e.g. "000123"


def _load_gray_png(p: Path) -> np.ndarray:
    # semseg_id was saved as L mode; keep uint8
    return np.array(Image.open(p).convert("L"), dtype=np.uint8)


def _maybe_load_rgb(run_dir: Path, stem: str) -> Optional[np.ndarray]:
    for ext in (".jpg", ".png"):
        p = run_dir / "rgb" / f"{stem}{ext}"
        if p.exists():
            return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
    return None


def main() -> None:
    args = argparse.Namespace(
        run_dir=Path("/home/juli/PycharmProjects/PEMOIN/carla_exports/20260129_013308_Town01_fps60_720x480_fov90_sem1_inst1_n600"),
        frame=None,
        label="Road",
        out=None,
        overlay=False,
        alpha=0.5,
    )

    run_dir: Path = args.run_dir
    sem_dir = run_dir / "semseg_id"
    if not sem_dir.exists():
        raise SystemExit(f"Missing {sem_dir} (run_dir must contain semseg_id/).")

    sem_files = sorted(sem_dir.glob("*.png"))
    if not sem_files:
        raise SystemExit(f"No PNGs found in {sem_dir}.")

    if args.frame is None:
        sem_path = sem_files[0]
    else:
        sem_path = sem_dir / f"{args.frame}.png"
        if not sem_path.exists():
            raise SystemExit(f"Frame not found: {sem_path}")

    stem = _infer_frame_stem(sem_path)
    label_map = _load_semseg_label_map(run_dir)

    if args.label not in label_map:
        available = ", ".join(sorted(label_map.keys()))
        raise SystemExit(f'Label "{args.label}" not in map. Available: {available}')

    label_id = int(label_map[args.label])

    sem = _load_gray_png(sem_path)
    mask = (sem == label_id).astype(np.uint8) * 255  # 0/255

    if args.out is None:
        suffix = "overlay" if args.overlay else "mask"
        out_path = run_dir / f"{args.label.lower()}_{suffix}_{stem}.png"
    else:
        out_path = args.out

    if args.overlay:
        rgb = _maybe_load_rgb(run_dir, stem)
        if rgb is None:
            raise SystemExit(f"Overlay requested but no rgb/{stem}.jpg|png found.")

        alpha = float(np.clip(args.alpha, 0.0, 1.0))
        out = rgb.copy()
        # red overlay where mask == 255
        red = np.zeros_like(out)
        red[..., 0] = 255
        m = mask.astype(bool)
        out[m] = (out[m] * (1.0 - alpha) + red[m] * alpha).astype(np.uint8)

        Image.fromarray(out).save(out_path)
    else:
        Image.fromarray(mask, mode="L").save(out_path)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
