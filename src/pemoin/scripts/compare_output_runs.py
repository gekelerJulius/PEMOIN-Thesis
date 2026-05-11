#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameCompare:
    frame_index: int
    normal_angle_deg: float
    height_at_camera_diff_m: float
    offset_diff_m: float
    residual_p90_diff_m: float | None


def _load_npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "metadata" not in data.files:
            return {}
        raw = data["metadata"]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.shape == ():
            item = raw.item()
            if isinstance(item, dict):
                return dict(item)
            return {}
        if raw.size == 1 and isinstance(raw.reshape(-1)[0], dict):
            return dict(raw.reshape(-1)[0])
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _load_trajectory_centers(run_dir: Path) -> dict[int, np.ndarray]:
    path = run_dir / "standard" / "trajectory" / "poses.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")
    with np.load(path, allow_pickle=True) as data:
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32).reshape(-1)
        c2w = np.asarray(data["camera_to_world"], dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Invalid camera_to_world shape in {path}: {c2w.shape}")
    return {int(f): c2w[i, :3, 3].astype(np.float32) for i, f in enumerate(frame_indices)}


def _load_plane_by_frame(run_dir: Path) -> dict[int, tuple[np.ndarray, float, dict[str, Any]]]:
    road_dir = run_dir / "standard" / "road_plane"
    if not road_dir.exists():
        raise FileNotFoundError(f"Missing road plane directory: {road_dir}")
    result: dict[int, tuple[np.ndarray, float, dict[str, Any]]] = {}
    for path in sorted(road_dir.glob("*.npz")):
        frame = int(path.stem)
        with np.load(path, allow_pickle=True) as data:
            normal = np.asarray(data["normal"], dtype=np.float32).reshape(3)
            offset = float(data["offset"])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal = normal / norm
        metadata = _load_npz_metadata(path)
        result[frame] = (normal, offset / norm, metadata)
    if not result:
        raise RuntimeError(f"No road planes found in {road_dir}")
    return result


def _angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    # Plane orientation is sign-invariant.
    dot = abs(dot)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def compare_runs(
    run_a: Path,
    run_b: Path,
    *,
    max_normal_angle_deg: float,
    max_height_diff_m: float,
    max_residual_p90_diff_m: float,
) -> dict[str, Any]:
    planes_a = _load_plane_by_frame(run_a)
    planes_b = _load_plane_by_frame(run_b)
    centers_a = _load_trajectory_centers(run_a)
    centers_b = _load_trajectory_centers(run_b)

    common = sorted(set(planes_a.keys()) & set(planes_b.keys()))
    if not common:
        raise RuntimeError("No common road-plane frames found between runs.")

    rows: list[FrameCompare] = []
    for frame in common:
        na, da, ma = planes_a[frame]
        nb, db, mb = planes_b[frame]
        ca = centers_a.get(frame)
        cb = centers_b.get(frame)
        if ca is None or cb is None:
            continue
        ha = float(np.dot(na, ca) + da)
        hb = float(np.dot(nb, cb) + db)
        ra = ma.get("residual_p90")
        rb = mb.get("residual_p90")
        residual_diff = None
        if ra is not None and rb is not None:
            residual_diff = abs(float(ra) - float(rb))
        rows.append(
            FrameCompare(
                frame_index=frame,
                normal_angle_deg=_angle_deg(na, nb),
                height_at_camera_diff_m=abs(ha - hb),
                offset_diff_m=abs(float(da - db)),
                residual_p90_diff_m=residual_diff,
            )
        )

    if not rows:
        raise RuntimeError("No comparable frames remained after trajectory intersection.")

    angle = np.asarray([r.normal_angle_deg for r in rows], dtype=np.float32)
    hdiff = np.asarray([r.height_at_camera_diff_m for r in rows], dtype=np.float32)
    odiff = np.asarray([r.offset_diff_m for r in rows], dtype=np.float32)
    rvals = np.asarray([r.residual_p90_diff_m for r in rows if r.residual_p90_diff_m is not None], dtype=np.float32)

    mismatches: list[dict[str, Any]] = []
    for r in rows:
        reasons: list[str] = []
        if r.normal_angle_deg > max_normal_angle_deg:
            reasons.append("normal_angle")
        if r.height_at_camera_diff_m > max_height_diff_m:
            reasons.append("height_at_camera")
        if r.residual_p90_diff_m is not None and r.residual_p90_diff_m > max_residual_p90_diff_m:
            reasons.append("residual_p90")
        if reasons:
            mismatches.append(
                {
                    "frame_index": r.frame_index,
                    "reasons": reasons,
                    "normal_angle_deg": r.normal_angle_deg,
                    "height_at_camera_diff_m": r.height_at_camera_diff_m,
                    "offset_diff_m": r.offset_diff_m,
                    "residual_p90_diff_m": r.residual_p90_diff_m,
                }
            )

    summary = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "frame_count_compared": len(rows),
        "thresholds": {
            "max_normal_angle_deg": max_normal_angle_deg,
            "max_height_diff_m": max_height_diff_m,
            "max_residual_p90_diff_m": max_residual_p90_diff_m,
        },
        "metrics": {
            "normal_angle_deg": {
                "median": float(np.median(angle)),
                "p90": float(np.percentile(angle, 90)),
                "max": float(np.max(angle)),
            },
            "height_at_camera_diff_m": {
                "median": float(np.median(hdiff)),
                "p90": float(np.percentile(hdiff, 90)),
                "max": float(np.max(hdiff)),
            },
            "offset_diff_m": {
                "median": float(np.median(odiff)),
                "p90": float(np.percentile(odiff, 90)),
                "max": float(np.max(odiff)),
            },
            "residual_p90_diff_m": {
                "median": float(np.median(rvals)) if rvals.size else None,
                "p90": float(np.percentile(rvals, 90)) if rvals.size else None,
                "max": float(np.max(rvals)) if rvals.size else None,
            },
        },
        "mismatch_count": len(mismatches),
        "mismatches_top20": mismatches[:20],
        "has_large_mismatch": bool(len(mismatches) > 0),
    }
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two PEMOIN output runs and detect large road-plane mismatches."
    )
    p.add_argument("run_a", type=Path, help="First output run folder")
    p.add_argument("run_b", type=Path, help="Second output run folder")
    p.add_argument("--max-normal-angle-deg", type=float, default=3.0)
    p.add_argument("--max-height-diff-m", type=float, default=0.25)
    p.add_argument("--max-residual-p90-diff-m", type=float, default=0.30)
    p.add_argument("--json-out", type=Path, help="Optional path to write JSON summary")
    p.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with status 2 when large mismatches are detected",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    summary = compare_runs(
        args.run_a,
        args.run_b,
        max_normal_angle_deg=float(args.max_normal_angle_deg),
        max_height_diff_m=float(args.max_height_diff_m),
        max_residual_p90_diff_m=float(args.max_residual_p90_diff_m),
    )
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_mismatch and bool(summary.get("has_large_mismatch", False)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
