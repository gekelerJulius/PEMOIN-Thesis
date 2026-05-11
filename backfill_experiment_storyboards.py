#!/usr/bin/env python3
"""Generate 3-frame mini storyboards for existing Experiment_* folders."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SERIF_FONT_CANDIDATE_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
]


def _resolve_serif_font_path() -> str | None:
    for candidate in SERIF_FONT_CANDIDATE_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


SERIF_FONT_PATH = _resolve_serif_font_path()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    font_size: int = 20,
    color: tuple[int, int, int] = (20, 20, 20),
    anchor: str = "la",
) -> np.ndarray:
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    if SERIF_FONT_PATH is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(SERIF_FONT_PATH, size=max(8, int(font_size)))
    draw.text(position, text, fill=color, font=font, anchor=anchor)
    rendered = np.asarray(pil_image, dtype=np.uint8)
    image[...] = rendered
    return image


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def _pick_evenly_spaced_rows(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered_rows = list(rows)
    if not ordered_rows or count <= 0:
        return []
    if len(ordered_rows) <= count:
        return ordered_rows
    targets = np.linspace(0, len(ordered_rows) - 1, num=count)
    chosen_positions: list[int] = []
    used_positions: set[int] = set()
    for target in targets:
        candidates = sorted(range(len(ordered_rows)), key=lambda idx: (abs(idx - target), idx))
        for candidate in candidates:
            if candidate not in used_positions:
                used_positions.add(candidate)
                chosen_positions.append(candidate)
                break
    return [ordered_rows[idx] for idx in sorted(chosen_positions)]


def _choose_storyboard_rows(rows: Sequence[dict[str, Any]], count: int = 5) -> tuple[str, list[dict[str, Any]]]:
    rows_list = list(rows)
    both_visible_near = [
        row
        for row in rows_list
        if int(row.get("unity_visible", 0)) > 0
        and int(row.get("pemoin_visible", 0)) > 0
        and (_safe_float(row.get("camera_distance_m")) is not None)
        and float(_safe_float(row.get("camera_distance_m"))) <= 5.0
    ]
    either_visible_near = [
        row
        for row in rows_list
        if (int(row.get("unity_visible", 0)) > 0 or int(row.get("pemoin_visible", 0)) > 0)
        and (_safe_float(row.get("camera_distance_m")) is not None)
        and float(_safe_float(row.get("camera_distance_m"))) <= 5.0
    ]
    both_visible = [
        row
        for row in rows_list
        if int(row.get("unity_visible", 0)) > 0 and int(row.get("pemoin_visible", 0)) > 0
    ]
    either_visible = [
        row
        for row in rows_list
        if int(row.get("unity_visible", 0)) > 0 or int(row.get("pemoin_visible", 0)) > 0
    ]
    if len(both_visible_near) >= count:
        return "both_visible_near_le_5m", _pick_evenly_spaced_rows(both_visible_near, count)
    if len(either_visible_near) >= count:
        return "either_visible_near_le_5m", _pick_evenly_spaced_rows(either_visible_near, count)
    if both_visible_near:
        return "both_visible_near_le_5m_partial", _pick_evenly_spaced_rows(both_visible_near, count)
    if either_visible_near:
        return "either_visible_near_le_5m_partial", _pick_evenly_spaced_rows(either_visible_near, count)
    if len(both_visible) >= count:
        return "both_visible", _pick_evenly_spaced_rows(both_visible, count)
    if len(either_visible) >= count:
        return "either_visible", _pick_evenly_spaced_rows(either_visible, count)
    if both_visible:
        return "both_visible_partial", _pick_evenly_spaced_rows(both_visible, count)
    if either_visible:
        return "either_visible_partial", _pick_evenly_spaced_rows(either_visible, count)
    return "all_frames_fallback", _pick_evenly_spaced_rows(rows_list, count)


def _fit_rgb_to_tile(image: np.ndarray, width: int, height: int, fill_value: int = 242) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), fill_value, dtype=np.uint8)
    offset_x = (width - resized_w) // 2
    offset_y = (height - resized_h) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas


def _make_mini_storyboard(
    rows: Sequence[dict[str, Any]],
    unity_frames: Sequence[np.ndarray],
    pemoin_frames: Sequence[np.ndarray],
    path: Path,
) -> None:
    storyboard_rows = list(rows)
    if not storyboard_rows:
        return
    row_h = 250
    gutter = 12
    header_h = 40
    panel_pad = 12
    label_gap = 20
    slot_w = 42
    board_w = 920
    image_h = row_h - 48
    image_w = (board_w - slot_w - panel_pad * 3) // 2
    board_h = header_h + len(storyboard_rows) * row_h + max(0, len(storyboard_rows) - 1) * gutter
    board = np.full((board_h, board_w, 3), 246, dtype=np.uint8)
    board = _draw_text(board, "Mini storyboard: visible pedestrian, distance <= 5 m", (18, 8), font_size=22, color=(20, 20, 20))
    for idx, row in enumerate(storyboard_rows):
        frame_idx = int(row["aligned_frame_index"])
        x0 = 0
        y0 = header_h + idx * (row_h + gutter)
        cv2.rectangle(board, (x0, y0), (x0 + board_w - 1, y0 + row_h - 1), (228, 228, 228), 1)
        board = _draw_text(board, f"{idx + 1}", (x0 + 16, y0 + row_h // 2 - 8), font_size=20, color=(25, 25, 25))
        unity_x = x0 + slot_w + panel_pad
        pemoin_x = unity_x + image_w + panel_pad
        image_y = y0 + 16 + label_gap
        board = _draw_text(board, "Unity", (unity_x, y0 + 1), font_size=14, color=(40, 40, 40))
        board = _draw_text(board, "PEMOIN", (pemoin_x, y0 + 1), font_size=14, color=(40, 40, 40))
        board[image_y : image_y + image_h, unity_x : unity_x + image_w] = _fit_rgb_to_tile(unity_frames[frame_idx], image_w, image_h)
        board[image_y : image_y + image_h, pemoin_x : pemoin_x + image_w] = _fit_rgb_to_tile(pemoin_frames[frame_idx], image_w, image_h)
    iio.imwrite(path, board)


def _resolve_video_paths(experiment_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    artifacts = summary.get("artifacts") or {}
    videos = artifacts.get("videos") or {}
    unity_video = Path(str(videos.get("unity") or experiment_dir / "videos" / "unity_comparable.mp4")).expanduser()
    pemoin_video = Path(str(videos.get("pemoin") or experiment_dir / "videos" / "pemoin_comparable.mp4")).expanduser()
    if not unity_video.exists():
        unity_video = experiment_dir / "videos" / "unity_comparable.mp4"
    if not pemoin_video.exists():
        pemoin_video = experiment_dir / "videos" / "pemoin_comparable.mp4"
    if not unity_video.exists() or not pemoin_video.exists():
        raise FileNotFoundError(f"Comparable videos missing for {experiment_dir}")
    return unity_video, pemoin_video


def _upsert_report_line(report_path: Path, line: str) -> None:
    if not report_path.exists():
        return
    lines = report_path.read_text(encoding="utf-8").splitlines()
    prefix = "- **Mini storyboard:**"
    for idx, existing in enumerate(lines):
        if existing.startswith(prefix):
            lines[idx] = line
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    try:
        outputs_idx = lines.index("## Outputs")
    except ValueError:
        lines.append("")
        lines.append("## Outputs")
        lines.append(line)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    insert_at = outputs_idx + 1
    while insert_at < len(lines) and (not lines[insert_at].startswith("## ") or lines[insert_at] == "## Outputs"):
        if insert_at > outputs_idx + 1 and lines[insert_at].startswith("## "):
            break
        insert_at += 1
    lines.insert(insert_at, line)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_experiment(experiment_dir: Path) -> None:
    summary_path = experiment_dir / "summary.json"
    per_frame_path = experiment_dir / "per_frame_metrics.csv"
    if not summary_path.exists() or not per_frame_path.exists():
        raise FileNotFoundError(f"Missing summary.json or per_frame_metrics.csv in {experiment_dir}")
    summary = _load_json(summary_path)
    rows = _load_csv(per_frame_path)
    if not rows:
        raise RuntimeError(f"No per-frame rows in {per_frame_path}")
    unity_video, pemoin_video = _resolve_video_paths(experiment_dir, summary)
    unity_frames = _read_video_frames(unity_video)
    pemoin_frames = _read_video_frames(pemoin_video)
    visibility_basis, storyboard_rows = _choose_storyboard_rows(rows, count=5)
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    storyboard_path = plots_dir / "mini_storyboard.png"
    _make_mini_storyboard(storyboard_rows, unity_frames, pemoin_frames, storyboard_path)

    plot_manifest = dict(summary.get("plot_manifest") or {})
    plot_manifest["mini_storyboard"] = str(storyboard_path)
    summary["plot_manifest"] = plot_manifest
    summary["storyboard_manifest"] = {
        "selection_basis": visibility_basis,
        "frames": [
            {
                "slot": idx + 1,
                "aligned_frame_index": int(row["aligned_frame_index"]),
                "time_s": float(row["time_s"]),
                "unity_visible": int(row.get("unity_visible", 0)),
                "pemoin_visible": int(row.get("pemoin_visible", 0)),
                "mask_iou": _safe_float(row.get("mask_iou")),
            }
            for idx, row in enumerate(storyboard_rows)
        ],
    }
    _write_json(summary_path, summary)
    _upsert_report_line(report_path=experiment_dir / "report.md", line=f"- **Mini storyboard:** {storyboard_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python backfill_experiment_storyboards.py <parent_dir_with_experiments>")
    parent_dir = Path(args[0]).expanduser().resolve()
    experiment_dirs = sorted(path for path in parent_dir.glob("Experiment_*") if path.is_dir())
    if not experiment_dirs:
        raise SystemExit(f"No Experiment_* folders found in {parent_dir}")
    processed = 0
    for experiment_dir in experiment_dirs:
        _process_experiment(experiment_dir)
        processed += 1
        print(f"Updated storyboard: {experiment_dir}")
    print(f"Processed experiments: {processed}")


if __name__ == "__main__":
    main()
