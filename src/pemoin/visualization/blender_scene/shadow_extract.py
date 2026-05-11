from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path:
    try:
        if Path(sys.path[0]).resolve() == _SCRIPT_DIR:
            sys.path.pop(0)
    except Exception:
        pass

try:
    import imageio.v2 as imageio  # type: ignore
except Exception as exc:  # pragma: no cover - depends on host env
    raise RuntimeError(
        "Shadow PNG synthesis requires the host Python environment to provide imageio."
    ) from exc


def _build_frame_index_map(frame_dir: Path) -> dict[int, Path]:
    frame_map: dict[int, Path] = {}
    for pattern in ("*.png", "*.exr"):
        for frame_path in sorted(frame_dir.glob(pattern)):
            match = re.search(r"(\d+)$", frame_path.stem)
            if not match:
                continue
            frame_map[int(match.group(1))] = frame_path
    if not frame_map:
        raise ValueError(f"No PNG or EXR frames found in {frame_dir}.")
    return frame_map


def _load_openexr_modules():
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Shadow EXR synthesis requires the host Python environment to provide "
            "`OpenEXR` and `Imath` when Blender exports EXR shadow frames."
        ) from exc
    return OpenEXR, Imath


def _channel_candidates(channels: dict[str, object], names: tuple[str, ...]) -> str:
    for name in names:
        if name in channels:
            return name
    available = ", ".join(sorted(str(name) for name in channels.keys()))
    raise ValueError(f"Expected one of {names!r}; available EXR channels: {available}")


def _read_exr_rgba(path: Path) -> np.ndarray:
    OpenEXR, Imath = _load_openexr_modules()
    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        channels = header.get("channels", {})
        data_window = header["dataWindow"]
        width = int(data_window.max.x - data_window.min.x + 1)
        height = int(data_window.max.y - data_window.min.y + 1)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        r_name = _channel_candidates(
            channels, ("R", "Image.R", "Combined.R", "Shadow.R")
        )
        g_name = _channel_candidates(
            channels, ("G", "Image.G", "Combined.G", "Shadow.G")
        )
        b_name = _channel_candidates(
            channels, ("B", "Image.B", "Combined.B", "Shadow.B")
        )
        a_name = _channel_candidates(
            channels, ("A", "Image.A", "Combined.A", "Shadow.A")
        )

        def _read_channel(name: str) -> np.ndarray:
            raw = exr.channel(name, pixel_type)
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            if arr.size != width * height:
                raise ValueError(
                    f"Unexpected pixel count while decoding {path} channel {name!r}: "
                    f"got {arr.size}, expected {width * height}."
                )
            return arr.reshape((height, width))

        rgba = np.stack(
            [
                _read_channel(r_name),
                _read_channel(g_name),
                _read_channel(b_name),
                _read_channel(a_name),
            ],
            axis=2,
        )
    finally:
        close = getattr(exr, "close", None)
        if callable(close):
            close()
    rgba = np.nan_to_num(rgba, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.rint(np.clip(rgba, 0.0, 1.0) * 255.0), 0.0, 255.0).astype(np.uint8)


def _read_rgba_frame(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".exr":
        return _read_exr_rgba(path)
    return np.asarray(imageio.imread(path), dtype=np.uint8)


def synthesize_shadow_rgba(
    *,
    shadow_render_rgba: np.ndarray,
    baseline_render_rgba: np.ndarray | None = None,
) -> np.ndarray:
    raw = np.asarray(shadow_render_rgba, dtype=np.uint8)
    if raw.ndim != 3 or raw.shape[2] < 4:
        raise ValueError(f"Expected RGBA shadow render frame, got {raw.shape}.")
    raw_rgb = np.asarray(raw[:, :, :3], dtype=np.float32)
    receiver_mask = np.asarray(raw[:, :, 3], dtype=np.float32) / 255.0
    raw_luma = (
        0.2126 * raw_rgb[:, :, 0]
        + 0.7152 * raw_rgb[:, :, 1]
        + 0.0722 * raw_rgb[:, :, 2]
    )
    valid = receiver_mask > 0.0
    if baseline_render_rgba is not None:
        baseline = np.asarray(baseline_render_rgba, dtype=np.uint8)
        if baseline.shape != raw.shape:
            raise ValueError(
                "Shadow baseline frame shape mismatch: "
                f"shadow={raw.shape} baseline={baseline.shape}."
            )
        baseline_rgb = np.asarray(baseline[:, :, :3], dtype=np.float32)
        baseline_mask = np.asarray(baseline[:, :, 3], dtype=np.float32) / 255.0
        valid = valid & (baseline_mask > 0.0)
        if not np.any(valid):
            return np.zeros(raw.shape[:2] + (4,), dtype=np.uint8)
        baseline_luma = (
            0.2126 * baseline_rgb[:, :, 0]
            + 0.7152 * baseline_rgb[:, :, 1]
            + 0.0722 * baseline_rgb[:, :, 2]
        )
        darkness = np.clip((baseline_luma - raw_luma) / np.maximum(baseline_luma, 1.0), 0.0, 1.0)
    else:
        if not np.any(valid):
            return np.zeros(raw.shape[:2] + (4,), dtype=np.uint8)
        baseline_luma = float(np.percentile(raw_luma[valid], 95.0))
        denom = max(baseline_luma, 1.0)
        darkness = np.clip((baseline_luma - raw_luma) / denom, 0.0, 1.0)
    alpha = np.clip(darkness * receiver_mask, 0.0, 1.0)
    alpha[alpha < (1.0 / 255.0)] = 0.0
    out = np.zeros(raw.shape[:2] + (4,), dtype=np.uint8)
    out[:, :, 3] = np.clip(np.rint(alpha * 255.0), 0.0, 255.0).astype(np.uint8)
    out[0, 0, 0] = 0  # keep RGB black
    return out


def materialize_shadow_png_sequence(
    *,
    shadow_render_dir: Path,
    shadow_output_dir: Path,
    baseline_render_dir: Path | None = None,
    blender_version: str | None = None,
    export_api: str | None = None,
) -> None:
    render_map = _build_frame_index_map(shadow_render_dir)
    baseline_map = None if baseline_render_dir is None else _build_frame_index_map(baseline_render_dir)
    shadow_output_dir.mkdir(parents=True, exist_ok=True)
    for output_path in shadow_output_dir.glob("*.png"):
        output_path.unlink()
    written = 0
    nonzero_alpha = 0
    for frame_idx, render_path in sorted(render_map.items()):
        render_rgba = _read_rgba_frame(render_path)
        baseline_rgba = None
        if baseline_map is not None:
            baseline_path = baseline_map.get(frame_idx)
            if baseline_path is None:
                raise ValueError(
                    f"Missing baseline shadow render for frame {frame_idx} in {baseline_render_dir}."
                )
            baseline_rgba = _read_rgba_frame(baseline_path)
        shadow_rgba = synthesize_shadow_rgba(
            shadow_render_rgba=render_rgba,
            baseline_render_rgba=baseline_rgba,
        )
        nonzero_alpha += int(np.count_nonzero(shadow_rgba[:, :, 3]))
        imageio.imwrite(shadow_output_dir / f"shadow_{frame_idx:04d}.png", shadow_rgba)
        written += 1
    if written == 0:
        raise ValueError(
            f"No indexed shadow frames were synthesized from {shadow_render_dir}."
        )
    (shadow_output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "mode": (
                    "single_pass_receiver_luma"
                    if baseline_render_dir is None
                    else "receiver_difference"
                ),
                "blender_version": None if blender_version is None else str(blender_version),
                "export_api": None if export_api is None else str(export_api),
                "frame_count": int(written),
                "nonzero_alpha_pixels": int(nonzero_alpha),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize PEMOIN shadow PNG frames from a single shadow-catcher render."
    )
    parser.add_argument("--shadow-render-dir", type=Path, required=True)
    parser.add_argument("--shadow-output-dir", type=Path, required=True)
    parser.add_argument("--baseline-render-dir", type=Path, default=None)
    parser.add_argument("--blender-version", type=str, default=None)
    parser.add_argument("--export-api", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    materialize_shadow_png_sequence(
        shadow_render_dir=args.shadow_render_dir.expanduser().resolve(),
        shadow_output_dir=args.shadow_output_dir.expanduser().resolve(),
        baseline_render_dir=(
            None
            if args.baseline_render_dir is None
            else args.baseline_render_dir.expanduser().resolve()
        ),
        blender_version=args.blender_version,
        export_api=args.export_api,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
