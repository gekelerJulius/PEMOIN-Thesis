from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _load_openexr_modules():
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Blender depth EXR decode requires the host Python environment to provide "
            "`OpenEXR` and `Imath`."
        ) from exc
    return OpenEXR, Imath


def _read_depth_exr(path: Path, *, channel_name: str = "Depth.V") -> np.ndarray:
    OpenEXR, Imath = _load_openexr_modules()
    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        channels = header.get("channels", {})
        if channel_name not in channels:
            available = ", ".join(sorted(str(name) for name in channels.keys()))
            raise ValueError(
                f"Depth channel {channel_name!r} was not found in {path}. "
                f"Available channels: {available}"
            )
        data_window = header["dataWindow"]
        width = int(data_window.max.x - data_window.min.x + 1)
        height = int(data_window.max.y - data_window.min.y + 1)
        raw = exr.channel(
            channel_name,
            Imath.PixelType(Imath.PixelType.FLOAT),
        )
    finally:
        close = getattr(exr, "close", None)
        if callable(close):
            close()
    depth = np.frombuffer(raw, dtype=np.float32).copy()
    if depth.size != width * height:
        raise ValueError(
            f"Unexpected pixel count while decoding {path}: "
            f"got {depth.size}, expected {width * height}."
        )
    depth = depth.reshape((height, width))
    invalid = (~np.isfinite(depth)) | (depth <= 0.0) | (depth > 1e10)
    if np.any(invalid):
        depth = depth.copy()
        depth[invalid] = 0.0
    return depth


def materialize_depth_npz_from_exr_sequence(
    *,
    depth_exr_dir: Path,
    depth_output_dir: Path,
    blender_version: str | None = None,
    export_api: str | None = None,
    channel_name: str = "Depth.V",
) -> None:
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    exr_paths = sorted(depth_exr_dir.glob("*.exr"))
    if not exr_paths:
        raise ValueError(f"No EXR depth frames found in {depth_exr_dir}.")
    frame_count = 0
    positive_value_count = 0
    for exr_path in exr_paths:
        match = re.search(r"(\d+)$", exr_path.stem)
        if not match:
            continue
        frame_idx = int(match.group(1))
        depth = _read_depth_exr(exr_path, channel_name=channel_name)
        positive_value_count += int(np.count_nonzero(depth > 0.0))
        np.savez_compressed(
            depth_output_dir / f"{frame_idx:06d}.npz",
            depth=np.asarray(depth, dtype=np.float32),
        )
        frame_count += 1
    if frame_count == 0:
        raise ValueError(f"No indexed EXR depth frames found in {depth_exr_dir}.")
    if positive_value_count <= 0:
        raise ValueError(
            "Decoded Blender depth EXR sequence contains no positive depth samples."
        )
    (depth_output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "mode": "z_pass_exr",
                "blender_version": None if blender_version is None else str(blender_version),
                "export_api": None if export_api is None else str(export_api),
                "channel_name": str(channel_name),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode Blender depth EXRs into PEMOIN depth NPZ frames."
    )
    parser.add_argument("--depth-exr-dir", type=Path, required=True)
    parser.add_argument("--depth-output-dir", type=Path, required=True)
    parser.add_argument("--blender-version", type=str, default=None)
    parser.add_argument("--export-api", type=str, default=None)
    parser.add_argument("--channel-name", type=str, default="Depth.V")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    materialize_depth_npz_from_exr_sequence(
        depth_exr_dir=args.depth_exr_dir.expanduser().resolve(),
        depth_output_dir=args.depth_output_dir.expanduser().resolve(),
        blender_version=args.blender_version,
        export_api=args.export_api,
        channel_name=str(args.channel_name),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
