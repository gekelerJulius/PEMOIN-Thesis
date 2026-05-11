from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


CARLA_ROOT = Path("/home/juli/carla/CARLA_0.9.15")
CARLA_SH = CARLA_ROOT / "CarlaUE4.sh"


def _parse_resolution(value: str) -> tuple[int, int]:
    text = str(value).strip().lower()
    if "x" not in text:
        raise argparse.ArgumentTypeError("Resolution must use WIDTHxHEIGHT format.")
    width_text, height_text = text.split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution must use integer WIDTHxHEIGHT.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be > 0.")
    return width, height


def build_command(args: argparse.Namespace) -> list[str]:
    width, height = args.resolution
    command = [
        str(CARLA_SH),
        f"-quality-level={args.quality_level}",
        "-carla-rpc-port={}".format(args.port),
        f"-ResX={width}",
        f"-ResY={height}",
        "-windowed",
    ]
    if args.opengl:
        command.append("-opengl")
    if args.offscreen:
        command.extend(["-RenderOffScreen", "-nosound"])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start CARLA with high-quality defaults.")
    parser.add_argument(
        "--quality-level",
        choices=("Epic", "Low"),
        default="Epic",
        help="CARLA rendering quality preset.",
    )
    parser.add_argument(
        "--res",
        dest="resolution",
        type=_parse_resolution,
        default=(1920, 1080),
        help="Window resolution in WIDTHxHEIGHT format.",
    )
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port.")
    parser.add_argument("--opengl", action="store_true", help="Force OpenGL rendering.")
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Run CARLA without an onscreen window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not CARLA_SH.exists():
        raise FileNotFoundError(f"CARLA launcher not found at '{CARLA_SH}'.")
    command = build_command(args)
    print("Launching CARLA:")
    print(" ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(CARLA_ROOT),
        env=dict(os.environ),
    )
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        return process.wait()


if __name__ == "__main__":
    sys.exit(main())
