#!/usr/bin/env python3
"""Replay prior Experiment_* folders by re-running compare_unity_pemoin_runs.py with stored source paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_experiment_paths(raw_paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        path = raw_path.expanduser().resolve()
        if path.is_dir() and path.name.startswith("Experiment_"):
            if path not in seen:
                resolved.append(path)
                seen.add(path)
            continue
        if path.is_dir():
            for child in sorted(path.glob("Experiment_*")):
                child = child.resolve()
                if child.is_dir() and child not in seen:
                    resolved.append(child)
                    seen.add(child)
    return resolved


def _extract_rerun_inputs(experiment_dir: Path) -> tuple[Path, Path]:
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")
    summary = _load_json(summary_path)
    inputs = summary.get("inputs") or {}
    unity_run = inputs.get("unity_run")
    pemoin_run = inputs.get("pemoin_run")
    if not unity_run:
        raise ValueError("missing inputs.unity_run")
    if not pemoin_run:
        raise ValueError("missing inputs.pemoin_run")
    unity_path = Path(str(unity_run)).expanduser()
    pemoin_path = Path(str(pemoin_run)).expanduser()
    if not unity_path.exists():
        raise FileNotFoundError(f"unity source path does not exist: {unity_path}")
    if not pemoin_path.exists():
        raise FileNotFoundError(f"pemoin source path does not exist: {pemoin_path}")
    return unity_path, pemoin_path


def _parse_created_experiment(stdout: str, stderr: str) -> str | None:
    for stream in (stdout, stderr):
        for line in reversed(stream.splitlines()):
            marker = "Created experiment output:"
            if marker in line:
                return line.split(marker, 1)[1].strip()
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Experiment_* directories or parent directories containing them.")
    parser.add_argument("--compare-script", type=Path, default=ROOT / "compare_unity_pemoin_runs.py")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--unity-sequence", default=None)
    parser.add_argument(
        "--pemoin-video-source",
        default=None,
        choices=("auto", "harmonized_overlays", "overlayed_frames", "output_mp4"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    experiment_dirs = _expand_experiment_paths(args.paths)
    if not experiment_dirs:
        raise SystemExit("No Experiment_* folders found in the provided paths.")

    compare_script = args.compare_script.expanduser().resolve()
    if not compare_script.exists():
        raise FileNotFoundError(f"Compare script not found: {compare_script}")

    validated: list[tuple[Path, Path, Path]] = []
    failures: list[str] = []
    for experiment_dir in experiment_dirs:
        try:
            unity_run, pemoin_run = _extract_rerun_inputs(experiment_dir)
            validated.append((experiment_dir, unity_run, pemoin_run))
        except Exception as exc:  # noqa: BLE001
            message = f"{experiment_dir}: {exc}"
            failures.append(message)
            if args.strict:
                raise SystemExit(message) from exc

    successes: list[tuple[Path, str | None]] = []
    if not args.dry_run:
        for experiment_dir, unity_run, pemoin_run in validated:
            command = [
                sys.executable,
                str(compare_script),
                "--unity-run",
                str(unity_run),
                "--pemoin-run",
                str(pemoin_run),
            ]
            if args.output_root is not None:
                command.extend(["--output-root", str(args.output_root.expanduser())])
            if args.unity_sequence is not None:
                command.extend(["--unity-sequence", args.unity_sequence])
            if args.pemoin_video_source is not None:
                command.extend(["--pemoin-video-source", args.pemoin_video_source])
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                failures.append(
                    f"{experiment_dir}: compare script failed with exit code {result.returncode}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
                if args.strict:
                    raise SystemExit(failures[-1])
                continue
            successes.append((experiment_dir, _parse_created_experiment(result.stdout, result.stderr)))

    print(f"Requested experiments: {len(experiment_dirs)}")
    print(f"Validated experiments: {len(validated)}")
    print(f"Successful reruns: {len(successes)}")
    print(f"Failures/skips: {len(failures)}")
    if args.dry_run:
        print("Dry run only. Planned reruns:")
        for experiment_dir, unity_run, pemoin_run in validated:
            print(f"- {experiment_dir} -> unity={unity_run} pemoin={pemoin_run}")
    else:
        print("Created outputs:")
        for experiment_dir, created_output in successes:
            print(f"- {experiment_dir} -> {created_output or 'created output path not detected'}")
    if failures:
        print("Failures/skips:")
        for failure in failures:
            print(f"- {failure}")
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
