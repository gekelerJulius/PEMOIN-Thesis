from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _load_diagnostics(run_dir: Path) -> dict:
    path = (
        run_dir
        / "standard"
        / "visualizations"
        / "blender_scene"
        / "render_backend_diagnostics.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing render backend diagnostics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_summary(run_dirs: list[Path]) -> dict:
    records = []
    for run_dir in run_dirs:
        payload = _load_diagnostics(run_dir)
        timings = payload.get("timings_seconds", {})
        total = float(sum(float(v) for v in timings.values()))
        records.append(
            {
                "run_dir": str(run_dir),
                "engine": payload.get("engine"),
                "resolution_scale": payload.get("render", {}).get("resolution_scale"),
                "samples": payload.get("render", {}).get("samples"),
                "shadow_map_resolution": payload.get("shadow", {}).get("map_resolution"),
                "shadow_softness": payload.get("shadow", {}).get("softness"),
                "timings_seconds": timings,
                "total_render_stage_seconds": total,
            }
        )
    totals = [float(record["total_render_stage_seconds"]) for record in records]
    return {
        "run_count": len(records),
        "mean_total_render_stage_seconds": mean(totals) if totals else 0.0,
        "runs": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize PEMOIN Blender render-backend diagnostics across runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more PEMOIN run directories under outputs/<run>.",
    )
    args = parser.parse_args(argv)
    summary = _build_summary([path.expanduser().resolve() for path in args.run_dirs])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
