#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import carla


def dump_enum(enum_type) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in dir(enum_type):
        if name.startswith("_"):
            continue
        try:
            v = getattr(enum_type, name)
            iv = int(v)  # carla enums can usually be cast to int
        except Exception:
            continue
        if 0 <= iv <= 255:
            out[name] = iv
    # sort by value for readability
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def main() -> None:
    out_dir = Path(".")
    city_labels = dump_enum(carla.CityObjectLabel)

    payload = {
        "carla_version_hint": getattr(carla, "__file__", None),
        "CityObjectLabel_name_to_id": city_labels,
        "CityObjectLabel_id_to_name": {str(v): k for k, v in city_labels.items()},
        "notes": [
            "Matches semantic segmentation camera tag IDs (red channel in Raw).",
            "For instance segmentation: red = semantic tag, green/blue = instance id.",
        ],
    }

    out_path = out_dir / "carla_label_map_dump.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
