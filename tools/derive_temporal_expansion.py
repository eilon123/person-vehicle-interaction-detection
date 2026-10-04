"""Derive a temporal-expansion experiment from one completed inference run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from person_vehicle.events import ClipOutput, expand_event_boundaries, merge_events
from person_vehicle.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--near-margin", type=float, required=True)
    parser.add_argument("--max-gap-s", type=float, required=True)
    parser.add_argument("--max-extension-s", type=float, required=True)
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(source / "config.json")
    config.update({
        "experiment_name": args.name,
        "experiment_change": ("Experiment 16 correlation/motion detections with temporal hysteresis: "
                              f"near margin {args.near_margin}, gap {args.max_gap_s}s, "
                              f"maximum extension {args.max_extension_s}s"),
        "temporal_expansion_enabled": True,
        "temporal_expansion_near_margin": args.near_margin,
        "temporal_expansion_max_gap_s": args.max_gap_s,
        "temporal_expansion_max_extension_s": args.max_extension_s,
    })
    write_json(output / "config.json", config)
    for clip_path in sorted((source / "clips").glob("*.json")):
        clip = ClipOutput.model_validate(read_json(clip_path)).model_dump()
        track_path = source / "tracks" / f"{clip_path.stem}.jsonl"
        rows = [json.loads(line) for line in track_path.read_text(encoding="utf-8").splitlines()]
        events = expand_event_boundaries(copy.deepcopy(clip["interactions"]), rows, config, clip["duration_s"])
        clip["interactions"] = merge_events(events, rows)
        write_json(output / "clips" / clip_path.name, ClipOutput.model_validate(clip).model_dump())
    print(output.resolve())


if __name__ == "__main__":
    main()
