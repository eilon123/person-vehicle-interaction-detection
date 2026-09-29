"""Create an ablation run by adding geometry transitions to a completed base run."""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from person_vehicle.events import ClipOutput, merge_events
from person_vehicle.io import fingerprint, read_json, sha256, write_json
from person_vehicle.transitions import infer_person_vehicle_transitions, fuse_transition_events


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser(description="Apply geometry transitions to a completed experiment")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    base, output = Path(args.base), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not config.get("geometry_transitions"):
        raise ValueError("Config must enable geometry_transitions")
    shutil.copytree(base, output, ignore=shutil.ignore_patterns("comparison", "annotated", "annotated_vs_gt",
                                                                "*.html", "*metrics.json"))
    write_json(output / "config.json", config)
    manifest = read_json(output / "manifest.json")
    manifest["config"] = config
    manifest["config_sha256"] = fingerprint(config)
    manifest["derived_from"] = str(base.resolve())
    manifest["derivation"] = "Base predictions plus deterministic geometry transition events"
    records = {record["clip_id"]: record for record in manifest["clips"]}
    for clip_path in sorted((output / "clips").glob("*.json")):
        clip_id = clip_path.stem
        clip = read_json(clip_path)
        candidates = read_json(output / "candidates" / f"{clip_id}.json")
        metadata = read_json(output / "audit" / f"{clip_id}.json")
        transition_events = infer_person_vehicle_transitions(
            rows(output / "tracks" / f"{clip_id}.jsonl"), candidates, metadata, config)
        clip["interactions"] = merge_events(fuse_transition_events(clip["interactions"], transition_events, config))
        validated = ClipOutput.model_validate(clip).model_dump()
        write_json(clip_path, validated)
        record = records[clip_id]
        record["events"] = len(validated["interactions"])
        record["geometry_transition_events"] = len(transition_events)
        record["json_sha256"] = sha256(clip_path)
    write_json(output / "manifest.json", manifest)
    print(output.resolve())


if __name__ == "__main__":
    main()
