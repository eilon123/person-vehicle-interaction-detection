import argparse
import json
from pathlib import Path

from person_vehicle.candidates import propose
from person_vehicle.events import (ClipOutput, apply_track_consistency_rules,
    clip_events_to_person_visibility, collapse_physical_episodes,
    expand_event_boundaries, merge_events, normalize_door_operations)
from person_vehicle.io import write_json
from person_vehicle.verify import QwenVerifier, verify_candidates
from person_vehicle.video import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    clip_id = args.video.stem
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (args.root / "tracks" / f"{clip_id}.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = probe(args.video)
    candidates = propose(rows, metadata["duration_s"], config)
    write_json(args.root / "config.json", config)
    write_json(args.root / "audit" / f"{clip_id}.json", metadata)
    write_json(args.root / "candidates" / f"{clip_id}.json", candidates)
    verifier = []
    factory = lambda settings: verifier[0] if verifier else (verifier.append(QwenVerifier(settings)) or verifier[0])
    events, reviews = verify_candidates(args.video, rows, candidates, metadata, config,
                                        args.root, factory, resume=True)
    assembled = merge_events(events, rows if config.get("exclusive_person_vehicle", True) else None)
    assembled = expand_event_boundaries(assembled, rows, config, metadata["duration_s"])
    assembled = merge_events(assembled, rows if config.get("exclusive_person_vehicle", True) else None)
    assembled = apply_track_consistency_rules(assembled, rows, config)
    assembled = collapse_physical_episodes(assembled, rows, config)
    assembled = clip_events_to_person_visibility(assembled, rows)
    assembled = normalize_door_operations(assembled)
    clip = ClipOutput(clip_id=clip_id, source_sha256=metadata["source_sha256"], status="ok",
                      duration_s=metadata["duration_s"], frame_count=metadata["frame_count"],
                      interactions=assembled).model_dump()
    write_json(args.root / "clips" / f"{clip_id}.json", clip)
    write_json(args.root / "review" / f"{clip_id}.json", reviews)
    print(json.dumps(clip, indent=2))


if __name__ == "__main__":
    main()
