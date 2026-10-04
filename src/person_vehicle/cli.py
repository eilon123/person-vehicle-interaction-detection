import argparse
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import yaml

from .candidates import propose
from .events import (ClipOutput, apply_track_consistency_rules, clip_events_to_person_visibility,
                     collapse_physical_episodes, expand_event_boundaries, merge_events,
                     normalize_door_operations)
from .evaluate import evaluate, proposal_recall, passerby_false_positive_rate, temporal_occupancy
from .io import fingerprint, read_json, sha256, videos, write_json
from .video import audit, probe


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["window_s"] <= config["window_overlap_s"] or config["window_overlap_s"] < 0:
        raise ValueError("window_s must exceed nonnegative window_overlap_s")
    if config["sample_frames"] < 2:
        raise ValueError("sample_frames must be >= 2")
    if config["verifier"] not in ("qwen", "review"):
        raise ValueError("verifier must be qwen or review")
    return config


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def run(args):
    from .tracking import track
    from .verify import verify_candidates, QwenVerifier
    from .render import render
    import torch
    config = load_config(args.config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    manifest = {"config_sha256": fingerprint(config), "config": config, "python": platform.python_version(),
                "platform": platform.platform(), "torch_cuda": torch.version.cuda,
                "packages": {name: importlib.metadata.version(name) for name in
                             ["numpy", "av", "opencv-python", "ultralytics", "torch", "transformers"]}, "clips": []}
    failed = False
    verifier_instance = []
    def verifier_factory(settings):
        if not verifier_instance:
            verifier_instance.append(QwenVerifier(settings))
        return verifier_instance[0]
    for path in videos(args.input):
        print(f"Processing {path.name}", flush=True)
        start = time.perf_counter()
        metadata = None
        try:
            metadata = probe(path)
            write_json(output / "audit" / f"{path.stem}.json", metadata)
            rows = track(path, output / "tracks" / f"{path.stem}.jsonl", metadata, config, not args.no_resume)
            candidates = propose(rows, metadata["duration_s"], config)
            write_json(output / "candidates" / f"{path.stem}.json", candidates)
            events, reviews = verify_candidates(path, rows, candidates, metadata, config, output, verifier_factory,
                                                resume=not args.no_resume)
            assembled = merge_events(events, rows if config.get("exclusive_person_vehicle", True) else None)
            assembled = expand_event_boundaries(assembled, rows, config, metadata["duration_s"])
            assembled = merge_events(assembled, rows if config.get("exclusive_person_vehicle", True) else None)
            assembled = apply_track_consistency_rules(assembled, rows, config)
            assembled = collapse_physical_episodes(assembled, rows, config)
            assembled = clip_events_to_person_visibility(assembled, rows)
            assembled = normalize_door_operations(assembled)
            clip = ClipOutput(clip_id=path.stem, source_sha256=metadata["source_sha256"], status="ok",
                              duration_s=metadata["duration_s"], frame_count=metadata["frame_count"],
                              interactions=assembled).model_dump()
            write_json(output / "clips" / f"{path.stem}.json", clip)
            write_json(output / "review" / f"{path.stem}.json", reviews)
            record = {"clip_id": path.stem, "status": "ok", "events": len(clip["interactions"]),
                      "uncertain_candidates": sum(r["decision"] == "uncertain" for r in reviews),
                      "json": f"clips/{path.stem}.json", "inference_seconds": time.perf_counter() - start}
            if args.annotate:
                render_start = time.perf_counter()
                destination = output / "annotated" / f"{path.stem}_annotated.mp4"
                record["render_qa"] = render(path, rows, clip, destination)
                record["video"] = str(destination.relative_to(output))
                record["render_seconds"] = time.perf_counter() - render_start
            record["json_sha256"] = sha256(output / record["json"])
        except Exception as error:
            import traceback
            traceback.print_exc()
            failed = True
            clip = ClipOutput(clip_id=path.stem, source_sha256=sha256(path), status="error",
                duration_s=metadata["duration_s"] if metadata else 0,
                frame_count=metadata["frame_count"] if metadata else 0,
                interactions=[], error=f"{type(error).__name__}: {error}").model_dump()
            write_json(output / "errors" / f"{path.stem}.json", clip)
            write_json(output / "clips" / f"{path.stem}.json", clip)
            record = {"clip_id": path.stem, "status": "error", "error": clip["error"]}
        manifest["clips"].append(record)
        write_json(output / "manifest.json", manifest)
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Detect person-vehicle events and render annotated videos")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="Decode inputs and generate metadata/contact sheets")
    audit_parser.add_argument("--input", required=True)
    audit_parser.add_argument("--output", default="outputs/audit")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", default="outputs")
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument("--annotate", action="store_true")
    run_parser.add_argument("--no-resume", action="store_true")
    tracking_parser = sub.add_parser("track", help="Cache tracks and candidate windows without loading the VLM")
    tracking_parser.add_argument("--input", required=True)
    tracking_parser.add_argument("--output", default="outputs")
    tracking_parser.add_argument("--config", default="configs/default.yaml")
    baseline_parser = sub.add_parser("baseline", help="Geometry-only ablation from cached tracks")
    baseline_parser.add_argument("--input", required=True)
    baseline_parser.add_argument("--tracks", default="outputs/tracks")
    baseline_parser.add_argument("--output", default="outputs/baseline/clips")
    baseline_parser.add_argument("--config", default="configs/default.yaml")
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--input", required=True)
    render_parser.add_argument("--events", default="outputs/clips")
    render_parser.add_argument("--tracks", default="outputs/tracks")
    render_parser.add_argument("--output", default="outputs/annotated")
    render_parser.add_argument("--reference", help="Manual events.json to show compact GT comparison labels")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--pred", required=True)
    eval_parser.add_argument("--reference", required=True)
    eval_parser.add_argument("--mapping", help="JSON mapping clip IDs to predicted-to-reference entity ID maps")
    eval_parser.add_argument("--split")
    eval_parser.add_argument("--candidates", help="Candidate JSON directory for proposal recall")
    eval_parser.add_argument("--negatives", help="Annotated negative-encounter JSON for passerby FP rate")
    eval_parser.add_argument("--binary-timeline", action="store_true",
                             help="Score temporal interaction presence without matching person/vehicle IDs")
    eval_parser.add_argument("--subset", default="test", choices=["test", "development", "all"])
    eval_parser.add_argument("--output", default="outputs/metrics.json")
    annotate_parser = sub.add_parser("annotate", help="Open a local GUI to create confirmed and uncertain time labels")
    annotate_parser.add_argument("--input", required=True, help="MP4 file or directory of MP4 files")
    annotate_parser.add_argument("--output", default="annotations/manual", help="Directory for events.json and uncertain.json")
    annotate_parser.add_argument("--pred", help="Pipeline output directory or clips directory to display active predictions")
    annotate_parser.add_argument("--tracks", help="Track JSONL directory; defaults to <pred>/tracks")
    sub.add_parser("schema")
    assets = sub.add_parser("download-assets")
    assets.add_argument("--detector-only", action="store_true")
    assets.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if args.command == "audit":
        summaries = [audit(path, args.output) for path in videos(args.input)]
        write_json(Path(args.output) / "inventory.json", summaries)
        print(json.dumps(summaries, indent=2))
    elif args.command == "run":
        run(args)
    elif args.command == "track":
        from .tracking import track
        import torch
        config = load_config(args.config)
        torch.manual_seed(config["seed"])
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        for path in videos(args.input):
            metadata = probe(path)
            write_json(Path(args.output) / "audit" / f"{path.stem}.json", metadata)
            rows = track(path, Path(args.output) / "tracks" / f"{path.stem}.jsonl", metadata, config)
            write_json(Path(args.output) / "candidates" / f"{path.stem}.json", propose(rows, metadata["duration_s"], config))
    elif args.command == "baseline":
        from .baseline import geometry_baseline
        config = load_config(args.config)
        for path in videos(args.input):
            metadata = probe(path)
            rows = load_rows(Path(args.tracks) / f"{path.stem}.jsonl")
            result = ClipOutput(clip_id=path.stem, source_sha256=metadata["source_sha256"], status="ok",
                duration_s=metadata["duration_s"], frame_count=metadata["frame_count"],
                interactions=geometry_baseline(rows, metadata, config))
            write_json(Path(args.output) / f"{path.stem}.json", result.model_dump())
    elif args.command == "render":
        from .render import render
        reference = read_json(args.reference) if args.reference else {}
        for path in videos(args.input):
            render(path, load_rows(Path(args.tracks) / f"{path.stem}.jsonl"),
                   read_json(Path(args.events) / f"{path.stem}.json"),
                   Path(args.output) / f"{path.stem}_annotated.mp4", reference.get(path.stem))
    elif args.command == "evaluate":
        pred_dir = Path(args.pred)
        if (pred_dir / "clips").exists():
            pred_dir = pred_dir / "clips"
        predictions = {p.stem: ClipOutput.model_validate(read_json(p)).model_dump() for p in pred_dir.glob("*.json")}
        references = read_json(args.reference)
        if args.split and args.subset != "all":
            selected = read_json(args.split)[args.subset]
            references = {key: references[key] for key in selected}
        if not references:
            raise ValueError("Reference annotations are empty; cannot claim accuracy metrics")
        mappings = read_json(args.mapping) if args.mapping else {}
        results = [evaluate(predictions, references, mappings, threshold, typed, args.binary_timeline)
                   for threshold in [0.3, 0.5, 0.7] for typed in [False, True]]
        report = {"event_metrics": results,
                  "evaluation_mode": "binary_timeline" if args.binary_timeline else "pair_correct_event",
                  "continuous_temporal_overlap": temporal_occupancy(predictions, references)}
        if args.candidates:
            candidates = {p.stem: read_json(p) for p in Path(args.candidates).glob("*.json")}
            report["proposal_recall"] = proposal_recall(candidates, references, mappings)
        if args.negatives:
            negatives = {k: v for k, v in read_json(args.negatives).items() if k in references}
            report["passerby_false_positive_rate"] = passerby_false_positive_rate(predictions, negatives, mappings)
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "annotate":
        from .annotate import launch
        launch(args.input, args.output, args.pred, args.tracks)
    elif args.command == "schema":
        write_json("schemas/clip_output.schema.json", ClipOutput.model_json_schema())
    elif args.command == "download-assets":
        import urllib.request
        config = load_config(args.config)
        target = Path(config["detector"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urllib.request.urlretrieve(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{target.name}", target)
        assets = {"detector": str(target), "detector_sha256": sha256(target)}
        if config.get("head_detection_enabled", False):
            from huggingface_hub import hf_hub_download
            head_target = Path(config["head_detector"])
            if not head_target.exists():
                downloaded = hf_hub_download(
                    repo_id="abhiWanKenobi/yolov8n_head_detection",
                    filename="yolov8n_head_detector.pt", local_dir=str(head_target.parent))
                if Path(downloaded) != head_target:
                    Path(downloaded).replace(head_target)
            assets["head_detector"] = str(head_target)
            assets["head_detector_sha256"] = sha256(head_target)
        if not args.detector_only:
            from huggingface_hub import snapshot_download
            location = snapshot_download(config["vlm_model"], revision=config["vlm_revision"],
                cache_dir=".cache/huggingface", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            assets["vlm_snapshot"] = location
            assets["vlm_revision"] = Path(location).name
            # Pin subsequent runs to the actually downloaded revision.
            config["vlm_revision"] = Path(location).name
            Path(args.config).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json("models/assets.json", assets)


if __name__ == "__main__":
    main()
