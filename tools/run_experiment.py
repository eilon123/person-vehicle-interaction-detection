"""Run, evaluate, render and archive one numbered experiment."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml
from person_vehicle.io import read_json, videos, write_json


PROJECT = Path(__file__).resolve().parents[1]


def next_number(root):
    numbers = [int(match.group(1)) for path in root.glob("experiment_*")
               if (match := re.fullmatch(r"experiment_(\d+)", path.name))]
    return max(numbers, default=0) + 1


def execute(*parts):
    print("\n>", subprocess.list2cmdline([str(part) for part in parts]), flush=True)
    subprocess.run([str(part) for part in parts], cwd=PROJECT, check=True)


def apply_final_postprocessing(input_path, output, config_path):
    """Reproduce the appearance split and emergence stages used by Experiment 40."""
    raw = output / "raw_person_detections"
    execute(sys.executable, "tools/cache_person_detections.py", "--input", input_path,
            "--config", config_path, "--output", raw)
    execute(sys.executable, "tools/batch_track_rules.py", "--videos", input_path,
            "--source-tracks", output / "tracks", "--raw", raw,
            "--output", output, "--emergence")
    for video in videos(input_path):
        execute(sys.executable, "tools/finish_scene_trial.py", "--root", output,
                "--video", video, "--config", output / "config.json")
    execute(sys.executable, "tools/apply_emergence_overrides.py", "--root", output)


def export_flat_interactions(output):
    interactions = []
    for clip_path in sorted((output / "clips").glob("*.json")):
        clip = read_json(clip_path)
        interactions.extend({"clip_id": clip.get("clip_id", clip_path.stem), **event}
                            for event in clip.get("interactions", []))
    write_json(output / "interactions.json", interactions)


def main():
    parser = argparse.ArgumentParser(description="Run a complete numbered experiment and refresh the history report")
    parser.add_argument("--input", required=True, help="Video file or directory")
    parser.add_argument("--config", required=True, help="YAML configuration for this experiment")
    parser.add_argument("--reference", default=str(PROJECT / "annotations" / "manual" / "events.json"),
                        help="Canonical manual GT; defaults to annotations/manual/events.json")
    parser.add_argument("--experiments-root", default="experiment_results")
    parser.add_argument("--number", type=int, help="Explicit experiment number; defaults to the next available number")
    parser.add_argument("--description", help="Short description of what changed in this experiment")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--final-postprocess", action="store_true",
                        help="Apply the appearance-split and emergence stages used by Experiment 40")
    args = parser.parse_args()

    root = Path(args.experiments_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    number = args.number or next_number(root)
    output = root / f"experiment_{number}"
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another --number")
    output.mkdir()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["experiment_name"] = f"experiment_{number}"
    if args.description:
        config["experiment_change"] = args.description
    generated_config = output / "experiment_config.yaml"
    generated_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    ground_truth = Path(args.reference).resolve()
    if not ground_truth.exists():
        raise FileNotFoundError(f"Canonical GT not found: {ground_truth}; create it with person_vehicle annotate")

    run_command = [sys.executable, "-m", "person_vehicle", "run", "--input", args.input,
                   "--output", output, "--config", generated_config]
    if args.no_resume:
        run_command.append("--no-resume")
    execute(*run_command)
    if args.final_postprocess:
        apply_final_postprocessing(args.input, output, generated_config)
    export_flat_interactions(output)
    execute(sys.executable, "-m", "person_vehicle", "evaluate", "--pred", output,
            "--reference", ground_truth, "--subset", "all", "--binary-timeline",
            "--output", output / "manual_binary_metrics.json")
    execute(sys.executable, "tools/kpi_dashboard.py", "--pred", output, "--reference", ground_truth,
            "--binary-timeline", "--output", output / "kpi_dashboard.html")
    execute(sys.executable, "tools/export_vlm_descriptions.py", "--clips", output / "clips",
            "--reviews", output / "review", "--output", output / "vlm_descriptions",
            "--experiment", f"experiment_{number}")
    execute(sys.executable, "tools/create_live_review.py", "--input", args.input,
            "--events", output / "clips", "--tracks", output / "tracks", "--reference", ground_truth,
            "--descriptions", output / "vlm_descriptions",
            "--output", output / "live_review.html")
    execute(sys.executable, "tools/experiment_history.py", "--experiments-root", root,
            "--reference", ground_truth)
    print(f"\nExperiment {number} complete: {output}")
    print(f"KPI report: {output / 'kpi_dashboard.html'}")
    print(f"Live ALG vs GT review: {output / 'live_review.html'}")
    print(f"All-experiment comparison: {root / 'all_experiments.html'}")


if __name__ == "__main__":
    main()
