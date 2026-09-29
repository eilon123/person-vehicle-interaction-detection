"""Run, evaluate, render and archive one numbered experiment."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def next_number(root):
    numbers = [int(match.group(1)) for path in root.glob("experiment_*")
               if (match := re.fullmatch(r"experiment_(\d+)", path.name))]
    return max(numbers, default=0) + 1


def execute(*parts):
    print("\n>", subprocess.list2cmdline([str(part) for part in parts]), flush=True)
    subprocess.run([str(part) for part in parts], cwd=PROJECT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run a complete numbered experiment and refresh the history report")
    parser.add_argument("--input", required=True, help="Video file or directory")
    parser.add_argument("--config", required=True, help="YAML configuration for this experiment")
    parser.add_argument("--reference", required=True, help="Manual GT events.json")
    parser.add_argument("--experiments-root", default="experiment_results")
    parser.add_argument("--number", type=int, help="Explicit experiment number; defaults to the next available number")
    parser.add_argument("--description", help="Short description of what changed in this experiment")
    parser.add_argument("--no-resume", action="store_true")
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
    ground_truth = output / "ground_truth" / "events.json"
    ground_truth.parent.mkdir()
    shutil.copy2(args.reference, ground_truth)

    run_command = [sys.executable, "-m", "person_vehicle", "run", "--input", args.input,
                   "--output", output, "--config", generated_config]
    if args.no_resume:
        run_command.append("--no-resume")
    execute(*run_command)
    execute(sys.executable, "-m", "person_vehicle", "evaluate", "--pred", output,
            "--reference", ground_truth, "--subset", "all", "--binary-timeline",
            "--output", output / "manual_binary_metrics.json")
    execute(sys.executable, "tools/kpi_dashboard.py", "--pred", output, "--reference", ground_truth,
            "--binary-timeline", "--output", output / "kpi_dashboard.html")
    execute(sys.executable, "-m", "person_vehicle", "render", "--input", args.input,
            "--events", output / "clips", "--tracks", output / "tracks", "--reference", ground_truth,
            "--output", output / "annotated_vs_gt")
    execute(sys.executable, "tools/experiment_history.py", "--experiments-root", root)
    print(f"\nExperiment {number} complete: {output}")
    print(f"KPI report: {output / 'kpi_dashboard.html'}")
    print(f"Annotated videos vs GT: {output / 'annotated_vs_gt'}")
    print(f"All-experiment comparison: {root / 'all_experiments.html'}")


if __name__ == "__main__":
    main()
