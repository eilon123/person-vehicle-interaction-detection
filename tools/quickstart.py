"""One-command v2 pipeline run with reports, Live Review, and annotated MP4s."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_experiment import next_number


PROJECT = Path(__file__).resolve().parents[1]


def execute(*parts: object) -> None:
    command = [str(part) for part in parts]
    print("\n>", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing the assignment MP4 files")
    parser.add_argument("--output", default="outputs/experiments", help="Numbered experiment root")
    parser.add_argument("--config", default="configs/v2.0.yaml")
    parser.add_argument("--reference", default="annotations/manual/events.json")
    parser.add_argument("--number", type=int, help="Unused experiment number; defaults to the next number")
    parser.add_argument("--skip-download", action="store_true", help="Require model assets to exist already")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    number = args.number or next_number(root)
    experiment = root / f"experiment_{number}"

    if not args.skip_download:
        execute(sys.executable, "-m", "person_vehicle", "download-assets", "--config", args.config)

    command: list[object] = [sys.executable, "tools/run_experiment.py", "--input", args.input,
        "--config", args.config, "--reference", args.reference, "--experiments-root", root,
        "--number", number, "--description", "Portable v2.0 quick-start run"]
    if args.no_resume:
        command.append("--no-resume")
    execute(*command)

    render_command: list[object] = [sys.executable, "-m", "person_vehicle", "render",
        "--input", args.input, "--events", experiment / "clips", "--tracks", experiment / "tracks",
        "--output", experiment / "annotated"]
    reference = Path(args.reference)
    if reference.exists():
        render_command.extend(["--reference", reference])
    execute(*render_command)

    print("\nComplete output:", experiment)
    print("KPI report:", experiment / "kpi_dashboard.html")
    print("Live Review:", experiment / "live_review.html")
    print("Annotated MP4s:", experiment / "annotated")
    print("Structured events:", experiment / "clips")
    print("VLM descriptions:", experiment / "vlm_descriptions")


if __name__ == "__main__":
    main()
