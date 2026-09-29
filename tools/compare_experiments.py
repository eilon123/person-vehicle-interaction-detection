"""Generate a self-contained HTML comparison for two pipeline runs."""

import argparse
import html
from pathlib import Path

from person_vehicle.evaluate import evaluate, temporal_occupancy
from person_vehicle.events import ClipOutput
from person_vehicle.io import read_json


def load_predictions(root):
    root = Path(root)
    clips = root / "clips" if (root / "clips").exists() else root
    return {path.stem: ClipOutput.model_validate(read_json(path)).model_dump()
            for path in clips.glob("*.json")}


def pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def delta(new, old):
    return "N/A" if new is None or old is None else f"{100 * (new - old):+.1f} pp"


def metrics(predictions, reference):
    event = evaluate(predictions, reference, threshold=0.5, typed=False, ignore_participants=True)
    time = temporal_occupancy(predictions, reference)
    return event, time


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    parser = argparse.ArgumentParser(description="Compare two person-vehicle experiments")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-name", default="Experiment 1")
    parser.add_argument("--candidate-name", default="Experiment 2")
    args = parser.parse_args()
    reference = read_json(args.reference)
    baseline = load_predictions(args.baseline)
    candidate = load_predictions(args.candidate)
    old_event, old_time = metrics(baseline, reference)
    new_event, new_time = metrics(candidate, reference)
    summary = [
        ["Event precision", pct(old_event["precision"]), pct(new_event["precision"]), delta(new_event["precision"], old_event["precision"])],
        ["Event recall", pct(old_event["recall"]), pct(new_event["recall"]), delta(new_event["recall"], old_event["recall"])],
        ["Event F1", pct(old_event["f1"]), pct(new_event["f1"]), delta(new_event["f1"], old_event["f1"])],
        ["Time overlap IoU", pct(old_time["temporal_iou"]), pct(new_time["temporal_iou"]), delta(new_time["temporal_iou"], old_time["temporal_iou"])],
        ["Time precision", pct(old_time["temporal_precision"]), pct(new_time["temporal_precision"]), delta(new_time["temporal_precision"], old_time["temporal_precision"])],
        ["Time recall", pct(old_time["temporal_recall"]), pct(new_time["temporal_recall"]), delta(new_time["temporal_recall"], old_time["temporal_recall"])],
        ["TP / FP / FN", f"{old_event['tp']} / {old_event['fp']} / {old_event['fn']}",
         f"{new_event['tp']} / {new_event['fp']} / {new_event['fn']}", ""],
    ]
    clips = []
    for clip_id in sorted(reference):
        old = old_event["per_clip"][clip_id]
        new = new_event["per_clip"][clip_id]
        old_iou = old_time["per_clip"][clip_id]["temporal_iou"]
        new_iou = new_time["per_clip"][clip_id]["temporal_iou"]
        clips.append([clip_id, pct(old["f1"]), pct(new["f1"]), delta(new["f1"], old["f1"]),
                      pct(old_iou), pct(new_iou), delta(new_iou, old_iou)])
    old_config = read_json(Path(args.baseline) / "config.json")
    new_config = read_json(Path(args.candidate) / "config.json")
    changed = [[key, old_config.get(key, "—"), new_config.get(key, "—")]
               for key in sorted(set(old_config) | set(new_config)) if old_config.get(key) != new_config.get(key)]
    comparison_title = f"Experiment comparison — {args.baseline_name} vs {args.candidate_name}"
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(comparison_title)}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f7f8fa;color:#1c2530}}main{{max-width:1200px;margin:auto}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{padding:9px 10px;border-bottom:1px solid #e4e8ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f7}}section{{margin-top:30px}}.note{{color:#52606d}}
</style></head><body><main><h1>{html.escape(comparison_title)}</h1>
<p class="note">Binary event matching at temporal IoU ≥ 0.5; participant IDs ignored. Positive delta favors {html.escape(args.candidate_name)}.</p>
<section><h2>Overall KPI comparison</h2>{table(["Metric", args.baseline_name, args.candidate_name, "Delta"], summary)}</section>
<section><h2>Per-clip comparison</h2>{table(["Clip", "Exp. 1 F1", "Exp. 2 F1", "F1 delta", "Exp. 1 Time IoU", "Exp. 2 Time IoU", "IoU delta"], clips)}</section>
<section><h2>Changed configuration</h2>{table(["Setting", args.baseline_name, args.candidate_name], changed)}</section>
</main></body></html>"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
