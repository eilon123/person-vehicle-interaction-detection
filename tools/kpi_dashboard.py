"""Create a self-contained HTML dashboard for person--vehicle KPI results."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from person_vehicle.evaluate import match_events, temporal_occupancy
from person_vehicle.events import ClipOutput
from person_vehicle.io import read_json


LABELS = ("enter", "exit", "door_operation", "load_unload", "other_interaction")


def rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def summarize(predictions, references, threshold: float, ignore_participants: bool):
    """Return total, per-clip, per-type and type-confusion counts at one tIoU."""
    total = Counter()
    per_clip, per_type = {}, {}
    confusion = defaultdict(Counter)

    for clip_id, truth_clip in references.items():
        predicted = predictions[clip_id]["interactions"]
        truth = truth_clip["interactions"]
        matches = match_events(predicted, truth, threshold=threshold,
                               ignore_participants=ignore_participants)
        matched_prediction = {left for left, _, _ in matches}
        matched_reference = {right for _, right, _ in matches}
        counts = Counter(tp=len(matches), fp=len(predicted) - len(matches), fn=len(truth) - len(matches))
        total.update(counts)
        per_clip[clip_id] = dict(counts)
        for left, right, _ in matches:
            confusion[truth[right]["type"]][predicted[left]["type"]] += 1
        for index, event in enumerate(truth):
            if index not in matched_reference:
                confusion[event["type"]]["missed"] += 1
        for index, event in enumerate(predicted):
            if index not in matched_prediction:
                confusion["false_alarm"][event["type"]] += 1

    for label in LABELS:
        typed_total = Counter()
        for clip_id, truth_clip in references.items():
            predicted = [event for event in predictions[clip_id]["interactions"] if event["type"] == label]
            truth = [event for event in truth_clip["interactions"] if event["type"] == label]
            matches = match_events(predicted, truth, threshold=threshold,
                                   typed=True, ignore_participants=ignore_participants)
            typed_total.update(tp=len(matches), fp=len(predicted) - len(matches), fn=len(truth) - len(matches))
        per_type[label] = dict(typed_total)
    return dict(total), per_clip, per_type, {key: dict(value) for key, value in confusion.items()}


def measures(counts):
    tp, fp, fn = counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0)
    precision = rate(tp, tp + fp)
    recall = rate(tp, tp + fn)
    return {**counts, "precision": precision, "recall": recall,
            "f1": rate(2 * tp, 2 * tp + fp + fn)}


def occupancy_for_type(predictions, references, label):
    """Compute threshold-free temporal KPIs using only one action label."""
    filtered_predictions = {
        clip_id: {**clip, "interactions": [event for event in clip["interactions"] if event["type"] == label]}
        for clip_id, clip in predictions.items()
    }
    filtered_references = {
        clip_id: {**clip, "interactions": [event for event in clip["interactions"] if event["type"] == label]}
        for clip_id, clip in references.items()
    }
    return temporal_occupancy(filtered_predictions, filtered_references)


def percent(value):
    return "N/A" if value is None else f"{value * 100:.1f}%"


def table(headers, rows, class_name=""):
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<table class="{class_name}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def algorithm_rows(config):
    if not config:
        return [["Run configuration", "Not found"]]
    rows = [
        ["Detector", f"{config.get('detector', 'unknown')} · image size {config.get('image_size', 'unknown')} · confidence {config.get('detection_confidence', 'unknown')}"],
        ["Tracker", config.get("tracker", "unknown")],
        ["Temporal verifier", f"{config.get('verifier', 'unknown')} · {config.get('vlm_model', 'unknown')}"],
        ["Verifier revision", config.get("vlm_revision", "unknown")],
        ["Candidate context", f"{config.get('context_s', 'unknown')} s context · {config.get('window_s', 'unknown')} s window · {config.get('window_overlap_s', 'unknown')} s overlap"],
        ["Verifier sampling", f"{config.get('sample_frames', 'unknown')} frames · max pixels {config.get('max_pixels', 'unknown')} · 4-bit {config.get('load_in_4bit', 'unknown')}"],
        ["Random seed", config.get("seed", "unknown")],
    ]
    if config.get("candidate_filter"):
        rows.extend([
            ["Learned candidate filter", config["candidate_filter"]],
            ["Training protocol", config.get("training_protocol", "unknown")],
            ["Training sample count", config.get("training_sample_count", "unknown")],
        ])
    return rows


def algorithm_summary(config, evaluation_mode, threshold):
    """Detailed, config-derived description so each report documents its own run."""
    value = lambda key, fallback="unknown": config.get(key, fallback)
    verifier = value("verifier")
    if verifier == "qwen":
        verification = (f"For each candidate, {value('vlm_model')} (revision {value('vlm_revision')}) examines "
                        f"{value('sample_frames')} chronologically selected frames. It decides whether an interaction "
                        "occurred, its action type, its person–vehicle pair, and supported evidence frames. "
                        f"Inference uses 4-bit loading: {value('load_in_4bit')}.")
    else:
        verification = f"Candidate windows are processed by the configured verifier: {verifier}."
    mode = ("Participant IDs are ignored in the thresholded event metric; this evaluates temporal event presence."
            if evaluation_mode == "binary_timeline" else
            "A thresholded event match requires the predicted person and vehicle to match the reference pair.")
    steps = [
        ("Detection.", f"Every decoded frame is processed by {value('detector')} at image size {value('image_size')} "
                         f"with detection confidence at least {value('detection_confidence')}."),
        ("Tracking.", f"{value('tracker')} assigns stable local person and vehicle IDs within a clip. IDs are reset for each clip."),
        ("Candidate generation.", f"The pipeline proposes person–vehicle windows from spatial proximity (near margin {value('near_margin')}), "
                                  f"track appearance/disappearance (gap {value('candidate_gap_s')} s), and temporal context of {value('context_s')} s."),
        ("Temporal verification.", verification),
        ("Event assembly.", "Overlapping duplicate proposals for the same supported action and pair are merged; disjoint supported spans remain separate. "
                            "The finalized event JSON is the source of truth for both metrics and video overlays."),
        ("Evaluation.", f"Thresholded event precision, recall, and F1 use temporal IoU ≥ {threshold:.1f}. {mode} "
                           "The separate Time overlap IoU has no threshold: it unions all predicted and reference interaction spans, "
                           "so simultaneous actions do not double-count time."),
    ]
    return "".join(f"<li><strong>{html.escape(stage)}</strong> {html.escape(detail)}</li>" for stage, detail in steps)


def dashboard(predictions, references, threshold, ignore_participants, config, experiment_name=None):
    total, per_clip, per_type, confusion = summarize(predictions, references, threshold, ignore_participants)
    overall = measures(total)
    occupancy = temporal_occupancy(predictions, references)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12] if config else "not available"
    clip_rows = []
    for clip_id, counts in sorted(per_clip.items(), key=lambda item: measures(item[1])["f1"] or 0):
        m = measures(counts)
        time = occupancy["per_clip"][clip_id]
        clip_rows.append([clip_id, m["tp"], m["fp"], m["fn"], percent(m["precision"]), percent(m["recall"]), percent(m["f1"]),
                          f"{time['intersection_s']:.2f}", f"{time['predicted_s']:.2f}", f"{time['reference_s']:.2f}",
                          percent(time["temporal_precision"]), percent(time["temporal_recall"]),
                          percent(time["temporal_f1"]), percent(time["temporal_iou"])])
    type_rows = []
    for label in LABELS:
        m = measures(per_type[label])
        time = occupancy_for_type(predictions, references, label)
        type_rows.append([label, m["tp"], m["fp"], m["fn"], percent(m["precision"]), percent(m["recall"]), percent(m["f1"]),
                          percent(time["temporal_precision"]), percent(time["temporal_recall"]),
                          percent(time["temporal_f1"]), percent(time["temporal_iou"])])
    columns = list(LABELS) + ["missed"]
    confusion_rows = [[row] + [confusion.get(row, {}).get(column, 0) for column in columns]
                      for row in list(LABELS) + ["false_alarm"]]
    mode = "Binary temporal event matching (participant IDs ignored)" if ignore_participants else "Pair-correct event matching"
    experiment_name = str(experiment_name or config.get("experiment_name", "Experiment")).replace("_", " ").title()
    report_title = f"{experiment_name} — Person–vehicle KPI report"
    training_note = ("<p class=\"note\"><strong>Training/evaluation note:</strong> This run used a small sample from the same videos being evaluated. "
                     "Its KPI measures dataset adaptation and is not an independent generalization estimate.</p>"
                     if "in_sample" in str(config.get("training_protocol", "")) else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(report_title)}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f7f8fa;color:#1c2530}} main{{max-width:1200px;margin:auto}} h1{{margin-bottom:4px}} .note{{color:#52606d}} .metrics{{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}} .metric{{background:#fff;border:1px solid #d7dde4;border-radius:10px;padding:14px 18px;min-width:120px}} .metric b{{display:block;font-size:26px;margin-top:5px}} section{{margin-top:30px}} table{{border-collapse:collapse;width:100%;background:#fff}} th,td{{padding:9px 10px;border-bottom:1px solid #e4e8ed;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#edf2f7;font-weight:600}} tr:hover td{{background:#f8fbff}} .f1{{font-weight:700}} @media(max-width:680px){{body{{margin:14px}}th,td{{padding:7px 5px;font-size:12px}}}}
</style></head><body><main>
<h1>{html.escape(report_title)}</h1><p class="note">Standalone results for {html.escape(experiment_name)} · {html.escape(mode)} · temporal IoU ≥ {threshold:.1f} · {len(references)} labelled clips</p>
{training_note}
<div class="metrics"><div class="metric">Precision<b>{percent(overall['precision'])}</b></div><div class="metric">Recall<b>{percent(overall['recall'])}</b></div><div class="metric">F1<b>{percent(overall['f1'])}</b></div><div class="metric">TP / FP / FN<b>{overall['tp']} / {overall['fp']} / {overall['fn']}</b></div><div class="metric">Class-agnostic Time IoU<b>{percent(occupancy['temporal_iou'])}</b></div><div class="metric">GT time covered<b>{percent(occupancy['temporal_recall'])}</b></div><div class="metric">Overlap time<b>{occupancy['intersection_s']:.2f}s</b></div></div>
<p class="note"><strong>Class-agnostic temporal overlap</strong> ignores the predicted and GT action labels. All predicted spans and all GT spans are unioned before comparison, so concurrent actions do not double-count time. Overlap: {occupancy['intersection_s']:.2f}s; predicted interaction time: {occupancy['predicted_s']:.2f}s; GT interaction time: {occupancy['reference_s']:.2f}s; union: {occupancy['union_s']:.2f}s.</p>
<section><h2>Algorithm summary</h2><ol>{algorithm_summary(config, 'binary_timeline' if ignore_participants else 'pair_correct_event', threshold)}</ol></section>
<section><h2>Algorithm run</h2><p class="note">Configuration fingerprint: <code>{config_hash}</code>. This section is generated from the config file associated with the selected prediction directory.</p>{table(['Setting', 'Value'], algorithm_rows(config))}</section>
<section><h2>Results by scene</h2><p class="note">Event KPIs use temporal IoU ≥ {threshold:.1f}. Class-agnostic time KPIs ignore action classification and compare only predicted interaction time against GT interaction time.</p>{table(['Scene', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'Overlap s', 'Predicted s', 'GT s', 'Time precision', 'GT time covered', 'Time F1', 'Class-agnostic Time IoU'], clip_rows, 'clip-table')}</section>
<section><h2>Complete KPI by interaction type</h2><p class="note">Time spans are unioned separately within each interaction type before measuring overlap.</p>{table(['Interaction type', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'Time precision', 'Time recall', 'Time F1', 'Time IoU'], type_rows)}</section>
<section><h2>Action-type confusion</h2><p class="note">Rows are manual labels; columns are predictions. “Missed” has no matched prediction; “false alarm” has no matched manual label.</p>{table(['Manual / prediction'] + columns, confusion_rows)}</section>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Create an HTML KPI dashboard")
    parser.add_argument("--pred", required=True, help="Pipeline output directory or clips directory")
    parser.add_argument("--reference", required=True, help="Manual events.json")
    parser.add_argument("--output", required=True, help="Destination HTML file")
    parser.add_argument("--config", help="Run config.json; defaults to <pred>/config.json")
    parser.add_argument("--experiment-name", help="Display name used in the report title")
    parser.add_argument("--tiou", type=float, default=0.5, choices=(0.3, 0.5, 0.7))
    parser.add_argument("--binary-timeline", action="store_true", help="Ignore person and vehicle IDs")
    args = parser.parse_args()
    pred_dir = Path(args.pred)
    run_root = pred_dir
    if (pred_dir / "clips").exists():
        pred_dir = pred_dir / "clips"
    elif pred_dir.name == "clips":
        run_root = pred_dir.parent
    predictions = {path.stem: ClipOutput.model_validate(read_json(path)).model_dump()
                   for path in pred_dir.glob("*.json")}
    references = read_json(args.reference)
    missing = set(references) - set(predictions)
    if missing:
        raise ValueError(f"Missing predictions for: {', '.join(sorted(missing))}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else run_root / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    output.write_text(dashboard(predictions, references, args.tiou, args.binary_timeline, config,
                                args.experiment_name), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
