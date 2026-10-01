"""Build one self-contained report comparing every archived experiment."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from person_vehicle.events import ClipOutput
from person_vehicle.evaluate import evaluate, temporal_occupancy
from person_vehicle.io import read_json
try:
    from tools.kpi_dashboard import LABELS, dashboard, measures, occupancy_for_type, summarize
except ModuleNotFoundError:  # Direct execution places tools/ rather than the project root on sys.path.
    from kpi_dashboard import LABELS, dashboard, measures, occupancy_for_type, summarize

PROJECT = Path(__file__).resolve().parents[1]


def pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def metric_at_05(report):
    return next(row for row in report["event_metrics"]
                if row["tiou_threshold"] == 0.5 and not row["type_aware"])


def display_name(path, config):
    return str(config.get("experiment_name", path.name)).replace("_", " ").title()


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build(root: Path, reference_path: Path | None = None):
    references = read_json(reference_path) if reference_path is not None else None
    experiments = []
    def experiment_order(path):
        suffix = path.name.removeprefix("experiment_")
        number, separator, variant = suffix.partition("_")
        if number.isdigit():
            # Keep derived variants beside their numbered base experiment,
            # before moving to the next experiment number.
            return (int(number), 0 if not separator else 1, variant)
        return (10 ** 9, 1, path.name)

    for path in sorted((p for p in root.iterdir() if p.is_dir()), key=experiment_order):
        config_path, metrics_path = path / "config.json", path / "manual_binary_metrics.json"
        if not config_path.exists() or (references is not None and not (path / "clips").exists()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        clips_path = path / "clips"
        predictions = {item.stem: ClipOutput.model_validate(read_json(item)).model_dump()
                       for item in clips_path.glob("*.json")} if clips_path.exists() else {}
        if references is None:
            if not metrics_path.exists():
                continue
            report = read_json(metrics_path)
        else:
            missing = set(references) - set(predictions)
            if missing:
                raise ValueError(f"{path.name} has no predictions for canonical GT clips: {', '.join(sorted(missing))}")
            metric_rows = [evaluate(predictions, references, threshold=threshold, typed=typed,
                                    ignore_participants=True)
                           for threshold in (0.15, 0.3, 0.5, 0.7) for typed in (False, True)]
            report = {"event_metrics": metric_rows,
                      "continuous_temporal_overlap": temporal_occupancy(predictions, references),
                      "reference_source": str(reference_path)}
            metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (path / "kpi_dashboard.html").write_text(
                dashboard(predictions, references, 0.5, True, config,
                          str(config.get("experiment_name", path.name))), encoding="utf-8")
        event, timing = metric_at_05(report), report["continuous_temporal_overlap"]
        typed_event = next((row for row in report["event_metrics"]
                            if row["tiou_threshold"] == 0.5 and row["type_aware"]), event)
        relaxed_event = next((row for row in report["event_metrics"]
                              if row["tiou_threshold"] == 0.15 and not row["type_aware"]), event)
        relaxed_typed_event = next((row for row in report["event_metrics"]
                                    if row["tiou_threshold"] == 0.15 and row["type_aware"]), typed_event)
        experiments.append((path, config, event, typed_event, relaxed_event, relaxed_typed_event, timing))
    if not experiments:
        raise ValueError(f"No complete experiments found under {root}")

    overview = []
    details = []
    typed_overview = []
    relaxed_overview = []
    relaxed_typed_overview = []
    previous_f1 = None
    for path, config, event, typed_event, relaxed_event, relaxed_typed_event, timing in experiments:
        predictions = {item.stem: ClipOutput.model_validate(read_json(item)).model_dump()
                       for item in (path / "clips").glob("*.json")}
        name = display_name(path, config)
        delta = "—" if previous_f1 is None else f"{100 * (event['f1'] - previous_f1):+.1f} pp"
        previous_f1 = event["f1"]
        overview.append([name, f"{event['tp']} / {event['fp']} / {event['fn']}", pct(event["precision"]),
                         pct(event["recall"]), pct(event["f1"]), delta, f"{timing.get('intersection_s', 0):.2f}",
                         pct(timing.get("temporal_recall")), pct(timing["temporal_iou"])])
        typed_overview.append([name, f"{typed_event['tp']} / {typed_event['fp']} / {typed_event['fn']}",
                              pct(typed_event["precision"]), pct(typed_event["recall"]), pct(typed_event["f1"])])
        relaxed_overview.append([name, f"{relaxed_event['tp']} / {relaxed_event['fp']} / {relaxed_event['fn']}",
                                pct(relaxed_event["precision"]), pct(relaxed_event["recall"]),
                                pct(relaxed_event["f1"])])
        relaxed_typed_overview.append([
            name, f"{relaxed_typed_event['tp']} / {relaxed_typed_event['fp']} / {relaxed_typed_event['fn']}",
            pct(relaxed_typed_event["precision"]), pct(relaxed_typed_event["recall"]),
            pct(relaxed_typed_event["f1"])])
        config_rows = [[key, value] for key, value in config.items()]
        clip_rows = [[clip, values["tp"], values["fp"], values["fn"], pct(values["f1"]),
                      pct(timing.get("per_clip", {}).get(clip, {}).get("temporal_iou"))]
                     for clip, values in sorted(event["per_clip"].items())]
        type_table = "<p>Detailed type metrics unavailable for this archive.</p>"
        if predictions and references is not None:
            _, _, per_type, _ = summarize(predictions, references, 0.5, True)
            type_rows = []
            for label in LABELS:
                values = measures(per_type[label])
                time_values = occupancy_for_type(predictions, references, label)
                type_rows.append([label, values["tp"], values["fp"], values["fn"],
                                  pct(values["precision"]), pct(values["recall"]), pct(values["f1"]),
                                  pct(time_values["temporal_iou"])])
            type_table = table(["Interaction type", "TP", "FP", "FN", "Precision", "Recall", "F1", "Time IoU"], type_rows)
        dashboard_file = "kpi_dashboard.html"
        live_review = "live_review.html"
        links = []
        if (path / dashboard_file).exists():
            links.append(f'<a href="{html.escape(path.name + "/" + dashboard_file)}">Full KPI report</a>')
        if (path / live_review).exists():
            links.append(f'<a href="{html.escape(path.name + "/" + live_review)}">Live ALG vs GT review</a>')
        change = config.get("experiment_change", "No change description recorded")
        details.append(f"<section><h2>{html.escape(name)}</h2><p>{html.escape(str(change))}</p>"
                       f"<p>{' · '.join(links)}</p><h3>Results by clip</h3>"
                       f"{table(['Scene','TP','FP','FN','F1','Time IoU'], clip_rows)}"
                       f"<h3>Complete KPI by interaction type</h3>{type_table}"
                       f"<h3>Full configuration</h3>{table(['Setting','Value'], config_rows)}</section>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>All person–vehicle experiments — comparison report</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f7f8fa;color:#1c2530}}main{{max-width:1250px;margin:auto}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{padding:9px 10px;border-bottom:1px solid #e4e8ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f7}}section{{margin-top:32px}}a{{margin-right:18px}}
</style></head><body><main><h1>All experiments — person–vehicle comparison</h1>
<h2>Interaction detection only — action type ignored</h2>
<p>An event is a TP when the VLM detects an interaction that temporally overlaps a GT interaction at IoU ≥ 0.5, even if it predicts the wrong action type. Participant IDs are also ignored. This is the requested class-agnostic detection metric.</p>
{table(['Experiment','TP / FP / FN','Precision','Recall','F1','F1 change','Overlap s','GT time covered','Class-agnostic Time IoU'], overview)}
<h2>Correct action classification required — temporal IoU ≥ 0.5</h2>
<p>A TP requires both sufficient temporal overlap and the same action label as the GT. Participant IDs remain ignored.</p>
{table(['Experiment','TP / FP / FN','Precision','Recall','F1'], typed_overview)}
<h2>Interaction detection only — relaxed temporal IoU ≥ 0.15</h2>
<p>Action type and participant IDs are ignored. A predicted interaction is a TP when its temporal IoU with a GT interaction is at least 15%.</p>
{table(['Experiment','TP / FP / FN','Precision','Recall','F1'], relaxed_overview)}
<h2>Correct action classification required â€” relaxed temporal IoU â‰¥ 0.15</h2>
<p>A TP requires the correct action label and at least 15% temporal IoU. This exposes correct classifications whose predicted interval is too broad for the strict 50% table.</p>
{table(['Experiment','TP / FP / FN','Precision','Recall','F1'], relaxed_typed_overview)}
{''.join(details)}<p>All reports above use this canonical GT file: {html.escape(str(reference_path)) if reference_path is not None else 'stored experiment metrics'}</p></main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Compare all archived experiments")
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--reference", default=str(PROJECT / "annotations" / "manual" / "events.json"),
                        help="Canonical GT file produced by the annotation tool")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.experiments_root).resolve()
    reference_path = Path(args.reference).resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"Canonical GT not found: {reference_path}")
    output = Path(args.output).resolve() if args.output else root / "all_experiments.html"
    output.write_text(build(root, reference_path), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
