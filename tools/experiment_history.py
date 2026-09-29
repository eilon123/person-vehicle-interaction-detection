"""Build one self-contained report comparing every archived experiment."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


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


def build(root: Path):
    experiments = []
    for path in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        config_path, metrics_path = path / "config.json", path / "manual_binary_metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        event, timing = metric_at_05(report), report["continuous_temporal_overlap"]
        experiments.append((path, config, event, timing))
    if not experiments:
        raise ValueError(f"No complete experiments found under {root}")

    overview = []
    details = []
    previous_f1 = None
    for path, config, event, timing in experiments:
        name = display_name(path, config)
        delta = "—" if previous_f1 is None else f"{100 * (event['f1'] - previous_f1):+.1f} pp"
        previous_f1 = event["f1"]
        overview.append([name, f"{event['tp']} / {event['fp']} / {event['fn']}", pct(event["precision"]),
                         pct(event["recall"]), pct(event["f1"]), delta, pct(timing["temporal_iou"])])
        config_rows = [[key, value] for key, value in config.items()]
        clip_rows = [[clip, values["tp"], values["fp"], values["fn"], pct(values["f1"])]
                     for clip, values in sorted(event["per_clip"].items())]
        dashboard = "kpi_dashboard.html"
        videos = "annotated_vs_gt"
        links = []
        if (path / dashboard).exists():
            links.append(f'<a href="{html.escape(path.name + "/" + dashboard)}">Full KPI report</a>')
        if (path / videos).exists():
            links.append(f'<a href="{html.escape(path.name + "/" + videos + "/")}">Annotated videos vs GT</a>')
        change = config.get("experiment_change", "No change description recorded")
        details.append(f"<section><h2>{html.escape(name)}</h2><p>{html.escape(str(change))}</p>"
                       f"<p>{' · '.join(links)}</p><h3>Results by clip</h3>"
                       f"{table(['Clip','TP','FP','FN','F1'], clip_rows)}"
                       f"<h3>Full configuration</h3>{table(['Setting','Value'], config_rows)}</section>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>All person–vehicle experiments — comparison report</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f7f8fa;color:#1c2530}}main{{max-width:1250px;margin:auto}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{padding:9px 10px;border-bottom:1px solid #e4e8ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f7}}section{{margin-top:32px}}a{{margin-right:18px}}
</style></head><body><main><h1>All experiments — person–vehicle comparison</h1>
<p>Binary temporal event matching at temporal IoU ≥ 0.5; participant IDs ignored. F1 delta is relative to the preceding experiment.</p>
{table(['Experiment','TP / FP / FN','Precision','Recall','F1','F1 change','Time IoU'], overview)}
{''.join(details)}</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Compare all archived experiments")
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.experiments_root).resolve()
    output = Path(args.output).resolve() if args.output else root / "all_experiments.html"
    output.write_text(build(root), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
