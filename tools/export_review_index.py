"""Create an offline HTML index for reviewing generated video/JSON pairs."""
import argparse
import html
from pathlib import Path

from person_vehicle.io import read_json


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/final")
args = parser.parse_args()
root = Path(args.output)
cards = []
for path in sorted((root / "clips").glob("*.json")):
    clip = read_json(path)
    name = html.escape(clip["clip_id"], quote=True)
    descriptions = []
    for event in clip["interactions"]:
        spans = ", ".join(f"{s['start_s']:.2f}–{s['end_s']:.2f}s" for s in event["spans"])
        descriptions.append("<li>" + html.escape(f"{event['event_id']} · {event['type']} · {spans} · "
            f"{event['persons'][0]['description']} → {event['vehicle']['description']}") + "</li>")
    content = "<ul>" + "".join(descriptions) + "</ul>" if descriptions else "<p>No confirmed interactions.</p>"
    if clip["status"] != "ok":
        content = "<p>Processing failed: " + html.escape(clip.get("error", "")) + "</p>"
    video = root / "annotated" / f"{clip['clip_id']}_annotated.mp4"
    player = f'<video controls preload="metadata" src="annotated/{name}_annotated.mp4"></video>' if video.exists() else "<p>Annotated video unavailable.</p>"
    cards.append(f'<article><h2>{name}</h2>{player}{content}<a href="clips/{name}.json">Event JSON</a></article>')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Person–vehicle interaction review</title><style>
body{font:16px system-ui,sans-serif;background:#111827;color:#e5e7eb;margin:0;padding:32px;max-width:1200px;margin:auto}
h1{font-size:30px}p,li{line-height:1.6}article{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:20px;margin:24px 0}
h2{font-size:20px}video{width:100%;max-height:650px;background:#000}a{color:#93c5fd}li{margin:10px 0}
</style><h1>Person–vehicle interaction review</h1><p>Automatic predictions, with explicit uncertainty in the companion review JSON. Semantic accuracy has not been independently established. Use the videos and event links to inspect each clip.</p>'''
root.mkdir(parents=True, exist_ok=True)
(root / "index.html").write_text(page + "".join(cards) + "</html>", encoding="utf-8")
print(root / "index.html")
