"""Select annotated frames at event boundaries and uniform points for visual QA."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from person_vehicle.io import read_json, write_json
from person_vehicle.video import frames


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/final")
parser.add_argument("--destination", default="outputs/render_review/final")
args = parser.parse_args()
root, destination = Path(args.output), Path(args.destination)
destination.mkdir(parents=True, exist_ok=True)
selections = {}
for event_path in sorted((root / "clips").glob("*.json")):
    clip = read_json(event_path)
    video = root / "annotated" / f"{clip['clip_id']}_annotated.mp4"
    if clip["status"] != "ok" or not video.exists():
        continue
    stamps = read_json(root / "audit" / f"{clip['clip_id']}.json")["timestamps"]
    selected = set(np.linspace(0, len(stamps) - 1, 8, dtype=int).tolist())
    for event in clip["interactions"]:
        for span in event["spans"]:
            for timestamp in (span["start_s"], (span["start_s"] + span["end_s"]) / 2, span["end_s"]):
                index = int(np.searchsorted(stamps, timestamp))
                selected.update(i for i in (index - 1, index) if 0 <= i < len(stamps))
    sheets, page, count = [], None, 0
    for index, timestamp, frame in frames(video):
        if index not in selected:
            continue
        if count % 8 == 0:
            page = Image.new("RGB", (1600, 1050), "#151b25")
            sheets.append(page)
        image = frame.to_image()
        image.thumbnail((800, 235))
        x, y = (count % 2) * 800, ((count % 8) // 2) * 262
        page.paste(image, (x, y))
        ImageDraw.Draw(page).text((x + 5, y + 240), f"{clip['clip_id']} | frame {index} | {timestamp:.3f}s", fill="white")
        count += 1
    for index, sheet in enumerate(sheets, 1):
        sheet.save(destination / f"{clip['clip_id']}_{index:02d}.jpg", quality=92)
    selections[clip["clip_id"]] = {"frame_indices": sorted(selected), "sheets": len(sheets), "review_status": "pending"}
write_json(destination / "selection.json", selections)
print(destination)
