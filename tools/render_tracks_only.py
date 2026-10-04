"""Render every cached person and vehicle track onto an MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source video")
    parser.add_argument("--tracks", required=True, help="Tracking JSONL")
    parser.add_argument("--output", required=True, help="Destination MP4")
    args = parser.parse_args()

    source = Path(args.input)
    rows = {row["frame_index"]: row for row in
            (json.loads(line) for line in Path(args.tracks).read_text(encoding="utf-8").splitlines())}
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")

    index = 0
    rendered = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        row = rows.get(index, {"objects": []})
        for obj in row.get("objects", []):
            x1, y1, x2, y2 = (int(round(value)) for value in obj["bbox"])
            color = (210, 45, 230) if obj["kind"] == "person" else (255, 200, 35)
            thickness = 2 if obj.get("observed", True) else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{obj['id']} {obj['kind']} {obj.get('confidence', 0):.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            top = max(0, y1 - text_height - 8)
            cv2.rectangle(frame, (x1, top), (min(width - 1, x1 + text_width + 6), y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(text_height + 1, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {index} | {index / fps:.2f}s", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        index += 1
        rendered += 1

    capture.release()
    writer.release()
    if not rendered:
        raise RuntimeError("No frames were rendered")
    print(json.dumps({"output": str(destination.resolve()), "frames": rendered,
                      "fps": fps, "width": width, "height": height}))


if __name__ == "__main__":
    main()
