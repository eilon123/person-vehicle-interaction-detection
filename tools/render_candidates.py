"""Render pre-VLM person-vehicle candidate windows onto a source video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = {row["frame_index"]: row for row in
            (json.loads(line) for line in Path(args.tracks).read_text(encoding="utf-8").splitlines())}
    candidates = read_json(args.candidates)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = index / fps
        row = rows.get(index, {"objects": []})
        objects = {obj["id"]: obj for obj in row.get("objects", [])}
        active = [c for c in candidates if c["start_s"] <= timestamp < c["end_s"]]
        active_ids = {identifier for c in active for identifier in (c["person_id"], c["vehicle_id"])}

        for obj in objects.values():
            x1, y1, x2, y2 = (int(round(v)) for v in obj["bbox"])
            selected = obj["id"] in active_ids
            color = (30, 230, 80) if obj["kind"] == "person" else (255, 210, 30)
            if not selected:
                color = tuple(int(v * .45) for v in color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if selected else 1)
            cv2.putText(frame, obj["id"], (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, color, 2 if selected else 1, cv2.LINE_AA)

        panel_lines = [f"PRE-VLM CANDIDATES | {timestamp:.2f}s | frame {index}"]
        for candidate in active:
            person = objects.get(candidate["person_id"])
            vehicle = objects.get(candidate["vehicle_id"])
            panel_lines.append(f"{candidate['candidate_id']}  {candidate['person_id']} -> {candidate['vehicle_id']}  "
                               f"[{candidate['start_s']:.2f}-{candidate['end_s']:.2f}s]")
            if person and vehicle:
                pc = tuple(int((person["bbox"][i] + person["bbox"][i + 2]) / 2) for i in (0, 1))
                vc = tuple(int((vehicle["bbox"][i] + vehicle["bbox"][i + 2]) / 2) for i in (0, 1))
                cv2.line(frame, pc, vc, (0, 170, 255), 3)
        panel_height = 9 + 22 * len(panel_lines)
        frame[:panel_height] = (frame[:panel_height].astype("float32") * .28).astype("uint8")
        for line_no, label in enumerate(panel_lines, 1):
            cv2.putText(frame, label, (8, line_no * 21), cv2.FONT_HERSHEY_SIMPLEX, .53,
                        (245, 245, 245) if line_no == 1 else (0, 210, 255), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, height - 14), (width - 1, height - 1), (25, 25, 25), -1)
        for candidate in candidates:
            left = round(candidate["start_s"] / duration * (width - 1))
            right = round(candidate["end_s"] / duration * (width - 1))
            cv2.rectangle(frame, (left, height - 12), (right, height - 4), (0, 165, 255), -1)
        cursor = round(timestamp / duration * (width - 1))
        cv2.line(frame, (cursor, height - 14), (cursor, height - 1), (255, 255, 255), 2)
        writer.write(frame)
        index += 1

    capture.release()
    writer.release()
    print(json.dumps({"output": str(destination.resolve()), "frames": index,
                      "fps": fps, "candidates": len(candidates)}))


if __name__ == "__main__":
    main()
