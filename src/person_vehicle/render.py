import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .events import ClipOutput, active_events
from .io import sha256, write_json
from .video import frames, probe


def draw_overlay(image, row, clip):
    image = image.copy()
    height, width = image.shape[:2]
    scale = max(0.45, min(2.0, width / 1000))
    line_height = max(15, int(25 * scale / 0.65))
    active = active_events(clip["interactions"], row["timestamp_s"])
    participants = {e["vehicle"]["vehicle_id"] for e in active} | {
        p["person_id"] for e in active for p in e["persons"]}
    centers = {}
    for obj in row["objects"]:
        x1, y1, x2, y2 = [int(x) for x in obj["bbox"]]
        color = (90, 220, 90) if obj["id"] in participants else (160, 160, 160)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_width = cv2.getTextSize(obj["id"], cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        label_x = max(0, min(x1, width - label_width - 3))
        cv2.putText(image, obj["id"], (label_x, max(line_height, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        centers[obj["id"]] = ((x1 + x2) // 2, (y1 + y2) // 2)
    lines = [f"{clip['clip_id']} | {row['timestamp_s']:.2f}s | frame {row['frame_index']}"]
    if not active:
        lines.append("No confirmed interaction")
    for event in active:
        person, vehicle = event["persons"][0], event["vehicle"]
        lines.append(f"{event['event_id']} {event['type'].upper()}")
        lines.append(f"{person['person_id']} -> {vehicle['vehicle_id']}")
        if person["person_id"] in centers and vehicle["vehicle_id"] in centers:
            cv2.line(image, centers[person["person_id"]], centers[vehicle["vehicle_id"]], (0, 215, 255), 2)
        # Wrap by rendered width rather than character count.
        for description in (person["description"], vehicle["description"]):
            current = ""
            for word in description.split():
                proposed = f"{current} {word}".strip()
                if cv2.getTextSize(proposed, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width - 16 and current:
                    lines.append(current)
                    current = word
                else:
                    current = proposed
            if current:
                lines.append(current)
    # Headers and action labels also need wrapping on low-resolution footage.
    wrapped = []
    for line in lines:
        current = ""
        for word in line.split():
            proposed = f"{current} {word}".strip()
            if cv2.getTextSize(proposed, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width - 16 and current:
                wrapped.append(current)
                current = word
            else:
                current = proposed
        wrapped.append(current)
    lines = wrapped
    # Choose top/bottom based on overlap with tracked entities rather than
    # covering the action by always placing the panel in the same corner.
    max_lines = max(2, height // (3 * line_height))
    shown = lines[:max_lines]
    if len(lines) > max_lines:
        shown[-1] = f"{len(active)} active; details in JSON"
    panel_height = len(shown) * line_height + 8
    def obstruction(y):
        return sum(max(0, min(y + panel_height, obj["bbox"][3]) - max(y, obj["bbox"][1])) *
                   max(0, obj["bbox"][2] - obj["bbox"][0]) for obj in row["objects"])
    panel_y = min((0, max(0, height - 16 - panel_height)), key=obstruction)
    image[panel_y:panel_y + panel_height] = (image[panel_y:panel_y + panel_height].astype(float) * 0.3).astype(np.uint8)
    for i, text in enumerate(shown):
        cv2.putText(image, text, (6, panel_y + (i + 1) * line_height), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, height - 12), (width, height), (35, 35, 35), -1)
    for event in clip["interactions"]:
        for span in event["spans"]:
            left = int(span["start_s"] / clip["duration_s"] * (width - 1))
            right = int(span["end_s"] / clip["duration_s"] * (width - 1))
            cv2.rectangle(image, (left, height - 10), (right, height - 3), (90, 220, 90), -1)
    cursor = int(row["timestamp_s"] / clip["duration_s"] * (width - 1))
    cv2.line(image, (cursor, height - 13), (cursor, height - 1), (255, 255, 255), 2)
    return image, [e["event_id"] for e in active]


def render(path, rows, clip, destination):
    ClipOutput.model_validate(clip)
    if clip["status"] != "ok":
        raise ValueError("Cannot render a failed inference as a successful clip")
    if sha256(path) != clip["source_sha256"]:
        raise ValueError("Source checksum differs from event artifact")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    time_base = Fraction(1, 90000)
    metadata = []
    stream = None
    with av.open(str(temporary), "w") as output:
        for index, timestamp, frame in frames(path):
            if index >= len(rows) or rows[index]["frame_index"] != index or abs(rows[index]["timestamp_s"] - timestamp) > 1e-6:
                raise ValueError("Track/frame alignment mismatch")
            image, active = draw_overlay(frame.to_ndarray(format="bgr24"), rows[index], clip)
            image = cv2.copyMakeBorder(image, 0, image.shape[0] % 2, 0, image.shape[1] % 2,
                                      cv2.BORDER_CONSTANT)
            if stream is None:
                rate = Fraction(clip["frame_count"] / clip["duration_s"]).limit_denominator(1001)
                stream = output.add_stream("libx264", rate=rate)
                stream.width, stream.height = image.shape[1], image.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.time_base = time_base
                stream.codec_context.time_base = time_base
                stream.options = {"crf": "20", "preset": "fast", "bf": "0"}
            encoded = av.VideoFrame.from_ndarray(image, format="bgr24")
            encoded.pts = round(timestamp / time_base)
            encoded.time_base = time_base
            for packet in stream.encode(encoded):
                output.mux(packet)
            metadata.append({"frame_index": index, "timestamp_s": timestamp, "active_event_ids": active})
        if stream is None:
            raise ValueError("Empty source")
        for packet in stream.encode():
            output.mux(packet)
    verified = probe(temporary)
    expected = [row["timestamp_s"] for row in rows]
    if verified["frame_count"] != len(rows):
        raise ValueError("Export changed frame count")
    error = max(abs(a - b) for a, b in zip(expected, verified["timestamps"]))
    tolerance = max(np.diff(expected), default=clip["duration_s"])
    duration_error = abs(verified["duration_s"] - clip["duration_s"])
    if error > tolerance or duration_error > tolerance + 1e-4:
        raise ValueError(f"Export timing mismatch: timestamps {error}s; duration {duration_error}s")
    temporary.replace(destination)
    with destination.with_suffix(".frames.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row) + "\n")
    qa = {"clip_id": clip["clip_id"], "fully_decoded": True, "frame_count": len(rows),
          "max_timestamp_error_s": error, "duration_error_s": duration_error,
          "overlay_agreement": sum(row["active_event_ids"] == [e["event_id"] for e in active_events(clip["interactions"], row["timestamp_s"])] for row in metadata) / len(metadata),
          "sha256": sha256(destination), "visual_review": "pending"}
    write_json(destination.with_suffix(".qa.json"), qa)
    return qa
