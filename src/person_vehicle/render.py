import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .events import ClipOutput, active_events
from .io import sha256, write_json
from .video import frames, probe


def draw_overlay(image, row, clip, reference=None):
    image = image.copy()
    height, width = image.shape[:2]
    scale = max(0.46, min(0.70, width / 1450))
    line_height = max(15, int(24 * scale / 0.52))
    active = active_events(clip["interactions"], row["timestamp_s"])
    reference_active = active_events(reference.get("interactions", []), row["timestamp_s"]) if reference else []
    participants = {e["vehicle"]["vehicle_id"] for e in active} | {
        p["person_id"] for e in active for p in e["persons"]}
    centers = {}
    for obj in row["objects"]:
        # Review output deliberately omits unrelated tracks: they create visual
        # clutter and make it hard to assess the algorithm's asserted pair.
        if obj["id"] not in participants:
            continue
        x1, y1, x2, y2 = [int(x) for x in obj["bbox"]]
        color = (230, 70, 230)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        label_width = cv2.getTextSize(obj["id"], cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        label_x = max(0, min(x1, width - label_width - 3))
        cv2.putText(image, obj["id"], (label_x, max(line_height, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        centers[obj["id"]] = ((x1 + x2) // 2, (y1 + y2) // 2)
    lines = [f"{clip['clip_id']}  {row['timestamp_s']:.2f}s  f{row['frame_index']}"]
    for event in active:
        person, vehicle = event["persons"][0], event["vehicle"]
        lines.append(f"ALG {event['type']}: {person['person_id']} -> {vehicle['vehicle_id']}")
        if person["person_id"] in centers and vehicle["vehicle_id"] in centers:
            cv2.line(image, centers[person["person_id"]], centers[vehicle["vehicle_id"]], (230, 70, 230), 1)
    for event in reference_active:
        person, vehicle = event["persons"][0], event["vehicle"]
        lines.append(f"GT  {event['type']}: {person['person_id']} -> {vehicle['vehicle_id']}")
    panel_height = len(lines) * line_height + 5
    image[:panel_height] = (image[:panel_height].astype(float) * 0.32).astype(np.uint8)
    for i, text in enumerate(lines):
        color = (230, 70, 230) if text.startswith("ALG") else (70, 220, 255) if text.startswith("GT") else (230, 230, 230)
        cv2.putText(image, text, (5, (i + 1) * line_height), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, height - 10), (width, height), (35, 35, 35), -1)
    for event in clip["interactions"]:
        for span in event["spans"]:
            left = int(span["start_s"] / clip["duration_s"] * (width - 1))
            right = int(span["end_s"] / clip["duration_s"] * (width - 1))
            cv2.rectangle(image, (left, height - 9), (right, height - 5), (230, 70, 230), -1)
    if reference:
        for event in reference.get("interactions", []):
            for span in event["spans"]:
                left = int(span["start_s"] / clip["duration_s"] * (width - 1))
                right = int(span["end_s"] / clip["duration_s"] * (width - 1))
                cv2.rectangle(image, (left, height - 4), (right, height - 1), (70, 220, 255), -1)
    cursor = int(row["timestamp_s"] / clip["duration_s"] * (width - 1))
    cv2.line(image, (cursor, height - 11), (cursor, height - 1), (255, 255, 255), 1)
    return image, [e["event_id"] for e in active]


def render(path, rows, clip, destination, reference=None):
    ClipOutput.model_validate(clip)
    if clip["status"] != "ok":
        raise ValueError("Cannot render a failed inference as a successful clip")
    if sha256(path) != clip["source_sha256"]:
        raise ValueError("Source checksum differs from event artifact")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    representative_dir = destination.parent / f"{destination.stem}_events"
    representative_dir.mkdir(parents=True, exist_ok=True)
    representatives = {}
    target_frames = {}
    timestamps = [row["timestamp_s"] for row in rows]
    for event in clip["interactions"]:
        span = event["spans"][0]
        points = {"start": span["start_s"], "middle": (span["start_s"] + span["end_s"]) / 2,
                  "end": span["end_s"]}
        representatives[event["event_id"]] = {}
        for phase, timestamp in points.items():
            index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - timestamp))
            target_frames.setdefault(index, []).append((event["event_id"], phase, timestamp))
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    time_base = Fraction(1, 90000)
    metadata = []
    stream = None
    with av.open(str(temporary), "w") as output:
        for index, timestamp, frame in frames(path):
            if index >= len(rows) or rows[index]["frame_index"] != index or abs(rows[index]["timestamp_s"] - timestamp) > 1e-6:
                raise ValueError("Track/frame alignment mismatch")
            image, active = draw_overlay(frame.to_ndarray(format="bgr24"), rows[index], clip, reference)
            for event_id, phase, target_timestamp in target_frames.get(index, []):
                filename = f"{event_id}_{phase}.jpg"
                cv2.imwrite(str(representative_dir / filename), image)
                representatives[event_id][phase] = {"frame_index": index,
                    "timestamp_s": timestamp, "target_timestamp_s": target_timestamp,
                    "path": str((representative_dir / filename).relative_to(destination.parent))}
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
          "sha256": sha256(destination), "visual_review": "pending",
          "representative_frames": representatives}
    write_json(destination.with_suffix(".qa.json"), qa)
    return qa
