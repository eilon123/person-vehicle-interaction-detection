"""Deliberately simple geometry baseline for ablation, not the final verifier.

Appearance/disappearance near a vehicle can also be occlusion. These predictions
must be evaluated as baseline guesses; the VLM path is the primary pipeline.
"""
from .candidates import near
from .events import merge_events


def geometry_baseline(rows, metadata, config):
    tracks = {}
    for row in rows:
        for obj in row["objects"]:
            if obj["kind"] == "person":
                tracks.setdefault(obj["id"], []).append((row, obj))
    events = []
    for identity, observations in tracks.items():
        if len(observations) < 3:
            continue
        for action, observation in (("exit", observations[0]), ("enter", observations[-1])):
            row, person = observation
            timestamp = row["timestamp_s"]
            if timestamp < 0.5 or timestamp > metadata["duration_s"] - 0.5:
                continue
            # Never interpret a scene-cut boundary as cabin entry/exit.
            i = row["frame_index"]
            neighbour = i - 1 if action == "exit" else i + 1
            if not 0 <= neighbour < len(rows) or rows[neighbour]["scene"] != row["scene"]:
                continue
            for vehicle in row["objects"]:
                if vehicle["kind"] == "person" or not near(person["bbox"], vehicle["bbox"], 0):
                    continue
                start = max(observations[0][0]["timestamp_s"], timestamp - 1.5) if action == "enter" else timestamp
                end = timestamp + metadata["last_frame_duration_s"] if action == "enter" else min(timestamp + 1.5, observations[-1][0]["timestamp_s"])
                if end <= start:
                    continue
                events.append({"event_id": "pending", "type": action,
                    "persons": [{"person_id": identity, "description": f"Tracked person {identity}; appearance not described by geometry baseline"}],
                    "vehicle": {"vehicle_id": vehicle["id"], "description": f"Tracked {vehicle['kind']} {vehicle['id']}; color unknown"},
                    "spans": [{"start_s": start, "end_s": min(end, metadata["duration_s"])}],
                    "evidence_frames": [i], "truncated_start": False, "truncated_end": False, "group_id": None})
    return merge_events(events)
