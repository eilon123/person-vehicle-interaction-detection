"""Geometry-based person/vehicle appearance transitions.

This is intentionally conservative: appearance/disappearance alone is insufficient.
Exit uses growth from an initial detection inside the vehicle; entry uses motion
toward the vehicle followed by disappearance.
"""

from __future__ import annotations

import math


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _inside(person, vehicle, margin=0.08):
    x, y = _center(person)
    width, height = vehicle[2] - vehicle[0], vehicle[3] - vehicle[1]
    return (vehicle[0] - margin * width <= x <= vehicle[2] + margin * width and
            vehicle[1] - margin * height <= y <= vehicle[3] + margin * height)


def _distance(person, vehicle):
    px, py = _center(person)
    vx, vy = _center(vehicle)
    scale = max(math.hypot(vehicle[2] - vehicle[0], vehicle[3] - vehicle[1]), 1)
    return math.hypot(px - vx, py - vy) / scale


def infer_person_vehicle_transitions(rows, candidates, metadata, config):
    """Infer enter/exit when a person track crosses a persistent vehicle boundary."""
    if not config.get("geometry_transitions", False):
        return []
    minimum = config.get("transition_min_observations", 4)
    distance_change = config.get("transition_distance_change", 0.18)
    size_ratio = config.get("transition_size_ratio", 1.25)
    persistence_s = config.get("transition_vehicle_persistence_s", 0.35)
    context_s = config.get("transition_event_context_s", 1.5)
    events = []
    pairs = sorted({(c["person_id"], c["vehicle_id"], c["scene"]) for c in candidates})
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)

    for person_id, vehicle_id, scene in pairs:
        scene_rows = by_scene.get(scene, [])
        observations = []
        for row in scene_rows:
            objects = {obj["id"]: obj for obj in row["objects"]}
            if person_id in objects and vehicle_id in objects:
                person, vehicle = objects[person_id], objects[vehicle_id]
                observations.append((row, person, vehicle, _distance(person["bbox"], vehicle["bbox"]),
                                     _area(person["bbox"]), _inside(person["bbox"], vehicle["bbox"])))
        if len(observations) < minimum:
            continue
        first, last = observations[0], observations[-1]
        person_times = [row["timestamp_s"] for row in scene_rows
                        if any(obj["id"] == person_id for obj in row["objects"])]
        vehicle_times = [row["timestamp_s"] for row in scene_rows
                         if any(obj["id"] == vehicle_id for obj in row["objects"])]
        if not person_times or not vehicle_times:
            continue
        first_time, last_time = person_times[0], person_times[-1]
        vehicle_before = first_time - min(vehicle_times) >= persistence_s
        vehicle_after = max(vehicle_times) - last_time >= persistence_s
        grows = last[4] >= max(first[4] * size_ratio, first[4] + 1)
        shrinks = first[4] >= max(last[4] * size_ratio, last[4] + 1)
        moves_in = first[3] - last[3] >= distance_change

        action = None
        if vehicle_before and first[5] and grows:
            action = "exit"
            start_s, end_s = first_time, min(metadata["duration_s"], last_time + context_s / 3)
            evidence = [first[0]["frame_index"], observations[len(observations)//2][0]["frame_index"], last[0]["frame_index"]]
        elif vehicle_after and last[5] and shrinks and moves_in:
            action = "enter"
            start_s, end_s = max(0, first_time - context_s / 3), min(metadata["duration_s"], last_time + persistence_s)
            evidence = [first[0]["frame_index"], observations[len(observations)//2][0]["frame_index"], last[0]["frame_index"]]
        if action and end_s > start_s:
            events.append({"event_id": "pending", "type": action,
                "persons": [{"person_id": person_id, "description": "Tracked person with geometric vehicle-boundary transition"}],
                "vehicle": {"vehicle_id": vehicle_id, "description": "Tracked vehicle associated with geometric transition"},
                "spans": [{"start_s": start_s, "end_s": end_s}],
                "truncated_start": start_s == 0, "truncated_end": end_s >= metadata["duration_s"],
                "evidence_frames": sorted(set(evidence)), "group_id": "geometry_transition"})
    return events
