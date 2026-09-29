"""Geometry-based person/vehicle appearance transitions.

This is intentionally conservative: appearance/disappearance alone is insufficient.
Exit uses growth from an initial detection inside the vehicle; entry uses motion
toward the vehicle followed by disappearance.
"""

from __future__ import annotations

import math
from copy import deepcopy


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


def _trend_fraction(values, increasing=True):
    changes = [right - left for left, right in zip(values, values[1:])]
    if not changes:
        return 0
    accepted = sum(change >= 0 if increasing else change <= 0 for change in changes)
    return accepted / len(changes)


def infer_person_vehicle_transitions(rows, candidates, metadata, config):
    """Infer enter/exit when a person track crosses a persistent vehicle boundary."""
    if not config.get("geometry_transitions", False):
        return []
    minimum = config.get("transition_min_observations", 4)
    distance_change = config.get("transition_distance_change", 0.18)
    size_ratio = config.get("transition_size_ratio", 1.25)
    persistence_s = config.get("transition_vehicle_persistence_s", 0.35)
    minimum_duration_s = config.get("transition_min_duration_s", 0.0)
    trend_fraction = config.get("transition_min_trend_fraction", 0.0)
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
                relative_area = _area(person["bbox"]) / max(_area(vehicle["bbox"]), 1)
                observations.append((row, person, vehicle, _distance(person["bbox"], vehicle["bbox"]),
                                     relative_area, _inside(person["bbox"], vehicle["bbox"])))
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
        if last_time - first_time < minimum_duration_s:
            continue
        vehicle_before = first_time - min(vehicle_times) >= persistence_s
        vehicle_after = max(vehicle_times) - last_time >= persistence_s
        sizes = [observation[4] for observation in observations]
        grows = last[4] >= first[4] * size_ratio and _trend_fraction(sizes) >= trend_fraction
        moves_in = first[3] - last[3] >= distance_change
        frame_step = min((b[0]["timestamp_s"] - a[0]["timestamp_s"] for a, b in zip(observations, observations[1:])
                          if b[0]["timestamp_s"] > a[0]["timestamp_s"]), default=0.04)

        action = None
        if vehicle_before and first[5] and grows:
            action = "exit"
            start_s, end_s = first_time, min(metadata["duration_s"], last_time + frame_step)
            evidence = [first[0]["frame_index"], observations[len(observations)//2][0]["frame_index"], last[0]["frame_index"]]
        elif vehicle_after and not first[5] and last[5] and moves_in:
            action = "enter"
            start_s, end_s = first_time, min(metadata["duration_s"], last_time + frame_step)
            evidence = [first[0]["frame_index"], observations[len(observations)//2][0]["frame_index"], last[0]["frame_index"]]
        if action and end_s > start_s:
            events.append({"event_id": "pending", "type": action,
                "persons": [{"person_id": person_id, "description": "Tracked person with geometric vehicle-boundary transition"}],
                "vehicle": {"vehicle_id": vehicle_id, "description": "Tracked vehicle associated with geometric transition"},
                "spans": [{"start_s": start_s, "end_s": end_s}],
                "truncated_start": start_s == 0, "truncated_end": end_s >= metadata["duration_s"],
                "evidence_frames": sorted(set(evidence)), "group_id": "geometry_transition"})
    return events


def fuse_transition_events(base_events, transition_events, config):
    """Use geometry to correct VLM events before optionally adding strong standalone events."""
    if config.get("geometry_fusion_mode", "add") == "add":
        return list(base_events) + list(transition_events)
    result = deepcopy(base_events)
    correctable = set(config.get("geometry_correctable_types",
                                 ["other_interaction", "door_operation", "enter", "exit"]))
    max_gap = config.get("geometry_correction_max_gap_s", 0.75)
    allow_standalone = config.get("geometry_allow_standalone", True)
    for transition in transition_events:
        start = transition["spans"][0]["start_s"]
        end = transition["spans"][0]["end_s"]
        matches = []
        for index, event in enumerate(result):
            if (event["type"] not in correctable or
                    event["persons"][0]["person_id"] != transition["persons"][0]["person_id"] or
                    event["vehicle"]["vehicle_id"] != transition["vehicle"]["vehicle_id"]):
                continue
            old_start = event["spans"][0]["start_s"]
            old_end = event["spans"][0]["end_s"]
            overlap = max(0, min(end, old_end) - max(start, old_start))
            gap = max(old_start - end, start - old_end, 0)
            if overlap > 0 or gap <= max_gap:
                matches.append((overlap, -gap, index))
        if matches:
            index = max(matches)[2]
            result[index]["type"] = transition["type"]
            result[index]["spans"] = deepcopy(transition["spans"])
            result[index]["evidence_frames"] = sorted(set(result[index]["evidence_frames"] + transition["evidence_frames"]))
            result[index]["group_id"] = "vlm_geometry_corrected"
        elif allow_standalone:
            result.append(transition)
    return result
