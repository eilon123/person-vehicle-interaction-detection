from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Span(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s <= self.start_s:
            raise ValueError("Span must have positive duration")
        return self


class Person(StrictModel):
    person_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Vehicle(StrictModel):
    vehicle_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Event(StrictModel):
    event_id: str
    type: Literal["enter", "exit", "door_operation", "load_unload", "other_interaction"]
    persons: list[Person] = Field(min_length=1, max_length=1)
    vehicle: Vehicle
    spans: list[Span] = Field(min_length=1)
    truncated_start: bool = False
    truncated_end: bool = False
    evidence_frames: list[int] = Field(min_length=1)
    group_id: str | None = None

    @model_validator(mode="after")
    def ordered(self):
        for left, right in zip(self.spans, self.spans[1:]):
            if right.start_s < left.end_s:
                raise ValueError("Spans must be sorted and disjoint")
        if any(i < 0 for i in self.evidence_frames):
            raise ValueError("Frame indices cannot be negative")
        return self


class ClipOutput(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    clip_id: str
    source_sha256: str
    status: Literal["ok", "error"]
    duration_s: float = Field(ge=0)
    frame_count: int = Field(ge=0)
    interactions: list[Event]
    error: str | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "error" and (not self.error or self.interactions):
            raise ValueError("Error outputs need a reason and no events")
        if self.status == "ok" and (self.error or self.frame_count == 0):
            raise ValueError("Successful outputs need decoded frames and no error")
        if len({e.event_id for e in self.interactions}) != len(self.interactions):
            raise ValueError("Duplicate event IDs")
        for event in self.interactions:
            if event.spans[-1].end_s > self.duration_s + 1e-6:
                raise ValueError("Event exceeds clip duration")
            if max(event.evidence_frames) >= self.frame_count:
                raise ValueError("Evidence frame outside clip")
        return self


def active_events(events, timestamp):
    return [e for e in events if any(s["start_s"] <= timestamp < s["end_s"] for s in e["spans"])]


def clip_events_to_person_visibility(events, rows):
    """Never start an event before its person first exists in the track.

    Candidate context and temporal expansion may begin before the detector has
    present the associated person. Clamp final spans after all merging and
    expansion so overlays and metrics do not claim an interaction before the
    tracker has a measured or predicted box. Synthetic ``untracked_`` probes
    remain governed by their visual verifier because they intentionally have
    no person track.
    """
    if not events or not rows:
        return events
    observed = {}
    frame_times = {}
    for row in rows:
        frame_times[row.get("frame_index")] = row["timestamp_s"]
        for obj in row["objects"]:
            if obj["kind"] == "person":
                observed.setdefault(obj["id"], []).append(
                    (row["timestamp_s"], row.get("frame_index")))
    result = []
    for original in events:
        event = dict(original)
        person_id = event["persons"][0]["person_id"]
        if person_id.startswith("untracked_"):
            result.append(event)
            continue
        sightings = observed.get(person_id, [])
        if not sightings:
            continue
        first_visible, first_frame = sightings[0]
        spans = []
        for span in event["spans"]:
            start = max(span["start_s"], first_visible)
            if span["end_s"] > start:
                spans.append({"start_s": start, "end_s": span["end_s"]})
        if not spans:
            continue
        event["spans"] = spans
        evidence = [frame for frame in event.get("evidence_frames", [])
                    if frame_times.get(frame, float("inf")) >= first_visible]
        event["evidence_frames"] = evidence or [first_frame]
        result.append(event)
    return result


def _vehicle_gap(person_box, vehicle_box):
    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = vehicle_box
    dx = max(vx1 - px2, px1 - vx2, 0)
    dy = max(vy1 - py2, py1 - vy2, 0)
    return ((dx * dx + dy * dy) ** 0.5) / max(vx2 - vx1, vy2 - vy1, 1e-6)


def enforce_single_vehicle(events, rows):
    """Trim overlapping events so each person has at most one active vehicle.

    During a conflict, the closest live person/vehicle track wins. If geometry
    is unavailable, prefer the event with more verifier evidence, then a stable
    vehicle-ID tie break. Non-overlapping events are retained unchanged.
    """
    if not events or not rows:
        return events
    frame_rows = sorted(rows, key=lambda row: row["timestamp_s"])
    frame_times = [row["timestamp_s"] for row in frame_rows]
    frame_by_index = {row["frame_index"]: row for row in frame_rows if "frame_index" in row}
    boundaries = set(frame_times)
    for event in events:
        boundaries.update(span["start_s"] for span in event["spans"])
        boundaries.update(span["end_s"] for span in event["spans"])
    cuts = sorted(boundaries)
    kept = [[] for _ in events]

    def active_at(event, timestamp):
        return any(span["start_s"] <= timestamp < span["end_s"] for span in event["spans"])

    for start, end in zip(cuts, cuts[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        active = [i for i, event in enumerate(events) if active_at(event, midpoint)]
        by_person = {}
        for i in active:
            person_id = events[i]["persons"][0]["person_id"]
            by_person.setdefault(person_id, []).append(i)
        winners = set(active)
        for person_id, indices in by_person.items():
            vehicle_ids = {events[i]["vehicle"]["vehicle_id"] for i in indices}
            if len(vehicle_ids) <= 1:
                continue
            nearest_index = min(range(len(frame_times)), key=lambda j: abs(frame_times[j] - midpoint))
            row = frame_rows[nearest_index]
            objects = {obj["id"]: obj for obj in row["objects"]}
            person_obj = objects.get(person_id)
            def rank(i):
                event = events[i]
                vehicle_id = event["vehicle"]["vehicle_id"]
                vehicle_obj = objects.get(vehicle_id)
                gap = (_vehicle_gap(person_obj["bbox"], vehicle_obj["bbox"])
                       if person_obj and vehicle_obj else float("inf"))
                evidence = len(event.get("evidence_frames", []))
                return (gap, -evidence, vehicle_id, i)
            winner = min(indices, key=rank)
            winning_vehicle = events[winner]["vehicle"]["vehicle_id"]
            winners.difference_update(i for i in indices
                                      if events[i]["vehicle"]["vehicle_id"] != winning_vehicle)
        for i in winners:
            if kept[i] and abs(kept[i][-1]["end_s"] - start) < 1e-9:
                kept[i][-1]["end_s"] = end
            else:
                kept[i].append({"start_s": start, "end_s": end})

    result = []
    for i, event in enumerate(events):
        if not kept[i]:
            continue
        event["spans"] = kept[i]
        evidence = [frame for frame in event.get("evidence_frames", [])
                    if frame in frame_by_index and active_at(event, frame_by_index[frame]["timestamp_s"])]
        if evidence:
            event["evidence_frames"] = evidence
        elif frame_by_index:
            midpoint = (kept[i][0]["start_s"] + kept[i][0]["end_s"]) / 2
            nearest_frame = min(frame_by_index, key=lambda frame: abs(frame_by_index[frame]["timestamp_s"] - midpoint))
            event["evidence_frames"] = [nearest_frame]
        result.append(event)
    return result


def merge_events(events, rows=None):
    """Merge overlapping proposals for the same pair and action, never across gaps."""
    result = []
    for event in sorted(events, key=lambda e: e["spans"][0]["start_s"]):
        match = next((old for old in result if old["type"] == event["type"]
                      and old["persons"][0]["person_id"] == event["persons"][0]["person_id"]
                      and old["vehicle"]["vehicle_id"] == event["vehicle"]["vehicle_id"]
                      and any(a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]
                              for a in old["spans"] for b in event["spans"])), None)
        if match is None:
            result.append(event)
            continue
        spans = sorted(match["spans"] + event["spans"], key=lambda s: s["start_s"])
        merged = []
        for span in spans:
            if merged and span["start_s"] <= merged[-1]["end_s"]:
                merged[-1]["end_s"] = max(merged[-1]["end_s"], span["end_s"])
            else:
                merged.append(dict(span))
        match["spans"] = merged
        match["evidence_frames"] = sorted(set(match["evidence_frames"] + event["evidence_frames"]))
        match["truncated_start"] |= event["truncated_start"]
        match["truncated_end"] |= event["truncated_end"]
    if rows is not None:
        result = enforce_single_vehicle(result, rows)
    for index, event in enumerate(result, 1):
        event["event_id"] = f"e{index:03d}"
    return result


def collapse_physical_episodes(events, rows, config):
    """Apply temporal NMS to duplicate descriptions of one interaction.

    Vehicle tracker IDs are treated as aliases only when their boxes strongly
    overlap in the same frame. Events are then clustered when their intervals
    overlap strongly. Disjoint events are never joined. Conflicting labels in
    the same duplicate cluster become ``other_interaction``.
    """
    if not config.get("collapse_physical_episodes", False) or len(events) < 2 or not rows:
        return events

    parent = {}
    def find(item):
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    def overlap_ratio(first, second):
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        a = max(1e-6, (first[2] - first[0]) * (first[3] - first[1]))
        b = max(1e-6, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / min(a, b)

    alias_overlap = float(config.get("episode_vehicle_containment", .8))
    nearby_vehicle_gap = float(config.get("episode_vehicle_near_gap", .15))
    nearby_vehicle_pairs = set()
    vehicle_frames = {}
    for row in rows:
        vehicles = [obj for obj in row["objects"] if obj["kind"] != "person"]
        for vehicle in vehicles:
            vehicle_frames[vehicle["id"]] = vehicle_frames.get(vehicle["id"], 0) + 1
        for i, first in enumerate(vehicles):
            for second in vehicles[i + 1:]:
                if overlap_ratio(first["bbox"], second["bbox"]) >= alias_overlap:
                    union(first["id"], second["id"])
                a, b = first["bbox"], second["bbox"]
                dx = max(a[0] - b[2], b[0] - a[2], 0)
                dy = max(a[1] - b[3], b[1] - a[3], 0)
                scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1e-6)
                if (dx * dx + dy * dy) ** .5 / scale <= nearby_vehicle_gap:
                    nearby_vehicle_pairs.add(frozenset((first["id"], second["id"])))

    bounds = [(min(span["start_s"] for span in event["spans"]),
               max(span["end_s"] for span in event["spans"])) for event in events]
    episode_parent = list(range(len(events)))
    def episode_find(index):
        if episode_parent[index] != index:
            episode_parent[index] = episode_find(episode_parent[index])
        return episode_parent[index]
    minimum_iou = float(config.get("event_nms_temporal_iou", .5))
    minimum_containment = float(config.get("event_nms_temporal_containment", .8))
    merge_overlapping_people = bool(config.get("episode_merge_overlapping_people", False))
    person_overlap_threshold = float(config.get("episode_person_overlap", .2))
    person_center_threshold = float(config.get("episode_person_center_distance", .25))

    def people_spatially_overlap(first, second, start_s, end_s):
        first_ids = {person["person_id"] for person in first.get("persons", [])}
        second_ids = {person["person_id"] for person in second.get("persons", [])}
        if first_ids & second_ids:
            return True
        for row in rows:
            if not start_s <= row["timestamp_s"] <= end_s:
                continue
            objects = {obj["id"]: obj for obj in row["objects"]}
            first_boxes = [objects[identity]["bbox"] for identity in first_ids if identity in objects]
            second_boxes = [objects[identity]["bbox"] for identity in second_ids if identity in objects]
            for first_box in first_boxes:
                for second_box in second_boxes:
                    if overlap_ratio(first_box, second_box) >= person_overlap_threshold:
                        return True
                    first_center = ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2)
                    second_center = ((second_box[0] + second_box[2]) / 2, (second_box[1] + second_box[3]) / 2)
                    scale = max(first_box[2] - first_box[0], first_box[3] - first_box[1],
                                second_box[2] - second_box[0], second_box[3] - second_box[1], 1e-6)
                    distance = ((first_center[0] - second_center[0]) ** 2 +
                                (first_center[1] - second_center[1]) ** 2) ** .5 / scale
                    if distance <= person_center_threshold:
                        return True
        return False
    for i, first in enumerate(events):
        for j in range(i + 1, len(events)):
            second = events[j]
            first_vehicle, second_vehicle = (first["vehicle"]["vehicle_id"],
                                             second["vehicle"]["vehicle_id"])
            same_vehicle = find(first_vehicle) == find(second_vehicle)
            same_person = first["persons"][0]["person_id"] == second["persons"][0]["person_id"]
            localized_handoff = same_person and frozenset((first_vehicle, second_vehicle)) in nearby_vehicle_pairs
            if not same_vehicle and not localized_handoff:
                continue
            intersection = max(0, min(bounds[i][1], bounds[j][1]) -
                               max(bounds[i][0], bounds[j][0]))
            if (merge_overlapping_people and not same_person and
                    not people_spatially_overlap(first, second,
                                                 max(bounds[i][0], bounds[j][0]),
                                                 min(bounds[i][1], bounds[j][1]))):
                continue
            first_duration = max(1e-6, bounds[i][1] - bounds[i][0])
            second_duration = max(1e-6, bounds[j][1] - bounds[j][0])
            temporal_iou = intersection / (first_duration + second_duration - intersection)
            containment = intersection / min(first_duration, second_duration)
            if temporal_iou < minimum_iou and containment < minimum_containment:
                continue
            left, right = episode_find(i), episode_find(j)
            if left != right:
                episode_parent[right] = left

    groups = {}
    for index in range(len(events)):
        groups.setdefault(episode_find(index), []).append(index)
    result = []
    for indices in groups.values():
        if len(indices) == 1:
            result.append(events[indices[0]])
            continue
        durations = {}
        for index in indices:
            duration = sum(span["end_s"] - span["start_s"] for span in events[index]["spans"])
            durations[events[index]["type"]] = durations.get(events[index]["type"], 0) + duration
        winning_type = (next(iter(durations)) if len(durations) == 1 else "other_interaction")
        eligible = [index for index in indices if events[index]["type"] == winning_type]
        if not eligible:
            eligible = indices
        representative_index = max(eligible, key=lambda index: sum(
            span["end_s"] - span["start_s"] for span in events[index]["spans"]))
        representative = dict(events[representative_index])
        representative["type"] = winning_type
        representative["persons"] = [dict(events[representative_index]["persons"][0])]
        vehicle_durations = {}
        for index in indices:
            identity = events[index]["vehicle"]["vehicle_id"]
            vehicle_durations[identity] = vehicle_durations.get(identity, 0) + sum(
                span["end_s"] - span["start_s"] for span in events[index]["spans"])
        vehicle_id = max(sorted(vehicle_durations), key=lambda identity: vehicle_durations[identity])
        vehicle_event = next(events[index] for index in indices
                             if events[index]["vehicle"]["vehicle_id"] == vehicle_id)
        representative["vehicle"] = dict(vehicle_event["vehicle"])
        representative["spans"] = [{"start_s": min(bounds[index][0] for index in indices),
                                     "end_s": max(bounds[index][1] for index in indices)}]
        representative["evidence_frames"] = sorted({frame for index in indices
                                                      for frame in events[index]["evidence_frames"]})
        representative["truncated_start"] = any(events[index]["truncated_start"] for index in indices)
        representative["truncated_end"] = any(events[index]["truncated_end"] for index in indices)
        result.append(representative)
    return merge_events(result)


def expand_event_boundaries(events, rows, config, duration):
    """Expand verified seeds with lower-threshold person-vehicle continuity."""
    if not config.get("temporal_expansion_enabled", False) or not events or not rows:
        return events
    ordered = sorted(rows, key=lambda row: row["timestamp_s"])
    margin = float(config.get("temporal_expansion_near_margin", .35))
    maximum_gap = float(config.get("temporal_expansion_max_gap_s", 1.0))
    maximum_extension = float(config.get("temporal_expansion_max_extension_s", 4.0))

    def supports(event, row):
        objects = {obj["id"]: obj for obj in row["objects"]}
        person = objects.get(event["persons"][0]["person_id"])
        vehicle = objects.get(event["vehicle"]["vehicle_id"])
        return bool(person and vehicle and _vehicle_gap(person["bbox"], vehicle["bbox"]) <= margin)

    for event in events:
        expanded = []
        for span in event["spans"]:
            start_index = min(range(len(ordered)), key=lambda i: abs(ordered[i]["timestamp_s"] - span["start_s"]))
            end_index = min(range(len(ordered)), key=lambda i: abs(ordered[i]["timestamp_s"] - span["end_s"]))
            start, end = span["start_s"], span["end_s"]
            last_support = start
            for index in range(start_index - 1, -1, -1):
                timestamp = ordered[index]["timestamp_s"]
                if start - timestamp > maximum_extension:
                    break
                if supports(event, ordered[index]):
                    last_support = timestamp
                    start = timestamp
                elif last_support - timestamp <= maximum_gap:
                    start = timestamp
                else:
                    break
            last_support = end
            for index in range(end_index + 1, len(ordered)):
                timestamp = ordered[index]["timestamp_s"]
                if timestamp - span["end_s"] > maximum_extension:
                    break
                if supports(event, ordered[index]):
                    last_support = timestamp
                    end = min(duration, ordered[index + 1]["timestamp_s"] if index + 1 < len(ordered) else duration)
                elif timestamp - last_support <= maximum_gap:
                    end = min(duration, ordered[index + 1]["timestamp_s"] if index + 1 < len(ordered) else duration)
                else:
                    break
            expanded.append({"start_s": max(0, start), "end_s": min(duration, max(end, start + 1e-6))})
        event["spans"] = expanded
    return events


def apply_track_consistency_rules(events, rows, config):
    """Clip events to vehicle visibility and apply guarded action corrections."""
    if not events or not rows or not config.get("track_consistency_rules", False):
        return events
    by_id = {}
    for row in rows:
        for obj in row["objects"]:
            by_id.setdefault(obj["id"], []).append((row["timestamp_s"], obj))
    disappear_window = float(config.get("door_enter_disappear_window_s", 1.5))
    disappearance_enter = bool(config.get("disappearance_enter_cue", False))
    disappearance_enter_min_approach = float(
        config.get("disappearance_enter_min_approach", .08))
    disappearance_enter_max_final_gap = float(
        config.get("disappearance_enter_max_final_gap", .1))
    exit_change = float(config.get("enter_exit_min_gap_change", .08))
    longitudinal_change = float(config.get("enter_exit_min_longitudinal_change", .45))
    vehicle_edge = float(config.get("enter_exit_vehicle_edge_fraction", .18))
    corrected = []
    for event in events:
        event = dict(event)
        person_id = event["persons"][0]["person_id"]
        vehicle_id = event["vehicle"]["vehicle_id"]
        vehicle_obs = by_id.get(vehicle_id, [])
        person_obs = by_id.get(person_id, [])
        if not vehicle_obs:
            continue
        first_vehicle, last_vehicle = vehicle_obs[0][0], vehicle_obs[-1][0]
        spans = []
        for span in event["spans"]:
            start = max(span["start_s"], first_vehicle)
            end = min(span["end_s"], last_vehicle)
            if end > start:
                spans.append({"start_s": start, "end_s": end})
        if not spans:
            continue
        event["spans"] = spans
        start = min(span["start_s"] for span in spans)
        end = max(span["end_s"] for span in spans)
        paired = []
        for timestamp, person in person_obs:
            if start <= timestamp <= end:
                vehicle = min(vehicle_obs, key=lambda item: abs(item[0] - timestamp))[1]
                person_box, vehicle_box = person["bbox"], vehicle["bbox"]
                px = (person_box[0] + person_box[2]) / 2
                vx1, _, vx2, _ = vehicle_box
                longitudinal = (px - vx1) / max(vx2 - vx1, 1e-6)
                paired.append((timestamp, _vehicle_gap(person_box, vehicle_box), longitudinal))
        if event["type"] == "door_operation" and person_obs:
            last_person = person_obs[-1][0]
            vehicle_continues = last_vehicle >= last_person + min(.3, disappear_window)
            disappears_near_end = end - .5 <= last_person <= end + disappear_window
            near_vehicle = bool(paired and min(item[1] for item in paired[-3:]) <= .1)
            if vehicle_continues and disappears_near_end and near_vehicle:
                event["type"] = "enter"
        if (disappearance_enter and event["type"] in ("other_interaction", "door_operation")
                and person_obs and len(paired) >= 3):
            # A person who steadily approaches a still-visible vehicle and whose
            # track terminates at its boundary supplies entry evidence by the
            # disappearance itself.  No later person track is required: entering
            # the cabin is precisely the case in which the original track ends.
            last_person = person_obs[-1][0]
            vehicle_continues = last_vehicle >= last_person + min(.3, disappear_window)
            disappears_near_end = end - disappear_window <= last_person <= end + disappear_window
            edge = min(3, len(paired) // 2)
            initial_gap = sum(item[1] for item in paired[:edge]) / edge
            final_gap = sum(item[1] for item in paired[-edge:]) / edge
            approaches_vehicle = initial_gap - final_gap >= disappearance_enter_min_approach
            ends_at_vehicle = final_gap <= disappearance_enter_max_final_gap
            if vehicle_continues and disappears_near_end and approaches_vehicle and ends_at_vehicle:
                event["type"] = "enter"
        if event["type"] == "enter" and len(paired) >= 4:
            edge = min(3, len(paired) // 2)
            initial = sum(item[1] for item in paired[:edge]) / edge
            final = sum(item[1] for item in paired[-edge:]) / edge
            initial_x = sum(item[2] for item in paired[:edge]) / edge
            final_x = sum(item[2] for item in paired[-edge:]) / edge
            terminal_x = paired[-1][2]
            walks_to_edge = (abs(terminal_x - initial_x) >= longitudinal_change and
                             vehicle_edge <= initial_x <= 1 - vehicle_edge and
                             (terminal_x <= vehicle_edge or terminal_x >= 1 - vehicle_edge))
            if initial <= .1 and (final - initial >= exit_change or walks_to_edge):
                event["type"] = "exit"
        corrected.append(event)
    return corrected


def normalize_door_operations(events):
    """Expose every remaining door operation as a load/unload interaction.

    Directional consistency rules run first, so a door operation with strong
    entry evidence may still become ``enter``.  Any event that remains a plain
    door operation is mapped to the requested public action taxonomy.
    """
    return [{**event, "type": "load_unload"} if event.get("type") == "door_operation"
            else event for event in events]
