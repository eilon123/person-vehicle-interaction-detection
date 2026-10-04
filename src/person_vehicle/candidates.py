import math


def near(person, vehicle, margin):
    px1, py1, px2, py2 = person
    vx1, vy1, vx2, vy2 = vehicle
    dx = max(vx1 - px2, px1 - vx2, 0)
    dy = max(vy1 - py2, py1 - vy2, 0)
    return math.hypot(dx, dy) <= margin * max(vx2 - vx1, vy2 - vy1)


def _pixel_gap(person, vehicle):
    px1, py1, px2, py2 = person
    vx1, vy1, vx2, vy2 = vehicle
    dx = max(vx1 - px2, px1 - vx2, 0)
    dy = max(vy1 - py2, py1 - vy2, 0)
    return math.hypot(dx, dy)


def near_objects(person, vehicle, config):
    """Measure proximity in head widths when a reliable head is visible."""
    head = person.get("head_bbox")
    if config.get("head_scaled_candidate_distance", False) and head:
        head_width = max(1.0, float(head[2]) - float(head[0]))
        maximum = float(config.get("candidate_max_head_widths", 3.0))
        return _pixel_gap(person["bbox"], vehicle["bbox"]) / head_width <= maximum
    return near(person["bbox"], vehicle["bbox"], config["near_margin"])


def _normalized_gap(person, vehicle):
    """Return the box-to-box gap normalized by vehicle size."""
    px1, py1, px2, py2 = person
    vx1, vy1, vx2, vy2 = vehicle
    dx = max(vx1 - px2, px1 - vx2, 0)
    dy = max(vy1 - py2, py1 - vy2, 0)
    scale = max(vx2 - vx1, vy2 - vy1, 1e-6)
    return math.hypot(dx, dy) / scale


def _normalized_center_distance(person, vehicle):
    """Measure center distance in vehicle-size units.

    Box-to-box distance is zero when a person touches or overlaps multiple
    vehicles.  The center term breaks that ambiguity in favour of the vehicle
    whose physical footprint best explains the person's location.
    """
    px1, py1, px2, py2 = person
    vx1, vy1, vx2, vy2 = vehicle
    px, py = (px1 + px2) / 2, (py1 + py2) / 2
    vx, vy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
    scale = max(vx2 - vx1, vy2 - vy1, 1e-6)
    return math.hypot(px - vx, py - vy) / scale


def _vehicle_rank(person, vehicle, config):
    """Rank nearby vehicles using boundary distance and spatial ownership."""
    gap = _normalized_gap(person["bbox"], vehicle["bbox"])
    center_weight = float(config.get("vehicle_assignment_center_weight", .2))
    center = _normalized_center_distance(person["bbox"], vehicle["bbox"])
    confidence_weight = float(config.get("vehicle_assignment_confidence_weight", .02))
    confidence = float(vehicle.get("confidence", 0.0))
    rank = gap + center_weight * center - confidence_weight * confidence
    if (config.get("horizontal_body_vehicle_assignment", False) and
            person.get("_ambiguous_vehicle_choice", False)):
        px1, py1, px2, py2 = person["bbox"]
        vx1, vy1, vx2, vy2 = vehicle["bbox"]
        person_width, person_height = max(px2 - px1, 1e-6), max(py2 - py1, 0.0)
        good_body = (person_height / person_width >= float(
            config.get("horizontal_body_min_aspect_ratio", 1.5)) and
            float(person.get("confidence", 0.0)) >= float(
                config.get("horizontal_body_min_confidence", .2)))
        if good_body:
            body_y = py1 + float(config.get("horizontal_body_anchor_fraction", .6)) * person_height
            vertical_margin = float(config.get("horizontal_body_vertical_margin", .08)) * max(vy2 - vy1, 1.0)
            crosses_body_axis = vy1 - vertical_margin <= body_y <= vy2 + vertical_margin
            body_x = (px1 + px2) / 2
            horizontal_gap = max(vx1 - body_x, body_x - vx2, 0.0) / max(vx2 - vx1, 1.0)
            # Crossing the body's horizontal axis is the primary perspective
            # cue; horizontal boundary distance resolves ties on that axis.
            miss_penalty = float(config.get("horizontal_body_miss_penalty", 2.0))
            rank = (0.0 if crosses_body_axis else miss_penalty) + horizontal_gap + .01 * rank
    return rank


def _short_track_vehicle_transition(observations, rows_by_time, config):
    """Keep a short, person-shaped track when it touches a vehicle boundary.

    The ordinary duration filter removes noisy one-off detections.  A real
    enter/exit can also be brief because the person is immediately occluded by
    the vehicle.  This exception requires multiple observations, a reasonably
    upright box, and vehicle proximity at the beginning or end of the track.
    Wide, car-part-shaped detections therefore remain filtered.
    """
    minimum = int(config.get("short_track_vehicle_min_detections", 2))
    if len(observations) < minimum:
        return False
    ratios = []
    for _, person in observations:
        x1, y1, x2, y2 = person["bbox"]
        ratios.append(max(0.0, y2 - y1) / max(1.0, x2 - x1))
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    if median_ratio < float(config.get("short_track_vehicle_min_aspect_ratio", .85)):
        return False
    endpoint_count = max(1, int(config.get("short_track_vehicle_endpoint_frames", 2)))
    endpoints = observations[:endpoint_count] + observations[-endpoint_count:]
    for timestamp, person in endpoints:
        vehicles = [obj for obj in rows_by_time[timestamp]["objects"] if obj["kind"] != "person"]
        if any(near_objects(person, vehicle, config) for vehicle in vehicles):
            return True
    return False


def propose(rows, duration, config):
    encounters = {}
    last_vehicle = {}
    person_observations = {}
    rows_by_time = {row["timestamp_s"]: row for row in rows}
    for row in rows:
        for obj in row["objects"]:
            if obj["kind"] == "person":
                person_observations.setdefault(obj["id"], []).append((row["timestamp_s"], obj))
    minimum_detections = int(config.get("person_track_min_detections", 0))
    minimum_duration = float(config.get("person_track_min_duration_s", 0.0))
    valid_people = {
        identity for identity, observations in person_observations.items()
        if (len(observations) >= minimum_detections and
            ((observations[-1][0] - observations[0][0]) if len(observations) > 1 else 0.0) >= minimum_duration)
        or (config.get("short_track_vehicle_exception", False) and
            _short_track_vehicle_transition(observations, rows_by_time, config))
    }
    for row in rows:
        people = [obj for obj in row["objects"]
                  if obj["kind"] == "person" and obj["id"] in valid_people]
        vehicles = [obj for obj in row["objects"] if obj["kind"] != "person"]
        for person in people:
            # A person can be associated with at most one vehicle at a given
            # frame. If several vehicle tracks are nearby, choose the closest
            # one (normalized by vehicle size) to avoid duplicate candidates.
            nearby = [vehicle for vehicle in vehicles if near_objects(person, vehicle, config)]
            if nearby and config.get("exclusive_person_vehicle", True):
                ranked_person = ({**person, "_ambiguous_vehicle_choice": True}
                                 if len(nearby) >= 2 else person)
                ranked = sorted(nearby, key=lambda item: (
                    _vehicle_rank(ranked_person, item, config), item["id"]))
                vehicle = ranked[0]
                if config.get("vehicle_assignment_hysteresis", False):
                    previous = last_vehicle.get(person["id"])
                    previous_obj = next((item for item in nearby if previous and item["id"] == previous[0]), None)
                    max_gap = float(config.get("vehicle_assignment_max_gap_s", 1.0))
                    switch_margin = float(config.get("vehicle_assignment_switch_margin", .05))
                    if previous and row["timestamp_s"] - previous[1] <= max_gap:
                        if previous_obj:
                            old_gap = _vehicle_rank(ranked_person, previous_obj, config)
                            new_gap = _vehicle_rank(ranked_person, vehicle, config)
                            if old_gap <= new_gap + switch_margin:
                                vehicle = previous_obj
                        elif config.get("vehicle_assignment_hold_when_missing", False):
                            # Do not transfer ownership to another nearby car
                            # because the established vehicle missed one frame
                            # or briefly failed the proximity test.
                            continue
                    last_vehicle[person["id"]] = (vehicle["id"], row["timestamp_s"])
                key = (person["id"], vehicle["id"], row["scene"])
                encounters.setdefault(key, []).append(row["timestamp_s"])
            else:
                for vehicle in nearby:
                    key = (person["id"], vehicle["id"], row["scene"])
                    encounters.setdefault(key, []).append(row["timestamp_s"])
    windows = []
    initial_probe = config.get("initial_vehicle_exit_probe_s", 0)
    if initial_probe and rows:
        first_scene = rows[0]["scene"]
        early = [row for row in rows if row["scene"] == first_scene
                 and row["timestamp_s"] < initial_probe]
        # When YOLO has no person track at the clip boundary, allow the VLM to
        # inspect the dominant vehicle for a visible person emerging from it.
        if early and not any(obj["kind"] == "person" for row in early for obj in row["objects"]):
            vehicle_area = {}
            for row in early:
                for obj in row["objects"]:
                    if obj["kind"] == "person":
                        continue
                    x1, y1, x2, y2 = obj["bbox"]
                    vehicle_area[obj["id"]] = vehicle_area.get(obj["id"], 0) + max(0, x2 - x1) * max(0, y2 - y1)
            if vehicle_area:
                vehicle_id = max(sorted(vehicle_area), key=lambda key: vehicle_area[key])
                windows.append({"person_id": f"untracked_{vehicle_id}", "vehicle_id": vehicle_id,
                                "scene": first_scene, "start_s": 0,
                                "end_s": min(duration, initial_probe + min(config["context_s"], 0.5))})
    for (person_id, vehicle_id, scene), times in sorted(encounters.items()):
        groups = [[times[0]]]
        for timestamp in times[1:]:
            if timestamp - groups[-1][-1] > config["candidate_gap_s"]:
                groups.append([])
            groups[-1].append(timestamp)
        for group in groups:
            start = max(0, group[0] - config["context_s"])
            end = min(duration, group[-1] + config["context_s"])
            while start < end:
                stop = min(end, start + config["window_s"])
                windows.append({"person_id": person_id, "vehicle_id": vehicle_id,
                                "scene": scene, "start_s": start, "end_s": stop})
                if stop == end:
                    break
                start = stop - config["window_overlap_s"]
    for i, window in enumerate(windows):
        window["candidate_id"] = f"c{i + 1:04d}"
    return windows
