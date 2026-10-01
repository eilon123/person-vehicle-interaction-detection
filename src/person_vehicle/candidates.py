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


def propose(rows, duration, config):
    encounters = {}
    last_vehicle = {}
    for row in rows:
        people = [obj for obj in row["objects"] if obj["kind"] == "person"]
        vehicles = [obj for obj in row["objects"] if obj["kind"] != "person"]
        for person in people:
            # A person can be associated with at most one vehicle at a given
            # frame. If several vehicle tracks are nearby, choose the closest
            # one (normalized by vehicle size) to avoid duplicate candidates.
            nearby = [vehicle for vehicle in vehicles if near_objects(person, vehicle, config)]
            if nearby and config.get("exclusive_person_vehicle", True):
                ranked = sorted(nearby, key=lambda item: (
                    _normalized_gap(person["bbox"], item["bbox"]), item["id"]))
                vehicle = ranked[0]
                if config.get("vehicle_assignment_hysteresis", False):
                    previous = last_vehicle.get(person["id"])
                    previous_obj = next((item for item in nearby if previous and item["id"] == previous[0]), None)
                    max_gap = float(config.get("vehicle_assignment_max_gap_s", 1.0))
                    switch_margin = float(config.get("vehicle_assignment_switch_margin", .05))
                    if previous_obj and row["timestamp_s"] - previous[1] <= max_gap:
                        old_gap = _normalized_gap(person["bbox"], previous_obj["bbox"])
                        new_gap = _normalized_gap(person["bbox"], vehicle["bbox"])
                        if old_gap <= new_gap + switch_margin:
                            vehicle = previous_obj
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
