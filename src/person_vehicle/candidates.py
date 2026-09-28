import math


def near(person, vehicle, margin):
    px1, py1, px2, py2 = person
    vx1, vy1, vx2, vy2 = vehicle
    dx = max(vx1 - px2, px1 - vx2, 0)
    dy = max(vy1 - py2, py1 - vy2, 0)
    return math.hypot(dx, dy) <= margin * max(vx2 - vx1, vy2 - vy1)


def propose(rows, duration, config):
    encounters = {}
    for row in rows:
        people = [obj for obj in row["objects"] if obj["kind"] == "person"]
        vehicles = [obj for obj in row["objects"] if obj["kind"] != "person"]
        for person in people:
            for vehicle in vehicles:
                if near(person["bbox"], vehicle["bbox"], config["near_margin"]):
                    key = (person["id"], vehicle["id"], row["scene"])
                    encounters.setdefault(key, []).append(row["timestamp_s"])
    windows = []
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
