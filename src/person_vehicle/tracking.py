import json
import os
from pathlib import Path

import cv2
import numpy as np

from .io import fingerprint, read_json, sha256, write_json
from .video import frames


def _iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1e-6, first_area + second_area - intersection)


def _deduplicate(detections, threshold=0.5):
    """Class-aware greedy NMS over [x1,y1,x2,y2,confidence,class]."""
    retained = []
    for detection in sorted(detections, key=lambda row: row[4], reverse=True):
        if any(int(detection[5]) == int(other[5]) and _iou(detection[:4], other[:4]) >= threshold
               for other in retained):
            continue
        retained.append(detection)
    return retained


def _contains_point(box, point, margin=0.0):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return (x1 - margin * width <= point[0] <= x2 + margin * width and
            y1 - margin * height <= point[1] <= y2 + margin * height)


def _head_person_detections(image, head_model, people, vehicles, config):
    """Convert uncovered head detections into conservative partial-person boxes.

    A head covered by (or very close to) an existing person box is discarded.
    Otherwise the inferred body extent is fed to the same tracker as a low
    confidence person observation, allowing a head-only track to become a normal
    body track as the person emerges from an occlusion.
    """
    result = head_model.predict(
        image, conf=float(config.get("head_detection_confidence", .25)),
        imgsz=int(config.get("head_detection_image_size", 1280)),
        device=config["device"], verbose=False)[0]
    if result.boxes is None:
        return [], [], []
    height, width = image.shape[:2]
    detections, head_regions, observed_heads = [], [], []
    proximity = float(config.get("head_person_proximity_margin", .12))
    vehicle_margin = float(config.get("head_vehicle_proximity_margin", .2))
    maximum_vehicle_ratio = float(config.get("head_max_vehicle_width_ratio", 0))
    body_width = float(config.get("head_body_width_factor", 3.2))
    body_height = float(config.get("head_body_height_factor", 6.5))
    for head, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                result.boxes.conf.cpu().tolist()):
        hx1, hy1, hx2, hy2 = head
        center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
        observed_heads.append([hx1, hy1, hx2, hy2])
        if any(_contains_point(person, center, proximity) for person in people):
            continue
        head_w, head_h = max(1, hx2 - hx1), max(1, hy2 - hy1)
        if maximum_vehicle_ratio and any(
                _contains_point(vehicle, center, vehicle_margin) and
                head_w > maximum_vehicle_ratio * max(1, vehicle[2] - vehicle[0])
                for vehicle in vehicles):
            continue
        half_width = body_width * head_w / 2
        inferred = [max(0, center[0] - half_width), max(0, hy1 - .2 * head_h),
                    min(width, center[0] + half_width), min(height, hy1 + body_height * head_h)]
        detection = [*inferred,
                     float(confidence) * float(config.get("head_person_confidence_scale", .65)), 0]
        detections.append(detection)
        head_regions.append((detection, [hx1, hy1, hx2, hy2]))
    return detections, head_regions, observed_heads


def _motion_mask(previous, current):
    """Camera-compensated frame difference used only to choose extra detector crops."""
    if previous is None:
        return np.zeros_like(current)
    prior = previous
    points = cv2.goodFeaturesToTrack(previous, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if points is not None:
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        if moved is not None and status is not None:
            source = points[status.ravel() == 1]
            target = moved[status.ravel() == 1]
            if len(source) >= 6:
                transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
                if transform is not None:
                    prior = cv2.warpAffine(previous, transform, (current.shape[1], current.shape[0]))
    difference = cv2.absdiff(current, prior)
    mask = cv2.threshold(difference, 22, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _vehicle_motion_crops(image, vehicles, motion, config):
    height, width = image.shape[:2]
    margin = float(config.get("motion_crop_margin", 0.6))
    minimum = float(config.get("motion_crop_min_fraction", 0.008))
    ranked = []
    for box in vehicles:
        x1, y1, x2, y2 = box
        size = max(x2 - x1, y2 - y1)
        left, top = max(0, int(x1 - margin * size)), max(0, int(y1 - margin * size))
        right, bottom = min(width, int(x2 + margin * size)), min(height, int(y2 + margin * size))
        if right - left < 48 or bottom - top < 48:
            continue
        region = motion[top:bottom, left:right]
        fraction = float(np.count_nonzero(region)) / max(1, region.size)
        if fraction >= minimum:
            ranked.append((fraction, [left, top, right, bottom]))
    crops = []
    for _, bounds in sorted(ranked, reverse=True):
        if any(_iou(bounds, prior) > 0.65 for prior in crops):
            continue
        crops.append(bounds)
        if len(crops) >= int(config.get("motion_crop_max_regions", 4)):
            break
    return crops


def _correlation_people(previous_gray, gray, states, observed, config):
    """Bridge short person-detector gaps with pyramidal LK correlation."""
    if previous_gray is None or not config.get("correlation_bridge_enabled", False):
        return []
    observed_ids = {obj["id"] for obj in observed}
    maximum = int(config.get("correlation_max_missing_frames", 12))
    minimum = int(config.get("correlation_min_points", 5))
    propagated = []
    for identity, state in list(states.items()):
        if identity in observed_ids or state["missing"] >= maximum:
            continue
        x1, y1, x2, y2 = state.get("correlation_bbox", state["bbox"])
        mask = np.zeros_like(previous_gray)
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(mask.shape[1], int(x2)), min(mask.shape[0], int(y2))
        if right - left < 8 or bottom - top < 8:
            continue
        mask[top:bottom, left:right] = 255
        points = cv2.goodFeaturesToTrack(previous_gray, mask=mask, maxCorners=40,
                                         qualityLevel=.01, minDistance=4)
        if points is None or len(points) < minimum:
            continue
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None)
        if moved is None or status is None:
            continue
        source = points[status.ravel() == 1].reshape(-1, 2)
        target = moved[status.ravel() == 1].reshape(-1, 2)
        if len(source) < minimum:
            continue
        shifts = target - source
        delta = np.median(shifts, axis=0)
        reliable = np.linalg.norm(shifts - delta, axis=1) <= float(config.get("correlation_max_residual_px", 8.0))
        if int(reliable.sum()) < minimum:
            continue
        dx, dy = np.median(shifts[reliable], axis=0)
        correlation_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        bx1, by1, bx2, by2 = state["bbox"]
        box = [bx1 + dx, by1 + dy, bx2 + dx, by2 + dy]
        box = [max(0, min(float(value), gray.shape[1] if i % 2 == 0 else gray.shape[0]))
               for i, value in enumerate(box)]
        correlation_box = [max(0, min(float(value), gray.shape[1] if i % 2 == 0 else gray.shape[0]))
                           for i, value in enumerate(correlation_box)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(obj["kind"] == "person" and _iou(box, obj["bbox"]) > .45 for obj in observed):
            continue
        # A head-only track must not jump to a newly observed head. Let the
        # detector/tracker resolve that identity instead of propagating both.
        if state.get("source") == "head" and any(
                obj.get("head_bbox") and _iou(correlation_box, obj["head_bbox"]) > .25
                for obj in observed):
            continue
        if state.get("source") == "head" and any(
                obj["kind"] == "person" and obj.get("source") == "detector" and
                _contains_point(obj["bbox"], ((correlation_box[0] + correlation_box[2]) / 2,
                                               (correlation_box[1] + correlation_box[3]) / 2), .12)
                for obj in observed):
            continue
        confidence = max(.01, state["confidence"] * float(config.get("correlation_confidence_decay", .85)))
        head_source = state.get("source") == "head"
        propagated.append({"id": identity, "kind": "person",
                           "bbox": [round(float(value), 2) for value in
                                    (correlation_box if head_source else box)],
                           "confidence": round(confidence, 4), "observed": False,
                           "source": "head_correlation" if head_source else "correlation",
                           **({"tracking_bbox": [round(float(value), 2) for value in box]}
                              if head_source else {}),
                           **({"head_bbox": [round(float(value), 2) for value in correlation_box]}
                              if head_source else {})})
    return propagated


def _stitch_person_ids(objects, states, aliases, frame_index, config):
    """Attach a new tracker ID to a recent head/person identity when geometry is strong."""
    maximum_gap = int(config.get("identity_stitch_max_gap_frames", 15))
    minimum_iou = float(config.get("identity_stitch_min_iou", .45))
    maximum_center = float(config.get("identity_stitch_max_center_distance", .2))

    def match_score(obj, identity, state):
        if frame_index - state["frame"] > maximum_gap:
            return None
        box = obj.get("tracking_bbox", obj["bbox"])
        prior = state["bbox"]
        overlap = _iou(box, prior)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        px, py = (prior[0] + prior[2]) / 2, (prior[1] + prior[3]) / 2
        scale = max(box[2] - box[0], box[3] - box[1], prior[2] - prior[0], prior[3] - prior[1], 1)
        distance = ((cx - px) ** 2 + (cy - py) ** 2) ** .5 / scale
        head_box = obj.get("head_bbox") or state.get("head_bbox")
        head_inside = False
        if head_box:
            head_center = ((head_box[0] + head_box[2]) / 2, (head_box[1] + head_box[3]) / 2)
            body = prior if obj.get("head_bbox") else box
            head_inside = _contains_point(body, head_center, .12)
        if overlap < minimum_iou and distance > maximum_center and not head_inside:
            return None
        return (-int(head_inside), -overlap, distance, identity)

    stitched = []
    # Full-body detections establish identities before head-only observations.
    ordered = sorted(objects, key=lambda obj: (obj["kind"] != "person",
                                                obj.get("source") in {"head", "head_correlation"}))
    claimed = set()
    for obj in ordered:
        if obj["kind"] != "person":
            stitched.append(obj)
            continue
        raw_id = obj["id"]
        logical_id = aliases.get(raw_id)
        if logical_id is None:
            matches = [(score, identity) for identity, state in states.items()
                       if (score := match_score(obj, identity, state)) is not None]
            logical_id = min(matches)[1] if matches else raw_id
            aliases[raw_id] = logical_id
        obj = dict(obj)
        obj["raw_id"] = raw_id
        obj["id"] = logical_id
        claimed.add(logical_id)
        prior = next((item for item in stitched if item.get("id") == logical_id), None)
        if prior is not None:
            # Prefer a real body observation to a head/correlation duplicate.
            prior_rank = (prior.get("source") != "detector", -prior.get("confidence", 0))
            current_rank = (obj.get("source") != "detector", -obj.get("confidence", 0))
            if current_rank < prior_rank:
                stitched[stitched.index(prior)] = obj
        else:
            stitched.append(obj)
        states[logical_id] = {"bbox": obj.get("tracking_bbox", obj["bbox"]),
                              "head_bbox": obj.get("head_bbox"), "frame": frame_index}
    for identity in list(states):
        if frame_index - states[identity]["frame"] > maximum_gap:
            del states[identity]
    return stitched


def _track_with_motion_crops(path, model, metadata, config, track_config, head_model=None):
    import torch
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml
    from ultralytics.utils.torch_utils import select_device

    tracker_settings = IterableSimpleNamespace(**YAML.load(check_yaml(config["tracker"])))
    tracker_settings.device = select_device(config["device"], verbose=False)
    tracker = BOTSORT(tracker_settings)
    rows, previous_gray, scene = [], None, 0
    correlation_states = {}
    identity_states, person_aliases = {}, {}
    previous_scene_thumb = None
    for index, timestamp, frame in frames(path):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scene_thumb = cv2.resize(gray, (64, 64))
        if previous_scene_thumb is not None and float(np.abs(scene_thumb.astype(float) - previous_scene_thumb).mean()) > 55:
            scene += 1
            tracker.reset()
            previous_gray = None
            correlation_states = {}
            identity_states, person_aliases = {}, {}
        motion = _motion_mask(previous_gray, gray)

        global_result = model.predict(image, classes=[0, 1, 2, 3, 5, 7],
                                      conf=min(config["detection_confidence"],
                                               config.get("motion_crop_person_confidence", 0.1)),
                                      imgsz=track_config["image_size"], device=config["device"],
                                      agnostic_nms=True, verbose=False)[0]
        detections = []
        head_detections, head_regions, observed_heads = [], [], []
        vehicles = []
        if global_result.boxes is not None:
            for box, category, confidence in zip(global_result.boxes.xyxy.cpu().tolist(),
                    global_result.boxes.cls.cpu().tolist(), global_result.boxes.conf.cpu().tolist()):
                threshold = (config.get("motion_crop_person_confidence", 0.1)
                             if int(category) == 0 else config["detection_confidence"])
                if confidence < threshold:
                    continue
                detections.append([*box, confidence, category])
                if int(category) != 0:
                    vehicles.append(box)

        for left, top, right, bottom in _vehicle_motion_crops(image, vehicles, motion, config):
            crop = image[top:bottom, left:right]
            crop_result = model.predict(crop, classes=[0],
                                        conf=config.get("motion_crop_person_confidence", 0.1),
                                        imgsz=config.get("motion_crop_image_size", 960),
                                        device=config["device"], verbose=False)[0]
            if crop_result.boxes is None:
                continue
            for box, confidence in zip(crop_result.boxes.xyxy.cpu().tolist(),
                                       crop_result.boxes.conf.cpu().tolist()):
                detections.append([box[0] + left, box[1] + top, box[2] + left, box[3] + top,
                                   confidence, 0])
        if head_model is not None:
            # Suppress heads against both global and motion-crop person boxes.
            detected_people = [row[:4] for row in detections if int(row[5]) == 0]
            head_detections, head_regions, observed_heads = _head_person_detections(
                image, head_model, detected_people, vehicles, config)
            detections.extend(head_detections)
        detections = _deduplicate(detections, config.get("motion_crop_nms_iou", 0.5))
        tensor = (torch.tensor(detections, dtype=torch.float32, device=tracker_settings.device)
                  if detections else torch.empty((0, 6), dtype=torch.float32, device=tracker_settings.device))
        boxes = Boxes(tensor, image.shape[:2]).cpu().numpy()
        tracked = tracker.update(boxes, image)
        objects = []
        for row in tracked:
            x1, y1, x2, y2, identity, confidence, category = row[:7]
            kind = global_result.names[int(category)]
            prefix = "p" if kind == "person" else "v"
            source = ("head" if kind == "person" and
                      any(_iou((x1, y1, x2, y2), detection[:4]) >= .35
                          for detection in head_detections) else "detector")
            obj = {"id": f"{prefix}{scene:02d}_{int(identity):03d}", "kind": kind,
                   "bbox": [round(float(x), 2) for x in (x1, y1, x2, y2)],
                   "confidence": round(float(confidence), 4), "observed": True,
                   "source": source}
            if source == "head" and head_regions:
                _, head_box = max(head_regions, key=lambda item: _iou((x1, y1, x2, y2), item[0][:4]))
                obj["head_bbox"] = [round(float(value), 2) for value in head_box]
                obj["tracking_bbox"] = obj["bbox"]
                obj["bbox"] = obj["head_bbox"]
            elif kind == "person" and observed_heads:
                contained = [head for head in observed_heads if _contains_point(
                    (x1, y1, x2, y2), ((head[0] + head[2]) / 2, (head[1] + head[3]) / 2), .05)]
                if contained:
                    head_box = min(contained, key=lambda head: (
                        (head[1] + head[3]) / 2, -(head[2] - head[0])))
                    obj["head_bbox"] = [round(float(value), 2) for value in head_box]
            objects.append(obj)
        objects = _stitch_person_ids(objects, identity_states, person_aliases, index, config)
        objects.extend(_correlation_people(previous_gray, gray, correlation_states, objects, config))
        live_ids = set()
        for obj in objects:
            if obj["kind"] != "person":
                continue
            live_ids.add(obj["id"])
            prior_missing = correlation_states.get(obj["id"], {}).get("missing", -1)
            correlation_states[obj["id"]] = {
                "bbox": obj.get("tracking_bbox", obj["bbox"]), "confidence": obj["confidence"],
                "correlation_bbox": obj.get("head_bbox", obj["bbox"]),
                "source": ("head" if obj.get("source") in {"head", "head_correlation"} else "person"),
                "missing": 0 if obj.get("observed", True) else prior_missing + 1,
            }
        for identity, state in correlation_states.items():
            if identity not in live_ids:
                state["missing"] += 1
        correlation_states = {
            identity: state for identity, state in correlation_states.items()
            if identity in live_ids or state["missing"] < int(config.get("correlation_max_missing_frames", 12))
        }
        rows.append({"frame_index": index, "timestamp_s": timestamp, "scene": scene,
                     "objects": objects})
        previous_gray, previous_scene_thumb = gray, scene_thumb.astype(float)
        if index % 100 == 0:
            print(f"{Path(path).stem}: tracking {index}/{metadata['frame_count']}", flush=True)
    return rows


def track(path, output, metadata, config, resume=True):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".meta.json")
    model_path = Path(config["detector"])
    if not model_path.exists():
        raise FileNotFoundError(f"Missing detector {model_path}; run the download-assets command")
    track_config = {key: config[key] for key in ["detector", "device", "image_size", "detection_confidence", "tracker", "seed"]}
    track_config["person_detection_confidence"] = config.get(
        "person_detection_confidence", config["detection_confidence"])
    track_config["person_detector"] = config.get("person_detector")
    track_config.update({key: config.get(key) for key in ["motion_crop_enabled", "motion_crop_person_confidence",
        "motion_crop_image_size", "motion_crop_margin", "motion_crop_min_fraction", "motion_crop_max_regions",
        "motion_crop_nms_iou", "correlation_bridge_enabled", "correlation_max_missing_frames",
        "correlation_min_points", "correlation_max_residual_px", "correlation_confidence_decay",
        "head_detection_enabled", "head_detector", "head_detection_confidence",
        "head_detection_image_size", "head_person_proximity_margin", "head_body_width_factor",
        "head_body_height_factor", "head_person_confidence_scale", "head_vehicle_proximity_margin",
        "head_max_vehicle_width_ratio", "identity_stitch_max_gap_frames", "identity_stitch_min_iou",
        "identity_stitch_max_center_distance"]})
    track_config["image_size"] = min(config["image_size"], max(640, metadata["width"], metadata["height"]))
    signature = fingerprint({"source": metadata["source_sha256"],
                             "model": sha256(model_path),
                             "person_model": (sha256(Path(config["person_detector"]))
                                              if config.get("person_detector") else None),
                             "head_model": (sha256(Path(config["head_detector"]))
                                            if config.get("head_detection_enabled") else None),
                             "config": track_config, "tracker_version": 10})
    if resume and output.exists() and sidecar.exists() and read_json(sidecar)["signature"] == signature:
        return [json.loads(line) for line in output.read_text().splitlines()]
    settings_dir = Path(".cache/ultralytics").resolve()
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO
    model = YOLO(str(model_path))  # A fresh model and tracker per clip.
    person_model = YOLO(str(Path(config["person_detector"]))) if config.get("person_detector") else None
    head_model = None
    if config.get("head_detection_enabled", False):
        head_path = Path(config["head_detector"])
        if not head_path.exists():
            raise FileNotFoundError(f"Missing head detector {head_path}")
        head_model = YOLO(str(head_path))
    if config.get("motion_crop_enabled", False):
        rows = _track_with_motion_crops(path, model, metadata, config, track_config, head_model)
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        temporary.replace(output)
        write_json(sidecar, {"signature": signature, "detector_sha256": sha256(model_path)})
        return rows
    rows = []
    previous = None
    scene = 0
    for index, timestamp, frame in frames(path):
        image = frame.to_ndarray(format="bgr24")
        thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 64))
        if previous is not None and float(np.abs(thumb.astype(float) - previous).mean()) > 55:
            scene += 1
            if model.predictor is not None and hasattr(model.predictor, "trackers"):
                for tracker in model.predictor.trackers:
                    tracker.reset()
            if person_model is not None and person_model.predictor is not None and hasattr(person_model.predictor, "trackers"):
                for tracker in person_model.predictor.trackers:
                    tracker.reset()
        previous = thumb.astype(float)
        classes = [1, 2, 3, 5, 7] if person_model is not None else [0, 1, 2, 3, 5, 7]
        result = model.track(image, persist=True, tracker=config["tracker"], classes=classes,
                             conf=config["detection_confidence"], imgsz=track_config["image_size"],
                             device=config["device"], agnostic_nms=True, verbose=False)[0]
        results = [result]
        if person_model is not None:
            results.append(person_model.track(
                image, persist=True, tracker=config["tracker"], classes=[0],
                conf=config.get("person_detection_confidence", config["detection_confidence"]),
                imgsz=track_config["image_size"], device=config["device"],
                agnostic_nms=True, verbose=False)[0])
        objects = []
        for tracked_result in results:
            if tracked_result.boxes is None or tracked_result.boxes.id is None:
                continue
            for box, category, confidence, identity in zip(tracked_result.boxes.xyxy.cpu().tolist(),
                    tracked_result.boxes.cls.cpu().tolist(), tracked_result.boxes.conf.cpu().tolist(),
                    tracked_result.boxes.id.cpu().tolist()):
                kind = tracked_result.names[int(category)]
                prefix = "p" if kind == "person" else "v"
                objects.append({"id": f"{prefix}{scene:02d}_{int(identity):03d}", "kind": kind,
                                "bbox": [round(x, 2) for x in box], "confidence": round(confidence, 4),
                                "observed": True})
        # Suppress nearly contained duplicate detections of the same entity kind.
        # An open door can otherwise become a second vehicle/person track.
        retained = []
        for obj in sorted(objects, key=lambda o: (o["bbox"][2]-o["bbox"][0])*(o["bbox"][3]-o["bbox"][1]), reverse=True):
            x1, y1, x2, y2 = obj["bbox"]
            duplicate = False
            for other in retained:
                if (obj["kind"] == "person") != (other["kind"] == "person"):
                    continue
                a, b, c, d = other["bbox"]
                intersection = max(0, min(x2, c)-max(x1, a))*max(0, min(y2, d)-max(y1, b))
                if intersection / max(1, (x2-x1)*(y2-y1)) > 0.9:
                    duplicate = True
                    break
            if not duplicate:
                retained.append(obj)
        rows.append({"frame_index": index, "timestamp_s": timestamp, "scene": scene, "objects": retained})
        if index % 100 == 0:
            print(f"{Path(path).stem}: tracking {index}/{metadata['frame_count']}", flush=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    temporary.replace(output)
    write_json(sidecar, {"signature": signature, "detector_sha256": sha256(model_path)})
    return rows
