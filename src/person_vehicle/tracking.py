import json
import os
from pathlib import Path

import cv2
import numpy as np

from .io import fingerprint, read_json, sha256, write_json
from .video import frames


def track(path, output, metadata, config, resume=True):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".meta.json")
    model_path = Path(config["detector"])
    if not model_path.exists():
        raise FileNotFoundError(f"Missing detector {model_path}; run the download-assets command")
    track_config = {key: config[key] for key in ["detector", "device", "image_size", "detection_confidence", "tracker", "seed"]}
    track_config["image_size"] = min(config["image_size"], max(640, metadata["width"], metadata["height"]))
    signature = fingerprint({"source": metadata["source_sha256"],
                             "model": sha256(model_path), "config": track_config, "tracker_version": 2})
    if resume and output.exists() and sidecar.exists() and read_json(sidecar)["signature"] == signature:
        return [json.loads(line) for line in output.read_text().splitlines()]
    settings_dir = Path(".cache/ultralytics").resolve()
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO
    model = YOLO(str(model_path))  # A fresh model and tracker per clip.
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
        previous = thumb.astype(float)
        result = model.track(image, persist=True, tracker=config["tracker"],
                             classes=[0, 1, 2, 3, 5, 7], conf=config["detection_confidence"],
                             imgsz=track_config["image_size"], device=config["device"], agnostic_nms=True, verbose=False)[0]
        objects = []
        if result.boxes is not None and result.boxes.id is not None:
            for box, category, confidence, identity in zip(result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.id.cpu().tolist()):
                kind = result.names[int(category)]
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
