"""Cache per-frame raw YOLO person detections required by the emergence rule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import yaml

from person_vehicle.io import fingerprint, read_json, sha256, videos, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    settings = Path(".cache/ultralytics").resolve()
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    model = YOLO(str(config["detector"]))
    args.output.mkdir(parents=True, exist_ok=True)
    confidence = float(config.get("detection_confidence", .2))
    image_size = int(config.get("image_size", 960))
    for video in videos(args.input):
        target = args.output / f"{video.stem}.json"
        sidecar = args.output / f"{video.stem}.meta.json"
        signature = fingerprint({"video": sha256(video), "detector": sha256(config["detector"]),
                                 "confidence": confidence, "image_size": image_size,
                                 "device": str(config["device"])})
        if (target.exists() and sidecar.exists() and not args.no_resume
                and read_json(sidecar).get("signature") == signature):
            print(f"{video.stem}: loaded raw person detections from cache", flush=True)
            continue
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
        rows = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, conf=confidence, classes=[0], imgsz=image_size,
                                   device=config["device"], verbose=False)[0]
            boxes = []
            if result.boxes is not None:
                boxes = [[*[round(float(value), 2) for value in box], round(float(score), 4)]
                         for box, score in zip(result.boxes.xyxy.cpu().tolist(),
                                               result.boxes.conf.cpu().tolist())]
            rows.append({"frame": frame_index, "t": round(frame_index / fps, 6), "boxes": boxes})
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"{video.stem}: raw person detection {frame_index}", flush=True)
        capture.release()
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        temporary.replace(target)
        write_json(sidecar, {"signature": signature, "frames": frame_index})


if __name__ == "__main__":
    main()
