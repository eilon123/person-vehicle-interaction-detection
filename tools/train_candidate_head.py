"""Low-data transfer learning: frozen ResNet embeddings plus a LOOCV linear head."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from person_vehicle.events import ClipOutput
from person_vehicle.io import fingerprint, read_json, sha256, videos, write_json


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def label_candidate(candidate, truth):
    for event in truth["interactions"]:
        for span in event["spans"]:
            duration = span["end_s"] - span["start_s"]
            if overlap(candidate["start_s"], candidate["end_s"], span["start_s"], span["end_s"]) >= .5 * duration:
                return 1
    return 0


def touches_uncertain(candidate, uncertain):
    return any(overlap(candidate["start_s"], candidate["end_s"], span["start_s"], span["end_s"]) > 0
               for event in uncertain.get("interactions", []) for span in event["spans"])


def sample_indices(rows, candidate, count=6):
    eligible = [row for row in rows if row["scene"] == candidate["scene"] and
                candidate["start_s"] <= row["timestamp_s"] < candidate["end_s"]]
    if not eligible:
        return [], None
    positions = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    boxes = [obj["bbox"] for row in eligible for obj in row["objects"]
             if obj["id"] in (candidate["person_id"], candidate["vehicle_id"])]
    if not boxes:
        return [], None
    crop = [min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes)]
    return [eligible[index]["frame_index"] for index in sorted(set(positions))], crop


def extract_features(video_paths, base, examples, device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()
    features = {}
    grouped = defaultdict(list)
    for example in examples:
        grouped[example["clip_id"]].append(example)
    for clip_id, clip_examples in grouped.items():
        rows = [json.loads(line) for line in (base / "tracks" / f"{clip_id}.jsonl").read_text(encoding="utf-8").splitlines()]
        requests = defaultdict(list)
        for example in clip_examples:
            indices, crop = sample_indices(rows, example, 6)
            for order, index in enumerate(indices):
                requests[index].append((example["key"], order, crop))
        captured = defaultdict(list)
        capture = cv2.VideoCapture(str(video_paths[clip_id]))
        frame_index = 0
        batch, owners = [], []
        def flush():
            if not batch:
                return
            with torch.inference_mode():
                output = model(torch.stack(batch).to(device)).cpu().numpy()
            for owner, vector in zip(owners, output):
                captured[owner].append(vector)
            batch.clear(); owners.clear()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for key, order, crop in requests.get(frame_index, []):
                x1, y1, x2, y2 = crop
                margin = .12 * max(x2 - x1, y2 - y1)
                h, w = frame.shape[:2]
                cut = frame[max(0, int(y1-margin)):min(h, int(y2+margin)),
                            max(0, int(x1-margin)):min(w, int(x2+margin))]
                if cut.size:
                    batch.append(transform(Image.fromarray(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))))
                    owners.append(key)
                if len(batch) >= 64:
                    flush()
            frame_index += 1
        capture.release(); flush()
        for example in clip_examples:
            vectors = captured[example["key"]]
            if vectors:
                array = np.stack(vectors)
                features[example["key"]] = np.concatenate([array.mean(0), array.std(0), array[-1] - array[0]])
    return features


def train_fold(train_x, train_y, test_x, seed=42):
    torch.manual_seed(seed)
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    head = nn.Linear(train_x.shape[1], 1)
    positives = train_y.sum().clamp_min(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(train_y) - positives) / positives)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=.05)
    for _ in range(250):
        optimizer.zero_grad()
        loss = criterion(head(train_x).squeeze(1), train_y)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        train_p = head(train_x).sigmoid().squeeze(1)
        test_p = head(test_x).sigmoid().squeeze(1)
    best = (0, .5)
    for threshold in np.linspace(.2, .8, 25):
        predicted = train_p >= threshold
        tp = int((predicted & (train_y == 1)).sum()); fp = int((predicted & (train_y == 0)).sum())
        fn = int(((~predicted) & (train_y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        best = max(best, (f1, float(threshold)))
    return test_p.numpy(), best[1]


def main():
    parser = argparse.ArgumentParser(description="Train a frozen-ResNet candidate filter with leave-one-video-out CV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--uncertain")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=("loocv", "small_sample_all"), default="loocv")
    parser.add_argument("--max-train-examples", type=int, default=20)
    args = parser.parse_args()
    base, output = Path(args.base), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    truth = read_json(args.reference)
    uncertain = read_json(args.uncertain) if args.uncertain else {}
    video_paths = {path.stem: path for path in videos(args.input)}
    examples = []
    for clip_id in truth:
        candidates = read_json(base / "candidates" / f"{clip_id}.json")
        for candidate in candidates:
            if touches_uncertain(candidate, uncertain.get(clip_id, {})):
                continue
            examples.append({**candidate, "clip_id": clip_id, "key": f"{clip_id}/{candidate['candidate_id']}",
                             "label": label_candidate(candidate, truth[clip_id])})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = extract_features(video_paths, base, examples, device)
    examples = [example for example in examples if example["key"] in features]
    scores, fold_info = {}, []
    if args.protocol == "loocv":
        for held_out in sorted(truth):
            train = [e for e in examples if e["clip_id"] != held_out]
            test = [e for e in examples if e["clip_id"] == held_out]
            train_x = torch.tensor(np.stack([features[e["key"]] for e in train]), dtype=torch.float32)
            train_y = torch.tensor([e["label"] for e in train], dtype=torch.float32)
            test_x = torch.tensor(np.stack([features[e["key"]] for e in test]), dtype=torch.float32)
            probabilities, threshold = train_fold(train_x, train_y, test_x)
            for example, probability in zip(test, probabilities):
                scores[example["key"]] = {"probability": float(probability), "threshold": threshold,
                                          "accepted": bool(probability >= threshold), "label": example["label"]}
            fold_info.append({"held_out": held_out, "train_examples": len(train), "test_examples": len(test),
                              "train_positive": sum(e["label"] for e in train), "threshold": threshold})
    else:
        positives = sorted((e for e in examples if e["label"] == 1), key=lambda e: (e["clip_id"], e["candidate_id"]))
        negatives = sorted((e for e in examples if e["label"] == 0), key=lambda e: (e["clip_id"], e["candidate_id"]))
        per_class = max(1, args.max_train_examples // 2)
        def spread(items, count):
            if len(items) <= count:
                return items
            return [items[index] for index in np.linspace(0, len(items) - 1, count, dtype=int)]
        train = spread(positives, per_class) + spread(negatives, per_class)
        train_x = torch.tensor(np.stack([features[e["key"]] for e in train]), dtype=torch.float32)
        train_y = torch.tensor([e["label"] for e in train], dtype=torch.float32)
        all_x = torch.tensor(np.stack([features[e["key"]] for e in examples]), dtype=torch.float32)
        probabilities, threshold = train_fold(train_x, train_y, all_x)
        training_keys = {e["key"] for e in train}
        for example, probability in zip(examples, probabilities):
            scores[example["key"]] = {"probability": float(probability), "threshold": threshold,
                                      "accepted": bool(probability >= threshold), "label": example["label"],
                                      "used_for_training": example["key"] in training_keys}
        fold_info.append({"protocol": "small_sample_all", "train_examples": len(train),
                          "train_positive": sum(e["label"] for e in train), "evaluated_examples": len(examples),
                          "threshold": threshold, "training_keys": sorted(training_keys)})

    shutil.copytree(base, output, ignore=shutil.ignore_patterns("comparison", "annotated", "annotated_vs_gt",
                                                                "*.html", "*metrics.json"))
    config = read_json(output / "config.json")
    protocol_name = "leave_one_video_out" if args.protocol == "loocv" else "small_sample_all_videos_in_sample_assisted"
    config.update({"experiment_name": "experiment_5",
                   "experiment_change": f"Frozen ResNet-18 candidate filter over Experiment 2 VLM events ({protocol_name})",
                   "candidate_filter": "frozen_resnet18_linear_head", "training_protocol": protocol_name,
                   "training_sample_count": sum(row["train_examples"] for row in fold_info)})
    write_json(output / "config.json", config)
    manifest = read_json(output / "manifest.json")
    manifest.update({"config": config, "config_sha256": fingerprint(config), "derived_from": str(base.resolve()),
                     "derivation": "Experiment 2 predictions filtered by held-out-fold candidate classifier"})
    for clip_path in sorted((output / "clips").glob("*.json")):
        clip = read_json(clip_path); clip_id = clip_path.stem
        candidates = read_json(output / "candidates" / f"{clip_id}.json")
        accepted = [candidate for candidate in candidates
                    if scores.get(f"{clip_id}/{candidate['candidate_id']}", {}).get("accepted")]
        def keep(event):
            for candidate in accepted:
                if (candidate["person_id"] == event["persons"][0]["person_id"] and
                        candidate["vehicle_id"] == event["vehicle"]["vehicle_id"] and
                        any(overlap(candidate["start_s"], candidate["end_s"], span["start_s"], span["end_s"]) > 0
                            for span in event["spans"])):
                    return True
            return False
        clip["interactions"] = [event for event in clip["interactions"] if keep(event)]
        validated = ClipOutput.model_validate(clip).model_dump(); write_json(clip_path, validated)
        record = next(row for row in manifest["clips"] if row["clip_id"] == clip_id)
        record["events"] = len(validated["interactions"]); record["json_sha256"] = sha256(clip_path)
    write_json(output / "manifest.json", manifest)
    write_json(output / "candidate_filter_scores.json", scores)
    write_json(output / "training_report.json", {"backbone": "ResNet-18 ImageNet-1K", "backbone_frozen": True,
        "trained_parameters": 1537, "examples": len(examples), "positive_examples": sum(e["label"] for e in examples),
        "excluded_uncertain": True, "protocol": protocol_name,
        "independent_evaluation": args.protocol == "loocv", "folds": fold_info})
    print(output.resolve())


if __name__ == "__main__":
    main()
