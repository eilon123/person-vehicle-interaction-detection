import numpy as np
from scipy.optimize import linear_sum_assignment


def union(spans):
    merged = []
    for start, end in sorted((s["start_s"], s["end_s"]) for s in spans):
        if end <= start:
            raise ValueError("Invalid span")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def temporal_iou(a, b):
    a, b = union(a), union(b)
    intersection = sum(max(0, min(a1, b1) - max(a0, b0)) for a0, a1 in a for b0, b1 in b)
    total = sum(end - start for start, end in a + b) - intersection
    return intersection / total if total else 0.0


def span_duration(spans):
    return sum(end - start for start, end in union(spans))


def span_intersection_duration(a, b):
    """Duration of the overlap between two unions of half-open time spans."""
    return sum(max(0, min(a1, b1) - max(a0, b0))
               for a0, a1 in union(a) for b0, b1 in union(b))


def temporal_occupancy(predictions, references):
    """Threshold-free binary interaction-time overlap, including per-clip values.

    Events are unioned before scoring, so parallel actions do not double-count
    time. The aggregate is duration-weighted rather than a mean of clip scores.
    """
    totals = {"intersection_s": 0.0, "predicted_s": 0.0, "reference_s": 0.0}
    per_clip = {}
    for clip_id, truth_clip in references.items():
        predicted_spans = [span for event in predictions[clip_id]["interactions"] for span in event["spans"]]
        reference_spans = [span for event in truth_clip["interactions"] for span in event["spans"]]
        predicted_s = span_duration(predicted_spans) if predicted_spans else 0.0
        reference_s = span_duration(reference_spans) if reference_spans else 0.0
        intersection_s = span_intersection_duration(predicted_spans, reference_spans) if predicted_spans and reference_spans else 0.0
        per_clip[clip_id] = {"intersection_s": intersection_s, "predicted_s": predicted_s,
                             "reference_s": reference_s}
        for key, value in per_clip[clip_id].items():
            totals[key] += value

    def metrics(values):
        intersection_s, predicted_s, reference_s = (values[key] for key in ("intersection_s", "predicted_s", "reference_s"))
        union_s = predicted_s + reference_s - intersection_s
        return {**values, "union_s": union_s,
                "temporal_precision": intersection_s / predicted_s if predicted_s else None,
                "temporal_recall": intersection_s / reference_s if reference_s else None,
                "temporal_iou": intersection_s / union_s if union_s else None,
                "temporal_f1": 2 * intersection_s / (predicted_s + reference_s)
                if predicted_s + reference_s else None}

    return {**metrics(totals), "per_clip": {clip_id: metrics(values) for clip_id, values in per_clip.items()}}


def pair(event, mapping=None):
    mapping = mapping or {}
    person = event["persons"][0]["person_id"]
    vehicle = event["vehicle"]["vehicle_id"]
    return mapping.get(person, person), mapping.get(vehicle, vehicle)


def match_events(predictions, references, threshold=0.5, typed=False, mapping=None, ignore_participants=False):
    if not predictions or not references:
        return []
    # Bonus larger than any possible summed IoU enforces maximum cardinality first.
    bonus = min(len(predictions), len(references)) + 1
    weights = np.zeros((len(predictions), len(references)))
    for i, prediction in enumerate(predictions):
        for j, reference in enumerate(references):
            if ((not ignore_participants and pair(prediction, mapping) != pair(reference))
                    or (typed and prediction["type"] != reference["type"])):
                continue
            overlap = temporal_iou(prediction["spans"], reference["spans"])
            if overlap >= threshold:
                weights[i, j] = bonus + overlap
    left, right = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(j), float(weights[i, j] - bonus)) for i, j in zip(left, right) if weights[i, j] > 0]


def rates(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def evaluate(predictions, references, mappings=None, threshold=0.5, typed=False, ignore_participants=False):
    mappings = mappings or {}
    tp = fp = fn = 0
    per_clip = {}
    starts, ends = [], []
    duration = 0
    for clip_id, reference in references.items():
        if clip_id not in predictions or predictions[clip_id]["status"] != "ok":
            raise ValueError(f"Missing or failed prediction for {clip_id}; report pipeline failure separately")
        predicted = predictions[clip_id]["interactions"]
        truth = reference["interactions"]
        matches = match_events(predicted, truth, threshold, typed, mappings.get(clip_id), ignore_participants)
        clip_tp = len(matches)
        clip_fp, clip_fn = len(predicted) - clip_tp, len(truth) - clip_tp
        per_clip[clip_id] = rates(clip_tp, clip_fp, clip_fn)
        tp, fp, fn = tp + clip_tp, fp + clip_fp, fn + clip_fn
        duration += reference["duration_s"]
        for i, j, _ in matches:
            if not truth[j].get("truncated_start", False):
                starts.append(abs(predicted[i]["spans"][0]["start_s"] - truth[j]["spans"][0]["start_s"]))
            if not truth[j].get("truncated_end", False):
                ends.append(abs(predicted[i]["spans"][-1]["end_s"] - truth[j]["spans"][-1]["end_s"]))
    summary = rates(tp, fp, fn)
    summary.update({"tiou_threshold": threshold, "type_aware": typed,
                    "participant_matching": "ignored" if ignore_participants else "required",
                    "per_clip": per_clip,
                    "false_alarms_per_minute": fp / (duration / 60) if duration else None})
    for label, errors in (("start", starts), ("end", ends)):
        summary[f"{label}_error_s"] = {"n": len(errors), "median": float(np.median(errors)) if errors else None,
                                     "p90": float(np.percentile(errors, 90)) if errors else None}
    return summary


def proposal_recall(candidates, references, mappings=None):
    mappings = mappings or {}
    covered = total = 0
    for clip_id, reference in references.items():
        mapping = mappings.get(clip_id, {})
        for event in reference["interactions"]:
            total += 1
            spans = union(event["spans"])
            duration = sum(b - a for a, b in spans)
            for candidate in candidates.get(clip_id, []):
                candidate_pair = (mapping.get(candidate["person_id"], candidate["person_id"]),
                                  mapping.get(candidate["vehicle_id"], candidate["vehicle_id"]))
                overlap = sum(max(0, min(b, candidate["end_s"]) - max(a, candidate["start_s"])) for a, b in spans)
                if candidate_pair == pair(event) and overlap >= 0.5 * duration:
                    covered += 1
                    break
    return {"covered": covered, "total": total, "recall": covered / total if total else None}


def passerby_false_positive_rate(predictions, negatives, mappings=None):
    mappings = mappings or {}
    false = total = 0
    for clip_id, encounters in negatives.items():
        for encounter in encounters:
            total += 1
            expected_pair = (encounter["person_id"], encounter["vehicle_id"])
            if any(pair(event, mappings.get(clip_id)) == expected_pair and
                   temporal_iou(event["spans"], encounter["spans"]) > 0
                   for event in predictions[clip_id]["interactions"]):
                false += 1
    return {"false_positive_encounters": false, "total_encounters": total,
            "rate": false / total if total else None}
