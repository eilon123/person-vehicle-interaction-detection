"""Summarize generated artifacts without fabricating semantic accuracy labels."""
import argparse
from pathlib import Path

from person_vehicle.events import ClipOutput
from person_vehicle.io import read_json, write_json


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/final")
parser.add_argument("--inventory", default="outputs/audit/inventory.json")
args = parser.parse_args()
root = Path(args.output)
inventory = read_json(args.inventory)
results = []
for source in inventory:
    clip_id = source["clip_id"]
    path = root / "clips" / f"{clip_id}.json"
    record = {"clip_id": clip_id, "input_duration_s": source["duration_s"], "schema_valid": False,
              "status": "missing", "events": None, "video_valid": False}
    if path.exists():
        clip = ClipOutput.model_validate(read_json(path))
        record.update(schema_valid=True, status=clip.status, events=len(clip.interactions))
        record["types"] = [e.type for e in clip.interactions]
    qa_path = root / "annotated" / f"{clip_id}_annotated.qa.json"
    if qa_path.exists():
        qa = read_json(qa_path)
        record.update(video_valid=qa["fully_decoded"], max_timestamp_error_s=qa["max_timestamp_error_s"],
                      duration_error_s=qa["duration_error_s"], overlay_agreement=qa["overlay_agreement"],
                      visual_review=qa["visual_review"])
    review_path = root / "review" / f"{clip_id}.json"
    if review_path.exists():
        reviews = read_json(review_path)
        record["candidates"] = len(reviews)
        record["uncertain_candidates"] = sum(r["decision"] == "uncertain" for r in reviews)
    results.append(record)
count = len(results)
report = {"clips": results, "input_clips": count,
          "successful_clips": sum(r["status"] == "ok" for r in results),
          "schema_valid_clips": sum(r["schema_valid"] for r in results),
          "valid_annotated_videos": sum(r["video_valid"] for r in results),
          "total_input_seconds": sum(r["input_duration_s"] for r in results),
          "full_corpus_event_precision": None, "full_corpus_event_recall": None,
          "full_corpus_event_f1": None,
          "semantic_accuracy_note": "Only a one-clip development sanity reference is available; no independent full-corpus accuracy claim."}
write_json(root / "artifact_metrics.json", report)
print(report)
