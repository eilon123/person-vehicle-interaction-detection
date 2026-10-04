"""Export final interactions with VLM descriptions aligned to post-processing."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from person_vehicle.io import read_json, write_json

ACTION_TEXT = {"enter": "enters", "exit": "exits",
               "load_unload": "loads or unloads",
               "other_interaction": "interacts with"}


def identities(event: dict) -> tuple[set[str], str | None]:
    people = {person.get("person_id") for person in event.get("persons", [])}
    return people, event.get("vehicle", {}).get("vehicle_id")


def overlaps(event: dict, review: dict) -> bool:
    people, vehicle = identities(event)
    return (review.get("decision") == "interaction" and review.get("person_id") in people
            and review.get("vehicle_id") == vehicle
            and any(span.get("start_s", 0) < review.get("end_s", 0)
                    and review.get("start_s", 0) < span.get("end_s", 0)
                    for span in event.get("spans", [])))


def person_text(person: dict) -> str:
    person_id = person.get("person_id") or "unknown"
    description = str(person.get("description", "")).strip().rstrip(".")
    description = re.sub(rf"^Person\s+{re.escape(person_id)}\s+is\s+", "", description, flags=re.I)
    description = re.sub(rf"^Person\s+{re.escape(person_id)}\s+", "", description, flags=re.I)
    description = re.sub(r"^Person\s+", "", description, flags=re.I)
    if not description or description.lower().startswith(("emerging from", "associated with")):
        return f"Person {person_id}"
    return f"Person {person_id} {description[0].lower() + description[1:]}"


def vehicle_text(vehicle: dict) -> str:
    vehicle_id = vehicle.get("vehicle_id") or "unknown"
    description = str(vehicle.get("description", "")).strip().rstrip(".")
    description = re.sub(rf"^Vehicle\s+{re.escape(vehicle_id)}\s+is\s+(?:a\s+)?", "", description, flags=re.I)
    description = re.sub(r"\s+and\s+appears\s+to\s+be\s+(?:a\s+)?", " ", description, flags=re.I)
    if not description or description.lower().startswith("vehicle associated"):
        return f"vehicle {vehicle_id}"
    if vehicle_id.lower() not in description.lower():
        description += f" ({vehicle_id})"
    return description[0].lower() + description[1:]


def final_description(event: dict, matched: list[dict]) -> tuple[str, list[str]]:
    raw = [str(review.get("reason", "")).strip() for review in matched
           if str(review.get("reason", "")).strip()]
    final_type = event.get("type", "other_interaction")
    person_parts = [person_text(person) for person in event.get("persons", [])]
    subject = " and ".join(person_parts) or "The person"
    action = ACTION_TEXT.get(final_type, "interacts with")
    return f"{subject} {action} {vehicle_text(event.get('vehicle', {}))}.", raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True); parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--experiment")
    args = parser.parse_args()
    clips_dir, reviews_dir, output = map(Path, (args.clips, args.reviews, args.output))
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for clip_path in sorted(clips_dir.glob("*.json")):
        clip = read_json(clip_path); review_path = reviews_dir / clip_path.name
        reviews = read_json(review_path) if review_path.exists() else []
        descriptions = []
        for index, event in enumerate(clip.get("interactions", []), 1):
            matched = [review for review in reviews if overlaps(event, review)]
            description, raw = final_description(event, matched)
            spans = event.get("spans", []); people, vehicle = identities(event)
            person_ids = sorted(item for item in people if item)
            descriptions.append({"description_id": f"d{index:04d}", "event_id": event.get("event_id"),
                "candidate_ids": [r.get("candidate_id") for r in matched if r.get("candidate_id")],
                "person_ids": person_ids, "person_id": person_ids[0] if person_ids else None,
                "vehicle_id": vehicle, "start_s": min((s.get("start_s", 0) for s in spans), default=0),
                "end_s": max((s.get("end_s", 0) for s in spans), default=0), "decision": "interaction",
                "final_type": event.get("type"), "description": description,
                "raw_vlm_descriptions": raw, "source": "final_event_with_vlm_evidence"})
        write_json(output / clip_path.name, {"schema_version": "2.0", "experiment": args.experiment,
                   "clip_id": clip.get("clip_id", clip_path.stem), "descriptions": descriptions})
        written += 1
    print(f"Wrote {written} final-interaction description files to {output.resolve()}")


if __name__ == "__main__":
    main()
