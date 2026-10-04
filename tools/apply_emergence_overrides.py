"""Add a conservative EXIT when a strong emergence cue was rejected by the VLM."""
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()

for rule_path in sorted((a.root / "rules").glob("*.json")):
    clip_path = a.root / "clips" / rule_path.name
    if not clip_path.exists():
        continue
    rules = json.loads(rule_path.read_text())
    clip = json.loads(clip_path.read_text())
    events = clip.setdefault("interactions", [])
    for cue in rules.get("emergence_cues", []):
        start = float(cue["start_s"])
        end = min(float(cue["end_s"]), start + 2.25)
        already = False
        for event in events:
            pids = {x.get("person_id") for x in event.get("persons", [])}
            vid = event.get("vehicle", {}).get("vehicle_id")
            overlap = any(min(end, s["end_s"]) > max(start, s["start_s"]) for s in event.get("spans", []))
            if cue["person_id"] in pids and cue["vehicle_id"] == vid and overlap:
                already = True
                break
        if already:
            continue
        events.append({
            "event_id": "e_emergence_%02d" % (len(events) + 1),
            "type": "exit",
            "persons": [{"person_id": cue["person_id"], "description": "Person emerging from vehicle"}],
            "vehicle": {"vehicle_id": cue["vehicle_id"], "description": "Vehicle associated with emergence"},
            "spans": [{"start_s": start, "end_s": end}],
            "truncated_start": False,
            "truncated_end": False,
            "evidence_frames": cue.get("frames", []),
            "group_id": None,
        })
    clip_path.write_text(json.dumps(clip, indent=2))
