"""Build a portable, checksum-indexed deliverable from a completed run."""
import argparse
import json
import zipfile
from pathlib import Path

from person_vehicle.events import ClipOutput
from person_vehicle.io import read_json, sha256, write_json


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/final")
parser.add_argument("--inventory", default="outputs/audit/inventory.json")
parser.add_argument("--destination", default="deliverables/annotated_results.zip")
args = parser.parse_args()
root = Path(args.output)
inventory = read_json(args.inventory)
manifest = read_json(root / "manifest.json")
expected = {entry["clip_id"] for entry in inventory}
actual = {entry["clip_id"] for entry in manifest["clips"] if entry["status"] == "ok"}
if actual != expected:
    raise ValueError(f"Run is incomplete: expected {sorted(expected)}, successful {sorted(actual)}")
files = []
combined = []
for clip_id in sorted(expected):
    event_path = root / "clips" / f"{clip_id}.json"
    clip = ClipOutput.model_validate(read_json(event_path))
    source = next(item for item in inventory if item["clip_id"] == clip_id)
    if clip.status != "ok" or clip.source_sha256 != source["source_sha256"]:
        raise ValueError(f"Failed or stale clip: {clip_id}")
    combined.extend({"clip_id": clip_id, **event.model_dump()} for event in clip.interactions)
    video = root / "annotated" / f"{clip_id}_annotated.mp4"
    qa_path = video.with_suffix(".qa.json")
    qa = read_json(qa_path)
    if not qa["fully_decoded"] or sha256(video) != qa["sha256"]:
        raise ValueError(f"Video missing, changed, or unverified: {clip_id}")
    files.extend([event_path, video, qa_path, root / "review" / f"{clip_id}.json"])
write_json(root / "interactions.json", combined)
files += [root / name for name in ("interactions.json", "manifest.json", "config.json", "index.html", "artifact_metrics.json")]
for name in ("development_metrics.json", "visual_review.json", "repeatability.json"):
    if (root / name).exists():
        files.append(root / name)
checksums = {path.relative_to(root).as_posix(): sha256(path) for path in files}
write_json(root / "checksums.json", checksums)
destination = Path(args.destination)
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(".tmp.zip")
with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for path in [*files, root / "checksums.json"]:
        archive.write(path, path.relative_to(root).as_posix())
    archive.writestr("README.txt", "Open index.html to review the annotated videos.\n"
        "interactions.json contains flattened automatic event predictions for all clips.\n"
        "Per-clip JSON also records zero-event clips. Predictions require semantic review.\n"
        "checksums.json lists SHA-256 digests of the included artifacts.\n")
temporary.replace(destination)
print(json.dumps({"archive": str(destination), "bytes": destination.stat().st_size,
                  "sha256": sha256(destination), "clips": len(expected), "events": len(combined)}))
