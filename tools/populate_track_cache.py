"""Populate the shared track cache from completed experiment directories."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".cache/track_results"))
    args = parser.parse_args()
    copied = skipped = 0
    for root in args.roots:
        for metadata_path in root.rglob("*.meta.json"):
            if metadata_path.parent.name != "tracks":
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                signature = metadata["signature"]
            except (OSError, ValueError, KeyError):
                skipped += 1
                continue
            tracks_path = metadata_path.with_suffix("").with_suffix(".jsonl")
            if not tracks_path.exists():
                skipped += 1
                continue
            destination = args.cache.resolve() / signature[:2] / f"{signature}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(tracks_path, destination)
                destination.with_suffix(".meta.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                copied += 1
            else:
                skipped += 1
    print(f"shared track cache: copied={copied}, skipped={skipped}")


if __name__ == "__main__":
    main()
