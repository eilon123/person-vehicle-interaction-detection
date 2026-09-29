# Person–Vehicle Interaction

Python pipeline that tracks people and vehicles, verifies candidate actions with
a local vision-language model, exports structured events, and renders full-length
annotated MP4s. The input clips have no audio. See [PLAN.md](PLAN.md) for the design
and KPI definitions, and [docs/ambiguities.md](docs/ambiguities.md) for decisions.

## Setup

Run commands from this repository root. Tested on Windows, Python 3.14, and an
RTX 4060 Laptop GPU with 8 GB VRAM. The implementation also accepts CPU devices,
but local vision-model inference will be substantially slower. Allow approximately 35 GB of free disk space for the 7B weights, CUDA
packages, caches, and exported videos.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip install -e . --no-deps
python -m person_vehicle download-assets
```

`requirements.lock` records the tested environment, including CUDA-specific torch
wheels. On a different platform, install a compatible PyTorch build from its
official instructions, then `pip install -e .`; this is a different environment
and needs its own validation. Set `device: cpu` for detection without CUDA.

For this working checkout, the tested interpreter is currently
`..\.venv\Scripts\python.exe` in the parent workspace. A fresh clone can use the
isolated `.venv` recipe above.

Model downloads: YOLO11s from the Ultralytics asset release and
`Qwen/Qwen2.5-VL-7B-Instruct` from Hugging Face, loaded in 4-bit mode. `download-assets` saves detector
checksums in `models/assets.json` and pins the VLM snapshot revision in the config.
Inference uses local files; no clip is uploaded to an external service. Downloads
require network access. Record model/library licenses when distributing: see
[Ultralytics licensing](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
and the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

## Run

```powershell
python -m person_vehicle audit --input "C:\black rover\Assignment26\Videos"
python -m person_vehicle run --input "C:\black rover\Assignment26\Videos" --output outputs/final --config configs/default.yaml --annotate
```

`--input` can also be a single MP4. Detection runs on every frame so render boxes
are observed rather than silently held across missing frames. The action verifier
sees deterministic timestamped samples from each candidate window. Use
`--no-resume` to recompute tracks and verifier responses. Verification responses are cached by source,
track evidence, prompt, and configuration hashes. Changing those inputs invalidates
the relevant cache. Failed clips produce explicit error artifacts and nonzero exit.

Regenerate videos without model inference:

```powershell
python -m person_vehicle render --input "C:\black rover\Assignment26\Videos" --events outputs/final/clips --tracks outputs/final/tracks --output outputs/final/annotated
```

For a compact algorithm-versus-ground-truth review video, add the manual labels.
Only participants in an active algorithm event receive boxes; small `ALG` and
`GT` lines show the asserted action and person-to-vehicle pair. The bottom
timeline uses magenta for algorithm events and yellow for GT spans.

```powershell
python -m person_vehicle render --input "C:\black rover\Assignment26\Videos" --events outputs/final/clips --tracks outputs/final/tracks --reference annotations/manual/events.json --output outputs/final/comparison
```

## Outputs

The deliverable run is under `outputs/final/`. Paths below are relative to the
chosen output directory; older smoke/development artifacts are not final results.

- `clips/<clip_id>.json`: validated events, descriptions, local participant
  IDs, visible time spans, evidence indices, and truncated-event flags.
- `annotated/<clip_id>_annotated.mp4`: boxes/IDs, active action labels,
  participant descriptions, timestamps, and event timeline. No-interaction periods
  say **No confirmed interaction**. Exports preserve original frame presentation timing.
- `annotated/*.qa.json`: full-decode/frame/timing verification; visual QA
  remains explicitly pending until reviewed.
- `review/*.json`: every candidate decision, including uncertainty/reasons.
- `tracks/*.jsonl`: per-frame geometry; `verification/`: raw model responses.
- `manifest.json`: inputs processed by the latest run, output links,
  checksums, model configuration, package versions, and timing.
- `audit/`: per-clip source metadata. The separate `outputs/audit/` directory
  contains the initial inventory and overview sheets.
- `interactions.json`: flattened list with `clip_id` on every event.
- `index.html`: offline video/JSON review page.

This pipeline makes discrete
action decisions, and does not report VLM self-confidence as calibrated probability.
An empty event list means no interaction was confirmed, not proof of absence.
Review mode (`verifier: review`) only exports uncertain candidates; it must not be
presented as a completed automatic action-detection run.

The schema is [schemas/clip_output.schema.json](schemas/clip_output.schema.json).
Times use half-open intervals `[start_s, end_s)` relative to the first frame;
evidence indices are zero-based. Per-clip IDs do not imply cross-clip identities.

Large MP4s, track caches, source footage, and weights are excluded from ordinary Git.
Deliver the annotated folder through a shared artifact folder or release and link
it in the final submission. Local video files remain available after generation. `tools/package_outputs.py`
creates `deliverables/annotated_results.zip` with all eight videos, event JSON,
review decisions, and a checksum index; it rejects incomplete or stale runs.
Extract the archive and open `index.html` to review it.

## Evaluation and tests

### Create manual temporal reference labels

For a small video collection, create the ground truth directly with the local
annotator. It opens one clip at a time and lets you choose a start frame, move to
an end frame, and add either a **confirmed** interval or an **uncertain** interval.
Confirmed intervals are saved in the evaluator-compatible reference file;
uncertain intervals are deliberately kept out of the primary KPI calculation.

```powershell
python -m person_vehicle annotate --input "C:\black rover\Assignment26\Videos" --output annotations/manual
```

To annotate while reviewing the algorithm's active predictions, pass its output
directory. The viewer draws only boxes for participants in an active predicted
event, with small `ALG` and `GT` lines at the top; it hides unrelated tracks.

```powershell
python -m person_vehicle annotate --input "C:\black rover\Assignment26\Videos" --output annotations/manual --pred outputs/final
```

The window saves after every interval and when it closes. Use a short visible
description and stable within-clip IDs for the person and vehicle. The first
time you evaluate, map those reference IDs to the prediction IDs (for example
`p01` and `v01`) in a JSON mapping file. Then run:

```powershell
python -m person_vehicle evaluate --pred outputs/final --reference annotations/manual/events.json --mapping annotations/entity_mapping_manual.json --subset all --output outputs/final/manual_metrics.json
```

If your annotation is deliberately binary over time and does not identify the
person/vehicle pair, add `--binary-timeline`. This produces a separately labelled
temporal-presence score and must not be reported as the pair-correct primary KPI:

```powershell
python -m person_vehicle evaluate --pred outputs/final --reference annotations/manual/events.json --subset all --binary-timeline --output outputs/final/manual_binary_metrics.json
```

Create a readable HTML dashboard with aggregate, per-clip, per-action, and
action-confusion results. It uses the same matching rule as evaluation:

```powershell
python tools/kpi_dashboard.py --pred outputs/final --reference annotations/manual/events.json --binary-timeline --output outputs/final/kpi_dashboard.html
```

The dashboard records the detector, tracker, verifier model/revision, temporal
sampling, thresholds, and seed from `outputs/final/config.json`. Pass `--config`
when the run configuration lives elsewhere; this makes result reports comparable
across pipeline variants.

`annotations/manual/uncertain.json` is a review queue, not ground truth. Resolve
or remove its spans before reporting KPI results. This tool uses nominal frame
timing for navigation; retain the pipeline's timestamp-aware decoder for final
rendering and metric validation.

Every evaluation report also includes `continuous_temporal_overlap`: a
threshold-free duration metric. It unions all predicted and GT interaction spans
within each clip, then reports overlap seconds, temporal precision/recall, F1,
and temporal IoU. Parallel actions therefore do not double-count shared time.

```powershell
python -m pytest -q
python -m person_vehicle evaluate --pred outputs/final --reference annotations/reference_dev.json --mapping annotations/entity_mapping_dev.json --output outputs/final/development_metrics.json
python tools/summarize_results.py --output outputs/final
python tools/export_review_index.py --output outputs/final
python tools/package_outputs.py --output outputs/final
```

The included reference covers **one development clip only**, reviewed by the coding
assistant, and is not a held-out benchmark. See
[reference notes](annotations/reference_dev_notes.md). Full-corpus reference
annotations must be supplied using [annotations/policy.md](annotations/policy.md).
The evaluator requires actual reference data; it does not create ground truth from
predictions. It reports one-to-one pair-correct event precision/recall/F1, type-aware
scores, per-clip counts, temporal-IoU sensitivity (0.3/0.5/0.7), boundary errors, and
false alarms per minute. Undefined metrics are JSON `null`. Without independent
reference annotations, event accuracy is **not measured**.

Run the geometry ablation from cached tracks without loading a VLM:

```powershell
python -m person_vehicle baseline --input "C:\black rover\Assignment26\Videos" --tracks outputs/final/tracks --output outputs/baseline/clips
python -m person_vehicle evaluate --pred outputs/baseline/clips --reference annotations/reference_dev.json --mapping annotations/entity_mapping_dev.json --output outputs/baseline/development_metrics.json
```

Use `tools/review_frames.py` to export dense timestamped sheets for annotation:

```powershell
python tools/review_frames.py --input clip.mp4 --start 4 --end 7 --fps 5
```

## Assumptions and limitations

Supported labels: enter, exit, door operation, load/unload, and other directed
interaction. Passing or standing nearby is not positive evidence. Associated door
closure belongs to an entry/exit episode. Model JSON is validated, unknown evidence
indices are rejected, and invalid replies become visible uncertain candidates.

The model can still hallucinate or miss an action, especially for tiny people,
occlusion, monochrome footage, and sparse samples. Pair proposals depend on detector
tracks, so missed detections can miss events. Current boundaries are sampled-frame
estimates, not a guarantee of frame-exact localization. Scene-cut detection is a
simple image-difference heuristic; camera motion can split tracks. Output quality
needs independent event annotation and review before claiming the plan's KPI targets.

Seeds, fixed sampling, greedy generation, pinned assets, and sorted file order
improve reproducibility. GPU kernels may remain hardware-dependent; the run requests
deterministic algorithms with warnings and does not claim bit-identical behavior
across platforms. Cached replay and fresh-inference repeatability are distinct.
