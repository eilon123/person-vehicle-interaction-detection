# Person–Vehicle Interaction

Python pipeline that tracks people and vehicles, verifies candidate actions with
a local vision-language model, exports structured events, and renders full-length
annotated MP4s. The input clips have no audio. See [PLAN.md](PLAN.md) for the design
and KPI definitions, and [docs/ambiguities.md](docs/ambiguities.md) for decisions.

## Quick start: clone to complete results

The following is the shortest supported path on Windows. It installs the project,
downloads the pinned detector and VLM assets, runs the full pipeline, evaluates
the canonical assignment clips, creates HTML reports and Live Review, and renders
annotated MP4 files.

Requirements: Python 3.11 or newer, Git, approximately 35 GB free disk space,
and preferably an NVIDIA GPU with at least 8 GB VRAM. CPU execution is supported
but VLM inference is substantially slower.

```powershell
git clone https://github.com/eilon123/person-vehicle-interaction-detection.git person-vehicle-interaction
Set-Location person-vehicle-interaction

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip install -e . --no-deps
```

Copy the eight assignment MP4 files into a directory such as `data\videos`.
Source videos and model weights are intentionally excluded from Git because of
their size. The expected clip names are listed under
[Paths used in this workspace](#paths-used-in-this-workspace).

Run everything with one command:

```powershell
python tools\quickstart.py --input "data\videos" --output "outputs\experiments"
```

The first run downloads the model assets and can take considerable time. Later
runs reuse matching tracking/model caches. When the command finishes, open:

```text
outputs\experiments\experiment_1\kpi_dashboard.html   KPI report
outputs\experiments\experiment_1\live_review.html     Interactive review
outputs\experiments\experiment_1\annotated\           Annotated MP4 files
outputs\experiments\experiment_1\clips\               Final event JSON
outputs\experiments\experiment_1\vlm_descriptions\    Concise and raw VLM text
outputs\experiments\all_experiments.html               Experiment comparison
```

Use `--skip-download` after the assets are installed. Add `--number N` to choose
an explicit unused experiment number. To run the detector on unrelated videos,
use `person_vehicle run`; the bundled KPI reference applies only to the supplied
assignment clip IDs and must not be used to claim accuracy on other footage.

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

## Operational guide for the current workspace

This section is the practical entry point for running the current algorithm,
creating reports, rendering videos, and finding the outputs. Commands use
PowerShell and assume the current local directory layout. Change the four path
variables when running from another machine.

### Paths used in this workspace

```powershell
$repo = "C:\Users\eilon\hw work\person-vehicle-interaction"
$python = "C:\Users\eilon\hw work\.venv\Scripts\python.exe"
$videos = "C:\black rover\Assignment26\Videos"
$results = "C:\Users\eilon\Documents\Codex\2026-09-28\new-chat\outputs\saved_results"

Set-Location $repo
$env:PYTHONPATH = "$repo\src"
```

The input directory contains the original MP4 files. A command that accepts
`--input` can normally receive either this directory or one specific MP4:

```text
C:\black rover\Assignment26\Videos\
  1THkHYIQ_bY_0.mp4
  gt1125_06.mp4
  HIu4lM4B8hA_1.mp4
  iMGR_0AG3a8_2_3.mp4
  mKzCQKTHizw_0.mp4
  mKzCQKTHizw_1.mp4
  NmlzoaDcOuI_1.mp4
  NmlzoaDcOuI_6.mp4
```

The current release configuration is `configs/experiment_39.yaml`. Experiment
40 uses the same detections and interaction algorithm and adds structured,
one-sentence VLM descriptions to the saved output and Live Review.

### Activate or call the virtual environment

Either activate the environment once:

```powershell
& "C:\Users\eilon\hw work\.venv\Scripts\Activate.ps1"
```

or use `& $python` in every command, as shown below. Calling the interpreter by
its full path is safer in scripts because it does not depend on shell activation.

### Run a complete numbered experiment

This is the recommended command for a new experiment. It runs the algorithm,
evaluates it against the manual GT, creates the KPI dashboard, exports structured
VLM descriptions, creates Live Review, and refreshes the experiment history.
Choose an unused number; the runner refuses to overwrite an existing experiment.

```powershell
& $python tools\run_experiment.py `
  --input $videos `
  --config configs\experiment_39.yaml `
  --reference "$results\ground_truth_7_scenes.json" `
  --experiments-root $results `
  --number 41 `
  --description "Experiment 39 algorithm with final interaction descriptions"
```

The reference above matches the current seven-scene cumulative report, which
excludes the deliberately omitted long `gt1125_06` run. Replace it with
`annotations\manual\events.json` when running and comparing a clean set of
complete eight-scene experiments.

Omit `--number` to choose the next available number automatically. Add
`--no-resume` only when every applicable cached stage must be recomputed. Tracking
uses a content-addressed shared cache and reads its saved result when the video,
model, and tracking configuration fingerprint match. VLM responses are resumed
from the experiment's `verification` directory, or from an explicitly configured
reuse directory. Changed inputs invalidate the relevant cache rather than
silently reusing stale results.

To run one scene only, pass the source MP4 instead of the directory and use a
separate experiment output root or a clearly named scene trial:

```powershell
& $python -m person_vehicle run `
  --input "$videos\iMGR_0AG3a8_2_3.mp4" `
  --output "$results\scene_trials\iMGR_trial" `
  --config configs\experiment_39.yaml
```

Do not compare a one-scene trial to the full GT as though it were a complete
experiment. The cumulative report expects every scene present in its chosen GT.
If a historical experiment deliberately omits `gt1125_06`, generate the history
with the corresponding seven-scene GT file rather than mixing evaluation sets.

### Run only tracking and candidate generation

Use this command when inspecting detection, tracking, and candidate generation
without loading the VLM:

```powershell
& $python -m person_vehicle track `
  --input $videos `
  --output "$results\tracking_trial" `
  --config configs\experiment_39.yaml
```

The important outputs are `tracks/*.jsonl` and `candidates/*.json`.

### Render annotated MP4 videos from saved results

Rendering does not rerun YOLO or the VLM. It reads the saved event and track
files and draws them over the original videos. For Experiment 40:

```powershell
$experiment = "$results\experiment_40"

& $python -m person_vehicle render `
  --input $videos `
  --events "$experiment\clips" `
  --tracks "$experiment\tracks" `
  --reference "annotations\manual\events.json" `
  --output "$experiment\annotated"
```

The resulting MP4 files are stored at:

```text
experiment_40\annotated\<clip_id>_annotated.mp4
```

Remove `--reference` when GT labels should not appear in the video. To render one
scene, pass one MP4 to `--input`; the event and track directories can remain the
same.

### Create or refresh Live Review

Live Review is an HTML viewer rather than a newly encoded video. It uses the
original MP4 files, overlays the saved tracks, shows algorithm and GT timelines,
and displays the final one-sentence VLM description active at the current time.

```powershell
& $python tools\create_live_review.py `
  --input $videos `
  --events "$experiment\clips" `
  --tracks "$experiment\tracks" `
  --descriptions "$experiment\vlm_descriptions" `
  --reference "annotations\manual\events.json" `
  --output "$experiment\live_review.html"
```

Add `--clip iMGR_0AG3a8_2_3` to create a viewer containing only one scene. Open
`live_review.html` in a browser. The source MP4 paths are local, so moving the
HTML to another computer without the source videos will leave the player empty.

### Export concise VLM descriptions

The exporter produces one JSON document per scene and includes final interactions
only. `description` is one decisive sentence based on the final action after
post-processing. `raw_vlm_descriptions` preserves the original detailed VLM text
for later investigation.

```powershell
& $python tools\export_vlm_descriptions.py `
  --clips "$experiment\clips" `
  --reviews "$experiment\review" `
  --output "$experiment\vlm_descriptions" `
  --experiment experiment_40
```

Example structure:

```json
{
  "event_id": "e001",
  "person_ids": ["p00_020"],
  "vehicle_id": "v00_001",
  "start_s": 4.2,
  "end_s": 7.8,
  "decision": "interaction",
  "final_type": "exit",
  "description": "Person p00_020 wearing a white shirt exits a silver sedan.",
  "raw_vlm_descriptions": ["Original detailed VLM response..."]
}
```

### Generate KPI and comparison reports

Evaluate interaction presence without requiring the predicted action type or
participant IDs to match:

```powershell
& $python -m person_vehicle evaluate `
  --pred $experiment `
  --reference "annotations\manual\events.json" `
  --subset all `
  --binary-timeline `
  --output "$experiment\manual_binary_metrics.json"
```

Create the readable per-experiment dashboard:

```powershell
& $python tools\kpi_dashboard.py `
  --pred $experiment `
  --reference "annotations\manual\events.json" `
  --binary-timeline `
  --output "$experiment\kpi_dashboard.html"
```

Refresh the report comparing all complete experiments in the same root:

```powershell
& $python tools\experiment_history.py `
  --experiments-root $results `
  --reference "$results\ground_truth_7_scenes.json" `
  --output "$results\all_experiments.html"
```

Use `annotations\manual\events.json` instead when every archived experiment in
the root contains all eight labelled scenes. Reports are found at:

```text
experiment_N\kpi_dashboard.html   Per-experiment KPI
experiment_N\live_review.html     Interactive video review
all_experiments.html              Cross-experiment comparison
```

### Open the manual annotation tool

```powershell
& $python -m person_vehicle annotate `
  --input $videos `
  --output "annotations\manual"
```

To display predictions while editing GT:

```powershell
& $python -m person_vehicle annotate `
  --input $videos `
  --output "annotations\manual" `
  --pred $experiment `
  --tracks "$experiment\tracks"
```

Confirmed annotations are saved immediately in
`annotations\manual\events.json`; unresolved intervals are stored separately in
`annotations\manual\uncertain.json`.

### Debug videos

Render every person and vehicle track for one scene:

```powershell
& $python tools\render_tracks_only.py `
  --input "$videos\iMGR_0AG3a8_2_3.mp4" `
  --tracks "$experiment\tracks\iMGR_0AG3a8_2_3.jsonl" `
  --output "$experiment\debug\iMGR_tracks.mp4"
```

Render the interaction candidates before VLM filtering:

```powershell
& $python tools\render_candidates.py `
  --input "$videos\iMGR_0AG3a8_2_3.mp4" `
  --tracks "$experiment\tracks\iMGR_0AG3a8_2_3.jsonl" `
  --candidates "$experiment\candidates\iMGR_0AG3a8_2_3.json" `
  --output "$experiment\debug\iMGR_candidates.mp4"
```

### Output directory reference

Each complete `experiment_N` directory can contain:

| Path | Contents |
|---|---|
| `config.json` or `experiment_config.yaml` | Exact configuration used by the run |
| `audit/` | Video metadata and input audit |
| `tracks/*.jsonl` | Per-frame people and vehicle tracks |
| `candidates/*.json` | Person–vehicle candidates before VLM filtering |
| `verification/` | Cached/raw verifier artifacts |
| `review/*.json` | Detailed VLM decisions for every reviewed candidate |
| `clips/*.json` | Final machine-readable interactions |
| `vlm_descriptions/*.json` | One-sentence descriptions of final interactions plus raw VLM text |
| `annotated/*.mp4` | Encoded result videos |
| `manual_binary_metrics.json` | Machine-readable KPI results |
| `kpi_dashboard.html` | Human-readable KPI report |
| `live_review.html` | Interactive review with tracks, GT, timelines, and descriptions |

For the current workspace, the main results are under
`C:\Users\eilon\Documents\Codex\2026-09-28\new-chat\outputs\saved_results`.
Experiment 39 is the released algorithm result; Experiment 40 adds the structured
VLM-description presentation without changing its interaction KPI.

## Run

### Complete numbered experiment

Use the experiment runner for every new algorithm variant. It automatically chooses
the next experiment number, preserves a snapshot of the configuration, uses the
single canonical GT produced by the annotation tool, runs inference and KPI
evaluation, creates a live `ALG` versus `GT` review page, and refreshes one
cumulative report covering every experiment in the same root directory.

```powershell
python tools/run_experiment.py `
  --input "C:\black rover\Assignment26\Videos" `
  --config configs/experiment_2.yaml `
  --experiments-root "C:\path\to\saved_results" `
  --description "Short explanation of the algorithm change"
```

Each run creates `experiment_N/kpi_dashboard.html`,
`experiment_N/live_review.html`, the raw outputs and metrics, plus
`all_experiments.html` at the root. Never reuse an experiment number; the command
refuses to overwrite an existing experiment. Use `--number N` only when assigning
a specific unused number. Pass `--reference PATH` only when intentionally using
a different GT source. The cumulative report recalculates every experiment from
the canonical GT each time it is generated, so edits made in the annotation tool
flow through to historical experiment reports too.

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
across pipeline variants. It also includes a compact description of the pipeline
used for the evaluated output. The detailed algorithm description is generated
from the same configuration, including a short configuration fingerprint. To
keep reports separate for different algorithms, write each one beside its run:

```powershell
python tools/kpi_dashboard.py --pred outputs/variant_a --reference annotations/manual/events.json --binary-timeline --output outputs/variant_a/kpi_dashboard.html
python tools/kpi_dashboard.py --pred outputs/variant_b --reference annotations/manual/events.json --binary-timeline --output outputs/variant_b/kpi_dashboard.html
```

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
