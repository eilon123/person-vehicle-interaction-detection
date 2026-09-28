# Person–vehicle interaction: implementation and evaluation plan

## 1. Scope and observed data

Build a Python CLI that accepts one MP4 or a directory and produces JSON listing
person–vehicle interactions, their intervals, and visible participant descriptions.
Also generate a full-length annotated MP4 for each of the eight input clips;
annotated videos are a required deliverable alongside the JSON outputs.
Keep identities local to a clip. No training from scratch, cross-camera identity
matching, service deployment, or real-time requirement.

The inspected directory is `C:\black rover\Assignment26\Videos` and contains:

- `1THkHYIQ_bY_0.mp4`
- `gt1125_06.mp4`
- `HIu4lM4B8hA_1.mp4`
- `iMGR_0AG3a8_2_3.mp4`
- `mKzCQKTHizw_0.mp4`
- `mKzCQKTHizw_1.mp4`
- `NmlzoaDcOuI_1.mp4`
- `NmlzoaDcOuI_6.mp4`

This is a filesystem inventory, not a visual review. Durations, frame rates,
camera motion, interaction counts, and visibility remain to be established.
Repeated filename prefixes suggest shared source videos; verify this visually.
The existing OpenMax module is not needed: unknown-class rejection alone does
not identify a temporal relationship between a person and a vehicle.

## 2. Define the annotation policy before implementation

An interaction is an observable action directed at a vehicle, rather than merely
occupying nearby image space. Use these labels:

| Label | Positive evidence |
|---|---|
| `enter` | A person transitions from outside into the passenger compartment. |
| `exit` | A person transitions from the passenger compartment to outside. |
| `door_operation` | A person opens or closes a door without a separately identifiable entry/exit. |
| `load_unload` | A person transfers an object to/from the vehicle or cargo compartment. |
| `other_interaction` | Visible directed contact/action such as pushing, cleaning, or working on the vehicle. |

Passersby, standing nearby, and unexplained overlap are negatives. Continuous
visible occupancy without a new action is not an event. Initially include cars,
vans, trucks, buses, motorcycles, and bicycles; map mounting/dismounting a two-wheeler
to `enter`/`exit` and document that convention. Revise this scope before freezing
labels if the initial review exposes a task-specific interpretation.

Decisions and alternatives:

- **Boundaries:** start at the first visible action evidence (e.g. reaching for a
  handle or beginning a body transition), not the start of an approach. End at
  completion of the action, including immediately associated door closure.
  Alternative: label the whole approach/departure; rejected because it inflates
  overlap and includes noninteraction time.
- **Door plus entry/exit:** report one entry/exit episode including its door action.
  Report a standalone door event only when it is a separate action. Distinct actions
  such as loading followed by entering remain distinct events.
- **Occlusion:** disappearance next to a car does not prove entry. Require a
  visible transition or corroborating before/after evidence. Otherwise record
  an `uncertain` candidate for review, not a confirmed interaction.
- **Partial events:** include visible evidence cut off by a clip boundary and set
  `truncated_start` or `truncated_end`. Do not infer unseen times outside the clip.
- **Multiple people:** use one person–vehicle event per pair, with an optional
  `group_id` for joint actions. This makes counting and attribution unambiguous.
- **Interrupted visibility:** represent one action with multiple visible spans
  only when identity and continuity are supported. Do not invent evidence across
  an occlusion. Keep separate episodes separate.
- **Descriptions:** describe visible clothing, accessories, vehicle type/color,
  and position. Use `unknown` for unclear attributes. Do not infer identity,
  intent, demographics, or vehicle make/model from weak evidence.
- **Uncertain reference cases:** keep an explicit uncertainty label and adjudicate
  where possible. Exclude unresolved cases from primary scoring, report their
  counts, and provide a sensitivity result counting them as positive/negative.

Store the policy and an ambiguity log with clip/time, decision, evidence,
alternatives considered, and expected effect on evaluation.

## 3. Data audit and reference annotations — 2–3 hours

1. Decode every clip and record SHA-256, dimensions, duration, frame count, and
   presentation timestamps. Use timestamps for variable-frame-rate clips; do not
   assume `frame_index / nominal_fps` is exact.
2. Watch each full clip, then inspect candidate actions frame by frame. Contact
   sheets help navigation but cannot establish entry/exit by themselves.
3. Annotate events and clear negative encounters before viewing model predictions.
   Give people and vehicles reference IDs, representative boxes/keyframes, visible
   descriptions, and action spans. Mark hard negatives such as walking behind a car.
4. Separate reference labels from generated output. If a second reviewer is
   available, double-label ambiguous cases and a sample of boundaries; otherwise
   report that this is a single-annotator evaluation.
5. Group clips by original source/scene, conservatively using repeated prefixes.
   Reserve approximately two source groups for a final test, chosen after checking
   that both sets have usable positive and negative examples. Publish the split.
   Do not split neighboring frames or related clips between development and test.

With only eight clips, report counts and per-clip results as well as aggregates.
If every clip is used to tune the pipeline, label results as development results,
not independent test performance. Running the final pipeline on all eight clips
is still required for the delivered output.

## 4. Proposed pipeline — 5–7 hours

### A. Detection and within-clip tracking

Use a pretrained Ultralytics detector with BoT-SORT, pinned to a tested package
version, checkpoint, checksum, and tracker configuration. Both BoT-SORT and
ByteTrack are supported through the documented tracking interface:
[Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/).
Start with a small detector checkpoint; compare one larger checkpoint only if
missed people/vehicles dominate development errors.

Process frames in timestamp order, initially targeting 8–10 FPS for analysis.
Retain original indices/timestamps. Reset tracker state for each clip and scene
cut. Cache detections and tracks. Retain plausible short person tracks: exiting
people may first appear late, and entering people may disappear early.

### B. Generate person–vehicle candidate windows

Use expanded vehicle regions and normalized person-to-vehicle distance to find
candidate pairs. Add windows around person-track appearance/disappearance near
a vehicle, even without a long dwell. Start with roughly two seconds of context
on either side; all margins are configuration values tuned on development data.
Keep competing vehicle candidates when association is ambiguous.

Proximity is only a proposal signal. A geometry-only state machine is a useful
baseline, but cannot reliably distinguish entry from walking behind a vehicle.
Also include a low-rate full-clip scan by the temporal verifier to propose missed
windows; validate any such proposals against decoded frames and entity evidence.
Measure proposal recall before spending time on the final classifier.

### C. Verify actions using temporal visual evidence

Use one pinned video-capable vision-language model, initially
`Qwen2.5-VL-7B-Instruct`, for candidate-window classification and descriptions.
Its documented interface accepts video inputs:
[Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl).
First run a short hardware/latency smoke test; select the smaller 3B variant if
needed and freeze the choice before evaluation. No external API is required by
the default plan. Any later API substitution must identify the provider/model,
data sent, parameters, latency/cost, and caching behavior.

Provide chronological frames with explicit timestamps, highlighted pair IDs,
and both full-scene context and pair crops. Ask for a constrained response:
`interaction`, `no_interaction`, or `uncertain`; action type; evidence frame IDs;
and observable descriptions. Validate the response against a JSON schema. Do not
accept model-generated entity IDs or timestamps absent from the supplied inputs.
Treat self-reported confidence as an uncalibrated score, not a probability.

Inspect accepted boundaries at the original frame rate or a denser local sample.
Do not claim sub-sampling precision when evidence only exists at sparse frames.
Cache model inputs/outputs and prompts. Use greedy decoding and fixed frame
selection. A temporal model is favored over frame-only classification because
entry and exit require direction over time; full-clip captioning alone is rejected
because it gives weak entity attribution and temporal localization.

### D. Assemble and validate events

Merge overlapping duplicate proposals only for the same pair and action episode.
Preserve supported disjoint spans. Keep `uncertain` cases in a separate review
artifact. Produce visible descriptions from representative clear frames, with
unknown attributes left explicit. Track fragments may be linked only with
spatiotemporal/appearance support; otherwise flag uncertain association.

Manual annotations are evaluation data. Do not silently hand-correct generated
results. If curated outputs are also provided, put them in a separate artifact
with an explicit corrections log.

### E. Render annotated videos — 2–3 additional hours

Generate `outputs/annotated/<clip_id>_annotated.mp4` for every input clip, including
clips with no confirmed interactions. Render from the finalized event JSON and
cached tracks so the video and machine-readable results share one source of truth.
Save per-frame boxes, timestamps, entity IDs, and observed/predicted status in
`outputs/tracks/<clip_id>.jsonl`; event JSON alone is insufficient to draw boxes.

Each video should display:

- Clip ID, elapsed timestamp, and original frame index.
- Person and vehicle boxes with stable within-clip IDs (`p02`, `v01`). Use muted
  boxes for tracked entities outside an interaction and highlighted boxes for
  participants during a confirmed event.
- An active-event label such as `e001 | p02 -> v01 | ENTER`, with a connector
  between visible participants. Use text as well as color to communicate status.
- A compact panel with the active participants' short visible descriptions;
  position it to minimize obstruction of the scene and wrap long text.
- A timeline showing confirmed event spans and the current playback position.
  Show `No confirmed interaction` when no event is active, without implying that
  every nearby person has been confidently classified as a passerby.

Display event labels only when `start_s <= frame_timestamp < end_s` for one of
the event's spans. During a visibility gap, do not draw a stale box or connector
as if the participant were visible. Optional diagnostic videos may show uncertain
candidates in amber and explicitly label them `UNCERTAIN`; keep these separate
from the required confirmed-event visualization.

Render at the source resolution and preserve presentation timing and all decoded
frames, even if action analysis used a lower sampling rate. Run or propagate the
tracker at full frame rate for rendering. Distinguish short predicted/interpolated
boxes with dashed outlines, cap propagation gaps in seconds, and never interpolate
across scene cuts or unsupported occlusions. Cache this geometry separately from
the frozen event decisions so rendering does not change the reported events.

Use OpenCV for drawing and a timestamp-aware encoder such as PyAV/FFmpeg for MP4
export; document the tested dependencies and encoder settings. Target H.264 with
`yuv420p` for convenient playback. If odd source dimensions require padding, add
at most one row/column and record it. Preserve variable frame timing rather than
silently exporting variable-frame-rate input at a nominal constant frame rate.
Save rendering configuration and font assets/versions for reproducibility.

Verify every exported MP4 can be fully decoded, and visually inspect the first
and last frame plus frames around each interaction's start/end and an occlusion
where available. Check legibility, correct entity association, and simultaneous
events. Full annotated clips are mandatory; short event excerpts are optional.

## 5. Machine-readable output contract

Use one JSON document per clip and an aggregate manifest. A clip with no detected
interactions still gets a successful record with `interactions: []`. Decode or
inference failure must be reported as an error, not as a negative clip.

Illustrative values below are not results from the supplied videos:

```json
{
  "schema_version": "1.0",
  "clip_id": "example_clip",
  "source_sha256": "<input checksum>",
  "status": "ok",
  "interactions": [
    {
      "event_id": "e001",
      "type": "enter",
      "persons": [
        {"person_id": "p02", "description": "Person in a dark jacket and light trousers"}
      ],
      "vehicle": {
        "vehicle_id": "v01",
        "description": "Light-colored passenger car on the right"
      },
      "spans": [{"start_s": 2.4, "end_s": 4.8}],
      "truncated_start": false,
      "truncated_end": false,
      "evidence_frames": [72, 96, 120],
      "group_id": null
    }
  ]
}
```

Time spans are half-open `[start_s, end_s)` relative to the clip's first displayed
frame. Evidence indices are zero-based decoded-frame indices. Validate that all
spans are ordered, nonoverlapping within an event, nonempty, and within duration;
IDs resolve within the clip and every required description is present. Save model,
environment, seed, prompt/config hashes, and run timing in a companion run manifest.

## 6. KPI definitions and scoring protocol — 2–3 hours

**Matching comes first.** Map predicted participant tracks to reference entities
using annotated boxes/keyframes, with identity ambiguities reviewed independently
of action labels. Local numeric IDs need not equal annotation IDs. A wrong person
or vehicle must fail the match. For event spans A and B, temporal IoU is
`duration(A intersection B) / duration(A union B)`, using the union of visible
spans rather than the enclosing interval. Build eligible matches within the same
clip and pair with temporal IoU >= 0.5. Use maximum-cardinality one-to-one matching,
breaking ties by total IoU. Each event can match at most once; duplicates become
false positives. Unmatched predictions are FP and unmatched references are FN.

The following are **provisional development goals**, not achieved scores or a
promise about these clips. Freeze thresholds and acceptance goals before the
final test. Always publish numerators/denominators with rates.

| KPI | Definition | Initial goal / interpretation |
|---|---|---|
| Event precision | `TP / (TP + FP)` under pair-correct matching at tIoU >= 0.5. Action subtype is ignored for this primary detection metric. | >= 0.90; penalizes fabricated interactions. |
| Event recall | `TP / (TP + FN)` under the same matching. | >= 0.85; penalizes missed interactions. |
| Event F1 — primary summary | `2TP / (2TP + FP + FN)`, micro-aggregated across clips. | >= 0.87, alongside precision and recall separately. |
| Type-aware event F1 | Repeat matching but also require the same action label. Report per-label support and F1. | Diagnostic; prioritize entry/exit and report their confusion counts. |
| Candidate recall | Fraction of reference events with a candidate window covering >= 50% of reference visible duration for the correct pair. | >= 0.95; a proposal bottleneck limits all later recall. |
| False alarms per minute | Unmatched predicted events / total evaluated video minutes. | <= 0.2/min initially; report raw FP because the corpus is small. |
| Passerby false-positive rate | Annotated noninteracting person–vehicle encounters receiving >= 1 predicted interaction / all such encounters. An encounter is a continuous near-vehicle episode under the frozen annotation policy. | <= 0.05; specifically tests the task's main negative case. |
| Boundary error | For matched events, absolute start and end errors in seconds, reported separately as median and P90; score internal span boundaries separately if present. | Median <= 0.5 s, P90 <= 1.0 s; report truncated events separately. |
| Description correctness | Human-reviewed correct, visually supported attribute claims / all asserted attribute claims, for matched entities. | >= 0.95; no invented attributes. |
| Description usefulness | Fraction of matched events where descriptions allow a reviewer to locate both entities; unclear cases explicitly count as not useful. | >= 0.90; prevents empty descriptions from gaming correctness. |
| Completion and schema validity | Successful clip outputs / input clips; schema-valid outputs / emitted outputs. | Both 100%; explicitly report failure reasons. |
| Repeatability | Identical canonical event outputs over two runs with identical inputs/config/environment, ignoring runtime metadata. | 100% in the documented environment; investigate any mismatch. |
| Runtime and memory | Wall-clock seconds / video seconds (real-time factor), total runtime, peak RAM/VRAM, and hardware. Separate model load from inference. | Report measured values; no real-time requirement. |
| Annotated-video coverage | Successfully exported and fully decodable annotated MP4s / input clips. | 100% (8/8). |
| Overlay agreement | Frames whose rendered active event-ID set equals the event JSON's active set / all decoded source frames, checked from renderer metadata. | 100%; this checks rendering consistency, not model accuracy. |
| Video timing fidelity | Decoded output frame count versus source; maximum corresponding-frame timestamp error and total duration difference. | No dropped/duplicated frames; timestamp/duration error <= one source-frame interval, allowing encoder time-base rounding. |
| Visual QA pass rate | Reviewed frames with readable labels, correct displayed IDs, and no stale boxes / all reviewed frames. Include each event's boundaries and representative occlusions. | 100% after rendering fixes; record reviewed frame IDs. |

Report undefined metrics as `N/A`, with counts, rather than silently substituting
perfect performance. Abstaining on an otherwise evaluable true interaction counts
as a miss in the confirmed-output evaluation; report uncertain-candidate counts
separately. Run temporal-IoU sensitivity at 0.3 and 0.7 to expose boundary effects.
Do not optimize frame accuracy: long periods with no interaction can make it
misleading. Detection mAP and tracking identity switches are optional debugging
metrics, not substitutes for end-to-end event quality.

Compare the geometry-only baseline to the temporal verifier on the same development
split. Inspect false positives, missed events, wrong pairs, action confusions,
boundary errors, and unsupported descriptions. Tune only a few documented settings.
Freeze the configuration, run the held-out groups once, then generate outputs for
all clips with that frozen configuration. Do not claim statistical certainty from
a handful of events; publish per-clip tables and exact counts.

## 7. Repository and reproducibility — 2 hours

Proposed structure:

```text
README.md
pyproject.toml
requirements.lock
configs/default.yaml
src/person_vehicle/{cli,video,tracking,candidates,verify,events,evaluate,render}.py
schemas/clip_output.schema.json
annotations/{policy.md,events.json,negatives.json,split.json}
prompts/verify.txt
outputs/{clips/,tracks/,annotated/,manifest.json,metrics.json,render_qa.json}
tests/
docs/{writeup.md,ambiguities.md}
```

Target run interface (to implement):

```powershell
python -m person_vehicle run --input "C:\black rover\Assignment26\Videos" --output outputs --config configs/default.yaml --annotate
python -m person_vehicle render --input "C:\black rover\Assignment26\Videos" --events outputs/clips --tracks outputs/tracks --output outputs/annotated
python -m person_vehicle evaluate --pred outputs --reference annotations/events.json --split annotations/split.json
```

Document the tested Python version, installation commands, model download URL and
checksum, hardware, exact prompts, tracker settings, sampling rates, and thresholds.
Fix Python/NumPy/PyTorch seeds, request deterministic kernels where available,
sort file processing order, and record remaining hardware-dependent limitations.
Support rerunning from raw clips as well as resuming from hashed caches.

Tests should cover timestamp conversion, multi-span IoU, one-to-one matching,
duplicate penalties, empty outputs, schema validation, tracker reset, malformed
model output, and decode failure. Include a small smoke run and two reproducibility
runs. Test overlay activation at exact boundaries, multi-span gaps, simultaneous
events, and clips without interactions. Validate annotated-video timing and run
the visual QA described above. Measure rendering time separately from inference.

Create a public GitHub repository during implementation and include the code,
machine-readable outputs, annotated videos (or a linked shared folder), metrics,
and write-up. Include a manifest mapping each clip ID to its JSON and annotated
video, with checksums. Document how to supply the videos;
do not assume permission to republish the source footage. Include model/library
license and external asset details. Verify the README from a clean environment
and make sure the final repository/shared-output links work without private access.
Publishing is a future implementation deliverable, not part of this planning step.

## 8. Two-page write-up and stopping rule

Keep the submitted write-up to two pages; this planning document is separate.

- **Page 1:** problem definition; compact pipeline; annotation policy and key
  ambiguity decisions; models and implementation choices.
- **Page 2:** KPI table with counts and split; representative failure cases;
  assumptions, limitations, reproducibility details, and next steps.

Main expected limitations: occluded doors/passengers, tiny people, track breaks,
camera cuts, ambiguous multiple vehicles, and model hallucinations. Next steps
would be more independent annotations, broader held-out scenes, better door/body
transition cues, and targeted action-model training if enough data becomes available.

Estimated scope: **13–18 focused hours**, plus model download/inference/rendering time.
Stop after one baseline, one temporal-verification variant, documented error
analysis, and a reproducible final run. The minimum completed submission must
account for all eight clips, emit valid required fields and eight annotated MP4s,
document every material
ambiguity, report honest measured KPIs, and include usable repository instructions.
