# Person–vehicle interaction: brief write-up

## Approach

The implementation accepts a clip or directory and emits validated per-clip JSON
and annotated MP4s. PyAV decodes presentation timestamps; YOLO11s and BoT-SORT
provide within-clip person/vehicle tracks. Nearby pairs generate overlapping
temporal windows with context. Qwen2.5-VL-7B-Instruct, locally quantized to 4-bit,
examines sixteen chronological frames from a fixed pair crop as video input.
It returns an action, visible descriptions, and evidence frame indices. Strict
validation rejects invented frame indices and inconsistent decisions; uncertain
cases remain in a separate review artifact. Overlapping duplicate events for the
same pair/action are merged. A geometry-only appearance/disappearance baseline
is included for comparison.

The renderer uses finalized JSON and per-frame tracks. It shows local IDs, active
event labels, participant descriptions, clip time, and a timeline. It does not
hold stale boxes through missed detections. H.264 exports preserve original
presentation timing; every export is decoded again to check frame count and timing.

## Assumptions and decisions

An interaction requires directed action, not proximity. Labels cover entry, exit,
door operation, loading/unloading, and other contact such as removing a car cover.
Associated door closure is included in entry/exit. Walking behind a vehicle is not
evidence of entry. Clip-boundary actions are marked truncated. Descriptions use
visible appearance rather than identity or demographics. Each event has one
person–vehicle pair; identities reset between clips. Time spans are half-open.

The input inventory contains eight clips, approximately 128 seconds in total,
with resolutions from 352×288 to 3840×2160 and differing frame rates. A low-resolution
monochrome clip exposed a detector failure caused by excessive upscaling, so the
inference size is capped according to source dimensions. The initial 3B verifier
produced inconsistent JSON and action confusions; the retained configuration uses
7B with local video input and focused crops. No video is sent to an external API.

## Evaluation and reproducibility

The primary semantic KPI is pair-correct event F1 at temporal IoU 0.5, using
one-to-one matching so duplicates count as false positives. The evaluator also
reports precision/recall, type-aware results, IoU sensitivity, boundary errors,
false alarms per minute, and optional proposal/passerby metrics. Undefined rates
are null. Artifact completion, schema validity, and video timing are measured
separately from semantic accuracy.

Only one development clip currently has an assistant-reviewed reference and
entity mapping. Its results are a development sanity check, not independent
test performance. The other clips require reference adjudication before reporting
full-corpus precision/recall/F1. Provisional KPI goals in the plan are not achieved
scores. Machine-readable run results and limitations accompany the outputs.

Packages, model revisions/checksums, configuration, and prompts are recorded.
Seeds, sorted input order, deterministic sampling, and greedy decoding reduce
variation; GPU determinism is not guaranteed across hardware. Caches are keyed
by inputs/configuration/evidence. Tests cover event matching, validation, candidate
generation, multi-span boundaries, and variable-frame-rate rendering.

## Limitations and next steps

Tracking fragments can split one real entity; detector misses prevent proposals.
The VLM can confuse entry with door operation or hallucinate contact. Sampled-frame
boundaries are approximate. Current inference does not include the proposed global
VLM rescue scan or a separate dense boundary-refinement pass. Full reference
annotation, negative-encounter labels, and description review take priority next,
followed by track-fragment association and targeted temporal-action improvements.
Generated predictions remain separate from reference annotations and manual review.
