# Person–vehicle interaction: brief write-up

## Approach

The system accepts one MP4 or a directory of independent clips and emits
schema-validated JSON interactions, concise descriptions, reports, Live Review,
and optional annotated MP4s. Experiment 40 is the submitted configuration.

YOLO26x detects people and vehicles. BoT-SORT creates clip-local tracks, while a
second YOLO pass on suspicious motion crops near vehicles recovers small or
partially occluded people. Nearby person–vehicle tracks create temporal
candidates. Qwen2.5-VL-7B-Instruct examines chronologically sampled frames from a
focused pair crop and decides whether an interaction exists and, if so, whether
it is entry, exit, load/unload, or another interaction.

Post-processing compensates for cases in which the VLM understands that contact
occurred but misreads its direction. It uses motion before and after the candidate,
disappearance near a vehicle, temporal continuity, and exclusive person–vehicle
assignment. Door-operation decisions are normalized to load/unload for this task.
Conservative appearance-change splitting avoids merging visibly different people
under one tracker ID. The Emergence rule recovers a person who is initially visible
only as a small region at a vehicle and then becomes larger while moving away;
this provides evidence for an exit whose beginning was occluded.

The final event description is aligned with the post-processed class, so the
machine-readable sentence and action field cannot disagree. Detector/tracker and
raw-person results are cached when their inputs and configuration match.

## Assumptions and ambiguity policy

An interaction requires directed activity with a vehicle, rather than proximity.
Walking past or behind a car is not an interaction. Entry and exit include the
associated door action; carrying items to or from a vehicle is load/unload. Any
other deliberate handling of a vehicle, such as removing a cover, is labeled
`other_interaction`. Descriptions use visible clothing and vehicle appearance,
without inferring identity or sensitive demographics. IDs are local to each clip,
and time spans are half-open.

Action boundaries are inherently ambiguous: reasonable annotators may disagree
about whether entry starts on approach, at door opening, or when the body crosses
the door. The manual labels use a relatively strict convention, while the project
goal is to recover the existence of as many real interactions as possible. We
therefore accepted some false positives, imperfect boundaries, and occasional
type errors in exchange for recall.

## Evaluation and reproducibility

The evaluator reports TP, FP, FN, precision, recall, and F1 with one-to-one event
matching. It provides both type-aware evaluation and binary interaction evaluation,
where detecting an interaction counts even if its subtype is wrong. Temporal IoU
and occupancy are also reported at several overlap thresholds. Because interaction
boundaries and even the scope of “interaction with a vehicle” are not uniquely
defined, these KPIs are primarily comparative indicators rather than absolute
ground truth. Some variants matched GT duration more closely; Experiment 40 was
selected because it prioritized finding interactions over maximizing temporal IoU.

The repository pins Python packages, model revisions, configuration, prompts, and
seed. Inputs are processed in sorted order and VLM decoding is greedy. GPU kernels
can still vary across hardware. The source videos are supplied separately by the
assignment and are not redistributed. `tools/quickstart.py` downloads the public
model assets and reproduces the complete Experiment 40 sequence, including its
post-processing and reports.

## Limitations and next steps

The small dataset makes threshold and branch selection vulnerable to overfitting;
a larger dataset with a fixed train/development/test split is the highest priority.
Human detection remains weaker than vehicle detection, especially during severe
occlusion, and should be evaluated with a dedicated person detector. Entry/exit
would benefit from an explicit temporal state model covering approach, door use,
disappearance, emergence, and departure. A vehicle-focused VLM or video model could
better distinguish loading from unloading, recognize door state, and reject
passers-by. Further work should also compare frame sampling strategies, separate
binary interaction detection from subtype classification, use adaptive temporal
expansion, automate error attribution by pipeline stage, and report runtime, GPU
memory, and VLM-call cost alongside accuracy.
