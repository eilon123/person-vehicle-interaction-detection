# Development sanity reference: not a held-out benchmark

`reference_dev.json` contains one visually reviewed clip, `mKzCQKTHizw_1`.
Reviewer: the coding assistant, using original-frame sheets exported from the
source video. This is not a second human annotation and not an independent test.
The same clip was used to diagnose verifier behavior during development.

The person in a light top and tan trousers reaches the open near-side front door,
turns into the cabin, sits inside, and closes the door. Start: first supported
entry/contact at approximately frame 155 (5.172 s). End: frame 201 timestamp
(6.707 s), after closure. Boundary interpretation has approximately 0.15 s
uncertainty. Approach from the bottom of the frame is excluded. Earlier opening
of the door has no sufficiently visible attributable person and is not separately
annotated. The two person track fragments were visually mapped to this same person.

Review material includes 0–4 s overview, 4–7 s dense samples, and original-rate
frames around 4.9–5.5 s and 6.4–6.85 s. Frame indices are preserved in filenames
and sheets under `outputs/manual_review` in the working checkout.

The other seven clips have review notes but do not yet have adjudicated reference
events or entity mappings. Full-corpus precision/recall/F1, passerby FP rate, and
description-accuracy claims must remain unmeasured until those labels exist.
The provisional split alone does not establish an independent benchmark.
