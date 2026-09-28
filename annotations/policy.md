# Reference annotation policy

Follow the action definitions and half-open time spans in `../PLAN.md`. Record one
person–vehicle pair per event. Entry/exit absorbs immediately associated door
operation. Proximity, walking past, and unexplained occlusion are negatives.

Reference annotations must be visually reviewed independently of predictions.
Store them as a JSON object keyed by clip ID, each value containing `duration_s`
and `interactions` in the event schema. Assign reference IDs and provide a separate
predicted-to-reference entity mapping for evaluation. Do not copy predictions
into reference files or use detector IDs as evidence of annotation correctness.

Unresolved ambiguous events belong in a separate review list, not the definitive
reference file. Annotate clear near-vehicle negative encounters in `negatives.json`
with pair IDs and time spans. Mark their source and reviewer explicitly.
