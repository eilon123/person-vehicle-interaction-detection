# FINAL results

This directory contains the machine-readable outputs and reports selected from
Experiment 40 as the final submission result.

- `interactions.json`: flat list of all final interactions; every record includes
  `clip_id`, time spans, person descriptions, vehicle description, action type,
  and a concise natural-language description.
- `clips/`: complete schema-validated output, one JSON document per clip.
- `vlm_descriptions/`: one decisive sentence per final interaction, plus the raw
  VLM observation retained for audit.
- `review/`: detailed VLM candidate decisions.
- `manual_binary_metrics.json`: machine-readable KPI report.
- `kpi_dashboard.html`: readable KPI report.
- `live_review.html`: local interactive review. Regenerate it after cloning so
  its video paths point to the local input directory.
- `config.json`: exact Experiment 40 configuration.

Annotated MP4 files are intentionally distributed in the `FINAL_outputs.zip`
GitHub Release asset because binary videos do not belong in ordinary Git history.
The source videos are not redistributed.
