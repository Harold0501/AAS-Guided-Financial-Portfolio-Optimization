# LLM Formula V2 Experiment

Isolated, resumable Step 0/Step 1 pipeline for PO-AAS.

## Safety contract

- All source repository, manuscript, and historical result files are read-only.
- Every generated artifact is written beneath this directory.
- No API secrets are stored here.
- Scripts use explicit instance-key joins and refuse duplicate instance IDs.
- `state.json`, logs, manifests, and versioned outputs permit resumption.

## Stages

1. `step0_audit.py`: input manifest, schema checks, known-result reconstruction, summary-bug diagnosis.
2. `generate_specs.py`: OpenRouter formula-spec generation with raw-response checkpoints.
3. `validate_and_compute.py`: validate specifications and compute deterministic values from historical data.
4. `run_experiments.py`: RF/MLP comparisons and feature ablations.
5. `analyze.py`: corrected tables, paired effects, controls, and final decision report.

No transformer work is included. The project stops after Step 1 for review.
