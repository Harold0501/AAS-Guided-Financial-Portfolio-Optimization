# Contributing

This repository contains active research and preserved historical artifacts.
Changes must remain auditable and must not contaminate held-out evaluation.

## Before making a change

1. Create a focused branch and state the scientific question being tested.
2. Record the source data, split manifest, feature set, selector, metric, and
   random seeds affected by the change.
3. Do not revise frozen prompts, feature contracts, or data partitions after
   inspecting final-test performance.
4. Keep credentials in environment variables; never place keys in code,
   notebooks, logs, prompts, requests, or committed `.env` files.

## Code and data conventions

- Use paths relative to the repository or the experiment directory.
- Join tables on explicit keys and validate key uniqueness before merging.
- Write new generated artifacts beneath the experiment that produced them.
- Preserve raw model requests/responses and input hashes when they are required
  to reconstruct an experiment.
- Do not overwrite a published result silently. Use a new versioned output and
  document the reason for the change.
- Avoid renaming historical directories without updating and testing every
  dependent script and notebook.

## Validation

For feature-engine changes, run:

```bash
python -m pytest -q \
  experiments/llm_formula_v2/tests \
  experiments/llm_discovery_v3/test_feature_engine.py
```

For broader changes, rerun the applicable experiment audit and compare its
manifests, row counts, missingness, seeds, and output hashes. Include the exact
commands and environment details in the pull request.

## Pull requests

A useful pull request explains:

- what changed and why;
- whether the change affects a frozen protocol;
- which data and results were regenerated;
- what checks were run; and
- any limitations or unresolved discrepancies.

Do not mix code cleanup with experimental-result changes unless separating them
would make the audit trail less clear.
