# Reliable LLM Portfolio-Instance Discovery (V3)

This additive workspace implements three frozen experimental arms:

- **A / FormulaV2:** preserved comparator generated without instance data.
- **B / Blind:** outcome-blind discovery from development-only raw-data summaries.
- **C / Basis-aware:** the same discovery context plus a frozen established feature basis.

The LLM never receives future returns, candidate-algorithm outcomes, selector scores,
held-out asset families, or final-test feedback. Every accepted feature must have a
complete reconstruction contract and is recomputed deterministically in Python.

## Safety and resume guarantees

- Original repository files are read-only inputs. All new files stay here.
- Every API request and response is checkpointed independently.
- Completed calls are skipped on restart.
- Feature and model outputs use temporary files followed by atomic replacement.
- `state.json` records the last completed stage and next action.
- Secrets are environment-only; `.env` and credential-like files are ignored.
- Input SHA-256 values and the frozen split are recorded before discovery.

## Order

1. `python bootstrap.py`
2. `OPENROUTER_API_KEY=... python generate_discoveries.py --arm B`
3. `OPENROUTER_API_KEY=... python generate_discoveries.py --arm C`
4. Contract review, independent reconstruction, feature freezing, factorial modeling,
   grouped robustness, multi-seed analysis, and final audit.

Do not use final-test results to revise prompts, contracts, or feature sets.
