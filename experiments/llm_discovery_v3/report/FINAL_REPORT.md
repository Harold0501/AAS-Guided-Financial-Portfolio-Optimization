# Experiments A, B, and C — Final Report

## Outcome

The reliability-centered discovery experiment succeeded. Qwen3-32B produced 100 accepted
reconstruction contracts across ten independent runs. Twenty consensus features were frozen
without using selector outcomes, computed deterministically for all 762 instances, and evaluated
under five-fold asset-family shift with RF and MLP selectors.

All ten frozen winner comparisons improved over the financial baseline in all five selector
seeds. Nine remained significant after Holm correction; MLP CR was positive in every seed but
uncertain after correction.

| Selector | Criterion | Frozen winner | Mean improvement | Positive seeds | Holm p |
|---|---|---:|---:|---:|---:|
| RF | AR | Fin+A+C | 0.03195 | 5/5 | 0.0023 |
| RF | CR | Fin+A+B | 0.25124 | 5/5 | 0.0012 |
| RF | MDD | Fin+A | 0.00346 | 5/5 | 0.0167 |
| RF | SHR | Fin+A+C | 0.01569 | 5/5 | 0.0232 |
| RF | SOR | Fin+A+C | 0.04612 | 5/5 | 0.0060 |
| MLP | AR | Fin+A+B | 0.05187 | 5/5 | 0.0043 |
| MLP | CR | Fin+GPT+A+C | 0.20562 | 5/5 | 0.0788 |
| MLP | MDD | Fin+A+B+C | 0.00750 | 5/5 | 0.0276 |
| MLP | SHR | Fin+A+B | 0.06717 | 5/5 | 0.0197 |
| MLP | SOR | Fin+A+B | 0.12915 | 5/5 | 0.0199 |

Higher selection outcome is better for every normalized criterion, including MDD.

## Arms

- **A:** the preserved 17-feature FormulaV2 comparator. Its general formula-generation mechanism
  is not claimed as the central novelty.
- **B:** blind, outcome-free discovery from development-only raw-data summaries. Ten consensus
  characteristics were reconstructed by Python.
- **C:** outcome-free discovery conditioned on a frozen traditional/GPT feature basis. Ten
  complementary characteristics were reconstructed by Python.

The full single-seed screen covered `Fin` and `Fin+GPT`, each crossed with A/B/C (16 configurations),
five criteria, and both selectors. Confirmation used five seeds for the baseline and one frozen
winner per selector/criterion.

## Reproducibility and reliability

- 762 instances, 23 asset families, 100% raw-data coverage.
- Discovery used 18 development families; five families were hidden from the LLM.
- No algorithm outcomes, selector scores, final-test feedback, or future returns entered prompts.
- Five generations per arm; 50 accepted contracts per arm.
- Six invalid API outputs were retained (truncation/count/schema failures), not silently deleted.
- Every request, raw response, parsed response, seed, model ID, usage record, prompt, split,
  formula, feature value, model result, and dependency version is checkpointed.
- All 20 computed features are finite and nonmissing for all instances.
- Determinism and asset/row-order invariance test passed.
- Original FormulaV2 and repository inputs were not modified.

## Interpretation

B/C are useful mainly as complementary representations. The strongest arms generally combine
them with A rather than replacing A. Blind discovery contributes most often through B in MLP;
basis-aware C is prominent in RF and in MLP CR. This supports the controlled comparison requested
by the professor rather than a claim that one prompting method universally dominates.

## Novelty boundary

Do not claim that LLM formula generation, executable feature code, or data-conditioned feature
engineering is new. The defensible contribution is their combination in a reliability-centered
portfolio algorithm-selection setting:

> Reproducible, outcome-blind LLM discovery of complementary portfolio-instance representations
> for robust algorithm selection under asset-family distribution shift.

The differentiating experimental evidence is: blind versus frozen-basis-aware discovery;
outcome isolation; deterministic reconstruction; explicit invalid-output accounting; repeated
generation and selector-seed reliability; and unseen asset-family evaluation. This is distinct
from stock-level alpha generation and from AutoFE methods optimized through downstream feedback.

## Limitations before submission

The screen-and-confirm winner was selected from the same outer grouped evaluation, so it should
be described as a two-stage empirical analysis, not a pristine untouched-test estimate. For a
conference-grade claim, add nested group selection or an additional external asset-family dataset.
The current evidence is strong and suitable for a credible workshop submission when reported
with this limitation. No transformer experiment is needed before analyzing and writing these results.
