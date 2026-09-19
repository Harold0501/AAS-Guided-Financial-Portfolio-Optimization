# AAS-Guided Financial Portfolio Optimization

> Research code and artifacts for representation-aware automated algorithm
> selection in financial portfolio optimization.

This repository studies whether a learned selector can choose an appropriate
portfolio optimizer for each market instance. The central idea is to represent
an instance with financial, language-model-derived, or deterministically
reconstructed features; predict the performance of each candidate optimizer;
and deploy the optimizer with the best predicted outcome.

The repository is under active development toward an ICLR 2027 submission. It
contains the original exploratory pipeline, the results used in the workshop
study, and newer leakage-controlled experiments. Historical code and outputs
are intentionally retained for auditability.

## Research problem

For a portfolio instance described by features \(x\), let
\(\mathcal{A}=\{a_1,\ldots,a_9\}\) be the candidate optimizer portfolio. A
selector learns one performance model per candidate and returns

\[
\hat a(x)=\arg\max_{a\in\mathcal A}\widehat m_a(x),
\]

where \(m\) is the evaluation criterion. The current artifact contains:

| Component | Current scope |
| --- | --- |
| Instances | 762 portfolio instances from 23 asset families |
| Candidate algorithms | EW, MCW, mSSRM, MS-M, OH-M, MS-T, OH-T, MS-L, and OH-L |
| Selection criteria | Annualized Return (AR), Cumulative Return (CR), Maximum Drawdown (MDD), Sharpe Ratio (SHR), and Sortino Ratio (SOR) |
| Selectors | Random Forest and multilayer perceptron regressors |
| Representations | Hand-engineered financial descriptors, GPT/LLM features, formula-derived features, and controlled feature-discovery variants |
| Reference policies | Single Best Solver (SBS) and Virtual Best Solver (VBS) |

The repository is a research artifact, not an investment product. Nothing here
constitutes financial advice.

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | Runs the legacy combined RF/MLP analysis, aggregation, and plotting pipeline. |
| `RF*.py`, `MLP*.py` | Per-criterion selector experiments and feature-set comparisons. |
| `Global_merge.py`, `merge_csv.py` | Assemble per-instance and summary result tables. |
| `Boxplot.py`, `Algo_count_summary.py`, `Wilcoxon.py` | Visualization and statistical-analysis utilities. |
| `RF/`, `MLP/`, `RF_combined/`, `MLP_combined/` | Experiment inputs and preserved selector outputs. |
| `LLM/` | Prompt templates, feature-definition generation, and direct LLM feature experiments. |
| `LLMD-AS-FPO/CleanLoader/` | Market-data loading, portfolio construction, model training, and backtesting. |
| `LLMD-AS-FPO/LLM_Generated_Feature/` | Earlier LLM feature-generation studies and their data. |
| `LLMD-AS-FPO/news_method/` | GDELT-based news collection utilities. |
| `experiments/llm_formula_v2/` | Resumable formula-specification, deterministic reconstruction, ablation, and audit workflow. |
| `experiments/llm_discovery_v3/` | Frozen-arm, outcome-blind feature discovery with grouped robustness and multi-seed evaluation. |
| `Result_tables/`, `Boxplots/`, `Algo_count_plot/`, `wilcoxon/` | Preserved tables and figures. |

Several directories retain their historical names because scripts and notebooks
refer to them by relative path. Run legacy entry points from the repository root
unless a subdirectory README says otherwise.

## Setup

Python 3.11 is the reference version (`.python-version`). The root environment
is locked with `uv.lock`:

```bash
git clone https://github.com/SeanWan514/AAS-Guided-Financial-Portfolio-Optimization.git
cd AAS-Guided-Financial-Portfolio-Optimization
uv sync --frozen
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux or
`.venv\Scripts\activate` on Windows. The deep portfolio-learning code has
additional dependencies documented in
[`LLMD-AS-FPO/CleanLoader/requirements.txt`](LLMD-AS-FPO/CleanLoader/requirements.txt).

Some data-acquisition and LLM stages call external services. Copy `.env.example`
to `.env`, add only the credentials needed for the selected stage, and never
commit the resulting `.env` file.

## Reproducing the main workflows

### Leakage-controlled discovery workflow (recommended)

The V3 workspace is the current research path. It freezes its experimental arms
and data splits before feature discovery, checkpoints API calls, reconstructs
accepted features deterministically, and keeps final-test feedback out of
feature design.

```bash
cd experiments/llm_discovery_v3
python bootstrap.py
OPENROUTER_API_KEY=... python generate_discoveries.py --arm B
OPENROUTER_API_KEY=... python generate_discoveries.py --arm C
python compute_features.py
python run_factorial.py
python analyze.py
```

Read [`experiments/llm_discovery_v3/README.md`](experiments/llm_discovery_v3/README.md)
before changing prompts, splits, feature contracts, or evaluation settings.

### Formula reconstruction workflow

```bash
cd experiments/llm_formula_v2
python step0_audit.py
OPENROUTER_API_KEY=... python generate_specs.py
python validate_and_compute.py
python run_experiments.py
python analyze_results.py
python final_audit.py
```

This workflow stores checkpoints, manifests, logs, computed features, and
analysis artifacts beneath its own directory.

### Preserved baseline workflow

```bash
python main.py
```

This reproduces the historical 10-fold shuffled-CV RF/MLP comparisons and then
regenerates aggregate tables and plots. It is retained for provenance and
backward comparison. Its random-fold estimates should not be presented as
evidence of generalization to unseen asset families; use the frozen/grouped V3
evaluations for that claim.

### Portfolio-learning and backtest workflow

```bash
cd LLMD-AS-FPO/CleanLoader
python -m pip install -r requirements.txt
python load_data.py
python portfolio_optimization.py -f <asset-file>
python experiment.py -a <asset> -p <portfolio> -o <model> -b <backtest-start> -z sophia
python backtest.py
```

See the local [CleanLoader guide](LLMD-AS-FPO/CleanLoader/README.md) for argument
semantics and GPU examples.

## Reproducibility principles

- Treat instance identifiers as explicit join keys; never rely on row order.
- Freeze train/development/test partitions before LLM-assisted discovery.
- Do not expose future returns, optimizer outcomes, held-out families, or
  final-test feedback to feature-generation prompts.
- Preserve raw requests, responses, manifests, seeds, and result files needed to
  audit a reported number.
- Fit preprocessing only on the training partition.
- Report SBS and VBS alongside learned selectors, with paired uncertainty and
  multiple random seeds where applicable.
- Keep all credentials in environment variables.

## Tests and validation

The deterministic feature engines include focused tests:

```bash
python -m pytest -q \
  experiments/llm_formula_v2/tests \
  experiments/llm_discovery_v3/test_feature_engine.py
```

Before sharing new results, also run the applicable audit stage and verify that
its recorded input hashes match the data used by the experiment.

## Contributing

Collaborators should read [CONTRIBUTING.md](CONTRIBUTING.md) before modifying
experimental code or results. In particular, keep changes to frozen protocols
separate from result-generation commits and document any unavoidable deviation.

## Citation and license

The archival citation will be added when the ICLR manuscript is released. If
you use this repository before then, please cite the repository URL and the
specific commit hash.

No open-source license has yet been selected. Until a license is added, the
repository is publicly viewable but the authors retain all rights.
