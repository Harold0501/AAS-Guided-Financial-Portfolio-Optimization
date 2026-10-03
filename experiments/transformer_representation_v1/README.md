# MS-T + OH-T Transformer representations

This workspace performs a one-repeat pilot of method 2: extract the final
causal Transformer encoder state from separately trained Max-Sortino (MS-T)
and One-Hot (OH-T) models, then concatenate the two 256-dimensional states for
algorithm selection.

## Protocol

- The original six-layer Transformer architecture and Sophia optimizer are
  retained.
- Each asset family has its own MS-T and OH-T model, matching the historical
  portfolio pipeline.
- The model is trained in expanding-window monthly order with seed 0.
- A representation for decision month `t` uses market observations only
  through month `t-1`.
- The best validation state is deep-copied before representation extraction.
- One repeat is run for each of the 23 asset families and each target.
- Raw OHLCV inputs come from this repository. Preserved target portfolio
  weights are read from `workspace/sDFLwDE/new_result` and are not modified.
- All generated files remain under this experiment directory.

## Commands

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py worker --asset asset_22 --smoke
```

Full extraction, with a hard 85-minute deadline:

```bash
python run_experiment.py launch --gpus 0,1,2,3 --timeout-minutes 85
```

The final validated matrix is
`outputs/transformer_representations_1repeat.csv`, containing 762 rows and 512
representation columns.
