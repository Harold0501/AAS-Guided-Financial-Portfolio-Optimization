# Direct Transformer algorithm selector

This experiment trains one shared, permutation-invariant market encoder to
select the best portfolio algorithm directly.

## Inputs and targets

- Input: the final 240 trading days strictly before each decision date.
- Per-stock fields: relative OHLC, log-volume change, and adjusted log return.
- Cross-sectional aggregation: shared stock MLP followed by masked mean and
  standard-deviation pooling.
- Temporal model: two-layer, 64-dimensional Transformer.
- Known planning horizon: one-to-nine-year embedding.
- Targets: five heads (AR, CR, MDD, SHR, SOR), each classifying among the nine
  candidate algorithms using the observed virtual-best algorithm as label.

## Evaluation

The experiment uses five-fold asset-family grouped evaluation. No held-out
asset family contributes to training, validation, or normalization statistics.
Validation is also split by complete asset families.

```bash
python run_experiment.py prepare
CUDA_VISIBLE_DEVICES=0 python run_experiment.py worker --fold 0 --smoke
python run_experiment.py launch --gpus 0,1,2,3 --timeout-minutes 60
```

Final out-of-fold predictions and summaries are written beneath `outputs/`.
