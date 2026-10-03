# Transformer algorithm-selection experiments

记录范围：2026-09-29 至 2026-10-01。除特别说明外，使用 762 instances、23 asset families 和 grouped 5-fold。

表示提取：MS-T encoder 256 维 + OH-T encoder 256 维，训练目标仍是原 PO 的 portfolio-weight objective。1-repeat 得到 762 行、512 个特征，约 6.7 分钟；后续 5 个 representation seeds 全部覆盖 762 instances。输出：`experiments/transformer_representation_v1/`。

## 1. Transformer representation + selector

### MLP / RF，seed 42

- 原始 512 维表示整体容易过拟合。
- MLP 的 `Fin+TransformerPCA32` 相对 Fin：
  - AR +0.0076
  - CR +0.0537
  - MDD +0.0088
  - SHR +0.0140
  - SOR +0.0096
- RF 整体弱于 MLP；OH-T 表示优于 MS-T。

### RF 组合筛选

5 个 representation seeds、RF seed 42：

- MDD：`Fin+OH-T-Raw`，平均 +0.00266，5/5 提升；
- SHR：`Fin+OH-T-PCA32`，平均 +0.01288，4/5 提升；
- SOR：`Fin+OH-T-PCA32`，平均 +0.01998，5/5 提升；
- AR、CR：Transformer 组合均未提升。

固定 `Fin+OH-T-PCA32`，5 representation seeds × 5 RF seeds：

- AR：-0.00837；
- CR：-0.13007；
- MDD：+0.00118，24/25 提升；
- SHR：+0.00978，20/25 提升；
- SOR：+0.00894，18/25 提升。

输出：

- `experiments/transformer_representation_v1/rf_stage12/`
- `experiments/transformer_representation_v1/repeated_representations/`

## 2. 原 PO Transformer + AS head + RF

- 保持原 PO 的 6 层、256 维 Transformer；
- A：直接训练 9 类 AS head；
- C_MS-T/C_OH-T：先用原 PO portfolio-weight objective 预训练，再冻结 encoder 训练 AS head；
- AS head 丢弃，只把 encoder 的 256 维表示交给 RF；
- 按 asset family 内时间切分，测试集共 16 个 instances，不用于 unseen-family 泛化结论。

RF 测试结果：

- A_AS：AR 0.2099，CR 1.2283，MDD -0.1675，SHR 0.8197，SOR 1.4780；
- C_MS-T：AR 0.2353，CR 1.1684，MDD -0.1648，SHR 0.8129，SOR 1.5161；
- C_OH-T：AR 0.2273，CR 1.2573，MDD -0.1653，SHR 0.7573，SOR 1.2878。

输出：`experiments/po_as_rf_v1/`

## 3. AS-Adapter-Grouped（无 test AS 标签）

- grouped 5-fold；
- train assets：OH-T PO 预训练后，用 AS labels 微调共享 body、shared 9-class head 和 train input adapters；
- test assets：仅用 OH-T PO target 适配 I/O adapters；test AS labels 使用数为 0；
- 丢弃 AS head，提取 256 维给 RF。

`Fin+EmbeddingPCA32` 相对 Fin：

- AR -0.01176；
- CR -0.31280；
- MDD +0.00393；
- SHR -0.02275；
- SOR -0.00669。

结论：仅 MDD 改善，其余指标没有提升。

输出：`experiments/as_adapter_grouped_v1/outputs/`

## 4. K=3 cluster-specific shared PO body + per-asset I/O adapters

- grouped 5-fold；聚类仅在 train fold 拟合；
- 每个 asset 保留独立 input adapter/portfolio head，共享 6 层、256 维 PO Transformer body；
- K=3 cluster bodies 使用 OH-T PO objective 微调；
- test asset 只做 PO-only I/O adapter adaptation，不使用 AS outcomes；
- 包含 global body、随机 cluster/body 和 Fin-only cluster RF 对照。

相对 global Fin RF：

- AR：cluster body + cluster-specific RF，+0.01418；对照均未提升；
- CR：仅 Fin-only cluster RF 提升 +0.06667，cluster body 未提升；
- MDD：cluster body + global RF 提升 +0.00273；随机 body 提升 +0.00127；
- SHR、SOR：均未提升。

输出：`experiments/cluster_po_adapter_v1/outputs/`

## 5. AS-Adapter-Grouped + cluster PO 微调

- 协议同第 3 节：grouped 5-fold，AS labels 仅 train assets，test I/O 只用 OH-T PO target；`test_label_instances_used=0`。
- AS-trained body 之后，train-fold K=3 聚类，再走两条路径：
  1. `body_ft`：复制 AS body，每簇小学习率 1e-5 全量 PO 微调 5 epoch；
  2. `residual`：冻结 AS body，每簇训 256→32→256 residual adapter。
- 对照：`as_only`、随机簇 `random_body_ft` / `random_residual`；RF 为 Fin 与 Fin+PCA32。

`Fin+EmbeddingPCA32` 相对 Fin：

| 表示 | AR | CR | MDD | SHR | SOR |
|---|---|---|---|---|---|
| as_only | -0.02491 | -0.26423 | +0.00109 | -0.06298 | -0.09111 |
| body_ft | -0.03534 | -0.27503 | -0.00160 | -0.03902 | -0.10720 |
| residual | -0.03343 | -0.24108 | +0.00155 | -0.06345 | -0.11365 |
| random_body_ft | -0.04295 | -0.20562 | +0.00110 | -0.06437 | -0.12600 |
| random_residual | -0.02774 | -0.18322 | +0.00093 | -0.06377 | -0.20061 |

结论：两条 cluster PO 路径都没有超过 Fin。相对 `as_only`，`body_ft` 只改善 SHR，`residual` 只轻微改善 MDD/CR；随机簇在 CR 上更好，说明聚类本身不稳定。

输出：`experiments/as_cluster_po_v1/outputs/`

## 6. Leaky oracle diagnostic（仅诊断，不是有效测试）

- 每个 asset 的 encoder 使用全部 AS/VBS 标签训练，包括 grouped RF test-fold 标签；
- A 从头训练；C 先做 MS-T/OH-T PO 预训练，再解冻整个 encoder 做 AS fine-tuning；
- grouped RF 覆盖 762 instances；结果不可用于泛化声明。

各指标最佳配置及相对 RF Fin：

- AR：C_MS-T + Fin + PCA32，0.2439（-0.0038）；
- CR：A_AS + Fin + PCA32，3.0271（+0.0205）；
- MDD：C_MS-T + Fin + PCA32，-0.2813（+0.0044）；
- SHR：C_MS-T + Fin + Raw，0.6787（+0.0013）；
- SOR：A_AS + Fin + PCA32，1.1246（-0.0272）。

输出：`experiments/po_as_rf_v1/leaky_oracle_diagnostic/`

## 当前结论

- 最稳定的 RF 配置：MDD 使用 `Fin+OH-T-Raw`；SHR/SOR 使用 `Fin+OH-T-PCA32`。
- Transformer 表示暂未改善 AR/CR。
- AS 后再做 cluster PO 全量微调或 residual adapter，均未超过 Fin。
- 正式论文结果仍需 nested configuration selection 和多 seed 显著性检验。
