import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

criteria = ["CR", "AR", "SHR", "SOR", "MDD"]
output_dir = Path(__file__).resolve().parent / "wilcoxon"

for criterion in criteria:
    print(f"Processing criterion: {criterion}")
    data = pd.read_csv(output_dir / f"All_{criterion}.csv")
    data = data.iloc[:, 38:49]

    # 获取列名
    columns = data.columns

    # 创建一个空的DataFrame来存储p-values
    wilcoxon_results = pd.DataFrame(index=columns, columns=columns, dtype=float)

    # 对每一对列进行Wilcoxon检验
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                # 对角线设为NaN或1
                wilcoxon_results.loc[col1, col2] = np.nan
            else:
                # 移除NaN值后进行配对检验
                paired_data = pd.DataFrame(
                    {"col1": data[col1], "col2": data[col2]}
                ).dropna()
                if len(paired_data) > 0:
                    try:
                        statistic, p_value = wilcoxon(
                            paired_data["col1"], paired_data["col2"]
                        )
                        wilcoxon_results.loc[col1, col2] = p_value
                    except:
                        wilcoxon_results.loc[col1, col2] = np.nan
                else:
                    wilcoxon_results.loc[col1, col2] = np.nan

    # 保存结果
    wilcoxon_results.to_csv(output_dir / f"wilcoxon_{criterion}.csv")
    print(wilcoxon_results)

    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        wilcoxon_results.astype(float),
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",  # 红色表示小p-value(显著),绿色表示大p-value(不显著)
        center=0.05,  # 以0.05为中心
        vmin=0,
        vmax=1,
        cbar_kws={"label": "p-value"},
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title(f"Wilcoxon Test P-values Heatmap ({criterion})", fontsize=16, pad=20)
    plt.xlabel("Algorithms", fontsize=12)
    plt.ylabel("Algorithms", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"wilcoxon_{criterion}_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
