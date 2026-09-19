import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# * This code is used to generate the boxplot in the text

def boxplot():
    criteria = ["CR", "AR", "SHR", "SOR", "MDD"]

    folders = ["RF", "MLP"]

    for folder in folders:
        for criterion in criteria:
            # 读取数据
            df = pd.read_csv(
                folder + "_combined/" + criterion + "/" + criterion + "_strategy_selection_summary.csv"
            )

            # 整理长格式数据
            cols = [
                "SBS",
                "VBS",
                "Fin Selection Result",
                "Fin+GPT Selection Result",
                "Fin+detail Selection Result",
                "Fin+formula Selection Result",
                "Fin+GPT+detail Selection Result",
                "Fin+GPT+formula Selection Result",
            ]
            df_long = df[cols].melt(var_name="Method", value_name="Value")

            plt.figure(figsize=(12, 6))
            # 绘制箱线图（保留原 order 不变）
            order = [
                "SBS",
                "VBS",
                "Fin Selection Result",
                "Fin+GPT Selection Result",
                "Fin+detail Selection Result",
                "Fin+formula Selection Result",
                "Fin+GPT+detail Selection Result",
                "Fin+GPT+formula Selection Result",
            ]

            ax = sns.boxplot(
                x="Method",
                y="Value",
                data=df_long,
                order=order,
                palette="Set3",
                showfliers=False,
            )

            # 自定义标签（短名称）
            new_labels = ["SBS", "VBS", "Fin", "Fin+GPT", "Fin+detail", "Fin+formula", "Fin+GPT+detail", "Fin+GPT+formula"]
            ax.set_xticklabels(new_labels)

            plt.title(folder + " " + criterion + " Performance Distribution")
            plt.ylabel(criterion + " Value")
            plt.xlabel("")
            plt.tight_layout()
            plt.savefig(
                "Boxplots/" + folder + "_" + criterion + "_strategy_selection_summary.png"
            )
            plt.close()

if __name__ == "__main__":
    boxplot()