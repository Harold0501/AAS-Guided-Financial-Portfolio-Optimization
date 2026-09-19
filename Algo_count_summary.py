import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# * This code is used for generating the graph about the algorithm selection count used in the text

def summary():
    folders = ["RF", "MLP"]

    criteria = ["CR", "AR", "SHR", "SOR", "MDD"]

    for folder in folders:
        for criterion in criteria:
            VBS = pd.read_csv(
                folder
                + "_combined/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_vbs_algo_counts.csv"
            )

            Fin = pd.read_csv(
                folder
                + "_combined/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_Fin_algo_counts.csv"
            )

            Fin_GPT = pd.read_csv(
                folder
                + "_combined/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_Fin+GPT_algo_counts.csv"
            )

            Fin_detail = pd.read_csv(
                folder
                + "_combined/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_Fin+detail_algo_counts.csv"
            )

            Fin_formula = pd.read_csv(
                folder
                + "_combined/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_Fin+formula_algo_counts.csv"
            )

            Fin_GPT_detail = pd.read_csv(
                folder
                + "_combined/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_Fin+GPT+detail_algo_counts.csv"
            )

            Fin_GPT_formular = pd.read_csv(
                folder
                + "_combined/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_Fin+GPT+formula_algo_counts.csv"
            )

            # Add feature-set labels
            VBS["FeatureSet"] = "VBS"
            Fin["FeatureSet"] = "Fin"
            Fin_GPT["FeatureSet"] = "Fin+GPT"
            Fin_detail["FeatureSet"] = "Fin+detail"
            Fin_formula["FeatureSet"] = "Fin+formula"
            Fin_GPT_detail["FeatureSet"] = "Fin+GPT+detail"
            Fin_GPT_formular["FeatureSet"] = "Fin+GPT+formular"

            # Merge all four dataframes
            combined = pd.concat(
                [VBS, Fin, Fin_GPT, Fin_detail, Fin_formula, Fin_GPT_detail, Fin_GPT_formular],
                ignore_index=True
            )

            combined.to_csv(f"Algo_count/{folder}_{criterion}_strategy_selection_algo_counts_summary.csv",index=False)

            # Get all algorithms
            all_algorithms = sorted(combined["Algorithm"].unique())

            # Ensure every FeatureSet contains every Algorithm, fill missing Count with 0
            full_index = pd.MultiIndex.from_product(
                [all_algorithms, ["VBS", "Fin", "Fin+GPT", "Fin+detail", "Fin+formula", "Fin+GPT+detail", "Fin+GPT+formular"]],
                names=["Algorithm", "FeatureSet"]
            )
            full_df = pd.DataFrame(index=full_index).reset_index()

            combined_full = full_df.merge(
                combined, on=["Algorithm", "FeatureSet"], how="left"
            )
            combined_full["Count"] = combined_full["Count"].fillna(0).astype(int)

            # Plot style
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 6))

            # Grouped bar chart
            ax = sns.barplot(
                data=combined_full,
                x="Algorithm",
                y="Count",
                hue="FeatureSet",
                palette="Set2"
            )

            for container in ax.containers:
                ax.bar_label(container, fmt="%d", padding=3, fontsize=9)

            # Labels and title
            ax.set_title(f"{folder} {criterion} Algorithm Selection Counts across Feature Sets", fontsize=14)
            ax.set_xlabel("Algorithm")
            ax.set_ylabel("Selection Count")
            ax.legend(title="Feature Set")

            # Rotate x-axis labels
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()
            plt.savefig(
                "Algo_count_plot/" 
                + folder
                + "_"
                + criterion
                + "_strategy_selection_algo_counts_summary.png"
            )
            plt.close()

if __name__ == "__main__":
    summary()