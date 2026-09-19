import pandas as pd
import numpy as np

# * This code is used to merge the result of different feature set selection result
# * so that a boxplot is available

def Merge():
    criteria = ["CR", "AR", "SHR", "SOR", "MDD"]

    folders = ["RF_combined", "MLP_combined"]

    for folder in folders:
        for criterion in criteria:

            Fin = pd.read_csv(
                folder + "/" + criterion + "/" + criterion + "_strategy_selection_Fin.csv"
            )

            Fin_GPT = pd.read_csv(
                folder + "/" + criterion + "/" + criterion + "_strategy_selection_Fin+GPT.csv"
            )

            Fin_detail = pd.read_csv(
                folder + "/" + criterion + "/" + criterion + "_strategy_selection_Fin+detail.csv"
            )

            Fin_formula = pd.read_csv(
                folder + "/" + criterion + "/" + criterion + "_strategy_selection_Fin+formula.csv"
            )

            Fin_GPT_detail = pd.read_csv(
                folder + "/" + criterion + "/" + criterion + "_strategy_selection_Fin+GPT+detail.csv"
            )

            Fin_GPT_formula = pd.read_csv(
                folder + "/" + criterion + "/" + criterion + "_strategy_selection_Fin+GPT+formula.csv"
            )

            Fin = Fin.drop(columns=["Selected Algo", "VBS Algo"])
            Fin.rename(columns={"Selection Result": "Fin Selection Result"}, inplace=True)

            Fin_GPT = Fin_GPT.drop(columns=["Selected Algo", "SBS", "VBS", "VBS Algo"])
            Fin_GPT.rename(
                columns={"Selection Result": "Fin+GPT Selection Result"}, inplace=True
            )

            Fin_detail = Fin_detail.drop(columns=["Selected Algo", "SBS", "VBS", "VBS Algo"])
            Fin_detail.rename(
                columns={"Selection Result": "Fin+detail Selection Result"}, inplace=True
            )

            Fin_formula = Fin_formula.drop(columns=["Selected Algo", "SBS", "VBS", "VBS Algo"])
            Fin_formula.rename(
                columns={"Selection Result": "Fin+formula Selection Result"}, inplace=True
            )

            Fin_GPT_detail = Fin_GPT_detail.drop(columns=["Selected Algo", "SBS", "VBS", "VBS Algo"])
            Fin_GPT_detail.rename(
                columns={"Selection Result": "Fin+GPT+detail Selection Result"}, inplace=True
            )

            Fin_GPT_formula = Fin_GPT_formula.drop(columns=["Selected Algo", "SBS", "VBS", "VBS Algo"])
            Fin_GPT_formula.rename(
                columns={"Selection Result": "Fin+GPT+formula Selection Result"}, inplace=True
            )

            Out = pd.merge(Fin, Fin_GPT, on="Instance")
            Out = pd.merge(Out, Fin_detail, on="Instance")
            Out = pd.merge(Out, Fin_formula, on="Instance")
            Out = pd.merge(Out, Fin_GPT_detail, on="Instance")
            Out = pd.merge(Out, Fin_GPT_formula, on="Instance")

            Out.to_csv(
                folder + "/" + criterion + "/" + criterion + "_strategy_selection_summary.csv",
                index=False
            )

if __name__ == "__main__":
    Merge()