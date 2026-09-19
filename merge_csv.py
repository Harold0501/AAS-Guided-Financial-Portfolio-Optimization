import pandas as pd

# * This code is to concat the result csvs into a full table so that 

def read_result_csv(path, result_name):
    # 读取 CSV
    df = pd.read_csv(path)

    # 把第一列改名为 Factors
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Factors", "Selection Result": result_name})

    # 删除全空行
    df = df.dropna(how="all")

    # 删除 Factors 为空的行（比如最后的 ,,,）
    df = df[df["Factors"].notna()]

    # 只保留需要的列
    keep_cols = ["Factors"]
    if "SBS" in df.columns:
        keep_cols.append("SBS")
    if "VBS" in df.columns:
        keep_cols.append("VBS")
    keep_cols.append(result_name)

    df = df[keep_cols]

    return df


def merge_all_results(fin_path, gpt_path, detail_path, formula_path, gpt_detail_path, gpt_formula_path, criteria_name):
    df_fin = read_result_csv(fin_path, "Fin")
    df_gpt = read_result_csv(gpt_path, "Fin+GPT")
    df_detail = read_result_csv(detail_path, "Fin+detail")
    df_formula = read_result_csv(formula_path, "Fin+formula")
    df_gpt_detail = read_result_csv(gpt_detail_path, "Fin+GPT+detail")
    df_gpt_formula = read_result_csv(gpt_detail_path, "Fin+GPT+formula")

    # 以 Fin 为主表，保留 SBS 和 VBS
    df = df_fin.merge(df_gpt[["Factors", "Fin+GPT"]], on="Factors", how="inner")
    df = df.merge(df_detail[["Factors", "Fin+detail"]], on="Factors", how="inner")
    df = df.merge(df_formula[["Factors", "Fin+formula"]], on="Factors", how="inner")
    df = df.merge(df_gpt_detail[["Factors", "Fin+GPT+detail"]], on="Factors", how="inner")
    df = df.merge(df_gpt_formula[["Factors", "Fin+GPT+formula"]], on="Factors", how="inner")

    # 在第一列插入 Criteria
    df.insert(0, "Criteria", criteria_name)

    return df



def tables():
    folders = ["RF_combined", "MLP_combined"]
    for folder in folders:
        df_AR = merge_all_results(
            folder + "/AR/AR_strategy_selection_Fin_stats.csv",
            folder + "/AR/AR_strategy_selection_Fin+GPT_stats.csv",
            folder + "/AR/AR_strategy_selection_Fin+detail_stats.csv",
            folder + "/AR/AR_strategy_selection_Fin+formula_stats.csv",
            folder + "/AR/AR_strategy_selection_Fin+GPT+detail_stats.csv",
            folder + "/AR/AR_strategy_selection_Fin+GPT+formula_stats.csv",
            "AR"
        )

        print(df_AR)

        all_dfs = [df_AR]

        df_CR = merge_all_results(
            folder + "/CR/CR_strategy_selection_Fin_stats.csv",
            folder + "/CR/CR_strategy_selection_Fin+GPT_stats.csv",
            folder + "/CR/CR_strategy_selection_Fin+detail_stats.csv",
            folder + "/CR/CR_strategy_selection_Fin+formula_stats.csv",
            folder + "/CR/CR_strategy_selection_Fin+GPT+detail_stats.csv",
            folder + "/CR/CR_strategy_selection_Fin+GPT+formula_stats.csv",
            "CR"
        )

        all_dfs.append(df_CR)

        df_SHR = merge_all_results(
            folder + "/SHR/SHR_strategy_selection_Fin_stats.csv",
            folder + "/SHR/SHR_strategy_selection_Fin+GPT_stats.csv",
            folder + "/SHR/SHR_strategy_selection_Fin+detail_stats.csv",
            folder + "/SHR/SHR_strategy_selection_Fin+formula_stats.csv",
            folder + "/SHR/SHR_strategy_selection_Fin+GPT+detail_stats.csv",
            folder + "/SHR/SHR_strategy_selection_Fin+GPT+formula_stats.csv",
            "SHR"
        )

        all_dfs.append(df_SHR)

        df_SOR = merge_all_results(
            folder + "/SOR/SOR_strategy_selection_Fin_stats.csv",
            folder + "/SOR/SOR_strategy_selection_Fin+GPT_stats.csv",
            folder + "/SOR/SOR_strategy_selection_Fin+detail_stats.csv",
            folder + "/SOR/SOR_strategy_selection_Fin+formula_stats.csv",
            folder + "/SOR/SOR_strategy_selection_Fin+GPT+detail_stats.csv",
            folder + "/SOR/SOR_strategy_selection_Fin+GPT+formula_stats.csv",
            "SOR"
        )

        all_dfs.append(df_SOR)

        df_MDD = merge_all_results(
            folder + "/MDD/MDD_strategy_selection_Fin_stats.csv",
            folder + "/MDD/MDD_strategy_selection_Fin+GPT_stats.csv",
            folder + "/MDD/MDD_strategy_selection_Fin+detail_stats.csv",
            folder + "/MDD/MDD_strategy_selection_Fin+formula_stats.csv",
            folder + "/MDD/MDD_strategy_selection_Fin+GPT+detail_stats.csv",
            folder + "/MDD/MDD_strategy_selection_Fin+GPT+formula_stats.csv",
            "MDD"
        )

        all_dfs.append(df_MDD)


        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv("Tex/" + folder + "_merged_results.csv", index=False)

if __name__ == "__main__":
    tables()