import pandas as pd

for folder in ["RF","MLP"]:
    result = None
    for criterion in ["AR", "CR", "MDD", "SHR", "SOR"]:
        df = pd.read_csv(f"{folder}_{criterion}_strategy_selection_algo_counts_summary.csv")

        current = (
            df.groupby(["Algorithm", "FeatureSet"])["Count"]
            .sum()
            .reset_index()
        )

        current = current.sort_values(by=["Algorithm", "FeatureSet"])
        current.rename(columns={"Count": f"{criterion}", "FeatureSet": "Feature Sets"},inplace=True)

        if result is None:
            result = current
        else:
            result = result.merge(current,on=["Algorithm", "Feature Sets"],how="outer")
   

    if result is not None:
        result = result.fillna(0)
        result[["AR", "CR", "MDD", "SHR", "SOR"]] = result[
            ["AR", "CR", "MDD", "SHR", "SOR"]
        ].astype(int)
        result.to_csv(f"../Result_tables/{folder}_Algo_Count.csv", index=False)
    else:
        raise ValueError()