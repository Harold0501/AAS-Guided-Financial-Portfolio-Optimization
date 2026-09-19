from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# * RF selection


def RF():
    folder = "RF_combined"
    criteria = ["CR", "AR", "SHR", "SOR", "MDD"]
    file_end = "_LLM_combined.csv"


    for criterion in criteria:
        path = folder + "/All_" + criterion + file_end
        df = pd.read_csv(path)

        # feature and target columns
        feature_cols1 = df.columns[1:25].tolist() + [df.columns[37]]  # Financial
        feature_cols2 = df.columns[1:38]  # Financial + GPT
        feature_cols3 = (
            df.columns[1:25].tolist() + [df.columns[37]] + df.columns[47:62].tolist()
        )  # Financial + detail
        feature_cols4 = (
            df.columns[1:25].tolist() + [df.columns[37]] + df.columns[62:].tolist()
        )  # Financial + formula
        feature_cols5 = (
            df.columns[1:38].tolist() + df.columns[47:62].tolist()
        )  # Financial + GPT + detail
        feature_cols6 = (
            df.columns[1:38].tolist() + df.columns[62:].tolist()
        )  # Financial + GPT + formula
        feature_cols7 = (
            df.columns[25:38].tolist() + df.columns[47:62].tolist()
        ) # GPT + detail
        feature_cols8 = (
            df.columns[25:38].tolist() + df.columns[62:].tolist()
        ) # GPT + formula
        target_cols = df.columns[38:47]
        feature_cols = [
            feature_cols1,
            feature_cols2,
            feature_cols3,
            feature_cols4,
            feature_cols5,
            feature_cols6,
            feature_cols7,
            feature_cols8,
        ]
        result_path = [
            "Fin",
            "Fin+GPT",
            "Fin+detail",
            "Fin+formula",
            "Fin+GPT+detail",
            "Fin+GPT+formula",
            "GPT+detail",
            "GPT+formula",
        ]

        for i in range(0, len(feature_cols)):
            X = df[feature_cols[i]]
            result_set = result_path[i]
            instances = df["instance"]

            # store actual and predicted values for each algorithm
            all_actual = pd.DataFrame(index=df.index, columns=target_cols)
            all_pred = pd.DataFrame(index=df.index, columns=target_cols)

            kf = KFold(n_splits=10, shuffle=True, random_state=42)

            def train_one_fold(target, train_idx, test_idx, fold_id):
                y = df[target]

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = RandomForestRegressor(
                    n_estimators=240,
                    random_state=42 + fold_id,
                    n_jobs=1
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                return target, y_test.index, y_test.values, y_pred


            tasks = []

            for target in target_cols:
                for fold_id, (train_idx, test_idx) in enumerate(kf.split(X)):
                    tasks.append((target, train_idx, test_idx, fold_id))


            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(train_one_fold)(target, train_idx, test_idx, fold_id)
                for target, train_idx, test_idx, fold_id in tasks
            )

            for target, test_index, y_true, y_pred in results:
                all_actual.loc[test_index, target] = y_true
                all_pred.loc[test_index, target] = y_pred

            # convert to numeric to avoid dtype issues
            all_actual = all_actual.apply(pd.to_numeric)
            all_pred = all_pred.apply(pd.to_numeric)

            # === summary results ===
            result_summary = pd.DataFrame(index=df.index)
            result_summary["Instance"] = instances

            # SBS: fixed baseline strategy
            sbs_algo = "OH-L" if criterion in ["CR", "AR"] else "MS-T"
            result_summary["SBS"] = all_actual[sbs_algo]

            # VBS: best actual value among all algorithms
            result_summary["VBS"] = all_actual.max(axis=1)

            # VBS algorithm: algorithm that actually gives the best value
            best_actual_algo = all_actual.idxmax(axis=1)
            result_summary["VBS Algo"] = best_actual_algo

            # Selection Result: choose algorithm by highest predicted score
            best_pred_algo = all_pred.idxmax(axis=1)
            result_summary["Selected Algo"] = best_pred_algo

            selection_result = []
            for i, algo in best_pred_algo.items():
                selection_result.append(all_actual.loc[i, algo])
            result_summary["Selection Result"] = selection_result

            # === summary statistics ===
            stats = pd.DataFrame(
                {
                    "Mean": result_summary[
                        ["SBS", "VBS", "Selection Result"]
                    ].mean(),
                    "Std": result_summary[["SBS", "VBS", "Selection Result"]].std(),
                    "Min": result_summary[["SBS", "VBS", "Selection Result"]].min(),
                    "Max": result_summary[["SBS", "VBS", "Selection Result"]].max(),
                }
            ).T

            # === selected algorithm counts ===
            algo_counts = (
                result_summary["Selected Algo"]
                .value_counts()
                .rename_axis("Algorithm")
                .reset_index(name="Count")
            )
            algo_counts["Proportion"] = algo_counts["Count"] / len(result_summary)

            # === VBS algorithm counts ===
            vbs_algo_counts = (
                result_summary["VBS Algo"]
                .value_counts()
                .rename_axis("Algorithm")
                .reset_index(name="Count")
            )
            vbs_algo_counts["Proportion"] = vbs_algo_counts["Count"] / len(
                result_summary
            )

            print("=== Per-instance results (head) ===")
            print(result_summary.head())

            print("\n=== Overall statistics ===")
            print(stats)

            print("\n=== Selected algorithm counts ===")
            print(algo_counts)

            print("\n=== VBS algorithm counts ===")
            print(vbs_algo_counts)

                # save
            path_result = (
                folder
                + "/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_"
                + result_set
                + ".csv"
            )
            path_stats = (
                folder
                + "/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_"
                + result_set
                + "_stats.csv"
            )
            path_counts = (
                folder
                + "/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_"
                + result_set
                + "_algo_counts.csv"
            )
            path_vbs_counts = (
                folder
                + "/"
                + criterion
                + "/"
                + criterion
                + "_strategy_selection_vbs_algo_counts.csv"
            )
            result_summary.to_csv(path_result, index=False)
            stats.to_csv(path_stats)
            algo_counts.to_csv(path_counts, index=False)
            vbs_algo_counts.to_csv(path_vbs_counts, index=False)


if __name__ == "__main__":
    RF()
