from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPRESENTATIONS = HERE / "outputs/transformer_representations_1repeat.csv"
MAPPING = ROOT / "LLM/Assets_and_Instances/combined_asset_components_time.csv"
RESULTS = HERE / "selector_combinations"
ALGORITHMS = ["EW", "MCW", "mSSRM", "MS-M", "OH-M", "MS-T", "OH-T", "MS-L", "OH-L"]
CRITERIA = ["AR", "CR", "MDD", "SHR", "SOR"]
SEED = 42


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_data(criterion: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    source = pd.read_csv(ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv")
    representations = pd.read_csv(REPRESENTATIONS)
    mapping = pd.read_csv(MAPPING)[["instance", "Asset"]]
    frame = (
        source.merge(
            representations.drop(columns=["Asset"]),
            on="instance",
            validate="one_to_one",
        )
        .merge(mapping, on="instance", validate="one_to_one")
    )
    groups = {
        "Fin": source.columns[1:25].tolist() + ["Period"],
        "GPT": source.columns[25:37].tolist(),
        "Detail": source.columns[47:62].tolist(),
        "Formula": source.columns[62:77].tolist(),
        "MS-T": [c for c in representations if c.startswith("mst_h_")],
        "OH-T": [c for c in representations if c.startswith("oht_h_")],
    }
    if len(frame) != 762 or any(len(groups[key]) == 0 for key in groups):
        raise RuntimeError("Unexpected selector input dimensions")
    numeric = sum(groups.values(), [])
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite selector inputs")
    return frame, groups


def configuration_specs(groups: dict[str, list[str]]) -> dict[str, list[tuple]]:
    fin, gpt = groups["Fin"], groups["GPT"]
    detail, formula = groups["Detail"], groups["Formula"]
    mst, oht = groups["MS-T"], groups["OH-T"]
    return {
        "Fin": [("raw", "Fin", fin)],
        "Fin+MS-T-Raw": [("raw", "Fin", fin), ("raw", "MS-T", mst)],
        "Fin+OH-T-Raw": [("raw", "Fin", fin), ("raw", "OH-T", oht)],
        "Fin+MS-T+OH-T-Raw": [
            ("raw", "Fin", fin),
            ("raw", "MS-T", mst),
            ("raw", "OH-T", oht),
        ],
        "Fin+MS-T-PCA32": [("raw", "Fin", fin), ("pca", "MS-T", mst)],
        "Fin+OH-T-PCA32": [("raw", "Fin", fin), ("pca", "OH-T", oht)],
        "Fin+MS-T+OH-T-PCA16each": [
            ("raw", "Fin", fin),
            ("pca16", "MS-T", mst),
            ("pca16", "OH-T", oht),
        ],
        "Fin+GPT+MS-T+OH-T-PCA16each": [
            ("raw", "Fin", fin),
            ("raw", "GPT", gpt),
            ("pca16", "MS-T", mst),
            ("pca16", "OH-T", oht),
        ],
        "Fin+Detail+MS-T+OH-T-PCA16each": [
            ("raw", "Fin", fin),
            ("raw", "Detail", detail),
            ("pca16", "MS-T", mst),
            ("pca16", "OH-T", oht),
        ],
        "Fin+Formula+MS-T+OH-T-PCA16each": [
            ("raw", "Fin", fin),
            ("raw", "Formula", formula),
            ("pca16", "MS-T", mst),
            ("pca16", "OH-T", oht),
        ],
        "Fin+GPT+Detail+Formula+MS-T+OH-T-PCA16each": [
            ("raw", "Fin", fin),
            ("raw", "GPT", gpt),
            ("raw", "Detail", detail),
            ("raw", "Formula", formula),
            ("pca16", "MS-T", mst),
            ("pca16", "OH-T", oht),
        ],
    }


def build_model(
    selector: str,
    specification: list[tuple],
    frame_columns: list[str],
    fold: int,
):
    transformers = []
    cursor = 0
    for kind, block_name, columns in specification:
        start, end = cursor, cursor + len(columns)
        cursor = end
        if kind == "raw":
            transform = "passthrough"
        elif kind == "pca":
            transform = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=32,
                            whiten=True,
                            random_state=SEED + fold,
                        ),
                    ),
                ]
            )
        elif kind == "pca16":
            transform = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=16,
                            whiten=True,
                            random_state=SEED + fold,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(kind)
        transformers.append((block_name, transform, list(range(start, end))))

    if selector == "RF":
        regressor = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=1,
        )
        outer_transform = "passthrough"
    else:
        regressor = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=5e-4,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=SEED + fold,
        )
        outer_transform = StandardScaler()

    feature_transform = ColumnTransformer(transformers)
    pipeline = Pipeline(
        [
            ("blocks", feature_transform),
            ("outer_scale", outer_transform),
            ("regressor", regressor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def fit_one(
    selector: str,
    specification: list[tuple],
    values: np.ndarray,
    targets: pd.DataFrame,
    algorithm: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fold: int,
    columns: list[str],
):
    model = build_model(selector, specification, columns, fold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(values[train_indices], targets[algorithm].iloc[train_indices])
    return algorithm, test_indices, model.predict(values[test_indices])


def run_one(
    selector: str,
    criterion: str,
    name: str,
    specification: list[tuple],
    jobs: int,
) -> dict:
    output_dir = RESULTS / selector / criterion / name
    result_path = output_dir / f"seed_{SEED}.csv"
    metadata_path = output_dir / f"seed_{SEED}.json"
    if result_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())

    frame, groups = load_data(criterion)
    columns = sum((block[2] for block in specification), [])
    values = frame[columns].to_numpy(dtype=float)
    targets = frame[ALGORITHMS].astype(float)
    folds = list(GroupKFold(n_splits=5).split(values, groups=frame["Asset"]))
    tasks = [
        (algorithm, train_indices, test_indices, fold)
        for algorithm in ALGORITHMS
        for fold, (train_indices, test_indices) in enumerate(folds)
    ]
    fitted = Parallel(n_jobs=jobs, verbose=0)(
        delayed(fit_one)(
            selector,
            specification,
            values,
            targets,
            algorithm,
            train_indices,
            test_indices,
            fold,
            columns,
        )
        for algorithm, train_indices, test_indices, fold in tasks
    )
    predictions = pd.DataFrame(
        index=frame.index, columns=ALGORITHMS, dtype=float
    )
    for algorithm, test_indices, prediction in fitted:
        predictions.loc[test_indices, algorithm] = prediction
    if predictions.isna().any().any():
        raise RuntimeError("Missing out-of-fold predictions")

    selected = predictions.idxmax(axis=1)
    actual = targets.to_numpy()
    selected_indices = np.array([ALGORITHMS.index(value) for value in selected])
    selection = actual[np.arange(len(frame)), selected_indices]
    vbs = actual.max(axis=1)
    result = pd.DataFrame(
        {
            "instance": frame["instance"],
            "Asset": frame["Asset"],
            "selected": selected,
            "selection": selection,
            "vbs": vbs,
            "regret": vbs - selection,
        }
    )
    atomic_csv(result_path, result)
    metadata = {
        "selector": selector,
        "criterion": criterion,
        "configuration": name,
        "seed": SEED,
        "input_features": len(columns),
        "selection_mean": float(result["selection"].mean()),
        "regret_mean": float(result["regret"].mean()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", choices=["RF", "MLP"], required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()

    _, groups = load_data("AR")
    configs = configuration_specs(groups)
    summaries = []
    for criterion in CRITERIA:
        for name, specification in configs.items():
            metadata = run_one(
                args.selector,
                criterion,
                name,
                specification,
                args.jobs,
            )
            summaries.append(metadata)
            print(json.dumps(metadata), flush=True)

    summary = pd.DataFrame(summaries)
    baseline = summary[summary["configuration"] == "Fin"][
        ["criterion", "selection_mean", "regret_mean"]
    ].rename(
        columns={
            "selection_mean": "fin_selection_mean",
            "regret_mean": "fin_regret_mean",
        }
    )
    summary = summary.merge(baseline, on="criterion", validate="many_to_one")
    summary["selection_delta_vs_fin"] = (
        summary["selection_mean"] - summary["fin_selection_mean"]
    )
    summary["regret_delta_vs_fin"] = (
        summary["regret_mean"] - summary["fin_regret_mean"]
    )
    atomic_csv(RESULTS / f"{args.selector}_summary.csv", summary)


if __name__ == "__main__":
    main()
