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
RESULTS = HERE / "selector_results"
ALGORITHMS = ["EW", "MCW", "mSSRM", "MS-M", "OH-M", "MS-T", "OH-T", "MS-L", "OH-L"]
CRITERIA = ["AR", "CR", "MDD", "SHR", "SOR"]
CONFIGURATIONS = [
    "Fin",
    "TransformerRaw",
    "TransformerPCA32",
    "Fin+TransformerRaw",
    "Fin+TransformerPCA32",
]
PCA_COMPONENTS = 32
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


def load_data(criterion: str):
    source = pd.read_csv(ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv")
    representations = pd.read_csv(REPRESENTATIONS)
    mapping = pd.read_csv(MAPPING)[["instance", "Asset"]]
    frame = source.merge(
        representations.drop(columns=["Asset"]),
        on="instance",
        validate="one_to_one",
    ).merge(mapping, on="instance", validate="one_to_one")

    financial = source.columns[1:25].tolist() + ["Period"]
    transformer = [
        column
        for column in representations
        if column.startswith("mst_h_") or column.startswith("oht_h_")
    ]
    if len(frame) != 762 or len(transformer) != 512:
        raise RuntimeError("Unexpected selector input dimensions")
    if not np.isfinite(frame[financial + transformer].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite selector inputs")
    return frame, financial, transformer


def build_regressor(
    selector: str,
    configuration: str,
    fold: int,
    financial_count: int,
    transformer_count: int,
):
    if selector == "RF":
        final_model = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=1,
        )
    else:
        final_model = MLPRegressor(
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

    if configuration == "Fin":
        if selector == "RF":
            feature_pipeline = "passthrough"
        else:
            feature_pipeline = StandardScaler()
    elif configuration == "TransformerRaw":
        if selector == "RF":
            feature_pipeline = "passthrough"
        else:
            feature_pipeline = StandardScaler()
    elif configuration == "TransformerPCA32":
        steps = [
            ("scale_before_pca", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=PCA_COMPONENTS,
                    whiten=True,
                    random_state=SEED + fold,
                ),
            ),
        ]
        feature_pipeline = Pipeline(steps)
    elif configuration == "Fin+TransformerRaw":
        if selector == "RF":
            feature_pipeline = "passthrough"
        else:
            feature_pipeline = StandardScaler()
    elif configuration == "Fin+TransformerPCA32":
        financial_transformer = (
            "passthrough" if selector == "RF" else StandardScaler()
        )
        feature_pipeline = ColumnTransformer(
            [
                (
                    "financial",
                    financial_transformer,
                    list(range(financial_count)),
                ),
                (
                    "transformer_pca",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "pca",
                                PCA(
                                    n_components=PCA_COMPONENTS,
                                    whiten=True,
                                    random_state=SEED + fold,
                                ),
                            ),
                        ]
                    ),
                    list(
                        range(
                            financial_count,
                            financial_count + transformer_count,
                        )
                    ),
                ),
            ]
        )
    else:
        raise ValueError(configuration)

    base = Pipeline(
        [
            ("features", feature_pipeline),
            ("regressor", final_model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=base,
        transformer=StandardScaler(),
    )


def fit_one(
    selector: str,
    configuration: str,
    values: np.ndarray,
    targets: pd.DataFrame,
    algorithm: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fold: int,
    financial_count: int,
    transformer_count: int,
):
    model = build_regressor(
        selector,
        configuration,
        fold,
        financial_count,
        transformer_count,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(values[train_indices], targets[algorithm].iloc[train_indices])
    return algorithm, test_indices, model.predict(values[test_indices])


def run_one(
    selector: str,
    criterion: str,
    configuration: str,
    jobs: int,
) -> dict:
    output_dir = RESULTS / selector / criterion / configuration
    result_path = output_dir / f"seed_{SEED}.csv"
    metadata_path = output_dir / f"seed_{SEED}.json"
    if result_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())

    frame, financial, transformer = load_data(criterion)
    if configuration == "Fin":
        columns = financial
        financial_count = len(financial)
        transformer_count = 0
    elif configuration.startswith("Transformer"):
        columns = transformer
        financial_count = 0
        transformer_count = len(transformer)
    else:
        columns = financial + transformer
        financial_count = len(financial)
        transformer_count = len(transformer)

    values = frame[columns].to_numpy(dtype=float)
    targets = frame[ALGORITHMS].astype(float)
    folds = list(
        GroupKFold(n_splits=5).split(values, groups=frame["Asset"])
    )
    tasks = [
        (algorithm, train_indices, test_indices, fold)
        for algorithm in ALGORITHMS
        for fold, (train_indices, test_indices) in enumerate(folds)
    ]
    fitted = Parallel(n_jobs=jobs, verbose=0)(
        delayed(fit_one)(
            selector,
            configuration,
            values,
            targets,
            algorithm,
            train_indices,
            test_indices,
            fold,
            financial_count,
            transformer_count,
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
    vbs = actual.max(axis=1)
    selection = actual[np.arange(len(frame)), selected_indices]
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
        "configuration": configuration,
        "seed": SEED,
        "raw_input_features": len(columns),
        "effective_transformer_features": (
            PCA_COMPONENTS if "PCA32" in configuration else transformer_count
        ),
        "selection_mean": float(result["selection"].mean()),
        "regret_mean": float(result["regret"].mean()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def make_comparison(selector: str, summaries: list[dict]) -> None:
    summary = pd.DataFrame(summaries)
    baseline = summary[summary["configuration"] == "Fin"][
        ["criterion", "selection_mean", "regret_mean"]
    ].rename(
        columns={
            "selection_mean": "fin_selection_mean",
            "regret_mean": "fin_regret_mean",
        }
    )
    comparison = summary.merge(baseline, on="criterion", validate="many_to_one")
    comparison["selection_delta_vs_fin"] = (
        comparison["selection_mean"] - comparison["fin_selection_mean"]
    )
    comparison["regret_delta_vs_fin"] = (
        comparison["regret_mean"] - comparison["fin_regret_mean"]
    )
    atomic_csv(RESULTS / f"{selector}_summary.csv", comparison)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", choices=["RF", "MLP"], required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()

    summaries = []
    for criterion in CRITERIA:
        for configuration in CONFIGURATIONS:
            metadata = run_one(
                args.selector,
                criterion,
                configuration,
                args.jobs,
            )
            summaries.append(metadata)
            print(json.dumps(metadata), flush=True)
    make_comparison(args.selector, summaries)


if __name__ == "__main__":
    main()
