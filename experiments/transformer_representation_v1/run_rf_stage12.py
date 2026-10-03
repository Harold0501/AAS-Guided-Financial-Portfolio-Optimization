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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAPPING = ROOT / "LLM/Assets_and_Instances/combined_asset_components_time.csv"
ALGORITHMS = ["EW", "MCW", "mSSRM", "MS-M", "OH-M", "MS-T", "OH-T", "MS-L", "OH-L"]
CRITERIA = ["AR", "CR", "MDD", "SHR", "SOR"]


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


def load_data(criterion: str, representation_path: Path):
    source = pd.read_csv(ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv")
    representation = pd.read_csv(representation_path)
    mapping = pd.read_csv(MAPPING)[["instance", "Asset"]]
    frame = (
        source.merge(
            representation.drop(columns=["Asset"]),
            on="instance",
            validate="one_to_one",
        )
        .merge(mapping, on="instance", validate="one_to_one")
    )
    groups = {
        "Fin": source.columns[1:25].tolist() + ["Period"],
        "MS-T": [c for c in representation if c.startswith("mst_h_")],
        "OH-T": [c for c in representation if c.startswith("oht_h_")],
    }
    if len(frame) != 762 or len(groups["MS-T"]) != 256:
        raise RuntimeError("Unexpected representation input")
    columns = sum(groups.values(), [])
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite input")
    return frame, groups


def specs(groups: dict[str, list[str]]) -> dict[str, list[tuple]]:
    fin, mst, oht = groups["Fin"], groups["MS-T"], groups["OH-T"]
    return {
        "Fin": [("raw", "Fin", fin)],
        "Fin+OH-T-Raw": [("raw", "Fin", fin), ("raw", "OH-T", oht)],
        "Fin+OH-T-PCA8": [("raw", "Fin", fin), ("pca8", "OH-T", oht)],
        "Fin+OH-T-PCA16": [("raw", "Fin", fin), ("pca16", "OH-T", oht)],
        "Fin+OH-T-PCA32": [("raw", "Fin", fin), ("pca32", "OH-T", oht)],
        "Fin+OH-T-PCA64": [("raw", "Fin", fin), ("pca64", "OH-T", oht)],
        "Fin+MS-T-PCA32": [("raw", "Fin", fin), ("pca32", "MS-T", mst)],
        "Fin+MS-T+OH-T-Raw": [
            ("raw", "Fin", fin),
            ("raw", "MS-T", mst),
            ("raw", "OH-T", oht),
        ],
        "Fin+MS-T+OH-T-PCA16each": [
            ("raw", "Fin", fin),
            ("pca16", "MS-T", mst),
            ("pca16", "OH-T", oht),
        ],
    }


def pca_transform(kind: str, fold: int):
    if kind == "raw":
        return "passthrough"
    components = int(kind.removeprefix("pca"))
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=components,
                    whiten=True,
                    random_state=fold,
                ),
            ),
        ]
    )


def fit_model(specification, columns, seed, fold):
    transformers = []
    cursor = 0
    for kind, block, block_columns in specification:
        start, end = cursor, cursor + len(block_columns)
        cursor = end
        transformers.append(
            (block, pca_transform(kind, fold), list(range(start, end)))
        )
    features = ColumnTransformer(transformers)
    regressor = RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        random_state=seed + fold,
        n_jobs=1,
    )
    pipeline = Pipeline([("features", features), ("rf", regressor)])
    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def fit_one(
    specification,
    columns,
    seed,
    frame,
    values,
    targets,
    algorithm,
    train_indices,
    test_indices,
    fold,
):
    model = fit_model(specification, columns, seed, fold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(values[train_indices], targets[algorithm].iloc[train_indices])
    return algorithm, test_indices, model.predict(values[test_indices])


def run_one(
    representation_path: Path,
    output_root: Path,
    representation_seed: int,
    rf_seed: int,
    criterion: str,
    name: str,
    specification: list[tuple],
    jobs: int,
) -> dict:
    output_dir = (
        output_root
        / f"representation_seed_{representation_seed}"
        / f"rf_seed_{rf_seed}"
        / criterion
        / name
    )
    result_path = output_dir / "results.csv"
    metadata_path = output_dir / "metadata.json"
    if result_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())

    frame, _ = load_data(criterion, representation_path)
    columns = sum((block[2] for block in specification), [])
    values = frame[columns].to_numpy(dtype=float)
    targets = frame[ALGORITHMS].astype(float)
    folds = list(GroupKFold(n_splits=5).split(values, groups=frame["Asset"]))
    tasks = [
        (algorithm, train, test, fold)
        for algorithm in ALGORITHMS
        for fold, (train, test) in enumerate(folds)
    ]
    fitted = Parallel(n_jobs=jobs, verbose=0)(
        delayed(fit_one)(
            specification,
            columns,
            rf_seed,
            frame,
            values,
            targets,
            algorithm,
            train,
            test,
            fold,
        )
        for algorithm, train, test, fold in tasks
    )
    predictions = pd.DataFrame(
        index=frame.index, columns=ALGORITHMS, dtype=float
    )
    for algorithm, test, prediction in fitted:
        predictions.loc[test, algorithm] = prediction
    if predictions.isna().any().any():
        raise RuntimeError("Missing out-of-fold predictions")
    chosen = predictions.idxmax(axis=1)
    actual = targets.to_numpy()
    chosen_index = np.array([ALGORITHMS.index(value) for value in chosen])
    selection = actual[np.arange(len(frame)), chosen_index]
    vbs = actual.max(axis=1)
    result = pd.DataFrame(
        {
            "instance": frame["instance"],
            "Asset": frame["Asset"],
            "selected": chosen,
            "selection": selection,
            "vbs": vbs,
            "regret": vbs - selection,
        }
    )
    atomic_csv(result_path, result)
    metadata = {
        "representation_seed": representation_seed,
        "rf_seed": rf_seed,
        "criterion": criterion,
        "configuration": name,
        "input_features": len(columns),
        "selection_mean": float(result["selection"].mean()),
        "regret_mean": float(result["regret"].mean()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--representation-seed", type=int, required=True)
    parser.add_argument("--rf-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()

    _, groups = load_data("AR", args.representations)
    all_configs = specs(groups)
    names = args.configs or list(all_configs)
    unknown = set(names) - set(all_configs)
    if unknown:
        raise SystemExit(f"Unknown configurations: {sorted(unknown)}")

    summaries = []
    for rf_seed in args.rf_seeds:
        for criterion in CRITERIA:
            for name in names:
                summary = run_one(
                    args.representations,
                    args.output_root,
                    args.representation_seed,
                    rf_seed,
                    criterion,
                    name,
                    all_configs[name],
                    args.jobs,
                )
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    summary = pd.DataFrame(summaries)
    atomic_csv(
        args.output_root
        / f"summary_representation_{args.representation_seed}.csv",
        summary,
    )


if __name__ == "__main__":
    main()
