"""LEAKY ORACLE DIAGNOSTIC.

This script deliberately trains each asset-specific encoder with all AS labels,
including labels later assigned to grouped RF test folds. Its results are not a
valid estimate of generalization and must not be reported as such.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import warnings
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
sys.path.insert(0, str(HERE))
import run_experiment as base


OUTPUTS = HERE / "leaky_oracle_diagnostic"
LOGS = OUTPUTS / "logs"
METHODS = ["A_AS", "C_MS_T", "C_OH_T"]
CONFIGS = ["EmbeddingRaw", "Fin+EmbeddingRaw", "Fin+EmbeddingPCA32"]


def train_asset(asset: str, smoke: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)

    mapping = base.load_mapping()
    local = mapping[mapping["Asset"] == asset].copy()
    data, _ = base.load_asset_data(asset)
    num_assets = len(data["log_tr"].columns)
    outcome_maps = {
        criterion: base.load_outcomes(criterion).set_index("instance")
        for criterion in base.CRITERIA
    }
    methods = ["A_AS"] if smoke else METHODS
    criteria = ["AR"] if smoke else base.CRITERIA
    as_epochs = 3 if smoke else 50
    pretrain_epochs = 1 if smoke else 10
    started = time.monotonic()

    for method in methods:
        pretrained = None
        if method in {"C_MS_T", "C_OH_T"}:
            pretrained = base.build_model(len(data.columns), num_assets).to(device)
            target = "maxsortino" if method == "C_MS_T" else "onehot"
            pretrained = base.pretrain_po(
                pretrained,
                asset,
                data,
                target,
                device,
                pretrain_epochs,
            )
        for criterion in criteria:
            model = (
                copy.deepcopy(pretrained)
                if pretrained is not None
                else base.build_model(len(data.columns), num_assets).to(device)
            )
            samples = []
            for row in local.itertuples(index=False):
                samples.append(
                    (
                        int(row.instance),
                        base.instance_tensor(data, row.start),
                        int(outcome_maps[criterion].loc[row.instance, "label"]),
                    )
                )
            # Deliberate leakage: every instance label is used.
            model = base.train_as(
                model,
                samples,
                device,
                as_epochs,
                freeze_encoder=False,
            )
            model.eval()
            rows = []
            with torch.no_grad():
                for instance, values, label in samples:
                    hidden = model.encode(values.to(device))[-1].cpu().numpy()
                    record = {
                        "instance": instance,
                        "Asset": asset,
                        "criterion": criterion,
                        "method": method,
                        "as_label": label,
                    }
                    record.update(
                        {
                            f"h_{index:03d}": float(value)
                            for index, value in enumerate(hidden)
                        }
                    )
                    rows.append(record)
            path = OUTPUTS / "embeddings" / method / criterion / f"{asset}.csv"
            base.atomic_csv(path, pd.DataFrame(rows))
            base.atomic_json(
                path.with_suffix(".json"),
                {
                    "warning": "LEAKY: all labels, including RF test-fold labels, trained the encoder",
                    "asset": asset,
                    "method": method,
                    "criterion": criterion,
                    "instances": len(rows),
                    "as_epochs": as_epochs,
                    "po_pretrain_epochs": (
                        pretrain_epochs if pretrained is not None else 0
                    ),
                    "encoder_finetuned_on_as": True,
                },
            )
    print(
        json.dumps(
            {
                "asset": asset,
                "methods": methods,
                "criteria": criteria,
                "elapsed_seconds": time.monotonic() - started,
                "warning": "LEAKY ORACLE DIAGNOSTIC",
            }
        ),
        flush=True,
    )


def feature_pipeline(config: str, fin_count: int, fold: int):
    if config == "EmbeddingRaw":
        return "passthrough"
    if config == "Fin+EmbeddingRaw":
        return "passthrough"
    if config == "Fin+EmbeddingPCA32":
        return ColumnTransformer(
            [
                ("fin", "passthrough", list(range(fin_count))),
                (
                    "embedding",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "pca",
                                PCA(
                                    n_components=32,
                                    whiten=True,
                                    random_state=base.SEED + fold,
                                ),
                            ),
                        ]
                    ),
                    list(range(fin_count, fin_count + base.D_MODEL)),
                ),
            ]
        )
    raise ValueError(config)


def fit_one(
    config,
    fin_count,
    fold,
    values,
    targets,
    algorithm,
    train_indices,
    test_indices,
):
    regressor = RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        random_state=base.SEED + fold,
        n_jobs=1,
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("features", feature_pipeline(config, fin_count, fold)),
                ("rf", regressor),
            ]
        ),
        transformer=StandardScaler(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(values[train_indices], targets[algorithm].iloc[train_indices])
    return algorithm, test_indices, model.predict(values[test_indices])


def run_rf(method: str, criterion: str, config: str, jobs: int) -> dict:
    result_path = OUTPUTS / "rf" / method / criterion / config / "results.csv"
    metadata_path = result_path.with_suffix(".json")
    if result_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())
    files = sorted(
        (OUTPUTS / "embeddings" / method / criterion).glob("asset_*.csv")
    )
    if len(files) != 23:
        raise RuntimeError(f"Expected 23 embedding files, got {len(files)}")
    embeddings = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    source = pd.read_csv(
        ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv"
    )
    mapping = pd.read_csv(base.MAPPING)[["instance", "Asset"]]
    frame = (
        source.merge(
            embeddings.drop(columns=["Asset", "criterion", "method", "as_label"]),
            on="instance",
            validate="one_to_one",
        )
        .merge(mapping, on="instance", validate="one_to_one")
    )
    financial = source.columns[1:25].tolist() + ["Period"]
    hidden = [f"h_{index:03d}" for index in range(base.D_MODEL)]
    if config == "EmbeddingRaw":
        columns, fin_count = hidden, 0
    else:
        columns, fin_count = financial + hidden, len(financial)
    values = frame[columns].to_numpy(dtype=float)
    targets = frame[base.ALGORITHMS].astype(float)
    folds = list(
        GroupKFold(n_splits=5).split(values, groups=frame["Asset"])
    )
    tasks = [
        (algorithm, train, test, fold)
        for algorithm in base.ALGORITHMS
        for fold, (train, test) in enumerate(folds)
    ]
    fitted = Parallel(n_jobs=jobs)(
        delayed(fit_one)(
            config,
            fin_count,
            fold,
            values,
            targets,
            algorithm,
            train,
            test,
        )
        for algorithm, train, test, fold in tasks
    )
    predictions = pd.DataFrame(
        index=frame.index, columns=base.ALGORITHMS, dtype=float
    )
    for algorithm, test, prediction in fitted:
        predictions.loc[test, algorithm] = prediction
    chosen = predictions.idxmax(axis=1)
    actual = targets.to_numpy()
    chosen_index = np.array(
        [base.ALGORITHMS.index(value) for value in chosen]
    )
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
    base.atomic_csv(result_path, result)
    metadata = {
        "warning": "LEAKY ORACLE DIAGNOSTIC; invalid generalization estimate",
        "method": method,
        "criterion": criterion,
        "configuration": config,
        "instances": len(result),
        "selection_mean": float(result["selection"].mean()),
        "regret_mean": float(result["regret"].mean()),
    }
    base.atomic_json(metadata_path, metadata)
    return metadata


def combine(jobs: int = 24) -> None:
    rows = []
    for method in METHODS:
        for criterion in base.CRITERIA:
            for config in CONFIGS:
                metadata = run_rf(method, criterion, config, jobs)
                rows.append(metadata)
                print(json.dumps(metadata), flush=True)
    base.atomic_csv(OUTPUTS / "rf_summary.csv", pd.DataFrame(rows))
    base.atomic_json(
        OUTPUTS / "validation_report.json",
        {
            "complete": True,
            "warning": "LEAKY ORACLE DIAGNOSTIC; do not report as valid evaluation",
            "rows": rows,
        },
    )


def launch(gpus: list[int], timeout_minutes: int) -> None:
    if not gpus or len(gpus) > 4:
        raise ValueError("Use one to four GPUs")
    assets = sorted(
        base.load_mapping()["Asset"].unique(),
        key=lambda value: int(value.split("_")[1]),
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, running, completed, failed = list(assets), {}, [], []
    deadline = time.monotonic() + timeout_minutes * 60
    while pending or running:
        if time.monotonic() >= deadline:
            for process in running:
                process.terminate()
            failed.extend(info["asset"] for info in running.values())
            break
        used = {info["gpu"] for info in running.values()}
        free = [gpu for gpu in gpus if gpu not in used]
        while pending and free:
            asset, gpu = pending.pop(0), free.pop(0)
            handle = (LOGS / f"{asset}.log").open("a")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--asset",
                    asset,
                ],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[process] = {
                "asset": asset,
                "gpu": gpu,
                "handle": handle,
            }
        time.sleep(2)
        for process, info in list(running.items()):
            if process.poll() is None:
                continue
            info["handle"].close()
            running.pop(process)
            (completed if process.returncode == 0 else failed).append(info["asset"])
    if failed:
        base.atomic_json(
            OUTPUTS / "run_state.json",
            {"status": "failed", "completed": completed, "failed": failed},
        )
        raise SystemExit(1)
    combine()
    base.atomic_json(
        OUTPUTS / "run_state.json",
        {
            "status": "complete",
            "completed": completed,
            "failed": [],
            "warning": "LEAKY ORACLE DIAGNOSTIC",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--asset", required=True)
    worker.add_argument("--smoke", action="store_true")
    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--gpus", default="0,1,2,3")
    launcher.add_argument("--timeout-minutes", type=int, default=120)
    combiner = subparsers.add_parser("combine")
    combiner.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.command == "worker":
        train_asset(args.asset, args.smoke)
    elif args.command == "launch":
        launch(
            [int(value) for value in args.gpus.split(",") if value],
            args.timeout_minutes,
        )
    else:
        combine(args.jobs)


if __name__ == "__main__":
    main()
