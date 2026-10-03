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
CLUSTER_EXPERIMENT = ROOT / "experiments/cluster_po_adapter_v1"
sys.path.insert(0, str(CLUSTER_EXPERIMENT))
import run_experiment as shared


OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
ALGORITHMS = shared.ALGORITHMS
CRITERIA = shared.CRITERIA
D_MODEL = shared.D_MODEL
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


def outcome_labels(mapping: pd.DataFrame, criterion: str) -> pd.DataFrame:
    source = pd.read_csv(
        ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv"
    )
    source["label"] = source[ALGORITHMS].to_numpy().argmax(axis=1)
    return mapping[["instance", "context_id", "Asset"]].merge(
        source[["instance", "label"]],
        on="instance",
        validate="one_to_one",
    )


def train_as(
    model,
    train_contexts,
    labels,
    tensors,
    device,
    epochs,
):
    import torch
    import torch.nn.functional as functional

    grouped_labels = (
        labels.groupby("context_id")["label"].apply(list).to_dict()
    )
    train_assets = sorted(train_contexts["Asset"].unique())
    parameters = [
        *model.body.parameters(),
        *model.as_head.parameters(),
    ]
    for asset in train_assets:
        parameters.extend(model.input_adapters[asset].parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=3e-4, weight_decay=1e-4
    )
    rows = train_contexts.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        for index in rng.permutation(len(rows)):
            row = rows.iloc[index]
            optimizer.zero_grad(set_to_none=True)
            hidden = model.encode(
                row.Asset,
                tensors[row.context_id].to(device),
            )[-1]
            logits = model.as_head(hidden)
            targets = torch.tensor(
                grouped_labels[row.context_id],
                dtype=torch.long,
                device=device,
            )
            loss = functional.cross_entropy(
                logits.unsqueeze(0).expand(len(targets), -1),
                targets,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite AS loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()


def adapt_test_io(
    model,
    test_contexts,
    tensors,
    targets,
    device,
    epochs,
):
    import torch

    for parameter in model.parameters():
        parameter.requires_grad = False
    for asset, group in test_contexts.groupby("Asset"):
        parameters = [
            *model.input_adapters[asset].parameters(),
            *model.output_heads[asset].parameters(),
        ]
        for parameter in parameters:
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            parameters, lr=1e-4, weight_decay=1e-4
        )
        rows = group.reset_index(drop=True)
        for _ in range(epochs):
            for row in rows.itertuples(index=False):
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    asset,
                    tensors[row.context_id].to(device),
                )
                loss = shared.po_loss(logits, targets[row.context_id])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
        for parameter in parameters:
            parameter.requires_grad = False
    for parameter in model.parameters():
        parameter.requires_grad = True


def rf_feature_transform(config: str, financial_count: int, fold: int):
    if config in {"Fin", "EmbeddingRaw", "Fin+EmbeddingRaw"}:
        return "passthrough"
    if config == "Fin+EmbeddingPCA32":
        return ColumnTransformer(
            [
                (
                    "fin",
                    "passthrough",
                    list(range(financial_count)),
                ),
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
                                    random_state=SEED + fold,
                                ),
                            ),
                        ]
                    ),
                    list(
                        range(
                            financial_count,
                            financial_count + D_MODEL,
                        )
                    ),
                ),
            ]
        )
    raise ValueError(config)


def fit_rf(
    config,
    financial_count,
    fold,
    values,
    targets,
    algorithm,
    train_indices,
    test_indices,
):
    model = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                (
                    "features",
                    rf_feature_transform(
                        config, financial_count, fold
                    ),
                ),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=120,
                        min_samples_leaf=2,
                        random_state=SEED + fold,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        transformer=StandardScaler(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(values[train_indices], targets[algorithm].iloc[train_indices])
    return algorithm, test_indices, model.predict(values[test_indices])


def evaluate_rf(
    frame,
    train_assets,
    test_assets,
    financial_columns,
    hidden_columns,
    criterion,
    config,
    fold,
):
    source = pd.read_csv(
        ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv"
    )
    data = frame.merge(
        source[["instance", *financial_columns, *ALGORITHMS]],
        on="instance",
        validate="one_to_one",
    )
    if config == "Fin":
        columns, financial_count = financial_columns, len(financial_columns)
    elif config == "EmbeddingRaw":
        columns, financial_count = hidden_columns, 0
    else:
        columns = financial_columns + hidden_columns
        financial_count = len(financial_columns)
    train_indices = np.flatnonzero(data["Asset"].isin(train_assets))
    test_indices = np.flatnonzero(data["Asset"].isin(test_assets))
    values = data[columns].to_numpy(dtype=float)
    targets = data[ALGORITHMS].astype(float)
    fitted = Parallel(n_jobs=18)(
        delayed(fit_rf)(
            config,
            financial_count,
            fold,
            values,
            targets,
            algorithm,
            train_indices,
            test_indices,
        )
        for algorithm in ALGORITHMS
    )
    predictions = pd.DataFrame(
        index=test_indices, columns=ALGORITHMS, dtype=float
    )
    for algorithm, indices, prediction in fitted:
        predictions.loc[indices, algorithm] = prediction
    chosen = predictions.idxmax(axis=1)
    actual = targets.iloc[test_indices].to_numpy()
    chosen_indices = np.asarray(
        [ALGORITHMS.index(value) for value in chosen]
    )
    selection = actual[np.arange(len(actual)), chosen_indices]
    vbs = actual.max(axis=1)
    return {
        "fold": fold,
        "criterion": criterion,
        "configuration": config,
        "test_instances": len(test_indices),
        "selection_mean": float(selection.mean()),
        "regret_mean": float((vbs - selection).mean()),
    }


def run_fold(fold: int, smoke: bool) -> None:
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)

    mapping, contexts, tensors, targets = shared.prepare_contexts()
    split = list(
        GroupKFold(n_splits=5).split(
            mapping, groups=mapping["Asset"]
        )
    )[fold]
    train_assets = set(mapping.iloc[split[0]]["Asset"].unique())
    test_assets = set(mapping.iloc[split[1]]["Asset"].unique())
    train_contexts = contexts[
        contexts["Asset"].isin(train_assets)
    ].reset_index(drop=True)
    test_contexts = contexts[
        contexts["Asset"].isin(test_assets)
    ].reset_index(drop=True)

    pretrained, _ = shared.build_network(contexts)
    pretrained = pretrained.to(device)
    pretrained.as_head = nn.Linear(D_MODEL, len(ALGORITHMS)).to(device)
    po_epochs = 2 if smoke else 10
    as_epochs = 2 if smoke else 30
    adapter_epochs = 1 if smoke else 15
    started = time.monotonic()
    shared.train_samples(
        pretrained,
        train_contexts,
        tensors,
        targets,
        device,
        po_epochs,
    )
    criteria = ["AR"] if smoke else CRITERIA
    feature_rows = []
    summaries = []
    fin_source = pd.read_csv(
        ROOT / "RF_combined/All_AR_LLM_combined.csv"
    )
    financial_columns = fin_source.columns[1:25].tolist() + ["Period"]
    for criterion in criteria:
        model = copy.deepcopy(pretrained)
        model.as_head = nn.Linear(D_MODEL, len(ALGORITHMS)).to(device)
        labels = outcome_labels(mapping, criterion)
        train_labels = labels[
            labels["Asset"].isin(train_assets)
        ]
        train_as(
            model,
            train_contexts,
            train_labels,
            tensors,
            device,
            as_epochs,
        )
        # Test adapters receive only PO targets; no test AS labels are passed.
        adapt_test_io(
            model,
            test_contexts,
            tensors,
            targets,
            device,
            adapter_epochs,
        )
        model.eval()
        context_records = []
        with torch.no_grad():
            for row in contexts.itertuples(index=False):
                hidden = model.encode(
                    row.Asset,
                    tensors[row.context_id].to(device),
                )[-1].cpu().numpy()
                record = {
                    "context_id": row.context_id,
                    "Asset": row.Asset,
                }
                record.update(
                    {
                        f"h_{index:03d}": float(value)
                        for index, value in enumerate(hidden)
                    }
                )
                context_records.append(record)
        context_frame = pd.DataFrame(context_records)
        instance_frame = mapping[
            ["instance", "context_id", "Asset"]
        ].merge(
            context_frame,
            on=["context_id", "Asset"],
            validate="many_to_one",
        )
        hidden_columns = [
            f"h_{index:03d}" for index in range(D_MODEL)
        ]
        output_dir = OUTPUTS / ("smoke" if smoke else "full")
        atomic_csv(
            output_dir / f"fold_{fold}_{criterion}_features.csv",
            instance_frame,
        )
        for config in [
            "Fin",
            "EmbeddingRaw",
            "Fin+EmbeddingRaw",
            "Fin+EmbeddingPCA32",
        ]:
            summaries.append(
                evaluate_rf(
                    instance_frame,
                    train_assets,
                    test_assets,
                    financial_columns,
                    hidden_columns,
                    criterion,
                    config,
                    fold,
                )
            )
        feature_rows.append(
            {
                "criterion": criterion,
                "train_label_instances": len(train_labels),
                "test_label_instances_used": 0,
            }
        )
    output_dir = OUTPUTS / ("smoke" if smoke else "full")
    atomic_csv(
        output_dir / f"fold_{fold}_summary.csv",
        pd.DataFrame(summaries),
    )
    atomic_json(
        output_dir / f"fold_{fold}_metadata.json",
        {
            "fold": fold,
            "smoke": smoke,
            "train_assets": sorted(train_assets),
            "test_assets": sorted(test_assets),
            "po_pretrain_epochs": po_epochs,
            "as_finetune_epochs": as_epochs,
            "test_adapter_epochs": adapter_epochs,
            "label_audit": feature_rows,
            "test_adapter_objective": "OH-T PO target only",
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        json.dumps(
            {
                "fold": fold,
                "criteria": criteria,
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def combine() -> dict:
    paths = [
        OUTPUTS / "full" / f"fold_{fold}_summary.csv"
        for fold in range(5)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"complete": False, "missing": missing}
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = []
    for keys, group in rows.groupby(["criterion", "configuration"]):
        weights = group["test_instances"].to_numpy()
        summary.append(
            {
                "criterion": keys[0],
                "configuration": keys[1],
                "test_instances": int(weights.sum()),
                "selection_mean": float(
                    np.average(group["selection_mean"], weights=weights)
                ),
                "regret_mean": float(
                    np.average(group["regret_mean"], weights=weights)
                ),
            }
        )
    frame = pd.DataFrame(summary)
    atomic_csv(OUTPUTS / "summary.csv", frame)
    report = {
        "complete": True,
        "summary": summary,
        "protocol": "asset-family grouped 5-fold; AS labels from train assets only; test I/O adapters use OH-T PO targets only",
    }
    atomic_json(OUTPUTS / "validation_report.json", report)
    return report


def launch(gpus: list[int], timeout_minutes: int) -> None:
    if not gpus or len(gpus) > 4:
        raise ValueError("Use one to four GPUs")
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, running, completed, failed = list(range(5)), {}, [], []
    deadline = time.monotonic() + timeout_minutes * 60
    while pending or running:
        if time.monotonic() >= deadline:
            for process in running:
                process.terminate()
            failed.extend(info["fold"] for info in running.values())
            break
        used = {info["gpu"] for info in running.values()}
        free = [gpu for gpu in gpus if gpu not in used]
        while pending and free:
            fold, gpu = pending.pop(0), free.pop(0)
            handle = (LOGS / f"fold_{fold}.log").open("a")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--fold",
                    str(fold),
                ],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[process] = {
                "fold": fold,
                "gpu": gpu,
                "handle": handle,
            }
        time.sleep(2)
        for process, info in list(running.items()):
            if process.poll() is None:
                continue
            info["handle"].close()
            running.pop(process)
            (completed if process.returncode == 0 else failed).append(info["fold"])
    report = combine()
    atomic_json(
        OUTPUTS / "run_state.json",
        {
            "status": "complete" if not failed and report.get("complete") else "failed",
            "completed": completed,
            "failed": failed,
            "report": report,
        },
    )
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--fold", type=int, choices=range(5), required=True)
    worker.add_argument("--smoke", action="store_true")
    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--gpus", default="0,1,2,3")
    launcher.add_argument("--timeout-minutes", type=int, default=120)
    subparsers.add_parser("combine")
    args = parser.parse_args()
    if args.command == "worker":
        run_fold(args.fold, args.smoke)
    elif args.command == "launch":
        launch(
            [int(value) for value in args.gpus.split(",") if value],
            args.timeout_minutes,
        )
    else:
        print(json.dumps(combine(), indent=2))


if __name__ == "__main__":
    main()
