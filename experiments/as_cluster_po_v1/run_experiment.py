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
from sklearn.cluster import KMeans
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
K = 3
SEED = 42
REPRESENTATIONS = [
    "as_only",
    "body_ft",
    "residual",
    "random_body_ft",
    "random_residual",
]


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


def train_as(model, train_contexts, labels, tensors, device, epochs):
    import torch
    import torch.nn.functional as functional

    grouped_labels = labels.groupby("context_id")["label"].apply(list).to_dict()
    train_assets = sorted(train_contexts["Asset"].unique())
    parameters = [
        *model.body.parameters(),
        *model.as_head.parameters(),
    ]
    for asset in train_assets:
        parameters.extend(model.input_adapters[asset].parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
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


def adapt_test_io(model, test_contexts, tensors, targets, device, epochs):
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
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
        rows = group.reset_index(drop=True)
        for _ in range(epochs):
            for row in rows.itertuples(index=False):
                optimizer.zero_grad(set_to_none=True)
                logits = model(asset, tensors[row.context_id].to(device))
                loss = shared.po_loss(logits, targets[row.context_id])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
        for parameter in parameters:
            parameter.requires_grad = False
    for parameter in model.parameters():
        parameter.requires_grad = True


def residual_adapter():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class ResidualAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(D_MODEL, 32)
            self.up = nn.Linear(32, D_MODEL)
            self.scale = nn.Parameter(torch.zeros(1))

        def forward(self, hidden):
            residual = self.up(functional.silu(self.down(hidden)))
            return hidden + self.scale * residual

    return ResidualAdapter()


def encode_path(model, asset, values, body, adapter=None):
    encoded = model.encode(asset, values, body=body)
    if adapter is not None:
        encoded = adapter(encoded)
    return encoded


def train_cluster_po(
    model,
    body,
    samples,
    tensors,
    targets,
    device,
    epochs,
    learning_rate,
    adapter=None,
):
    import torch

    rows = samples.reset_index(drop=True)
    if rows.empty or epochs <= 0:
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in body.parameters():
        parameter.requires_grad = False
    if adapter is None:
        for parameter in body.parameters():
            parameter.requires_grad = True
        parameters = list(body.parameters())
    else:
        for parameter in adapter.parameters():
            parameter.requires_grad = True
        parameters = list(adapter.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        for index in rng.permutation(len(rows)):
            row = rows.iloc[index]
            optimizer.zero_grad(set_to_none=True)
            hidden = encode_path(
                model,
                row.Asset,
                tensors[row.context_id].to(device),
                body,
                adapter,
            )[-1]
            logits = model.output_heads[row.Asset](hidden)
            loss = shared.po_loss(logits, targets[row.context_id])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite cluster PO loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad = True
    for parameter in body.parameters():
        parameter.requires_grad = False
    if adapter is not None:
        for parameter in adapter.parameters():
            parameter.requires_grad = False


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
                    list(range(financial_count, financial_count + D_MODEL)),
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
                    rf_feature_transform(config, financial_count, fold),
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
    representation,
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
    else:
        columns = financial_columns + hidden_columns
        financial_count = len(financial_columns)
    train_indices = np.flatnonzero(data["Asset"].isin(train_assets))
    test_indices = np.flatnonzero(data["Asset"].isin(test_assets))
    values = data[columns].to_numpy(dtype=float)
    targets = data[ALGORITHMS].astype(float)
    fitted = Parallel(n_jobs=9)(
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
    predictions = pd.DataFrame(index=test_indices, columns=ALGORITHMS, dtype=float)
    for algorithm, indices, prediction in fitted:
        predictions.loc[indices, algorithm] = prediction
    chosen = predictions.idxmax(axis=1)
    actual = targets.iloc[test_indices].to_numpy()
    chosen_indices = np.asarray([ALGORITHMS.index(value) for value in chosen])
    selection = actual[np.arange(len(actual)), chosen_indices]
    vbs = actual.max(axis=1)
    return {
        "fold": fold,
        "criterion": criterion,
        "representation": representation,
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
        GroupKFold(n_splits=5).split(mapping, groups=mapping["Asset"])
    )[fold]
    train_assets = set(mapping.iloc[split[0]]["Asset"].unique())
    test_assets = set(mapping.iloc[split[1]]["Asset"].unique())
    train_contexts = contexts[contexts["Asset"].isin(train_assets)].reset_index(
        drop=True
    )
    test_contexts = contexts[contexts["Asset"].isin(test_assets)].reset_index(
        drop=True
    )
    summary_columns = [f"s_{i}" for i in range(8)]
    scaler = StandardScaler().fit(train_contexts[summary_columns])
    kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=20).fit(
        scaler.transform(train_contexts[summary_columns])
    )
    train_contexts["cluster"] = kmeans.labels_
    test_contexts["cluster"] = kmeans.predict(
        scaler.transform(test_contexts[summary_columns])
    )
    rng = np.random.default_rng(SEED + fold)
    train_contexts["random_cluster"] = rng.integers(0, K, len(train_contexts))
    test_contexts["random_cluster"] = rng.integers(0, K, len(test_contexts))
    cluster_of = {
        **dict(zip(train_contexts["context_id"], train_contexts["cluster"])),
        **dict(zip(test_contexts["context_id"], test_contexts["cluster"])),
    }
    random_of = {
        **dict(
            zip(train_contexts["context_id"], train_contexts["random_cluster"])
        ),
        **dict(
            zip(test_contexts["context_id"], test_contexts["random_cluster"])
        ),
    }

    pretrained, _ = shared.build_network(contexts)
    pretrained = pretrained.to(device)
    pretrained.as_head = nn.Linear(D_MODEL, len(ALGORITHMS)).to(device)
    po_epochs = 2 if smoke else 10
    as_epochs = 2 if smoke else 20
    adapter_epochs = 1 if smoke else 10
    cluster_epochs = 1 if smoke else 5
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
    summaries = []
    label_audit = []
    fin_source = pd.read_csv(ROOT / "RF_combined/All_AR_LLM_combined.csv")
    financial_columns = fin_source.columns[1:25].tolist() + ["Period"]

    for criterion in criteria:
        model = copy.deepcopy(pretrained)
        model.as_head = nn.Linear(D_MODEL, len(ALGORITHMS)).to(device)
        labels = outcome_labels(mapping, criterion)
        train_labels = labels[labels["Asset"].isin(train_assets)]
        train_as(
            model,
            train_contexts,
            train_labels,
            tensors,
            device,
            as_epochs,
        )
        adapt_test_io(
            model,
            test_contexts,
            tensors,
            targets,
            device,
            adapter_epochs,
        )
        as_body = copy.deepcopy(model.body)
        real_bodies = [copy.deepcopy(as_body).to(device) for _ in range(K)]
        random_bodies = [copy.deepcopy(as_body).to(device) for _ in range(K)]
        real_adapters = [residual_adapter().to(device) for _ in range(K)]
        random_adapters = [residual_adapter().to(device) for _ in range(K)]
        for cluster in range(K):
            real_rows = train_contexts[
                train_contexts["cluster"] == cluster
            ].reset_index(drop=True)
            random_rows = train_contexts[
                train_contexts["random_cluster"] == cluster
            ].reset_index(drop=True)
            train_cluster_po(
                model,
                real_bodies[cluster],
                real_rows,
                tensors,
                targets,
                device,
                cluster_epochs,
                1e-5,
            )
            train_cluster_po(
                model,
                random_bodies[cluster],
                random_rows,
                tensors,
                targets,
                device,
                cluster_epochs,
                1e-5,
            )
            train_cluster_po(
                model,
                as_body,
                real_rows,
                tensors,
                targets,
                device,
                cluster_epochs,
                1e-4,
                adapter=real_adapters[cluster],
            )
            train_cluster_po(
                model,
                as_body,
                random_rows,
                tensors,
                targets,
                device,
                cluster_epochs,
                1e-4,
                adapter=random_adapters[cluster],
            )

        model.eval()
        records = []
        with torch.no_grad():
            for row in contexts.itertuples(index=False):
                cluster = int(cluster_of[row.context_id])
                random_cluster = int(random_of[row.context_id])
                values = tensors[row.context_id].to(device)
                hidden_map = {
                    "as_only": encode_path(model, row.Asset, values, as_body)[-1],
                    "body_ft": encode_path(
                        model, row.Asset, values, real_bodies[cluster]
                    )[-1],
                    "residual": encode_path(
                        model,
                        row.Asset,
                        values,
                        as_body,
                        real_adapters[cluster],
                    )[-1],
                    "random_body_ft": encode_path(
                        model,
                        row.Asset,
                        values,
                        random_bodies[random_cluster],
                    )[-1],
                    "random_residual": encode_path(
                        model,
                        row.Asset,
                        values,
                        as_body,
                        random_adapters[random_cluster],
                    )[-1],
                }
                record = {
                    "context_id": row.context_id,
                    "Asset": row.Asset,
                    "cluster": cluster,
                    "random_cluster": random_cluster,
                }
                for prefix, hidden in hidden_map.items():
                    vector = hidden.detach().cpu().numpy()
                    record.update(
                        {
                            f"{prefix}_h_{index:03d}": float(value)
                            for index, value in enumerate(vector)
                        }
                    )
                records.append(record)
        instance_frame = mapping[["instance", "context_id", "Asset"]].merge(
            pd.DataFrame(records),
            on=["context_id", "Asset"],
            validate="many_to_one",
        )
        output_dir = OUTPUTS / ("smoke" if smoke else "full")
        atomic_csv(
            output_dir / f"fold_{fold}_{criterion}_features.csv",
            instance_frame,
        )
        summaries.append(
            evaluate_rf(
                instance_frame,
                train_assets,
                test_assets,
                financial_columns,
                [f"as_only_h_{index:03d}" for index in range(D_MODEL)],
                criterion,
                "Fin",
                fold,
                "fin",
            )
        )
        for representation in REPRESENTATIONS:
            hidden_columns = [
                f"{representation}_h_{index:03d}" for index in range(D_MODEL)
            ]
            summaries.append(
                evaluate_rf(
                    instance_frame,
                    train_assets,
                    test_assets,
                    financial_columns,
                    hidden_columns,
                    criterion,
                    "Fin+EmbeddingPCA32",
                    fold,
                    representation,
                )
            )
        label_audit.append(
            {
                "criterion": criterion,
                "train_label_instances": int(len(train_labels)),
                "test_label_instances_used": 0,
                "train_cluster_counts": {
                    int(key): int(value)
                    for key, value in train_contexts["cluster"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            }
        )

    output_dir = OUTPUTS / ("smoke" if smoke else "full")
    atomic_csv(output_dir / f"fold_{fold}_summary.csv", pd.DataFrame(summaries))
    atomic_json(
        output_dir / f"fold_{fold}_metadata.json",
        {
            "fold": fold,
            "smoke": smoke,
            "K": K,
            "train_assets": sorted(train_assets),
            "test_assets": sorted(test_assets),
            "po_pretrain_epochs": po_epochs,
            "as_finetune_epochs": as_epochs,
            "cluster_po_epochs": cluster_epochs,
            "test_adapter_epochs": adapter_epochs,
            "body_ft_lr": 1e-5,
            "residual_lr": 1e-4,
            "label_audit": label_audit,
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
    paths = [OUTPUTS / "full" / f"fold_{fold}_summary.csv" for fold in range(5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"complete": False, "missing": missing}
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = []
    for keys, group in rows.groupby(
        ["criterion", "representation", "configuration"]
    ):
        weights = group["test_instances"].to_numpy()
        summary.append(
            {
                "criterion": keys[0],
                "representation": keys[1],
                "configuration": keys[2],
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
        "protocol": (
            "grouped 5-fold; AS labels from train assets only; "
            "cluster PO body fine-tune and residual adapters on train contexts; "
            "test I/O adapters use OH-T PO targets only"
        ),
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
            running[process] = {"fold": fold, "gpu": gpu, "handle": handle}
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
