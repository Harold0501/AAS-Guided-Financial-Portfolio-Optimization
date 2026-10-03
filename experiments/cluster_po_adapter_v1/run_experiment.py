from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
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
MAPPING = ROOT / "LLM/Assets_and_Instances/combined_asset_components_time.csv"
RAW_DIR = ROOT / "LLMD-AS-FPO/LLM_Generated_Feature/data"
TARGET_DIR = Path(
    os.environ.get(
        "SDFLWDE_TARGET_DIR",
        "/data1/home/hz314/workspace/sDFLwDE/new_result",
    )
)
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"
ALGORITHMS = ["EW", "MCW", "mSSRM", "MS-M", "OH-M", "MS-T", "OH-T", "MS-L", "OH-L"]
CRITERIA = ["AR", "CR", "MDD", "SHR", "SOR"]
WINDOW = 252
D_MODEL = 256
K = 3
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


def load_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING)
    mapping["start"] = pd.to_datetime(mapping["start"], format="mixed")
    mapping["end"] = pd.to_datetime(mapping["end"], format="mixed")
    mapping["context_id"] = (
        mapping["Asset"] + "__" + mapping["start"].dt.strftime("%Y-%m")
    )
    mapping["horizon_months"] = (
        (mapping["end"].dt.year - mapping["start"].dt.year) * 12
        + mapping["end"].dt.month
        - mapping["start"].dt.month
    )
    if len(mapping) != 762 or mapping["context_id"].nunique() != 165:
        raise RuntimeError("Expected 762 instances and 165 contexts")
    return mapping


def preprocess_asset(asset: str):
    raw = pd.read_csv(RAW_DIR / f"{asset}.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["company", "date"])
    raw["open"] = np.log(raw["open"] / raw["close"])
    raw["high"] = np.log(raw["high"] / raw["close"])
    raw["low"] = np.log(raw["low"] / raw["close"])
    raw["volume"] = np.log(raw["volume"])
    raw["log_tr"] = np.log(
        raw.groupby("company")["div_adj_close"].pct_change() + 1
    )
    raw = raw.drop(["close", "div_adj_close"], axis=1)
    data = raw.set_index(["company", "date"]).unstack(level=0)
    data = data[(~data["log_tr"].isna()).sum(axis=1) > 1]
    data = data[(~(data["log_tr"].isna() | (data["log_tr"] == 0))).sum(axis=1) > 0]
    return data.fillna(0)


def context_summary(data: pd.DataFrame, start: pd.Timestamp) -> np.ndarray:
    returns = data["log_tr"].loc[data.index < start].iloc[-WINDOW:]
    values = returns.to_numpy(dtype=float)
    correlation = returns.corr().to_numpy(dtype=float)
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    portfolio = returns.mean(axis=1).to_numpy(dtype=float)
    return np.asarray(
        [
            np.log1p(returns.shape[1]),
            np.nanmean(values),
            np.nanstd(values),
            np.nanmedian(np.nanstd(values, axis=0)),
            np.nanmedian(np.nanstd(values, axis=1)),
            np.nanmean(off_diagonal),
            np.mean(values < 0),
            np.nanstd(portfolio),
        ],
        dtype=np.float32,
    )


def context_tensor(data: pd.DataFrame, start: pd.Timestamp):
    import torch

    values = data.loc[data.index < start].to_numpy(dtype=np.float32)[-WINDOW:]
    if len(values) == 0:
        raise RuntimeError(f"No history before {start}")
    return torch.tensor(np.nan_to_num(values), dtype=torch.float32)


def load_target(asset: str, start: pd.Timestamp):
    path = TARGET_DIR / f"{asset}_weights_onehot.pkl"
    with path.open("rb") as handle:
        targets = pickle.load(handle)
    key = start.strftime("%Y-%m")
    if key not in targets:
        raise KeyError(f"Missing OH-T PO target: {asset}/{key}")
    return targets[key].float()


def prepare_contexts():
    mapping = load_mapping()
    contexts = mapping[
        ["context_id", "Asset", "start"]
    ].drop_duplicates().sort_values(["Asset", "start"])
    asset_data = {}
    records = []
    tensors = {}
    targets = {}
    for row in contexts.itertuples(index=False):
        if row.Asset not in asset_data:
            asset_data[row.Asset] = preprocess_asset(row.Asset)
        data = asset_data[row.Asset]
        tensors[row.context_id] = context_tensor(data, row.start)
        targets[row.context_id] = load_target(row.Asset, row.start)
        summary = context_summary(data, row.start)
        record = {
            "context_id": row.context_id,
            "Asset": row.Asset,
            "start": row.start,
            "input_size": tensors[row.context_id].shape[1],
            "num_assets": len(data["log_tr"].columns),
        }
        record.update({f"s_{i}": float(value) for i, value in enumerate(summary)})
        records.append(record)
    return mapping, pd.DataFrame(records), tensors, targets


def build_network(contexts: pd.DataFrame):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    asset_specs = (
        contexts.groupby("Asset")[["input_size", "num_assets"]]
        .first()
        .to_dict("index")
    )

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.position = nn.Embedding(WINDOW, D_MODEL)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=8,
                dim_feedforward=D_MODEL,
                activation=functional.silu,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=6)

        def forward(self, hidden):
            positions = torch.arange(
                hidden.shape[0], device=hidden.device
            ).flip(0)
            hidden = hidden + self.position(positions)
            mask = (
                torch.triu(
                    torch.ones(
                        (hidden.shape[0], hidden.shape[0]),
                        dtype=torch.bool,
                        device=hidden.device,
                    )
                )
                != 1
            ).transpose(0, 1)
            return functional.silu(
                self.encoder(hidden, mask=mask, is_causal=True)
            )

    class MultiAssetPO(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_adapters = nn.ModuleDict(
                {
                    asset: nn.Linear(spec["input_size"], D_MODEL)
                    for asset, spec in asset_specs.items()
                }
            )
            self.output_heads = nn.ModuleDict(
                {
                    asset: nn.Linear(D_MODEL, spec["num_assets"])
                    for asset, spec in asset_specs.items()
                }
            )
            self.body = Body()

        def encode(self, asset, values, body=None):
            hidden = functional.silu(self.input_adapters[asset](values))
            return (body or self.body)(hidden)

        def forward(self, asset, values, body=None):
            hidden = self.encode(asset, values, body)[-1]
            return self.output_heads[asset](hidden)

    return MultiAssetPO(), Body


def po_loss(logits, target):
    return -(logits.log_softmax(dim=-1) * target.to(logits.device)).sum()


def train_samples(
    model,
    samples,
    tensors,
    targets,
    device,
    epochs,
    body=None,
    body_only=False,
):
    import torch

    if body_only:
        parameters = list(body.parameters())
    else:
        parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-4, weight_decay=1e-4
    )
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        for index in rng.permutation(len(samples)):
            row = samples.iloc[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                row.Asset,
                tensors[row.context_id].to(device),
                body=body,
            )
            loss = po_loss(logits, targets[row.context_id])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PO loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()


def adapt_test_assets(
    model,
    samples,
    tensors,
    targets,
    device,
    epochs,
):
    import torch

    for parameter in model.body.parameters():
        parameter.requires_grad = False
    for asset, group in samples.groupby("Asset"):
        parameters = [
            *model.input_adapters[asset].parameters(),
            *model.output_heads[asset].parameters(),
        ]
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
                loss = po_loss(logits, targets[row.context_id])
                loss.backward()
                optimizer.step()
    for parameter in model.body.parameters():
        parameter.requires_grad = True


def rf_predict(train, test, feature_columns, criterion):
    source = pd.read_csv(
        ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv"
    )
    train = train.merge(
        source[["instance", *ALGORITHMS]], on="instance", validate="one_to_one"
    )
    test = test.merge(
        source[["instance", *ALGORITHMS]], on="instance", validate="one_to_one"
    )
    predictions = pd.DataFrame(index=test.index, columns=ALGORITHMS)
    for algorithm in ALGORITHMS:
        model = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=4,
        )
        model.fit(train[feature_columns], train[algorithm])
        predictions[algorithm] = model.predict(test[feature_columns])
    chosen = predictions.idxmax(axis=1)
    actual = test[ALGORITHMS].to_numpy()
    indices = np.asarray([ALGORITHMS.index(value) for value in chosen])
    selection = actual[np.arange(len(test)), indices]
    vbs = actual.max(axis=1)
    return chosen, selection, vbs


def run_fold(fold: int, smoke: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    mapping, contexts, tensors, targets = prepare_contexts()
    assets = sorted(mapping["Asset"].unique())
    train_asset_indices, test_asset_indices = list(
        GroupKFold(n_splits=5).split(assets, groups=assets)
    )[fold]
    train_assets = {assets[index] for index in train_asset_indices}
    test_assets = {assets[index] for index in test_asset_indices}
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
    rng = np.random.default_rng(SEED)
    train_contexts["random_cluster"] = rng.integers(
        0, K, len(train_contexts)
    )
    test_contexts["random_cluster"] = rng.integers(0, K, len(test_contexts))

    model, Body = build_network(contexts)
    model = model.to(device)
    global_epochs = 2 if smoke else 30
    cluster_epochs = 1 if smoke else 15
    adapter_epochs = 1 if smoke else 15
    started = time.monotonic()
    train_samples(
        model,
        train_contexts,
        tensors,
        targets,
        device,
        global_epochs,
    )
    adapt_test_assets(
        model,
        test_contexts,
        tensors,
        targets,
        device,
        adapter_epochs,
    )
    cluster_bodies = [copy.deepcopy(model.body).to(device) for _ in range(K)]
    random_bodies = [copy.deepcopy(model.body).to(device) for _ in range(K)]
    for cluster in range(K):
        selected = train_contexts[train_contexts["cluster"] == cluster].reset_index(
            drop=True
        )
        if len(selected):
            train_samples(
                model,
                selected,
                tensors,
                targets,
                device,
                cluster_epochs,
                body=cluster_bodies[cluster],
                body_only=True,
            )
        random_selected = train_contexts[
            train_contexts["random_cluster"] == cluster
        ].reset_index(drop=True)
        if len(random_selected):
            train_samples(
                model,
                random_selected,
                tensors,
                targets,
                device,
                cluster_epochs,
                body=random_bodies[cluster],
                body_only=True,
            )

    model.eval()
    records = []
    with torch.no_grad():
        for row in contexts.itertuples(index=False):
            split = "train" if row.Asset in train_assets else "test"
            lookup = train_contexts if split == "train" else test_contexts
            assignment = lookup.set_index("context_id").loc[row.context_id]
            values = tensors[row.context_id].to(device)
            global_hidden = model.encode(row.Asset, values)[-1].cpu().numpy()
            cluster_hidden = model.encode(
                row.Asset,
                values,
                cluster_bodies[int(assignment.cluster)],
            )[-1].cpu().numpy()
            random_hidden = model.encode(
                row.Asset,
                values,
                random_bodies[int(assignment.random_cluster)],
            )[-1].cpu().numpy()
            record = {
                "context_id": row.context_id,
                "Asset": row.Asset,
                "split": split,
                "cluster": int(assignment.cluster),
                "random_cluster": int(assignment.random_cluster),
            }
            for prefix, hidden in [
                ("global", global_hidden),
                ("clustered", cluster_hidden),
                ("random", random_hidden),
            ]:
                record.update(
                    {
                        f"{prefix}_h_{index:03d}": float(value)
                        for index, value in enumerate(hidden)
                    }
                )
            records.append(record)
    context_features = pd.DataFrame(records)
    instance_features = mapping[
        ["instance", "context_id", "Asset", "horizon_months"]
    ].merge(context_features, on=["context_id", "Asset"], validate="many_to_one")
    output_dir = OUTPUTS / ("smoke" if smoke else "full")
    atomic_csv(output_dir / f"fold_{fold}_features.csv", instance_features)

    summaries = []
    fin_source = pd.read_csv(
        ROOT / "RF_combined/All_AR_LLM_combined.csv"
    )
    financial_columns = fin_source.columns[1:25].tolist() + ["Period"]
    financial = fin_source[["instance", *financial_columns]]
    frame = instance_features.merge(financial, on="instance", validate="one_to_one")
    train_frame = frame[frame["split"] == "train"].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    for criterion in CRITERIA:
        _, selection, vbs = rf_predict(
            train_frame,
            test_frame,
            financial_columns,
            criterion,
        )
        summaries.append(
            {
                "fold": fold,
                "representation": "fin_only",
                "rf": "global",
                "criterion": criterion,
                "test_instances": len(test_frame),
                "selection_mean": float(selection.mean()),
                "regret_mean": float((vbs - selection).mean()),
            }
        )
    for representation in ["global", "clustered", "random"]:
        embedding_columns = [
            f"{representation}_h_{index:03d}" for index in range(D_MODEL)
        ]
        for criterion in CRITERIA:
            chosen, selection, vbs = rf_predict(
                train_frame,
                test_frame,
                financial_columns + embedding_columns,
                criterion,
            )
            summaries.append(
                {
                    "fold": fold,
                    "representation": representation,
                    "rf": "global",
                    "criterion": criterion,
                    "test_instances": len(test_frame),
                    "selection_mean": float(selection.mean()),
                    "regret_mean": float((vbs - selection).mean()),
                }
            )
            if representation == "clustered":
                selections, vbss = [], []
                for cluster in range(K):
                    cluster_train = train_frame[train_frame["cluster"] == cluster]
                    cluster_test = test_frame[test_frame["cluster"] == cluster]
                    if cluster_test.empty:
                        continue
                    _, cluster_selection, cluster_vbs = rf_predict(
                        cluster_train,
                        cluster_test,
                        financial_columns + embedding_columns,
                        criterion,
                    )
                    selections.extend(cluster_selection)
                    vbss.extend(cluster_vbs)
                summaries.append(
                    {
                        "fold": fold,
                        "representation": representation,
                        "rf": "cluster_specific",
                        "criterion": criterion,
                        "test_instances": len(selections),
                        "selection_mean": float(np.mean(selections)),
                        "regret_mean": float(
                            np.mean(np.asarray(vbss) - np.asarray(selections))
                        ),
                    }
                )
    atomic_csv(output_dir / f"fold_{fold}_summary.csv", pd.DataFrame(summaries))
    atomic_json(
        output_dir / f"fold_{fold}_metadata.json",
        {
            "fold": fold,
            "smoke": smoke,
            "K": K,
            "train_assets": sorted(train_assets),
            "test_assets": sorted(test_assets),
            "train_cluster_counts": train_contexts["cluster"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "test_cluster_counts": test_contexts["cluster"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "global_epochs": global_epochs,
            "cluster_epochs": cluster_epochs,
            "adapter_epochs": adapter_epochs,
            "elapsed_seconds": time.monotonic() - started,
            "test_adapter_rule": "PO-only adaptation; no AS outcomes",
        },
    )
    print(
        json.dumps(
            {
                "fold": fold,
                "elapsed_seconds": time.monotonic() - started,
                "summaries": len(summaries),
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
    weighted = []
    for keys, group in rows.groupby(["representation", "rf", "criterion"]):
        weight = group["test_instances"].to_numpy()
        weighted.append(
            {
                "representation": keys[0],
                "rf": keys[1],
                "criterion": keys[2],
                "test_instances": int(weight.sum()),
                "selection_mean": float(
                    np.average(group["selection_mean"], weights=weight)
                ),
                "regret_mean": float(
                    np.average(group["regret_mean"], weights=weight)
                ),
            }
        )
    summary = pd.DataFrame(weighted)
    atomic_csv(OUTPUTS / "summary.csv", summary)
    report = {
        "complete": True,
        "K": K,
        "summary": weighted,
        "protocol": "asset-family grouped 5-fold; train-fold clustering; PO-only test I/O adapter adaptation",
    }
    atomic_json(OUTPUTS / "validation_report.json", report)
    return report


def run_cluster_rf_control(
    train_frame,
    test_frame,
    assignment_column,
    feature_columns,
    criterion,
):
    selections, vbss = [], []
    for cluster in range(K):
        cluster_train = train_frame[
            train_frame[assignment_column] == cluster
        ]
        cluster_test = test_frame[
            test_frame[assignment_column] == cluster
        ]
        if cluster_test.empty:
            continue
        _, selection, vbs = rf_predict(
            cluster_train,
            cluster_test,
            feature_columns,
            criterion,
        )
        selections.extend(selection)
        vbss.extend(vbs)
    return np.asarray(selections), np.asarray(vbss)


def evaluate_controls() -> None:
    rows = []
    fin_source = pd.read_csv(ROOT / "RF_combined/All_AR_LLM_combined.csv")
    financial_columns = fin_source.columns[1:25].tolist() + ["Period"]
    financial = fin_source[["instance", *financial_columns]]
    for fold in range(5):
        frame = pd.read_csv(OUTPUTS / "full" / f"fold_{fold}_features.csv").merge(
            financial, on="instance", validate="one_to_one"
        )
        train = frame[frame["split"] == "train"].copy()
        test = frame[frame["split"] == "test"].copy()
        configurations = [
            ("real_cluster_fin_only", "cluster", financial_columns),
            ("random_cluster_fin_only", "random_cluster", financial_columns),
            (
                "random_cluster_random_body",
                "random_cluster",
                financial_columns
                + [f"random_h_{index:03d}" for index in range(D_MODEL)],
            ),
        ]
        for name, assignment, columns in configurations:
            for criterion in CRITERIA:
                selection, vbs = run_cluster_rf_control(
                    train,
                    test,
                    assignment,
                    columns,
                    criterion,
                )
                rows.append(
                    {
                        "fold": fold,
                        "configuration": name,
                        "criterion": criterion,
                        "test_instances": len(selection),
                        "selection_mean": float(selection.mean()),
                        "regret_mean": float((vbs - selection).mean()),
                    }
                )
    detail = pd.DataFrame(rows)
    atomic_csv(OUTPUTS / "control_fold_results.csv", detail)
    aggregate = []
    for keys, group in detail.groupby(["configuration", "criterion"]):
        weights = group["test_instances"].to_numpy()
        aggregate.append(
            {
                "configuration": keys[0],
                "criterion": keys[1],
                "test_instances": int(weights.sum()),
                "selection_mean": float(
                    np.average(group["selection_mean"], weights=weights)
                ),
                "regret_mean": float(
                    np.average(group["regret_mean"], weights=weights)
                ),
            }
        )
    atomic_csv(OUTPUTS / "control_summary.csv", pd.DataFrame(aggregate))


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
    subparsers.add_parser("controls")
    args = parser.parse_args()
    if args.command == "worker":
        run_fold(args.fold, args.smoke)
    elif args.command == "launch":
        launch(
            [int(value) for value in args.gpus.split(",") if value],
            args.timeout_minutes,
        )
    elif args.command == "combine":
        print(json.dumps(combine(), indent=2))
    else:
        evaluate_controls()


if __name__ == "__main__":
    main()
