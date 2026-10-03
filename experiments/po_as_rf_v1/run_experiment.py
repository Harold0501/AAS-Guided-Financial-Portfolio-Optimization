from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


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
METHODS = ["A_AS", "C_MS_T", "C_OH_T"]
WINDOW = 252
D_MODEL = 256
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
    mapping["context_key"] = (
        mapping["Asset"] + "__" + mapping["start"].dt.strftime("%Y-%m-%d")
    )
    if len(mapping) != 762 or mapping["instance"].nunique() != 762:
        raise RuntimeError("Expected 762 unique instances")
    # Chronological split inside each asset; all horizons sharing a start date
    # stay in the same split.
    mapping["split"] = "train"
    for asset, group in mapping.groupby("Asset", sort=True):
        starts = sorted(group["start"].unique())
        test_starts = set(starts[max(1, int(np.ceil(len(starts) * 0.8))) :])
        mapping.loc[
            (mapping["Asset"] == asset) & mapping["start"].isin(test_starts),
            "split",
        ] = "test"
    return mapping


def load_outcomes(criterion: str) -> pd.DataFrame:
    source = pd.read_csv(
        ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv"
    )
    if not np.isfinite(source[ALGORITHMS].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite outcomes for {criterion}")
    source["label"] = source[ALGORITHMS].to_numpy().argmax(axis=1)
    return source[["instance", *ALGORITHMS, "label"]]


def load_asset_data(asset: str):
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
    return data.fillna(0), raw


def instance_tensor(data: pd.DataFrame, start: pd.Timestamp):
    import torch

    dates = data.index[data.index < start]
    dates = dates[-WINDOW:]
    if len(dates) == 0:
        raise RuntimeError(f"No history before {start} in instance")
    values = data.loc[dates].to_numpy(dtype=np.float32)
    return torch.tensor(np.nan_to_num(values), dtype=torch.float32)


def month_keys(data: pd.DataFrame) -> list[str]:
    periods = data.index.to_period("M")
    return sorted({str(period) for period in periods})


def masked_portfolio_loss(prediction, target, missing_mask):
    import torch

    return -(
        torch.where(missing_mask, -torch.inf, prediction)
        .log_softmax(dim=-1)
        .nan_to_num(0)
        * torch.where(missing_mask, 0, target)
    ).sum()


def build_model(input_size: int, num_assets: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class OriginalPOTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(input_size, D_MODEL, bias=True)
            self.positional_embedding = nn.Embedding(WINDOW, D_MODEL)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=8,
                dim_feedforward=D_MODEL,
                activation=functional.silu,
            )
            self.trf = nn.TransformerEncoder(layer, num_layers=6)
            self.portfolio_head = nn.Linear(D_MODEL, num_assets, bias=True)
            self.as_head = nn.Linear(D_MODEL, len(ALGORITHMS), bias=True)

        def encode(self, x):
            x = functional.silu(self.l1(x))
            positions = torch.arange(x.shape[0], device=x.device).flip(0)
            x = x + self.positional_embedding(positions)
            return functional.silu(
                self.trf(
                    x,
                    mask=causal_mask(x.shape[0], x.device),
                    is_causal=True,
                )
            )

        def portfolio(self, x):
            return self.portfolio_head(self.encode(x)[-1])

        def classify(self, x):
            return self.as_head(self.encode(x)[-1])

    def causal_mask(size, device):
        return (
            torch.triu(torch.ones((size, size), dtype=torch.bool, device=device))
            != 1
        ).transpose(0, 1)

    return OriginalPOTransformer()


def load_target_weights(asset: str, target: str):
    path = TARGET_DIR / f"{asset}_weights_{target}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def pretrain_po(
    model,
    asset: str,
    data: pd.DataFrame,
    target: str,
    device,
    epochs: int,
):
    import torch

    targets = load_target_weights(asset, target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    dates = data.index
    missing_by_month = {}
    for key in targets:
        period = pd.Period(key)
        month_dates = dates[dates.to_period("M") == period]
        if len(month_dates) == 0:
            continue
        missing_by_month[key] = torch.tensor(
            data["log_tr"].loc[month_dates[-1]].isna().values,
            dtype=torch.bool,
            device=device,
        )
    usable = [key for key in targets if key in missing_by_month]
    for _ in range(epochs):
        np.random.default_rng(SEED).shuffle(usable)
        for key in usable:
            period = pd.Period(key)
            month_dates = dates[dates.to_period("M") <= period]
            x = torch.tensor(
                data.loc[month_dates].to_numpy(dtype=np.float32)[-WINDOW:],
                device=device,
            )
            target_tensor = targets[key].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_portfolio_loss(
                model.portfolio(x),
                target_tensor,
                missing_by_month[key],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite PO loss for {asset}/{target}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def train_as(
    model,
    samples: list[tuple[int, object, int]],
    device,
    epochs: int,
    freeze_encoder: bool,
):
    import torch
    import torch.nn.functional as functional

    if freeze_encoder:
        for parameter in model.l1.parameters():
            parameter.requires_grad = False
        for parameter in model.positional_embedding.parameters():
            parameter.requires_grad = False
        for parameter in model.trf.parameters():
            parameter.requires_grad = False
        parameters = model.as_head.parameters()
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
    for _ in range(epochs):
        order = np.random.permutation(len(samples))
        for index in order:
            _, x, label = samples[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(x.to(device))
            loss = functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([label], dtype=torch.long, device=device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite AS loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
    return model


def run_asset(asset: str, smoke: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mapping = load_mapping()
    local = mapping[mapping["Asset"] == asset].copy()
    data, _ = load_asset_data(asset)
    num_assets = len(data["log_tr"].columns)
    outcomes = {criterion: load_outcomes(criterion) for criterion in CRITERIA}
    outcome_maps = {
        criterion: outcomes[criterion].set_index("instance")
        for criterion in CRITERIA
    }
    methods = METHODS
    criteria = ["AR"] if smoke else CRITERIA
    epochs = 2 if smoke else 30
    pretrain_epochs = 1 if smoke else 10
    started = time.monotonic()

    for method in methods:
        pretrained_model = None
        if method == "C_MS_T":
            pretrained_model = build_model(len(data.columns), num_assets).to(device)
            pretrained_model = pretrain_po(
                pretrained_model,
                asset,
                data,
                "maxsortino",
                device,
                pretrain_epochs,
            )
        elif method == "C_OH_T":
            pretrained_model = build_model(len(data.columns), num_assets).to(device)
            pretrained_model = pretrain_po(
                pretrained_model,
                asset,
                data,
                "onehot",
                device,
                pretrain_epochs,
            )

        for criterion in criteria:
            model = (
                copy.deepcopy(pretrained_model)
                if pretrained_model is not None
                else build_model(len(data.columns), num_assets).to(device)
            )
            samples = []
            for row in local.itertuples(index=False):
                label = int(outcome_maps[criterion].loc[row.instance, "label"])
                x = instance_tensor(data, row.start)
                samples.append((int(row.instance), x, label))
            train_samples = [
                sample
                for sample, split in zip(
                    samples, local["split"].tolist()
                )
                if split == "train"
            ]
            test_ids = {
                int(row.instance)
                for row in local.itertuples(index=False)
                if row.split == "test"
            }
            model = train_as(
                model,
                train_samples,
                device,
                epochs,
                freeze_encoder=method != "A_AS",
            )
            model.eval()
            rows = []
            with torch.no_grad():
                for instance, x, label in samples:
                    hidden = model.encode(x.to(device))[-1].cpu().numpy()
                    row = {
                        "instance": instance,
                        "Asset": asset,
                        "criterion": criterion,
                        "method": method,
                        "split": "test" if instance in test_ids else "train",
                        "as_label": label,
                    }
                    row.update({f"h_{i:03d}": float(value) for i, value in enumerate(hidden)})
                    rows.append(row)
            output = OUTPUTS / "embeddings" / method / criterion / f"{asset}.csv"
            atomic_csv(output, pd.DataFrame(rows))
            atomic_json(
                output.with_suffix(".json"),
                {
                    "asset": asset,
                    "criterion": criterion,
                    "method": method,
                    "instances": len(rows),
                    "train_instances": len(train_samples),
                    "test_instances": len(test_ids),
                    "pretrained": method != "A_AS",
                    "freeze_encoder": method != "A_AS",
                    "epochs": epochs,
                    "pretrain_epochs": pretrain_epochs if method != "A_AS" else 0,
                    "completed_at": pd.Timestamp.utcnow().isoformat(),
                },
            )
    print(
        json.dumps(
            {
                "asset": asset,
                "methods": methods,
                "criteria": criteria,
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def combine_and_select() -> None:
    from sklearn.ensemble import RandomForestRegressor

    mapping = load_mapping()
    summaries = []
    for method in METHODS:
        for criterion in CRITERIA:
            files = list(
                (OUTPUTS / "embeddings" / method / criterion).glob("asset_*.csv")
            )
            if len(files) != mapping["Asset"].nunique():
                continue
            embeddings = pd.concat([pd.read_csv(path) for path in files])
            outcomes = load_outcomes(criterion).merge(
                mapping[["instance", "Asset", "split"]],
                on="instance",
                validate="one_to_one",
            )
            frame = outcomes.merge(
                embeddings.drop(columns=["Asset", "criterion", "method", "split"]),
                on="instance",
                validate="one_to_one",
            )
            train = frame[frame["split"] == "train"]
            test = frame[frame["split"] == "test"]
            feature_columns = [f"h_{i:03d}" for i in range(D_MODEL)]
            predictions = pd.DataFrame(index=test.index, columns=ALGORITHMS)
            for algorithm in ALGORITHMS:
                model = RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=2,
                    random_state=SEED,
                    n_jobs=-1,
                )
                model.fit(train[feature_columns], train[algorithm])
                predictions.loc[:, algorithm] = model.predict(
                    test[feature_columns]
                )
            chosen = predictions.idxmax(axis=1)
            actual = test[ALGORITHMS].to_numpy()
            selected_index = np.array([ALGORITHMS.index(value) for value in chosen])
            selection = actual[np.arange(len(test)), selected_index]
            vbs = actual.max(axis=1)
            summaries.append(
                {
                    "method": method,
                    "criterion": criterion,
                    "test_instances": len(test),
                    "selection_mean": float(selection.mean()),
                    "regret_mean": float((vbs - selection).mean()),
                    "accuracy": float(
                        (selected_index == actual.argmax(axis=1)).mean()
                    ),
                }
            )
    atomic_csv(OUTPUTS / "rf_summary.csv", pd.DataFrame(summaries))


def launch(gpus: list[int], timeout_minutes: int) -> None:
    if len(gpus) > 4 or not gpus:
        raise ValueError("Use one to four GPUs")
    assets = sorted(
        load_mapping()["Asset"].unique(),
        key=lambda value: int(value.split("_")[1]),
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = list(assets)
    running, completed, failed = {}, [], []
    deadline = time.monotonic() + timeout_minutes * 60
    while pending or running:
        if time.monotonic() >= deadline:
            for process in running:
                process.terminate()
            failed.extend(
                {"asset": info["asset"], "reason": "timeout"}
                for info in running.values()
            )
            break
        free = [gpu for gpu in gpus if gpu not in {v["gpu"] for v in running.values()}]
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
            running[process] = {"asset": asset, "gpu": gpu, "handle": handle}
        time.sleep(2)
        for process, info in list(running.items()):
            if process.poll() is None:
                continue
            info["handle"].close()
            running.pop(process)
            (completed if process.returncode == 0 else failed).append(info["asset"])
    combine_and_select()
    atomic_json(
        OUTPUTS / "run_state.json",
        {
            "status": "complete" if not failed else "failed",
            "completed": completed,
            "failed": failed,
            "finished_at": pd.Timestamp.utcnow().isoformat(),
        },
    )
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--asset", required=True)
    worker.add_argument("--smoke", action="store_true")
    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--gpus", default="0,1,2,3")
    launcher.add_argument("--timeout-minutes", type=int, default=120)
    subparsers.add_parser("combine")
    args = parser.parse_args()
    if args.command == "worker":
        run_asset(args.asset, args.smoke)
    elif args.command == "launch":
        launch(
            [int(value) for value in args.gpus.split(",") if value],
            args.timeout_minutes,
        )
    else:
        combine_and_select()


if __name__ == "__main__":
    main()
