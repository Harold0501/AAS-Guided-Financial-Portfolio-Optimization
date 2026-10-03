from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAPPING = ROOT / "LLM/Assets_and_Instances/combined_asset_components_time.csv"
RAW_DIR = ROOT / "LLMD-AS-FPO/LLM_Generated_Feature/data"
CACHE = HERE / "cache"
CONTEXTS = CACHE / "contexts"
OUTPUTS = HERE / "outputs"
LOGS = HERE / "logs"

ALGORITHMS = ["EW", "MCW", "mSSRM", "MS-M", "OH-M", "MS-T", "OH-T", "MS-L", "OH-L"]
CRITERIA = ["AR", "CR", "MDD", "SHR", "SOR"]
WINDOW = 240
RAW_FEATURES = 5
MODEL_DIM = 64
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def prepare_context(asset: str, starts: list[pd.Timestamp]) -> list[dict]:
    raw = pd.read_csv(RAW_DIR / f"{asset}.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["company", "date"])
    raw["f_open"] = np.log(raw["open"] / raw["close"])
    raw["f_high"] = np.log(raw["high"] / raw["close"])
    raw["f_low"] = np.log(raw["low"] / raw["close"])
    raw["f_volume"] = raw.groupby("company")["volume"].transform(
        lambda values: np.log1p(values).diff()
    )
    raw["f_return"] = raw.groupby("company")["div_adj_close"].transform(
        lambda values: np.log(values).diff()
    )
    companies = sorted(raw["company"].unique())
    feature_names = ["f_open", "f_high", "f_low", "f_volume", "f_return"]
    matrices = {
        feature: raw.pivot_table(
            index="date",
            columns="company",
            values=feature,
            aggfunc="last",
        )
        .reindex(columns=companies)
        .sort_index()
        for feature in feature_names
    }
    dates = matrices["f_return"].index
    records = []
    for start in starts:
        selected_dates = dates[dates < start][-WINDOW:]
        if len(selected_dates) < WINDOW:
            raise RuntimeError(
                f"{asset}/{start.date()} has only {len(selected_dates)} days"
            )
        values = np.stack(
            [
                matrices[feature].loc[selected_dates].to_numpy(dtype=np.float32)
                for feature in feature_names
            ],
            axis=-1,
        )
        mask = np.isfinite(values).all(axis=-1)
        values = np.where(np.isfinite(values), values, 0).astype(np.float32)
        context_id = f"{asset}__{start.strftime('%Y-%m')}"
        path = CONTEXTS / f"{context_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                values=values,
                mask=mask,
                companies=np.asarray(companies),
                dates=selected_dates.astype(str).to_numpy(),
            )
        temporary.replace(path)
        records.append(
            {
                "context_id": context_id,
                "Asset": asset,
                "start": start.strftime("%Y-%m-%d"),
                "observations": len(selected_dates),
                "stocks": len(companies),
                "valid_fraction": float(mask.mean()),
                "input_first_date": str(selected_dates[0].date()),
                "input_last_date": str(selected_dates[-1].date()),
                "path": str(path),
            }
        )
    return records


def prepare() -> None:
    mapping = pd.read_csv(MAPPING)
    mapping["start"] = pd.to_datetime(mapping["start"], format="mixed")
    mapping["end"] = pd.to_datetime(mapping["end"], format="mixed")
    mapping["horizon_months"] = (
        (mapping["end"].dt.year - mapping["start"].dt.year) * 12
        + mapping["end"].dt.month
        - mapping["start"].dt.month
    )
    mapping["horizon_index"] = mapping["horizon_months"] // 12 - 1
    mapping["context_id"] = (
        mapping["Asset"] + "__" + mapping["start"].dt.strftime("%Y-%m")
    )
    if len(mapping) != 762 or mapping["instance"].nunique() != 762:
        raise RuntimeError("Expected 762 unique instances")
    if not mapping["horizon_index"].between(0, 8).all():
        raise RuntimeError("Expected horizons from one to nine years")

    context_records = []
    for asset, group in mapping.groupby("Asset", sort=True):
        starts = sorted(group["start"].unique())
        context_records.extend(prepare_context(asset, starts))
    contexts = pd.DataFrame(context_records)
    if len(contexts) != 165 or contexts["context_id"].nunique() != 165:
        raise RuntimeError("Expected 165 unique market contexts")
    if contexts["input_last_date"].ge(contexts["start"]).any():
        # The decision date is often a non-trading year end. Equality would still
        # be backward-looking, but this experiment deliberately stops before it.
        bad = contexts[contexts["input_last_date"].ge(contexts["start"])]
        raise RuntimeError(f"Context reaches decision date:\n{bad}")

    labels = mapping[
        [
            "instance",
            "Asset",
            "start",
            "end",
            "context_id",
            "horizon_months",
            "horizon_index",
        ]
    ].copy()
    for criterion in CRITERIA:
        source = pd.read_csv(
            ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv"
        )[["instance", *ALGORITHMS]]
        if not np.isfinite(source[ALGORITHMS].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite outcomes for {criterion}")
        renamed = source.rename(
            columns={algorithm: f"{criterion}__{algorithm}" for algorithm in ALGORITHMS}
        )
        renamed[f"{criterion}__label"] = source[ALGORITHMS].to_numpy().argmax(axis=1)
        labels = labels.merge(renamed, on="instance", validate="one_to_one")

    atomic_csv(CACHE / "contexts.csv", contexts)
    atomic_csv(CACHE / "labels.csv", labels)
    atomic_json(
        CACHE / "manifest.json",
        {
            "created_at": utc_now(),
            "instances": len(labels),
            "contexts": len(contexts),
            "asset_families": int(labels["Asset"].nunique()),
            "window": WINDOW,
            "raw_features": [
                "log_open_close",
                "log_high_close",
                "log_low_close",
                "log_volume_change",
                "log_adjusted_return",
            ],
            "criteria": CRITERIA,
            "algorithms": ALGORITHMS,
            "input_rule": "last 240 trading days strictly before decision date",
        },
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "instances": len(labels),
                "contexts": len(contexts),
                "min_valid_fraction": contexts["valid_fraction"].min(),
            }
        )
    )


def load_contexts() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    contexts = pd.read_csv(CACHE / "contexts.csv")
    result = {}
    for row in contexts.itertuples(index=False):
        with np.load(row.path) as data:
            result[row.context_id] = (
                data["values"].astype(np.float32),
                data["mask"].astype(bool),
            )
    return result


def split_indices(labels: pd.DataFrame, fold: int):
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    outer = list(
        GroupKFold(n_splits=5).split(labels, groups=labels["Asset"])
    )
    outer_train, test = outer[fold]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED + fold
    )
    subtrain_local, validation_local = next(
        splitter.split(
            labels.iloc[outer_train],
            groups=labels.iloc[outer_train]["Asset"],
        )
    )
    subtrain = outer_train[subtrain_local]
    validation = outer_train[validation_local]
    return subtrain, validation, test


def normalization_stats(
    labels: pd.DataFrame,
    contexts: dict[str, tuple[np.ndarray, np.ndarray]],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context_ids = labels.iloc[train_indices]["context_id"].unique()
    count = np.zeros(RAW_FEATURES, dtype=np.float64)
    total = np.zeros(RAW_FEATURES, dtype=np.float64)
    total_square = np.zeros(RAW_FEATURES, dtype=np.float64)
    for context_id in context_ids:
        values, mask = contexts[context_id]
        selected = values[mask]
        count += len(selected)
        total += selected.sum(axis=0, dtype=np.float64)
        total_square += np.square(selected, dtype=np.float64).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_square / count - mean**2, 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def train_fold(fold: int, smoke: bool) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)

    labels = pd.read_csv(CACHE / "labels.csv")
    contexts = load_contexts()
    train_indices, validation_indices, test_indices = split_indices(labels, fold)
    mean, std = normalization_stats(labels, contexts, train_indices)

    class InstanceDataset(Dataset):
        def __init__(self, indices):
            self.indices = np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            row_index = int(self.indices[item])
            row = labels.iloc[row_index]
            values, mask = contexts[row["context_id"]]
            normalized = np.where(
                mask[..., None],
                (values - mean) / std,
                0,
            ).astype(np.float32)
            targets = np.asarray(
                [row[f"{criterion}__label"] for criterion in CRITERIA],
                dtype=np.int64,
            )
            return {
                "row_index": row_index,
                "values": normalized,
                "mask": mask,
                "horizon": int(row["horizon_index"]),
                "targets": targets,
            }

    def collate(batch):
        max_stocks = max(item["values"].shape[1] for item in batch)
        values = np.zeros(
            (len(batch), WINDOW, max_stocks, RAW_FEATURES), dtype=np.float32
        )
        mask = np.zeros((len(batch), WINDOW, max_stocks), dtype=bool)
        for index, item in enumerate(batch):
            stocks = item["values"].shape[1]
            values[index, :, :stocks] = item["values"]
            mask[index, :, :stocks] = item["mask"]
        return {
            "row_index": torch.tensor(
                [item["row_index"] for item in batch], dtype=torch.long
            ),
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "horizon": torch.tensor(
                [item["horizon"] for item in batch], dtype=torch.long
            ),
            "targets": torch.tensor(
                np.stack([item["targets"] for item in batch]), dtype=torch.long
            ),
        }

    class DirectSelector(nn.Module):
        def __init__(self):
            super().__init__()
            self.stock_encoder = nn.Sequential(
                nn.Linear(RAW_FEATURES, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
            )
            self.day_projection = nn.Linear(64, MODEL_DIM)
            self.position = nn.Embedding(WINDOW, MODEL_DIM)
            layer = nn.TransformerEncoderLayer(
                d_model=MODEL_DIM,
                nhead=4,
                dim_feedforward=128,
                dropout=0.2,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.horizon = nn.Embedding(9, MODEL_DIM)
            self.normalization = nn.LayerNorm(MODEL_DIM)
            self.heads = nn.ModuleList(
                [nn.Linear(MODEL_DIM, len(ALGORITHMS)) for _ in CRITERIA]
            )

        def forward(self, values, stock_mask, horizon):
            stock_hidden = self.stock_encoder(values)
            weights = stock_mask.unsqueeze(-1).to(stock_hidden.dtype)
            denominator = weights.sum(dim=2).clamp_min(1)
            stock_mean = (stock_hidden * weights).sum(dim=2) / denominator
            centered = stock_hidden - stock_mean.unsqueeze(2)
            stock_variance = (
                centered.square() * weights
            ).sum(dim=2) / denominator
            daily = self.day_projection(
                torch.cat([stock_mean, stock_variance.sqrt()], dim=-1)
            )
            positions = torch.arange(WINDOW, device=values.device)
            daily = daily + self.position(positions).unsqueeze(0)
            day_mask = ~stock_mask.any(dim=2)
            encoded = self.temporal_encoder(
                daily, src_key_padding_mask=day_mask
            )
            state = self.normalization(
                encoded[:, -1] + self.horizon(horizon)
            )
            return torch.stack([head(state) for head in self.heads], dim=1)

    batch_size = 32
    train_loader = DataLoader(
        InstanceDataset(train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        InstanceDataset(validation_indices),
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        InstanceDataset(test_indices),
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
    )
    model = DirectSelector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )
    criterion_loss = nn.CrossEntropyLoss()
    max_epochs = 3 if smoke else 100
    patience = 2 if smoke else 15
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    started = time.monotonic()

    def evaluate_loss(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    batch["values"].to(device),
                    batch["mask"].to(device),
                    batch["horizon"].to(device),
                )
                targets = batch["targets"].to(device)
                loss = torch.stack(
                    [
                        criterion_loss(logits[:, index], targets[:, index])
                        for index in range(len(CRITERIA))
                    ]
                ).mean()
                losses.append(float(loss))
        return float(np.mean(losses))

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["values"].to(device),
                batch["mask"].to(device),
                batch["horizon"].to(device),
            )
            targets = batch["targets"].to(device)
            loss = torch.stack(
                [
                    criterion_loss(logits[:, index], targets[:, index])
                    for index in range(len(CRITERIA))
                ]
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in fold {fold}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation_loss = evaluate_loss(validation_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"No best model for fold {fold}")
    model.load_state_dict(best_state)
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["values"].to(device),
                batch["mask"].to(device),
                batch["horizon"].to(device),
            )
            predictions = logits.argmax(dim=-1).cpu().numpy()
            for local_index, row_index in enumerate(batch["row_index"].numpy()):
                source = labels.iloc[int(row_index)]
                row = {
                    "instance": int(source["instance"]),
                    "Asset": source["Asset"],
                    "fold": fold,
                }
                for criterion_index, criterion in enumerate(CRITERIA):
                    selected_index = int(predictions[local_index, criterion_index])
                    actual_index = int(source[f"{criterion}__label"])
                    utilities = np.asarray(
                        [
                            source[f"{criterion}__{algorithm}"]
                            for algorithm in ALGORITHMS
                        ],
                        dtype=float,
                    )
                    selected_value = float(utilities[selected_index])
                    vbs_value = float(utilities.max())
                    row.update(
                        {
                            f"{criterion}__selected": ALGORITHMS[selected_index],
                            f"{criterion}__label": ALGORITHMS[actual_index],
                            f"{criterion}__correct": int(
                                selected_index == actual_index
                            ),
                            f"{criterion}__selection": selected_value,
                            f"{criterion}__vbs": vbs_value,
                            f"{criterion}__regret": vbs_value - selected_value,
                        }
                    )
                rows.append(row)

    output_dir = OUTPUTS / ("smoke" if smoke else "full")
    prediction_path = output_dir / f"fold_{fold}_predictions.csv"
    metadata_path = output_dir / f"fold_{fold}_metadata.json"
    atomic_csv(prediction_path, pd.DataFrame(rows))
    atomic_json(
        metadata_path,
        {
            "fold": fold,
            "smoke": smoke,
            "seed": SEED + fold,
            "train_instances": len(train_indices),
            "validation_instances": len(validation_indices),
            "test_instances": len(test_indices),
            "train_assets": sorted(labels.iloc[train_indices]["Asset"].unique()),
            "validation_assets": sorted(
                labels.iloc[validation_indices]["Asset"].unique()
            ),
            "test_assets": sorted(labels.iloc[test_indices]["Asset"].unique()),
            "epochs_completed": len(history),
            "best_validation_loss": best_loss,
            "elapsed_seconds": time.monotonic() - started,
            "history": history,
            "completed_at": utc_now(),
        },
    )
    print(
        json.dumps(
            {
                "fold": fold,
                "test_instances": len(rows),
                "epochs": len(history),
                "best_validation_loss": best_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def combine() -> dict:
    paths = [OUTPUTS / "full" / f"fold_{fold}_predictions.csv" for fold in range(5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"complete": False, "missing": missing}
    predictions = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(predictions) != 762 or predictions["instance"].nunique() != 762:
        raise RuntimeError("Outer-fold predictions do not cover 762 instances")
    predictions = predictions.sort_values("instance").reset_index(drop=True)
    atomic_csv(OUTPUTS / "direct_selector_predictions.csv", predictions)

    rows = []
    for criterion in CRITERIA:
        selection = predictions[f"{criterion}__selection"]
        regret = predictions[f"{criterion}__regret"]
        accuracy = predictions[f"{criterion}__correct"]
        row = {
            "criterion": criterion,
            "selection_mean": float(selection.mean()),
            "regret_mean": float(regret.mean()),
            "accuracy": float(accuracy.mean()),
        }
        for selector in ["RF", "MLP"]:
            baseline_path = (
                ROOT
                / "experiments/transformer_representation_v1"
                / f"selector_results/{selector}/{criterion}/Fin/seed_42.csv"
            )
            if baseline_path.exists():
                baseline = pd.read_csv(baseline_path)
                row[f"{selector}_fin_selection_mean"] = float(
                    baseline["selection"].mean()
                )
                row[f"delta_vs_{selector}_fin"] = (
                    row["selection_mean"]
                    - row[f"{selector}_fin_selection_mean"]
                )
        rows.append(row)
    summary = pd.DataFrame(rows)
    atomic_csv(OUTPUTS / "summary.csv", summary)
    report = {
        "complete": True,
        "instances": len(predictions),
        "asset_families": int(predictions["Asset"].nunique()),
        "summary": rows,
        "completed_at": utc_now(),
    }
    atomic_json(OUTPUTS / "validation_report.json", report)
    return report


def launch(gpus: list[int], timeout_minutes: int) -> None:
    if not gpus or len(gpus) > 4:
        raise ValueError("Use one to four GPUs")
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = list(range(5))
    running: dict[subprocess.Popen, dict] = {}
    completed = []
    failed = []
    deadline = time.monotonic() + timeout_minutes * 60
    started = time.monotonic()
    while pending or running:
        if time.monotonic() >= deadline:
            for process in running:
                process.terminate()
            failed.extend(
                {"fold": value["fold"], "reason": "timeout"}
                for value in running.values()
            )
            break
        used = {value["gpu"] for value in running.values()}
        free = [gpu for gpu in gpus if gpu not in used]
        while pending and free:
            fold = pending.pop(0)
            gpu = free.pop(0)
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
            return_code = process.poll()
            if return_code is None:
                continue
            info["handle"].close()
            running.pop(process)
            if return_code == 0:
                completed.append(info["fold"])
            else:
                failed.append(
                    {
                        "fold": info["fold"],
                        "return_code": return_code,
                        "log": str(LOGS / f"fold_{info['fold']}.log"),
                    }
                )
            atomic_json(
                OUTPUTS / "run_state.json",
                {
                    "status": "running",
                    "completed": completed,
                    "failed": failed,
                    "pending": pending,
                    "running": [
                        {"fold": value["fold"], "gpu": value["gpu"]}
                        for value in running.values()
                    ],
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    report = combine()
    status = "complete" if not failed and report.get("complete") else "failed"
    atomic_json(
        OUTPUTS / "run_state.json",
        {
            "status": status,
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": time.monotonic() - started,
            "report": report,
            "finished_at": utc_now(),
        },
    )
    if status != "complete":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--fold", type=int, choices=range(5), required=True)
    worker.add_argument("--smoke", action="store_true")
    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--gpus", default="0,1,2,3")
    launcher.add_argument("--timeout-minutes", type=int, default=60)
    subparsers.add_parser("combine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        train_fold(args.fold, args.smoke)
    elif args.command == "launch":
        launch(
            [int(value) for value in args.gpus.split(",") if value],
            args.timeout_minutes,
        )
    else:
        print(json.dumps(combine(), indent=2))


if __name__ == "__main__":
    main()
