from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
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
RAW_DATA = ROOT / "LLMD-AS-FPO/LLM_Generated_Feature/data"
TARGET_WEIGHTS = Path(
    os.environ.get(
        "SDFLWDE_TARGET_DIR",
        "/data1/home/hz314/workspace/sDFLwDE/new_result",
    )
)
SOPHIA_DIR = ROOT / "LLMD-AS-FPO/CleanLoader/src"
OUTPUTS = Path(
    os.environ.get("TRANSFORMER_REP_OUTPUTS", str(HERE / "outputs"))
)
LOGS = Path(
    os.environ.get("TRANSFORMER_REP_LOGS", str(OUTPUTS / "logs"))
)

D_MODEL = 256
WINDOW_SIZE = 252
NUM_EPOCHS = 50
STOP_COUNT = 10
LEARNING_RATE = 1e-4
SEED = int(os.environ.get("TRANSFORMER_REP_SEED", "0"))
TARGETS = ("maxsortino", "onehot")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING)
    mapping["start"] = pd.to_datetime(mapping["start"], format="mixed")
    mapping["end"] = pd.to_datetime(mapping["end"], format="mixed")
    if len(mapping) != 762 or mapping["instance"].nunique() != 762:
        raise RuntimeError("Expected exactly 762 unique mapped instances")
    return mapping


def get_yearmons(data: pd.DataFrame) -> list[str]:
    first_date = data.index.min()
    last_date = data.index.max()
    result: list[str] = []
    for year in range(first_date.year, last_date.year + 1):
        for month_zero_based in range(12):
            month = month_zero_based + 1
            if year == first_date.year and month < first_date.month:
                continue
            if year == last_date.year and month == last_date.month:
                break
            result.append(f"{year}-{month:02d}")
    return result


def preprocess_data(
    path: Path,
    first_target_month: str,
    device,
):
    import torch

    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"])
    # Target portfolios in the preserved sDFLwDE artifacts start in either
    # 2015-01 or 2020-01. This also resolves the longer asset_18 source file.
    raw = raw[raw["date"] >= pd.Timestamp(first_target_month + "-01")].copy()
    raw = raw.sort_values(["company", "date"]).set_index(["company", "date"])

    raw["open"] = np.log(raw["open"] / raw["close"])
    raw["high"] = np.log(raw["high"] / raw["close"])
    raw["low"] = np.log(raw["low"] / raw["close"])
    raw["volume"] = np.log(raw["volume"])
    raw["log_tr"] = np.log(
        raw.groupby(level="company")["div_adj_close"].pct_change() + 1
    )
    raw = raw.drop(["close", "div_adj_close"], axis=1)
    data = raw.unstack(level=0)
    data = data[(~data["log_tr"].isna()).sum(axis=1) > 1]
    data = data[(~(data["log_tr"].isna() | (data["log_tr"] == 0))).sum(axis=1) > 0]

    tensor = torch.tensor(
        data.values, dtype=torch.float32, device=device
    ).nan_to_num(0)
    yearmons = get_yearmons(data)
    missing_mask = {
        yearmon: torch.tensor(
            data["log_tr"].loc[yearmon].iloc[0].isna().values,
            dtype=torch.bool,
            device=device,
        )
        for yearmon in yearmons
    }
    return yearmons, data.fillna(0), tensor, missing_mask


def build_model(input_size: int, num_output: int, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class TransformerWithRepresentation(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(input_size, D_MODEL, bias=True)
            self.positional_embedding = nn.Embedding(WINDOW_SIZE, D_MODEL)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=8,
                dim_feedforward=D_MODEL,
                activation=functional.silu,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=6)
            self.output_projection = nn.Linear(D_MODEL, num_output, bias=True)

        @staticmethod
        def causal_mask(length: int, target_device):
            return (
                torch.triu(
                    torch.ones(
                        (length, length), dtype=torch.bool, device=target_device
                    )
                )
                != 1
            ).transpose(0, 1)

        def encode(self, values):
            hidden = functional.silu(self.input_projection(values))
            positions = torch.arange(
                hidden.shape[0], device=hidden.device
            ).flip(0)
            hidden = hidden + self.positional_embedding(positions)
            hidden = self.encoder(
                hidden,
                mask=self.causal_mask(hidden.shape[0], hidden.device),
                is_causal=True,
            )
            return functional.silu(hidden)

        def forward(self, values):
            return self.output_projection(self.encode(values))

    return TransformerWithRepresentation().to(device)


def portfolio_loss(prediction, target, missing_mask):
    import torch

    return -(
        torch.where(missing_mask, -torch.inf, prediction)
        .log_softmax(dim=-1)
        .nan_to_num(0)
        * torch.where(missing_mask, 0, target)
    ).sum(dim=-1).sum(dim=0)


def train_target(
    asset: str,
    target_name: str,
    requested_months: set[str],
    smoke: bool,
) -> tuple[pd.DataFrame, dict]:
    import torch

    sys.path.insert(0, str(SOPHIA_DIR))
    from sophia import SophiaG

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    target_path = TARGET_WEIGHTS / f"{asset}_weights_{target_name}.pkl"
    with target_path.open("rb") as handle:
        target_weights = pickle.load(handle)
    first_target_month = next(iter(target_weights))

    yearmons, data, tensor, missing_mask = preprocess_data(
        RAW_DATA / f"{asset}.csv", first_target_month, device
    )
    date_to_index = {date: index for index, date in enumerate(data.index)}
    num_items = len(data["log_tr"].columns)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = build_model(tensor.shape[1], num_items, device)
    optimizer = SophiaG(model.parameters(), lr=LEARNING_RATE)

    effective_epochs = 3 if smoke else NUM_EPOCHS
    effective_stop_count = 2 if smoke else STOP_COUNT
    wanted = {min(requested_months)} if smoke else requested_months
    last_wanted = max(wanted)
    rows: list[dict] = []
    trained_months = 0
    started = time.monotonic()

    for index in range(3, len(yearmons)):
        prediction_month = yearmons[index]
        train_month = yearmons[index - 3]
        valid_month = yearmons[index - 2]
        input_month = yearmons[index - 1]

        if train_month not in target_weights or valid_month not in target_weights:
            continue

        train_indices = [date_to_index[x] for x in data.loc[:train_month].index]
        valid_indices = [date_to_index[x] for x in data.loc[:valid_month].index]
        test_indices = [date_to_index[x] for x in data.loc[:input_month].index]
        train_values = tensor[train_indices][-WINDOW_SIZE:]
        valid_values = tensor[valid_indices][-WINDOW_SIZE:]
        test_values = tensor[test_indices][-WINDOW_SIZE:]

        best_loss = float("inf")
        best_state = None
        stale = 0
        for _ in range(effective_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(train_values)
            target = target_weights[train_month].to(device)
            loss = portfolio_loss(output, target, missing_mask[train_month])
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss for {asset}/{target_name}/{train_month}"
                )
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_output = model(valid_values)[-1]
                validation_target = target_weights[valid_month].to(device)
                validation_loss = portfolio_loss(
                    validation_output,
                    validation_target,
                    missing_mask[valid_month],
                )
                value = float(validation_loss)
            if value < best_loss:
                best_loss = value
                # state_dict tensors otherwise continue changing during training.
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale == effective_stop_count:
                    break

        if best_state is None:
            raise RuntimeError(f"No best state for {asset}/{target_name}/{train_month}")
        model.load_state_dict(best_state)
        model.eval()
        trained_months += 1

        if prediction_month in wanted:
            with torch.no_grad():
                representation = model.encode(test_values)[-1]
            values = representation.detach().float().cpu().numpy()
            if values.shape != (D_MODEL,) or not np.isfinite(values).all():
                raise RuntimeError(
                    f"Invalid representation for {asset}/{target_name}/{prediction_month}"
                )
            row = {
                "Asset": asset,
                "start_month": prediction_month,
                "target": target_name,
                "input_through_month": input_month,
            }
            row.update({f"h_{i:03d}": float(v) for i, v in enumerate(values)})
            rows.append(row)

        if prediction_month >= last_wanted:
            break

    frame = pd.DataFrame(rows)
    if set(frame.get("start_month", [])) != wanted:
        missing = wanted - set(frame.get("start_month", []))
        raise RuntimeError(f"Missing requested representations: {sorted(missing)}")
    metadata = {
        "asset": asset,
        "target": target_name,
        "smoke": smoke,
        "seed": SEED,
        "epochs": effective_epochs,
        "window_size": WINDOW_SIZE,
        "d_model": D_MODEL,
        "trained_months": trained_months,
        "representations": len(frame),
        "elapsed_seconds": time.monotonic() - started,
        "raw_data_sha256": sha256(RAW_DATA / f"{asset}.csv"),
        "target_weights_sha256": sha256(target_path),
        "completed_at": utc_now(),
    }
    return frame, metadata


def run_worker(asset: str, smoke: bool) -> None:
    mapping = load_mapping()
    local = mapping[mapping["Asset"] == asset]
    if local.empty:
        raise RuntimeError(f"Unknown asset in mapping: {asset}")
    requested_months = set(local["start"].dt.strftime("%Y-%m"))
    output_dir = OUTPUTS / ("smoke" if smoke else "full") / asset
    output_dir.mkdir(parents=True, exist_ok=True)

    for target_name in TARGETS:
        output_path = output_dir / f"{target_name}_representations.csv"
        metadata_path = output_dir / f"{target_name}_metadata.json"
        if not smoke and output_path.exists() and metadata_path.exists():
            continue
        frame, metadata = train_target(
            asset, target_name, requested_months, smoke
        )
        atomic_csv(output_path, frame)
        atomic_json(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "asset": asset,
                    "target": target_name,
                    "representations": len(frame),
                    "elapsed_seconds": metadata["elapsed_seconds"],
                }
            ),
            flush=True,
        )


def combine_outputs() -> dict:
    mapping = load_mapping()
    hidden_columns = [f"h_{i:03d}" for i in range(D_MODEL)]
    target_frames: dict[str, pd.DataFrame] = {}
    missing_assets: list[str] = []

    for target_name in TARGETS:
        frames = []
        for asset in sorted(mapping["Asset"].unique()):
            path = OUTPUTS / "full" / asset / f"{target_name}_representations.csv"
            if not path.exists():
                missing_assets.append(f"{asset}/{target_name}")
                continue
            frames.append(pd.read_csv(path))
        if frames:
            target_frames[target_name] = pd.concat(frames, ignore_index=True)

    if missing_assets:
        return {"complete": False, "missing": sorted(missing_assets)}

    key = ["Asset", "start_month"]
    left = target_frames["maxsortino"].rename(
        columns={column: f"mst_{column}" for column in hidden_columns}
    )
    right = target_frames["onehot"].rename(
        columns={column: f"oht_{column}" for column in hidden_columns}
    )
    representations = left[key + [f"mst_{c}" for c in hidden_columns]].merge(
        right[key + [f"oht_{c}" for c in hidden_columns]],
        on=key,
        validate="one_to_one",
    )

    instance_map = mapping[["instance", "Asset", "start", "end"]].copy()
    instance_map["start_month"] = instance_map["start"].dt.strftime("%Y-%m")
    combined = instance_map.merge(
        representations, on=["Asset", "start_month"], validate="many_to_one"
    )
    feature_columns = [
        column
        for column in combined
        if column.startswith("mst_h_") or column.startswith("oht_h_")
    ]
    values = combined[feature_columns].to_numpy(dtype=float)
    if len(combined) != 762 or combined["instance"].nunique() != 762:
        raise RuntimeError("Combined representation does not cover 762 instances")
    if len(feature_columns) != 2 * D_MODEL or not np.isfinite(values).all():
        raise RuntimeError("Combined representation has invalid feature values")

    output_path = OUTPUTS / "transformer_representations_1repeat.csv"
    atomic_csv(output_path, combined)
    report = {
        "complete": True,
        "instances": len(combined),
        "asset_families": int(combined["Asset"].nunique()),
        "features": len(feature_columns),
        "ms_t_features": D_MODEL,
        "oh_t_features": D_MODEL,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "completed_at": utc_now(),
    }
    atomic_json(OUTPUTS / "validation_report.json", report)
    return report


def launch(gpu_ids: list[int], timeout_minutes: int) -> None:
    if not gpu_ids or len(gpu_ids) > 4:
        raise ValueError("Use between one and four GPU IDs")
    mapping = load_mapping()
    assets = sorted(
        mapping["Asset"].unique(),
        key=lambda value: int(value.split("_")[1]),
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUTS / "run_state.json"
    deadline = time.monotonic() + timeout_minutes * 60
    pending = list(assets)
    running: dict[subprocess.Popen, dict] = {}
    completed: list[str] = []
    failed: list[dict] = []
    started = time.monotonic()
    atomic_json(
        state_path,
        {
            "status": "running",
            "started_at": utc_now(),
            "assets_total": len(assets),
            "gpu_ids": gpu_ids,
            "timeout_minutes": timeout_minutes,
        },
    )

    while pending or running:
        if time.monotonic() >= deadline:
            for process in running:
                process.terminate()
            failed.extend(
                {"asset": info["asset"], "reason": "global_timeout"}
                for info in running.values()
            )
            break

        used_gpus = {info["gpu"] for info in running.values()}
        free_gpus = [gpu for gpu in gpu_ids if gpu not in used_gpus]
        while pending and free_gpus:
            asset = pending.pop(0)
            gpu = free_gpus.pop(0)
            log_path = LOGS / f"{asset}.log"
            handle = log_path.open("a")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--asset",
                asset,
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[process] = {
                "asset": asset,
                "gpu": gpu,
                "handle": handle,
                "started": time.monotonic(),
            }

        time.sleep(2)
        for process, info in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            info["handle"].close()
            running.pop(process)
            if return_code == 0:
                completed.append(info["asset"])
            else:
                failed.append(
                    {
                        "asset": info["asset"],
                        "return_code": return_code,
                        "log": str(LOGS / f"{info['asset']}.log"),
                    }
                )
            atomic_json(
                state_path,
                {
                    "status": "running",
                    "started_at": utc_now(),
                    "completed": completed,
                    "failed": failed,
                    "pending": pending,
                    "running": [
                        {"asset": value["asset"], "gpu": value["gpu"]}
                        for value in running.values()
                    ],
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

    validation = combine_outputs()
    status = "complete" if not failed and validation.get("complete") else "failed"
    atomic_json(
        state_path,
        {
            "status": status,
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": time.monotonic() - started,
            "validation": validation,
            "finished_at": utc_now(),
        },
    )
    if status != "complete":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--asset", required=True)
    worker.add_argument("--smoke", action="store_true")

    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--gpus", default="0,1,2,3")
    launcher.add_argument("--timeout-minutes", type=int, default=85)

    subparsers.add_parser("combine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if args.command == "worker":
        run_worker(args.asset, args.smoke)
    elif args.command == "launch":
        gpu_ids = [int(value) for value in args.gpus.split(",") if value]
        launch(gpu_ids, args.timeout_minutes)
    else:
        report = combine_outputs()
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
