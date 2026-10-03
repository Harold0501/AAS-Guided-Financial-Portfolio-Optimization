from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_experiment as base


OUTPUTS = HERE / "bandit_outputs"
SEED = 42
ENTROPY_COEF = 0.01
TEMPERATURE = 1.0
MODEL_DIM = 64


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


def build_data(labels, contexts, indices, mean, std):
    import torch
    from torch.utils.data import Dataset

    class InstanceDataset(Dataset):
        def __init__(self):
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
            rewards = np.asarray(
                [
                    [
                        row[f"{criterion}__{algorithm}"]
                        for algorithm in base.ALGORITHMS
                    ]
                    for criterion in base.CRITERIA
                ],
                dtype=np.float32,
            )
            return {
                "row_index": row_index,
                "values": normalized,
                "mask": mask,
                "horizon": int(row["horizon_index"]),
                "rewards": rewards,
            }

    def collate(batch):
        max_stocks = max(item["values"].shape[1] for item in batch)
        values = np.zeros(
            (len(batch), base.WINDOW, max_stocks, base.RAW_FEATURES),
            dtype=np.float32,
        )
        mask = np.zeros((len(batch), base.WINDOW, max_stocks), dtype=bool)
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
            "rewards": torch.tensor(
                np.stack([item["rewards"] for item in batch]),
                dtype=torch.float32,
            ),
        }

    return InstanceDataset(), collate


def build_model():
    import torch
    import torch.nn as nn

    class BanditPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.stock_encoder = nn.Sequential(
                nn.Linear(base.RAW_FEATURES, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
            )
            self.day_projection = nn.Linear(64, MODEL_DIM)
            self.position = nn.Embedding(base.WINDOW, MODEL_DIM)
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
            self.policy_heads = nn.ModuleList(
                [
                    nn.Linear(MODEL_DIM, len(base.ALGORITHMS))
                    for _ in base.CRITERIA
                ]
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
            positions = torch.arange(base.WINDOW, device=values.device)
            daily = daily + self.position(positions).unsqueeze(0)
            day_mask = ~stock_mask.any(dim=2)
            encoded = self.temporal_encoder(
                daily, src_key_padding_mask=day_mask
            )
            state = self.normalization(
                encoded[:, -1] + self.horizon(horizon)
            )
            return torch.stack(
                [head(state) for head in self.policy_heads], dim=1
            )

    return BanditPolicy()


def reward_statistics(labels, train_indices):
    means = []
    scales = []
    train = labels.iloc[train_indices]
    for criterion in base.CRITERIA:
        values = train[
            [f"{criterion}__{algorithm}" for algorithm in base.ALGORITHMS]
        ].to_numpy(dtype=np.float32)
        means.append(float(values.mean()))
        scales.append(float(max(values.std(), 1e-6)))
    return np.asarray(means, dtype=np.float32), np.asarray(scales, dtype=np.float32)


def normalize_rewards(rewards, means, scales):
    standardized = (rewards - means[None, :, None]) / scales[None, :, None]
    return standardized - standardized.mean(axis=2, keepdims=True)


def run_fold(fold: int, smoke: bool) -> None:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)

    labels = pd.read_csv(base.CACHE / "labels.csv")
    contexts = base.load_contexts()
    train_indices, validation_indices, test_indices = base.split_indices(
        labels, fold
    )
    mean, std = base.normalization_stats(
        labels, contexts, train_indices
    )
    reward_means, reward_scales = reward_statistics(labels, train_indices)

    train_dataset, collate = build_data(
        labels, contexts, train_indices, mean, std
    )
    validation_dataset, _ = build_data(
        labels, contexts, validation_indices, mean, std
    )
    test_dataset, _ = build_data(labels, contexts, test_indices, mean, std)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=64, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate
    )
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )
    epochs = 3 if smoke else 100
    patience = 2 if smoke else 15
    best_score = -float("inf")
    best_state = None
    stale = 0
    history = []
    started = time.monotonic()

    def objective(logits, rewards):
        normalized = normalize_rewards(
            rewards.detach().cpu().numpy(),
            reward_means,
            reward_scales,
        )
        advantages = torch.from_numpy(normalized).to(logits.device)
        probabilities = functional.softmax(
            logits / TEMPERATURE, dim=-1
        )
        expected_reward = (probabilities * advantages).sum(dim=-1).mean()
        entropy = -(
            probabilities
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        return -expected_reward - ENTROPY_COEF * entropy, expected_reward, entropy

    def validation_score():
        model.eval()
        values = []
        with torch.no_grad():
            for batch in validation_loader:
                logits = model(
                    batch["values"].to(device),
                    batch["mask"].to(device),
                    batch["horizon"].to(device),
                )
                _, expected, _ = objective(
                    logits, batch["rewards"].to(device)
                )
                values.append(float(expected))
        return float(np.mean(values))

    for epoch in range(epochs):
        model.train()
        train_objectives = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["values"].to(device),
                batch["mask"].to(device),
                batch["horizon"].to(device),
            )
            loss, _, _ = objective(
                logits, batch["rewards"].to(device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite bandit loss in fold {fold}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            train_objectives.append(float(loss.detach()))
        validation_expected = validation_score()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_objectives)),
                "validation_expected_reward": validation_expected,
            }
        )
        if validation_expected > best_score + 1e-5:
            best_score = validation_expected
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
        raise RuntimeError(f"No best policy in fold {fold}")
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
            chosen = logits.argmax(dim=-1).cpu().numpy()
            for local_index, row_index in enumerate(batch["row_index"].numpy()):
                source = labels.iloc[int(row_index)]
                row = {
                    "instance": int(source["instance"]),
                    "Asset": source["Asset"],
                    "fold": fold,
                }
                for criterion_index, criterion in enumerate(base.CRITERIA):
                    action = int(chosen[local_index, criterion_index])
                    utilities = np.asarray(
                        [
                            source[f"{criterion}__{algorithm}"]
                            for algorithm in base.ALGORITHMS
                        ],
                        dtype=float,
                    )
                    vbs = float(utilities.max())
                    value = float(utilities[action])
                    row.update(
                        {
                            f"{criterion}__selected": base.ALGORITHMS[action],
                            f"{criterion}__label": base.ALGORITHMS[
                                int(utilities.argmax())
                            ],
                            f"{criterion}__correct": int(
                                action == int(utilities.argmax())
                            ),
                            f"{criterion}__selection": value,
                            f"{criterion}__vbs": vbs,
                            f"{criterion}__regret": vbs - value,
                        }
                    )
                rows.append(row)

    output_dir = OUTPUTS / ("smoke" if smoke else "full")
    atomic_csv(
        output_dir / f"fold_{fold}_predictions.csv",
        pd.DataFrame(rows),
    )
    atomic_json(
        output_dir / f"fold_{fold}_metadata.json",
        {
            "fold": fold,
            "smoke": smoke,
            "seed": SEED + fold,
            "train_instances": len(train_indices),
            "validation_instances": len(validation_indices),
            "test_instances": len(test_indices),
            "epochs_completed": len(history),
            "best_validation_expected_reward": best_score,
            "elapsed_seconds": time.monotonic() - started,
            "history": history,
            "reward_definition": "train-fold standardized, per-instance action-centered utility",
        },
    )
    print(
        json.dumps(
            {
                "fold": fold,
                "test_instances": len(rows),
                "epochs": len(history),
                "best_validation_expected_reward": best_score,
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


def combine() -> dict:
    paths = [
        OUTPUTS / "full" / f"fold_{fold}_predictions.csv"
        for fold in range(5)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"complete": False, "missing": missing}
    predictions = pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True
    )
    if len(predictions) != 762 or predictions["instance"].nunique() != 762:
        raise RuntimeError("Bandit predictions do not cover 762 instances")
    predictions = predictions.sort_values("instance").reset_index(drop=True)
    atomic_csv(OUTPUTS / "bandit_predictions.csv", predictions)
    rows = []
    for criterion in base.CRITERIA:
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
                baseline_mean = float(baseline["selection"].mean())
                row[f"{selector}_fin_selection_mean"] = baseline_mean
                row[f"delta_vs_{selector}_fin"] = (
                    row["selection_mean"] - baseline_mean
                )
        rows.append(row)
    summary = pd.DataFrame(rows)
    atomic_csv(OUTPUTS / "summary.csv", summary)
    report = {
        "complete": True,
        "instances": len(predictions),
        "asset_families": int(predictions["Asset"].nunique()),
        "summary": rows,
        "method": "full-information offline contextual bandit",
        "completed_at": base.utc_now(),
    }
    atomic_json(OUTPUTS / "validation_report.json", report)
    return report


def launch(gpus: list[int], timeout_minutes: int) -> None:
    if not gpus or len(gpus) > 4:
        raise ValueError("Use one to four GPUs")
    running = {}
    pending = list(range(5))
    completed = []
    failed = []
    deadline = time.monotonic() + timeout_minutes * 60
    started = time.monotonic()
    (HERE / "bandit_logs").mkdir(exist_ok=True)
    while pending or running:
        if time.monotonic() >= deadline:
            for process in running:
                process.terminate()
            failed.extend(
                {"fold": info["fold"], "reason": "timeout"}
                for info in running.values()
            )
            break
        used = {info["gpu"] for info in running.values()}
        free = [gpu for gpu in gpus if gpu not in used]
        while pending and free:
            fold = pending.pop(0)
            gpu = free.pop(0)
            log = (HERE / "bandit_logs" / f"fold_{fold}.log").open("a")
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
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[process] = {"fold": fold, "gpu": gpu, "log": log}
        time.sleep(2)
        for process, info in list(running.items()):
            if process.poll() is None:
                continue
            info["log"].close()
            running.pop(process)
            if process.returncode == 0:
                completed.append(info["fold"])
            else:
                failed.append(
                    {
                        "fold": info["fold"],
                        "return_code": process.returncode,
                    }
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
            "finished_at": base.utc_now(),
        },
    )
    if status != "complete":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--fold", type=int, choices=range(5), required=True)
    worker.add_argument("--smoke", action="store_true")
    launcher = subparsers.add_parser("launch")
    launcher.add_argument("--gpus", default="0,1,2,3")
    launcher.add_argument("--timeout-minutes", type=int, default=60)
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
