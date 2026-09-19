from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
ROOT = Path(CFG["source_root"])
OUT = HERE / "outputs" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CRITERIA = CFG["criteria"]
ALGOS = CFG["algorithms"]
SEED = CFG["random_seed"]


def load_frame(criterion: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    source = pd.read_csv(ROOT / "RF_combined" / f"All_{criterion}_LLM_combined.csv")
    v2 = pd.read_csv(HERE / "outputs" / "features" / "formula_v2_features.csv")
    df = source.merge(v2, on="instance", validate="one_to_one")
    v2_cols = [c for c in v2 if c != "instance"]
    pruned = [c for c in v2_cols if c != "downside_beta_concentration"]
    fin = source.columns[1:25].tolist() + [source.columns[37]]
    gpt = source.columns[25:37].tolist()
    old_formula = source.columns[62:77].tolist()

    rng = np.random.default_rng(SEED)
    random_cols, permuted_cols = [], []
    for i, col in enumerate(v2_cols):
        rc = f"control_random_{i+1:02d}"
        pc = f"control_permuted_{col}"
        df[rc] = rng.normal(size=len(df))
        df[pc] = rng.permutation(df[col].to_numpy())
        random_cols.append(rc)
        permuted_cols.append(pc)

    configs = {
        "Fin": fin,
        "Fin+GPT": fin + gpt,
        "Fin+OldFormula": fin + old_formula,
        "FormulaV2Only": v2_cols,
        "Fin+FormulaV2All": fin + v2_cols,
        "Fin+FormulaV2Pruned": fin + pruned,
        "Fin+GPT+FormulaV2Pruned": fin + gpt + pruned,
        "Fin+Random17": fin + random_cols,
        "Fin+PermutedFormulaV2": fin + permuted_cols,
    }
    return df, configs


def build_model(selector: str, fold: int):
    if selector == "RF":
        return RandomForestRegressor(
            n_estimators=CFG["rf_estimators"], random_state=SEED + fold, n_jobs=1
        )
    base = Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=1e-3, learning_rate_init=5e-4, learning_rate="adaptive",
            max_iter=1000, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=SEED + fold,
        )),
    ])
    return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())


def one_fit(selector, X, y, target, train_idx, test_idx, fold):
    model = build_model(selector, fold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X.iloc[train_idx], y[target].iloc[train_idx])
    return target, test_idx, model.predict(X.iloc[test_idx])


def run_one(selector: str, criterion: str, config_name: str, jobs: int) -> dict:
    out_dir = OUT / selector / criterion
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{config_name}.csv"
    metadata_path = out_dir / f"{config_name}.metadata.json"
    if result_path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())

    df, configs = load_frame(criterion)
    cols = configs[config_name]
    X = df[cols].astype(float)
    y = df[ALGOS].astype(float)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Non-finite values in {criterion}/{config_name}")

    folds = list(KFold(n_splits=CFG["folds"], shuffle=True, random_state=SEED).split(X))
    tasks = [(target, tr, te, fold) for target in ALGOS for fold, (tr, te) in enumerate(folds)]
    results = Parallel(n_jobs=jobs, verbose=0)(
        delayed(one_fit)(selector, X, y, target, tr, te, fold)
        for target, tr, te, fold in tasks
    )
    pred = pd.DataFrame(index=df.index, columns=ALGOS, dtype=float)
    for target, test_idx, values in results:
        pred.iloc[test_idx, pred.columns.get_loc(target)] = values

    selected = pred.idxmax(axis=1)
    actual = y.to_numpy()
    chosen_idx = np.array([ALGOS.index(a) for a in selected])
    selection = actual[np.arange(len(df)), chosen_idx]
    vbs_idx = np.argmax(actual, axis=1)
    sbs_algo = "OH-L" if criterion in {"AR", "CR"} else "MS-T"
    result = pd.DataFrame({
        "Instance": df["instance"].astype(int),
        "SBS": y[sbs_algo],
        "VBS": actual[np.arange(len(df)), vbs_idx],
        "VBS Algo": np.array(ALGOS)[vbs_idx],
        "Selected Algo": selected,
        "Selection Result": selection,
        "Selection Regret": actual[np.arange(len(df)), vbs_idx] - selection,
    })
    tmp = result_path.with_suffix(".tmp")
    result.to_csv(tmp, index=False)
    tmp.replace(result_path)
    meta = {
        "selector": selector, "criterion": criterion, "configuration": config_name,
        "n_instances": len(df), "n_features": len(cols), "folds": CFG["folds"],
        "seed": SEED, "protocol": "exact repository shuffled KFold; one scalar model per target/fold",
        "selection_mean": float(result["Selection Result"].mean()),
        "regret_mean": float(result["Selection Regret"].mean()),
        "sbs_mean": float(result["SBS"].mean()), "vbs_mean": float(result["VBS"].mean()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", choices=["RF", "MLP"], required=True)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()
    _, all_configs = load_frame("AR")
    chosen = args.configs or list(all_configs)
    unknown = set(chosen) - set(all_configs)
    if unknown:
        raise SystemExit(f"Unknown configs: {sorted(unknown)}")
    summaries = []
    for criterion in CRITERIA:
        for config in chosen:
            meta = run_one(args.selector, criterion, config, args.jobs)
            summaries.append(meta)
            print(json.dumps({k: meta[k] for k in ["selector", "criterion", "configuration", "selection_mean", "regret_mean"]}), flush=True)
    pd.DataFrame(summaries).to_csv(OUT / f"{args.selector}_summary.csv", index=False)


if __name__ == "__main__":
    main()
