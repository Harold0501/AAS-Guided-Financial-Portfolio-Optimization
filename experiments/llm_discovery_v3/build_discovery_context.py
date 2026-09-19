from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
ROOT = Path(CFG["source_root"])


def q(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {}
    return {k: round(float(v), 6) for k, v in zip(
        ["min", "q10", "median", "q90", "max"], x.quantile([0, .1, .5, .9, 1]).values
    )}


def main() -> None:
    mapping = pd.read_csv(ROOT / CFG["mapping_csv"])
    mapping["start"] = pd.to_datetime(mapping["start"], format="mixed")
    mapping["end"] = pd.to_datetime(mapping["end"], format="mixed")
    folds = pd.read_csv(HERE / "manifests/frozen_asset_folds.csv")
    discovery_test = set(folds.loc[folds.outer_fold == 0, "Asset"])
    dev = mapping[~mapping.Asset.isin(discovery_test)].copy()
    raw_dir = ROOT / CFG["raw_data_dir"]
    rows = []
    # Deterministically summarize every development instance. No outcomes are read.
    for asset, group in dev.groupby("Asset", sort=True):
        raw = pd.read_csv(raw_dir / f"{asset}.csv")
        raw["date"] = pd.to_datetime(raw["date"])
        for _, r in group.sort_values("instance").iterrows():
            d = raw[(raw.date >= r.start) & (raw.date <= r.end)].copy()
            close = d.pivot(index="date", columns="company", values="div_adj_close").sort_index()
            ret = close.pct_change(fill_method=None)
            rows.append({
                "instance": int(r.instance), "family": asset,
                "observations": int(len(close)), "assets": int(close.shape[1]),
                "missing_fraction": float(close.isna().mean().mean()),
                "median_asset_volatility": float(ret.std().median()),
                "median_asset_mean_return": float(ret.mean().median()),
                "cross_sectional_return_dispersion": float(ret.std(axis=1).median()),
                "mean_pairwise_correlation": float(ret.corr().where(~np.eye(ret.shape[1], dtype=bool)).stack().mean()),
                "negative_return_fraction": float((ret < 0).mean().mean()),
                "volume_log_change_volatility": float(d.sort_values(["company", "date"]).groupby("company")["volume"].transform(lambda x: np.log1p(x).diff()).std()),
            })
    frame = pd.DataFrame(rows)
    context = {
        "protocol": {
            "purpose": "development-only, outcome-blind portfolio-instance feature discovery",
            "development_families": sorted(dev.Asset.unique().tolist()),
            "heldout_family_count": len(discovery_test),
            "heldout_names_disclosed_to_llm": False,
            "algorithm_outcomes_included": False,
            "future_returns_included": False,
        },
        "schema": {
            "raw_fields": CFG["allowed_raw_fields"],
            "instance_definition": "A variable-length multi-asset OHLCV history ending at its decision date.",
            "required_output": "One finite scalar per portfolio instance, invariant to asset ordering.",
        },
        "development_distribution": {
            "instances": len(frame), "families": frame.family.nunique(),
            **{col: q(frame[col]) for col in frame.columns if col not in {"instance", "family"}},
        },
        # A small deterministic, anonymized sample makes this genuinely data-conditioned.
        "anonymized_examples": frame.drop(columns=["instance", "family"]).iloc[::max(1, len(frame)//24)].head(24).round(6).to_dict("records"),
    }
    path = HERE / "manifests/discovery_context.json"
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(context, indent=2) + "\n"); tmp.replace(path)
    split = {"discovery_development_assets": sorted(dev.Asset.unique()),
             "discovery_test_assets": sorted(discovery_test),
             "development_instances": len(dev),
             "test_instances": len(mapping) - len(dev)}
    path = HERE / "manifests/discovery_split_private.json"
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(split, indent=2) + "\n"); tmp.replace(path)
    print(json.dumps({"context_instances": len(frame), "development_families": frame.family.nunique(),
                      "heldout_families": len(discovery_test)}))


if __name__ == "__main__":
    main()
