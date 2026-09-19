from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
ROOT = Path(CFG["source_root"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    for name in ["manifests", "prompts", "requests", "responses", "features", "results", "logs", "report"]:
        (HERE / name).mkdir(exist_ok=True)

    mapping_path = ROOT / CFG["mapping_csv"]
    mapping = pd.read_csv(mapping_path)
    if len(mapping) != 762 or mapping["instance"].nunique() != 762:
        raise RuntimeError("Expected exactly 762 unique instances")

    raw_dir = ROOT / CFG["raw_data_dir"]
    inputs = [mapping_path, *sorted(raw_dir.glob("*.csv"))]
    formula_dir = ROOT / CFG["formula_v2_dir"]
    inputs += [
        formula_dir / "outputs/features/formula_v2_features.csv",
        formula_dir / "outputs/features/formula_v2_validated_features.csv",
        ROOT / "RF_combined/All_AR_LLM_combined.csv",
        ROOT / "RF_combined/All_CR_LLM_combined.csv",
        ROOT / "RF_combined/All_MDD_LLM_combined.csv",
        ROOT / "RF_combined/All_SHR_LLM_combined.csv",
        ROOT / "RF_combined/All_SOR_LLM_combined.csv",
    ]
    rows = []
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty input: {path}")
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(rows)
    tmp = HERE / "manifests/input_manifest.tmp"
    manifest.to_csv(tmp, index=False)
    tmp.replace(HERE / "manifests/input_manifest.csv")

    # Freeze grouped folds before any new LLM output exists.
    assets = pd.DataFrame({"Asset": sorted(mapping["Asset"].unique())})
    # Deterministic round-robin assignment is sufficient here: each row is one
    # complete asset family, and no instance-level information influences folds.
    assets["outer_fold"] = [i % CFG["heldout_asset_folds"] for i in range(len(assets))]
    tmp = HERE / "manifests/frozen_asset_folds.tmp"
    assets.to_csv(tmp, index=False)
    tmp.replace(HERE / "manifests/frozen_asset_folds.csv")

    atomic_json(HERE / "manifests/environment.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "instances": int(len(mapping)),
        "asset_families": int(mapping["Asset"].nunique()),
        "raw_csv_files": len(list(raw_dir.glob("*.csv"))),
        "model": CFG["model"],
    })
    state = json.loads((HERE / "state.json").read_text())
    state.update(status="bootstrapped", completed_stages=["bootstrap"], next_stage="api_check_and_A")
    atomic_json(HERE / "state.json", state)
    print(json.dumps({"status": "ok", "inputs": len(inputs), "instances": 762,
                      "assets": int(mapping["Asset"].nunique())}))


if __name__ == "__main__":
    main()
