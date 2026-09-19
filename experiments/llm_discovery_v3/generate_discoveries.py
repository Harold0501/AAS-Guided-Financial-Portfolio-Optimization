from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text[text.find("{"):text.rfind("}") + 1])


def prompt_for(arm: str) -> str:
    common = (HERE / "prompts/common_contract.txt").read_text()
    template = (HERE / f"prompts/{arm}_{'blind' if arm == 'B' else 'basis_aware'}.txt").read_text()
    context = (HERE / "manifests/discovery_context.json").read_text()
    basis = (HERE / "manifests/frozen_basis.json").read_text() if arm == "C" else ""
    n = CFG["features_per_generation"]
    return (template.replace("<COMMON>", common).replace("<CONTEXT>", context)
            .replace("<BASIS>", basis).replace("<N>", str(n))
            + f"\nFINAL COUNT CHECK: features array length MUST equal {n}; do not return {n+1} or more.\n")


def validate_shape(doc: dict, arm: str) -> None:
    required = {"feature_id", "name", "discovered_gap", "financial_interpretation", "inputs",
                "preprocessing", "mathematical_definition", "aggregation", "parameters",
                "minimum_sample", "missing_and_zero_policy", "expected_range", "invariances",
                "synthetic_test", "reconstruction_pseudocode", "leakage_argument"}
    if doc.get("arm") != arm or len(doc.get("features", [])) != CFG["features_per_generation"]:
        raise ValueError("Wrong arm or feature count")
    ids = []
    for f in doc["features"]:
        if not required <= set(f): raise ValueError(f"Missing fields: {required-set(f)}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", f["feature_id"]): raise ValueError("Invalid feature_id")
        ids.append(f["feature_id"])
    if len(ids) != len(set(ids)): raise ValueError("Duplicate feature IDs within generation")


def run_one(arm: str, index: int, key: str) -> None:
    label = f"{arm}_generation_{index}"
    parsed_path = HERE / "responses" / f"{label}.json"
    raw_path = HERE / "responses" / f"{label}_raw.json"
    if parsed_path.exists() and raw_path.exists():
        print(f"[resume] {label}", flush=True); return
    count_guard = "\nReturn EXACTLY ten (10) feature objects. Number them mentally 1 through 10 and stop after number 10."
    payload = {
        "model": CFG["model"], "temperature": CFG["temperature"],
        "max_tokens": 16000,
        "seed": CFG["generation_seeds"][index - 1],
        "messages": [
            {"role": "system", "content": "Return strict JSON. Be conservative, mathematically complete, and leakage-aware."},
            {"role": "user", "content": prompt_for(arm) + count_guard},
        ], "response_format": {"type": "json_object"},
    }
    req_path = HERE / "requests" / f"{label}.json"; atomic_json(req_path, payload)
    cmd = ["curl", "-sS", "--fail-with-body", "--connect-timeout", "30", "--max-time", "600",
           "--retry", "2", "--retry-all-errors", "https://openrouter.ai/api/v1/chat/completions",
           "-H", f"Authorization: Bearer {key}", "-H", "Content-Type: application/json",
           "-H", "HTTP-Referer: https://openai.com/codex", "--data-binary", f"@{req_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        # Error logs contain no request headers or credential.
        (HERE / "logs" / f"{label}_error.log").write_text(result.stderr + "\n" + result.stdout)
        raise RuntimeError(f"{label} network/API failure; safe to rerun")
    response = json.loads(result.stdout); atomic_json(raw_path, response)
    finish = response.get("choices", [{}])[0].get("finish_reason")
    if finish not in {"stop", None}:
        raise RuntimeError(f"{label} incomplete response ({finish}); raw checkpoint retained")
    doc = extract_json(response["choices"][0]["message"]["content"]); validate_shape(doc, arm)
    doc["_provenance"] = {"requested_model": CFG["model"], "returned_model": response.get("model"),
        "response_id": response.get("id"), "created": response.get("created"),
        "usage": response.get("usage"), "seed": CFG["generation_seeds"][index - 1],
        "generated_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(parsed_path, doc); print(f"[complete] {label}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["B", "C"], required=True)
    args = ap.parse_args(); key = os.environ.get("OPENROUTER_API_KEY")
    if not key: raise SystemExit("OPENROUTER_API_KEY must be passed through the environment")
    for i in range(1, len(CFG["generation_seeds"]) + 1):
        run_one(args.arm, i, key); time.sleep(1)
    state = json.loads((HERE / "state.json").read_text())
    done = set(state["completed_stages"]); done.add(f"generate_{args.arm}")
    state.update(status=f"{args.arm}_generated", completed_stages=sorted(done),
                 next_stage="generate_C" if args.arm == "B" else "validate_contracts")
    atomic_json(HERE / "state.json", state)


if __name__ == "__main__": main()
