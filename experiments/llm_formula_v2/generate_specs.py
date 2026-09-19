from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "specs"
LOG = HERE / "logs"
OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-32b")
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set; no request was made.")


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start: raise
        return json.loads(text[start:end+1])


def run_one(label: str) -> dict:
    parsed_path = OUT / f"{label}.json"
    raw_path = OUT / f"{label}_raw_response.json"
    if parsed_path.exists() and raw_path.exists():
        print(f"[resume] {label} already complete")
        return json.loads(parsed_path.read_text())
    prompt = (HERE / "prompts" / "formula_generation_prompt.txt").read_text().replace("<provided label>", label)
    payload = {
        "model": MODEL,
        "temperature": 0.35,
        "seed": 42 + int(label.rsplit("_", 1)[-1]),
        "messages": [
            {"role": "system", "content": "Return strictly valid JSON. Be mathematically precise and conservative."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    payload_path = OUT / f"{label}_request.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    cmd = ["curl", "-sS", "--fail-with-body", "--max-time", "600", "https://openrouter.ai/api/v1/chat/completions",
           "-H", f"Authorization: Bearer {API_KEY}", "-H", "Content-Type: application/json", "-H", "HTTP-Referer: https://openai.com/codex",
           "--data-binary", f"@{payload_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        (LOG / f"{label}_error.log").write_text(result.stderr + "\n" + result.stdout)
        raise RuntimeError(f"OpenRouter request failed for {label}; see log")
    raw_path.write_text(result.stdout)
    response = json.loads(result.stdout)
    content = response["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    parsed["_provenance"] = {
        "requested_model": MODEL,
        "returned_model": response.get("model"),
        "response_id": response.get("id"),
        "created": response.get("created"),
        "usage": response.get("usage"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if len(parsed.get("features", [])) != 12:
        raise ValueError(f"{label} returned {len(parsed.get('features', []))}, expected 12")
    parsed_path.write_text(json.dumps(parsed, indent=2) + "\n")
    print(f"[complete] {label}: 12 features")
    return parsed


def main():
    outputs = [run_one(f"qwen3_32b_generation_{i}") for i in range(1, 4)]
    ids = []
    for o in outputs:
        ids.extend(x["feature_id"] for x in o["features"])
    summary = {"model": MODEL, "generations": 3, "features_before_dedup": len(ids), "unique_ids": len(set(ids)), "feature_ids": ids}
    (OUT / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    state_path = HERE / "state.json"; state = json.loads(state_path.read_text())
    state["status"] = "formula_specs_generated"; state["completed_stages"] = sorted(set(state["completed_stages"] + ["generate_formula_specs"]))
    state["next_stage"] = "validate_and_compute"; state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__": main()
