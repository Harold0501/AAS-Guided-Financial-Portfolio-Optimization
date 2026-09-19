import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# =========================
# Config
# =========================

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen3-14b"  # Change to your LM Studio model name if needed

ASSET_CSV = Path("Assets_and_Instances/combined_asset_components_time.csv")

FEATURE_JSON_FILES = [
    "detailed_ChatGPT_DeepResearch_response.json",
    "detailed_ChatGPT_Optimized_response.json",
    "detailed_Ref_Fatemi2024_response.json",
    "detailed_Ref_Koa2024_response.json",
    "formulate_ChatGPT_DeepResearch_response.json",
    "formulate_ChatGPT_Optimized_response.json",
    "formulate_Ref_Fatemi2024_response.json",
    "formulate_Ref_Koa2024_response.json",
]

OUTPUT_DIR = Path("llm_direct_feature_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_RESPONSE_DIR = OUTPUT_DIR / "raw_responses"
RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_DIR = OUTPUT_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# If testing, set this to a small number such as 5.
# If running all 762 rows, set to None.
MAX_ASSET_ROWS: Optional[int] = None

REQUEST_TIMEOUT = 120
SLEEP_BETWEEN_CALLS = 0.2

# Lower temperature makes output slightly more stable.
TEMPERATURE = 0.1


# =========================
# Helpers
# =========================

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_filename(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Tries to parse a JSON object from LLM output.
    Handles code fences and extra explanation text.
    """
    cleaned = strip_code_fences(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Parsed JSON is not an object.")
    except Exception:
        pass

    # Fallback: extract first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    candidate = match.group(0)
    parsed = json.loads(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Extracted JSON is not an object.")

    return parsed


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except Exception:
        return None


def make_feature_list(feature_json: Dict[str, Any]) -> List[str]:
    features = feature_json.get("features", [])
    names = []
    for f in features:
        name = f.get("name")
        if name:
            names.append(name)
    return names


def build_prompt(asset_row: Dict[str, Any], feature_json: Dict[str, Any], feature_set_name: str) -> str:
    """
    This intentionally asks the LLM to directly generate numeric values.
    It also forces the model to admit estimation because raw market time series are not provided.
    """
    feature_names = make_feature_list(feature_json)

    asset_description = {
        "instance": asset_row.get("instance"),
        "Asset": asset_row.get("Asset"),
        "start": asset_row.get("start"),
        "end": asset_row.get("end"),
        "Components": asset_row.get("Components"),
    }

    return f"""
You are generating numeric portfolio feature values for an algorithm-selection experiment.

IMPORTANT:
You are NOT given raw historical price-volume time series.
You are only given asset/component descriptions and feature definitions.
Therefore, the numeric feature values must be treated as LLM-estimated proxy values, not deterministic computed values.

Task:
Given the asset description and the feature definitions, directly generate one numeric value for each feature.

Feature set name:
{feature_set_name}

Asset description:
{json.dumps(asset_description, ensure_ascii=False, indent=2)}

Feature definitions:
{json.dumps(feature_json, ensure_ascii=False, indent=2)}

Required output:
Return ONLY a valid JSON object.
Do not include markdown.
Do not include explanation outside JSON.

JSON schema:
{{
  "instance": <same instance id>,
  "Asset": <same Asset>,
  "start": <same start>,
  "end": <same end>,
  "feature_set": "{feature_set_name}",
  "value_type": "llm_estimated_not_computed",
  "feature_values": {{
    {", ".join([json.dumps(name) + ": <numeric value>" for name in feature_names])}
  }},
  "confidence": {{
    {", ".join([json.dumps(name) + ": <number from 0 to 1>" for name in feature_names])}
  }},
  "warnings": [
    "Values are estimated by LLM from asset/component descriptions only.",
    "No raw div_adj_close or volume time series were provided.",
    "Values are not reproducible deterministic calculations."
  ]
}}

Rules:
1. Every feature listed in the feature definitions must appear exactly once in feature_values.
2. Each feature value must be numeric, not a string.
3. If the feature output_range is bounded, try to keep the value inside the stated range.
4. If the feature cannot be inferred from asset description, still provide a conservative numeric estimate and set confidence <= 0.30.
5. Do not invent raw price or volume data.
6. Do not say you cannot do the task; generate the requested estimated values.
""".strip()


def call_lm_studio(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator. "
                    "You must output valid JSON only. "
                    "No markdown, no commentary."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": TEMPERATURE,
        "stream": False,
    }

    response = requests.post(
        LM_STUDIO_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def normalize_llm_result(
    parsed: Dict[str, Any],
    asset_row: Dict[str, Any],
    feature_set_name: str,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Converts the nested LLM JSON into one flat CSV row.
    """
    feature_values = parsed.get("feature_values", {})
    confidence = parsed.get("confidence", {})

    row = {
        "instance": asset_row.get("instance"),
        "Asset": asset_row.get("Asset"),
        "start": asset_row.get("start"),
        "end": asset_row.get("end"),
        "feature_set": feature_set_name,
        "value_type": parsed.get("value_type", "llm_estimated_not_computed"),
    }

    for feature_name in feature_names:
        row[feature_name] = safe_float(feature_values.get(feature_name))
        row[f"{feature_name}__confidence"] = safe_float(confidence.get(feature_name))

    warnings = parsed.get("warnings", [])
    if isinstance(warnings, list):
        row["warnings"] = " | ".join(map(str, warnings))
    else:
        row["warnings"] = str(warnings)

    return row


# =========================
# Main pipeline
# =========================

def main() -> None:
    if not ASSET_CSV.exists():
        raise FileNotFoundError(f"Asset CSV not found: {ASSET_CSV.resolve()}")

    asset_df = pd.read_csv(ASSET_CSV)

    required_cols = {"instance", "Asset", "start", "end", "Components"}
    missing_cols = required_cols - set(asset_df.columns)
    if missing_cols:
        raise ValueError(f"Asset CSV missing required columns: {missing_cols}")

    if MAX_ASSET_ROWS is not None:
        asset_df = asset_df.head(MAX_ASSET_ROWS)

    summary_records = []

    for feature_file in FEATURE_JSON_FILES:
        feature_path = Path("Response/" + feature_file)

        if not feature_path.exists():
            print(f"[SKIP] Feature file not found: {feature_path}")
            summary_records.append({
                "feature_file": feature_file,
                "status": "missing_file",
                "num_rows_success": 0,
                "num_rows_failed": 0,
            })
            continue

        feature_set_name = sanitize_filename(feature_file)
        feature_json = load_json(feature_path)
        feature_names = make_feature_list(feature_json)

        output_rows = []
        audit_rows = []

        print(f"\n=== Processing feature set: {feature_set_name} ===")
        print(f"Number of features: {len(feature_names)}")
        print(f"Number of asset rows: {len(asset_df)}")

        for idx, asset_row in asset_df.iterrows():
            asset_dict = asset_row.to_dict()
            instance_id = asset_dict.get("instance")

            prompt = build_prompt(asset_dict, feature_json, feature_set_name)

            raw_response_path = RAW_RESPONSE_DIR / f"{feature_set_name}__row_{idx}__instance_{instance_id}.txt"

            try:
                raw_text = call_lm_studio(prompt)
                raw_response_path.write_text(raw_text, encoding="utf-8")

                parsed = extract_json_object(raw_text)
                flat_row = normalize_llm_result(
                    parsed=parsed,
                    asset_row=asset_dict,
                    feature_set_name=feature_set_name,
                    feature_names=feature_names,
                )

                output_rows.append(flat_row)

                audit_rows.append({
                    "row_index": idx,
                    "instance": instance_id,
                    "feature_set": feature_set_name,
                    "status": "success",
                    "raw_response_file": str(raw_response_path),
                    "error": "",
                })

                print(f"[OK] {feature_set_name} | row={idx} | instance={instance_id}")

            except Exception as e:
                audit_rows.append({
                    "row_index": idx,
                    "instance": instance_id,
                    "feature_set": feature_set_name,
                    "status": "failed",
                    "raw_response_file": str(raw_response_path),
                    "error": repr(e),
                })

                print(f"[FAIL] {feature_set_name} | row={idx} | instance={instance_id} | {repr(e)}")

            time.sleep(SLEEP_BETWEEN_CALLS)

        output_csv = OUTPUT_DIR / f"{feature_set_name}__llm_direct_feature_values.csv"
        audit_csv = AUDIT_DIR / f"{feature_set_name}__audit.csv"

        pd.DataFrame(output_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(audit_rows).to_csv(audit_csv, index=False, encoding="utf-8-sig")

        num_success = sum(1 for r in audit_rows if r["status"] == "success")
        num_failed = sum(1 for r in audit_rows if r["status"] == "failed")

        summary_records.append({
            "feature_file": feature_file,
            "feature_set": feature_set_name,
            "status": "done",
            "num_features": len(feature_names),
            "num_asset_rows": len(asset_df),
            "num_rows_success": num_success,
            "num_rows_failed": num_failed,
            "output_csv": str(output_csv),
            "audit_csv": str(audit_csv),
        })

    summary_csv = OUTPUT_DIR / "run_summary.csv"
    pd.DataFrame(summary_records).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n=== Finished ===")
    print(f"Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()