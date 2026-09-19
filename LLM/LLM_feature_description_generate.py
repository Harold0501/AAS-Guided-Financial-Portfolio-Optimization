import json
import time
import re
from pathlib import Path

import requests


# Config
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen3-14b"

PROMPT_DIR = Path("Prompt")
RESPONSE_DIR = Path("Response")
PREVIOUS_FEATURES_PATH = Path("original_response.json")

TEMPERATURE = 0.4
MAX_TOKENS = 8192
REQUEST_TIMEOUT = 300

ENABLE_THINKING = True

EXPECTED_PROMPT_COUNT = 8

# Basic IO
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Output cleaning
def remove_qwen_thinking(text: str) -> str:
    """
    Qwen3 may output:
    <think>...</think>
    {...}

    This removes the thinking block before JSON parsing.
    """
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    return text.strip()


def extract_json_object(text: str) -> dict:
    """
    Parse model output into JSON.

    This is robust to:
    - pure JSON
    - <think>...</think> + JSON
    - accidental text before or after JSON
    """
    text = remove_qwen_thinking(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    json_text = text[start:end + 1]
    return json.loads(json_text)


# Prompt injection
FEATURE_PLACEHOLDERS = [
    "{{PREVIOUS_FEATURES}}",
    "{PREVIOUS_FEATURES}",
    "{{previous_features}}",
    "{previous_features}",

    "{{FEATURES}}",
    "{FEATURES}",
    "{{features}}",
    "{features}",

    "{{ORIGINAL_FEATURES}}",
    "{ORIGINAL_FEATURES}",
    "{{original_features}}",
    "{original_features}",

    "{{VERSION_B_FEATURES}}",
    "{VERSION_B_FEATURES}",
    "{{version_b_features}}",
    "{version_b_features}",

    "{{VERSIONB_FEATURES}}",
    "{VERSIONB_FEATURES}",
    "{{versionb_features}}",
    "{versionb_features}",
]


def inject_features_into_prompt(prompt_template: str, feature_data: dict) -> str:
    """
    Replace any known feature placeholder with feature_data.

    This handles different prompt styles, including:
    - {{PREVIOUS_FEATURES}}
    - {features}
    - {version_b_features}
    - {{VERSION_B_FEATURES}}
    """
    feature_str = json.dumps(feature_data, ensure_ascii=False, indent=2)

    prompt = prompt_template
    replaced = False

    for placeholder in FEATURE_PLACEHOLDERS:
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, feature_str)
            replaced = True

    if not replaced:
        prompt = append_features_if_no_placeholder(prompt, feature_str)

    if ENABLE_THINKING:
        prompt = "/think\n\n" + prompt
    else:
        prompt = "/no_think\n\n" + prompt

    return prompt


def append_features_if_no_placeholder(prompt_template: str, feature_str: str) -> str:
    """
    Fallback when the prompt has no explicit placeholder.

    This prevents the script from failing just because one prompt writes
    instructions but forgets to include a placeholder token.
    """
    suffix = (
        "\n\n"
        "--------------------------------------------------\n"
        "FEATURE INPUT\n"
        "--------------------------------------------------\n"
        f"{feature_str}\n"
    )
    return prompt_template.rstrip() + suffix


# LM Studio call
def call_lm_studio(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful quantitative finance feature engineering assistant. "
                    "Return only valid JSON. Do not include markdown or explanations outside JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }

    response = requests.post(
        LM_STUDIO_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


# Prompt type detection
def is_detailed_prompt(prompt_path: Path) -> bool:
    return prompt_path.stem.lower().startswith("detailed_")


def is_formulate_prompt(prompt_path: Path) -> bool:
    return prompt_path.stem.lower().startswith("formulate_")


def corresponding_detailed_name(formulate_prompt_path: Path) -> str:
    """
    Example:
    formulate_Ref_Koa2024.txt
    -> detailed_Ref_Koa2024_response.json
    """
    stem = formulate_prompt_path.stem

    if stem.startswith("formulate_"):
        suffix = stem[len("formulate_"):]
        return f"detailed_{suffix}_response.json"

    return ""


def select_input_features(prompt_path: Path, original_features: dict) -> dict:
    """
    Decide which feature JSON to inject.

    detailed_*:
        uses original_response.json

    formulate_*:
        preferably uses corresponding detailed_* output as Version B.
        If unavailable, falls back to original_response.json.
    """
    if is_detailed_prompt(prompt_path):
        return original_features

    if is_formulate_prompt(prompt_path):
        detailed_output_name = corresponding_detailed_name(prompt_path)
        detailed_output_path = RESPONSE_DIR / detailed_output_name

        if detailed_output_path.exists():
            print(f"  Using Version B input: {detailed_output_path}")
            return load_json(detailed_output_path)

        print(
            f"  Warning: corresponding Version B file not found: {detailed_output_path}. "
            f"Fallback to original_response.json."
        )
        return original_features

    return original_features


# Run one prompt
def run_one_prompt(prompt_path: Path, feature_input: dict) -> dict:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    final_prompt = inject_features_into_prompt(prompt_template, feature_input)

    raw_output = call_lm_studio(final_prompt)
    parsed_output = extract_json_object(raw_output)

    return {
        "prompt_name": prompt_path.stem,
        "raw_output": raw_output,
        "parsed_output": parsed_output
    }


# Main pipeline
def main():
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

    original_features = load_json(PREVIOUS_FEATURES_PATH)

    prompt_files = sorted(PROMPT_DIR.glob("*.txt"))

    if EXPECTED_PROMPT_COUNT is not None and len(prompt_files) != EXPECTED_PROMPT_COUNT:
        print(
            f"[Warning] Expected {EXPECTED_PROMPT_COUNT} prompt files, "
            f"but found {len(prompt_files)} in {PROMPT_DIR}."
        )

    if not prompt_files:
        raise FileNotFoundError(f"No .txt prompt files found in {PROMPT_DIR}")

    detailed_prompts = [p for p in prompt_files if is_detailed_prompt(p)]
    formulate_prompts = [p for p in prompt_files if is_formulate_prompt(p)]
    other_prompts = [
        p for p in prompt_files
        if not is_detailed_prompt(p) and not is_formulate_prompt(p)
    ]

    ordered_prompts = detailed_prompts + formulate_prompts + other_prompts

    summary = []

    for i, prompt_path in enumerate(ordered_prompts, start=1):
        print(f"\n[{i}/{len(ordered_prompts)}] Running prompt: {prompt_path.name}")

        output_json_path = RESPONSE_DIR / f"{prompt_path.stem}_response.json"
        raw_output_path = RESPONSE_DIR / f"{prompt_path.stem}_raw.txt"
        error_path = RESPONSE_DIR / f"{prompt_path.stem}_error.txt"

        try:
            feature_input = select_input_features(prompt_path, original_features)
            result = run_one_prompt(prompt_path, feature_input)

            save_text(raw_output_path, result["raw_output"])
            save_json(output_json_path, result["parsed_output"])

            print(f"  Saved JSON: {output_json_path}")
            print(f"  Saved raw output: {raw_output_path}")

            summary.append({
                "prompt": prompt_path.name,
                "status": "success",
                "json_output": str(output_json_path),
                "raw_output": str(raw_output_path)
            })

        except Exception as e:
            error_message = (
                f"Failed on {prompt_path.name}\n\n"
                f"{type(e).__name__}: {e}"
            )
            save_text(error_path, error_message)

            print(f"  Failed. Error saved to: {error_path}")

            summary.append({
                "prompt": prompt_path.name,
                "status": "failed",
                "error_file": str(error_path),
                "error": str(e)
            })

        time.sleep(1)

    summary_path = RESPONSE_DIR / "run_summary.json"
    save_json(summary_path, {"runs": summary})

    print(f"\nDone. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()