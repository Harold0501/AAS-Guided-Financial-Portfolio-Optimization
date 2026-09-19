import pandas as pd
import json
import os
from openai import OpenAI
from tqdm import tqdm
import time

# Configuration
VLLM_BASE_URL = "http://localhost:8003/v1/"
MODEL_NAME = "Qwen/Qwen3-32B"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_RANGE_FILE = os.path.join(SCRIPT_DIR, "time_range.xlsx")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FEATURE_DEF_FILE = os.path.join(SCRIPT_DIR, "response.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "all_by_llm.csv")
RAW_RESPONSE_FILE = os.path.join(SCRIPT_DIR, "iterate_response.json")  # Store raw LLM responses

# Data sampling for each instance (to avoid token overflow)
MAX_ROWS_PER_INSTANCE = 100  # Max rows to send to LLM per instance

# Initialize OpenAI client for vLLM with extended timeout
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="dummy-key",
    timeout=1200.0  # Extended timeout to 120 seconds
)

def load_feature_definitions():
    """
    Load feature definitions from response.json.
    """
    if not os.path.exists(FEATURE_DEF_FILE):
        print(f"❌ Error: Feature definition file not found: {FEATURE_DEF_FILE}")
        print("   Please run all_by_llm.py first to generate feature definitions.")
        return None

    with open(FEATURE_DEF_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'features' not in data or len(data['features']) < 3:
        print(f"❌ Error: Invalid feature definition file. Expected 3 features.")
        return None

    return data['features']

def load_instance_data(asset_file, start_time, end_time):
    """
    Load data for a specific instance based on asset file and time range.
    """
    asset_path = os.path.join(DATA_DIR, f"{asset_file}.csv")

    if not os.path.exists(asset_path):
        return None

    try:
        df = pd.read_csv(asset_path)
        df['date'] = pd.to_datetime(df['date'])

        start_date = pd.to_datetime(start_time)
        end_date = pd.to_datetime(end_time)

        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df[mask].copy()
        filtered_df = filtered_df.sort_values('date')

        if len(filtered_df) == 0:
            return None

        return filtered_df

    except Exception as e:
        return None

def format_dataframe_compact(df, max_rows=100):
    """
    Format DataFrame as compact text representation.
    Limit the number of rows to avoid exceeding token limits.
    """
    if len(df) > max_rows:
        # Sample evenly across the time period
        indices = [int(i * len(df) / max_rows) for i in range(max_rows)]
        df_sample = df.iloc[indices]
    else:
        df_sample = df

    lines = []
    lines.append(f"Records: {len(df)}, Companies: {df['company'].nunique()}, Period: {df['date'].min().date()} to {df['date'].max().date()}")
    lines.append(f"Companies: {', '.join(df['company'].unique()[:10])}{'...' if df['company'].nunique() > 10 else ''}")
    lines.append("")

    for idx, row in df_sample.iterrows():
        line = f"{row['date'].date()}|{row['company']}|C:{row['close']:.1f} V:{row['volume']:.0f}"
        lines.append(line)

    return "\n".join(lines)

def create_system_prompt(features):
    """
    Create system prompt for computing feature values.
    """
    feature_list = "\n".join([f"{i+1}. {f['name']}" for i, f in enumerate(features)])

    return f"""You are a financial analyst. Compute numerical values for these portfolio features:

{feature_list}

Important: The features to be extracted must be complex, abstract, and deep semantic features. They should not be calculated using simple mathematical formulas. Instead, derive them by observing the overall relationships and patterns in the data.

Output JSON only:
{{
  "feature_1": <number>,
  "feature_2": <number>,
  "feature_3": <number>
}}"""

def create_user_prompt(instance_num, asset_file, start_time, end_time, df, features):
    """
    Create user prompt for a specific instance.
    Limit the number of rows to avoid exceeding token limits.
    """
    data_summary = format_dataframe_compact(df, max_rows=100)  # Limit to 100 rows

    prompt = f"""Portfolio {instance_num} ({asset_file}, {start_time.date()} to {end_time.date()}):

{data_summary}

Compute feature values as JSON."""

    return prompt

def call_llm_for_instance(instance_num, asset_file, start_time, end_time, df, features, max_retries=3):
    """
    Call LLM to compute feature values for a specific instance.
    Returns: (feature_values_dict, raw_response_string) or (None, raw_response_string)
    """
    system_prompt = create_system_prompt(features)
    user_prompt = create_user_prompt(instance_num, asset_file, start_time, end_time, df, features)

    raw_response = None  # Store the raw response

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=20480  # Remove token limit to allow full output
            )

            content = response.choices[0].message.content.strip()
            raw_response = content  # Store raw response

            print(f"\n=== Instance {instance_num} Full LLM Response ===")
            print(content)
            print(f"=== End Response ===\n")

            json_search_start = 0
            if "<think>" in content and "</think>" in content:
                think_end = content.find("</think>")
                json_search_start = think_end + 8

            json_start = content.find("{", json_search_start)
            json_end = content.rfind("}")

            if json_start == -1 or json_end == -1:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}: No JSON found, retrying...")
                    time.sleep(2)
                    continue
                print("All attempts: No JSON brackets found")
                return None, raw_response

            json_content = content[json_start:json_end + 1]
            print(f"Extracted JSON: {json_content}")

            result = json.loads(json_content)

            if all(key in result for key in ['feature_1', 'feature_2', 'feature_3']):
                return result, raw_response
            else:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}: Missing feature keys, retrying...")
                    time.sleep(2)
                    continue
                print("All attempts: Invalid JSON structure")
                return None, raw_response

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: JSON decode error ({str(e)}), retrying...")
                time.sleep(2)
                continue
            print(f"All attempts: JSON decode failed - {str(e)}")
            return None, raw_response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Error ({str(e)}), retrying...")
                time.sleep(2)
                continue
            print(f"All attempts: Exception - {str(e)}")
            return None, raw_response

    return None, raw_response

def save_partial_results(instance_num, feature_values, raw_response):
    """
    Save partial results to output files.
    """
    # Save raw response
    with open(RAW_RESPONSE_FILE, 'a', encoding='utf-8') as f:
        json.dump({
            "instance": instance_num,
            "raw_llm_response": raw_response,
            "extracted_features": feature_values
        }, f, ensure_ascii=False)
        f.write("\n")

    # Save feature values to CSV
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        if feature_values:
            f.write(f"{instance_num},{feature_values['feature_1']},{feature_values['feature_2']},{feature_values['feature_3']}\n")
        else:
            f.write(f"{instance_num},,,\n")

def main():
    """
    Main execution function.
    """
    print("="*80)
    print("LLM-Based Feature Value Generation for All Instances")
    print("="*80)

    features = load_feature_definitions()
    if features is None:
        return

    if not os.path.exists(TIME_RANGE_FILE):
        print(f"❌ Error: {TIME_RANGE_FILE} not found!")
        return

    time_range_df = pd.read_excel(TIME_RANGE_FILE)
    print(f"✓ Loaded {len(time_range_df)} instances")

    for idx, row in tqdm(time_range_df.iterrows(), total=len(time_range_df), desc="Processing"):
        instance_num = row['Instance #']
        asset_file = row['Asset #']
        start_time = row['Start Time']
        end_time = row['End Time']

        df = load_instance_data(asset_file, start_time, end_time)
        if df is None:
            print(f"⚠️ Instance {instance_num}: Failed to load data")
            save_partial_results(instance_num, None, "Failed to load data")
            continue

        feature_values, raw_response = call_llm_for_instance(
            instance_num, asset_file, start_time, end_time, df, features
        )

        save_partial_results(instance_num, feature_values, raw_response)

if __name__ == "__main__":
    main()
