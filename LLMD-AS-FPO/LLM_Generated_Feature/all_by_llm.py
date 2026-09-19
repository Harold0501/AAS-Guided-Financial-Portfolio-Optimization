import pandas as pd
import json
import os
from openai import OpenAI
from datetime import datetime
import numpy as np

# Configuration
VLLM_BASE_URL = "http://localhost:8003/v1/"
MODEL_NAME = "Qwen/Qwen3-8B"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_RANGE_FILE = os.path.join(SCRIPT_DIR, "time_range.xlsx")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "response.json")

# Sampling configuration
NUM_INSTANCES_TO_SAMPLE = 10  # Number of instances to sample
MAX_ROWS_PER_INSTANCE = 20    # Max rows to show per instance in prompt

# Initialize OpenAI client for vLLM
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="dummy-key"  # vLLM doesn't require a real key
)

def format_dataframe_as_markdown(df, max_rows=100):
    """
    Convert DataFrame to Markdown table format.
    Limit to max_rows to avoid token overflow.
    """
    if len(df) > max_rows:
        df_sample = df.head(max_rows)
        truncated = True
    else:
        df_sample = df
        truncated = False
    
    # Create markdown table
    headers = " | ".join(df_sample.columns)
    separator = " | ".join(["---"] * len(df_sample.columns))
    
    rows = []
    for _, row in df_sample.iterrows():
        row_str = " | ".join([str(val) for val in row.values])
        rows.append(row_str)
    
    markdown_table = f"{headers}\n{separator}\n" + "\n".join(rows)
    
    if truncated:
        markdown_table += f"\n\n(Showing first {max_rows} rows out of {len(df)} total rows)"
    
    return markdown_table

def load_instance_data(asset_file, start_time, end_time):
    """
    Load data for a specific instance based on asset file and time range.
    """
    asset_path = os.path.join(DATA_DIR, f"{asset_file}.csv")
    
    if not os.path.exists(asset_path):
        print(f"  ⚠️  Asset file not found: {asset_path}")
        return None
    
    try:
        # Load CSV data
        df = pd.read_csv(asset_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Parse time range
        start_date = pd.to_datetime(start_time)
        end_date = pd.to_datetime(end_time)
        
        # Filter data by time range
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df[mask].copy()
        
        # Sort by date
        filtered_df = filtered_df.sort_values('date')
        
        if len(filtered_df) == 0:
            print(f"  ⚠️  No data in time range {start_time} to {end_time}")
            return None
        
        return filtered_df
    
    except Exception as e:
        print(f"  ❌ Error loading data: {str(e)}")
        return None

def create_system_prompt():
    """
    Create system prompt for LLM to generate universal features.
    """
    return """You are an expert financial data analyst specializing in feature engineering for portfolio-level stock market analysis.

Your task is to analyze multiple instances of stock PORTFOLIO data and propose THREE universal numerical features that can capture deep, complex patterns across an entire asset collection (portfolio) during a specific time period.

Each "instance" represents a PORTFOLIO (asset collection) of multiple stocks during a specific time period.    Features should describe characteristics of the ENTIRE PORTFOLIO, not individual stocks.    Features should capture portfolio-level patterns, inter-stock relationships, and collective behavior.

Requirements for the features:
1. Each feature must be a NUMERICAL metric that describes portfolio-level characteristics but not necessarily to be able to be computed from stock price time series data
2. Features should capture COMPLEX, DEEP-LEVEL patterns that are NOT easily computable through simple formulas (e.g., not just mean, variance, or simple moving averages), such as abstract semantic features that capture high-level market concepts
3. Features should be UNIVERSAL - applicable to any stock portfolio across different time periods and market compositions
4. Features should have predictive value for understanding portfolio behavior, collective risk, or aggregate returns
5. Each feature should capture a DISTINCT aspect of portfolio dynamics (avoid redundancy)

You will be provided with multiple portfolio instances.    Each instance contains time series data for MULTIPLE stocks:
- date: Trading date
- company: Stock ticker symbol (multiple companies per instance)
- open, high, low, close: Daily price data
- volume: Trading volume
- div_adj_close: Dividend-adjusted closing price

Based on these portfolio samples, propose THREE universal features that describe PORTFOLIO-LEVEL characteristics.

Output format (JSON):
{
  "features": [
    {
      "name": "Feature Name",
      "description": "Detailed description of what this portfolio-level feature captures, how it relates to the collective behavior of stocks in the portfolio, and why it's valuable for portfolio analysis and prediction"
    },
    {
      "name": "Feature Name 2",
      "description": "..."
    },
    {
      "name": "Feature Name 3",
      "description": "..."
    }
  ]
}

Only output valid JSON. DO NOT include any additional text or explanations outside the JSON structure."""

def create_user_prompt(instances_data):
    """
    Create user prompt with all instances data.
    """
    prompt = "I am providing you with multiple instances of stock market time series data. Please analyze these samples and propose three universal numerical features.\n\n"
    
    for i, (instance_num, asset_file, start_time, end_time, df) in enumerate(instances_data, 1):
        prompt += f"### Instance {instance_num}: {asset_file}\n"
        prompt += f"Time Period: {start_time} to {end_time}\n"
        prompt += f"Number of records: {len(df)}\n"
        prompt += f"Number of companies: {df['company'].nunique()}\n\n"
        
        # Format data as markdown table (limit rows to avoid token overflow)
        markdown_table = format_dataframe_as_markdown(df, max_rows=MAX_ROWS_PER_INSTANCE)
        prompt += f"Data:\n```\n{markdown_table}\n```\n\n"
        
        # Add separator between instances
        if i < len(instances_data):
            prompt += "---\n\n"
    
    prompt += "Based on these data samples, please provide three universal numerical features in JSON format as specified."
    
    return prompt

def call_llm(system_prompt, user_prompt):
    """
    Call vLLM API to generate features.
    """
    try:
        print("\n🤖 Calling LLM to generate universal features...")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=40960
        )
        
        content = response.choices[0].message.content
        
        # Clean response: Remove <think> tags if present (Qwen models sometimes add them)
        if "<think>" in content and "</think>" in content:
            print("  ℹ️  Detected <think> tags, extracting actual response...")
            # Extract content after </think>
            think_end = content.find("</think>")
            content = content[think_end + 8:].strip()
        
        # Try to find JSON in the response
        # Look for the first { and last }
        json_start = content.find("{")
        json_end = content.rfind("}")
        
        if json_start == -1 or json_end == -1:
            print(f"  ❌ No JSON found in response")
            print(f"  Raw response: {content}")
            return None
        
        json_content = content[json_start:json_end + 1]
        
        # Parse JSON response
        result = json.loads(json_content)
        
        return result
    
    except json.JSONDecodeError as e:
        print(f"  ❌ Error parsing JSON response: {str(e)}")
        print(f"  Raw response: {content}")
        return None
    except Exception as e:
        print(f"  ❌ Error calling LLM: {str(e)}")
        return None

def main():
    """
    Main execution function.
    """
    print("="*80)
    print("LLM-Based Universal Feature Generation")
    print("="*80)
    
    # Load time_range.xlsx
    print(f"\n📂 Loading time range data from: {TIME_RANGE_FILE}")
    
    if not os.path.exists(TIME_RANGE_FILE):
        print(f"❌ Error: {TIME_RANGE_FILE} not found!")
        return
    
    time_range_df = pd.read_excel(TIME_RANGE_FILE)
    print(f"✓ Loaded {len(time_range_df)} instances")
    
    # Sample instances if there are more than NUM_INSTANCES_TO_SAMPLE
    if len(time_range_df) > NUM_INSTANCES_TO_SAMPLE:
        print(f"\n🎲 Sampling {NUM_INSTANCES_TO_SAMPLE} instances from {len(time_range_df)} total instances...")
        # Use random sampling for better diversity
        sampled_indices = np.random.choice(len(time_range_df), NUM_INSTANCES_TO_SAMPLE, replace=False)
        sampled_indices = sorted(sampled_indices)  # Sort for easier tracking
        time_range_df = time_range_df.iloc[sampled_indices].reset_index(drop=True)
        print(f"✓ Sampled instances: {time_range_df['Instance #'].tolist()}")
    else:
        print(f"\n📊 Using all {len(time_range_df)} instances (less than {NUM_INSTANCES_TO_SAMPLE})")
    
    # Load data for sampled instances
    print(f"\n📊 Loading data for {len(time_range_df)} instances...")
    instances_data = []
    
    for idx, row in time_range_df.iterrows():
        instance_num = row['Instance #']
        asset_file = row['Asset #']
        start_time = row['Start Time']
        end_time = row['End Time']
        
        print(f"\n  Instance {instance_num}: {asset_file}")
        print(f"    Time range: {start_time} to {end_time}")
        
        df = load_instance_data(asset_file, start_time, end_time)
        
        if df is not None:
            print(f"    ✓ Loaded {len(df)} records, {df['company'].nunique()} companies")
            instances_data.append((instance_num, asset_file, start_time, end_time, df))
        else:
            print(f"    ✗ Failed to load data")
    
    if len(instances_data) == 0:
        print("\n❌ No valid data loaded. Exiting.")
        return
    
    print(f"\n✓ Successfully loaded {len(instances_data)} instances")
    
    # Create prompts
    print("\n📝 Creating prompts...")
    system_prompt = create_system_prompt()
    user_prompt = create_user_prompt(instances_data)
    
    print(f"  System prompt length: {len(system_prompt)} characters")
    print(f"  User prompt length: {len(user_prompt)} characters")
    
    # Call LLM
    result = call_llm(system_prompt, user_prompt)
    
    if result is None:
        print("\n❌ Failed to generate features. Exiting.")
        return
    
    # Save result
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("✓ Results saved successfully!")
    
    # Display results
    print("\n" + "="*80)
    print("Generated Universal Features:")
    print("="*80)
    
    if 'features' in result:
        for i, feature in enumerate(result['features'], 1):
            print(f"\n{i}. {feature.get('name', 'N/A')}")
            print(f"   {feature.get('description', 'N/A')}")
    
    print("\n" + "="*80)
    print("🎉 Process completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
