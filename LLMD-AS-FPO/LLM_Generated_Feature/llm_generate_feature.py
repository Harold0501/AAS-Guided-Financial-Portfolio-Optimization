import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from openai import OpenAI
from pydantic import BaseModel
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import partial

class ResearchPaperExtraction(BaseModel):
    expected_return: float

# Make model_name arg parser
parser = argparse.ArgumentParser(description='Generate LLM-based stock predictions with parallel processing')
parser.add_argument('-m', "--model_name", type=str, default="qwen", 
                    choices=['llama', 'gemma', 'qwen'],
                    help="Model name: llama, gemma, or qwen")
parser.add_argument('-n', "--num_samples", type=int, default=30, 
                    help="Number of LLM samples per stock (default: 30, try 10 for 3x faster)")
parser.add_argument('-w', "--max_workers", type=int, default=10, 
                    help="Number of parallel workers for LLM calls (default: 10, max recommended: 10)")
parser.add_argument('-r', "--resume", action='store_true', 
                    help="Resume from existing results (skip already processed predictions)")
args = parser.parse_args()

"""
Model Configuration
"""
model_name = args.model_name
if model_name == 'llama':
    client = OpenAI(
        base_url="http://localhost:8001/v1/", 
        api_key="-",
    )
elif model_name == 'gemma':
    client = OpenAI(
        base_url="http://localhost:8002/v1/",
        api_key="-",
    )
elif model_name == 'qwen':
    client = OpenAI(
        base_url="http://localhost:8003/v1/", 
        api_key="-",
    )
else:
    raise ValueError(f"Unknown model: {model_name}")

"""
Data Loading and Preparation
"""
# Create directories if they don't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
response_dir = os.path.join(script_dir, 'response')
os.makedirs(response_dir, exist_ok=True)

# Load time_range.xlsx to get instances
time_range_path = os.path.join(script_dir, 'time_range.xlsx')
print(f"Loading time range data from {time_range_path}")
time_range_df = pd.read_excel(time_range_path)
print(f"Loaded {len(time_range_df)} instances from time_range.xlsx")

# Load S&P500 ticker data for company information
sp500_table_path = os.path.join(os.path.dirname(script_dir), "sp500_table.csv")
if os.path.exists(sp500_table_path):
    sp500_table = pd.read_csv(sp500_table_path)[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    print(f"Loaded S&P500 table with {len(sp500_table)} companies")
else:
    print(f"Warning: S&P500 table not found at {sp500_table_path}")
    sp500_table = pd.DataFrame(columns=['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry'])

"""
Define Training Windows and Prediction Horizons
"""
# Configuration from arguments
num_samples = args.num_samples
max_workers = args.max_workers
resume_mode = args.resume

print(f"\n{'='*80}")
print(f"CONFIGURATION:")
print(f"  Model: {model_name}")
print(f"  Total instances: {len(time_range_df)}")
print(f"  Samples per stock: {num_samples}")
print(f"  Parallel workers: {max_workers}")
print(f"  Resume mode: {resume_mode}")
print(f"{'='*80}\n")

"""
Prompt Engineering Functions
"""
def make_system_prompt(horizon_years):
    """
    Create system prompt in English for LLM to predict annualized returns.
    """
    return f"""You are a financial analyst AI designed to predict stock returns based on historical price data.

Given the following information:
- Stock ticker symbol
- Company name (if available)
- GICS sector and sub-industry classification (if available)
- Historical daily returns (percentage changes) from dividend-adjusted closing prices over the past year

Your task is to predict the ANNUALIZED RETURN for the next {horizon_years} year(s).

Important:
- The input data is based on dividend-adjusted closing prices
- Historical data represents daily percentage changes over the training period
- You should output a single float value representing the expected annualized return (as a decimal, e.g., 0.05 for 5% annual return)
- Do NOT include any commentary, explanations, or additional text
- Output ONLY the numerical value

Example: If you predict a 7.5% annualized return over {horizon_years} year(s), output: 0.075"""

def make_user_prompt(ticker, data_dict):
    """
    Create user prompt with stock-specific information.
    """
    security = data_dict[ticker]['Security'] if data_dict[ticker]['Security'] else 'N/A'
    sector = data_dict[ticker]['GICS Sector'] if data_dict[ticker]['GICS Sector'] else 'N/A'
    sub_industry = data_dict[ticker]['GICS Sub-Industry'] if data_dict[ticker]['GICS Sub-Industry'] else 'N/A'
    pct_change = data_dict[ticker]['pct_change']
    
    return f"""Stock Information:
Ticker: {ticker}
Company Name: {security}
GICS Sector: {sector}
GICS Sub-Industry: {sub_industry}

Historical Daily Returns (dividend-adjusted):
{pct_change}

Please predict the annualized return for the next {data_dict[ticker]['prediction_horizon']} year(s)."""

"""
Parallel LLM Prediction Functions
"""
def predict_single_stock(ticker, data_dict, system_prompt, model_name, client, num_samples):
    """
    Generate predictions for a single stock (called in parallel).
    This function is thread-safe and can be executed concurrently.
    """
    user_prompt = make_user_prompt(ticker, data_dict)
    
    # Concatenate prompts for gemma (doesn't support system prompt)
    if model_name == 'gemma':
        user_prompt = system_prompt + "\n\n" + user_prompt
    
    # Get responses num_samples times to estimate uncertainty
    answers = []
    for attempt in range(num_samples):
        try:
            if model_name == 'llama':
                completion = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_body={"guided_json": ResearchPaperExtraction.model_json_schema()}
                )
            elif model_name == 'gemma':
                completion = client.chat.completions.create(
                    model="google/gemma-7b",
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"guided_json": ResearchPaperExtraction.model_json_schema()}
                )
            elif model_name == 'qwen':
                completion = client.chat.completions.create(
                    model="Qwen/Qwen2-7B-Instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"guided_json": ResearchPaperExtraction.model_json_schema()}
                )
            
            expected_return_dict = eval(completion.choices[0].message.content)
            answers.append(expected_return_dict['expected_return'])
            
        except Exception as e:
            # Silently skip failed attempts
            continue
    
    return ticker, answers

"""
Main Processing Loop
"""
overall_start_time = time.time()
processed_count = 0
skipped_count = 0

# Store results for CSV generation
all_instance_stats = []

# Process each instance from time_range.xlsx
for idx, row in tqdm(time_range_df.iterrows(), total=len(time_range_df), desc="Processing instances"):
    instance_num = row['Instance #']
    asset_file = row['Asset #']
    start_time_str = row['Start Time']
    end_time_str = row['End Time']
    
    # Parse dates
    start_date = pd.to_datetime(start_time_str)
    end_date = pd.to_datetime(end_time_str)
    
    # Calculate training period (year before start_date)
    train_start_date = start_date - pd.DateOffset(years=1) + pd.DateOffset(days=1)
    train_end_date = start_date
    
    # Calculate prediction horizon in years
    horizon_years = round((end_date - start_date).days / 365.25, 1)
    
    print(f"\n{'='*80}")
    print(f"Instance {instance_num}: {asset_file}")
    print(f"  Training: {train_start_date.date()} to {train_end_date.date()}")
    print(f"  Prediction: {start_date.date()} to {end_date.date()} ({horizon_years} years)")
    print(f"{'='*80}")
    
    # Load asset data
    asset_path = os.path.join(script_dir, 'data', f'{asset_file}.csv')
    if not os.path.exists(asset_path):
        print(f"  ⚠️  Asset file not found: {asset_path}, skipping...")
        continue
    
    try:
        raw_data = pd.read_csv(asset_path)
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        
        # Pivot data
        pivot_data = raw_data.pivot(index='date', columns='company', values='div_adj_close')
        pivot_data = pivot_data.sort_index()
        
        # Extract training data
        train_mask = (pivot_data.index >= train_start_date) & (pivot_data.index <= train_end_date)
        train_data = pivot_data[train_mask]
        
        if len(train_data) == 0:
            print(f"  ⚠️  No training data available, skipping...")
            continue
        
        print(f"  Training data: {len(train_data)} trading days")
        
        # Calculate daily returns
        returns = train_data.pct_change(fill_method=None).iloc[1:]
        companies = returns.columns.tolist()
        print(f"  Number of companies: {len(companies)}")
        
    except Exception as e:
        print(f"  ⚠️  Error loading data: {str(e)}, skipping...")
        continue
    
    # Define output file path
    output_filename = f"{model_name}_instance_{instance_num}.json"
    output_path = os.path.join(response_dir, output_filename)
    
    # Check if we should skip (resume mode and file exists)
    if resume_mode and os.path.exists(output_path):
        print(f"  ⏭️  Skipping (file already exists)")
        skipped_count += 1
        
        # Load existing results for statistics
        try:
            with open(output_path, 'r') as f:
                data_dict = json.load(f)
            
            # Calculate statistics
            stock_avgs = []
            stock_vars = []
            for company in companies:
                if company in data_dict and len(data_dict[company].get('expected_return', [])) > 0:
                    returns_list = data_dict[company]['expected_return']
                    stock_avgs.append(np.mean(returns_list))
                    stock_vars.append(np.var(returns_list))
            
            if len(stock_avgs) > 0:
                instance_stats = {
                    'instance': instance_num,
                    'max_avg': np.max(stock_avgs),
                    'min_avg': np.min(stock_avgs),
                    'mean_avg': np.mean(stock_avgs),
                    'median_avg': np.median(stock_avgs),
                    'std_avg': np.std(stock_avgs),
                    'max_var': np.max(stock_vars),
                    'min_var': np.min(stock_vars),
                    'mean_var': np.mean(stock_vars),
                    'median_var': np.median(stock_vars),
                    'std_var': np.std(stock_vars)
                }
                all_instance_stats.append(instance_stats)
        except Exception as e:
            print(f"  ⚠️  Error loading existing results: {str(e)}")
        
        continue
    
    # Organize data for prediction
    data_dict = {}
    for company in companies:
        # Get company info from sp500_table if available
        if len(sp500_table) > 0 and company in sp500_table['Symbol'].values:
            company_info = sp500_table[sp500_table['Symbol'] == company].iloc[0]
            security = company_info['Security']
            sector = company_info['GICS Sector']
            sub_industry = company_info['GICS Sub-Industry']
        else:
            security = ''
            sector = ''
            sub_industry = ''
        
        data_dict[company] = {
            'ticker': company,
            'Security': security,
            'GICS Sector': sector,
            'GICS Sub-Industry': sub_industry,
            'pct_change': returns[company].dropna().tolist(),
            'instance': instance_num,
            'asset_file': asset_file,
            'train_start': str(train_start_date.date()),
            'train_end': str(train_end_date.date()),
            'pred_start': str(start_date.date()),
            'pred_end': str(end_date.date()),
            'prediction_horizon': horizon_years
        }
    
    # Generate LLM predictions
    system_prompt = make_system_prompt(horizon_years)
    
    print(f"  🚀 Starting parallel predictions with {max_workers} workers...")
    start_time = time.time()
    
    # Create partial function
    predict_func = partial(
        predict_single_stock,
        data_dict=data_dict,
        system_prompt=system_prompt,
        model_name=model_name,
        client=client,
        num_samples=num_samples
    )
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(predict_func, ticker): ticker for ticker in companies}
        
        with tqdm(total=len(companies), desc=f"  Predicting", leave=False, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_result, answers = future.result()
                    if len(answers) > 0:
                        data_dict[ticker_result]['expected_return'] = answers
                    else:
                        data_dict[ticker_result]['expected_return'] = []
                except Exception as e:
                    print(f"\n      ⚠️  Error processing {ticker}: {str(e)}")
                    data_dict[ticker]['expected_return'] = []
                finally:
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    api_calls_made = len(companies) * num_samples
    print(f"  ✓ Completed in {elapsed_time:.1f}s ({elapsed_time/len(companies):.2f}s/stock, "
          f"{elapsed_time/api_calls_made:.3f}s/call)")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"  💾 Saved: {output_filename}")
    
    # Calculate statistics for this instance
    stock_avgs = []
    stock_vars = []
    valid_predictions = []
    
    for company in companies:
        if len(data_dict[company].get('expected_return', [])) > 0:
            returns_list = data_dict[company]['expected_return']
            stock_avgs.append(np.mean(returns_list))
            stock_vars.append(np.var(returns_list))
            valid_predictions.append(company)
    
    avg_samples = np.mean([len(data_dict[ticker].get('expected_return', [])) 
                          for ticker in valid_predictions]) if valid_predictions else 0
    print(f"  📈 Success: {len(valid_predictions)}/{len(companies)} companies "
          f"(avg {avg_samples:.1f}/{num_samples} samples per stock)")
    
    # Store statistics
    if len(stock_avgs) > 0:
        instance_stats = {
            'instance': instance_num,
            'max_avg': np.max(stock_avgs),
            'min_avg': np.min(stock_avgs),
            'mean_avg': np.mean(stock_avgs),
            'median_avg': np.median(stock_avgs),
            'std_avg': np.std(stock_avgs),
            'max_var': np.max(stock_vars),
            'min_var': np.min(stock_vars),
            'mean_var': np.mean(stock_vars),
            'median_var': np.median(stock_vars),
            'std_var': np.std(stock_vars)
        }
        all_instance_stats.append(instance_stats)
    
    processed_count += 1

# Save statistics to CSV
csv_output_path = os.path.join(script_dir, 'llm_generated_feature.csv')
if len(all_instance_stats) > 0:
    stats_df = pd.DataFrame(all_instance_stats)
    stats_df = stats_df.sort_values('instance')
    stats_df.to_csv(csv_output_path, index=False)
    print(f"\n{'='*80}")
    print(f"📊 STATISTICS SAVED")
    print(f"{'='*80}")
    print(f"  Output file: {csv_output_path}")
    print(f"  Total instances with statistics: {len(stats_df)}")
    print(f"{'='*80}\n")
else:
    print(f"\n⚠️  No statistics to save (no valid predictions)")

# Final summary
total_elapsed = time.time() - overall_start_time
print(f"\n{'='*80}")
print(f"🎉 ALL PREDICTIONS COMPLETED!")
print(f"{'='*80}")
print(f"  Processed: {processed_count} instances")
print(f"  Skipped: {skipped_count} (already existed)")
print(f"  Total instances: {len(time_range_df)}")
print(f"  Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
print(f"  Results saved in: {response_dir}")
print(f"  Statistics saved in: {csv_output_path}")
print(f"{'='*80}\n")

# Performance tip
if num_samples == 30 and max_workers == 5:
    print("💡 Performance tip:")
    print("   For faster runs, try: --num_samples 10 --max_workers 10")
    print("   This reduces runtime by ~70% with minimal accuracy loss\n")
