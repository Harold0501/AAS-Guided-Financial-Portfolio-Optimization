import pandas as pd
import json
from openai import OpenAI
import os
import csv
from dotenv import load_dotenv

load_dotenv()
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

df = pd.read_excel("asset.xlsx")

features = [
    "Average Pairwise Correlation",
    "Dispersion of Returns",
    "Concentration Index (HHI)",
    "Turnover Rate",
    "Liquidity (Amihud Illiquidity)",
    "Mean P/E Ratio",
    "Mean P/B Ratio",
    "Dividend Yield (avg)",
    "Debt-to-Equity",
    "Earnings Growth Rate (avg)",
    "ROE / ROA",
    "Market Cap Weighted Average Size",
    "Value at Risk (VaR, 95%)",
    "Conditional VaR (CVaR)",
    "Upside Capture Ratio",
    "Downside Capture Ratio"
]
output_file = "asset_features.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["asset", "start_time", "end_time"] + features
    writer.writerow(header)

for idx, row in df.iterrows():
    asset = row["Asset"]
    start_time = row["start_time"]
    end_time = start_time + pd.DateOffset(years=1)
    start_bound = row["start_time"]
    end_bound = row["end_time"]

    prompt = f"""You are a professional financial assistant designed to generate feature values for an asset of stocks. 
        You do not need exact data but please refer to the previous data you have to generate the predictions. 
        Here is a list of stock tickers: {asset}.
        Your task is to generate {len(features)} features: {", ".join(features)}, for this asset.
        Use stock data from {start_time} to {end_time}.
        Return the results in a JSON file with the following schema:
        {{
            "start_time": "{start_time}",
            "end_time": "{end_time}",
            "features": {{
                "{features[0]}": {{value}},
                ...
            }}
        }}
        Do not include any information beyond this schema.
        """

    while start_time < end_time and start_time < end_bound:

        end_time = start_time + pd.DateOffset(years=1)

        while end_time <= end_bound:

            for i in range(3):
                response = client.chat.completions.create(
                    model="qwen/qwen3-235b-a22b-07-25",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8
                )

                content = response.choices[0].message.content
                print(content)

                try:
                    json_data = json.loads(content)

                    with open(output_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        row_data = [idx, start_time, end_time]
                        for feat in features:
                            row_data.append(json_data["features"].get(feat, None))
                        writer.writerow(row_data)

                    print(f"Write asset {idx} on {start_time} to {end_time} to csv.")
                    break

                except Exception as e:
                    print(f"[{i}] Error parsing JSON for asset {idx} on {start_time} to {end_time}: {e}")

            else:
                raise ValueError(f"Failed to get valid JSON response for asset {idx}  on {start_time} to {end_time} after 3 attempts.")

            end_time += pd.DateOffset(years=1)

        start_time += pd.DateOffset(years=1)

print("SUCCESS: Feature Data saved to asset_features.csv")