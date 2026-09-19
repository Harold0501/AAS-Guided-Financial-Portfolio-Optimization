import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def fetch_gdelt_news(keyword, start_date, end_date, max_records=250):
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={keyword}"
        f"&mode=artlist"
        f"&maxrecords={max_records}"
        f"&format=csv"
        f"&startdatetime={start_date}000000"
        f"&enddatetime={end_date}235959"
    )

    try:
        df = pd.read_csv(url)
        df["keyword"] = keyword
        return df
    except Exception as e:
        print(f"[Error] Failed to fetch for {keyword}: {e}")
        return pd.DataFrame()

stock_keywords = ["Apple", "Tesla", "Alibaba", "贵州茅台"]

start_date = "20240101"
end_date = "20240331"

all_data = []
for kw in tqdm(stock_keywords, desc="Fetching news"):
    df_kw = fetch_gdelt_news(kw, start_date, end_date, max_records=250)
    all_data.append(df_kw)

df_all = pd.concat(all_data, ignore_index=True)

df_all.to_csv("gdelt_stock_news.csv", index=False, encoding="utf-8-sig")

print(f"共抓取 {len(df_all)} 条新闻。")
print(df_all.columns.tolist())
