"""
完整测试：处理第1个instance的所有股票
"""
import pandas as pd
import yfinance as yf
import akshare as ak
from datetime import timedelta
import time
from pathlib import Path
from urllib.parse import quote

# 中国知名公司英文名映射字典
CHINESE_COMPANY_ENGLISH_NAMES = {
    "600519": "Kweichow Moutai",
    "000333": "Midea Group",
    "300750": "Contemporary Amperex Technology",
    "600030": "CITIC Securities",
    "600036": "China Merchants Bank",
    "600900": "China Yangtze Power",
    "601166": "Industrial Bank",
    "601318": "Ping An Insurance",
    "601899": "Zijin Mining Group",
    "000858": "Wuliangye Yibin",
}

def is_chinese_stock(ticker):
    return ticker.isdigit() and len(ticker) == 6

def get_company_name(ticker):
    try:
        if is_chinese_stock(ticker):
            if ticker in CHINESE_COMPANY_ENGLISH_NAMES:
                return CHINESE_COMPANY_ENGLISH_NAMES[ticker], 'success'
            try:
                stock_info = ak.stock_individual_info_em(symbol=ticker)
                cn_name = stock_info[stock_info['item'] == '股票简称']['value'].values[0]
                return cn_name, 'fallback'
            except Exception as e:
                print(f"  [Warning] Failed to get Chinese stock info for {ticker}: {e}")
                return None, 'failed'
        else:
            try:
                stock = yf.Ticker(ticker)
                company_name = stock.info.get('longName') or stock.info.get('shortName')
                if company_name:
                    return company_name, 'success'
                else:
                    return None, 'failed'
            except Exception as e:
                print(f"  [Warning] Failed to get US stock info for {ticker}: {e}")
                return None, 'failed'
    except Exception as e:
        print(f"  [Error] Unexpected error for {ticker}: {e}")
        return None, 'failed'

def fetch_gdelt_news(keyword, start_date, end_date, max_records=250, max_retries=3):
    encoded_keyword = quote(keyword)
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={encoded_keyword}"
        f"&mode=artlist"
        f"&maxrecords={max_records}"
        f"&format=csv"
        f"&startdatetime={start_date}000000"
        f"&enddatetime={end_date}235959"
    )
    
    for attempt in range(max_retries):
        try:
            df = pd.read_csv(url)
            df["keyword"] = keyword
            df["ticker"] = ""
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [Retry {attempt + 1}/{max_retries}] Failed: {e}")
                time.sleep(1)
            else:
                print(f"    [Error] All retries failed: {e}")
                return None
    return None

# 主逻辑
base_dir = Path(r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method")
df_assets = pd.read_excel(base_dir / "asset_components.xlsx")
df_time_range = pd.read_excel(base_dir / "time_range.xlsx")

# 处理第1个instance
row = df_time_range.iloc[0]
instance_num = row['Instance #']
asset_id_str = row['Asset #']
start_time = row['Start Time']

asset_num = int(asset_id_str.split('_')[1])
start_date_obj = pd.to_datetime(start_time)
query_start = start_date_obj - timedelta(days=365)
query_end = start_date_obj

start_date_str = query_start.strftime('%Y%m%d')
end_date_str = query_end.strftime('%Y%m%d')

print(f"Instance {instance_num}: Asset {asset_id_str}")
print(f"Time range: {start_date_str} - {end_date_str}")

asset_row = df_assets[df_assets['Asset #'] == asset_num]
components_str = asset_row.iloc[0]['Components']
tickers = [t.strip() for t in components_str.split(',')]

print(f"Processing {len(tickers)} tickers...\n")

all_news = []
success_count = 0
failed_tickers = []

for i, ticker in enumerate(tickers, 1):
    print(f"[{i}/{len(tickers)}] Processing {ticker}...", end=" ")
    
    company_name, status = get_company_name(ticker)
    
    if status == 'failed' or company_name is None:
        print(f"SKIP (Failed to get company name)")
        failed_tickers.append(ticker)
        continue
    
    print(f"{company_name} ({status})")
    
    df_news = fetch_gdelt_news(company_name, start_date_str, end_date_str, max_records=250)
    
    if df_news is not None and not df_news.empty:
        df_news['ticker'] = ticker
        all_news.append(df_news)
        print(f"    -> Found {len(df_news)} articles")
        success_count += 1
    else:
        print(f"    -> No articles found")
        failed_tickers.append(ticker)
    
    time.sleep(1)

print(f"\n{'='*80}")
print(f"Summary: {success_count}/{len(tickers)} successful")
print(f"Failed tickers: {failed_tickers}")

if all_news:
    df_combined = pd.concat(all_news, ignore_index=True)
    output_file = base_dir / "news" / f"instance_{instance_num}_news.csv"
    output_file.parent.mkdir(exist_ok=True)
    df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(df_combined)} total articles to {output_file}")
    print(f"Columns: {df_combined.columns.tolist()}")
else:
    print("\nNo news collected!")
