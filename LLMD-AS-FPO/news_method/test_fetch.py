"""
测试脚本：只处理前2个instance来验证功能
"""

import requests
import pandas as pd
import yfinance as yf
import akshare as ak
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
from urllib.parse import quote


# 中国知名公司英文名映射字典
CHINESE_COMPANY_ENGLISH_NAMES = {
    "600519": "Kweichow Moutai",  # 贵州茅台
    "000333": "Midea Group",  # 美的集团
    "300750": "Contemporary Amperex Technology",  # 宁德时代 (CATL)
    "600030": "CITIC Securities",  # 中信证券
    "600036": "China Merchants Bank",  # 招商银行
    "600900": "China Yangtze Power",  # 长江电力
    "601166": "Industrial Bank",  # 兴业银行
    "601318": "Ping An Insurance",  # 中国平安
    "601899": "Zijin Mining Group",  # 紫金矿业
    "000858": "Wuliangye Yibin",  # 五粮液
}


def is_chinese_stock(ticker):
    """判断是否为中国股票（6位数字）"""
    return ticker.isdigit() and len(ticker) == 6


def get_company_name(ticker):
    """获取股票对应的公司名称"""
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
    """从GDELT获取新闻，带重试机制"""
    # URL编码关键词，处理空格等特殊字符
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
                print(f"  [Retry {attempt + 1}/{max_retries}] Failed for {keyword}: {e}")
                time.sleep(1)
            else:
                print(f"  [Error] All retries failed for {keyword}: {e}")
                return None
    
    return None


# 测试主逻辑
base_dir = Path(r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method")
asset_file = base_dir / "asset_components.xlsx"
time_range_file = base_dir / "time_range.xlsx"

print("Loading data files...")
df_assets = pd.read_excel(asset_file)
df_time_range = pd.read_excel(time_range_file)

# 只测试第1个instance
print("\n" + "="*80)
print("Testing with Instance 1")
print("="*80)

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

print(f"\nInstance {instance_num}: Asset {asset_id_str}, Time: {start_date_str} - {end_date_str}")

# 获取股票列表
asset_row = df_assets[df_assets['Asset #'] == asset_num]
components_str = asset_row.iloc[0]['Components']
tickers = [t.strip() for t in components_str.split(',')]

print(f"Found {len(tickers)} tickers: {tickers[:5]}...")

# 测试前3个股票
print("\nTesting first 3 tickers:")
for ticker in tickers[:3]:
    print(f"\n--- Testing {ticker} ---")
    company_name, status = get_company_name(ticker)
    print(f"Company Name: {company_name}, Status: {status}")
    
    if company_name:
        df_news = fetch_gdelt_news(company_name, start_date_str, end_date_str, max_records=10)
        if df_news is not None:
            print(f"News found: {len(df_news)} articles")
        else:
            print("No news found")
    time.sleep(1)

print("\n" + "="*80)
print("Test completed!")
