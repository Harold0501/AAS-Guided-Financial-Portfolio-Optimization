"""
批量获取instance对应股票的GDELT新闻
遍历time_range.xlsx中的每个instance，获取其asset对应的所有股票的新闻数据
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
    """
    获取股票对应的公司名称
    - 美股：使用yfinance获取longName
    - 中国股票：使用映射字典或akshare获取中文名
    
    返回: (company_name, status)
        - company_name: 公司名称
        - status: 'success', 'fallback', 'failed'
    """
    try:
        if is_chinese_stock(ticker):
            # 中国股票：优先使用英文映射
            if ticker in CHINESE_COMPANY_ENGLISH_NAMES:
                return CHINESE_COMPANY_ENGLISH_NAMES[ticker], 'success'
            
            # 使用akshare获取中文名
            try:
                stock_info = ak.stock_individual_info_em(symbol=ticker)
                cn_name = stock_info[stock_info['item'] == '股票简称']['value'].values[0]
                # 返回中文名作为fallback
                return cn_name, 'fallback'
            except Exception as e:
                print(f"  [Warning] Failed to get Chinese stock info for {ticker}: {e}")
                return None, 'failed'
        else:
            # 美股：使用yfinance
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
    """
    从GDELT获取新闻，带重试机制
    
    参数:
        keyword: 搜索关键词（公司名）
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        max_records: 最大记录数
        max_retries: 最大重试次数
    
    返回: DataFrame或None
    """
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
            df["ticker"] = ""  # 添加ticker列，稍后填充
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [Retry {attempt + 1}/{max_retries}] Failed for {keyword}: {e}")
                time.sleep(1)  # 重试前等待1秒
            else:
                print(f"  [Error] All retries failed for {keyword}: {e}")
                return None
    
    return None


def process_instance(instance_row, df_assets):
    """
    处理单个instance，获取其所有股票的新闻
    
    参数:
        instance_row: time_range.xlsx中的一行数据
        df_assets: asset_components.xlsx的DataFrame
    
    返回: DataFrame（合并后的新闻）或None
    """
    instance_num = instance_row['Instance #']
    asset_id_str = instance_row['Asset #']  # 如 "asset_22"
    start_time = instance_row['Start Time']
    
    # 提取asset编号
    asset_num = int(asset_id_str.split('_')[1])
    
    # 计算时间范围：开始时间前一年到开始时间
    start_date_obj = pd.to_datetime(start_time)
    query_start = start_date_obj - timedelta(days=365)
    query_end = start_date_obj
    
    start_date_str = query_start.strftime('%Y%m%d')
    end_date_str = query_end.strftime('%Y%m%d')
    
    print(f"\n[Instance {instance_num}] Asset: {asset_id_str}, Time: {start_date_str} - {end_date_str}")
    
    # 获取asset对应的股票列表
    asset_row = df_assets[df_assets['Asset #'] == asset_num]
    if asset_row.empty:
        print(f"  [Error] Asset {asset_num} not found in asset_components.xlsx")
        return None
    
    components_str = asset_row.iloc[0]['Components']
    tickers = [t.strip() for t in components_str.split(',')]
    
    print(f"  Found {len(tickers)} tickers in asset {asset_num}")
    
    # 收集所有股票的新闻
    all_news = []
    success_count = 0
    failed_tickers = []
    
    for ticker in tqdm(tickers, desc=f"  Fetching news for instance {instance_num}", leave=False):
        # 获取公司名称
        company_name, status = get_company_name(ticker)
        
        if status == 'failed' or company_name is None:
            print(f"  [Skip] {ticker}: Failed to get company name")
            failed_tickers.append(ticker)
            continue
        
        if status == 'fallback':
            print(f"  [Fallback] {ticker}: Using Chinese name '{company_name}'")
        
        # 获取新闻
        df_news = fetch_gdelt_news(company_name, start_date_str, end_date_str)
        
        if df_news is not None and not df_news.empty:
            df_news['ticker'] = ticker  # 记录ticker
            all_news.append(df_news)
            success_count += 1
        else:
            failed_tickers.append(ticker)
        
        # 避免API限流
        time.sleep(1)
    
    print(f"  [Summary] Success: {success_count}/{len(tickers)}, Failed: {len(failed_tickers)}")
    if failed_tickers:
        print(f"  [Failed Tickers] {', '.join(failed_tickers)}")
    
    # 合并所有新闻
    if all_news:
        df_combined = pd.concat(all_news, ignore_index=True)
        return df_combined
    else:
        print(f"  [Warning] No news collected for instance {instance_num}")
        return None


def main():
    """主函数"""
    # 路径设置
    base_dir = Path(r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method")
    asset_file = base_dir / "asset_components.xlsx"
    time_range_file = base_dir / "time_range.xlsx"
    news_dir = base_dir / "news"
    
    # 创建news目录
    news_dir.mkdir(exist_ok=True)
    
    # 读取数据
    print("Loading data files...")
    df_assets = pd.read_excel(asset_file)
    df_time_range = pd.read_excel(time_range_file)
    
    print(f"Total instances: {len(df_time_range)}")
    print(f"Total assets: {len(df_assets)}")
    
    # 遍历每个instance
    for idx, row in df_time_range.iterrows():
        instance_num = row['Instance #']
        output_file = news_dir / f"instance_{instance_num}_news.csv"
        
        # 检查是否已存在
        if output_file.exists():
            print(f"\n[Skip] Instance {instance_num} already exists: {output_file}")
            continue
        
        # 处理instance
        df_news = process_instance(row, df_assets)
        
        # 保存结果
        if df_news is not None:
            df_news.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"  [Saved] {len(df_news)} news articles to {output_file}")
        else:
            # 即使没有新闻也创建空文件，标记为已处理
            pd.DataFrame().to_csv(output_file, index=False)
            print(f"  [Saved] Empty file (no news) to {output_file}")
    
    print("\n" + "="*80)
    print("All instances processed!")


if __name__ == "__main__":
    main()
