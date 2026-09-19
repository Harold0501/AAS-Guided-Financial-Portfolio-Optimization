
import sys
sys.path.insert(0, r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method")

from fetch_instance_news import process_instance
import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method")
df_assets = pd.read_excel(base_dir / "asset_components.xlsx")
df_time = pd.read_excel(base_dir / "time_range.xlsx")

# 处理instance 45
row = df_time[df_time['Instance #'] == 45].iloc[0]
df_news = process_instance(row, df_assets)

if df_news is not None:
    print(f"\n成功获取 {len(df_news)} 条新闻")
    print(f"Columns: {df_news.columns.tolist()}")
    print("\n前3条新闻:")
    print(df_news[['ticker', 'keyword', 'Date', 'Title']].head(3))
    
    # 保存测试结果
    output_file = base_dir / "news" / f"instance_{row['Instance #']}_news.csv"
    output_file.parent.mkdir(exist_ok=True)
    df_news.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n已保存到: {output_file}")
else:
    print("\n未获取到任何新闻")
