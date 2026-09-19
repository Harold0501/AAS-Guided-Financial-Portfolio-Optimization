"""
最终验证测试：处理一个2023年的instance（应该能获取到新闻）
"""
import subprocess
import sys
import pandas as pd
from pathlib import Path

# 找一个2023年的instance
base_dir = Path(r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method")
df_time = pd.read_excel(base_dir / "time_range.xlsx")

# 筛选2023年的instance
df_time['Start Year'] = pd.to_datetime(df_time['Start Time']).dt.year
recent_instances = df_time[df_time['Start Year'] == 2023]

if len(recent_instances) > 0:
    test_instance = recent_instances.iloc[0]
    print(f"选择测试Instance: {test_instance['Instance #']}")
    print(f"Asset: {test_instance['Asset #']}")
    print(f"时间: {test_instance['Start Time']} - {test_instance['End Time']}")
    
    # 创建临时测试脚本（只处理这一个instance）
    test_script = f"""
import sys
sys.path.insert(0, r"{base_dir}")

from fetch_instance_news import process_instance
import pandas as pd
from pathlib import Path

base_dir = Path(r"{base_dir}")
df_assets = pd.read_excel(base_dir / "asset_components.xlsx")
df_time = pd.read_excel(base_dir / "time_range.xlsx")

# 处理instance {test_instance['Instance #']}
row = df_time[df_time['Instance #'] == {test_instance['Instance #']}].iloc[0]
df_news = process_instance(row, df_assets)

if df_news is not None:
    print(f"\\n成功获取 {{len(df_news)}} 条新闻")
    print(f"Columns: {{df_news.columns.tolist()}}")
    print("\\n前3条新闻:")
    print(df_news[['ticker', 'keyword', 'Date', 'Title']].head(3))
    
    # 保存测试结果
    output_file = base_dir / "news" / f"instance_{{row['Instance #']}}_news.csv"
    output_file.parent.mkdir(exist_ok=True)
    df_news.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\\n已保存到: {{output_file}}")
else:
    print("\\n未获取到任何新闻")
"""
    
    with open(base_dir / "run_single_test.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print(f"\n开始测试...")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, str(base_dir / "run_single_test.py")],
        capture_output=False,
        text=True
    )
    
    print("="*80)
    if result.returncode == 0:
        print("\n✓ 测试成功！")
    else:
        print(f"\n✗ 测试失败，返回码: {result.returncode}")
else:
    print("未找到2023年的instance")
