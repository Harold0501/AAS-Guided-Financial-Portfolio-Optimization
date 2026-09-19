import pandas as pd
from urllib.parse import quote

def quick_test(keyword, start, end):
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote(keyword)}&mode=artlist&maxrecords=10&format=csv&startdatetime={start}000000&enddatetime={end}235959"
    try:
        df = pd.read_csv(url)
        print(f"{keyword} ({start}-{end}): {len(df)} articles")
    except Exception as e:
        print(f"{keyword}: Error - {e}")

# 测试不同时间段
print("Testing 2014-2015:")
quick_test("Apple", "20141231", "20151231")
quick_test("Microsoft", "20141231", "20151231")

print("\nTesting 2023-2024:")
quick_test("Apple", "20230101", "20240101")
quick_test("Kweichow Moutai", "20230101", "20240101")

# 检查time_range中是否有更近的时间
df_time = pd.read_excel(r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method\time_range.xlsx")
print(f"\n最新的时间范围:")
print(df_time.sort_values('Start Time', ascending=False).head(5)[['Instance #', 'Asset #', 'Start Time', 'End Time']])
