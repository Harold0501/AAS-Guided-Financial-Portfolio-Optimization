import pandas as pd
from datetime import timedelta
from urllib.parse import quote

# 简单测试GDELT API
def test_gdelt(keyword, start_date, end_date):
    encoded_keyword = quote(keyword)
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={encoded_keyword}"
        f"&mode=artlist"
        f"&maxrecords=10"
        f"&format=csv"
        f"&startdatetime={start_date}000000"
        f"&enddatetime={end_date}235959"
    )
    print(f"Testing: {keyword}")
    print(f"URL: {url}")
    try:
        df = pd.read_csv(url)
        print(f"Result: {len(df)} articles found\n")
        if len(df) > 0:
            print(df[['url', 'title']].head(3))
    except Exception as e:
        print(f"Error: {e}\n")

# 测试美股 - Apple（2014-2015年应该有很多新闻）
print("="*80)
test_gdelt("Apple", "20141231", "20151231")

print("\n" + "="*80)
# 测试中国公司英文名
test_gdelt("Midea Group", "20201231", "20211231")  # 使用较新的日期

print("\n" + "="*80)
# 测试茅台
test_gdelt("Kweichow Moutai", "20201231", "20211231")
