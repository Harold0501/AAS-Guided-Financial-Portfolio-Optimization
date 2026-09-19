import pandas as pd
from urllib.parse import quote

# 测试并查看GDELT返回的列
def test_gdelt_columns(keyword, start_date, end_date):
    encoded_keyword = quote(keyword)
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={encoded_keyword}"
        f"&mode=artlist"
        f"&maxrecords=5"
        f"&format=csv"
        f"&startdatetime={start_date}000000"
        f"&enddatetime={end_date}235959"
    )
    print(f"Testing: {keyword}")
    try:
        df = pd.read_csv(url)
        print(f"Found: {len(df)} articles")
        print(f"Columns: {df.columns.tolist()}")
        if len(df) > 0:
            print("\nFirst row:")
            print(df.iloc[0])
    except Exception as e:
        print(f"Error: {e}")

print("="*80)
test_gdelt_columns("Kweichow Moutai", "20201231", "20211231")
