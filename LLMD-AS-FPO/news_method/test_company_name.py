import yfinance as yf
import akshare as ak

# 测试美股
print("测试美股 - AAPL:")
ticker = yf.Ticker("AAPL")
print(f"Long Name: {ticker.info.get('longName', 'N/A')}")
print(f"Short Name: {ticker.info.get('shortName', 'N/A')}")

# 测试中国股票
print("\n测试中国股票 - 600519 (贵州茅台):")
try:
    # akshare获取股票信息
    stock_info = ak.stock_individual_info_em(symbol="600519")
    print(stock_info)
except Exception as e:
    print(f"Error: {e}")

print("\n测试中国股票 - 000333:")
try:
    stock_info = ak.stock_individual_info_em(symbol="000333")
    print(stock_info)
except Exception as e:
    print(f"Error: {e}")
