import os
import time
import tqdm
import pandas as pd
import yfinance as yf
import akshare as ak

input_df = pd.read_excel("asset_components.xlsx").set_index("Asset #")

for asset_id, row in input_df.iterrows():
    tickers = [t.replace('.', '-') for t in row["Components"].split(', ')]
    is_cn = any(c.isdigit() for c in tickers[0])
    records = []
    for ticker in tqdm.tqdm(tickers, desc=f"Asset_{asset_id}"):
        try:
            if is_cn:
                ticker = 'sh' + ticker if ticker.startswith('6') else 'sz' + ticker
                df = ak.stock_zh_a_daily(symbol=ticker, start_date='2015-01-01', end_date='2025-01-01', adjust="hfq")
                df = df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                                        'low': 'low', 'close': 'close', 'volume': 'volume'})
                df['div_adj_close'] = df['close']
                df['company'] = ticker
                df = df[['date', 'company', 'open', 'high', 'low', 'close', 'volume', 'div_adj_close']]
                time.sleep(0.5)
            else:
                df = yf.download(ticker, start='2020-01-01', end='2025-01-01', group_by="ticker", auto_adjust=False, progress=False)
                df = df.reset_index()
                df['company'] = ticker
                df.columns = [b if b else a for a, b in df.columns]
                df = df[['Date', 'company', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
                df.columns = ['date', 'company', 'open', 'high', 'low', 'close', 'volume', 'div_adj_close']
            records.append(df)
        except Exception as e:
            print(f"ERROR: Dowloading {ticker} failed: {e}")
            continue

    if records:
        final_df = pd.concat(records, ignore_index=True)
        final_df = final_df.sort_values(['company', 'date']).dropna()

        folder = "new data" if is_cn else "data"
        os.makedirs(folder, exist_ok=True)
        final_df.to_csv(f"{folder}/asset_{asset_id}.csv", index=False)
        print(f"SUCCESS: Asset_{asset_id} saved successfully, {len(final_df)} lines of data in total")
    else:
        print(f"WARNING: No available data in Asset_{asset_id}")