import pandas as pd

# 读取asset_components
df_assets = pd.read_excel(r'd:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method\asset_components.xlsx')
print('Asset Components Info:')
print(df_assets.head(3))
print('\nOrigins:', df_assets['Origin'].unique())
print('\n检查是否有中国股票:')
cn_assets = df_assets[df_assets['Origin'] != 'S&P 500']
print(cn_assets.head(3))
if len(cn_assets) > 0:
    print('\n中国股票示例Components:')
    print(cn_assets.iloc[0]['Components'])

# 读取time_range
df_time = pd.read_excel(r'd:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method\time_range.xlsx')
print('\n\nTime Range Info:')
print(df_time.head(3))
print('\nAsset numbers in time_range:', df_time['Asset #'].unique()[:10])
