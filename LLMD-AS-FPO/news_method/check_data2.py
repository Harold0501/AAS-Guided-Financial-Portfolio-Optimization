import pandas as pd

# 读取asset_components
df_assets = pd.read_excel(r'd:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method\asset_components.xlsx')

# 读取time_range
df_time = pd.read_excel(r'd:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method\time_range.xlsx')

print('Time range中的Asset格式示例:')
print(df_time['Asset #'].head(20).tolist())

print('\n\nAsset components中的Asset #范围:')
print(f'Min: {df_assets["Asset #"].min()}, Max: {df_assets["Asset #"].max()}')

print('\n\n查看asset_22对应的信息:')
asset_22 = df_assets[df_assets['Asset #'] == 22]
print(asset_22)

print('\n\n查看是否有中国股票 (CSI 300):')
csi_assets = df_assets[df_assets['Origin'].str.contains('CSI_300', na=False)]
print(csi_assets[['Asset #', 'Description', 'Origin']].head())
if len(csi_assets) > 0:
    print('\n中国股票Components示例:')
    print(csi_assets.iloc[0]['Components'][:200])
