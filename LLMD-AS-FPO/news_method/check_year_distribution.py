import pandas as pd

df_time = pd.read_excel(r"d:\郑皓元\life at DKU\July 7 - August 31, 2025\LLMD-AS-FPO\news_method\time_range.xlsx")

df_time['Start Year'] = pd.to_datetime(df_time['Start Time']).dt.year
year_counts = df_time['Start Year'].value_counts().sort_index()

print("Instance分布按开始年份:")
print(year_counts)

print(f"\n总Instance数: {len(df_time)}")
print(f"2017年及以后的Instance数: {len(df_time[df_time['Start Year'] >= 2017])}")
print(f"2020年及以后的Instance数: {len(df_time[df_time['Start Year'] >= 2020])}")

# GDELT在2015年后数据更完整
print(f"\n建议: GDELT在2015年后的数据覆盖更好")
print(f"但脚本会处理所有instance，早期数据可能新闻较少")
