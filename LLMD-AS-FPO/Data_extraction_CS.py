import akshare as ak
from datetime import datetime
import math
import time
import pandas as pd
import numpy as np
import winsound
from typing import Union
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start', type=int)
args = parser.parse_args()

# 读文件，只需要这些文件

df_tmp = pd.read_csv("CR_asset_22-38.csv")
dt_tmp = pd.read_csv("new_asset_components.csv")
instances = df_tmp["instance"]

# 一个用于安全提取数据的函数
def safe_get(df, row_idx, col_key) -> Union[float, None]:
    if col_key in df.columns:
        return df.loc[row_idx, col_key]
    else:
        return None

# 读取instance数字，目前跑到180，第一个数是目前跑到的，第二个是想跑到几
# 建议3-5个一组跑，少了效率低，多了跑不动（也是服了）
# output窗口会显示目前跑到哪个股票了
# 如果超过4分钟output卡在同一个股票上，说明网炸了，停止运行并考虑重启kernel
for instance in instances[args.start:args.start+5]: 
    # 175说明上一组最后一个instance是175，180是说跑到180
    # 同理，下一组应该是[180,185]
    local_df = df_tmp[df_tmp["instance"] == instance]
    Inx_df = local_df.index[0]
    asset = local_df["Asset"][Inx_df]

    local_dt = dt_tmp[dt_tmp["Asset"] == asset]
    Inx_dt = local_dt.index[0]

    code = str(local_dt.loc[Inx_dt,"Components"])

    start_date = str(local_df["start"][Inx_df])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(local_df["end"][Inx_df])
    period = int((end.year - start.year) * 12 + (end.month - start.month))
    start_date = start_date.replace("-","")
    # print(start_date)
    # print(period)

    ans =[]
    word = ""
    for chars in code:
        if chars == ',' or chars == ' ':
            if word != " " and word != "":
                ans.append(word)
            word = ""
        else:
            word += chars
    if word != " " and word != "":
        ans.append(word)

    # print(ans)

    cl = ans
    org = pd.DataFrame()
    df = pd.DataFrame(index=[
        "P/E","P/B","P/S","P/CF","EV/EBITDA","Gross Margin",
        "Net Margin","ROA","ROE","FCF","FCF per Share","SPS",
        "PVS","RAPS","EVC","VEC","PLF","IQS","Period"
        ],columns=cl)

    # 读取计算写入，理论上是没问题的
    # 注：如果出现JSON Decode报错，不是代码的问题，是akshare的服务器炸了，打开下面这个链接，
    # （不用往后看，这一行就这个网址）https://vip.stock.finance.sina.com.cn/corp/go.php/vFD_FinanceSummary/stockid/600004.phtml
    # 刷新，直到表格出现即可继续运行
    for i in cl:
        j = i[2:]
        org = ak.stock_financial_abstract(symbol=j)
        print(j, end=', ') # 显示目前跑到哪个stock
        price_data = ak.stock_zh_a_daily(symbol=i,start_date=start_date)
        price = price_data["close"].values[0] if not price_data.empty else None

        pe = price / safe_get(org, 8, start_date) if price and safe_get(org, 8, start_date) else None
        pb = price / safe_get(org, 9, start_date) if price and safe_get(org, 9, start_date) else None
        fcf = safe_get(org, 7, start_date)
        fcf_per_share = safe_get(org, 10, start_date) if safe_get(org, 10, start_date) else None
        pcf = price / fcf_per_share if fcf_per_share != 0 and fcf_per_share and price else None
        roe = safe_get(org, 11, start_date) if safe_get(org, 11, start_date) else None
        roa = safe_get(org, 12, start_date) if safe_get(org, 12, start_date) else None
        gross_margin = safe_get(org, 13, start_date) if safe_get(org, 13, start_date) else None
        net_margin = safe_get(org, 14, start_date) if safe_get(org, 14, start_date) else None
        sps = safe_get(org, 31, start_date) if safe_get(org, 31, start_date) else None
        ps = price / sps if sps != 0 and sps and price else None
        ev_ebitda = price / safe_get(org, 33, start_date) if price and safe_get(org, 33, start_date) else None

        pvs = roe / pe if roe and pe else None
        raps = roe / pe / 2 if roe and pe else None
        evc = 1 / roa / ev_ebitda / pcf if roa and ev_ebitda and pcf else None
        vec = (pe + roe + fcf_per_share)/ 3 if pe and roe and fcf_per_share else None
        plf = roe * gross_margin / pe if roe and gross_margin and pe else None
        iqs = roe * math.log(sps, 10) / pe / pb if roe and sps and pe and pb else None


        df.loc["P/E",i] = pe
        df.loc["P/B",i] = pb
        df.loc["P/S",i] = ps
        df.loc["P/CF",i] = pcf
        df.loc["EV/EBITDA",i] = ev_ebitda
        df.loc["Gross Margin",i] = gross_margin
        df.loc["Net Margin",i] = net_margin
        df.loc["ROA",i] = roa
        df.loc["ROE",i] = roe
        df.loc["FCF",i] = fcf
        df.loc["FCF per Share",i] = fcf_per_share
        df.loc["SPS",i] = sps
        df.loc["PVS",i] = pvs
        df.loc["RAPS",i] = raps
        df.loc["EVC",i] = evc
        df.loc["VEC",i] = vec
        df.loc["PLF",i] = plf
        df.loc["IQS",i] = iqs
        df.loc["Period",i] = period

        # 防止拒绝访问
        time.sleep(2)

    # 写到一个csv中，需要新建一个instances_CS文件夹
    df.to_csv("instances_CS\\instance{}.csv".format(instance), encoding='utf-8')
# 提示跑完了
winsound.Beep(1000,500)