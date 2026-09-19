import os
import glob
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import argparse
import tqdm

from src.mSSRM_PGA import mSSRM_PGA
from src.utils import preprocess_backtest_data
from src.backtest import aggregate_weights, backtest, backtest_baseline

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--asset', default='asset_1', type=str)
args = parser.parse_args()

asset = args.asset

train_start = '2015-01-01'
train_end = '2025-01-01'
trading_cost = 0.004
lagging = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_series = [d.strftime('%Y-%m-%d') for d in pd.date_range(start=train_start, end=train_end, freq='Y')]

ports = ['maxsortino', 'onehot']
models = ['mamba', 'trf', 'lstm']
optimizers = ['sophia']

directory = Path('new_result')
data_dir = Path('new_data')
out_csv = Path('result_asset.csv')

def max_drawdown(cum_returns: pd.Series) -> float:
    arr = cum_returns.to_numpy()
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    drawdowns = arr / running_max - 1.0
    return float(np.min(drawdowns))


def load_pickles(prefixes: List[str], folder: Path) -> Dict[str, Any]:
    loaded = {}
    for p in prefixes:
        path = folder / (p + '.pkl')
        if path.exists():
            with path.open('rb') as f:
                loaded[p] = pickle.load(f)
    return loaded


def list_expected_weight_names(asset: str, ports: List[str], models: List[str], optimizers: List[str]) -> Tuple[List[str], List[str]]:
    single = [f"{asset}_weights_{port.lower()}" for port in ports]
    allw = [f"{asset}_{port.lower()}_{model.lower()}_{optimizer}_all_weights" for port in ports for model in models for optimizer in optimizers]
    return single, allw

backtest_yearmons, backtest_data = {}, {}
yearmons, data = preprocess_backtest_data(str(data_dir / f'{asset}.csv'))
backtest_yearmons[asset] = yearmons
backtest_data[asset] = data

single_prefixes, all_prefixes = list_expected_weight_names(asset, ports, models, optimizers)
weights_cache = load_pickles(single_prefixes, directory)
all_weights_cache = load_pickles(all_prefixes, directory)

baseline_backtest: Dict[str, pd.DataFrame] = {}

for start_idx in tqdm.tqdm(range(len(time_series) - 1), desc='start idx'):
    backtest_start = time_series[start_idx]

    baseline_df = backtest_baseline(yearmons, data, backtest_start, trading_cost, lagging)

    cols = data['log_tr'].columns
    bb_weights = pd.DataFrame(index=yearmons, columns=cols, dtype=float)
    start_pos = yearmons.get_loc(backtest_start) if backtest_start in yearmons else 0
    relevant_yearmons = yearmons[start_pos:]

    for ym in relevant_yearmons:
        past = data['log_tr'].loc[data.index < ym].iloc[-252:].fillna(0)
        if past.shape[0] == 0:
            bb_weights.loc[ym] = 1.0 / len(cols)
            continue
        matR = torch.tensor(past.values, dtype=torch.float32, device=device)
        try:
            w = mSSRM_PGA(m=len(cols), iternum=1e4, tol=1e-5, matR=matR, vecmu=matR.mean(0))
            w = w.detach().cpu().numpy().astype(float)
            bb_weights.loc[ym] = w
        except Exception as e:
            bb_weights.loc[ym] = 1.0 / len(cols)
            print(f"[WARN] mSSRM_PGA failed at yearmon {ym}: {e}")

    baseline_df = baseline_df.copy()
    baseline_df['mSSRM_PGA'] = backtest(yearmons, data, bb_weights, backtest_start, trading_cost, lagging)
    baseline_backtest[asset] = baseline_df

    for end_idx in range(start_idx + 1, len(time_series)):
        backtest_end = time_series[end_idx]
        print(f"\nRange: {backtest_start} to {backtest_end}")

        base_slice = baseline_backtest[asset].loc[backtest_start:backtest_end]

        cumret_df = base_slice.apply(np.exp).cumprod()
        final_cum = cumret_df.iloc[-1]
        EW_cr = final_cum.get('equal_weight', np.nan)
        MCW_cr = final_cum.get('value_weight', np.nan)
        MVO_cr = final_cum.get('mSSRM_PGA', np.nan)

        years = (pd.to_datetime(backtest_end) - pd.to_datetime(backtest_start)).days / 365.25
        EW_ar = (EW_cr ** (1 / years) - 1) if not pd.isna(EW_cr) else np.nan
        MCW_ar = (MCW_cr ** (1 / years) - 1) if not pd.isna(MCW_cr) else np.nan
        MVO_ar = (MVO_cr ** (1 / years) - 1) if not pd.isna(MVO_cr) else np.nan

        mean_vec = base_slice.mean()
        std_vec = base_slice.std()
        neg_std = base_slice.clip(upper=0).std()

        EW_shr = (mean_vec.get('equal_weight', np.nan) / std_vec.get('equal_weight', np.nan)) * np.sqrt(252) if std_vec.get('equal_weight', np.nan) != 0 else np.nan
        MCW_shr = (mean_vec.get('value_weight', np.nan) / std_vec.get('value_weight', np.nan)) * np.sqrt(252) if std_vec.get('value_weight', np.nan) != 0 else np.nan
        MVO_shr = (mean_vec.get('mSSRM_PGA', np.nan) / std_vec.get('mSSRM_PGA', np.nan)) * np.sqrt(252) if std_vec.get('mSSRM_PGA', np.nan) != 0 else np.nan

        EW_sor = (mean_vec.get('equal_weight', np.nan) / neg_std.get('equal_weight', np.nan)) * np.sqrt(252) if neg_std.get('equal_weight', np.nan) != 0 else np.nan
        MCW_sor = (mean_vec.get('value_weight', np.nan) / neg_std.get('value_weight', np.nan)) * np.sqrt(252) if neg_std.get('value_weight', np.nan) != 0 else np.nan
        MVO_sor = (mean_vec.get('mSSRM_PGA', np.nan) / neg_std.get('mSSRM_PGA', np.nan)) * np.sqrt(252) if neg_std.get('mSSRM_PGA', np.nan) != 0 else np.nan

        EW_mdd = max_drawdown(cumret_df['equal_weight']) if 'equal_weight' in cumret_df else np.nan
        MCW_mdd = max_drawdown(cumret_df['value_weight']) if 'value_weight' in cumret_df else np.nan
        MVO_mdd = max_drawdown(cumret_df['mSSRM_PGA']) if 'mSSRM_PGA' in cumret_df else np.nan

        bt_results = {}
        for port in ports:
            for model in models:
                for optimizer in optimizers:
                    obj_prefix = f"{asset}_{port.lower()}_{model.lower()}_{optimizer}_all_weights"
                    target_name = f"{asset}_{port.lower()}_{model.lower()}_{optimizer}_backtest"

                    if obj_prefix not in all_weights_cache:
                        print(f"[WARN] weights object missing: {obj_prefix} (skipping)")
                        continue

                    weight_df = aggregate_weights(backtest_yearmons[asset], backtest_data[asset], all_weights_cache[obj_prefix], backtest_start, backtest_end)
                    weight_df.iloc[:3] = 1.0 / weight_df.shape[1]
                    try:
                        bt_results[target_name] = backtest(backtest_yearmons[asset], backtest_data[asset], weight_df, backtest_start, trading_cost, lagging)
                    except Exception as e:
                        print(f"[ERROR] backtest failed for {target_name} in range {backtest_start}-{backtest_end}: {e}")

        cr_list, ar_list, shr_list, sor_list, mdd_list = [], [], [], [], []
        for model in models:
            for port in ports:
                key = f"{asset}_{port.lower()}_{model.lower()}_{optimizers[0]}_backtest"
                df_bt = bt_results.get(key)
                if df_bt is None:
                    cr_list.append(np.nan); ar_list.append(np.nan); shr_list.append(np.nan); sor_list.append(np.nan); mdd_list.append(np.nan)
                    continue
                obj_slice = df_bt.loc[backtest_start:backtest_end]
                cum_df = obj_slice.apply(np.exp).cumprod()
                final_val = cum_df.iloc[-1, 0]
                cr_list.append(final_val)
                ar_list.append(final_val ** (1 / years) - 1 if years > 0 else np.nan)
                mean_r = obj_slice.mean().iloc[0]
                std_r = obj_slice.std().iloc[0]
                neg_std_r = obj_slice.clip(upper=0).std().iloc[0]
                shr_list.append((mean_r / std_r) * np.sqrt(252) if std_r != 0 else np.nan)
                sor_list.append((mean_r / neg_std_r) * np.sqrt(252) if neg_std_r != 0 else np.nan)
                mdd_list.append(max_drawdown(cum_df.iloc[:, 0]))

        col_top = ['Asset', 'start', 'end'] + sum([[metric] * 9 for metric in ['AR', 'CR', 'SHR', 'SOR', 'MDD']], [])
        col_bottom = ['Asset', 'start', 'end'] + (['EW', 'MCW', 'MVO', 'MS-M', 'OH-M', 'MS-T', 'OH-T', 'MS-L', 'OH-L'] * 5)
        columns = pd.MultiIndex.from_arrays([col_top, col_bottom])

        row_data = [asset, backtest_start, backtest_end] + \
                   [EW_ar, MCW_ar, MVO_ar] + ar_list + \
                   [EW_cr, MCW_cr, MVO_cr] + cr_list + \
                   [EW_shr, MCW_shr, MVO_shr] + shr_list + \
                   [EW_sor, MCW_sor, MVO_sor] + sor_list + \
                   [EW_mdd, MCW_mdd, MVO_mdd] + mdd_list

        result_df = pd.DataFrame([row_data], columns=columns)
        result_df.to_csv(out_csv, mode='a', header=not out_csv.exists(), index=False)

print("All done.")
