import numpy as np
import pandas as pd
from src.formula_engine import compute_all


def synthetic(common=False, seed=42):
    rng=np.random.default_rng(seed); dates=pd.date_range("2020-01-01",periods=180,freq="B"); rows=[]
    base=rng.normal(0,0.01,len(dates))
    for j in range(8):
        r=base+rng.normal(0,0.001,len(dates)) if common else rng.normal(0,0.01,len(dates))
        p=100*np.exp(np.cumsum(r)); v=rng.lognormal(14,0.3,len(dates))
        for d,px,vol in zip(dates,p,v): rows.append((d,f"A{j}",px,px*1.01,px*.99,px,vol,px))
    return pd.DataFrame(rows,columns=["date","company","open","high","low","close","volume","div_adj_close"])


def test_determinism_and_asset_permutation():
    d=synthetic(); a=compute_all(d); b=compute_all(d.sample(frac=1,random_state=7))
    assert a.keys()==b.keys()
    assert all(np.isclose(a[k],b[k]) for k in a)


def test_all_finite():
    assert all(np.isfinite(x) for x in compute_all(synthetic()).values())


def test_common_factor_reduces_effective_rank_and_increases_downside_corr():
    independent=compute_all(synthetic(False)); common=compute_all(synthetic(True))
    assert common["effective_rank"] < independent["effective_rank"]
    assert common["downside_correlation"] > independent["downside_correlation"]
