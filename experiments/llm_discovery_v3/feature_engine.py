from __future__ import annotations
import numpy as np
import pandas as pd

EPS=1e-12

def _safe(x):
    x=float(x); return x if np.isfinite(x) else 0.0
def _mat(df,col):
    return df.pivot_table(index='date',columns='company',values=col,aggfunc='last').sort_index().sort_index(axis=1)
def _corr(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    return _safe(np.corrcoef(a[m],b[m])[0,1]) if m.sum()>=5 and np.std(a[m])>EPS and np.std(b[m])>EPS else 0.0
def _skew(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if len(x)<4:return 0.
    z=(x-x.mean())/(x.std()+EPS);return _safe(np.mean(z**3))
def _kurt(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if len(x)<5:return 0.
    z=(x-x.mean())/(x.std()+EPS);return _safe(np.mean(z**4)-3)

def compute(df):
    p=_mat(df,'div_adj_close'); v=_mat(df,'volume'); r=p.pct_change(fill_method=None)
    lv=np.log1p(v).diff(); market=r.mean(axis=1); rv=r.std(); miss=p.isna().mean()
    ar=r.to_numpy().ravel(); av=lv.to_numpy().ravel()
    # B
    b={}
    b['B_volume_return_correlation']=_corr(ar,av)
    b['B_cross_sectional_return_skewness']=_safe(r.apply(lambda x:_skew(x.values),axis=1).median())
    neg=(market<0).astype(float); b['B_negative_return_persistence']=_corr(neg.iloc[:-1].values,neg.iloc[1:].values)
    b['B_volume_volatility_ratio']=_safe(np.nanmedian(lv.std()/(rv+EPS)))
    hi=_mat(df,'high');lo=_mat(df,'low');cl=_mat(df,'close'); b['B_daily_price_range']=_safe(np.nanmedian(((hi-lo)/(cl.abs()+EPS)).values))
    hist=np.histogram(ar[np.isfinite(ar)],bins=20)[0].astype(float); prob=hist/(hist.sum()+EPS); b['B_return_entropy']=_safe(-(prob[prob>0]*np.log(prob[prob>0])).sum()/np.log(20))
    q=r.quantile(.1); tails=r.lt(q,axis=1); b['B_tail_return_co_occurrence']=_safe((tails.sum(axis=1)>=max(2,int(np.ceil(.25*r.shape[1])))).mean())
    b['B_volatility_concentration']=_safe(rv.max()/(rv.sum()+EPS))
    b['B_return_serial_correlation']=_safe(np.nanmedian([_corr(r[c].iloc[:-1].values,r[c].iloc[1:].values) for c in r]))
    b['B_missingness_volatility_correlation']=_corr(miss.values,rv.values)
    # C: complementarity-aware higher-order/dynamic structure
    c={}; windows=[]
    for end in range(40,len(r)+1,20): windows.append(r.iloc[max(0,end-60):end].corr())
    if len(windows)>=2:
        vals=[np.nanmean(np.abs((windows[i]-windows[i-1]).to_numpy())) for i in range(1,len(windows))]
        c['C_correlation_stability']=_safe(1/(1+np.nanmean(vals)))
    else:c['C_correlation_stability']=0.
    c['C_cross_sectional_return_kurtosis']=_safe(r.apply(lambda x:_kurt(x.values),axis=1).median())
    m=np.isfinite(ar)&np.isfinite(av); zr=(ar[m]-np.mean(ar[m]))/(np.std(ar[m])+EPS);zv=(av[m]-np.mean(av[m]))/(np.std(av[m])+EPS)
    c['C_return_volume_joint_skewness']=_safe(np.mean(zr*zv*zv)) if m.sum() else 0.
    c['C_return_volume_joint_kurtosis']=_safe(np.mean((zr*zv)**2)-1) if m.sum() else 0.
    z=r.div(rv+EPS); c['C_extreme_return_participation']=_safe((z.abs()>2).mean(axis=1).median())
    w=v.div(v.sum(axis=1),axis=0); wd=((r.sub(r.mean(axis=1),axis=0)**2*w).sum(axis=1))**.5;c['C_volume_adjusted_dispersion']=_safe(wd.median())
    c['C_skew_volatility_ratio']=_safe(np.nanmedian([abs(_skew(r[x].values))/(r[x].std()+EPS) for x in r]))
    c['C_return_mad_ratio']=_safe(np.nanmedian(r.apply(lambda x:np.nanmedian(np.abs(x-np.nanmedian(x)))/(np.nanstd(x)+EPS))))
    cors=[]
    for i in range(20,len(r)+1,20): cors.append(np.nanmean(r.iloc[i-20:i].corr().where(~np.eye(r.shape[1],dtype=bool))))
    c['C_correlation_duration']=_corr(np.asarray(cors[:-1]),np.asarray(cors[1:])) if len(cors)>2 else 0.
    up=r.where(market>=0).mean();down=r.where(market<0).mean();c['C_cross_asset_return_asymmetry']=_safe(np.nanstd(up-down))
    return {**b,**c}
