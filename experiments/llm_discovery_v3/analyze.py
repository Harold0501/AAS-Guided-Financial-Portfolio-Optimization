from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import ttest_rel
H=Path(__file__).resolve().parent
winners={'RF':{'AR':'Fin+A+C','CR':'Fin+A+B','MDD':'Fin+A','SHR':'Fin+A+C','SOR':'Fin+A+C'},'MLP':{'AR':'Fin+A+B','CR':'Fin+GPT+A+C','MDD':'Fin+A+B+C','SHR':'Fin+A+B','SOR':'Fin+A+B'}}
rows=[]
for sel,cs in winners.items():
 for crit,cfg in cs.items():
  ds=[]
  for seed in [42,142,242,342,442]:
   b=pd.read_csv(H/'results'/sel/crit/'Fin'/f'seed_{seed}.csv')
   x=pd.read_csv(H/'results'/sel/crit/cfg/f'seed_{seed}.csv')
   ds.append(x.selection.mean()-b.selection.mean())
  a=np.array(ds);t,p=ttest_rel(a,np.zeros_like(a));rows.append({'selector':sel,'criterion':crit,'winner':cfg,'mean_delta':a.mean(),'min_seed_delta':a.min(),'max_seed_delta':a.max(),'positive_seeds':int((a>0).sum()),'seed_t_p':p})
d=pd.DataFrame(rows);d['holm_p']=np.nan
order=np.argsort(d.seed_t_p.values);m=len(d)
for rank,idx in enumerate(order):d.loc[idx,'holm_p']=min(1,d.loc[idx,'seed_t_p']*(m-rank))
d.to_csv(H/'results/confirmatory_summary.csv',index=False);print(d.to_string(index=False))
