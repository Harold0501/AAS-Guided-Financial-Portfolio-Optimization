from __future__ import annotations
import argparse,json,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;CFG=json.loads((HERE/'config.json').read_text());ROOT=Path(CFG['source_root']);ALG=['EW','MCW','mSSRM','MS-M','OH-M','MS-T','OH-T','MS-L','OH-L']
def load(crit):
 s=pd.read_csv(ROOT/'RF_combined'/f'All_{crit}_LLM_combined.csv'); d=pd.read_csv(HERE/'features/discovered_features.csv');a=pd.read_csv(ROOT/CFG['formula_v2_dir']/'outputs/features/formula_v2_features.csv');m=pd.read_csv(ROOT/CFG['mapping_csv'])[['instance','Asset']]
 x=s.merge(a,on='instance').merge(d,on='instance').merge(m,on='instance');fin=s.columns[1:25].tolist()+['Period'];gpt=s.columns[25:37].tolist();A=[c for c in a if c!='instance'];B=[c for c in d if c.startswith('B_')];C=[c for c in d if c.startswith('C_')]
 cfg={}
 for prefix,base in [('Fin',fin),('Fin+GPT',fin+gpt)]:
  for mask in range(8):
   names=[];cols=list(base)
   for bit,name,z in [(1,'A',A),(2,'B',B),(4,'C',C)]:
    if mask&bit:names.append(name);cols+=z
   cfg[prefix+('+'+'+'.join(names) if names else '')]=cols
 return x,cfg
def model(sel,seed,fold):
 if sel=='RF':return RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=seed+fold,n_jobs=1)
 base=Pipeline([('scale',StandardScaler()),('mlp',MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,learning_rate_init=5e-4,max_iter=600,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,random_state=seed+fold))])
 return TransformedTargetRegressor(regressor=base,transformer=StandardScaler())
def fit(sel,X,y,t,tr,te,seed,k):
 z=model(sel,seed,k)
 with warnings.catch_warnings():warnings.simplefilter('ignore');z.fit(X.iloc[tr],y[t].iloc[tr])
 return t,te,z.predict(X.iloc[te])
def one(sel,crit,name,seed,jobs):
 out=HERE/'results'/sel/crit/name;out.mkdir(parents=True,exist_ok=True);p=out/f'seed_{seed}.csv';meta=out/f'seed_{seed}.json'
 if p.exists() and meta.exists():return json.loads(meta.read_text())
 d,cfg=load(crit);X=d[cfg[name]].astype(float);y=d[ALG].astype(float);folds=list(GroupKFold(n_splits=5).split(X,groups=d.Asset));tasks=[(a,tr,te,k) for a in ALG for k,(tr,te) in enumerate(folds)]
 res=Parallel(n_jobs=jobs)(delayed(fit)(sel,X,y,a,tr,te,seed,k) for a,tr,te,k in tasks);pred=pd.DataFrame(index=d.index,columns=ALG,dtype=float)
 for a,te,v in res:pred.iloc[te,pred.columns.get_loc(a)]=v
 chosen=pred.idxmax(axis=1);actual=y.to_numpy();idx=np.array([ALG.index(a) for a in chosen]);vbs=actual.max(axis=1);r=pd.DataFrame({'instance':d.instance,'Asset':d.Asset,'selected':chosen,'selection':actual[np.arange(len(d)),idx],'vbs':vbs,'regret':vbs-actual[np.arange(len(d)),idx]})
 tmp=p.with_suffix('.tmp');r.to_csv(tmp,index=False);tmp.replace(p);z={'selector':sel,'criterion':crit,'configuration':name,'seed':seed,'features':X.shape[1],'selection_mean':r.selection.mean(),'regret_mean':r.regret.mean(),'completed_at':datetime.now(timezone.utc).isoformat()};meta.write_text(json.dumps(z,indent=2)+'\n');return z
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selector',choices=['RF','MLP'],required=True);ap.add_argument('--jobs',type=int,default=-1);ap.add_argument('--seeds',type=int,nargs='+',default=[42]);ap.add_argument('--configs',nargs='*',default=None);a=ap.parse_args();_,cfg=load('AR');rows=[];names=a.configs or list(cfg)
 for crit in CFG['criteria']:
  for name in names:
   for seed in a.seeds:
    z=one(a.selector,crit,name,seed,a.jobs);rows.append(z);print(json.dumps(z),flush=True)
 pd.DataFrame(rows).to_csv(HERE/'results'/f'{a.selector}_summary.csv',index=False)
if __name__=='__main__':main()
