from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from feature_engine import compute
HERE=Path(__file__).resolve().parent;CFG=json.loads((HERE/'config.json').read_text());ROOT=Path(CFG['source_root'])
def main():
 out=HERE/'features/discovered_features.csv';audit=HERE/'features/computation_audit.csv';rows=[];aud=[];done=set()
 if out.exists(): rows=pd.read_csv(out).to_dict('records');done={int(x['instance']) for x in rows}
 if audit.exists():aud=pd.read_csv(audit).to_dict('records')
 m=pd.read_csv(ROOT/CFG['mapping_csv']);m['start']=pd.to_datetime(m.start,format='mixed');m['end']=pd.to_datetime(m.end,format='mixed')
 for asset,g in m.groupby('Asset',sort=True):
  raw=pd.read_csv(ROOT/CFG['raw_data_dir']/f'{asset}.csv');raw.date=pd.to_datetime(raw.date)
  for _,x in g.sort_values('instance').iterrows():
   iid=int(x.instance)
   if iid in done:continue
   d=raw[(raw.date>=x.start)&(raw.date<=x.end)].copy()
   try: vals=compute(d);status='ok';warning=''
   except Exception as e: vals={};status='failed';warning=repr(e)
   rows.append({'instance':iid,**vals});aud.append({'instance':iid,'Asset':asset,'rows':len(d),'companies':d.company.nunique(),'status':status,'warning':warning})
   for path,data in [(out,rows),(audit,aud)]:
    tmp=path.with_suffix('.tmp');pd.DataFrame(data).sort_values('instance').to_csv(tmp,index=False);tmp.replace(path)
   if iid%50==0:print(iid,flush=True)
 f=pd.read_csv(out);num=f.drop(columns='instance');q={'instances':len(f),'features':len(num.columns),'finite':bool(np.isfinite(num).all().all()),'missing':int(num.isna().sum().sum()),'unique':num.nunique().to_dict()}
 (HERE/'features/quality.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
