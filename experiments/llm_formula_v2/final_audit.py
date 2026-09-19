from __future__ import annotations
import hashlib,json,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

manifest=pd.read_csv(HERE/'manifests'/'input_manifest.csv')
checks=[]
for r in manifest.itertuples(index=False):
 p=Path(r.path); checks.append({'path':r.path,'exists':p.exists(),'unchanged':p.exists() and sha(p)==r.sha256})
pd.DataFrame(checks).to_csv(HERE/'manifests'/'input_integrity_final.csv',index=False)
outputs=[]
for p in sorted(HERE.rglob('*')):
 if p.is_file() and '.venv' not in p.parts and p.name!='output_manifest_final.csv':
  outputs.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
pd.DataFrame(outputs).to_csv(HERE/'manifests'/'output_manifest_final.csv',index=False)
env={'python':sys.version,'platform':platform.platform()}
for package in ['numpy','pandas','scipy','sklearn','joblib','statsmodels','matplotlib','seaborn']:
 try:
  mod=__import__(package);env[package]=getattr(mod,'__version__','unknown')
 except Exception as e:env[package]=f'unavailable: {e}'
(HERE/'manifests'/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
state=json.loads((HERE/'state.json').read_text()); state.update({'status':'step1_complete_analysis_stop','completed_stages':sorted(set(state.get('completed_stages',[])+['formula_computation','main_rf_mlp','negative_controls','group_asset_robustness','individual_feature_diagnostics','statistical_analysis','final_audit'])),'next_stage':'human_review_and_step2_decision','updated_at':datetime.now(timezone.utc).isoformat()});(HERE/'state.json').write_text(json.dumps(state,indent=2)+'\n')
summary={'manifested_inputs':len(checks),'inputs_unchanged':sum(x['unchanged'] for x in checks),'missing_inputs':sum(not x['exists'] for x in checks),'outputs_manifested':len(outputs),'completed_at':datetime.now(timezone.utc).isoformat()}
(HERE/'outputs'/'analysis'/'final_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
