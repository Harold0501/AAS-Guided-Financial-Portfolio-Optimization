import numpy as np,pandas as pd
from feature_engine import compute
def synthetic():
 rows=[]
 for t in range(100):
  for j,c in enumerate(['a','b','c']):
   p=100*np.exp(.001*t+.01*np.sin(t/7+j));rows.append({'date':pd.Timestamp('2020-01-01')+pd.Timedelta(days=t),'company':c,'open':p*.99,'high':p*1.01,'low':p*.98,'close':p,'div_adj_close':p,'volume':1000+10*t+20*j})
 return pd.DataFrame(rows)
def test_deterministic_and_invariant():
 d=synthetic();a=compute(d);b=compute(d.sample(frac=1,random_state=4));assert a.keys()==b.keys();assert np.allclose(list(a.values()),list(b.values()),atol=1e-12);assert np.isfinite(list(a.values())).all()
