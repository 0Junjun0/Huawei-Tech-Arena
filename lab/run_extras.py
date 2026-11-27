import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from cv import build, oof, score
recs = load_all()
mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
for use_time in (True, False):
    for ex in (False, True):
        items = build(recs, lambda df: compute(df, use_time=use_time, extras=ex))
        f = [[score(items, P, (1,1,1), s) for s in (1,3,5)] for P in [oof(items, mk, seed=s) for s in (0,1,2,3)]]
        print(f"time={use_time} extras={ex}: smooth1/3/5 {np.round(np.mean(f,0),4)}", flush=True)
