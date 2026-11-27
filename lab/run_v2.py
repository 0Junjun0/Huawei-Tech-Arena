import sys, time
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from cv import build, oof, score, tune

recs = load_all()
def mk(): return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
for use_time in (False, True):
    items = build(recs, lambda df: compute(df, use_time=use_time))
    t=time.perf_counter(); P = oof(items, mk); print(f"cv {time.perf_counter()-t:.0f}s")
    print(f"== use_time={use_time}"); score(items, P, verbose=True)
    print("tuned:", tune(items, P))
