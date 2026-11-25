import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from sim import evaluate
recs = load_all()
SC = ["full", "skipstart", "chunk600", "chunk300", "chunk180", "chunk120", "chunk60", "rand300"]
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=2)
print("scenarios:", SC, flush=True)
for nm, fn, w in [("bebe_time", lambda d: compute(d, use_time=True), (1, 1, 1)), ("bebe_notime", lambda d: compute(d, use_time=False), (1, 1.4, 0.8)), ("local", lambda d: compute(d, use_time=False, local=True), (1, 1, 1))]:
    r = evaluate(recs, fn, lg, SC, weights=w, smooth=3); print(f"{nm:12s}", {k: round(v, 3) for k, v in r.items()}, flush=True)
