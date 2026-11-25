import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from cv import build, oof, score
recs = load_all()
def mkf(**kw):
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
    base.update(kw); return lambda: lgb.LGBMClassifier(**base)
cfgs = {"base": {}, "small": dict(num_leaves=15, n_estimators=400), "big": dict(num_leaves=63, min_child_samples=80),
        "slow": dict(n_estimators=700, learning_rate=0.025), "reg": dict(min_child_samples=150, reg_lambda=5.0, colsample_bytree=0.3)}
for use_time in (False, True):
    items = build(recs, lambda df: compute(df, use_time=use_time))
    for name, kw in cfgs.items():
        fs = []
        for seed in (0, 1, 2):
            P = oof(items, mkf(**kw), seed=seed)
            fs.append([score(items, P, (1, 1, 1), s) for s in (1, 3, 5)])
        fs = np.array(fs)
        print(f"time={use_time} {name:6s} smooth1/3/5 mean over 3 splits: {np.round(fs.mean(0),4)}  sd {np.round(fs.std(0),4)}", flush=True)
