import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from cv import build, oof, score
recs = load_all()
items = build(recs, lambda df: compute(df, use_time=True))
for ne, lr, nl in [(400,.05,15),(200,.1,15),(120,.15,15),(150,.1,7),(80,.2,15)]:
    mk = lambda: lgb.LGBMClassifier(n_estimators=ne, learning_rate=lr, num_leaves=nl, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
    f = [score(items, oof(items, mk, seed=s), (1,1,1), 3) for s in (0,1,2)]
    print(ne, lr, nl, round(np.mean(f),4), flush=True)
