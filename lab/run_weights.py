import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from cv import build, oof, score
recs = load_all()
mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
for use_time in (False, True):
    items = build(recs, lambda df: compute(df, use_time=use_time))
    Ps = [oof(items, mk, seed=s) for s in (0, 1, 2, 3)]
    grid = [(ws, wa, sm) for ws in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4) for wa in (0.8, 0.9, 1.0, 1.1) for sm in (1, 3, 5, 9)]
    # pick on splits 0-1, evaluate on splits 2-3
    pick = max(grid, key=lambda g: np.mean([score(items, P, (1, g[0], g[1]), g[2]) for P in Ps[:2]]))
    held = lambda g: np.mean([score(items, P, (1, g[0], g[1]), g[2]) for P in Ps[2:]])
    print(f"time={use_time}: picked ws,wa,smooth={pick}  held-out {held(pick):.4f}  vs plain(1,1,smooth3) {held((1.0,1.0,3)):.4f}")
