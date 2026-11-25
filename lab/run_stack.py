import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from cv import build, oof, score
from stack import stage2_features
recs = load_all()
items = build(recs, lambda df: compute(df, use_time=True))
mk1 = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
mk2 = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=60, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, class_weight="balanced", reg_lambda=2.0, verbose=-1, n_jobs=8)
res = []
for seed in (0, 1, 2):
    P1 = oof(items, mk1, seed=seed)
    # stage-2 trained on OOF stage-1 probs, evaluated with a *different* subject split (seed+10) to avoid optimistic reuse
    items2 = [dict(it, X=stage2_features(p)) for it, p in zip(items, P1)]
    P2 = oof(items2, mk2, seed=seed + 10)
    # also stage-2 with original features appended
    items3 = [dict(it, X=np.hstack([it["X"], stage2_features(p)])) for it, p in zip(items, P1)]
    P3 = oof(items3, mk1, seed=seed + 10)
    r = [score(items, P1, smooth=3), score(items, P2, smooth=1), score(items, P2, smooth=3), score(items, P3, smooth=1), score(items, P3, smooth=3)]
    res.append(r); print(seed, np.round(r, 4), flush=True)
print("mean  stage1(s3)  stage2(s1)  stage2(s3)  stage2+raw(s1)  stage2+raw(s3)\n     ", np.round(np.mean(res, 0), 4))
score(items, P2, smooth=3, verbose=True)
