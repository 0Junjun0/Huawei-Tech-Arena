"""Compare RF (leaderboard 73) vs bebe_time (67) vs bebe_notime under recording-holdout vs subject-holdout."""
import sys
sys.path.insert(0, "lab"); sys.path.insert(0, "submission"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import load_all
from feats import compute
from holdout import *
import submission_model as tm
recs = load_all()
FS = {
  "rf": feats_cached(recs, "rf", lambda d: tm._compute_window_features(d)[1]),
  "time": feats_cached(recs, "time", lambda d: compute(d, use_time=True)[1]),
  "notime": feats_cached(recs, "notime", lambda d: compute(d, use_time=False)[1]),
  "local": feats_cached(recs, "local", lambda d: compute(d, use_time=False, local=True)[1]),
}
rf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=4))])
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=4)
CFG = [("RF(lb73)", "rf", rf, dict(weights=(0.95, 1.8, 0.85), vote=True)),
       ("bebe_time(lb67)", "time", lg, dict(weights=(1, 1, 1), smooth=3)),
       ("bebe_notime", "notime", lg, dict(weights=(1, 1.4, 0.8), smooth=3)),
       ("local_lgb", "local", lg, dict(weights=(1, 1, 1), smooth=3))]
for split in ("recording", "subject"):
    for name, fk, mk, pp in CFG:
        res = []
        for seed in range(3):
            g = folds_recording(recs, 7, seed) if split == "recording" else folds_subject(recs, 5, seed)
            P = oof_probs(recs, FS[fk], mk, g)
            res.append(score_sets(recs, [predict(p, **pp) for p in P], g))
        print(f"{split:9s} {name:16s} per-set mean {np.mean([r[0] for r in res]):.3f} (sd {np.mean([r[1] for r in res]):.3f})  pooled {np.mean([r[2] for r in res]):.3f}", flush=True)
