import sys
sys.path.insert(0, "lab"); sys.path.insert(0, "submission"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import load_all
from feats import compute
from sim3 import *
import submission_model as tm
recs = load_all()
ALL = sorted(set(EVAL_SC) | set(TRAIN_SC))
FF = {"local": lambda d: compute(d, use_time=False, local=True)[1],
      "notime": lambda d: compute(d, use_time=False)[1],
      "time": lambda d: compute(d, use_time=True)[1],
      "localx": lambda d: compute(d, use_time=False, local=True, extras=True)[1],
      "rf": lambda d: tm._compute_window_features(d)[1]}
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=4)
rf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=4))])
which = sys.argv[1:] or list(FF)
for k in which:
    CF = chunk_feats(recs, k, FF[k], ALL)
    if k == "rf":
        print("RF-exact      full-train", fmt(evaluate(recs, CF, rf, ("full",), weights=(0.95, 1.8, 0.85), vote=True)), flush=True)
        print("RF-orig1.35   full-train", fmt(evaluate(recs, CF, rf, ("full",), weights=(0.9, 1.35, 0.9), vote=True)), flush=True)
    for ts in (("full",), TRAIN_SC):
        print(f"lgb {k:8s} train={'aug' if len(ts) > 1 else 'full'}", fmt(evaluate(recs, CF, lg, ts)), flush=True)
