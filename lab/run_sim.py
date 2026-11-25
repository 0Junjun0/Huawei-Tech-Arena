import sys
sys.path.insert(0, "lab"); sys.path.insert(0, "submission"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import load_all
from feats import compute
from sim import evaluate
import submission_model as tm
recs = load_all()
SC = ["full", "skipstart", "chunk300", "chunk120", "chunk60", "chunk30", "rand300"]
def rf_feat(df):
    idx, X = tm._compute_window_features(df); return None, X, None
rf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
print("scenarios:", SC, flush=True)
r = evaluate(recs, rf_feat, rf, SC, weights=(0.95, 1.8, 0.85), smooth=3); print("teammate RF ", {k: round(v, 3) for k, v in r.items()}, flush=True)
r = evaluate(recs, lambda d: compute(d, use_time=True), lg, SC); print("bebe_time   ", {k: round(v, 3) for k, v in r.items()}, flush=True)
r = evaluate(recs, lambda d: compute(d, use_time=False), lg, SC, weights=(1, 1.4, 0.8)); print("bebe_notime ", {k: round(v, 3) for k, v in r.items()}, flush=True)
