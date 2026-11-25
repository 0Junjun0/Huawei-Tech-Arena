import sys, pathlib
sys.path.insert(0, "TechArena-Finland-2026"); sys.path.insert(0, "."); sys.path.insert(0, "lab")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import submission_model as sm
from data import load_all
from cv import build, oof, score, tune

def feat(df):
    idx, X = sm._compute_window_features(df)
    return idx, X, None

recs = load_all()
print(len(recs), "recordings")
items = build(recs, feat)
mk = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
P = oof(items, mk)
print("raw:"); score(items, P, verbose=True)
print("their weights+3-smooth (approx):"); score(items, P, (0.95, 1.8, 0.85), 3, verbose=True)
print("best tuned:", tune(items, P))
