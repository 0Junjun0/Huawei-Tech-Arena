"""Scenario: grader feeds one person's sessions concatenated in a single run() call."""
import sys
sys.path.insert(0, "lab"); sys.path.insert(0, "submission"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import load_all
from feats import compute
from sim import predict_rows, window_majority
import submission_model as tm
recs = load_all()
subj = np.array([r["subject"] for r in recs]); us = np.unique(subj)
rng = np.random.RandomState(0); perm = {s: i % 5 for i, s in enumerate(rng.permutation(us))}
def run(featfn, mk, w, gap_min):
    feats = [featfn(r["df"])[1] for r in recs]
    Y, Pp = [], []
    for k in range(5):
        tr = [i for i in range(len(recs)) if perm[subj[i]] != k]
        X = np.vstack([feats[i][:len(recs[i]["y"])] for i in tr]); y = np.concatenate([recs[i]["y"][:len(feats[i])] for i in tr])
        model = mk(); model.fit(X[y >= 0], y[y >= 0])
        for s in [s for s in us if perm[s] == k]:
            parts, ys = [], []
            off = pd.Timedelta(0); first = None
            for i in [i for i in range(len(recs)) if subj[i] == s]:
                d = recs[i]["df"].copy()
                if first is None: first = d.index[0]
                # shift sessions so they follow each other with a gap
                d.index = d.index - d.index[0] + first + off
                off = (d.index[-1] - first) + pd.Timedelta(minutes=gap_min) + pd.Timedelta(seconds=10 - (d.index[-1] - first).total_seconds() % 10)
                parts.append(d)
                yy = recs[i]["y"]
                ys.append((d.index, yy))
            big = pd.concat(parts)
            row = predict_rows(big, featfn, model, w, 3)
            t = big.index.values.astype("datetime64[ns]").view(np.int64)
            pw = window_majority(t, row)
            for (idx, yy) in ys:
                w0 = (idx[0].value // 10**10) - (t[0] // 10**10)
                seg = pw[w0:w0 + len(yy)]; n = min(len(seg), len(yy)); m = (yy[:n] >= 0) & (seg[:n] >= 0)
                Y.append(yy[:n][m]); Pp.append(seg[:n][m])
    return f1_score(np.concatenate(Y), np.concatenate(Pp), average="macro")
def rf_feat(df):
    idx, X = tm._compute_window_features(df); return None, X, None
rf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
for gap in (0,):
    print("concat gap", gap, "RF", round(run(rf_feat, rf, (0.95, 1.8, 0.85), gap), 3), flush=True)
    print("concat gap", gap, "bebe_time", round(run(lambda d: compute(d, use_time=True), lg, (1, 1, 1), gap), 3), flush=True)
    print("concat gap", gap, "bebe_notime", round(run(lambda d: compute(d, use_time=False), lg, (1, 1.4, 0.8), gap), 3), flush=True)
