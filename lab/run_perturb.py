"""Train per recording-fold on clean data, score held-out recordings whose *input* is perturbed."""
import sys
sys.path.insert(0, "lab"); sys.path.insert(0, "submission"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import load_all
from feats import compute
from holdout import *
import submission_model as tm
recs = load_all()
PERT = {
 "clean": lambda d: d,
 "ffill": lambda d: d.ffill(),
 "ffill_bfill": lambda d: d.ffill().bfill(),
 "interp": lambda d: d.interpolate(limit_direction="both"),
 "fill0": lambda d: d.fillna(0.0),
 "no_ibi": lambda d: d.assign(IBI_dt=np.nan, IBI_duration=np.nan),
 "f32": lambda d: d.astype("float32"),
}
FF = {"rf": lambda d: tm._compute_window_features(d)[1], "time": lambda d: compute(d, use_time=True)[1],
      "notime": lambda d: compute(d, use_time=False)[1], "local": lambda d: compute(d, use_time=False, local=True)[1]}
FS = {k: feats_cached(recs, k, f) for k, f in FF.items()}
rf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=4))])
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=4)
CFG = [("RF", "rf", rf, dict(weights=(0.95, 1.8, 0.85), vote=True)), ("bebe_time", "time", lg, dict(smooth=3)),
       ("bebe_notime", "notime", lg, dict(weights=(1, 1.4, 0.8), smooth=3)), ("local", "local", lg, dict(smooth=3))]
g = folds_recording(recs, 7, 0)
res = {}
for name, fk, mk, pp in CFG:
    models = {}
    for k in np.unique(g):
        tr = np.flatnonzero(g != k)
        X = np.vstack([FS[fk][i] for i in tr]); y = np.concatenate([recs[i]["y"][:len(FS[fk][i])] for i in tr]); m = y >= 0
        models[k] = mk().fit(X[m], y[m])
    for pn, pf in PERT.items():
        preds = []
        for i, r in enumerate(recs):
            X = FF[fk](pf(r["df"]))[:len(r["y"])]
            P = models[g[i]].predict_proba(np.nan_to_num(X.astype(np.float32)) if fk != "rf" else X)
            preds.append(predict(P, **pp))
        s = score_sets(recs, preds, g)
        print(f"{name:12s} {pn:12s} set-mean {s[0]:.3f} pooled {s[2]:.3f}", flush=True)
