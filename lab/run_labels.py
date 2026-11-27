"""Score OOF predictions under alternative ground-truth / grid definitions the grader might use."""
import sys
sys.path.insert(0, "lab"); sys.path.insert(0, "submission"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from data import load_all, files, phase_to_id
from feats import compute
from holdout import *
from oofcache import oof_cached
import submission_model as tm
recs = load_all()
ph = {(f.parent.name, f.stem): pd.read_parquet(f, columns=["Phase"]).sort_index()["Phase"] for f in files()}
rf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=4))])
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=4)
CFG = [("RF", "rf", rf, dict(weights=(0.95, 1.8, 0.85), vote=True)), ("bebe_time", "time", lg, dict(smooth=3)),
       ("bebe_notime", "notime", lg, dict(weights=(1, 1.4, 0.8), smooth=3)), ("local", "local", lg, dict(smooth=3))]
FS = {k: feats_cached(recs, k, None) for k in ("rf", "time", "notime", "local")}
def labels(p, mode):
    s = p.astype(str).str.strip().str.lower()
    lab = s.map(phase_to_id).to_numpy()
    if mode == "undef_ffill":
        x = pd.Series(np.where(lab >= 0, lab, np.nan)).ffill().bfill().to_numpy(); lab = x.astype(int)
    elif mode == "undef_rest":
        lab = np.where(lab >= 0, lab, 0)
    elif mode == "undef_stress_in_stress":  # handled outside
        pass
    return lab
W = 10**10
def win_major(t, v, start_first):
    base = t[0] if start_first else (t[0] // W) * W
    wid = (t - base) // W
    n = int(wid[-1]) + 1
    c = np.zeros((n, 3)); m = v >= 0
    np.add.at(c, (wid[m], v[m]), 1)
    o = c.argmax(1); o[c.sum(1) == 0] = -1
    return o
for name, fk, mk, pp in CFG:
    P, g = oof_cached(recs, name, FS[fk], mk)
    out = []
    for mode in ("std", "undef_ffill", "undef_rest", "first_ts_grid", "row_level", "drop_first_last_min"):
        Y, Pr, G = [], [], []
        for i, r in enumerate(recs):
            p = ph[(r["protocol"], r["subject"])]
            t = p.index.values.astype("datetime64[ns]").view(np.int64)
            wid = t // W - t[0] // W
            pw = predict(P[i], **pp)
            rowp = pw[np.clip(wid, 0, len(pw) - 1)]
            lab = labels(p, "std" if mode in ("std", "first_ts_grid", "row_level", "drop_first_last_min") else mode)
            if mode == "row_level":
                y, q = lab, rowp
            else:
                y = win_major(t, lab, mode == "first_ts_grid"); q = win_major(t, rowp, mode == "first_ts_grid")
                if mode == "drop_first_last_min":
                    y = y.copy(); y[:6] = -1; y[-6:] = -1
            m = (y >= 0) & (q >= 0)
            Y.append(y[m]); Pr.append(q[m]); G += [g[i]] * int(m.sum())
        Y = np.concatenate(Y); Pr = np.concatenate(Pr); G = np.array(G)
        per = [f1_score(Y[G == k], Pr[G == k], average="macro") for k in np.unique(G)]
        out.append(f"{mode}={np.mean(per):.3f}")
    print(f"{name:12s}", " ".join(out), flush=True)
