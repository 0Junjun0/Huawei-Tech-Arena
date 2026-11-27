"""Score model families on the hard benchmark (lab/hard.py); optionally train with fault-augmented copies.
usage: run_hard.py fam1,fam2 [aug_weight] [fold_seeds]"""
import sys, pathlib
sys.path.insert(0, "lab")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from data import load_all
from feats import compute
from holdout import folds_subject, smooth_probs
import hard

recs = load_all()
_d = recs[0]["df"].iloc[:640]
NX = list(compute(_d, **hard.KW_NXE)[2]); LX = list(compute(_d, use_time=False, local=True, extras=True)[2])
LXI = np.array([NX.index(n) for n in LX])
LG = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=12)
SPEC = {  # name: (cache family, column selector, model, smooth, class weights)
    "time": ("time", None, LG, 3, (1, 1, 1)),
    "nxe": ("nxe", None, LG, 5, (1, 1.4, 1)),
    "lx": ("nxe", LXI, LG, 9, (1, 1.8, 1.2)),
    "fa1": ("fa1", None, lambda: HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, learning_rate=0.075, l2_regularization=5,
                                                               class_weight="balanced", early_stopping=False, random_state=42), 7, (1, 1.4, 1)),
    "rf": ("rf", None, lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=12))]), 3, (0.95, 1.8, 0.85)),
}
LB = {"time": 0.60, "nxe": 0.675, "lx": 0.687, "fa1": 0.686, "rf": 0.694}


def data(fam, kind, seed, sel):
    out = hard.build(fam, kind, seed, recs)
    return [(X if sel is None else X[:, sel], y) for X, y in out]


def run(name, aug_w=0.0, fold_seed=0, aug_kinds=("crop", "conn", "nanblk", "place", "mix")):
    fam, sel, mk, sm, w = SPEC[name]
    EV = {k: data(fam, k, 0, sel) for k in hard.SCEN}
    TR = [(data(fam, "clean", 0, sel), 1.0)]
    if aug_w > 0:
        TR += [(data(fam, k, s, sel), aug_w) for k in aug_kinds for s in (1, 2)]
    g = folds_subject(recs, 5, fold_seed); res = {k: ([], []) for k in hard.SCEN}
    for f in np.unique(g):
        tr = np.flatnonzero(g != f)
        Xs, ys, ws = [], [], []
        for D, wt in TR:
            for i in tr:
                X, y = D[i]; m = y >= 0; Xs.append(X[m]); ys.append(y[m]); ws.append(np.full(m.sum(), wt))
        X, y, sw = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
        mod = mk()
        if isinstance(mod, Pipeline): mod.fit(X, y, rf__sample_weight=sw)
        else: mod.fit(X, y, sample_weight=sw)
        for i in np.flatnonzero(g == f):
            for k in hard.SCEN:
                X, y = EV[k][i]
                P = np.zeros((len(X), 3)); P[:, list(mod.classes_)] = mod.predict_proba(X)
                q = (smooth_probs(P, sm) * np.asarray(w)).argmax(1); m = y >= 0
                res[k][0].append(y[m]); res[k][1].append(q[m])
    F = {k: f1_score(np.concatenate(a), np.concatenate(b), average="macro") for k, (a, b) in res.items()}
    F["faults"] = np.mean([F[k] for k in hard.SCEN if k != "clean"])
    F["lb_like"] = 0.5 * F["clean"] + 0.5 * F["faults"]   # half the hidden sessions are flagged
    return F


if __name__ == "__main__":
    names = sys.argv[1].split(","); aug = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
    for n in names:
        Fs = [run(n, aug, s) for s in seeds]; F = {k: np.mean([f[k] for f in Fs]) for k in Fs[0]}
        print(f"{n:5s} aug={aug:<4} " + " ".join(f"{k}={v:.3f}" for k, v in F.items()) + (f"   | LB-implied {LB[n]:.3f}" if aug == 0 else ""), flush=True)
