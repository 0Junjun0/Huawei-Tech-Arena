"""Which scoring rule / subset reproduces the leaderboard order (RF 73 ~ Fares1 74 ~ nxe_p 73 >> bebe_time 67)?
OOF whole-session predictions per recording, then several ways of aggregating."""
import sys, pickle, pathlib
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from data import load_all
from feats import compute
from holdout import folds_subject, smooth_probs
recs = load_all(); C = pathlib.Path("lab/cache")
full = lambda k: [p[0][3] for p in pickle.loads((C / f"cf_{k}_full.pkl").read_bytes())]
L = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=6)
M = {
 "rf": (full("e_rf"), lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=6))]), 3, (0.95, 1.8, 0.85)),
 "fares1": (full("e_fa1"), lambda: HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, learning_rate=0.075, l2_regularization=5, class_weight="balanced", early_stopping=False, random_state=42), 7, (1, 1.4, 1)),
 "nxe": (full("e_nxe"), L, 5, (1, 1.4, 1)),
 "bebe_time": ([compute(r["df"], use_time=True)[1] for r in recs], L, 3, (1, 1, 1)),
 "bebe_notime": ([compute(r["df"], use_time=False)[1] for r in recs], L, 3, (1, 1, 1)),
}
p = C / "scoring_per.pkl"
PER = pickle.loads(p.read_bytes()) if p.exists() else {}
g = folds_subject(recs, 5, 0)
for name, (X, mk, sm, w) in M.items():
    if name in PER: continue
    per = {}
    for k in np.unique(g):
        tr = np.flatnonzero(g != k)
        Xt = np.vstack([X[i][:len(recs[i]["y"])] for i in tr]); yt = np.concatenate([recs[i]["y"][:len(X[i])] for i in tr]); m = yt >= 0
        mod = mk(); mod.fit(Xt[m], yt[m])
        for i in np.flatnonzero(g == k):
            y = recs[i]["y"]; n = min(len(y), len(X[i]))
            P = mod.predict_proba(X[i][:n])
            per[i] = (y[:n], P)
    PER[name] = per; p.write_bytes(pickle.dumps(PER))
    print(name, "done", flush=True)

def preds(name, idx):
    _, _, sm, w = M[name]; out = []
    for i in idx:
        y, P = PER[name][i]; q = (smooth_probs(P, sm) * np.asarray(w)).argmax(1); m = y >= 0
        out.append((y[m], q[m]))
    return out
def pooled(pp): return f1_score(np.concatenate([a for a, _ in pp]), np.concatenate([b for _, b in pp]), average="macro")
def per_union(pp): return np.mean([f1_score(a, b, average="macro") for a, b in pp])
def per_fixed(pp, zd): return np.mean([f1_score(a, b, labels=[0, 1, 2], average="macro", zero_division=zd) for a, b in pp])
def per_true(pp): return np.mean([f1_score(a, b, labels=np.unique(a), average="macro") for a, b in pp])
def acc(pp): return np.mean(np.concatenate([a == b for a, b in pp]))
allidx = list(range(len(recs)))
subsets = {"all": allidx, "S-stage": [i for i in allidx if recs[i]["subject"][0] == "S"], "f-stage": [i for i in allidx if recs[i]["subject"][0] == "f"]}
for pr in ("STRESS", "AEROBIC", "ANAEROBIC"): subsets[pr] = [i for i in allidx if recs[i]["protocol"] == pr]
print(f"{'':12s}" + "".join(f"{k:>12s}" for k in ["pooled", "sess-union", "sess-fix0", "sess-fix1", "sess-true", "acc"]))
for sn, idx in subsets.items():
    print("--", sn, len(idx))
    for name in M:
        pp = preds(name, idx)
        print(f"{name:12s}" + "".join(f"{v:12.3f}" for v in [pooled(pp), per_union(pp), per_fixed(pp, 0), per_fixed(pp, 1), per_true(pp), acc(pp)]), flush=True)
