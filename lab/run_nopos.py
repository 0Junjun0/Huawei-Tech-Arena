"""Position-free variants of nxe (no input length / edge distance; optionally no recording-level constants). Whole-session CV."""
import sys, pickle, pathlib
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from sklearn.metrics import f1_score
from data import load_all
from feats import compute
from holdout import folds_subject, smooth_probs
import hard
recs = load_all(); C = pathlib.Path("lab/cache")
p = C / "nopos_X.pkl"
if p.exists(): X, NM = pickle.loads(p.read_bytes())
else:
    X = []; NM = None
    for r in recs:
        _, x, nm = compute(r["df"], **hard.KW_NXE); X.append(x); NM = nm
    p.write_bytes(pickle.dumps((X, NM)))
NM = list(NM)
def cols(drop):
    return np.array([i for i, n in enumerate(NM) if not any(n.startswith(d) or n == d for d in drop)])
POS = ["log_n_win", "edge_start", "edge_end"]
REC = [n for n in NM if "_rec_" in n]
ZR = [n for n in NM if n.endswith(("_zrec", "_c19_zrec", "_c19_mp10", "_c19_rng", "_c19_rank"))]
LXN = list(compute(recs[0]["df"].iloc[:640], use_time=False, local=True, extras=True)[2])
V = {"nxe (as shipped)": cols([]), "nxe - position": cols(POS), "nxe - position - rec consts": cols(POS + REC),
     "nxe - position - all session-relative": cols(POS + REC + ZR), "lx": np.array([NM.index(n) for n in LXN])}
LG = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=12)
for name, sel in V.items():
    out = []
    for seed in (0, 1):
        g = folds_subject(recs, 5, seed); Y, P = [], []
        for k in np.unique(g):
            tr = np.flatnonzero(g != k)
            Xt = np.vstack([X[i][:len(recs[i]["y"]), sel] for i in tr]); yt = np.concatenate([recs[i]["y"][:len(X[i])] for i in tr]); m = yt >= 0
            mod = LG().fit(Xt[m], yt[m])
            for i in np.flatnonzero(g == k):
                y = recs[i]["y"]; n = min(len(y), len(X[i])); q = (smooth_probs(mod.predict_proba(X[i][:n, sel]), 5) * np.array([1, 1.4, 1])).argmax(1)
                mm = y[:n] >= 0; Y.append(y[:n][mm]); P.append(q[mm])
        out.append(f1_score(np.concatenate(Y), np.concatenate(P), average="macro"))
    print(f"{name:40s} feats={len(sel):3d}  CV {np.mean(out):.3f} {np.round(out, 3)}", flush=True)
