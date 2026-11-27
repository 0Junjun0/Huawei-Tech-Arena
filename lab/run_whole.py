"""Whole-session regime (probe_len confirmed the grader feeds whole recordings).
Subject-fold CV over several seeds; eval on full sessions and on sessions missing their start.
Variants differ in feature set / training scenarios / model; post-processing swept afterwards."""
import sys, pickle, pathlib
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from data import load_all
import sim3
from sim3 import oof_chunk_probs, score_probs_fast

recs = load_all()
C = pathlib.Path("lab/cache")


def cf(key, scs):
    return {s: pickle.loads((C / f"cf_{key}_{s}.pkl").read_bytes()) for s in scs}


def concat(*cfs):
    out = {}
    for s in cfs[0]:
        out[s] = [[(a, b, w0, np.hstack([c[s][i][j][3] for c in cfs])) for j, (a, b, w0, _) in enumerate(cfs[0][s][i])]
                  for i in range(len(recs))]
    return out


ALL = ["full", "skipstart", "gain_full"]
nxe = cf("c_nxe", ALL)
f3 = cf("c_fares3", ALL)
fa1 = cf("e_fa1", ["full"])
nf3 = concat(nxe, f3)
nfa1 = concat({"full": nxe["full"]}, fa1)


def lgbm(n=400, lr=0.05, leaves=15):
    return lambda: lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=40, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0,
                                      verbose=-1, n_jobs=12)


hgb = lambda: HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, learning_rate=0.075, l2_regularization=5,
                                             class_weight="balanced", early_stopping=False, random_state=42)

V = {
    "nxe/full": (nxe, ["full"], lgbm(), ["full", "skipstart"]),
    "nxe/full+skip+gain": (nxe, ["full", "skipstart", "gain_full"], lgbm(), ["full", "skipstart"]),
    "fa1-hgb/full": (fa1, ["full"], hgb, ["full"]),
    "nxe+fa1/full": (nfa1, ["full"], lgbm(), ["full"]),
    "f3/full": (f3, ["full"], lgbm(), ["full", "skipstart"]),
    "nxe+f3/full": (nf3, ["full"], lgbm(), ["full", "skipstart"]),
    "nxe+f3/full+skip+gain": (nf3, ["full", "skipstart", "gain_full"], lgbm(), ["full", "skipstart"]),
}
only = sys.argv[1:] or list(V)
POST = [(s, w) for s in (5, 9, 15, 25) for w in ((1, 1, 1), (1, 1.4, 1))]
for name in only:
    CF, tr, mk, ev = V[name]
    sim3._CFREF.clear(); sim3._CFREF.update(CF)
    rows = {p: [] for p in POST}
    for seed in (0, 1, 2):
        OP = oof_chunk_probs(recs, CF, mk, train_sc=tr, eval_sc=ev, seed=seed,
                             cache_name="whole_" + name.replace("/", "_").replace("+", "p"))
        for s, w in POST:
            r = score_probs_fast(recs, OP, weights=w, smooth=s, eval_sc=ev)
            rows[(s, w)].append([r[e] for e in ev])
    best = max(POST, key=lambda p: np.mean(rows[p], 0)[0])
    for p in POST:
        m = np.mean(rows[p], 0)
        print(f"{name:24s} smooth={p[0]:2d} w={p[1]}  " + " ".join(f"{e}={v:.3f}" for e, v in zip(ev, m))
              + ("  <- best" if p == best else ""), flush=True)
