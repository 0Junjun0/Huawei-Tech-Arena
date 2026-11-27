"""Whole-session comparison: session-level vs local-only feature sets (subject 5-fold, 3 seeds)."""
import sys, pickle
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from fares_feats import fares_local
import sim3
from sim3 import chunk_feats, oof_chunk_probs, score_probs_fast
recs = load_all()
SC = ["full", "skipstart"]
mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=12)
LOC = dict(use_time=False, local=True, extras=True, fb=(3, 9, 18, 36))
CF = {"nxe": chunk_feats(recs, "c_nxe", None, SC),
      "ours_local": chunk_feats(recs, "c_loc", lambda d: compute(d, **LOC)[1], SC),
      "fares_local": chunk_feats(recs, "c_floc", fares_local, SC)}
cat = lambda A, B: {s: [[(a, b, w0, np.hstack([x1, x2[:len(x1)]])) for (a, b, w0, x1), (_, _, _, x2) in zip(A[s][i], B[s][i])] for i in range(len(recs))] for s in SC}
CF["ours_local+fares_local"] = cat(CF["ours_local"], CF["fares_local"])
for name, C in CF.items():
    sim3._CFREF.clear(); sim3._CFREF.update(C)
    res = {}
    for seed in (0, 1, 2):
        OP = oof_chunk_probs(recs, C, mk, train_sc=["full"], eval_sc=SC, seed=seed, cache_name=f"loc_{name.replace('+','p')}")
        for sm in (3, 9, 15):
            for ws in (1.0, 1.4):
                r = score_probs_fast(recs, OP, (1, ws, 1), sm, eval_sc=SC)
                res.setdefault((sm, ws), []).append((r["full"], r["skipstart"]))
    best = max(res, key=lambda k: np.mean(res[k], 0)[0])
    m = np.mean(res[best], 0)
    print(f"{name:24s} cols={C['full'][0][0][3].shape[1]:4d}  full={m[0]:.3f} skipstart={m[1]:.3f}  (smooth {best[0]}, stress w {best[1]})", flush=True)
