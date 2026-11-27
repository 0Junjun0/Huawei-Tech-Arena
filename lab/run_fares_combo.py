"""Ours (nxe) vs Fares ver3 features vs both, same chunk-augmented LightGBM, scenario blend."""
import sys, json
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from fares_feats import fares_feats
import sim3
from sim3 import chunk_feats, oof_chunk_probs, score_probs_fast, EVAL_SC, TRAIN_SC, fmt
recs = load_all()
kw = json.load(open("submissions/robust_nxe_pg/config.json"))["feat_kw"]
SC = sorted(set(EVAL_SC) | set(TRAIN_SC))
SW = {"full": 1.0, "chunk600": 0.1, "chunk300": 0.1, "chunk180": 0.1, "chunk120": 0.1}
mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
CFn = chunk_feats(recs, "c_nxe", lambda d: compute(d, **kw)[1], SC); print("nxe feats done", flush=True)
CFf = chunk_feats(recs, "c_fares3", fares_feats, SC); print("fares feats done", flush=True)
CFb = {sc: [[(a, b, w0, np.hstack([X1, X2[:len(X1)]])) for (a, b, w0, X1), (_, _, _, X2) in zip(CFn[sc][i], CFf[sc][i])] for i in range(len(recs))] for sc in SC}
for name, CF in [("ours(nxe)", CFn), ("fares3", CFf), ("both", CFb)]:
    sim3._CFREF.clear(); sim3._CFREF.update(CF)
    for seed in (0, 1):
        OP = oof_chunk_probs(recs, CF, mk, train_sc=TRAIN_SC, seed=seed, sc_weight=SW, cache_name=f"cmp_{name}_{seed}".replace("(", "").replace(")", ""))
        best = max(((ws, sm) for ws in (1.0, 1.2, 1.4, 1.7) for sm in (3, 5, 9)), key=lambda g: score_probs_fast(recs, OP, (1, g[0], 1), g[1])["blend"])
        print(f"{name:10s} seed{seed} plain(1,1.4,1,s5): {fmt(score_probs_fast(recs, OP, (1, 1.4, 1), 5))}", flush=True)
        print(f"{name:10s} seed{seed} tuned{best}: blend={score_probs_fast(recs, OP, (1, best[0], 1), best[1])['blend']:.3f}", flush=True)
