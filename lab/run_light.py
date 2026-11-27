import sys, json, time
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from fares_feats import fares_light
import sim3
from sim3 import chunk_feats, oof_chunk_probs, score_probs_fast, EVAL_SC, TRAIN_SC, fmt
recs = load_all()
d = recs[0]["df"]; ch = d.iloc[:np.searchsorted(d.index.values, d.index.values[0] + np.timedelta64(600, "s"))]
t = time.perf_counter(); [fares_light(ch) for _ in range(20)]; print(f"fares_light on 600 s: {(time.perf_counter()-t)/20*1000:.1f} ms", flush=True)
SC = sorted(set(EVAL_SC) | set(TRAIN_SC))
SW = {"full": 1.0, "chunk600": 0.1, "chunk300": 0.1, "chunk180": 0.1, "chunk120": 0.1}
mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
CFn = chunk_feats(recs, "c_nxe", None, SC)
CFl = chunk_feats(recs, "c_fareslite", fares_light, SC)
CF = {sc: [[(a, b, w0, np.hstack([X1, X2[:len(X1)]])) for (a, b, w0, X1), (_, _, _, X2) in zip(CFn[sc][i], CFl[sc][i])] for i in range(len(recs))] for sc in SC}
sim3._CFREF.clear(); sim3._CFREF.update(CF)
for seed in (0, 1):
    OP = oof_chunk_probs(recs, CF, mk, train_sc=TRAIN_SC, seed=seed, sc_weight=SW, cache_name=f"cmp_light_{seed}")
    print(f"ours+fares_light seed{seed}: {fmt(score_probs_fast(recs, OP, (1, 1.4, 1), 5))}", flush=True)
