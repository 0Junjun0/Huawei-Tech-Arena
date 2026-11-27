"""combine cached chunk features: run_combo.py name key1 key2 ... [aug]"""
import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb, pickle
from data import load_all
from sim3 import *
recs = load_all()
name = sys.argv[1]; keys = [k for k in sys.argv[2:] if k != "aug"]; aug = "aug" in sys.argv
SC = sorted(set(EVAL_SC) | set(TRAIN_SC))
CFs = [chunk_feats(recs, k, None, SC) for k in keys]
CF = {sc: [[(a, b, w0, np.hstack([C[sc][i][j][3][:min(len(C2[sc][i][j][3]) for C2 in CFs)] for C in CFs]))
            for j, (a, b, w0, _) in enumerate(CFs[0][sc][i])] for i in range(len(recs))] for sc in SC}
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=1)
OP = oof_chunk_probs(recs, CF, lg, TRAIN_SC if aug else ("full",), cache_name=f"{name}_{'aug' if aug else 'full'}")
print(f"lgb {name:10s} {'aug' if aug else 'full'}", fmt(score_probs(recs, OP, (1, 1, 1), 3)), flush=True)
