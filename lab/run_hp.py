"""hp sweep: run_hp.py <key> <json list of lgb param overrides> [train_sc csv]"""
import sys, json
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from sim3 import *
recs = load_all()
key = sys.argv[1]; grid = json.loads(sys.argv[2]); tsc = sys.argv[3].split(",") if len(sys.argv) > 3 else ["full"]
CF = chunk_feats(recs, key, None, sorted(set(EVAL_SC) | set(tsc)))
base = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=1)
for ov in grid:
    p = dict(base, **ov)
    OP = oof_chunk_probs(recs, CF, lambda: lgb.LGBMClassifier(**p), tsc)
    print(key, "+".join(tsc), json.dumps(ov), fmt(score_probs(recs, OP, (1, 1, 1), 3)), flush=True)
