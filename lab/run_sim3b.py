"""generic: run_sim3b.py <key> '<feat_kw json>' [aug]"""
import sys, json
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
from feats import compute
from sim3 import *
recs = load_all()
key = sys.argv[1]; kw = json.loads(sys.argv[2]); aug = "aug" in sys.argv[3:]
CF = chunk_feats(recs, key, lambda d: compute(d, **kw)[1], sorted(set(EVAL_SC) | set(TRAIN_SC)))
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=1)
OP = oof_chunk_probs(recs, CF, lg, TRAIN_SC if aug else ("full",), cache_name=f"{key}_{'aug' if aug else 'full'}")
print(f"lgb {key:10s} {'aug' if aug else 'full'}", fmt(score_probs(recs, OP, (1, 1, 1), 3)), flush=True)
