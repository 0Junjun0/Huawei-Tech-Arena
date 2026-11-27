"""Tune smoothing + class weights on the scenario blend. usage: run_tune3.py <featkey> [aug]"""
import sys
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb, itertools
from data import load_all
import sim3
from sim3 import *
recs = load_all()
import json
key = sys.argv[1]; aug = len(sys.argv) > 2 and sys.argv[2] in ("aug", "light")
light = len(sys.argv) > 2 and sys.argv[2] == "light"
SW = {"full": 1.0, "chunk600": 0.3, "chunk300": 0.3, "chunk180": 0.3, "chunk120": 0.3} if light else None
CF = chunk_feats(recs, key, None, sorted(set(EVAL_SC) | set(TRAIN_SC)))
sim3._CFREF.update(CF)
lg = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=1)
OP = oof_chunk_probs(recs, CF, lg, TRAIN_SC if aug else ("full",), cache_name=f"{key}_{sys.argv[2] if len(sys.argv) > 2 else 'full'}", sc_weight=SW)
best = []
for sm, ws, wa in itertools.product((1, 3, 5, 7, 9, 13, 19), (0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0), (0.8, 0.9, 1.0, 1.15, 1.3)):
    r = score_probs_fast(recs, OP, (1, ws, wa), sm)
    best.append((round(r["blend"], 4), sm, ws, wa))
best.sort(reverse=True)
print("default", [b for b in best if b[1:4] == (3, 1.0, 1.0)][0])
for b in best[:5]:
    print(b, fmt(score_probs(recs, OP, (1, b[2], b[3]), b[1])), flush=True)
