"""usage: run_exp.py <key> '<json: {sw: float|null, hp: {}, seed: 0, train_sc: [...], weights, smooth}>' tag"""
import sys, json
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from data import load_all
import sim3
from sim3 import *
recs = load_all()
key, o, tag = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
tsc = o.get("train_sc", TRAIN_SC)
CF = chunk_feats(recs, key, None, sorted(set(EVAL_SC) | set(tsc)))
sw = o.get("sw", 0.3)
SW = None if sw is None else {sc: (1.0 if sc == "full" else sw) for sc in tsc}
base = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=1)
base.update(o.get("hp", {}))
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
fam = o.get("family", "lgb")
mk = {"lgb": lambda: lgb.LGBMClassifier(**base),
      "et": lambda: ExtraTreesClassifier(n_estimators=300, max_depth=o.get("depth", 16), min_samples_leaf=o.get("leaf", 5), max_features=o.get("mf", 0.3), class_weight="balanced_subsample", n_jobs=1, random_state=0),
      "rf": lambda: RandomForestClassifier(n_estimators=200, max_depth=o.get("depth", 16), min_samples_leaf=o.get("leaf", 5), max_features=o.get("mf", 0.2), class_weight="balanced_subsample", n_jobs=1, random_state=0)}[fam]
OP = oof_chunk_probs(recs, CF, mk, tsc, seed=o.get("seed", 0), sc_weight=SW, cache_name=f"x_{tag}", split=o.get("split", "subject"))
print(tag, key, json.dumps(o), fmt(score_probs(recs, OP, o.get("weights", (1, 1.4, 1.0)), o.get("smooth", 5))), flush=True)

if o.get("split") == "recording":
    from sklearn.metrics import f1_score
    g = folds_recording(recs, 7, o.get("seed", 0))
    for sc in ("full", "chunk600", "chunk300"):
        per = []
        for k in range(7):
            sub = [i for i in range(len(recs)) if g[i] == k]
            r = score_probs([recs[i] for i in sub], {sc: [OP[sc][i] for i in sub]}, o.get("weights", (1, 1.4, 1.0)), o.get("smooth", 5))
            sim3._T.clear(); per.append(r[sc])
        print(tag, sc, "per-12-rec-set F1: mean %.3f sd %.3f min %.3f max %.3f" % (np.mean(per), np.std(per), np.min(per), np.max(per)), flush=True)
