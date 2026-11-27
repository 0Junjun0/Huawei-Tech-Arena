"""Protocol classifier variants: richer aggregates. usage: run_proto2.py <variant>"""
import sys, pickle
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data import load_all
import sim3
from sim3 import *
recs = load_all()
SC = sorted(set(EVAL_SC) | set(TRAIN_SC))
CF = chunk_feats(recs, "nxe", None, SC)
var = sys.argv[1]
def agg17(X):
    B = X[:, :17].astype(np.float64)
    B[:, :4] = np.log1p(np.abs(B[:, :4])); B[:, 13] = np.log1p(np.abs(B[:, 13]))
    return np.concatenate([B.mean(0), B.std(0), np.percentile(B, 10, 0), np.percentile(B, 90, 0), [np.log1p(len(X))]])
def aggall(X):
    Xl = np.sign(X) * np.log1p(np.abs(X.astype(np.float64)))
    return np.concatenate([Xl.mean(0), np.percentile(Xl, 10, 0), np.percentile(Xl, 90, 0), [np.log1p(len(X))]])
agg = agg17 if var.startswith("a17") else aggall
A = {sc: [[agg(X) for a, b, w0, X in CF[sc][i]] for i in range(len(recs))] for sc in SC}
ex = np.array([r["protocol"] != "STRESS" for r in recs])
mk = {"a17_lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
      "a17_lr1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
      "a17_lr03": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000)),
      "all_lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=5000)),
      "all_lr01": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=5000)),
      "all_lgb": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, verbose=-1, n_jobs=1)}[var]
for seed in (0, 1):
    g = folds_subject(recs, 5, seed)
    PE = {sc: [None] * len(recs) for sc in EVAL_SC}
    for k in range(5):
        tr = np.flatnonzero(g != k)
        Z = np.array([z for sc in TRAIN_SC for i in tr for z in A[sc][i]]); y = np.array([ex[i] for sc in TRAIN_SC for i in tr for z in A[sc][i]])
        m = mk().fit(Z, y)
        for i in np.flatnonzero(g == k):
            for sc in EVAL_SC: PE[sc][i] = m.predict_proba(np.array(A[sc][i]))[:, 1]
    OP = pickle.loads((CACHE / f"ocp_x_nxe_sw07{'_s1' if seed else ''}_subject{seed}.pkl").read_bytes())
    sim3._CFREF.update(CF)
    err = " ".join(f"{sc}:{np.mean(np.concatenate([(PE[sc][i] > .5) != ex[i] for i in range(len(recs))])):.3f}" for sc in EVAL_SC)
    OP2 = {sc: [[(a, b, P * np.array([1, 1 - pe, pe])) for (a, b, P), pe in zip(OP[sc][i], PE[sc][i])] for i in range(len(recs))] for sc in EVAL_SC}
    print(var, "seed", seed, err, "|", fmt(score_probs_fast(recs, OP2, (1, 1.4, 1.0), 5)), flush=True)
