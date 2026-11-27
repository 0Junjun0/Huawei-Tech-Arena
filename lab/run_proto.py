"""Input-level protocol classifier (exercise vs stress session) on chunk aggregates of the first 17 base window features."""
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
def agg(X):
    B = X[:, :17]
    Bl = B.copy(); Bl[:, :4] = np.log1p(np.abs(Bl[:, :4])); Bl[:, 13] = np.log1p(np.abs(Bl[:, 13]))
    return np.concatenate([Bl.mean(0), Bl.std(0), np.percentile(Bl, 10, 0), np.percentile(Bl, 90, 0), [np.log1p(len(X))]])
A = {sc: [[agg(X) for a, b, w0, X in CF[sc][i]] for i in range(len(recs))] for sc in SC}
ex = np.array([r["protocol"] != "STRESS" for r in recs])
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
g = folds_subject(recs, 5, SEED)
mk = {"lgb": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2),
      "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))}
PE = {}
for name, f in mk.items():
    PE[name] = {sc: [None] * len(recs) for sc in EVAL_SC}
    for k in range(5):
        tr = np.flatnonzero(g != k)
        Z = np.array([z for sc in TRAIN_SC for i in tr for z in A[sc][i]])
        y = np.array([ex[i] for sc in TRAIN_SC for i in tr for z in A[sc][i]])
        m = f().fit(Z, y)
        for i in np.flatnonzero(g == k):
            for sc in EVAL_SC:
                PE[name][sc][i] = m.predict_proba(np.array(A[sc][i]))[:, 1]
    print(name, " ".join(f"{sc}:{np.mean(np.concatenate([(PE[name][sc][i] > .5) != ex[i] for i in range(len(recs))])):.3f}" for sc in EVAL_SC), flush=True)
pickle.dump(PE, open(CACHE / f"proto_pe{SEED}.pkl", "wb"))
OP = pickle.loads((CACHE / f"ocp_x_nxe_sw07{'_s1' if SEED else ''}_subject{SEED}.pkl").read_bytes())
sim3._CFREF.update(CF)
print("base", fmt(score_probs_fast(recs, OP, (1, 1.4, 1.0), 5)))
for name in mk:
    for gam in (0.5, 1, 2, 4):
        OP2 = {sc: [[(a, b, P * np.array([1, (1 - pe) ** gam, pe ** gam])) for (a, b, P), pe in zip(OP[sc][i], PE[name][sc][i])] for i in range(len(recs))] for sc in EVAL_SC}
        print(name, gam, fmt(score_probs_fast(recs, OP2, (1, 1.4, 1.0), 5)), flush=True)
