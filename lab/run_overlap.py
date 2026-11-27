"""Does subject overlap (hidden recordings' subjects are in train under other protocols) change the ranking?
Same models under subject-grouped folds vs recording-level folds, whole sessions."""
import sys, pickle, pathlib
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from data import load_all
from holdout import folds_subject, folds_recording, smooth_probs
recs = load_all(); C = pathlib.Path("lab/cache")
full = lambda k: [p[0][3] for p in pickle.loads((C / f"cf_{k}_full.pkl").read_bytes())]
M = {
 "rf": (full("e_rf"), lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=4))]), 3, (0.95, 1.8, 0.85)),
 "fares1": (full("e_fa1"), lambda: HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, learning_rate=0.075, l2_regularization=5, class_weight="balanced", early_stopping=False, random_state=42), 7, (1, 1.4, 1)),
 "nxe": (full("e_nxe"), lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=4), 5, (1, 1.4, 1)),
}
def run(name, g):
    X, mk, sm, w = M[name]; Y, P = [], []; per = {}
    for k in np.unique(g):
        tr = np.flatnonzero(g != k)
        Xt = np.vstack([X[i][:len(recs[i]["y"])] for i in tr]); yt = np.concatenate([recs[i]["y"][:len(X[i])] for i in tr]); m = yt >= 0
        mod = mk(); mod.fit(Xt[m], yt[m])
        for i in np.flatnonzero(g == k):
            y = recs[i]["y"]; n = min(len(y), len(X[i]))
            p = (smooth_probs(mod.predict_proba(X[i][:n]), sm) * np.asarray(w)).argmax(1)
            mm = y[:n] >= 0; Y.append(y[:n][mm]); P.append(p[mm]); per[i] = (y[:n][mm], p[mm])
    return f1_score(np.concatenate(Y), np.concatenate(P), average="macro"), per
for name in sys.argv[1:] or list(M):
    s = [run(name, folds_subject(recs, 5, sd))[0] for sd in (0, 1)]
    r = [run(name, folds_recording(recs, 7, sd))[0] for sd in (0, 1)]
    print(f"{name:8s} subject-folds {np.mean(s):.3f} {np.round(s,3)}   recording-folds {np.mean(r):.3f} {np.round(r,3)}", flush=True)
