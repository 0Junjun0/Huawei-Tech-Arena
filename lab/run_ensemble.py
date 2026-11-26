"""Ours (nxe, chunk-aug LGBM) vs teammate RF vs Fares ver1 (HGB) vs their probability average.
Each family retrained per subject fold with its own recipe; scored on held-out chunks."""
import sys, json
sys.path.insert(0, "lab"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, lightgbm as lgb, importlib.util
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from data import load_all
from feats import compute
from holdout import smooth_probs
import sim3
from sim3 import chunk_feats, oof_chunk_probs

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
RF = load("rf_sub", "lab/teammates/rf/submission_model.py")
FA = load("fares1_sub", "lab/teammates/fares1/submission_model.py")

recs = load_all()
kw = json.load(open("submissions/robust_nxe_pg/config.json"))["feat_kw"]
EVAL = ["full", "chunk600", "chunk300", "chunk180", "rand300"]
TRAIN_AUG = ["full", "chunk600", "chunk300", "chunk180", "chunk120"]
SW = {"full": 1.0, "chunk600": 0.1, "chunk300": 0.1, "chunk180": 0.1, "chunk120": 0.1}
SC = sorted(set(EVAL) | set(TRAIN_AUG))

def rf_feats(df):
    return RF._compute_window_features(df)[1]
def fa_feats(df):
    x, rows, clock = FA.window_features(df, "long")
    t = df.index.values.astype("datetime64[ns]").view(np.int64)
    w0 = t[0] // 10**10
    n = int(t[-1] // 10**10 - w0) + 1
    out = np.zeros((n, x.shape[1]), np.float32)
    off = clock - w0  # clock is the absolute 10 s window number
    ok = (off >= 0) & (off < n)
    out[off[ok]] = x[ok]
    return out

fam = {
    "ours": dict(CF=chunk_feats(recs, "e_nxe", lambda d: compute(d, **kw)[1], SC), train=TRAIN_AUG, sw=SW,
                 mk=lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
                                               subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8),
                 smooth=5, w=(1, 1.4, 1)),
    "rf": dict(CF=chunk_feats(recs, "e_rf", rf_feats, EVAL), train=["full"], sw=None,
               mk=lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
               smooth=3, w=(0.95, 1.8, 0.85)),
    "fares1": dict(CF=chunk_feats(recs, "e_fa1", fa_feats, EVAL), train=["full"], sw=None,
                   mk=lambda: HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, learning_rate=0.075, l2_regularization=5, class_weight="balanced", early_stopping=False, random_state=42),
                   smooth=7, w=(1, 1.4, 1)),
}
print("features done", flush=True)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OP = {}
for name, f in fam.items():
    OP[name] = oof_chunk_probs(recs, f["CF"], f["mk"], train_sc=f["train"], eval_sc=EVAL, seed=seed, sc_weight=f["sw"], cache_name=f"ens_{name}")
    print(name, "oof done", flush=True)

def post(name, P):
    f = fam[name]; Q = smooth_probs(P, f["smooth"]) * np.asarray(f["w"])
    return Q / Q.sum(1, keepdims=True)

def score(members, wts=None):
    wts = wts or [1.0] * len(members); res = {}
    for sc in EVAL:
        Y, Pp = [], []
        for i in range(len(recs)):
            yt = recs[i]["y"]; pw = np.full(len(yt), -1)
            for j, (_, _, w0, _) in enumerate(fam["ours"]["CF"][sc][i]):
                Q = sum(wt * post(m, OP[m][sc][i][j][2]) for m, wt in zip(members, wts))
                q = Q.argmax(1); e = min(len(yt), w0 + len(q)); pw[w0:e] = q[:e - w0]
            m = (yt >= 0) & (pw >= 0); Y.append(yt[m]); Pp.append(pw[m])
        res[sc] = f1_score(np.concatenate(Y), np.concatenate(Pp), average="macro")
    res["blend"] = float(np.mean([res[s] for s in EVAL]))
    return " ".join(f"{k}={v:.3f}" for k, v in res.items())

for members in (["ours"], ["rf"], ["fares1"], ["ours", "fares1"], ["ours", "rf"], ["rf", "fares1"], ["ours", "rf", "fares1"]):
    print(f"{'+'.join(members):18s}", score(members), flush=True)
print("ours x2 + rf + fares1 ", score(["ours", "rf", "fares1"], [2, 1, 1]), flush=True)
print("ours + fares1 x2      ", score(["ours", "fares1"], [1, 2]), flush=True)
