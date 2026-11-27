"""Calibrate a 'mixedness' statistic: does one input contain both Stress-like and Activity-like stretches?
Fold-trained nxe model (ungated), real whole sessions vs simulated composites of two/three protocols."""
import sys, json, pickle, pathlib
sys.path.insert(0, "lab")
import numpy as np, pandas as pd, lightgbm as lgb
from data import load_all
from feats import compute
from holdout import folds_subject, smooth_probs
recs = load_all(); C = pathlib.Path("lab/cache")
kw = json.load(open("submissions/robust_nxe_pg/config.json"))["feat_kw"]
X = [p[0][3] for p in pickle.loads((C / "cf_e_nxe_full.pkl").read_bytes())]
g = folds_subject(recs, 5, 0)
def stat(P):
    Q = smooth_probs(P, 5) * np.array([1, 1.4, 1]); a = Q.argmax(1)
    fS, fA = (a == 1).mean(), (a == 2).mean()
    return min(fS, fA), min(P[:, 1].mean(), P[:, 2].mean())
real, comp = [], []
rng = np.random.RandomState(0)
for k in np.unique(g):
    tr = np.flatnonzero(g != k)
    Xt = np.vstack([X[i][:len(recs[i]["y"])] for i in tr]); yt = np.concatenate([recs[i]["y"][:len(X[i])] for i in tr]); m = yt >= 0
    mod = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8).fit(Xt[m], yt[m])
    te = np.flatnonzero(g == k)
    for i in te:
        real.append((recs[i]["protocol"],) + stat(mod.predict_proba(X[i])))
    st = [i for i in te if recs[i]["protocol"] == "STRESS"]; ex = [i for i in te if recs[i]["protocol"] != "STRESS"]
    for i in st:
        for j in rng.permutation(ex)[:2]:
            for frac in (0.5, 0.25):   # share of the exercise session kept
                a = recs[i]["df"]; b = recs[j]["df"]
                na, nb = int(len(a) * 0.5), int(len(b) * frac)
                a = a.iloc[rng.randint(0, len(a) - na + 1):][:na]; b = b.iloc[rng.randint(0, len(b) - nb + 1):][:nb]
                parts = [a, b] if rng.rand() < 0.5 else [b, a]
                p1 = parts[1].copy(); p1.index = p1.index - p1.index[0] + parts[0].index[-1] + pd.Timedelta(milliseconds=16)
                big = pd.concat([parts[0], p1])
                comp.append((frac,) + stat(mod.predict_proba(compute(big, **kw)[1])))
R = pd.DataFrame(real, columns=["proto", "hard", "soft"]); Cm = pd.DataFrame(comp, columns=["frac", "hard", "soft"])
print("REAL sessions:"); print(R.groupby("proto")[["hard", "soft"]].describe(percentiles=[.5, .9, .95]).T.round(3).to_string())
print("COMPOSITES:"); print(Cm.groupby("frac")[["hard", "soft"]].describe(percentiles=[.05, .1, .5]).T.round(3).to_string())
print("real mean hard %.3f soft %.3f | composite mean hard %.3f soft %.3f" % (R.hard.mean(), R.soft.mean(), Cm.hard.mean(), Cm.soft.mean()))
