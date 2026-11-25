import sys, pickle
sys.path.insert(0, "lab")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import confusion_matrix, f1_score
from data import load_all, files
from feats import compute
from cv import build, oof, smooth_probs
recs = load_all()
items = build(recs, lambda df: compute(df, use_time=True))
mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0, verbose=-1, n_jobs=8)
P = oof(items, mk, seed=0)
pickle.dump(P, open("lab/cache/oof_time_s0.pkl", "wb"))
# per-window phase names
rows = []
for f, it, p in zip(files(), items, P):
    df = pd.read_parquet(f, columns=["Phase"])
    t = df.index.values.astype("datetime64[ns]").view(np.int64); wid = t // 10**10 - t[0] // 10**10
    ph = pd.Series(df["Phase"].values).groupby(wid).agg(lambda s: s.mode().iloc[0]).reindex(range(len(it["y"]))).values
    pred = smooth_probs(p, 3).argmax(1)
    for w in range(len(it["y"])):
        if it["y"][w] >= 0:
            rows.append((it["protocol"], it["subject"], ph[w], it["y"][w], pred[w], w))
d = pd.DataFrame(rows, columns=["proto", "subj", "phase", "y", "pred", "w"])
L = np.array(["Rest", "Stress", "Act"])
print("overall F1", round(f1_score(d.y, d.pred, average="macro"), 4))
for pr, g in d.groupby("proto"):
    print(f"\n{pr}: n={len(g)}  true->pred\n", pd.crosstab(L[g.y], L[g.pred]))
d["err"] = d.y != d.pred
ph = d.assign(phase=d.phase.str.replace(r"\d+(-\d+)? rpm", "rpm", regex=True)).groupby(["proto", "phase"]).agg(n=("err", "size"), err_rate=("err", "mean")).sort_values("n", ascending=False)
print("\nerror rate by phase:\n", ph[ph.n > 50].round(3).to_string())
print("\nworst subjects:\n", d.groupby("subj").err.mean().sort_values(ascending=False).head(8).round(3).to_string())
