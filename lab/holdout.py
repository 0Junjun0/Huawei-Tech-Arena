"""Recording-level holdout simulation of the hidden test set.

subject-info.csv lists 12 sessions that are NOT in the public data:
  STRESS: S02 S04 f07 f08 f14 | AEROBIC: S11 S13 f06 | ANAEROBIC: S01 S15 S16 f10
The hidden test set is most likely these 12 recordings. 11 of the 12 subjects appear in the
training data under another protocol, so the right simulation holds out *recordings* (not subjects),
~12 at a time with a similar protocol mix, and scores pooled macro-F1 per held-out set.
"""
import sys, pathlib, pickle, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import f1_score
from data import load_all

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
HIDDEN = {"STRESS": ["S02", "S04", "f07", "f08", "f14"], "AEROBIC": ["S11", "S13", "f06"],
          "ANAEROBIC": ["S01", "S15", "S16", "f10"]}


def feats_cached(recs, key, featfn):
    p = CACHE / f"feat_{key}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = []
    for r in recs:
        X = featfn(r["df"])
        n = min(len(X), len(r["y"]))
        out.append(np.asarray(X[:n], dtype=np.float32))
    p.write_bytes(pickle.dumps(out, protocol=4))
    return out


def folds_recording(recs, n_folds=7, seed=0):
    """protocol-stratified random assignment of recordings to folds (~12 recordings per fold)."""
    rng = np.random.RandomState(seed)
    g = np.zeros(len(recs), int)
    off = rng.randint(n_folds)
    for proto in ("STRESS", "AEROBIC", "ANAEROBIC"):
        idx = [i for i, r in enumerate(recs) if r["protocol"] == proto]
        idx = list(rng.permutation(idx))
        for j, i in enumerate(idx):
            g[i] = (j + off) % n_folds
        off += len(idx)
    return g


def folds_subject(recs, n_folds=5, seed=0):
    subj = np.array([r["subject"] for r in recs])
    rng = np.random.RandomState(seed)
    perm = {s: i for i, s in enumerate(rng.permutation(np.unique(subj)))}
    return np.array([perm[s] % n_folds for s in subj])


def oof_probs(recs, X, make_model, groups, sample_weight_fn=None):
    probs = [None] * len(recs)
    for k in np.unique(groups):
        tr = np.flatnonzero(groups != k)
        Xt = np.vstack([X[i] for i in tr]); yt = np.concatenate([recs[i]["y"][:len(X[i])] for i in tr])
        m = yt >= 0
        model = make_model()
        if sample_weight_fn is not None:
            model.fit(Xt[m], yt[m], **sample_weight_fn(yt[m]))
        else:
            model.fit(Xt[m], yt[m])
        cls = list(model.classes_)
        for i in np.flatnonzero(groups == k):
            P = model.predict_proba(X[i])
            Q = np.zeros((len(P), 3)); Q[:, cls] = P
            probs[i] = Q
    return probs


def smooth_probs(P, k):
    if k <= 1:
        return P
    h = k // 2; n = len(P)
    c = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def vote3(pred):
    """the RF submission's post-processing: 3-window majority vote, ties -> own label."""
    n = len(pred); out = pred.copy()
    for i in range(n):
        ch = pred[max(0, i - 1):min(n, i + 2)]
        c = np.bincount(ch, minlength=3)
        best = c.max()
        out[i] = pred[i] if c[pred[i]] == best else int(np.argmax(c))
    return out


def predict(P, weights=(1, 1, 1), smooth=3, vote=False):
    if vote:
        return vote3((P * np.asarray(weights)).argmax(1))
    return (smooth_probs(P, smooth) * np.asarray(weights)).argmax(1)


def score_sets(recs, preds, groups):
    """macro-F1 pooled per held-out set (mean, std) + pooled over all recordings."""
    per = []
    for k in np.unique(groups):
        idx = np.flatnonzero(groups == k)
        y = np.concatenate([recs[i]["y"][:len(preds[i])] for i in idx]); p = np.concatenate([preds[i] for i in idx])
        m = y >= 0
        per.append(f1_score(y[m], p[m], average="macro"))
    y = np.concatenate([r["y"][:len(p)] for r, p in zip(recs, preds)]); p = np.concatenate(preds)
    m = y >= 0
    return float(np.mean(per)), float(np.std(per)), float(f1_score(y[m], p[m], average="macro")), per
