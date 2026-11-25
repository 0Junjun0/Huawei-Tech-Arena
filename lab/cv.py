"""Subject-grouped CV harness. Scores pooled 10 s windows with macro-F1 like the competition."""
import sys, time, pathlib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import load_all, LABELS  # noqa: E402


def build(recs, featfn):
    out = []
    t0 = time.perf_counter()
    for r in recs:
        ws, X, names = featfn(r["df"])
        y = r["y"]
        n = min(len(y), len(X))
        out.append(dict(subject=r["subject"], protocol=r["protocol"], X=X[:n], y=y[:n], names=names))
    print(f"features: {out[0]['X'].shape[1]} cols, {time.perf_counter()-t0:.1f}s for {len(recs)} recs")
    return out


def smooth_probs(P, k):
    """centered moving average of class probabilities over k windows."""
    if k <= 1:
        return P
    h = k // 2
    c = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])
    n = len(P)
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def oof(items, make_model, n_splits=5, seed=0):
    """returns per-recording OOF probability arrays (aligned with items)."""
    subj = np.array([it["subject"] for it in items])
    usub = np.unique(subj)
    rng = np.random.RandomState(seed)
    perm = {s: i for i, s in enumerate(rng.permutation(usub))}
    groups = np.array([perm[s] % n_splits for s in subj])
    probs = [None] * len(items)
    for k in range(n_splits):
        tr = [i for i in range(len(items)) if groups[i] != k]
        va = [i for i in range(len(items)) if groups[i] == k]
        Xtr = np.vstack([items[i]["X"] for i in tr])
        ytr = np.concatenate([items[i]["y"] for i in tr])
        m = ytr >= 0
        model = make_model()
        model.fit(Xtr[m], ytr[m])
        for i in va:
            probs[i] = model.predict_proba(items[i]["X"])
    return probs


def score(items, probs, weights=(1, 1, 1), smooth=1, verbose=False):
    ys, ps = [], []
    w = np.asarray(weights)
    for it, P in zip(items, probs):
        Q = smooth_probs(P, smooth) * w
        pred = Q.argmax(1)
        m = it["y"] >= 0
        ys.append(it["y"][m]); ps.append(pred[m])
    y = np.concatenate(ys); p = np.concatenate(ps)
    f1 = f1_score(y, p, average="macro")
    if verbose:
        print(f"macro-F1 {f1:.4f}  per-class {np.round(f1_score(y, p, average=None), 4)}")
        print(confusion_matrix(y, p))
    return f1


def tune(items, probs, smooths=(1, 3, 5, 7, 9, 13, 17, 25)):
    best = (0, None, None)
    for s in smooths:
        for ws in np.arange(0.6, 3.01, 0.1):
            for wa in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25):
                f = score(items, probs, (1.0, ws, wa), s)
                if f > best[0]:
                    best = (f, s, (1.0, round(float(ws), 2), wa))
    return best
