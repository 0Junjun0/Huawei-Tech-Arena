"""Re-score models under different ways the grader might feed data to run().

Models are trained per fold on full training recordings; held-out recordings are then
fed in pieces (scenario) and scored per 10 s window on the full recording, like the grader.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import f1_score
from cv import smooth_probs

W = 10_000_000_000


def chunks(n_rows, t_ns, scenario, rng):
    """yield (start,end) row slices for a scenario."""
    if scenario == "full":
        yield 0, n_rows; return
    if scenario.startswith("chunk"):           # fixed-length chunks, e.g. chunk60 = 60 s
        L = int(scenario[5:]) * 1_000_000_000
        edges = np.searchsorted(t_ns, np.arange(t_ns[0], t_ns[-1] + L, L))
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a: yield a, b
        return
    if scenario.startswith("rand"):            # random chunk lengths between 30 s and N s
        hi = int(scenario[4:])
        a = 0
        while a < n_rows:
            L = rng.uniform(30, hi) * 1e9
            b = int(np.searchsorted(t_ns, t_ns[a] + L))
            b = max(b, a + 1)
            yield a, b; a = b
        return
    if scenario == "skipstart":                # session missing its first 0-5 minutes
        s = int(np.searchsorted(t_ns, t_ns[0] + rng.uniform(60, 300) * 1e9))
        yield s, n_rows; return
    raise ValueError(scenario)


def predict_rows(df, featfn, model, weights, smooth):
    ws, X, _ = featfn(df)
    P = model.predict_proba(X)
    pred = (smooth_probs(P, smooth) * np.asarray(weights)).argmax(1)
    t = df.index.values.astype("datetime64[ns]").view(np.int64)
    wid = t // W - t[0] // W
    wid = np.clip(wid, 0, len(pred) - 1)
    return pred[wid]


def window_majority(t_ns, row_pred, mask=None):
    wid = t_ns // W - t_ns[0] // W
    n = int(wid[-1]) + 1
    cnt = np.zeros((n, 3))
    m = row_pred >= 0 if mask is None else mask
    np.add.at(cnt, (wid[m], row_pred[m]), 1)
    out = cnt.argmax(1); out[cnt.sum(1) == 0] = -1
    return out


def evaluate(recs, featfn, make_model, scenarios, weights=(1, 1, 1), smooth=3, n_splits=5, seed=0):
    subj = np.array([r["subject"] for r in recs])
    rng = np.random.RandomState(seed)
    perm = {s: i for i, s in enumerate(rng.permutation(np.unique(subj)))}
    groups = np.array([perm[s] % n_splits for s in subj])
    feats = [featfn(r["df"]) for r in recs]
    ys = {s: [] for s in scenarios}; ps = {s: [] for s in scenarios}
    for k in range(n_splits):
        tr = np.flatnonzero(groups != k)
        X = np.vstack([feats[i][1][:len(recs[i]["y"])] for i in tr])
        y = np.concatenate([recs[i]["y"][:len(feats[i][1])] for i in tr])
        m = y >= 0
        model = make_model(); model.fit(X[m], y[m])
        for i in np.flatnonzero(groups == k):
            df = recs[i]["df"]
            t = df.index.values.astype("datetime64[ns]").view(np.int64)
            for sc in scenarios:
                r = np.random.RandomState(1000 + i)
                row = np.full(len(df), -1)
                for a, b in chunks(len(df), t, sc, r):
                    row[a:b] = predict_rows(df.iloc[a:b], featfn, model, weights, smooth)
                pw = window_majority(t, row)
                yt = recs[i]["y"]; n = min(len(yt), len(pw))
                keep = (yt[:n] >= 0) & (pw[:n] >= 0)
                ys[sc].append(yt[:n][keep]); ps[sc].append(pw[:n][keep])
    return {sc: f1_score(np.concatenate(ys[sc]), np.concatenate(ps[sc]), average="macro") for sc in scenarios}
