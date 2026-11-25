"""Scenario-blend harness with chunk-augmented training.

For each recording and scenario the input is cut into chunks (like the grader might feed run()),
features are computed per chunk, and chunk windows are mapped back to the recording's global 10 s grid.
Training can use features from any set of scenarios (augmentation); evaluation is row-level like the
grader (row predictions -> 10 s majority vs ground truth).
"""
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import f1_score
from sim import chunks
from holdout import smooth_probs, vote3, folds_subject, folds_recording

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
W = 10_000_000_000
EVAL_SC = ["full", "skipstart", "chunk600", "chunk300", "chunk180", "chunk120", "rand300"]
TRAIN_SC = ["full", "chunk600", "chunk300", "chunk180", "chunk120"]


def chunk_feats(recs, key, featfn, scenarios):
    """dict sc -> list per rec of list of (a, b, w_off, X)."""
    out = {}
    for sc in scenarios:
        p = CACHE / f"cf_{key}_{sc}.pkl"
        if p.exists():
            out[sc] = pickle.loads(p.read_bytes()); continue
        lst = []
        for i, r in enumerate(recs):
            df = r["df"]
            base_sc = sc
            if sc.startswith("gain_"):  # augmentation: random sensor gains (+ BVP noise) per recording
                base_sc = sc[5:]
                gr = np.random.RandomState(500 + i)
                df = df.copy()
                df["BVP"] = df["BVP"] * np.exp(gr.uniform(np.log(0.33), np.log(3.0)))
                if gr.rand() < 0.3:
                    df["BVP"] = df["BVP"] + gr.randn(len(df)) * gr.uniform(5, 40)
                df["EDA"] = df["EDA"] * np.exp(gr.uniform(np.log(0.2), np.log(5.0)))
            t = df.index.values.astype("datetime64[ns]").view(np.int64)
            g0 = t[0] // W
            rng = np.random.RandomState(1000 + i)
            parts = []
            for a, b in chunks(len(df), t, base_sc, rng):
                X = np.asarray(featfn(df.iloc[a:b]), dtype=np.float32)
                parts.append((a, b, int(t[a] // W - g0), X))
            lst.append(parts)
        p.write_bytes(pickle.dumps(lst, protocol=4))
        out[sc] = lst
    return out


def train_xy(recs, CF, idx, train_sc, sc_weight=None):
    Xs, ys, ws = [], [], []
    for sc in train_sc:
        for i in idx:
            y = recs[i]["y"]
            for a, b, w0, X in CF[sc][i]:
                yy = y[w0:w0 + len(X)]
                n = len(yy)
                Xs.append(X[:n]); ys.append(yy); ws.append(np.full(n, (sc_weight or {}).get(sc, 1.0)))
    X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws); m = y >= 0
    if sc_weight is not None:
        return X[m], y[m], w[m]
    return X[m], y[m]


def row_pred(recs, i, parts, model, weights, smooth, vote=False, cache=None):
    df = recs[i]["df"]
    t = df.index.values.astype("datetime64[ns]").view(np.int64)
    row = np.full(len(df), -1)
    for a, b, w0, X in parts:
        P = model.predict_proba(X)
        if vote:
            pw = vote3((P * np.asarray(weights)).argmax(1))
        else:
            pw = (smooth_probs(P, smooth) * np.asarray(weights)).argmax(1)
        tt = t[a:b]
        wid = np.clip(tt // W - tt[0] // W, 0, len(pw) - 1)
        row[a:b] = pw[wid]
    return row


def win_major(t, v):
    wid = t // W - t[0] // W
    n = int(wid[-1]) + 1
    c = np.zeros((n, 3)); m = v >= 0
    np.add.at(c, (wid[m], v[m]), 1)
    o = c.argmax(1); o[c.sum(1) == 0] = -1
    return o


def evaluate(recs, CF, make_model, train_sc=("full",), eval_sc=EVAL_SC, weights=(1, 1, 1), smooth=3, vote=False,
             seed=0, split="subject", fit_kw=None, return_probs=False):
    g = folds_subject(recs, 5, seed) if split == "subject" else folds_recording(recs, 7, seed)
    Y = {s: [] for s in eval_sc}; Pp = {s: [] for s in eval_sc}
    for k in np.unique(g):
        tr = np.flatnonzero(g != k)
        X, y = train_xy(recs, CF, tr, train_sc)
        model = make_model(); model.fit(X, y, **(fit_kw or {}))
        for i in np.flatnonzero(g == k):
            t = recs[i]["df"].index.values.astype("datetime64[ns]").view(np.int64)
            yt = recs[i]["y"]
            for sc in eval_sc:
                row = row_pred(recs, i, CF[sc][i], model, weights, smooth, vote)
                pw = win_major(t, row)
                n = min(len(pw), len(yt)); m = (yt[:n] >= 0) & (pw[:n] >= 0)
                Y[sc].append(yt[:n][m]); Pp[sc].append(pw[:n][m])
    res = {s: f1_score(np.concatenate(Y[s]), np.concatenate(Pp[s]), average="macro") for s in eval_sc}
    res["blend"] = float(np.mean([res[s] for s in eval_sc]))
    return res


def fmt(res):
    return " ".join(f"{k}={v:.3f}" for k, v in res.items())


def oof_chunk_probs(recs, CF, make_model, train_sc=("full",), eval_sc=EVAL_SC, seed=0, split="subject", fit_kw=None,
                    cache_name=None, sc_weight=None):
    """per scenario, per recording: list of (a, b, P) window probabilities of held-out chunks."""
    if cache_name:
        p = CACHE / f"ocp_{cache_name}_{split}{seed}.pkl"
        if p.exists():
            return pickle.loads(p.read_bytes())
    g = folds_subject(recs, 5, seed) if split == "subject" else folds_recording(recs, 7, seed)
    out = {s: [None] * len(recs) for s in eval_sc}
    for k in np.unique(g):
        tr = np.flatnonzero(g != k)
        if sc_weight is not None:
            X, y, sw = train_xy(recs, CF, tr, train_sc, sc_weight)
            model = make_model(); model.fit(X, y, sample_weight=sw)
        else:
            X, y = train_xy(recs, CF, tr, train_sc)
            model = make_model(); model.fit(X, y, **(fit_kw or {}))
        for i in np.flatnonzero(g == k):
            for sc in eval_sc:
                out[sc][i] = [(a, b, model.predict_proba(X)) for a, b, w0, X in CF[sc][i]]
    if cache_name:
        p.write_bytes(pickle.dumps(out, protocol=4))
    return out


_T = {}


def _times(recs, i):
    if i not in _T:
        _T[i] = recs[i]["df"].index.values.astype("datetime64[ns]").view(np.int64)
    return _T[i]


def score_probs(recs, OP, weights=(1, 1, 1), smooth=3, eval_sc=None, vote=False, post=None):
    eval_sc = eval_sc or list(OP)
    res = {}
    w = np.asarray(weights)
    for sc in eval_sc:
        Y, Pp = [], []
        for i, parts in enumerate(OP[sc]):
            t = _times(recs, i); yt = recs[i]["y"]
            row = np.full(len(t), -1)
            for a, b, P in parts:
                if post is not None:
                    P = post(P)
                pw = vote3((P * w).argmax(1)) if vote else (smooth_probs(P, smooth) * w).argmax(1)
                tt = t[a:b]
                row[a:b] = pw[np.clip(tt // W - tt[0] // W, 0, len(pw) - 1)]
            pw = win_major(t, row)
            n = min(len(pw), len(yt)); m = (yt[:n] >= 0) & (pw[:n] >= 0)
            Y.append(yt[:n][m]); Pp.append(pw[:n][m])
        res[sc] = f1_score(np.concatenate(Y), np.concatenate(Pp), average="macro")
    res["blend"] = float(np.mean([res[s] for s in eval_sc]))
    return res


def score_probs_fast(recs, OP, weights=(1, 1, 1), smooth=3, eval_sc=None, post=None):
    """window-level approximation of score_probs (chunk-boundary windows: later chunk wins). For tuning."""
    eval_sc = eval_sc or list(OP)
    w = np.asarray(weights); res = {}
    for sc in eval_sc:
        Y, Pp = [], []
        for i, parts in enumerate(OP[sc]):
            yt = recs[i]["y"]; pw = np.full(len(yt), -1)
            for (a, b, P), (_, _, w0, _) in zip(parts, _CFREF[sc][i]):
                if post is not None:
                    P = post(P)
                q = (smooth_probs(P, smooth) * w).argmax(1)
                e = min(len(yt), w0 + len(q)); pw[w0:e] = q[:e - w0]
            m = (yt >= 0) & (pw >= 0); Y.append(yt[m]); Pp.append(pw[m])
        res[sc] = f1_score(np.concatenate(Y), np.concatenate(Pp), average="macro")
    res["blend"] = float(np.mean([res[s] for s in eval_sc]))
    return res


_CFREF = {}


def fold_score(recs, OP, fold, weights, smooth, post=None, seed=0):
    """score_probs restricted to the recordings of one subject fold (to compare with e2e.py)."""
    g = folds_subject(recs, 5, seed)
    sub = [i for i in range(len(recs)) if g[i] == fold]
    R = [recs[i] for i in sub]
    _T.clear()
    r = score_probs(R, {sc: [OP[sc][i] for i in sub] for sc in OP}, weights, smooth, post=post)
    _T.clear()
    return r
