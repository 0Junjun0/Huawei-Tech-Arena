"""Hard benchmark: whole-session inputs with the faults the hidden sessions are flagged for
(connection loss, wrong wristband placement, cut/early-ended sessions), feature caches per model family.

Calibration target (implied hidden-test macro-F1 from the leaderboard):
    bebe_time 0.60 < robust_nxe_p 0.675 ~ trio 0.68 < robust_lx_tuned 0.687 ~ Fares ver1 0.686 ~ RF 0.69
A benchmark that reproduces this order is one we can optimise on.
"""
import sys, json, pickle, pathlib, importlib.util
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from data import load_all
from feats import compute

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "lab" / "cache"
W = 10_000_000_000
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
SCEN = ["clean", "crop", "conn", "nanblk", "place", "mix"]


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


# ---------------------------------------------------------------- faults
def _gaps(d, rng, k, lo, hi):
    t = d.index.values.astype("datetime64[ns]").view(np.int64); keep = np.ones(len(d), bool)
    for _ in range(k):
        L = rng.uniform(lo, hi) * 1e9; a = t[0] + rng.uniform(0, max(1.0, (t[-1] - t[0]) - L))
        keep &= ~((t >= a) & (t < a + L))
    return d[keep]


def _nanblk(d, rng, k, lo, hi):
    d = d.copy(); t = d.index.values.astype("datetime64[ns]").view(np.int64)
    for _ in range(k):
        L = rng.uniform(lo, hi) * 1e9; a = t[0] + rng.uniform(0, max(1.0, (t[-1] - t[0]) - L))
        d.loc[(t >= a) & (t < a + L), COLS] = np.nan
    return d


def _crop(d, rng, s_hi=480, e_hi=360):
    t = d.index.values.astype("datetime64[ns]").view(np.int64)
    a = t[0] + rng.uniform(0, s_hi) * 1e9; b = t[-1] - rng.uniform(0, e_hi) * 1e9
    return d[(t >= a) & (t <= b)]


def _place(d, rng):
    d = d.copy(); n = len(d)
    d["BVP"] = d["BVP"] * np.exp(rng.uniform(np.log(0.3), np.log(3.0)))
    if rng.rand() < 0.6:
        d["BVP"] = d["BVP"] + rng.randn(n) * rng.uniform(5, 40)
    d["EDA"] = d["EDA"] * np.exp(rng.uniform(np.log(0.1), np.log(0.6)))
    d["Temperature"] = d["Temperature"] - rng.uniform(0, 3)
    if rng.rand() < 0.5:   # poor contact: most beats lost
        m = d["IBI_duration"].notna().to_numpy(); idx = np.flatnonzero(m); drop = rng.rand(len(idx)) < rng.uniform(0.4, 0.9)
        d.iloc[idx[drop], [COLS.index("IBI_dt"), COLS.index("IBI_duration")]] = np.nan
    return d


def corrupt(df, kind, rng):
    d = df[COLS]
    if kind == "clean": return d
    if kind == "crop": return _crop(d, rng)
    if kind == "conn": return _gaps(d, rng, rng.randint(1, 4), 30, 540)
    if kind == "nanblk": return _nanblk(d, rng, rng.randint(1, 4), 30, 480)
    if kind == "place": return _place(d, rng)
    if kind == "mix":
        d = _crop(d, rng, 300, 300)
        d = _gaps(d, rng, rng.randint(0, 3), 30, 400) if rng.rand() < 0.7 else _nanblk(d, rng, rng.randint(1, 3), 30, 300)
        return _place(d, rng) if rng.rand() < 0.5 else d
    raise ValueError(kind)


# ---------------------------------------------------------------- feature families (grid = floor(t0 / 10 s) .. floor(t_end / 10 s))
_RF = _FA = None
KW_NXE = json.load(open(ROOT / "submissions/robust_nxe_pg/config.json"))["feat_kw"]


def _grid(df):
    t = df.index.values.astype("datetime64[ns]").view(np.int64); w0 = t[0] // W; return t, w0, int(t[-1] // W - w0) + 1


def f_nxe(df): return compute(df, **KW_NXE)[1]
def f_time(df): return compute(df, use_time=True)[1]
def f_rf(df):
    global _RF
    _RF = _RF or _load("rf_sub", ROOT / "lab/teammates/rf/submission_model.py")
    return np.asarray(_RF._compute_window_features(df)[1], np.float32)
def f_fa1(df):
    global _FA
    _FA = _FA or _load("fa1_sub", ROOT / "lab/teammates/fares1/submission_model.py")
    x, rows, clock = _FA.window_features(df, "long"); t, w0, n = _grid(df)
    out = np.full((n, x.shape[1]), np.nan, np.float32); off = clock - w0; ok = (off >= 0) & (off < n); out[off[ok]] = x[ok]; return out


FAM = {"nxe": f_nxe, "time": f_time, "rf": f_rf, "fa1": f_fa1}


def split_feats(fn, df, gap_s=600):
    """like our submissions: parts separated by > 10 min are featurised on their own, then put on one grid."""
    t, w0, n = _grid(df)
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > gap_s * 1e9) + 1, len(t)]
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        X = np.asarray(fn(df.iloc[a:b]), np.float32); o = int(t[a] // W - w0); parts.append((o, X))
    F = parts[0][1].shape[1]; out = np.full((n, F), np.nan, np.float32)
    for o, X in parts:
        k = min(len(X), n - o); out[o:o + k] = X[:k]
    return out


def labels_on(df_in, rec):
    """true window labels of rec on the grid of df_in; -1 where the corrupted input has no rows."""
    t_in, w0, n = _grid(df_in); t0 = rec["df"].index.values.astype("datetime64[ns]").view(np.int64)[0] // W
    y = np.full(n, -1); src = rec["y"]; off = w0 - t0
    idx = np.arange(n) + off; ok = (idx >= 0) & (idx < len(src)); y[ok] = src[idx[ok]]
    has = np.zeros(n, bool); has[np.unique(t_in // W - w0)] = True; y[~has] = -1
    return y


def build(fam, kind, seed, recs=None):
    """list per recording of (X, y) for one fault scenario; cached."""
    p = CACHE / f"hard_{fam}_{kind}_{seed}.pkl"
    if p.exists(): return pickle.loads(p.read_bytes())
    recs = recs or load_all(); out = []
    for i, r in enumerate(recs):
        d = corrupt(r["df"], kind, np.random.RandomState(seed * 1000 + i))
        X = split_feats(FAM[fam], d); y = labels_on(d, r); n = min(len(X), len(y)); out.append((X[:n], y[:n]))
    p.write_bytes(pickle.dumps(out, protocol=4)); return out


if __name__ == "__main__":
    fams = sys.argv[1].split(","); seeds = [int(s) for s in sys.argv[2].split(",")]
    recs = load_all()
    for fam in fams:
        for s in seeds:
            for k in SCEN:
                build(fam, k, s, recs)
            print(fam, s, "done", flush=True)
