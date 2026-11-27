"""Load recordings + 10 s window ground truth exactly as the competition scores it."""
import pathlib, pickle
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent / "TechArena-Finland-2026" / "data"
CACHE = pathlib.Path(__file__).resolve().parent / "cache"
LABELS = ["Rest", "Stress", "Activity"]
STRESS = {"stroop", "tmct", "real opinion", "opposite opinion", "subtraction"}
ACT = {"warm-up", "sprint", "cooldown"}
REST = {"baseline", "rest"}
SENSORS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def phase_to_id(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return -1
    s = str(p).strip().lower()
    if s in STRESS:
        return 1
    if s in REST:
        return 0
    if s in ACT or "rpm" in s:
        return 2
    return -1


def window_labels(df):
    """majority scored state per 10 s window (floor-aligned); -1 if window has no scored rows."""
    t = df.index.values.astype("datetime64[ns]").view(np.int64)
    W = 10_000_000_000
    wid = t // W - t[0] // W
    n = int(wid[-1]) + 1
    lab = df["Phase"].map(phase_to_id).to_numpy()
    m = lab >= 0
    counts = np.zeros((n, 3))
    np.add.at(counts, (wid[m], lab[m]), 1)
    y = counts.argmax(1)
    y[counts.sum(1) == 0] = -1
    return y


def files():
    return sorted(ROOT.rglob("*.parquet"))


def load_all():
    """list of dicts: subject, protocol, df (sensors only), y (window labels). Cached as pickle."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / "recs.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    recs = []
    for f in files():
        df = pd.read_parquet(f)
        df = df.sort_index()
        recs.append(dict(subject=f.stem, protocol=f.parent.name, y=window_labels(df), df=df[SENSORS]))
    p.write_bytes(pickle.dumps(recs, protocol=4))
    return recs
