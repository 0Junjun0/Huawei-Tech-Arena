"""Fares' ver3 feature_matrix as a plain feature function on our 10 s grid."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "TechArena-Finland-2026"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "teammates"))
import numpy as np, pandas as pd
import fares3 as F

def fares_feats(df):
    """one row per 10 s window from the first window to the last (gaps filled with zeros)."""
    t = pd.DatetimeIndex(df.index).as_unit("ns").asi8
    n_win = int(t[-1] // F.BIN_NS - t[0] // F.BIN_NS) + 1
    out = None
    for rows, ids, start, X in F.feature_segments(df):
        if out is None:
            out = np.zeros((n_win, X.shape[1]), dtype=np.float32)
        off = int(start - t[0] // F.BIN_NS)
        out[off:off + len(X)] = X
    return out


def fares_light_matrix(frame, *, _grid=None):
    """Fares' feature_matrix without the window-stat block (compute_window_features) and seen flags:
    physiology channels + their r2/8/24 context + recording-relative terms. Same maths otherwise."""
    values, _, clock = F.physiology_windows(frame, _grid=_grid)
    seen = np.isfinite(values)
    scaled = np.arcsinh(np.where(seen, values, 0.0) / F.UNITS).astype(np.float32)
    local = scaled[:, F.CONTEXT_CHANNELS]
    valid = seen[:, F.CONTEXT_CHANNELS]
    counts = np.vstack([np.zeros(local.shape[1]), valid.cumsum(axis=0)])
    total = np.vstack([np.zeros(local.shape[1]), local.cumsum(axis=0, dtype=np.float64)])
    sq = np.vstack([np.zeros(local.shape[1]), (local ** 2).cumsum(axis=0, dtype=np.float64)])
    pos = np.arange(len(clock))
    parts = [scaled]
    for radius in (2, 8, 24):
        l = np.maximum(pos - radius, 0); h = np.minimum(pos + radius + 1, len(clock))
        count = np.maximum(counts[h] - counts[l], 1)
        avg = (total[h] - total[l]) / count
        var = np.maximum((sq[h] - sq[l]) / count - avg ** 2, 0.0)
        mid = pos + 1
        before = (total[mid] - total[l]) / np.maximum(counts[mid] - counts[l], 1)
        after = (total[h] - total[pos]) / np.maximum(counts[h] - counts[pos], 1)
        parts.extend([avg, np.sqrt(var), np.where(valid, local - avg, 0), after - before])
    ordered = np.sort(np.where(valid, local, np.inf), axis=0)
    size = valid.sum(0); features = np.arange(local.shape[1]); qs = []
    for q in (0.15, 0.50, 0.85):
        index = np.floor(q * np.maximum(size - 1, 0)).astype(int)
        qs.append(np.where(size > 0, ordered[index, features], 0.0))
    lo, med, hi = qs
    parts.extend([np.where(valid, local - med, 0), np.where(valid, (local - med) / (0.08 + hi - lo), 0), np.broadcast_to(hi - lo, local.shape)])
    return np.nan_to_num(np.column_stack(parts).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fares_light(df):
    t = pd.DatetimeIndex(df.index).as_unit("ns").asi8
    n_win = int(t[-1] // F.BIN_NS - t[0] // F.BIN_NS) + 1
    out = None
    for rows, ids, start, section in F.recording_segments(df):
        grid = (section.index.as_unit("ns").asi8, start, ids)
        X = fares_light_matrix(section, _grid=grid)
        if out is None:
            out = np.zeros((n_win, X.shape[1]), dtype=np.float32)
        off = int(start - t[0] // F.BIN_NS)
        m = min(len(X), n_win - off)
        if m > 0:
            out[off:off + m] = X[:m]
    return out


def fares_local(df):
    """fares_light without the whole-recording quantile block: physiology + r2/8/24 context only."""
    X = fares_light(df)
    nC = len(F.CONTEXT_CHANNELS)
    return X[:, : X.shape[1] - 3 * nC]
