"""Fast numpy feature extraction on 10 s windows (shared by training + submission).

Windows are aligned to 10 s boundaries of the timestamp (same as pandas floor("10s")).
"""
import numpy as np
import pandas as pd

WIN_NS = 10_000_000_000
SIG = ["BVP", "EDA", "HR", "IBI_duration", "Temperature"]


def _win_stats(wid, vals, n_win):
    """count, mean, std, min, max per window for one signal (NaNs already dropped)."""
    cnt = np.bincount(wid, minlength=n_win).astype(np.float64)
    s = np.bincount(wid, weights=vals, minlength=n_win)
    ss = np.bincount(wid, weights=vals * vals, minlength=n_win)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / cnt
        var = np.maximum(ss / cnt - mean * mean, 0.0)
    mn = np.full(n_win, np.nan)
    mx = np.full(n_win, np.nan)
    if len(vals):
        # wid is sorted (time ordered) -> reduceat over run starts
        starts = np.flatnonzero(np.r_[True, wid[1:] != wid[:-1]])
        u = wid[starts]
        mn[u] = np.minimum.reduceat(vals, starts)
        mx[u] = np.maximum.reduceat(vals, starts)
    return cnt, mean, np.sqrt(var), mn, mx


def _ffill(a):
    """forward then backward fill NaNs along axis 0 (2-D)."""
    a = a.copy()
    n = a.shape[0]
    idx = np.where(np.isnan(a), 0, np.arange(n)[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    a = a[idx, np.arange(a.shape[1])]
    # backward fill leading NaNs
    rev = a[::-1]
    idx = np.where(np.isnan(rev), 0, np.arange(n)[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    a = rev[idx, np.arange(a.shape[1])][::-1]
    return np.nan_to_num(a, nan=0.0)


def _roll_mean(a, k):
    """centered rolling mean along axis 0 with window k (edges use available values)."""
    n = a.shape[0]
    h = k // 2
    c = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def _roll_std(a, k):
    m = _roll_mean(a, k)
    m2 = _roll_mean(a * a, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def window_grid(index):
    t = np.asarray(index, dtype="datetime64[ns]").view(np.int64)
    w0 = (t[0] // WIN_NS)
    wid = (t // WIN_NS - w0).astype(np.int64)
    n_win = int(wid[-1]) + 1
    return t, wid, w0, n_win


def compute(df: pd.DataFrame, use_time=True):
    """Returns (window_start_ns[int64], X[float32], names)."""
    t, wid, w0, n_win = window_grid(df.index)
    base_cols, names = [], []

    for c in SIG:
        v = df[c].to_numpy(dtype=np.float64) if c in df.columns else np.full(len(t), np.nan)
        m = ~np.isnan(v)
        cnt, mean, std, mn, mx = _win_stats(wid[m], v[m], n_win)
        if c == "BVP":
            d = np.abs(np.diff(v, prepend=v[0]))
            d[~m] = np.nan
            mm = ~np.isnan(d)
            _, dmean, _, _, dmax = _win_stats(wid[mm], d[mm], n_win)
            base_cols += [std, mx - mn, dmean, dmax]
            names += ["BVP_std", "BVP_ptp", "BVP_absdiff", "BVP_absdiff_max"]
        elif c == "IBI_duration":
            # successive beat differences (RMSSD proxy) + beat count
            vv = v[m]
            sd = np.abs(np.diff(vv, prepend=vv[0] if len(vv) else 0.0))
            _, sdm, _, _, _ = _win_stats(wid[m], sd * sd, n_win)
            base_cols += [mean, std, cnt, sdm]
            names += ["IBI_mean", "IBI_std", "IBI_count", "IBI_ssd"]
        else:
            base_cols += [mean, std, mx - mn]
            names += [f"{c}_mean", f"{c}_std", f"{c}_ptp"]

    B = np.column_stack(base_cols)
    # IBI count: missing => 0 beats, keep 0 rather than ffill
    ibi_cnt_i = names.index("IBI_count")
    B[:, ibi_cnt_i] = np.nan_to_num(B[:, ibi_cnt_i])
    B = _ffill(B)

    # Signals used for context / recording-relative features
    key = [names.index(k) for k in
           ["HR_mean", "EDA_mean", "Temperature_mean", "BVP_std", "BVP_absdiff", "IBI_mean", "EDA_std", "IBI_ssd"]]
    K = B[:, key]
    knames = [names[i] for i in key]
    Klog = K.copy()
    # BVP amplitudes are heavy tailed -> log
    for j, nm in enumerate(knames):
        if nm.startswith("BVP") or nm == "IBI_ssd":
            Klog[:, j] = np.log1p(np.abs(K[:, j]))

    feats = [B]
    fnames = list(names)

    for k in (3, 7, 19, 61):
        R = _roll_mean(Klog, k)
        feats.append(R)
        fnames += [f"{n}_c{k}" for n in knames]
    for k in (7, 31):
        S = _roll_std(Klog, k)
        feats.append(S)
        fnames += [f"{n}_s{k}" for n in knames]

    # slopes over ~1 and ~3 minutes (centered difference of smoothed signal)
    sm = _roll_mean(Klog, 7)
    for lag in (3, 9):
        fwd = np.vstack([sm[lag:], np.repeat(sm[-1:], lag, axis=0)])
        bwd = np.vstack([np.repeat(sm[:1], lag, axis=0), sm[:-lag]])
        feats.append(fwd - bwd)
        fnames += [f"{n}_slope{lag}" for n in knames]

    # recording-relative: vs median, vs 10th pct (baseline-ish), scaled by IQR; rank in recording
    med = np.median(Klog, axis=0)
    p10 = np.percentile(Klog, 10, axis=0)
    p90 = np.percentile(Klog, 90, axis=0)
    iqr = np.percentile(Klog, 75, axis=0) - np.percentile(Klog, 25, axis=0) + 1e-6
    sm19 = _roll_mean(Klog, 19)
    feats += [(Klog - med) / iqr, (sm19 - med) / iqr, sm19 - p10, (sm19 - p10) / (p90 - p10 + 1e-6)]
    fnames += [f"{n}_zrec" for n in knames] + [f"{n}_c19_zrec" for n in knames] + \
              [f"{n}_c19_mp10" for n in knames] + [f"{n}_c19_rng" for n in knames]
    ranks = np.argsort(np.argsort(sm19, axis=0), axis=0) / max(n_win - 1, 1)
    feats.append(ranks)
    fnames += [f"{n}_c19_rank" for n in knames]

    # recording-level context (same for every window): spread of HR/EDA/BVP in this recording
    rec = np.concatenate([p90 - p10, p90, p10])
    feats.append(np.repeat(rec[None, :], n_win, axis=0))
    fnames += [f"{n}_rec_rng" for n in knames] + [f"{n}_rec_p90" for n in knames] + [f"{n}_rec_p10" for n in knames]

    if use_time:
        pos = np.arange(n_win, dtype=np.float64)
        feats.append(np.column_stack([pos * 10.0, pos / max(n_win - 1, 1), np.full(n_win, n_win * 10.0)]))
        fnames += ["t_sec", "t_frac", "rec_len_sec"]

    X = np.hstack(feats).astype(np.float32)
    win_start = (w0 + np.arange(n_win)) * WIN_NS
    return win_start, X, fnames
