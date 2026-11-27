import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pathlib
import numpy as np
import pandas as pd
import onnxruntime as ort
from adm_base_class import ADM_Base

# ---- feature extraction (copied from lab/feats.py) ----

from scipy.ndimage import minimum_filter1d, maximum_filter1d

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


def _extras(df, t, wid, n_win, Klog, fb=(3, 9)):
    """EDA response dynamics + one-sided (before/after) context."""
    cols, nm = [], []
    v = df["EDA"].to_numpy(dtype=np.float64) if "EDA" in df.columns else np.full(len(t), np.nan)
    m = ~np.isnan(v)
    ev, ew = v[m], wid[m]
    if len(ev) > 2:
        d = np.diff(ev, prepend=ev[0])
        pos = np.maximum(d, 0.0)
        _, pmean, _, _, pmax = _win_stats(ew, pos, n_win)
        _, _, _, dmin, _ = _win_stats(ew, d, n_win)
        # count of rising-edge onsets (SCR-like): derivative crosses small threshold upward
        thr = 0.01 * (np.percentile(ev, 90) - np.percentile(ev, 10) + 1e-6) + 1e-4
        on = ((d > thr) & (np.r_[0.0, d[:-1]] <= thr)).astype(np.float64)
        _, onr, _, _, _ = _win_stats(ew, on, n_win)
    else:
        pmean = pmax = dmin = onr = np.zeros(n_win)
    E = _ffill(np.column_stack([pmean, pmax, dmin, onr]))
    cols.append(E); nm += ["EDA_dpos_mean", "EDA_dpos_max", "EDA_dmin", "EDA_onset_rate"]
    cols.append(_roll_mean(E, 7)); nm += [f"{x}_c7" for x in ["EDA_dpos_mean", "EDA_dpos_max", "EDA_dmin", "EDA_onset_rate"]]
    # one-sided context: mean of next k minus mean of previous k (onset/offset detectors)
    c = np.vstack([np.zeros((1, Klog.shape[1])), np.cumsum(Klog, 0)])
    i = np.arange(n_win)
    for k in fb:
        lo_b, hi_b = np.clip(i - k, 0, n_win), i
        lo_f, hi_f = np.clip(i + 1, 0, n_win), np.clip(i + 1 + k, 0, n_win)
        nb = np.maximum(hi_b - lo_b, 1)[:, None]; nf = np.maximum(hi_f - lo_f, 1)[:, None]
        before = (c[hi_b] - c[lo_b]) / nb
        after = (c[hi_f] - c[lo_f]) / nf
        before = np.where((hi_b - lo_b)[:, None] > 0, before, Klog)
        after = np.where((hi_f - lo_f)[:, None] > 0, after, Klog)
        cols += [after - before, Klog - before]
        nm += [f"k{j}_fb{k}" for j in range(Klog.shape[1])] + [f"k{j}_nowb{k}" for j in range(Klog.shape[1])]
    return np.hstack(cols), nm


def compute(df: pd.DataFrame, use_time=True, extras=False, local=False, lbase=(), fb=(3, 9), edge=0, inlen=False):
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
    ii = np.arange(n_win)
    for lag in (3, 9):
        # index clipping == edge padding; works for any n_win >= 1
        fwd = sm[np.minimum(ii + lag, n_win - 1)]
        bwd = sm[np.maximum(ii - lag, 0)]
        feats.append(fwd - bwd)
        fnames += [f"{n}_slope{lag}" for n in knames]

    if lbase:
        # bounded local baseline: position of the smoothed signal within its +-k/2 window range
        sm3 = _roll_mean(Klog, 3)
        for k in lbase:
            lo = minimum_filter1d(sm3, size=min(k, 2 * n_win + 1), axis=0, mode="nearest")
            hi = maximum_filter1d(sm3, size=min(k, 2 * n_win + 1), axis=0, mode="nearest")
            feats += [sm3 - lo, (sm3 - lo) / (hi - lo + 1e-6)]
            fnames += [f"{n}_lb{k}_mmin" for n in knames] + [f"{n}_lb{k}_pos" for n in knames]

    if inlen:
        feats.append(np.full((n_win, 1), np.log1p(n_win)))
        fnames += ["log_n_win"]

    if edge:
        # distance (windows) to the start / end of this input, capped: lets the model discount edge context
        pos = np.arange(n_win, dtype=np.float64)
        feats.append(np.column_stack([np.minimum(pos, edge), np.minimum(n_win - 1 - pos, edge)]))
        fnames += ["edge_start", "edge_end"]

    if local:
        # no whole-recording statistics: only bounded context (robust to chunked / concatenated input)
        if extras:
            Xe, ne = _extras(df, t, wid, n_win, Klog, fb)
            feats.append(Xe); fnames += ne
        X = np.hstack(feats).astype(np.float32)
        return (w0 + np.arange(n_win)) * WIN_NS, X, fnames

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

    if extras:
        Xe, ne = _extras(df, t, wid, n_win, Klog, fb)
        feats.append(Xe)
        fnames += ne

    if use_time:
        pos = np.arange(n_win, dtype=np.float64)
        feats.append(np.column_stack([pos * 10.0, pos / max(n_win - 1, 1), np.full(n_win, n_win * 10.0)]))
        fnames += ["t_sec", "t_frac", "rec_len_sec"]

    X = np.hstack(feats).astype(np.float32)
    win_start = (w0 + np.arange(n_win)) * WIN_NS
    return win_start, X, fnames
# ---- model ----
LABELS = np.array(["Rest", "Stress", "Activity"], dtype=object)
FEAT_KW = {'use_time': False, 'extras': True, 'edge': 30, 'inlen': True, 'lbase': [31, 61, 121], 'fb': [3, 9, 18, 36]}
USE_TIME = FEAT_KW.get("use_time", False)
CLASS_WEIGHTS = np.array([1, 1.4, 1.0], dtype=np.float32)
SMOOTH = 5
GATE = None  # (ratio, alpha): input-level protocol gate, or None


def _smooth(P, k):
    if k <= 1:
        return P
    h = k // 2
    n = len(P)
    c = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


class TechArenaModel(ADM_Base):
    def load_model(self, onnx_file):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.prob_idx = [i for i, o in enumerate(self.session.get_outputs()) if "prob" in o.name.lower()][0]

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        n = len(input)
        if n == 0:
            return pd.DataFrame({"Activity": pd.Series(dtype=object)}, index=input.index)
        try:
            labels = self._predict(input)
        except Exception:
            labels = np.full(n, "Rest", dtype=object)
        return pd.DataFrame({"Activity": pd.Series(labels, index=input.index, dtype=object, copy=False)})

    def _predict(self, input):
        idx = input.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        t = np.asarray(idx, dtype="datetime64[ns]").view(np.int64)
        out = np.full(len(t), "Rest", dtype=object)
        cols = {}
        for c in ("BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"):
            if c in input.columns:
                col = input[c]
                if col.dtype.kind not in "fiub":
                    col = pd.to_numeric(col, errors="coerce")
                cols[c] = col.to_numpy(dtype=np.float64, na_value=np.nan)
        ok = t != np.iinfo(np.int64).min  # NaT
        if ok.all() and (t[1:] >= t[:-1]).all():
            pos, tt = None, t
        else:
            if not ok.any():
                return out
            pos = np.flatnonzero(ok)
            o = np.argsort(t[pos], kind="stable"); pos = pos[o]; tt = t[pos]
            cols = {c: v[pos] for c, v in cols.items()}
        _, X, _ = compute(_Frame(tt, cols), **FEAT_KW)
        P = np.asarray(self.session.run(None, {self.input_name: X})[self.prob_idx], dtype=np.float64)
        if GATE is not None:
            # whole input looks like an exercise session -> no Stress, else no Activity
            if P[:, 2].mean() > GATE[0] * P[:, 1].mean():
                P[:, 1] *= GATE[1]
            else:
                P[:, 2] *= GATE[1]
        pred = (_smooth(P, SMOOTH) * CLASS_WEIGHTS).argmax(1)
        lab = LABELS[pred[tt // WIN_NS - tt[0] // WIN_NS]]
        if pos is None:
            return lab
        out[pos] = lab
        return out


class _Col:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def to_numpy(self, dtype=None):
        return self.a


class _Frame:
    """minimal DataFrame stand-in for compute(): .index (datetime64[ns]), .columns, [col].to_numpy()"""

    def __init__(self, t, cols):
        self.index = t.view("datetime64[ns]")
        self.columns = cols
        self._c = cols

    def __getitem__(self, c):
        return _Col(self._c[c])


def get_model() -> ADM_Base:
    m = TechArenaModel()
    m.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return m
