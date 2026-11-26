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


def compute(df: pd.DataFrame, use_time=True, extras=False, local=False, lbase=(), fb=(3, 9), edge=0, inlen=False, pmin=()):
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

    if pmin:
        # rise above the minimum of the previous k windows (one-sided: onset of stress responses)
        sm3 = _roll_mean(Klog, 3)
        for k in pmin:
            kk = min(k + 1, 2 * n_win - 1) if n_win > 1 else 1
            lo = minimum_filter1d(sm3, size=kk, axis=0, mode="nearest", origin=(kk - 1) // 2)
            feats.append(sm3 - lo); fnames += [f"{n}_pmin{k}" for n in knames]

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
SPLIT_GAP_NS = 600 * 1_000_000_000
GATE = None  # (ratio, alpha): input-level protocol gate, or None
PROTO = {'mean': [3.778308950601882, 5.456085289985822, 2.1140796754361824, 3.5179882761666903, 8.293526965799947, 0.20353024857124308, 0.7379579466216769, 90.83563253540933, 0.5387919357798318, 1.6495415708799284, 0.6126236305013997, 0.027634024647849212, 4.89016202173651, 0.007121798399492643, 36.24871852134361, 0.10443789123462431, 0.2862591937699698, 0.3787857444241235, 0.501370407850574, 0.23463234716985523, 0.42892474195438585, 1.0894750742366168, 0.15688109782745202, 0.5290105661827197, 6.704985281622831, 0.4473294318656691, 1.3696497134250554, 0.058006574158397495, 0.016729198997956475, 3.4918211147522173, 0.00902882218946518, 0.5181184482825714, 0.10205514684072414, 0.25294753434512485, 3.349709875085021, 4.896178401144971, 1.8475096533557396, 3.053091360735406, 6.978265338973115, 0.07285945311891896, 0.2747246715054839, 82.62314952977023, 0.1223708322019569, 0.37126230066367255, 0.5470870701763387, 0.00984094308645992, 1.23447109471094, 0.0009244621779278691, 35.65992961709943, 0.042296035683865266, 0.11705166030667023, 4.234849738904453, 6.068805487228982, 2.3961714029184056, 4.030824331994496, 9.668913248034107, 0.3751409571515762, 1.345077117839697, 99.5058769429011, 1.1081141223279314, 3.393558118459366, 0.6804834654401212, 0.047902266477274184, 9.122878228782328, 0.016748156760912864, 36.84184118677915, 0.1545069883069051, 0.44104581566322, 3.0020992761247673], 'scale': [0.7556370893164924, 0.8315093837997913, 0.8464830629646186, 0.9144482822085546, 17.19176389313459, 0.31871089425633925, 1.1420384201752858, 22.498257188413724, 0.4642330374399055, 1.393843960697954, 0.23985413298203603, 0.018359669065547028, 4.033874864437087, 0.01691091937041647, 26.686532079114198, 0.5138433201480089, 1.2845599233352754, 0.2117559782458292, 0.25186230551108774, 0.16172868506215216, 0.2567323844576747, 2.493769404822456, 0.3083900677607901, 0.954215120638401, 6.6438247268655335, 0.5268748912548562, 1.5733734540443305, 0.056411589505442054, 0.012494111494841652, 1.9787126161846864, 0.02039164997772797, 4.876169187528644, 0.9415958447818337, 1.936921268845325, 0.8065834337345632, 0.8976709314316436, 0.9158350629413935, 1.0379415690143825, 15.79643473402687, 0.1266396504330149, 0.47609936762680094, 21.5647070492794, 0.12142009681376378, 0.3596357935211896, 0.22068275771978096, 0.012302091766156825, 2.45761022723149, 0.00245276125434106, 25.65848010139247, 0.28707820653534327, 0.7724344343422569, 0.7882900130134184, 0.868434488672302, 0.8109765326738475, 0.8633059476803924, 18.96578353387566, 0.6043290785335241, 2.117459942843488, 25.76650598606263, 1.1318214947314877, 3.3445894877090803, 0.27163823850943825, 0.0322142444073254, 5.94951799808401, 0.044473578172798005, 28.529980799730232, 0.6968701894200267, 1.866870189140702, 0.6359953104608912], 'coef': [-1.077847483304152, 0.42281616861691373, 1.0284989353747989, -1.4858365195027918, -0.19690049246284486, -0.5157288953860355, 0.2218929097678789, 1.3656694524119397, -0.09500619666206632, -0.5880220130897722, -0.5309966700983154, -0.4572738291732854, -2.0209589558002206, 0.9939775766024322, 0.0189523594510326, 0.8278594114839651, 1.1100837170586388, 0.24383608570760626, -0.638293290500946, 0.8183939335265197, -0.26864075386662495, 0.3472287984323219, -0.1818914862433015, 0.06783860467711966, -0.5904223697987934, 1.1370887765623607, 0.6899810034832798, 0.00595933917967098, 0.14955420960125648, 1.000100791439259, -0.664938380045578, 0.6551143645515003, 0.18683161802695908, 0.29890128590593973, -0.31390400676460767, 1.000399356583022, 1.213640649920864, -1.1903197112680886, -0.12644215612108997, 0.0657771266222361, 0.6613065461163067, -0.5593833207584478, 0.7780850636415121, -0.5946977528120467, -0.16873380226062631, 0.016322905852556188, 0.48465337255485974, 0.3610725787805184, -0.14569122467819984, 0.8719656164827752, 1.0445567983104633, -0.9950292896893898, 0.24319014568698247, 0.8276573398978035, -0.13986054603081008, -0.17516291568011086, -0.2819051001441334, 1.171020291938113, 0.8045411706878969, -0.26935867293756294, -0.2420757635971035, -0.14915602519428825, -0.2380267760072223, -0.060143800377902415, 0.6270957152020369, 0.15261018629783307, 0.8986050301837225, 1.1753338414296743, 0.1774043794817803], 'intercept': 2.4152743985729423, 'gamma': 1.0}  # logistic regression: P(input is an exercise session) from input aggregates, or None


FARES_DIM = 375  # >0: append Fares' ver3 features (isolated module below)
_FARES = None
_FARES_SRC = '"""Huawei wrist classifier. The same feature code serves training and deployment."""\n\nimport pathlib\n\nimport numpy as np\nimport pandas as pd\nfrom scipy.ndimage import maximum_filter1d, minimum_filter1d\n\nLABELS = ["Rest", "Stress", "Activity"]\nSENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]\n\n\ndef _ffill_bfill_2d(arr: np.ndarray) -> np.ndarray:\n    """Forward-fills missing window features and backfills only leading empty bins."""\n    n_bins, n_cols = arr.shape\n    if n_bins == 0:\n        return arr\n    mask = np.isnan(arr)\n    if not mask.any():\n        return np.nan_to_num(arr) if np.isinf(arr).any() else arr\n    idx_f = np.where(~mask, np.arange(n_bins)[:, None], 0)\n    np.maximum.accumulate(idx_f, axis=0, out=idx_f)\n    out = arr[idx_f, np.arange(n_cols)]\n\n    mask2 = np.isnan(out)\n    if np.any(mask2):\n        idx_b = np.where(~mask2, np.arange(n_bins)[:, None], n_bins - 1)\n        idx_b = np.minimum.accumulate(idx_b[::-1], axis=0)[::-1]\n        out = out[idx_b, np.arange(n_cols)]\n    return np.nan_to_num(out, nan=0.0)\n\n\ndef _ffill_bfill_1d(arr: np.ndarray) -> np.ndarray:\n    """Forward-fills a 1D feature array and backfills only leading missing values."""\n    n = len(arr)\n    if n == 0:\n        return arr\n    m = np.isnan(arr)\n    if not m.any():\n        return np.nan_to_num(arr) if np.isinf(arr).any() else arr\n    if np.all(m):\n        return np.zeros(n, dtype=np.float32)\n    idx_f = np.where(~m, np.arange(n), 0)\n    np.maximum.accumulate(idx_f, out=idx_f)\n    out = arr[idx_f]\n\n    m2 = np.isnan(out)\n    if np.any(m2):\n        idx_b = np.where(~m2, np.arange(n), n - 1)\n        idx_b = np.minimum.accumulate(idx_b[::-1])[::-1]\n        out = out[idx_b]\n    return np.nan_to_num(out, nan=0.0)\n\n\ndef _symmetric_rolling_mean(arr: np.ndarray, radius: int) -> np.ndarray:\n    """Fast centered rolling average over [i - radius, i + radius] via cumulative sums."""\n    n = len(arr)\n    if n == 0:\n        return arr\n    cs = np.zeros((n + 1, arr.shape[1]), dtype=np.float64)\n    cs[1:] = np.cumsum(arr, axis=0, dtype=np.float64)\n    idx = np.arange(n)\n    left = np.maximum(0, idx - radius)\n    right = np.minimum(n, idx + radius + 1)\n    counts = (right - left)[:, None]\n    return ((cs[right] - cs[left]) / counts).astype(np.float32)\n\n\ndef _symmetric_rolling_std_1d(arr: np.ndarray, radius: int) -> np.ndarray:\n    """Fast centered rolling std over [i - radius, i + radius] via prefix sums."""\n    n = len(arr)\n    if n == 0:\n        return arr\n    x = arr.astype(np.float64)\n    cs1 = np.zeros(n + 1, dtype=np.float64)\n    cs2 = np.zeros(n + 1, dtype=np.float64)\n    cs1[1:] = np.cumsum(x)\n    cs2[1:] = np.cumsum(x * x)\n    idx = np.arange(n)\n    left = np.maximum(0, idx - radius)\n    right = np.minimum(n, idx + radius + 1)\n    cnt = right - left\n    mean = (cs1[right] - cs1[left]) / cnt\n    var = np.maximum(0.0, (cs2[right] - cs2[left]) / cnt - mean * mean)\n    return np.sqrt(var).astype(np.float32)\n\n\ndef _symmetric_rolling_ptp_1d(\n    mins_1d: np.ndarray, maxs_1d: np.ndarray, radius: int\n) -> np.ndarray:\n    """Centered peak-to-peak swing across [i - radius, i + radius] windows."""\n    n = len(mins_1d)\n    if n == 0:\n        return np.empty(0, dtype=np.float32)\n    width = 2 * radius + 1\n    return maximum_filter1d(maxs_1d, width, mode="nearest") - minimum_filter1d(\n        mins_1d, width, mode="nearest"\n    )\n\n\ndef _sensor_column(df, name, dtype):\n    """Read native numeric columns without making a NaN-filled second copy."""\n    if name not in df:\n        return np.full(len(df), np.nan, dtype=dtype)\n    column = df[name]\n    if isinstance(column.dtype, np.dtype) and column.dtype.kind in "fiu":\n        return column.to_numpy(dtype=dtype, copy=False)\n    return column.to_numpy(dtype=dtype, na_value=np.nan)\n\n\ndef _optical_features(pulse, origin, bins, means, start_bin, bin_ns, width):\n    """Reduce observed samples by bin. Resample irregular windows on their actual timestamps."""\n    n = len(means)\n    wave = np.zeros((n, 3), dtype=np.float32)\n    bands = np.full((n, 3), np.nan, dtype=np.float32)\n    if not len(pulse):\n        return wave.T, bands.T\n    starts = np.r_[0, np.flatnonzero(bins[1:] != bins[:-1]) + 1]\n    ends = np.r_[starts[1:], len(bins)]\n    lengths = ends - starts\n    occupied = bins[starts]\n    cadence = 15_625_000\n    gaps = np.r_[np.int64(0), np.diff(origin)]\n    gaps[starts] = 0\n    adjacent = (gaps > 0) & (gaps <= 23_437_500)\n    pairs = adjacent & np.r_[False, adjacent[:-1]]\n    delta = np.zeros(len(pulse), dtype=np.float32)\n    delta[1:] = np.diff(pulse)\n    second = np.empty_like(delta)\n    second[0] = 0\n    second[1:] = np.diff(delta)\n    centered = pulse - means[bins]\n    crossings = np.zeros(len(pulse), dtype=bool)\n    crossings[1:] = (centered[1:] * centered[:-1]) < 0\n    n_adjacent = np.maximum(1, np.add.reduceat(adjacent, starts, dtype=np.int64))\n    n_pairs = np.maximum(1, np.add.reduceat(pairs, starts, dtype=np.int64))\n    delta[~adjacent] = 0\n    second[~pairs] = 0\n    wave[occupied, 0] = np.sqrt(\n        np.add.reduceat(np.square(delta, dtype=np.float64), starts) / n_adjacent\n    )\n    wave[occupied, 1] = np.sqrt(\n        np.add.reduceat(np.square(second, dtype=np.float64), starts) / n_pairs\n    )\n    wave[occupied, 2] = (\n        np.add.reduceat(crossings & adjacent, starts, dtype=np.int64) / n_adjacent\n    )\n\n    offset = origin[starts] - (start_bin + occupied) * bin_ns\n    last = origin[ends - 1] - (start_bin + occupied) * bin_ns\n    complete = (\n        (lengths >= width * 0.85)\n        & (lengths <= width * 1.1)\n        & (offset <= 125_000_000)\n        & (last >= bin_ns - 125_000_000)\n        & (np.maximum.reduceat(gaps, starts) <= 125_000_000)\n    )\n    irregular = gaps != cadence\n    irregular[starts] = False\n    on_grid = (\n        (lengths == width) & (offset == 0) & ~np.logical_or.reduceat(irregular, starts)\n    )\n    grid = np.arange(width)\n    target = grid * cadence\n    taper = np.hanning(width)\n\n    def store(bins_in_batch, signals):\n        power = (np.abs(np.fft.rfft(signals * taper, axis=1)) ** 2).astype(np.float32)\n        cardiac = power[:, 7:36].mean(axis=1)\n        motion = power[:, 40:151].mean(axis=1)\n        bands[occupied[bins_in_batch]] = np.column_stack(\n            (np.log1p(cardiac), np.log1p(motion), cardiac / (cardiac + motion + 1e-6))\n        )\n\n    regular_bins = np.flatnonzero(complete & on_grid)\n    for j in range(0, len(regular_bins), 96):\n        idx = regular_bins[j : j + 96]\n        loc = starts[idx]\n        if loc[-1] - loc[0] == (len(idx) - 1) * width and np.all(np.diff(loc) == width):\n            signal = centered[loc[0] : loc[-1] + width].reshape(-1, width)\n        else:\n            signal = centered[loc[:, None] + grid]\n        store(idx, signal)\n    other_bins = np.flatnonzero(complete & ~on_grid)\n    for j in range(0, len(other_bins), 96):\n        idx = other_bins[j : j + 96]\n        signal = np.empty((len(idx), width), dtype=np.float64)\n        for row, k in enumerate(idx):\n            sl = slice(starts[k], ends[k])\n            t = origin[sl] - (start_bin + occupied[k]) * bin_ns\n            signal[row] = np.interp(target, t, centered[sl])\n        store(idx, signal)\n    return wave.T, bands.T\n\n\ndef compute_window_features(\n    df: pd.DataFrame, win_sec: int = 10, *, reference_optical: bool = False, _grid=None\n) -> tuple[int, np.ndarray, np.ndarray]:\n    """Extracts 95 defensive, vectorized physiological, spectral, HRV, context, and subject-normalized features."""\n    bin_ns = int(win_sec * 1_000_000_000)\n    if _grid is None:\n        dt_index = pd.DatetimeIndex(df.index).as_unit("ns")\n        order = (\n            None\n            if dt_index.is_monotonic_increasing\n            else np.argsort(dt_index.asi8, kind="stable")\n        )\n        asi8 = dt_index.asi8 if order is None else dt_index.asi8[order]\n        raw_bins = asi8 // bin_ns\n        start_bin = int(raw_bins[0])\n        bin_ids = raw_bins - start_bin\n    else:\n        asi8, start_bin, bin_ids = _grid\n        order = None\n    n_bins = int(bin_ids[-1]) + 1\n\n    n_cols = len(SENSOR_COLS)\n    means = np.full((n_bins, n_cols), np.nan, dtype=np.float32)\n    stds = np.full_like(means, np.nan)\n    mins = np.full_like(means, np.nan)\n    maxs = np.full_like(means, np.nan)\n    within_slope = np.zeros_like(means)\n    observations = []\n    valid_ranges = (\n        (None, None),\n        (0, 150),\n        (25, 240),\n        (None, None),\n        (0.20, 2.50),\n        (15, 45),\n    )\n    for c, (name, (low, high)) in enumerate(zip(SENSOR_COLS, valid_ranges)):\n        col = _sensor_column(df, name, np.float32)\n        if order is not None:\n            col = col[order]\n        valid = np.isfinite(col) if low is None else (col >= low) & (col <= high)\n        if valid.all():\n            loc = slice(None)\n        else:\n            loc = np.flatnonzero(valid)\n        b, values = bin_ids[loc], col[loc]\n        observations.append((b, values))\n        if c == 0:\n            optical_positions = loc\n            if reference_optical:\n                bvp = np.where(valid, col, np.float32(0))\n        if not len(b):\n            continue\n        starts = np.r_[0, np.flatnonzero(b[1:] != b[:-1]) + 1]\n        ends = np.r_[starts[1:], len(b)]\n        bins = b[starts]\n        count = ends - starts\n        v = values.astype(np.float64)\n        if c == 0:\n            total = np.add.reduceat(v, starts)\n            squares = np.add.reduceat(v * v, starts)\n        else:\n            # Keep the original accumulation order for near-constant temperature and beat signals.\n            total = np.bincount(b, weights=v, minlength=n_bins)[bins]\n            squares = np.bincount(b, weights=v * v, minlength=n_bins)[bins]\n        m = total / count\n        variance = np.maximum(0.0, squares / count - m * m)\n        means[bins, c] = m\n        stds[bins, c] = np.sqrt(variance)\n        mins[bins, c] = np.minimum.reduceat(values, starts)\n        maxs[bins, c] = np.maximum.reduceat(values, starts)\n        within_slope[bins, c] = values[ends - 1] - values[starts]\n\n    means = _ffill_bfill_2d(means)\n    stds = _ffill_bfill_2d(stds)\n    mins = _ffill_bfill_2d(mins)\n    maxs = _ffill_bfill_2d(maxs)\n    ptp = maxs - mins\n    delta = np.zeros_like(means)\n    delta[1:] = means[1:] - means[:-1]\n\n    if reference_optical:\n        # 64 Hz BVP waveform sharpness, zero-crossing rate, and bin-aligned cardiac-vs-motion FFT energy\n        bvp_centered = bvp - means[bin_ids, 0]\n        same_b = bin_ids[1:] == bin_ids[:-1]\n\n        d1 = np.diff(bvp)\n        d1_sq = np.bincount(\n            bin_ids[1:][same_b],\n            weights=(d1[same_b] ** 2).astype(np.float64),\n            minlength=n_bins,\n        )\n        d1_cnt = np.maximum(1, np.bincount(bin_ids[1:][same_b], minlength=n_bins))\n        bvp_diff1_rms = np.sqrt(d1_sq / d1_cnt).astype(np.float32)\n\n        d2 = np.diff(d1)\n        same_b2 = bin_ids[2:] == bin_ids[:-2]\n        d2_sq = np.bincount(\n            bin_ids[2:][same_b2],\n            weights=(d2[same_b2] ** 2).astype(np.float64),\n            minlength=n_bins,\n        )\n        d2_cnt = np.maximum(1, np.bincount(bin_ids[2:][same_b2], minlength=n_bins))\n        bvp_diff2_rms = np.sqrt(d2_sq / d2_cnt).astype(np.float32)\n\n        zc = ((bvp_centered[1:] * bvp_centered[:-1]) < 0) & same_b\n        bvp_zcr = (np.bincount(bin_ids[1:][zc], minlength=n_bins) / d1_cnt).astype(\n            np.float32\n        )\n        bvp_crest = ptp[:, 0] / (stds[:, 0] + 1e-4)\n\n        win_len_64 = win_sec * 64\n        bvp_cardiac = np.full(n_bins, np.nan, dtype=np.float32)\n        bvp_motion = np.full(n_bins, np.nan, dtype=np.float32)\n        bvp_cardiac_frac = np.full(n_bins, np.nan, dtype=np.float32)\n        bin_starts = np.flatnonzero(np.r_[True, bin_ids[1:] != bin_ids[:-1]])\n        bin_ends = np.r_[bin_starts[1:], len(bvp)]\n        seg_lens = bin_ends - bin_starts\n        valid_fft_segs = np.flatnonzero(seg_lens >= 64)\n        if len(valid_fft_segs) > 0:\n            bvp_mat = np.zeros((len(valid_fft_segs), win_len_64), dtype=np.float32)\n            scales = np.ones(len(valid_fft_segs), dtype=np.float32)\n            target_bins = bin_ids[bin_starts[valid_fft_segs]]\n            for idx_k, seg_i in enumerate(valid_fft_segs):\n                s0 = bin_starts[seg_i]\n                s1 = bin_ends[seg_i]\n                take = min(win_len_64, s1 - s0)\n                bvp_mat[idx_k, :take] = bvp_centered[s0 : s0 + take]\n                scales[idx_k] = float(win_len_64) / float(take)\n            spec = np.abs(np.fft.rfft(bvp_mat, axis=1)) ** 2\n            c_pow = np.mean(spec[:, 7:36], axis=1).astype(np.float32) * scales\n            m_pow = np.mean(spec[:, 40:151], axis=1).astype(np.float32) * scales\n            bvp_cardiac[target_bins] = np.log1p(c_pow)\n            bvp_motion[target_bins] = np.log1p(m_pow)\n            bvp_cardiac_frac[target_bins] = c_pow / (c_pow + m_pow + 1e-6)\n        bvp_cardiac = _ffill_bfill_1d(bvp_cardiac)\n        bvp_motion = _ffill_bfill_1d(bvp_motion)\n        bvp_cardiac_frac = _ffill_bfill_1d(bvp_cardiac_frac)\n        bvp_motion_ratio = bvp_motion - bvp_cardiac\n\n    else:\n        pulse_bins, pulse = observations[0]\n        wave, bands = _optical_features(\n            pulse,\n            asi8[optical_positions],\n            pulse_bins,\n            means[:, 0],\n            start_bin,\n            bin_ns,\n            64 * win_sec,\n        )\n        bvp_diff1_rms, bvp_diff2_rms, bvp_zcr = wave\n        bvp_cardiac, bvp_motion, bvp_cardiac_frac = bands\n        bvp_crest = ptp[:, 0] / (stds[:, 0] + 1e-4)\n        bvp_cardiac = _ffill_bfill_1d(bvp_cardiac)\n        bvp_motion = _ffill_bfill_1d(bvp_motion)\n        bvp_cardiac_frac = _ffill_bfill_1d(bvp_cardiac_frac)\n        bvp_motion_ratio = bvp_motion - bvp_cardiac\n\n    # EDA tonic/phasic decomposition + arousal spike energy\n    eda_log = np.log1p(np.maximum(means[:, 1], 0.0))\n    eda_tonic = _symmetric_rolling_mean(means[:, 1:2], radius=3)[:, 0]\n    eda_phasic = means[:, 1] - eda_tonic\n    eda_phasic_ratio = eda_phasic / (np.maximum(eda_tonic, 0.0) + 0.1)\n\n    e_bins, e_vals = observations[1]\n    eda_rises = np.zeros(n_bins, dtype=np.float32)\n    eda_max_jump = np.zeros(n_bins, dtype=np.float32)\n    if len(e_bins):\n        same_e = e_bins[1:] == e_bins[:-1]\n        if np.any(same_e):\n            ediffs = e_vals[1:][same_e] - e_vals[:-1][same_e]\n            eb = e_bins[1:][same_e]\n            eda_rises = np.bincount(\n                eb, weights=(ediffs > 0.01).astype(np.float64), minlength=n_bins\n            ).astype(np.float32)\n            pos_mask = ediffs > 0\n            if np.any(pos_mask):\n                np.maximum.at(eda_max_jump, eb[pos_mask], ediffs[pos_mask])\n\n    eda_phasic_energy = _symmetric_rolling_mean(np.abs(eda_phasic)[:, None], radius=1)[\n        :, 0\n    ]\n    eda_rises_ctx30 = _symmetric_rolling_mean(eda_rises[:, None], radius=1)[:, 0]\n\n    # Beat-to-beat HRV + IBI_dt step delta (from 42-feature baseline)\n    ibi_bins, ibi_vals = observations[4]\n    rmssd = np.full(n_bins, np.nan, dtype=np.float32)\n    pnn50 = np.full(n_bins, np.nan, dtype=np.float32)\n    if len(ibi_bins):\n        same_bin = ibi_bins[1:] == ibi_bins[:-1]\n        if np.any(same_bin):\n            diffs = ibi_vals[1:][same_bin] - ibi_vals[:-1][same_bin]\n            d_bins = ibi_bins[1:][same_bin]\n            d_cnt = np.bincount(d_bins, minlength=n_bins)\n            d_sq = np.bincount(\n                d_bins, weights=(diffs * diffs).astype(np.float64), minlength=n_bins\n            )\n            d_50 = np.bincount(\n                d_bins,\n                weights=(np.abs(diffs) > 0.05).astype(np.float64),\n                minlength=n_bins,\n            )\n            has_d = d_cnt > 0\n            rmssd[has_d] = np.sqrt(d_sq[has_d] / d_cnt[has_d]).astype(np.float32)\n            pnn50[has_d] = (d_50[has_d] / d_cnt[has_d]).astype(np.float32)\n\n    rmssd = _ffill_bfill_1d(rmssd)\n    pnn50 = _ffill_bfill_1d(pnn50)\n    sdnn = stds[:, 4]\n    ibi_cv = sdnn / (means[:, 4] + 1e-4)\n    ibi_dropout = maxs[:, 3] / (means[:, 4] + 1e-4)\n\n    idt_bins, idt_vals = observations[3]\n    ibi_dt_abs_diff = np.zeros(n_bins, dtype=np.float32)\n    if len(idt_bins):\n        same_idt = idt_bins[1:] == idt_bins[:-1]\n        if np.any(same_idt):\n            adiffs = np.abs(idt_vals[1:][same_idt] - idt_vals[:-1][same_idt]).astype(\n                np.float64\n            )\n            ab = idt_bins[1:][same_idt]\n            acnt = np.bincount(ab, minlength=n_bins)\n            asum = np.bincount(ab, weights=adiffs, minlength=n_bins)\n            has_a = acnt > 0\n            ibi_dt_abs_diff[has_a] = (asum[has_a] / acnt[has_a]).astype(np.float32)\n\n    # Per-recording resting floor & robust z-scores\n    p15_floor = np.percentile(means, 15, axis=0, keepdims=True).astype(np.float32)\n    above_floor = means - p15_floor\n    hr_ratio = means[:, 2] / (p15_floor[0, 2] + 1.0)\n    eda_ratio = means[:, 1] / (p15_floor[0, 1] + 0.1)\n\n    normalized_channels = np.stack(\n        (\n            eda_log,\n            means[:, 2],\n            means[:, 4],\n            means[:, 5],\n            bvp_motion_ratio,\n            rmssd,\n            eda_phasic,\n        )\n    )\n    q25, median, q75 = np.percentile(normalized_channels, [25, 50, 75], axis=1)\n    min_scale = np.array([0.05, 2.0, 0.02, 0.1, 0.2, 0.01, 0.02])\n    scale = np.maximum(\n        np.maximum((q75 - q25) * 0.7413, normalized_channels.std(1)), min_scale\n    )\n    eda_z, hr_z, ibi_z, temp_z, motion_z, rmssd_z, eda_phasic_z = (\n        normalized_channels - median.astype(np.float32)[:, None]\n    ) / scale.astype(np.float32)[:, None]\n\n    # Symmetric context + 42-feature context variability terms\n    sym30 = _symmetric_rolling_mean(means, radius=1)\n    diff_sym30 = means - sym30\n    sym70_sub = _symmetric_rolling_mean(means[:, [1, 2, 5]], radius=3)\n    diff_sym70_sub = means[:, [1, 2, 5]] - sym70_sub\n\n    hr_std_ctx50 = _symmetric_rolling_std_1d(means[:, 2], radius=2)\n    eda_std_ctx50 = _symmetric_rolling_std_1d(means[:, 1], radius=2)\n    eda_ctx70_swing = _symmetric_rolling_ptp_1d(mins[:, 1], maxs[:, 1], radius=3)\n\n    # Cross-modal Stress-vs-Sweat interactions & 60s trends\n    hr_above = above_floor[:, 2]\n    hr_per_bvp_std = hr_above / (stds[:, 0] + 1.0)\n    eda_phasic_x_hr = eda_phasic * hr_above\n    motion_x_hr = bvp_motion_ratio * hr_above\n\n    temp_trend_60s = np.zeros(n_bins, dtype=np.float32)\n    hr_trend_60s = np.zeros(n_bins, dtype=np.float32)\n    eda_trend_60s = np.zeros(n_bins, dtype=np.float32)\n    if n_bins > 6:\n        temp_trend_60s[6:] = means[6:, 5] - means[:-6, 5]\n        hr_trend_60s[6:] = means[6:, 2] - means[:-6, 2]\n        eda_trend_60s[6:] = means[6:, 1] - means[:-6, 1]\n\n    stress_arousal_score = eda_phasic_z + hr_z - motion_z\n    hr_minus_temp_z = hr_z - temp_z\n    rmssd_z_x_hr_z = rmssd_z * hr_z\n\n    X = np.hstack(\n        [\n            means,\n            stds,\n            ptp,\n            delta,\n            within_slope,\n            bvp_diff1_rms[:, None],\n            bvp_diff2_rms[:, None],\n            bvp_zcr[:, None],\n            bvp_crest[:, None],\n            bvp_cardiac[:, None],\n            bvp_motion[:, None],\n            bvp_motion_ratio[:, None],\n            bvp_cardiac_frac[:, None],\n            eda_log[:, None],\n            eda_phasic[:, None],\n            eda_phasic_ratio[:, None],\n            eda_rises[:, None],\n            eda_max_jump[:, None],\n            eda_phasic_energy[:, None],\n            eda_rises_ctx30[:, None],\n            rmssd[:, None],\n            pnn50[:, None],\n            sdnn[:, None],\n            ibi_cv[:, None],\n            ibi_dropout[:, None],\n            ibi_dt_abs_diff[:, None],\n            above_floor,\n            hr_ratio[:, None],\n            eda_ratio[:, None],\n            eda_z[:, None],\n            hr_z[:, None],\n            ibi_z[:, None],\n            temp_z[:, None],\n            motion_z[:, None],\n            rmssd_z[:, None],\n            sym30,\n            diff_sym30,\n            sym70_sub,\n            diff_sym70_sub,\n            hr_std_ctx50[:, None],\n            eda_std_ctx50[:, None],\n            eda_ctx70_swing[:, None],\n            hr_per_bvp_std[:, None],\n            eda_phasic_x_hr[:, None],\n            motion_x_hr[:, None],\n            temp_trend_60s[:, None],\n            hr_trend_60s[:, None],\n            eda_trend_60s[:, None],\n            stress_arousal_score[:, None],\n            hr_minus_temp_z[:, None],\n            rmssd_z_x_hr_z[:, None],\n        ]\n    ).astype(np.float32)\n\n    if order is not None:\n        orig_bin_ids = np.empty_like(bin_ids)\n        orig_bin_ids[order] = bin_ids\n        bin_ids = orig_bin_ids\n\n    return start_bin, bin_ids, X\n\n\ndef smooth_window_probs(probs: np.ndarray) -> np.ndarray:\n    """Smooths window class probabilities across a 50s Gaussian kernel to remove state jitter."""\n    if len(probs) < 3:\n        return probs\n    kernel = np.array([0.10, 0.22, 0.36, 0.22, 0.10], dtype=np.float32)\n    padded = np.pad(probs, ((2, 2), (0, 0)), mode="edge")\n    return (\n        kernel[0] * padded[:-4]\n        + kernel[1] * padded[1:-3]\n        + kernel[2] * padded[2:-2]\n        + kernel[3] * padded[3:-1]\n        + kernel[4] * padded[4:]\n    )\n\n\nPERIOD_NS = {\n    "HR": 1_000_000_000,\n    "EDA": 250_000_000,\n    "Temperature": 250_000_000,\n    "BVP": 15_625_000,\n}\n\nBIN_NS = 10_000_000_000\n\n\ndef beat_pairs(duration, times, reported_dt):\n    """Return consecutive observed beats, allowing one optical sample of timestamp jitter."""\n    elapsed = np.diff(times) / 1e9\n    tolerance = PERIOD_NS["BVP"] / 1e9\n    agrees = ~np.isfinite(reported_dt[1:]) | (\n        np.abs(reported_dt[1:] - elapsed) <= tolerance\n    )\n    return (elapsed > 0) & (np.abs(elapsed - duration[1:]) <= tolerance) & agrees\n\n\ndef recover_intervals(duration, dt, hr, times):\n    """Accept a reported gap as one beat only when timestamps and rate support it."""\n    measured = np.isfinite(duration) & (duration > 0)\n    dt_ok = np.isfinite(dt)\n    if not (dt_ok & ~measured).any():\n        return duration\n    idx = np.flatnonzero(measured | dt_ok)\n    elapsed = np.diff(times[idx]) / 1e9\n    values = dt[idx]\n    consistent = np.r_[False, np.abs(values[1:] - elapsed) <= PERIOD_NS["BVP"] / 1e9]\n    plausible = consistent & (values >= 0.30) & (values <= 2.0)\n    hr_valid = np.flatnonzero(np.isfinite(hr) & (hr >= 30) & (hr <= 210))\n    supported = np.zeros(len(idx), dtype=bool)\n    has_hr = np.zeros(len(idx), dtype=bool)\n    if len(hr_valid):\n        ht = times[hr_valid]\n        j = ht.searchsorted(times[idx])\n        left, right = np.maximum(j - 1, 0), np.minimum(j, len(ht) - 1)\n        near = np.where(\n            abs(ht[left] - times[idx]) <= abs(ht[right] - times[idx]), left, right\n        )\n        has_hr = abs(times[idx] - ht[near]) <= 15_000_000_000\n        expected = 60.0 / hr[hr_valid[near]]\n        supported = has_hr & (values >= 0.65 * expected) & (values <= 1.45 * expected)\n    if len(idx) > 1:\n        similar = (\n            plausible[1:]\n            & plausible[:-1]\n            & (elapsed <= 2.0)\n            & (\n                np.abs(values[1:] - values[:-1])\n                < 0.18 * np.minimum(values[1:], values[:-1])\n            )\n        )\n        supported |= ~has_hr & (np.r_[False, similar] | np.r_[similar, False])\n    take = idx[~measured[idx] & plausible & supported]\n    if not len(take):\n        return duration\n    recovered = duration.copy()\n    recovered[take] = dt[take]\n    return recovered\n\n\ndef repair_hr(hr, intervals, times, positions, clock):\n    valid = np.flatnonzero(np.isfinite(hr))\n    if len(valid) and (\n        times[valid[0]] - times[0] <= 2_000_000_000\n        and times[-1] - times[valid[-1]] <= 2_000_000_000\n        and np.all(np.diff(times[valid]) <= 2_000_000_000)\n    ):\n        return hr\n    recovered = heart_rate_from_beats(intervals, times, positions, clock)\n    if not len(valid):\n        return recovered\n    available = np.flatnonzero(np.isfinite(recovered))\n    if not len(available):\n        return hr\n    observed_seconds = times[valid] // 1_000_000_000\n    seconds = times[available] // 1_000_000_000\n    found = np.searchsorted(observed_seconds, seconds)\n    fill = available[\n        (found == len(valid))\n        | (observed_seconds[np.minimum(found, len(valid) - 1)] != seconds)\n    ]\n    if not len(fill):\n        return hr\n    result = hr.copy()\n    result[fill] = recovered[fill]\n    return result\n\n\ndef heart_rate_from_beats(intervals, times, positions, clock):\n    """Recover a missing HR stream near observed beat intervals, at most once per second."""\n    usable = np.isfinite(intervals) & (intervals >= 0.3) & (intervals <= 2.0)\n    result = np.full(len(times), np.nan)\n    counts = np.bincount(positions[usable], minlength=len(clock))\n    anchors = counts >= 2\n    if not anchors.any():\n        return result\n    sums = np.bincount(\n        positions[usable], weights=intervals[usable], minlength=len(clock)\n    )\n    hr = 60 * counts[anchors] / sums[anchors]\n    # Reference times are relative to the first occupied bin, preserving timestamp precision.\n    anchor_times = (clock[anchors] - clock[0]) * 10.0 + 5.0\n    samples = np.r_[0, np.flatnonzero(np.diff(times // 1_000_000_000)) + 1]\n    seconds = (times[samples] - clock[0] * BIN_NS) / 1e9\n    right = anchor_times.searchsorted(seconds)\n    previous = anchor_times[np.maximum(right - 1, 0)]\n    following = anchor_times[np.minimum(right, len(anchor_times) - 1)]\n    near = np.minimum(np.abs(seconds - previous), np.abs(seconds - following)) <= 30\n    result[samples[near]] = np.interp(seconds[near], anchor_times, hr)\n    return result\n\n\ndef pulse_measurements(values, times, index, counts, rate, spectrum_cache=None):\n    """Waveform shape and spectral concentration using only observed 64 Hz pulse data."""\n    n = len(counts)\n    result = np.full((n, 15), np.nan)\n    starts = np.cumsum(counts) - counts\n    regular = counts == 640\n    discontinuities = np.flatnonzero(np.diff(times) != PERIOD_NS["BVP"])\n    inside_bin = index[discontinuities] == index[discontinuities + 1]\n    regular[index[discontinuities[inside_bin]]] = False\n    complete = np.flatnonzero(regular)\n    if spectrum_cache is not None:\n        spectrum_cache["pulse"] = (times, values, index, counts, starts, complete)\n    grid = np.arange(640)\n    taper = np.hanning(640)\n\n    def measure(bins, signal, complete_observations=False):\n        zeroed = signal - signal.mean(axis=1, keepdims=True)\n        sd = np.sqrt(np.mean(zeroed**2, axis=1))\n        normalized = zeroed / np.maximum(1.0e-7, sd[:, None])\n        square = normalized * normalized\n        derivative = np.diff(normalized, axis=1)\n        power = np.abs(np.fft.rfft(zeroed * taper)) ** 2\n        if complete_observations and spectrum_cache is not None:\n            if "power" not in spectrum_cache:\n                spectrum_cache.update(power=np.empty((n, 321)), seen=np.zeros(n, bool))\n            spectrum_cache["power"][bins] = power\n            spectrum_cache["seen"][bins] = True\n        cardiac = power[:, 6:36]\n        dominant = 6 + np.argmax(cardiac, axis=1)\n        row = np.arange(len(bins))\n        total = np.maximum(1.0e-12, power[:, 4:121].sum(1))\n        fundamental = sum(power[row, dominant + offset] for offset in (-1, 0, 1))\n        harmonic = sum(power[row, 2 * dominant + offset] for offset in (-1, 0, 1))\n        period = np.clip(np.rint(640 / dominant).astype(int), 16, 107)\n        # The central portion avoids wraparound. All samples belong to the same ten seconds.\n        central = np.arange(108, 532, 2)[None, :]\n        autocorr = (\n            normalized[:, central[0]]\n            * normalized[row[:, None], central - period[:, None]]\n        ).mean(1)\n        segment = power[:, 4:75]\n        freq = np.arange(4, 75)\n        centroid = (segment * freq).sum(1) / np.maximum(1.0e-12, segment.sum(1))\n        z = np.column_stack(\n            (\n                np.log1p(sd),\n                np.clip(np.mean(square * normalized, 1), -8, 8),\n                np.log1p(np.mean(square * square, 1)),\n                (derivative > 0).mean(1),\n                np.log1p(np.abs(derivative).mean(1)),\n                dominant * 6,\n                np.abs(rate[bins] - dominant * 6),\n                fundamental / total,\n                np.log1p(harmonic / np.maximum(fundamental, 1.0e-12)),\n                power[:, 36:101].sum(1) / total,\n                np.clip(autocorr, -2, 2),\n                cardiac.sum(1) / total,\n                centroid / 10,\n                power[:, 1:5].sum(1) / np.maximum(power[:, 1:121].sum(1), 1.0e-12),\n                np.mean(np.abs(np.diff(derivative, axis=1)), 1),\n            )\n        )\n        z[sd < 1.0e-7] = np.nan\n        result[bins] = z\n\n    for j in range(0, len(complete), 128):\n        bins = complete[j : j + 128]\n        measure(bins, values[starts[bins, None] + grid], True)\n    partial = np.flatnonzero((counts >= 576) & (counts < 640))\n    for j in range(0, len(partial), 128):\n        bins = []\n        signals = []\n        for b in partial[j : j + 128]:\n            sl = slice(starts[b], starts[b] + counts[b])\n            slots = (times[sl] % BIN_NS) // PERIOD_NS["BVP"]\n            gaps = np.diff(slots)\n            if slots[0] <= 1 and slots[-1] >= 638 and np.all((gaps > 0) & (gaps <= 4)):\n                bins.append(b)\n                signals.append(np.interp(grid, slots, values[sl]))\n        if bins:\n            measure(np.asarray(bins), np.asarray(signals))\n    return result\n\n\ndef physiology_windows(frame, spectrum_cache=None, *, _grid=None):\n    idx = frame.index\n    if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans:\n        raise ValueError("Expected a timestamp index without missing timestamps")\n    if not len(idx):\n        return (\n            np.empty((0, PHYSIOLOGY_WIDTH), np.float32),\n            np.empty(0, int),\n            np.empty(0, np.int64),\n        )\n    if _grid is None:\n        original = idx.as_unit("ns").asi8\n        order = (\n            None if idx.is_monotonic_increasing else np.argsort(original, kind="stable")\n        )\n        times = original if order is None else original[order]\n        bins = times // BIN_NS\n        start = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]\n        clock = bins[start]\n        rows = np.repeat(np.arange(len(clock)), np.diff(np.r_[start, len(times)]))\n    else:\n        times, start_bin, rows = _grid\n        clock = start_bin + np.arange(int(rows[-1]) + 1)\n        order = None\n    n = len(clock)\n\n    def sensor(key):\n        if key not in frame:\n            return np.full(len(times), np.nan)\n        x = _sensor_column(frame, key, np.float64)\n        return x if order is None else x[order]\n\n    def count(ix):\n        return np.bincount(ix, minlength=n)\n\n    def avg(values, groups, counts):\n        return np.divide(\n            np.bincount(groups, weights=values, minlength=n),\n            counts,\n            out=np.full(n, np.nan),\n            where=counts > 0,\n        )\n\n    def stats(v, groups, counts):\n        level = avg(v, groups, counts)\n        return level, np.sqrt(np.maximum(0, avg(v * v, groups, counts) - level**2))\n\n    hr = sensor("HR")\n    dt = sensor("IBI_dt")\n    ibi = recover_intervals(sensor("IBI_duration"), dt, hr, times)\n    hr = repair_hr(hr, ibi, times, rows, clock)\n    output = []\n    rate = np.full(n, np.nan)\n    for key, raw, low, high in (\n        ("HR", hr, 25, 235),\n        ("EDA", sensor("EDA"), 0, 100),\n        ("Temperature", sensor("Temperature"), 0, 50),\n        ("IBI_duration", ibi, 0.30, 2),\n        ("BVP", sensor("BVP"), -1.0e5, 1.0e5),\n    ):\n        valid = np.isfinite(raw) & (raw >= low) & (raw <= high)\n        loc = slice(None) if valid.all() else np.flatnonzero(valid)\n        g, v, t = rows[loc], raw[loc], times[loc]\n        if key == "BVP":\n            k = np.diff(np.searchsorted(g, np.arange(n + 1)))\n            output.extend(pulse_measurements(v, t, g, k, rate, spectrum_cache).T)\n            output.append(np.minimum(k / 640, 1))\n            continue\n        k = count(g)\n        m, sd = stats(v, g, k)\n        if key == "HR":\n            rate = m\n        elif key == "EDA" and spectrum_cache is not None:\n            spectrum_cache["eda"] = (t, v, g)\n        output.extend((m, sd))\n        if key == "IBI_duration":\n            consecutive = beat_pairs(v, t, dt[loc])\n            groups = g[1:][consecutive]\n            delta = np.diff(v)[consecutive]\n            pairs = count(groups)\n            output.extend(\n                (\n                    avg(np.abs(delta), groups, pairs),\n                    np.sqrt(avg(delta * delta, groups, pairs)),\n                    avg(np.abs(delta) > 0.05, groups, pairs),\n                    np.clip(np.bincount(g, weights=v, minlength=n) / 10, 0, 1),\n                    np.minimum(pairs / 12, 1),\n                )\n            )\n            continue\n        elapsed = np.diff(t) / 1.0e9\n        contiguous = np.abs(np.diff(t) - PERIOD_NS[key]) <= PERIOD_NS[key] // 3\n        groups = g[1:][contiguous]\n        step = np.diff(v)[contiguous] / elapsed[contiguous]\n        pairs = count(groups)\n        output.extend((avg(np.abs(step), groups, pairs), avg(step, groups, pairs)))\n        if key == "EDA":\n            output.append(avg(np.maximum(0, step), groups, pairs))\n            # Positive skin conductance deflections against short tonic histories.\n            end = np.arange(1, len(t) + 1)\n            prefix = np.r_[0.0, np.cumsum(v)]\n            even = len(t) > 1 and np.all(np.diff(t) == PERIOD_NS["EDA"])\n            for sec in (2, 8, 20):\n                begin = (\n                    np.maximum(0, end - 1 - 4 * sec)\n                    if even\n                    else t.searchsorted(t - sec * 1_000_000_000)\n                )\n                support = end - begin\n                good = support >= 3 * sec\n                amplitude = (\n                    v[good] - (prefix[end[good]] - prefix[begin[good]]) / support[good]\n                )\n                target = g[good]\n                nn = count(target)\n                output.extend(\n                    (\n                        avg(np.maximum(0, amplitude), target, nn),\n                        avg(amplitude > 0.03, target, nn),\n                    )\n                )\n        output.append(\n            np.minimum(k / ({"HR": 10, "EDA": 40, "Temperature": 40}[key]), 1)\n        )\n    values = np.column_stack(output).astype("float32")\n    if values.shape[1] != PHYSIOLOGY_WIDTH:\n        raise ValueError(f"Expected {PHYSIOLOGY_WIDTH} features, got {values.shape[1]}")\n    if order is not None:\n        aligned = np.empty_like(rows)\n        aligned[order] = rows\n        rows = aligned\n    return values, rows, clock\n\n\nPHYSIOLOGY_WIDTH = 45\n\nUNITS = np.asarray(\n    [\n        30,\n        2,\n        1,\n        1,\n        1,\n        0.5,\n        0.04,\n        0.03,\n        0.03,\n        0.03,\n        0.02,\n        0.25,\n        0.04,\n        0.25,\n        0.08,\n        0.25,\n        1,\n        5,\n        0.1,\n        0.01,\n        0.01,\n        1,\n        0.25,\n        0.03,\n        0.03,\n        0.03,\n        0.10,\n        0.3,\n        1,\n        2,\n        1,\n        1,\n        0.3,\n        0.3,\n        30,\n        10,\n        0.4,\n        0.5,\n        0.2,\n        0.4,\n        0.4,\n        2,\n        0.3,\n        0.3,\n        1,\n    ],\n    dtype=float,\n)\n\n\n# Multiscale summaries retain absolute levels as a reference when clips are short.\nCONTEXT_CHANNELS = np.array(\n    [0, 1, 3, 5, 6, 11, 12, 13, 16, 17, 22, 23, 26, 27, 30, 32, 35, 36, 38, 39, 42, 43]\n)\nFEATURE_SCHEMA = "merged-crop-v2"\n\n\ndef feature_matrix(frame, *, _grid=None):\n    _, _, merged = compute_window_features(frame, _grid=_grid)\n    values, _, clock = physiology_windows(frame, _grid=_grid)\n    if len(merged) != len(clock):\n        raise ValueError(\n            "Expected a contiguous sequence of occupied ten-second windows"\n        )\n    seen = np.isfinite(values)\n    scaled = np.arcsinh(np.where(seen, values, 0.0) / UNITS).astype(np.float32)\n    local = scaled[:, CONTEXT_CHANNELS]\n    valid = seen[:, CONTEXT_CHANNELS]\n    counts = np.vstack([np.zeros(local.shape[1]), valid.cumsum(axis=0)])\n    total = np.vstack(\n        [np.zeros(local.shape[1]), local.cumsum(axis=0, dtype=np.float64)]\n    )\n    sq = np.vstack(\n        [np.zeros(local.shape[1]), (local**2).cumsum(axis=0, dtype=np.float64)]\n    )\n    pos = np.arange(len(clock))\n    parts = [merged, scaled, seen.astype(np.float32)]\n    for radius in (2, 8, 24):\n        l = np.maximum(pos - radius, 0)\n        h = np.minimum(pos + radius + 1, len(clock))\n        count = np.maximum(counts[h] - counts[l], 1)\n        avg = (total[h] - total[l]) / count\n        var = np.maximum((sq[h] - sq[l]) / count - avg**2, 0.0)\n        mid = pos + 1\n        before = (total[mid] - total[l]) / np.maximum(counts[mid] - counts[l], 1)\n        after = (total[h] - total[pos]) / np.maximum(counts[h] - counts[pos], 1)\n        parts.extend(\n            [avg, np.sqrt(var), np.where(valid, local - avg, 0), after - before]\n        )\n    # Recording-level references use a stabilised scale; missing channels keep their own indicators.\n    ordered = np.sort(np.where(valid, local, np.inf), axis=0)\n    size = valid.sum(0)\n    features = np.arange(local.shape[1])\n    quantiles = []\n    for q in (0.15, 0.50, 0.85):\n        index = np.floor(q * np.maximum(size - 1, 0)).astype(int)\n        quantiles.append(np.where(size > 0, ordered[index, features], 0.0))\n    lo, med, hi = quantiles\n    parts.extend(\n        [\n            np.where(valid, local - med, 0),\n            np.where(valid, (local - med) / (0.08 + hi - lo), 0),\n            np.broadcast_to(hi - lo, local.shape),\n        ]\n    )\n    return np.nan_to_num(\n        np.column_stack(parts).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0\n    )\n\n\ndef recording_segments(frame, preserve_reference_grid=False, *, slices=False):\n    """Yield features and original row positions, without padding long timestamp gaps."""\n    time = pd.DatetimeIndex(frame.index).as_unit("ns").asi8\n    sorted_index = frame.index.is_monotonic_increasing\n    order = None if sorted_index else np.argsort(time, kind="stable")\n    ordered_time = time if sorted_index else time[order]\n    ticks = ordered_time // BIN_NS\n    steps = np.diff(ticks)\n    occupied = 1 + np.count_nonzero(steps)\n    # The original predictor fills empty bins. Preserve its grid when finite gaps\n    # fit in memory, while bounding the allocation for sparse multi-day input.\n    if preserve_reference_grid and ticks[-1] - ticks[0] + 1 <= max(128, 4 * occupied):\n        edges = np.array([0, len(frame)])\n    else:\n        edges = np.r_[0, np.flatnonzero(steps > 1) + 1, len(frame)]\n    for left, right in zip(edges[:-1], edges[1:]):\n        if right > left:\n            if sorted_index:\n                rows = slice(left, right) if slices else np.arange(left, right)\n                section = (\n                    frame\n                    if left == 0 and right == len(frame)\n                    else frame.iloc[left:right]\n                )\n            else:\n                rows = order[left:right]\n                section = frame.iloc[rows]\n            ids = ticks[left:right] - ticks[left]\n            yield rows, ids, ticks[left], section\n\n\ndef feature_segments(frame):\n    for rows, ids, start, section in recording_segments(frame):\n        grid = (section.index.as_unit("ns").asi8, start, ids)\n        yield rows, ids, start, feature_matrix(section, _grid=grid)\n\n\ndef has_reported_cardiac_data(frame):\n    for signal, low, high in (("HR", 25, 240), ("IBI_duration", 0.2, 2.5)):\n        if signal in frame:\n            values = _sensor_column(frame, signal, np.float32)\n            if np.any((values >= low) & (values <= high)):\n                return True\n    return False\n\n\ndef inference_grid(section, grid):\n    """Form complete ten-second windows when a caller cuts through the original grid.\n\n    Keep the input index for the result. A trailing fragment outside the modeled\n    windows receives the last available prediction. All context is local to this call.\n    """\n    times, start_bin, row_bins = grid\n    edge_tolerance = 125_000_000\n    phase = int(times[0] % BIN_NS)\n    last_phase = int(times[-1] % BIN_NS)\n    count = int(row_bins[-1]) + 1\n    if count < 2 or phase <= edge_tolerance:\n        return section, grid, row_bins\n    span = int(times[-1] - times[0])\n    remainder = span % BIN_NS\n    last_partial = last_phase < BIN_NS - edge_tolerance\n    complete_on_original_grid = count - 1 - int(last_partial)\n    complete_from_start = (span + edge_tolerance) // BIN_NS\n    near_boundary = remainder <= edge_tolerance or remainder >= BIN_NS - edge_tolerance\n    recovers_window = near_boundary and complete_from_start > max(\n        0, complete_on_original_grid\n    )\n    # Twenty-four bins is the longest context radius in the trained model.\n    # In short clips both partial ends can influence the entire recording context.\n    short_clip = count <= 25 and last_partial and span >= BIN_NS\n    if not (recovers_window or short_clip):\n        return section, grid, row_bins\n    shifted_times = times - phase\n    shifted_bins = shifted_times // BIN_NS - start_bin\n    if np.any(np.diff(shifted_bins) > 1):\n        return section, grid, row_bins\n    end = (\n        shifted_bins.searchsorted(complete_from_start)\n        if short_clip or remainder <= edge_tolerance\n        else len(section)\n    )\n    if end == 0:\n        return section, grid, row_bins\n    if end < len(section):\n        analysis = (shifted_times[:end], start_bin, shifted_bins[:end])\n        return (\n            section.iloc[:end],\n            analysis,\n            np.minimum(shifted_bins, shifted_bins[end - 1]),\n        )\n    return section, (shifted_times, start_bin, shifted_bins), shifted_bins\n\n\n\ndef fares_light_matrix(frame, *, _grid=None):\n    """Fares\' feature_matrix without the window-stat block (compute_window_features) and seen flags:\n    physiology channels + their r2/8/24 context + recording-relative terms. Same maths otherwise."""\n    values, _, clock = physiology_windows(frame, _grid=_grid)\n    seen = np.isfinite(values)\n    scaled = np.arcsinh(np.where(seen, values, 0.0) / UNITS).astype(np.float32)\n    local = scaled[:, CONTEXT_CHANNELS]\n    valid = seen[:, CONTEXT_CHANNELS]\n    counts = np.vstack([np.zeros(local.shape[1]), valid.cumsum(axis=0)])\n    total = np.vstack([np.zeros(local.shape[1]), local.cumsum(axis=0, dtype=np.float64)])\n    sq = np.vstack([np.zeros(local.shape[1]), (local ** 2).cumsum(axis=0, dtype=np.float64)])\n    pos = np.arange(len(clock))\n    parts = [scaled]\n    for radius in (2, 8, 24):\n        l = np.maximum(pos - radius, 0); h = np.minimum(pos + radius + 1, len(clock))\n        count = np.maximum(counts[h] - counts[l], 1)\n        avg = (total[h] - total[l]) / count\n        var = np.maximum((sq[h] - sq[l]) / count - avg ** 2, 0.0)\n        mid = pos + 1\n        before = (total[mid] - total[l]) / np.maximum(counts[mid] - counts[l], 1)\n        after = (total[h] - total[pos]) / np.maximum(counts[h] - counts[pos], 1)\n        parts.extend([avg, np.sqrt(var), np.where(valid, local - avg, 0), after - before])\n    ordered = np.sort(np.where(valid, local, np.inf), axis=0)\n    size = valid.sum(0); features = np.arange(local.shape[1]); qs = []\n    for q in (0.15, 0.50, 0.85):\n        index = np.floor(q * np.maximum(size - 1, 0)).astype(int)\n        qs.append(np.where(size > 0, ordered[index, features], 0.0))\n    lo, med, hi = qs\n    parts.extend([np.where(valid, local - med, 0), np.where(valid, (local - med) / (0.08 + hi - lo), 0), np.broadcast_to(hi - lo, local.shape)])\n    return np.nan_to_num(np.column_stack(parts).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)\n\n\n\ndef light_segments(frame):\n    for rows, ids, start, section in recording_segments(frame):\n        grid = (section.index.as_unit("ns").asi8, start, ids)\n        yield rows, ids, start, fares_light_matrix(section, _grid=grid)\n'
_FARES_FN = 'light_segments'


def _fares_feats(tt, cols):
    """Fares' feature_matrix per contiguous segment, laid on our 10 s grid (gaps -> zeros)."""
    global _FARES
    if _FARES is None:
        import types
        _FARES = types.ModuleType("_fares")
        exec(compile(_FARES_SRC, "_fares", "exec"), _FARES.__dict__)
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(tt.view("datetime64[ns]")))
    n_win = int(tt[-1] // WIN_NS - tt[0] // WIN_NS) + 1
    out = np.zeros((n_win, FARES_DIM), dtype=np.float32)
    for rows, ids, start, F in getattr(_FARES, _FARES_FN)(df):
        off = int(start - tt[0] // WIN_NS)
        m = min(len(F), n_win - off)
        if m > 0:
            out[off:off + m] = F[:m]
    return out


def _proto_agg(X):
    B = X[:, :17].astype(np.float64)
    B[:, :4] = np.log1p(np.abs(B[:, :4])); B[:, 13] = np.log1p(np.abs(B[:, 13]))
    return np.concatenate([B.mean(0), B.std(0), np.percentile(B, 10, 0), np.percentile(B, 90, 0), [np.log1p(len(X))]])


def _proto_pe(X):
    z = (_proto_agg(X) - np.asarray(PROTO["mean"])) / np.asarray(PROTO["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(PROTO["coef"]) + PROTO["intercept"])))


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

    def _segment(self, tt, cols):
        _, X, _ = compute(_Frame(tt, cols), **FEAT_KW)
        if FARES_DIM:
            Xf = np.zeros((len(X), FARES_DIM), dtype=np.float32)
            try:
                F = _fares_feats(tt, cols)
                n = min(len(F), len(X))
                Xf[:n] = F[:n]
            except Exception:
                pass
            X = np.hstack([X, Xf])
        P = np.asarray(self.session.run(None, {self.input_name: X})[self.prob_idx], dtype=np.float64)
        if PROTO is not None:
            pe = _proto_pe(X)
            P = P * np.array([1.0, (1.0 - pe) ** PROTO["gamma"], pe ** PROTO["gamma"]])
        if GATE is not None:
            # whole input looks like an exercise session -> no Stress, else no Activity
            if P[:, 2].mean() > GATE[0] * P[:, 1].mean():
                P[:, 1] *= GATE[1]
            else:
                P[:, 2] *= GATE[1]
        pred = (_smooth(P, SMOOTH) * CLASS_WEIGHTS).argmax(1)
        return LABELS[pred[tt // WIN_NS - tt[0] // WIN_NS]]

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
        # inputs holding several sessions separated by long gaps: process each segment on its own
        cuts = np.flatnonzero(np.diff(tt) > SPLIT_GAP_NS) + 1
        bounds = np.r_[0, cuts, len(tt)]
        lab = np.empty(len(tt), dtype=object)
        for a, b in zip(bounds[:-1], bounds[1:]):
            lab[a:b] = self._segment(tt[a:b], {c: v[a:b] for c, v in cols.items()} if len(bounds) > 2 else cols)
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
