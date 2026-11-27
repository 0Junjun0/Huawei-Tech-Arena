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
PROTO = {'mean': [3.7833713897252026, 5.4613915825761925, 2.1271188293924035, 3.5320496825390464, 6.4854099472668345, 0.17981396070363548, 0.6513305928021211, 90.83563253540936, 0.5387919357798313, 1.6495415708799273, 0.6126236305013995, 0.027634024647849215, 4.890162021736505, 0.007121798399492656, 36.24871852134361, 0.10443789123462434, 0.2862591937699698, 0.37288762425338884, 0.49356982549570116, 0.22953279566558227, 0.4201308697136661, 0.9568648381276251, 0.14536973841737985, 0.48846255247183584, 6.7049852816228235, 0.4473294318656689, 1.3696497134250563, 0.05800657415839747, 0.016729198997956468, 3.491821114752219, 0.009028822189465179, 0.5181184482825716, 0.10205514684072407, 0.25294753434512474, 3.363116904864996, 4.913577027497319, 1.8670409388204006, 3.079426501321057, 5.3387703307051755, 0.06192684245276547, 0.2337514864624504, 82.62314952977027, 0.12237083220195698, 0.37126230066367233, 0.5470870701763387, 0.009840943086459919, 1.23447109471094, 0.0009244621779278696, 35.6599296170994, 0.04229603568386524, 0.11705166030667023, 4.234310759229343, 6.0673022189875345, 2.4046931534168743, 4.035742952804095, 7.691494904594804, 0.34012664816471605, 1.2171792763991764, 99.50587694290105, 1.1081141223279316, 3.3935581184593717, 0.6804834654401217, 0.04790226647727408, 9.122878228782328, 0.016748156760912875, 36.84184118677914, 0.15450698830690515, 0.44104581566321993, 3.0020992761247705], 'scale': [0.7084682455213946, 0.7832278939729683, 0.8160565049848785, 0.8789620940445018, 10.849656881765146, 0.2877553211824921, 1.0223015350676312, 22.498257188413746, 0.4642330374399055, 1.3938439606979534, 0.23985413298203592, 0.018359669065547028, 4.033874864437086, 0.01691091937041647, 26.686532079114194, 0.5138433201480088, 1.2845599233352758, 0.20843798914543665, 0.2440232669185737, 0.16163272594941636, 0.25505863901270753, 2.2006431424940023, 0.3203012016099856, 1.0124283153680942, 6.643824726865527, 0.5268748912548562, 1.5733734540443303, 0.056411589505442075, 0.012494111494841656, 1.9787126161846862, 0.020391649977727973, 4.876169187528644, 0.9415958447818338, 1.936921268845325, 0.7430804039225697, 0.8195701834051083, 0.8845750227333737, 0.9966538861922482, 9.640997799954555, 0.09771530525927172, 0.3720912132493888, 21.56470704927939, 0.12142009681376381, 0.3596357935211898, 0.2206827577197809, 0.012302091766156827, 2.457610227231486, 0.002452761254341057, 25.65848010139247, 0.2870782065353435, 0.7724344343422568, 0.7556477012743953, 0.8392459396089466, 0.7838521504378666, 0.8365722095938087, 12.49798127402476, 0.6253104838172958, 2.149030612069823, 25.76650598606262, 1.1318214947314882, 3.34458948770908, 0.2716382385094381, 0.03221424440732541, 5.949517998084012, 0.04447357817279801, 28.529980799730236, 0.6968701894200267, 1.866870189140703, 0.635995310460891], 'coef': [-0.8869933246416832, 0.385385550651982, 0.4370924870466972, -1.1814879691447222, -0.23005390877979148, -0.286467251687647, 0.24896813658110095, 1.3273764940530592, -0.21736657602980422, -0.529444507280335, -0.5386176543811796, -0.5549384390541895, -1.9416299027531632, 1.2760625882073628, 0.08880831628477416, 0.8563458060531974, 1.1566559457829897, 0.7765189180625133, -0.599450033820919, 0.23183031661901263, -0.42697203404567086, 0.4253163756293744, 0.2499689807673659, -0.12993501782912045, -0.6568470011766219, 1.1598890906207482, 0.6811299042664276, -0.031617977314376065, 0.14141040191011361, 1.0992899077978329, -0.73911236850514, 0.7091964506914887, 0.2784491922981438, 0.4355921878114116, -0.45627510873751576, 1.0283810197996635, 0.7963543957051809, -1.008628999508636, 0.020673364864157044, -0.11194661078985195, 0.840442808514541, -0.5156706262649123, 0.8500262406337625, -0.6710830000640416, -0.21915004910824135, 0.04992580316657649, 0.5870135247637045, 0.40932152070872835, -0.08928665092211081, 0.7849238214198068, 0.9654707025969101, -0.6171281093402534, 0.04505636213320424, 0.8492096488343387, -0.20802953065769073, -0.13009524824039378, -0.4977830515326527, 0.7123693288177216, 0.8731903887626717, -0.21693745038029716, -0.2395192809180592, -0.10551294938064555, -0.2157501225639662, -0.2750916023379309, 0.4602310368660725, 0.22628517160419628, 0.9914391793953582, 1.280523109973697, 0.19847936829720883], 'intercept': 2.4534503985355065, 'gamma': 1.0}  # logistic regression: P(input is an exercise session) from input aggregates, or None


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
