# This section enforces single-threaded execution across all numerical libraries before importing them.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pathlib
import numpy as np
import pandas as pd
import onnxruntime as ort
from adm_base_class import ADM_Base

LABELS = ["Rest", "Stress", "Activity"]
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
# Auto-updated by train_model.py after nested out-of-fold Macro-F1 tuning
CLASS_WEIGHTS = [1.0, 1.0, 0.92]


def _ffill_bfill_2d(arr: np.ndarray) -> np.ndarray:
    """Forward-fills missing window features and backfills only leading empty bins."""
    n_bins, n_cols = arr.shape
    if n_bins == 0:
        return arr
    mask = np.isnan(arr)
    idx_f = np.where(~mask, np.arange(n_bins)[:, None], 0)
    np.maximum.accumulate(idx_f, axis=0, out=idx_f)
    out = arr[idx_f, np.arange(n_cols)]

    mask2 = np.isnan(out)
    if np.any(mask2):
        idx_b = np.where(~mask2, np.arange(n_bins)[:, None], n_bins - 1)
        idx_b = np.minimum.accumulate(idx_b[::-1], axis=0)[::-1]
        out = out[idx_b, np.arange(n_cols)]
    return np.nan_to_num(out, nan=0.0)


def _ffill_bfill_1d(arr: np.ndarray) -> np.ndarray:
    """Forward-fills a 1D feature array and backfills only leading missing values."""
    n = len(arr)
    if n == 0:
        return arr
    m = np.isnan(arr)
    if np.all(m):
        return np.zeros(n, dtype=np.float32)
    idx_f = np.where(~m, np.arange(n), 0)
    np.maximum.accumulate(idx_f, out=idx_f)
    out = arr[idx_f]

    m2 = np.isnan(out)
    if np.any(m2):
        idx_b = np.where(~m2, np.arange(n), n - 1)
        idx_b = np.minimum.accumulate(idx_b[::-1])[::-1]
        out = out[idx_b]
    return np.nan_to_num(out, nan=0.0)


def _symmetric_rolling_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast centered rolling average over [i - radius, i + radius] via cumulative sums."""
    n = len(arr)
    if n == 0:
        return arr
    cs = np.zeros((n + 1, arr.shape[1]), dtype=np.float64)
    cs[1:] = np.cumsum(arr, axis=0, dtype=np.float64)
    idx = np.arange(n)
    left = np.maximum(0, idx - radius)
    right = np.minimum(n, idx + radius + 1)
    counts = (right - left)[:, None]
    return ((cs[right] - cs[left]) / counts).astype(np.float32)


def _symmetric_rolling_std_1d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast centered rolling std over [i - radius, i + radius] via prefix sums."""
    n = len(arr)
    if n == 0:
        return arr
    x = arr.astype(np.float64)
    cs1 = np.zeros(n + 1, dtype=np.float64)
    cs2 = np.zeros(n + 1, dtype=np.float64)
    cs1[1:] = np.cumsum(x)
    cs2[1:] = np.cumsum(x * x)
    idx = np.arange(n)
    left = np.maximum(0, idx - radius)
    right = np.minimum(n, idx + radius + 1)
    cnt = right - left
    mean = (cs1[right] - cs1[left]) / cnt
    var = np.maximum(0.0, (cs2[right] - cs2[left]) / cnt - mean * mean)
    return np.sqrt(var).astype(np.float32)


def _symmetric_rolling_ptp_1d(mins_1d: np.ndarray, maxs_1d: np.ndarray, radius: int) -> np.ndarray:
    """Centered peak-to-peak swing across [i - radius, i + radius] windows."""
    n = len(mins_1d)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        l = 0 if i < radius else i - radius
        r = n if i + radius + 1 > n else i + radius + 1
        out[i] = float(np.max(maxs_1d[l:r]) - np.min(mins_1d[l:r]))
    return out


def _robust_z_1d(arr: np.ndarray, min_scale: float = 1e-3) -> np.ndarray:
    """Per-recording robust z-score using median and IQR to normalize subject reactivity."""
    if len(arr) == 0:
        return arr
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    iqr = float(q75 - q25)
    std = float(np.std(arr))
    scale = max(iqr * 0.7413, std, min_scale)
    return ((arr - float(q50)) / scale).astype(np.float32)


def _extract_sensor_array(df: pd.DataFrame, col_name: str, order: np.ndarray | None) -> np.ndarray:
    """Safely extracts a float32 sensor column (NaN-filled if missing) and applies sort permutation if needed."""
    if col_name in df.columns:
        arr = df[col_name].to_numpy(dtype=np.float32, na_value=np.nan)
    else:
        arr = np.full(len(df), np.nan, dtype=np.float32)
    if order is not None:
        arr = arr[order]
    return arr


def compute_window_features(df: pd.DataFrame, win_sec: int = 10) -> tuple[int, np.ndarray, np.ndarray]:
    """Extracts 95 defensive, vectorized physiological, spectral, HRV, context, and subject-normalized features."""
    dt_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    dt_index = dt_index.as_unit("ns")

    if dt_index.is_monotonic_increasing:
        order = None
        asi8 = dt_index.asi8
    else:
        order = np.argsort(dt_index.asi8, kind="mergesort")
        asi8 = dt_index.asi8[order]

    bin_ns = int(win_sec * 1_000_000_000)
    raw_bins = asi8 // bin_ns
    start_bin = int(raw_bins[0])
    end_bin = int(raw_bins[-1])
    n_bins = end_bin - start_bin + 1
    bin_ids = raw_bins - start_bin

    cols_data = [_extract_sensor_array(df, c, order) for c in SENSOR_COLS]

    # Mask physiologically impossible sensor spikes (e.g., 200C Temperature glitches in S02/S05/S07)
    cols_data[1] = np.where((cols_data[1] >= 0.0) & (cols_data[1] <= 150.0), cols_data[1], np.nan)
    cols_data[2] = np.where((cols_data[2] >= 25.0) & (cols_data[2] <= 240.0), cols_data[2], np.nan)
    cols_data[4] = np.where((cols_data[4] >= 0.20) & (cols_data[4] <= 2.50), cols_data[4], np.nan)
    cols_data[5] = np.where((cols_data[5] >= 15.0) & (cols_data[5] <= 45.0), cols_data[5], np.nan)

    n_cols = len(cols_data)
    means = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    stds = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    mins = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    maxs = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    within_slope = np.zeros((n_bins, n_cols), dtype=np.float32)

    for c, col in enumerate(cols_data):
        valid = ~np.isnan(col)
        if not np.any(valid):
            continue
        b = bin_ids[valid]
        v_f32 = col[valid]
        v = v_f32.astype(np.float64)

        cnt = np.bincount(b, minlength=n_bins)
        s1 = np.bincount(b, weights=v, minlength=n_bins)
        s2 = np.bincount(b, weights=v * v, minlength=n_bins)

        has = cnt > 0
        m = np.zeros(n_bins, dtype=np.float64)
        m[has] = s1[has] / cnt[has]
        var = np.zeros(n_bins, dtype=np.float64)
        var[has] = np.maximum(0.0, s2[has] / cnt[has] - m[has] * m[has])

        means[has, c] = m[has].astype(np.float32)
        stds[has, c] = np.sqrt(var[has]).astype(np.float32)

        change_idx = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        end_idx = np.r_[change_idx[1:] - 1, len(b) - 1]
        active_bins = b[change_idx]
        mins[active_bins, c] = np.minimum.reduceat(v_f32, change_idx)
        maxs[active_bins, c] = np.maximum.reduceat(v_f32, change_idx)
        within_slope[active_bins, c] = v_f32[end_idx] - v_f32[change_idx]

    means = _ffill_bfill_2d(means)
    stds = _ffill_bfill_2d(stds)
    mins = _ffill_bfill_2d(mins)
    maxs = _ffill_bfill_2d(maxs)
    ptp = maxs - mins
    delta = np.zeros_like(means)
    delta[1:] = means[1:] - means[:-1]

    # 64 Hz BVP waveform sharpness, zero-crossing rate, and bin-aligned cardiac-vs-motion FFT energy
    bvp = np.nan_to_num(cols_data[0], nan=0.0)
    bvp_centered = bvp - means[bin_ids, 0]
    same_b = bin_ids[1:] == bin_ids[:-1]

    d1 = np.diff(bvp)
    d1_sq = np.bincount(bin_ids[1:][same_b], weights=(d1[same_b] ** 2).astype(np.float64), minlength=n_bins)
    d1_cnt = np.maximum(1, np.bincount(bin_ids[1:][same_b], minlength=n_bins))
    bvp_diff1_rms = np.sqrt(d1_sq / d1_cnt).astype(np.float32)

    d2 = np.diff(d1)
    same_b2 = bin_ids[2:] == bin_ids[:-2]
    d2_sq = np.bincount(bin_ids[2:][same_b2], weights=(d2[same_b2] ** 2).astype(np.float64), minlength=n_bins)
    d2_cnt = np.maximum(1, np.bincount(bin_ids[2:][same_b2], minlength=n_bins))
    bvp_diff2_rms = np.sqrt(d2_sq / d2_cnt).astype(np.float32)

    zc = ((bvp_centered[1:] * bvp_centered[:-1]) < 0) & same_b
    bvp_zcr = (np.bincount(bin_ids[1:][zc], minlength=n_bins) / d1_cnt).astype(np.float32)
    bvp_crest = ptp[:, 0] / (stds[:, 0] + 1e-4)

    win_len_64 = win_sec * 64
    bvp_cardiac = np.full(n_bins, np.nan, dtype=np.float32)
    bvp_motion = np.full(n_bins, np.nan, dtype=np.float32)
    bvp_cardiac_frac = np.full(n_bins, np.nan, dtype=np.float32)
    bin_starts = np.flatnonzero(np.r_[True, bin_ids[1:] != bin_ids[:-1]])
    bin_ends = np.r_[bin_starts[1:], len(bvp)]
    seg_lens = bin_ends - bin_starts
    valid_fft_segs = np.flatnonzero(seg_lens >= 64)
    if len(valid_fft_segs) > 0:
        bvp_mat = np.zeros((len(valid_fft_segs), win_len_64), dtype=np.float32)
        scales = np.ones(len(valid_fft_segs), dtype=np.float32)
        target_bins = bin_ids[bin_starts[valid_fft_segs]]
        for idx_k, seg_i in enumerate(valid_fft_segs):
            s0 = bin_starts[seg_i]
            s1 = bin_ends[seg_i]
            take = min(win_len_64, s1 - s0)
            bvp_mat[idx_k, :take] = bvp_centered[s0 : s0 + take]
            scales[idx_k] = float(win_len_64) / float(take)
        spec = np.abs(np.fft.rfft(bvp_mat, axis=1)) ** 2
        c_pow = np.mean(spec[:, 7:36], axis=1).astype(np.float32) * scales
        m_pow = np.mean(spec[:, 40:151], axis=1).astype(np.float32) * scales
        bvp_cardiac[target_bins] = np.log1p(c_pow)
        bvp_motion[target_bins] = np.log1p(m_pow)
        bvp_cardiac_frac[target_bins] = c_pow / (c_pow + m_pow + 1e-6)
    bvp_cardiac = _ffill_bfill_1d(bvp_cardiac)
    bvp_motion = _ffill_bfill_1d(bvp_motion)
    bvp_cardiac_frac = _ffill_bfill_1d(bvp_cardiac_frac)
    bvp_motion_ratio = bvp_motion - bvp_cardiac

    # EDA tonic/phasic decomposition + arousal spike energy
    eda_log = np.log1p(np.maximum(means[:, 1], 0.0))
    eda_tonic = _symmetric_rolling_mean(means[:, 1:2], radius=3)[:, 0]
    eda_phasic = means[:, 1] - eda_tonic
    eda_phasic_ratio = eda_phasic / (np.maximum(eda_tonic, 0.0) + 0.1)

    eda_col = cols_data[1]
    valid_eda = ~np.isnan(eda_col)
    eda_rises = np.zeros(n_bins, dtype=np.float32)
    eda_max_jump = np.zeros(n_bins, dtype=np.float32)
    if np.any(valid_eda):
        e_bins = bin_ids[valid_eda]
        e_vals = eda_col[valid_eda]
        same_e = e_bins[1:] == e_bins[:-1]
        if np.any(same_e):
            ediffs = e_vals[1:][same_e] - e_vals[:-1][same_e]
            eb = e_bins[1:][same_e]
            eda_rises = np.bincount(eb, weights=(ediffs > 0.01).astype(np.float64), minlength=n_bins).astype(np.float32)
            pos_mask = ediffs > 0
            if np.any(pos_mask):
                np.maximum.at(eda_max_jump, eb[pos_mask], ediffs[pos_mask])

    eda_phasic_energy = _symmetric_rolling_mean(np.abs(eda_phasic)[:, None], radius=1)[:, 0]
    eda_rises_ctx30 = _symmetric_rolling_mean(eda_rises[:, None], radius=1)[:, 0]

    # Beat-to-beat HRV + IBI_dt step delta (from 42-feature baseline)
    ibi_col = cols_data[4]
    valid_ibi = ~np.isnan(ibi_col)
    rmssd = np.full(n_bins, np.nan, dtype=np.float32)
    pnn50 = np.full(n_bins, np.nan, dtype=np.float32)
    if np.any(valid_ibi):
        ibi_bins = bin_ids[valid_ibi]
        ibi_vals = ibi_col[valid_ibi]
        same_bin = ibi_bins[1:] == ibi_bins[:-1]
        if np.any(same_bin):
            diffs = ibi_vals[1:][same_bin] - ibi_vals[:-1][same_bin]
            d_bins = ibi_bins[1:][same_bin]
            d_cnt = np.bincount(d_bins, minlength=n_bins)
            d_sq = np.bincount(d_bins, weights=(diffs * diffs).astype(np.float64), minlength=n_bins)
            d_50 = np.bincount(d_bins, weights=(np.abs(diffs) > 0.05).astype(np.float64), minlength=n_bins)
            has_d = d_cnt > 0
            rmssd[has_d] = np.sqrt(d_sq[has_d] / d_cnt[has_d]).astype(np.float32)
            pnn50[has_d] = (d_50[has_d] / d_cnt[has_d]).astype(np.float32)

    rmssd = _ffill_bfill_1d(rmssd)
    pnn50 = _ffill_bfill_1d(pnn50)
    sdnn = stds[:, 4]
    ibi_cv = sdnn / (means[:, 4] + 1e-4)
    ibi_dropout = maxs[:, 3] / (means[:, 4] + 1e-4)

    ibi_dt_col = cols_data[3]
    valid_idt = ~np.isnan(ibi_dt_col)
    ibi_dt_abs_diff = np.zeros(n_bins, dtype=np.float32)
    if np.any(valid_idt):
        idt_bins = bin_ids[valid_idt]
        idt_vals = ibi_dt_col[valid_idt]
        same_idt = idt_bins[1:] == idt_bins[:-1]
        if np.any(same_idt):
            adiffs = np.abs(idt_vals[1:][same_idt] - idt_vals[:-1][same_idt]).astype(np.float64)
            ab = idt_bins[1:][same_idt]
            acnt = np.bincount(ab, minlength=n_bins)
            asum = np.bincount(ab, weights=adiffs, minlength=n_bins)
            has_a = acnt > 0
            ibi_dt_abs_diff[has_a] = (asum[has_a] / acnt[has_a]).astype(np.float32)

    # Per-recording resting floor & robust z-scores
    p15_floor = np.percentile(means, 15, axis=0, keepdims=True).astype(np.float32)
    above_floor = means - p15_floor
    hr_ratio = means[:, 2] / (p15_floor[0, 2] + 1.0)
    eda_ratio = means[:, 1] / (p15_floor[0, 1] + 0.1)

    eda_z = _robust_z_1d(eda_log, min_scale=0.05)
    hr_z = _robust_z_1d(means[:, 2], min_scale=2.0)
    ibi_z = _robust_z_1d(means[:, 4], min_scale=0.02)
    temp_z = _robust_z_1d(means[:, 5], min_scale=0.1)
    motion_z = _robust_z_1d(bvp_motion_ratio, min_scale=0.2)
    rmssd_z = _robust_z_1d(rmssd, min_scale=0.01)
    eda_phasic_z = _robust_z_1d(eda_phasic, min_scale=0.02)

    # Symmetric context + 42-feature context variability terms
    sym30 = _symmetric_rolling_mean(means, radius=1)
    diff_sym30 = means - sym30
    sym70_sub = _symmetric_rolling_mean(means[:, [1, 2, 5]], radius=3)
    diff_sym70_sub = means[:, [1, 2, 5]] - sym70_sub

    hr_std_ctx50 = _symmetric_rolling_std_1d(means[:, 2], radius=2)
    eda_std_ctx50 = _symmetric_rolling_std_1d(means[:, 1], radius=2)
    eda_ctx70_swing = _symmetric_rolling_ptp_1d(mins[:, 1], maxs[:, 1], radius=3)

    # Cross-modal Stress-vs-Sweat interactions & 60s trends
    hr_above = above_floor[:, 2]
    hr_per_bvp_std = hr_above / (stds[:, 0] + 1.0)
    eda_phasic_x_hr = eda_phasic * hr_above
    motion_x_hr = bvp_motion_ratio * hr_above

    temp_trend_60s = np.zeros(n_bins, dtype=np.float32)
    hr_trend_60s = np.zeros(n_bins, dtype=np.float32)
    eda_trend_60s = np.zeros(n_bins, dtype=np.float32)
    if n_bins > 6:
        temp_trend_60s[6:] = means[6:, 5] - means[:-6, 5]
        hr_trend_60s[6:] = means[6:, 2] - means[:-6, 2]
        eda_trend_60s[6:] = means[6:, 1] - means[:-6, 1]

    stress_arousal_score = eda_phasic_z + hr_z - motion_z
    hr_minus_temp_z = hr_z - temp_z
    rmssd_z_x_hr_z = rmssd_z * hr_z

    X = np.hstack([
        means, stds, ptp, delta, within_slope,
        bvp_diff1_rms[:, None], bvp_diff2_rms[:, None], bvp_zcr[:, None],
        bvp_crest[:, None], bvp_cardiac[:, None], bvp_motion[:, None],
        bvp_motion_ratio[:, None], bvp_cardiac_frac[:, None],
        eda_log[:, None], eda_phasic[:, None], eda_phasic_ratio[:, None],
        eda_rises[:, None], eda_max_jump[:, None],
        eda_phasic_energy[:, None], eda_rises_ctx30[:, None],
        rmssd[:, None], pnn50[:, None], sdnn[:, None], ibi_cv[:, None],
        ibi_dropout[:, None], ibi_dt_abs_diff[:, None],
        above_floor, hr_ratio[:, None], eda_ratio[:, None],
        eda_z[:, None], hr_z[:, None], ibi_z[:, None], temp_z[:, None],
        motion_z[:, None], rmssd_z[:, None],
        sym30, diff_sym30, sym70_sub, diff_sym70_sub,
        hr_std_ctx50[:, None], eda_std_ctx50[:, None], eda_ctx70_swing[:, None],
        hr_per_bvp_std[:, None], eda_phasic_x_hr[:, None],
        motion_x_hr[:, None], temp_trend_60s[:, None],
        hr_trend_60s[:, None], eda_trend_60s[:, None],
        stress_arousal_score[:, None], hr_minus_temp_z[:, None], rmssd_z_x_hr_z[:, None],
    ]).astype(np.float32)

    if order is not None:
        orig_bin_ids = np.empty_like(bin_ids)
        orig_bin_ids[order] = bin_ids
        bin_ids = orig_bin_ids

    return start_bin, bin_ids, X


def smooth_window_probs(probs: np.ndarray) -> np.ndarray:
    """Smooths window class probabilities across a 50s Gaussian kernel to remove state jitter."""
    if len(probs) < 3:
        return probs
    kernel = np.array([0.10, 0.22, 0.36, 0.22, 0.10], dtype=np.float32)
    padded = np.pad(probs, ((2, 2), (0, 0)), mode="edge")
    return (
        kernel[0] * padded[:-4]
        + kernel[1] * padded[1:-3]
        + kernel[2] * padded[2:-2]
        + kernel[3] * padded[3:-1]
        + kernel[4] * padded[4:]
    )


def _extract_ort_probs(ort_outs: list, n_classes: int) -> np.ndarray | None:
    if len(ort_outs) <= 1:
        return None
    raw_probs = ort_outs[1]
    if isinstance(raw_probs, list) and len(raw_probs) > 0 and isinstance(raw_probs[0], dict):
        probs = np.zeros((len(raw_probs), n_classes), dtype=np.float32)
        for row_idx, prob_dict in enumerate(raw_probs):
            for cls_key, prob_val in prob_dict.items():
                cls_i = int(cls_key)
                if 0 <= cls_i < n_classes:
                    probs[row_idx, cls_i] = float(prob_val)
        return probs
    return np.asarray(raw_probs, dtype=np.float32)


class TechArenaModel(ADM_Base):
    """Implements ADM_Base using strictly single-threaded ONNX Runtime inference."""

    def __init__(self, onnx_file: pathlib.Path) -> None:
        self.load_model(onnx_file)
        self.input_name = self.session.get_inputs()[0].name
        self.class_weights = np.array(CLASS_WEIGHTS, dtype=np.float32)

    def load_model(self, onnx_file: pathlib.Path) -> None:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        if input.empty:
            return pd.DataFrame({"Activity": pd.Series(dtype=str)}, index=input.index)

        _, bin_ids, feats = compute_window_features(input, win_sec=10)
        ort_outs = self.session.run(None, {self.input_name: feats})

        probs = _extract_ort_probs(ort_outs, len(LABELS))
        if probs is not None and probs.ndim == 2 and probs.shape[1] == len(LABELS):
            smoothed_probs = smooth_window_probs(probs)
            pred_ids = np.argmax(smoothed_probs * self.class_weights, axis=-1)
        else:
            pred_ids = np.asarray(ort_outs[0]).reshape(-1).astype(int)

        win_labels = np.array(LABELS, dtype=object)[np.clip(pred_ids, 0, len(LABELS) - 1)]
        mapped = win_labels[bin_ids]
        return pd.DataFrame({"Activity": mapped}, index=input.index)


def get_model() -> ADM_Base:
    onnx_path = pathlib.Path(__file__).with_suffix(".onnx")
    return TechArenaModel(onnx_path)
