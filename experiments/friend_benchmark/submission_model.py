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
CLASS_WEIGHTS = [0.92, 1.3, 0.85]


def _ffill_bfill_2d(arr: np.ndarray) -> np.ndarray:
    """This subprogram causally forward-fills missing window features and backfills only leading empty bins."""
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
    """This subprogram causally forward-fills a 1D feature array and backfills only leading missing values."""
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
    """This subprogram computes a fast centered rolling average over [i - radius, i + radius] via cumulative sums."""
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


def compute_window_features(df: pd.DataFrame, win_sec: int = 10) -> tuple[int, np.ndarray, np.ndarray]:
    """This subprogram extracts 77 physiological, spectral BVP, tonic/phasic EDA, and HRV features per 10s window."""
    dt_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    bin_ns = int(win_sec * 1_000_000_000)
    raw_bins = dt_index.asi8 // bin_ns
    start_bin = int(raw_bins.min())
    end_bin = int(raw_bins.max())
    n_bins = end_bin - start_bin + 1
    bin_ids = raw_bins - start_bin

    cols_data = [df[c].to_numpy(dtype=np.float32) for c in SENSOR_COLS]
    n_cols = len(cols_data)

    means = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    stds = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    mins = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    maxs = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    within_slope = np.zeros((n_bins, n_cols), dtype=np.float32)

    # Compute per-bin mean, std, min, max, and within-window slope using fast vectorized reductions
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

    # Extract 64 Hz BVP waveform sharpness, zero-crossing rate, and cardiac-vs-motion FFT energy
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
    n_full = len(bvp) // win_len_64
    bvp_cardiac = np.zeros(n_bins, dtype=np.float32)
    bvp_motion = np.zeros(n_bins, dtype=np.float32)
    if n_full > 0:
        bvp_mat = bvp[:n_full * win_len_64].reshape(n_full, win_len_64)
        bvp_mat = bvp_mat - np.mean(bvp_mat, axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(bvp_mat, axis=1)) ** 2
        c_pow = np.mean(spec[:, 7:36], axis=1).astype(np.float32)
        m_pow = np.mean(spec[:, 40:151], axis=1).astype(np.float32)
        target_bins = bin_ids[np.arange(n_full) * win_len_64]
        bvp_cardiac[target_bins] = np.log1p(c_pow)
        bvp_motion[target_bins] = np.log1p(m_pow)
        if n_full < n_bins:
            bvp_cardiac[target_bins[-1] + 1:] = bvp_cardiac[target_bins[-1]]
            bvp_motion[target_bins[-1] + 1:] = bvp_motion[target_bins[-1]]
    bvp_motion_ratio = bvp_motion - bvp_cardiac

    # Decompose EDA into slow tonic sweat level and rapid phasic arousal spikes
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

    # Extract true beat-to-beat HRV (RMSSD, pNN50, SDNN, CV) and PPG beat-dropout ratio
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

    # Anchor channels to each participant's 15th-percentile resting floor
    p15_floor = np.percentile(means, 15, axis=0, keepdims=True).astype(np.float32)
    above_floor = means - p15_floor
    hr_ratio = means[:, 2] / (p15_floor[0, 2] + 1.0)
    eda_ratio = means[:, 1] / (p15_floor[0, 1] + 0.1)

    # Build symmetric 30s and 70s temporal context features
    sym30 = _symmetric_rolling_mean(means, radius=1)
    diff_sym30 = means - sym30
    sym70_sub = _symmetric_rolling_mean(means[:, [1, 2, 5]], radius=3)
    diff_sym70_sub = means[:, [1, 2, 5]] - sym70_sub

    # Compute cross-modal Stress-vs-Sweat interaction features
    hr_above = above_floor[:, 2]
    hr_per_bvp_std = hr_above / (stds[:, 0] + 1.0)
    eda_phasic_x_hr = eda_phasic * hr_above
    motion_x_hr = bvp_motion_ratio * hr_above
    temp_trend_60s = np.zeros(n_bins, dtype=np.float32)
    if n_bins > 6:
        temp_trend_60s[6:] = means[6:, 5] - means[:-6, 5]

    X = np.hstack([
        means, stds, ptp, delta, within_slope,
        bvp_diff1_rms[:, None], bvp_diff2_rms[:, None], bvp_zcr[:, None],
        bvp_crest[:, None], bvp_cardiac[:, None], bvp_motion[:, None],
        bvp_motion_ratio[:, None],
        eda_log[:, None], eda_phasic[:, None], eda_phasic_ratio[:, None],
        eda_rises[:, None], eda_max_jump[:, None],
        rmssd[:, None], pnn50[:, None], sdnn[:, None], ibi_cv[:, None],
        ibi_dropout[:, None],
        above_floor, hr_ratio[:, None], eda_ratio[:, None],
        sym30, diff_sym30, sym70_sub, diff_sym70_sub,
        hr_per_bvp_std[:, None], eda_phasic_x_hr[:, None],
        motion_x_hr[:, None], temp_trend_60s[:, None],
    ]).astype(np.float32)

    return start_bin, bin_ids, X


def smooth_window_probs(probs: np.ndarray) -> np.ndarray:
    """This subprogram smooths window class probabilities across a 50s Gaussian kernel to remove state jitter."""
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


class TechArenaModel(ADM_Base):
    """This class implements ADM_Base using strictly single-threaded ONNX Runtime inference."""

    def __init__(self, onnx_file: pathlib.Path) -> None:
        self.load_model(onnx_file)
        self.input_name = self.session.get_inputs()[0].name
        self.class_weights = np.array(CLASS_WEIGHTS, dtype=np.float32)

    def load_model(self, onnx_file: pathlib.Path) -> None:
        """This subprogram loads the ONNX classifier with single-threaded sequential execution enforced."""
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
        """This subprogram extracts 10s features, runs single-threaded ONNX inference, smooths probabilities, and maps labels."""
        if input.empty:
            return pd.DataFrame({"Activity": pd.Series(dtype=str)}, index=input.index)

        _, bin_ids, feats = compute_window_features(input, win_sec=10)
        ort_outs = self.session.run(None, {self.input_name: feats})

        if len(ort_outs) > 1:
            probs = np.asarray(ort_outs[1], dtype=np.float32)
            if probs.ndim == 2 and probs.shape[1] == len(LABELS):
                smoothed_probs = smooth_window_probs(probs)
                pred_ids = np.argmax(smoothed_probs * self.class_weights, axis=-1)
            else:
                pred_ids = np.asarray(ort_outs[0]).reshape(-1).astype(np.int64)
        else:
            pred_ids = np.asarray(ort_outs[0]).reshape(-1).astype(np.int64)

        win_labels = np.array(LABELS, dtype=object)[np.clip(pred_ids, 0, len(LABELS) - 1)]
        mapped = win_labels[bin_ids]
        return pd.DataFrame({"Activity": mapped}, index=input.index)


def get_model() -> ADM_Base:
    """This subprogram instantiates TechArenaModel and loads submission_model.onnx from the same directory."""
    onnx_path = pathlib.Path(__file__).with_suffix(".onnx")
    return TechArenaModel(onnx_path)
