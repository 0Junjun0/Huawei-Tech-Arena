"""Huawei wrist classifier. The same feature code serves training and deployment."""

import pathlib

import numpy as np
import onnxruntime as ort
import pandas as pd
from adm_base_class import ADM_Base
from scipy.ndimage import maximum_filter1d, minimum_filter1d

LABELS = ["Rest", "Stress", "Activity"]
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def _ffill_bfill_2d(arr: np.ndarray) -> np.ndarray:
    """Forward-fills missing window features and backfills only leading empty bins."""
    n_bins, n_cols = arr.shape
    if n_bins == 0:
        return arr
    mask = np.isnan(arr)
    if not mask.any():
        return np.nan_to_num(arr) if np.isinf(arr).any() else arr
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
    if not m.any():
        return np.nan_to_num(arr) if np.isinf(arr).any() else arr
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


def _symmetric_rolling_ptp_1d(
    mins_1d: np.ndarray, maxs_1d: np.ndarray, radius: int
) -> np.ndarray:
    """Centered peak-to-peak swing across [i - radius, i + radius] windows."""
    n = len(mins_1d)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    width = 2 * radius + 1
    return maximum_filter1d(maxs_1d, width, mode="nearest") - minimum_filter1d(
        mins_1d, width, mode="nearest"
    )


def _sensor_column(df, name, dtype):
    """Read native numeric columns without making a NaN-filled second copy."""
    if name not in df:
        return np.full(len(df), np.nan, dtype=dtype)
    column = df[name]
    if isinstance(column.dtype, np.dtype) and column.dtype.kind in "fiu":
        return column.to_numpy(dtype=dtype, copy=False)
    return column.to_numpy(dtype=dtype, na_value=np.nan)


def _optical_features(pulse, origin, bins, means, start_bin, bin_ns, width):
    """Reduce observed samples by bin. Resample irregular windows on their actual timestamps."""
    n = len(means)
    wave = np.zeros((n, 3), dtype=np.float32)
    bands = np.full((n, 3), np.nan, dtype=np.float32)
    if not len(pulse):
        return wave.T, bands.T
    starts = np.r_[0, np.flatnonzero(bins[1:] != bins[:-1]) + 1]
    ends = np.r_[starts[1:], len(bins)]
    lengths = ends - starts
    occupied = bins[starts]
    cadence = 15_625_000
    gaps = np.r_[np.int64(0), np.diff(origin)]
    gaps[starts] = 0
    adjacent = (gaps > 0) & (gaps <= 23_437_500)
    pairs = adjacent & np.r_[False, adjacent[:-1]]
    delta = np.zeros(len(pulse), dtype=np.float32)
    delta[1:] = np.diff(pulse)
    second = np.empty_like(delta)
    second[0] = 0
    second[1:] = np.diff(delta)
    centered = pulse - means[bins]
    crossings = np.zeros(len(pulse), dtype=bool)
    crossings[1:] = (centered[1:] * centered[:-1]) < 0
    n_adjacent = np.maximum(1, np.add.reduceat(adjacent, starts, dtype=np.int64))
    n_pairs = np.maximum(1, np.add.reduceat(pairs, starts, dtype=np.int64))
    delta[~adjacent] = 0
    second[~pairs] = 0
    wave[occupied, 0] = np.sqrt(
        np.add.reduceat(np.square(delta, dtype=np.float64), starts) / n_adjacent
    )
    wave[occupied, 1] = np.sqrt(
        np.add.reduceat(np.square(second, dtype=np.float64), starts) / n_pairs
    )
    wave[occupied, 2] = (
        np.add.reduceat(crossings & adjacent, starts, dtype=np.int64) / n_adjacent
    )

    offset = origin[starts] - (start_bin + occupied) * bin_ns
    last = origin[ends - 1] - (start_bin + occupied) * bin_ns
    complete = (
        (lengths >= width * 0.85)
        & (lengths <= width * 1.1)
        & (offset <= 125_000_000)
        & (last >= bin_ns - 125_000_000)
        & (np.maximum.reduceat(gaps, starts) <= 125_000_000)
    )
    irregular = gaps != cadence
    irregular[starts] = False
    on_grid = (
        (lengths == width) & (offset == 0) & ~np.logical_or.reduceat(irregular, starts)
    )
    grid = np.arange(width)
    target = grid * cadence
    taper = np.hanning(width)

    def store(bins_in_batch, signals):
        power = (np.abs(np.fft.rfft(signals * taper, axis=1)) ** 2).astype(np.float32)
        cardiac = power[:, 7:36].mean(axis=1)
        motion = power[:, 40:151].mean(axis=1)
        bands[occupied[bins_in_batch]] = np.column_stack(
            (np.log1p(cardiac), np.log1p(motion), cardiac / (cardiac + motion + 1e-6))
        )

    regular_bins = np.flatnonzero(complete & on_grid)
    for j in range(0, len(regular_bins), 96):
        idx = regular_bins[j : j + 96]
        loc = starts[idx]
        if loc[-1] - loc[0] == (len(idx) - 1) * width and np.all(np.diff(loc) == width):
            signal = centered[loc[0] : loc[-1] + width].reshape(-1, width)
        else:
            signal = centered[loc[:, None] + grid]
        store(idx, signal)
    other_bins = np.flatnonzero(complete & ~on_grid)
    for j in range(0, len(other_bins), 96):
        idx = other_bins[j : j + 96]
        signal = np.empty((len(idx), width), dtype=np.float64)
        for row, k in enumerate(idx):
            sl = slice(starts[k], ends[k])
            t = origin[sl] - (start_bin + occupied[k]) * bin_ns
            signal[row] = np.interp(target, t, centered[sl])
        store(idx, signal)
    return wave.T, bands.T


def compute_window_features(
    df: pd.DataFrame, win_sec: int = 10, *, reference_optical: bool = False, _grid=None
) -> tuple[int, np.ndarray, np.ndarray]:
    """Extracts 95 defensive, vectorized physiological, spectral, HRV, context, and subject-normalized features."""
    bin_ns = int(win_sec * 1_000_000_000)
    if _grid is None:
        dt_index = pd.DatetimeIndex(df.index).as_unit("ns")
        order = (
            None
            if dt_index.is_monotonic_increasing
            else np.argsort(dt_index.asi8, kind="stable")
        )
        asi8 = dt_index.asi8 if order is None else dt_index.asi8[order]
        raw_bins = asi8 // bin_ns
        start_bin = int(raw_bins[0])
        bin_ids = raw_bins - start_bin
    else:
        asi8, start_bin, bin_ids = _grid
        order = None
    n_bins = int(bin_ids[-1]) + 1

    n_cols = len(SENSOR_COLS)
    means = np.full((n_bins, n_cols), np.nan, dtype=np.float32)
    stds = np.full_like(means, np.nan)
    mins = np.full_like(means, np.nan)
    maxs = np.full_like(means, np.nan)
    within_slope = np.zeros_like(means)
    observations = []
    valid_ranges = (
        (None, None),
        (0, 150),
        (25, 240),
        (None, None),
        (0.20, 2.50),
        (15, 45),
    )
    for c, (name, (low, high)) in enumerate(zip(SENSOR_COLS, valid_ranges)):
        col = _sensor_column(df, name, np.float32)
        if order is not None:
            col = col[order]
        valid = np.isfinite(col) if low is None else (col >= low) & (col <= high)
        if valid.all():
            loc = slice(None)
        else:
            loc = np.flatnonzero(valid)
        b, values = bin_ids[loc], col[loc]
        observations.append((b, values))
        if c == 0:
            optical_positions = loc
            if reference_optical:
                bvp = np.where(valid, col, np.float32(0))
        if not len(b):
            continue
        starts = np.r_[0, np.flatnonzero(b[1:] != b[:-1]) + 1]
        ends = np.r_[starts[1:], len(b)]
        bins = b[starts]
        count = ends - starts
        v = values.astype(np.float64)
        if c == 0:
            total = np.add.reduceat(v, starts)
            squares = np.add.reduceat(v * v, starts)
        else:
            # Keep the original accumulation order for near-constant temperature and beat signals.
            total = np.bincount(b, weights=v, minlength=n_bins)[bins]
            squares = np.bincount(b, weights=v * v, minlength=n_bins)[bins]
        m = total / count
        variance = np.maximum(0.0, squares / count - m * m)
        means[bins, c] = m
        stds[bins, c] = np.sqrt(variance)
        mins[bins, c] = np.minimum.reduceat(values, starts)
        maxs[bins, c] = np.maximum.reduceat(values, starts)
        within_slope[bins, c] = values[ends - 1] - values[starts]

    means = _ffill_bfill_2d(means)
    stds = _ffill_bfill_2d(stds)
    mins = _ffill_bfill_2d(mins)
    maxs = _ffill_bfill_2d(maxs)
    ptp = maxs - mins
    delta = np.zeros_like(means)
    delta[1:] = means[1:] - means[:-1]

    if reference_optical:
        # 64 Hz BVP waveform sharpness, zero-crossing rate, and bin-aligned cardiac-vs-motion FFT energy
        bvp_centered = bvp - means[bin_ids, 0]
        same_b = bin_ids[1:] == bin_ids[:-1]

        d1 = np.diff(bvp)
        d1_sq = np.bincount(
            bin_ids[1:][same_b],
            weights=(d1[same_b] ** 2).astype(np.float64),
            minlength=n_bins,
        )
        d1_cnt = np.maximum(1, np.bincount(bin_ids[1:][same_b], minlength=n_bins))
        bvp_diff1_rms = np.sqrt(d1_sq / d1_cnt).astype(np.float32)

        d2 = np.diff(d1)
        same_b2 = bin_ids[2:] == bin_ids[:-2]
        d2_sq = np.bincount(
            bin_ids[2:][same_b2],
            weights=(d2[same_b2] ** 2).astype(np.float64),
            minlength=n_bins,
        )
        d2_cnt = np.maximum(1, np.bincount(bin_ids[2:][same_b2], minlength=n_bins))
        bvp_diff2_rms = np.sqrt(d2_sq / d2_cnt).astype(np.float32)

        zc = ((bvp_centered[1:] * bvp_centered[:-1]) < 0) & same_b
        bvp_zcr = (np.bincount(bin_ids[1:][zc], minlength=n_bins) / d1_cnt).astype(
            np.float32
        )
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

    else:
        pulse_bins, pulse = observations[0]
        wave, bands = _optical_features(
            pulse,
            asi8[optical_positions],
            pulse_bins,
            means[:, 0],
            start_bin,
            bin_ns,
            64 * win_sec,
        )
        bvp_diff1_rms, bvp_diff2_rms, bvp_zcr = wave
        bvp_cardiac, bvp_motion, bvp_cardiac_frac = bands
        bvp_crest = ptp[:, 0] / (stds[:, 0] + 1e-4)
        bvp_cardiac = _ffill_bfill_1d(bvp_cardiac)
        bvp_motion = _ffill_bfill_1d(bvp_motion)
        bvp_cardiac_frac = _ffill_bfill_1d(bvp_cardiac_frac)
        bvp_motion_ratio = bvp_motion - bvp_cardiac

    # EDA tonic/phasic decomposition + arousal spike energy
    eda_log = np.log1p(np.maximum(means[:, 1], 0.0))
    eda_tonic = _symmetric_rolling_mean(means[:, 1:2], radius=3)[:, 0]
    eda_phasic = means[:, 1] - eda_tonic
    eda_phasic_ratio = eda_phasic / (np.maximum(eda_tonic, 0.0) + 0.1)

    e_bins, e_vals = observations[1]
    eda_rises = np.zeros(n_bins, dtype=np.float32)
    eda_max_jump = np.zeros(n_bins, dtype=np.float32)
    if len(e_bins):
        same_e = e_bins[1:] == e_bins[:-1]
        if np.any(same_e):
            ediffs = e_vals[1:][same_e] - e_vals[:-1][same_e]
            eb = e_bins[1:][same_e]
            eda_rises = np.bincount(
                eb, weights=(ediffs > 0.01).astype(np.float64), minlength=n_bins
            ).astype(np.float32)
            pos_mask = ediffs > 0
            if np.any(pos_mask):
                np.maximum.at(eda_max_jump, eb[pos_mask], ediffs[pos_mask])

    eda_phasic_energy = _symmetric_rolling_mean(np.abs(eda_phasic)[:, None], radius=1)[
        :, 0
    ]
    eda_rises_ctx30 = _symmetric_rolling_mean(eda_rises[:, None], radius=1)[:, 0]

    # Beat-to-beat HRV + IBI_dt step delta (from 42-feature baseline)
    ibi_bins, ibi_vals = observations[4]
    rmssd = np.full(n_bins, np.nan, dtype=np.float32)
    pnn50 = np.full(n_bins, np.nan, dtype=np.float32)
    if len(ibi_bins):
        same_bin = ibi_bins[1:] == ibi_bins[:-1]
        if np.any(same_bin):
            diffs = ibi_vals[1:][same_bin] - ibi_vals[:-1][same_bin]
            d_bins = ibi_bins[1:][same_bin]
            d_cnt = np.bincount(d_bins, minlength=n_bins)
            d_sq = np.bincount(
                d_bins, weights=(diffs * diffs).astype(np.float64), minlength=n_bins
            )
            d_50 = np.bincount(
                d_bins,
                weights=(np.abs(diffs) > 0.05).astype(np.float64),
                minlength=n_bins,
            )
            has_d = d_cnt > 0
            rmssd[has_d] = np.sqrt(d_sq[has_d] / d_cnt[has_d]).astype(np.float32)
            pnn50[has_d] = (d_50[has_d] / d_cnt[has_d]).astype(np.float32)

    rmssd = _ffill_bfill_1d(rmssd)
    pnn50 = _ffill_bfill_1d(pnn50)
    sdnn = stds[:, 4]
    ibi_cv = sdnn / (means[:, 4] + 1e-4)
    ibi_dropout = maxs[:, 3] / (means[:, 4] + 1e-4)

    idt_bins, idt_vals = observations[3]
    ibi_dt_abs_diff = np.zeros(n_bins, dtype=np.float32)
    if len(idt_bins):
        same_idt = idt_bins[1:] == idt_bins[:-1]
        if np.any(same_idt):
            adiffs = np.abs(idt_vals[1:][same_idt] - idt_vals[:-1][same_idt]).astype(
                np.float64
            )
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

    normalized_channels = np.stack(
        (
            eda_log,
            means[:, 2],
            means[:, 4],
            means[:, 5],
            bvp_motion_ratio,
            rmssd,
            eda_phasic,
        )
    )
    q25, median, q75 = np.percentile(normalized_channels, [25, 50, 75], axis=1)
    min_scale = np.array([0.05, 2.0, 0.02, 0.1, 0.2, 0.01, 0.02])
    scale = np.maximum(
        np.maximum((q75 - q25) * 0.7413, normalized_channels.std(1)), min_scale
    )
    eda_z, hr_z, ibi_z, temp_z, motion_z, rmssd_z, eda_phasic_z = (
        normalized_channels - median.astype(np.float32)[:, None]
    ) / scale.astype(np.float32)[:, None]

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

    X = np.hstack(
        [
            means,
            stds,
            ptp,
            delta,
            within_slope,
            bvp_diff1_rms[:, None],
            bvp_diff2_rms[:, None],
            bvp_zcr[:, None],
            bvp_crest[:, None],
            bvp_cardiac[:, None],
            bvp_motion[:, None],
            bvp_motion_ratio[:, None],
            bvp_cardiac_frac[:, None],
            eda_log[:, None],
            eda_phasic[:, None],
            eda_phasic_ratio[:, None],
            eda_rises[:, None],
            eda_max_jump[:, None],
            eda_phasic_energy[:, None],
            eda_rises_ctx30[:, None],
            rmssd[:, None],
            pnn50[:, None],
            sdnn[:, None],
            ibi_cv[:, None],
            ibi_dropout[:, None],
            ibi_dt_abs_diff[:, None],
            above_floor,
            hr_ratio[:, None],
            eda_ratio[:, None],
            eda_z[:, None],
            hr_z[:, None],
            ibi_z[:, None],
            temp_z[:, None],
            motion_z[:, None],
            rmssd_z[:, None],
            sym30,
            diff_sym30,
            sym70_sub,
            diff_sym70_sub,
            hr_std_ctx50[:, None],
            eda_std_ctx50[:, None],
            eda_ctx70_swing[:, None],
            hr_per_bvp_std[:, None],
            eda_phasic_x_hr[:, None],
            motion_x_hr[:, None],
            temp_trend_60s[:, None],
            hr_trend_60s[:, None],
            eda_trend_60s[:, None],
            stress_arousal_score[:, None],
            hr_minus_temp_z[:, None],
            rmssd_z_x_hr_z[:, None],
        ]
    ).astype(np.float32)

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


PERIOD_NS = {
    "HR": 1_000_000_000,
    "EDA": 250_000_000,
    "Temperature": 250_000_000,
    "BVP": 15_625_000,
}

BIN_NS = 10_000_000_000


def beat_pairs(duration, times, reported_dt):
    """Return consecutive observed beats, allowing one optical sample of timestamp jitter."""
    elapsed = np.diff(times) / 1e9
    tolerance = PERIOD_NS["BVP"] / 1e9
    agrees = ~np.isfinite(reported_dt[1:]) | (
        np.abs(reported_dt[1:] - elapsed) <= tolerance
    )
    return (elapsed > 0) & (np.abs(elapsed - duration[1:]) <= tolerance) & agrees


def recover_intervals(duration, dt, hr, times):
    """Accept a reported gap as one beat only when timestamps and rate support it."""
    measured = np.isfinite(duration) & (duration > 0)
    dt_ok = np.isfinite(dt)
    if not (dt_ok & ~measured).any():
        return duration
    idx = np.flatnonzero(measured | dt_ok)
    elapsed = np.diff(times[idx]) / 1e9
    values = dt[idx]
    consistent = np.r_[False, np.abs(values[1:] - elapsed) <= PERIOD_NS["BVP"] / 1e9]
    plausible = consistent & (values >= 0.30) & (values <= 2.0)
    hr_valid = np.flatnonzero(np.isfinite(hr) & (hr >= 30) & (hr <= 210))
    supported = np.zeros(len(idx), dtype=bool)
    has_hr = np.zeros(len(idx), dtype=bool)
    if len(hr_valid):
        ht = times[hr_valid]
        j = ht.searchsorted(times[idx])
        left, right = np.maximum(j - 1, 0), np.minimum(j, len(ht) - 1)
        near = np.where(
            abs(ht[left] - times[idx]) <= abs(ht[right] - times[idx]), left, right
        )
        has_hr = abs(times[idx] - ht[near]) <= 15_000_000_000
        expected = 60.0 / hr[hr_valid[near]]
        supported = has_hr & (values >= 0.65 * expected) & (values <= 1.45 * expected)
    if len(idx) > 1:
        similar = (
            plausible[1:]
            & plausible[:-1]
            & (elapsed <= 2.0)
            & (
                np.abs(values[1:] - values[:-1])
                < 0.18 * np.minimum(values[1:], values[:-1])
            )
        )
        supported |= ~has_hr & (np.r_[False, similar] | np.r_[similar, False])
    take = idx[~measured[idx] & plausible & supported]
    if not len(take):
        return duration
    recovered = duration.copy()
    recovered[take] = dt[take]
    return recovered


def repair_hr(hr, intervals, times, positions, clock):
    valid = np.flatnonzero(np.isfinite(hr))
    if len(valid) and (
        times[valid[0]] - times[0] <= 2_000_000_000
        and times[-1] - times[valid[-1]] <= 2_000_000_000
        and np.all(np.diff(times[valid]) <= 2_000_000_000)
    ):
        return hr
    recovered = heart_rate_from_beats(intervals, times, positions, clock)
    if not len(valid):
        return recovered
    available = np.flatnonzero(np.isfinite(recovered))
    if not len(available):
        return hr
    observed_seconds = times[valid] // 1_000_000_000
    seconds = times[available] // 1_000_000_000
    found = np.searchsorted(observed_seconds, seconds)
    fill = available[
        (found == len(valid))
        | (observed_seconds[np.minimum(found, len(valid) - 1)] != seconds)
    ]
    if not len(fill):
        return hr
    result = hr.copy()
    result[fill] = recovered[fill]
    return result


def heart_rate_from_beats(intervals, times, positions, clock):
    """Recover a missing HR stream near observed beat intervals, at most once per second."""
    usable = np.isfinite(intervals) & (intervals >= 0.3) & (intervals <= 2.0)
    result = np.full(len(times), np.nan)
    counts = np.bincount(positions[usable], minlength=len(clock))
    anchors = counts >= 2
    if not anchors.any():
        return result
    sums = np.bincount(
        positions[usable], weights=intervals[usable], minlength=len(clock)
    )
    hr = 60 * counts[anchors] / sums[anchors]
    # Reference times are relative to the first occupied bin, preserving timestamp precision.
    anchor_times = (clock[anchors] - clock[0]) * 10.0 + 5.0
    samples = np.r_[0, np.flatnonzero(np.diff(times // 1_000_000_000)) + 1]
    seconds = (times[samples] - clock[0] * BIN_NS) / 1e9
    right = anchor_times.searchsorted(seconds)
    previous = anchor_times[np.maximum(right - 1, 0)]
    following = anchor_times[np.minimum(right, len(anchor_times) - 1)]
    near = np.minimum(np.abs(seconds - previous), np.abs(seconds - following)) <= 30
    result[samples[near]] = np.interp(seconds[near], anchor_times, hr)
    return result


def pulse_measurements(values, times, index, counts, rate, spectrum_cache=None):
    """Waveform shape and spectral concentration using only observed 64 Hz pulse data."""
    n = len(counts)
    result = np.full((n, 15), np.nan)
    starts = np.cumsum(counts) - counts
    regular = counts == 640
    discontinuities = np.flatnonzero(np.diff(times) != PERIOD_NS["BVP"])
    inside_bin = index[discontinuities] == index[discontinuities + 1]
    regular[index[discontinuities[inside_bin]]] = False
    complete = np.flatnonzero(regular)
    if spectrum_cache is not None:
        spectrum_cache["pulse"] = (times, values, index, counts, starts, complete)
    grid = np.arange(640)
    taper = np.hanning(640)

    def measure(bins, signal, complete_observations=False):
        zeroed = signal - signal.mean(axis=1, keepdims=True)
        sd = np.sqrt(np.mean(zeroed**2, axis=1))
        normalized = zeroed / np.maximum(1.0e-7, sd[:, None])
        square = normalized * normalized
        derivative = np.diff(normalized, axis=1)
        power = np.abs(np.fft.rfft(zeroed * taper)) ** 2
        if complete_observations and spectrum_cache is not None:
            if "power" not in spectrum_cache:
                spectrum_cache.update(power=np.empty((n, 321)), seen=np.zeros(n, bool))
            spectrum_cache["power"][bins] = power
            spectrum_cache["seen"][bins] = True
        cardiac = power[:, 6:36]
        dominant = 6 + np.argmax(cardiac, axis=1)
        row = np.arange(len(bins))
        total = np.maximum(1.0e-12, power[:, 4:121].sum(1))
        fundamental = sum(power[row, dominant + offset] for offset in (-1, 0, 1))
        harmonic = sum(power[row, 2 * dominant + offset] for offset in (-1, 0, 1))
        period = np.clip(np.rint(640 / dominant).astype(int), 16, 107)
        # The central portion avoids wraparound. All samples belong to the same ten seconds.
        central = np.arange(108, 532, 2)[None, :]
        autocorr = (
            normalized[:, central[0]]
            * normalized[row[:, None], central - period[:, None]]
        ).mean(1)
        segment = power[:, 4:75]
        freq = np.arange(4, 75)
        centroid = (segment * freq).sum(1) / np.maximum(1.0e-12, segment.sum(1))
        z = np.column_stack(
            (
                np.log1p(sd),
                np.clip(np.mean(square * normalized, 1), -8, 8),
                np.log1p(np.mean(square * square, 1)),
                (derivative > 0).mean(1),
                np.log1p(np.abs(derivative).mean(1)),
                dominant * 6,
                np.abs(rate[bins] - dominant * 6),
                fundamental / total,
                np.log1p(harmonic / np.maximum(fundamental, 1.0e-12)),
                power[:, 36:101].sum(1) / total,
                np.clip(autocorr, -2, 2),
                cardiac.sum(1) / total,
                centroid / 10,
                power[:, 1:5].sum(1) / np.maximum(power[:, 1:121].sum(1), 1.0e-12),
                np.mean(np.abs(np.diff(derivative, axis=1)), 1),
            )
        )
        z[sd < 1.0e-7] = np.nan
        result[bins] = z

    for j in range(0, len(complete), 128):
        bins = complete[j : j + 128]
        measure(bins, values[starts[bins, None] + grid], True)
    partial = np.flatnonzero((counts >= 576) & (counts < 640))
    for j in range(0, len(partial), 128):
        bins = []
        signals = []
        for b in partial[j : j + 128]:
            sl = slice(starts[b], starts[b] + counts[b])
            slots = (times[sl] % BIN_NS) // PERIOD_NS["BVP"]
            gaps = np.diff(slots)
            if slots[0] <= 1 and slots[-1] >= 638 and np.all((gaps > 0) & (gaps <= 4)):
                bins.append(b)
                signals.append(np.interp(grid, slots, values[sl]))
        if bins:
            measure(np.asarray(bins), np.asarray(signals))
    return result


def physiology_windows(frame, spectrum_cache=None, *, _grid=None):
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans:
        raise ValueError("Expected a timestamp index without missing timestamps")
    if not len(idx):
        return (
            np.empty((0, PHYSIOLOGY_WIDTH), np.float32),
            np.empty(0, int),
            np.empty(0, np.int64),
        )
    if _grid is None:
        original = idx.as_unit("ns").asi8
        order = (
            None if idx.is_monotonic_increasing else np.argsort(original, kind="stable")
        )
        times = original if order is None else original[order]
        bins = times // BIN_NS
        start = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
        clock = bins[start]
        rows = np.repeat(np.arange(len(clock)), np.diff(np.r_[start, len(times)]))
    else:
        times, start_bin, rows = _grid
        clock = start_bin + np.arange(int(rows[-1]) + 1)
        order = None
    n = len(clock)

    def sensor(key):
        if key not in frame:
            return np.full(len(times), np.nan)
        x = _sensor_column(frame, key, np.float64)
        return x if order is None else x[order]

    def count(ix):
        return np.bincount(ix, minlength=n)

    def avg(values, groups, counts):
        return np.divide(
            np.bincount(groups, weights=values, minlength=n),
            counts,
            out=np.full(n, np.nan),
            where=counts > 0,
        )

    def stats(v, groups, counts):
        level = avg(v, groups, counts)
        return level, np.sqrt(np.maximum(0, avg(v * v, groups, counts) - level**2))

    hr = sensor("HR")
    dt = sensor("IBI_dt")
    ibi = recover_intervals(sensor("IBI_duration"), dt, hr, times)
    hr = repair_hr(hr, ibi, times, rows, clock)
    output = []
    rate = np.full(n, np.nan)
    for key, raw, low, high in (
        ("HR", hr, 25, 235),
        ("EDA", sensor("EDA"), 0, 100),
        ("Temperature", sensor("Temperature"), 0, 50),
        ("IBI_duration", ibi, 0.30, 2),
        ("BVP", sensor("BVP"), -1.0e5, 1.0e5),
    ):
        valid = np.isfinite(raw) & (raw >= low) & (raw <= high)
        loc = slice(None) if valid.all() else np.flatnonzero(valid)
        g, v, t = rows[loc], raw[loc], times[loc]
        if key == "BVP":
            k = np.diff(np.searchsorted(g, np.arange(n + 1)))
            output.extend(pulse_measurements(v, t, g, k, rate, spectrum_cache).T)
            output.append(np.minimum(k / 640, 1))
            continue
        k = count(g)
        m, sd = stats(v, g, k)
        if key == "HR":
            rate = m
        elif key == "EDA" and spectrum_cache is not None:
            spectrum_cache["eda"] = (t, v, g)
        output.extend((m, sd))
        if key == "IBI_duration":
            consecutive = beat_pairs(v, t, dt[loc])
            groups = g[1:][consecutive]
            delta = np.diff(v)[consecutive]
            pairs = count(groups)
            output.extend(
                (
                    avg(np.abs(delta), groups, pairs),
                    np.sqrt(avg(delta * delta, groups, pairs)),
                    avg(np.abs(delta) > 0.05, groups, pairs),
                    np.clip(np.bincount(g, weights=v, minlength=n) / 10, 0, 1),
                    np.minimum(pairs / 12, 1),
                )
            )
            continue
        elapsed = np.diff(t) / 1.0e9
        contiguous = np.abs(np.diff(t) - PERIOD_NS[key]) <= PERIOD_NS[key] // 3
        groups = g[1:][contiguous]
        step = np.diff(v)[contiguous] / elapsed[contiguous]
        pairs = count(groups)
        output.extend((avg(np.abs(step), groups, pairs), avg(step, groups, pairs)))
        if key == "EDA":
            output.append(avg(np.maximum(0, step), groups, pairs))
            # Positive skin conductance deflections against short tonic histories.
            end = np.arange(1, len(t) + 1)
            prefix = np.r_[0.0, np.cumsum(v)]
            even = len(t) > 1 and np.all(np.diff(t) == PERIOD_NS["EDA"])
            for sec in (2, 8, 20):
                begin = (
                    np.maximum(0, end - 1 - 4 * sec)
                    if even
                    else t.searchsorted(t - sec * 1_000_000_000)
                )
                support = end - begin
                good = support >= 3 * sec
                amplitude = (
                    v[good] - (prefix[end[good]] - prefix[begin[good]]) / support[good]
                )
                target = g[good]
                nn = count(target)
                output.extend(
                    (
                        avg(np.maximum(0, amplitude), target, nn),
                        avg(amplitude > 0.03, target, nn),
                    )
                )
        output.append(
            np.minimum(k / ({"HR": 10, "EDA": 40, "Temperature": 40}[key]), 1)
        )
    values = np.column_stack(output).astype("float32")
    if values.shape[1] != PHYSIOLOGY_WIDTH:
        raise ValueError(f"Expected {PHYSIOLOGY_WIDTH} features, got {values.shape[1]}")
    if order is not None:
        aligned = np.empty_like(rows)
        aligned[order] = rows
        rows = aligned
    return values, rows, clock


PHYSIOLOGY_WIDTH = 45

UNITS = np.asarray(
    [
        30,
        2,
        1,
        1,
        1,
        0.5,
        0.04,
        0.03,
        0.03,
        0.03,
        0.02,
        0.25,
        0.04,
        0.25,
        0.08,
        0.25,
        1,
        5,
        0.1,
        0.01,
        0.01,
        1,
        0.25,
        0.03,
        0.03,
        0.03,
        0.10,
        0.3,
        1,
        2,
        1,
        1,
        0.3,
        0.3,
        30,
        10,
        0.4,
        0.5,
        0.2,
        0.4,
        0.4,
        2,
        0.3,
        0.3,
        1,
    ],
    dtype=float,
)


# Multiscale summaries retain absolute levels as a reference when clips are short.
CONTEXT_CHANNELS = np.array(
    [0, 1, 3, 5, 6, 11, 12, 13, 16, 17, 22, 23, 26, 27, 30, 32, 35, 36, 38, 39, 42, 43]
)
FEATURE_SCHEMA = "merged-crop-v2"


def feature_matrix(frame, *, _grid=None):
    _, _, merged = compute_window_features(frame, _grid=_grid)
    values, _, clock = physiology_windows(frame, _grid=_grid)
    if len(merged) != len(clock):
        raise ValueError(
            "Expected a contiguous sequence of occupied ten-second windows"
        )
    seen = np.isfinite(values)
    scaled = np.arcsinh(np.where(seen, values, 0.0) / UNITS).astype(np.float32)
    local = scaled[:, CONTEXT_CHANNELS]
    valid = seen[:, CONTEXT_CHANNELS]
    counts = np.vstack([np.zeros(local.shape[1]), valid.cumsum(axis=0)])
    total = np.vstack(
        [np.zeros(local.shape[1]), local.cumsum(axis=0, dtype=np.float64)]
    )
    sq = np.vstack(
        [np.zeros(local.shape[1]), (local**2).cumsum(axis=0, dtype=np.float64)]
    )
    pos = np.arange(len(clock))
    parts = [merged, scaled, seen.astype(np.float32)]
    for radius in (2, 8, 24):
        l = np.maximum(pos - radius, 0)
        h = np.minimum(pos + radius + 1, len(clock))
        count = np.maximum(counts[h] - counts[l], 1)
        avg = (total[h] - total[l]) / count
        var = np.maximum((sq[h] - sq[l]) / count - avg**2, 0.0)
        mid = pos + 1
        before = (total[mid] - total[l]) / np.maximum(counts[mid] - counts[l], 1)
        after = (total[h] - total[pos]) / np.maximum(counts[h] - counts[pos], 1)
        parts.extend(
            [avg, np.sqrt(var), np.where(valid, local - avg, 0), after - before]
        )
    # Recording-level references use a stabilised scale; missing channels keep their own indicators.
    ordered = np.sort(np.where(valid, local, np.inf), axis=0)
    size = valid.sum(0)
    features = np.arange(local.shape[1])
    quantiles = []
    for q in (0.15, 0.50, 0.85):
        index = np.floor(q * np.maximum(size - 1, 0)).astype(int)
        quantiles.append(np.where(size > 0, ordered[index, features], 0.0))
    lo, med, hi = quantiles
    parts.extend(
        [
            np.where(valid, local - med, 0),
            np.where(valid, (local - med) / (0.08 + hi - lo), 0),
            np.broadcast_to(hi - lo, local.shape),
        ]
    )
    return np.nan_to_num(
        np.column_stack(parts).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def recording_segments(frame, preserve_reference_grid=False, *, slices=False):
    """Yield features and original row positions, without padding long timestamp gaps."""
    time = pd.DatetimeIndex(frame.index).as_unit("ns").asi8
    sorted_index = frame.index.is_monotonic_increasing
    order = None if sorted_index else np.argsort(time, kind="stable")
    ordered_time = time if sorted_index else time[order]
    ticks = ordered_time // BIN_NS
    steps = np.diff(ticks)
    occupied = 1 + np.count_nonzero(steps)
    # The original predictor fills empty bins. Preserve its grid when finite gaps
    # fit in memory, while bounding the allocation for sparse multi-day input.
    if preserve_reference_grid and ticks[-1] - ticks[0] + 1 <= max(128, 4 * occupied):
        edges = np.array([0, len(frame)])
    else:
        edges = np.r_[0, np.flatnonzero(steps > 1) + 1, len(frame)]
    for left, right in zip(edges[:-1], edges[1:]):
        if right > left:
            if sorted_index:
                rows = slice(left, right) if slices else np.arange(left, right)
                section = (
                    frame
                    if left == 0 and right == len(frame)
                    else frame.iloc[left:right]
                )
            else:
                rows = order[left:right]
                section = frame.iloc[rows]
            ids = ticks[left:right] - ticks[left]
            yield rows, ids, ticks[left], section


def feature_segments(frame):
    for rows, ids, start, section in recording_segments(frame):
        grid = (section.index.as_unit("ns").asi8, start, ids)
        yield rows, ids, start, feature_matrix(section, _grid=grid)


def has_reported_cardiac_data(frame):
    for signal, low, high in (("HR", 25, 240), ("IBI_duration", 0.2, 2.5)):
        if signal in frame:
            values = _sensor_column(frame, signal, np.float32)
            if np.any((values >= low) & (values <= high)):
                return True
    return False


def inference_grid(section, grid):
    """Form complete ten-second windows when a caller cuts through the original grid.

    Keep the input index for the result. A trailing fragment outside the modeled
    windows receives the last available prediction. All context is local to this call.
    """
    times, start_bin, row_bins = grid
    edge_tolerance = 125_000_000
    phase = int(times[0] % BIN_NS)
    last_phase = int(times[-1] % BIN_NS)
    count = int(row_bins[-1]) + 1
    if count < 2 or phase <= edge_tolerance:
        return section, grid, row_bins
    span = int(times[-1] - times[0])
    remainder = span % BIN_NS
    last_partial = last_phase < BIN_NS - edge_tolerance
    complete_on_original_grid = count - 1 - int(last_partial)
    complete_from_start = (span + edge_tolerance) // BIN_NS
    near_boundary = remainder <= edge_tolerance or remainder >= BIN_NS - edge_tolerance
    recovers_window = near_boundary and complete_from_start > max(
        0, complete_on_original_grid
    )
    # Twenty-four bins is the longest context radius in the trained model.
    # In short clips both partial ends can influence the entire recording context.
    short_clip = count <= 25 and last_partial and span >= BIN_NS
    if not (recovers_window or short_clip):
        return section, grid, row_bins
    shifted_times = times - phase
    shifted_bins = shifted_times // BIN_NS - start_bin
    if np.any(np.diff(shifted_bins) > 1):
        return section, grid, row_bins
    end = (
        shifted_bins.searchsorted(complete_from_start)
        if short_clip or remainder <= edge_tolerance
        else len(section)
    )
    if end == 0:
        return section, grid, row_bins
    if end < len(section):
        analysis = (shifted_times[:end], start_bin, shifted_bins[:end])
        return (
            section.iloc[:end],
            analysis,
            np.minimum(shifted_bins, shifted_bins[end - 1]),
        )
    return section, (shifted_times, start_bin, shifted_bins), shifted_bins


class TechArenaModel(ADM_Base):
    def __init__(self, path):
        self.load_model(path)

    def load_model(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        if (
            self.session.get_modelmeta().custom_metadata_map.get("schema")
            != FEATURE_SCHEMA
        ):
            raise ValueError(
                "The Python file requires the matching merged-crop ONNX model."
            )
        # ONNX If reads only the features of the chosen branch.
        self.placeholders = {
            arg.name: np.zeros((1, arg.shape[-1]), dtype=np.float32)
            for arg in self.session.get_inputs()
            if arg.name != "has_cardiac"
        }
        self.reference_weights = np.array([1.08, 1.0, 0.92], dtype=np.float32)

    def run(self, input):
        result = np.zeros(len(input), dtype=np.int8)
        if len(input):
            reference_only = not has_reported_cardiac_data(input)
            for positions, ids, start, section in recording_segments(
                input, preserve_reference_grid=reference_only, slices=True
            ):
                cardiac = not reference_only and (
                    len(section) == len(input) or has_reported_cardiac_data(section)
                )
                grid = (section.index.as_unit("ns").asi8, start, ids)
                feed = dict(self.placeholders, has_cardiac=np.array(cardiac))
                if cardiac:
                    section, grid, ids = inference_grid(section, grid)
                    features = feature_matrix(section, _grid=grid)
                    feed["features"] = features
                    if "context" in feed:
                        feed["context"] = context_features(features)
                else:
                    _, _, features = compute_window_features(
                        section, reference_optical=True, _grid=grid
                    )
                    feed["reference_features"] = features
                probability = self.session.run(["probabilities"], feed)[0]
                smoothed = smooth_window_probs(probability)
                if not cardiac:
                    smoothed = smoothed * self.reference_weights
                result[positions] = smoothed.argmax(1)[ids]
        return pd.DataFrame(
            {"Activity": pd.Categorical.from_codes(result, categories=LABELS)},
            index=input.index,
        )


def get_model():
    return TechArenaModel(pathlib.Path(__file__).with_suffix(".onnx"))


def context_features(features):
    """Absolute, ranked, and quiet-reference signal features for the independent context head."""
    from scipy.stats import rankdata

    phys = features[:, 95:140]
    selected = np.array(
        [
            0,
            1,
            2,
            3,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            15,
            16,
            17,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            37,
            38,
            39,
            40,
            41,
            43,
        ]
    )
    stable = np.arcsinh(
        features[
            :, [30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 49, 50, 51]
        ]
    )
    z = np.column_stack([phys[:, selected], stable])
    low, med, hi = np.quantile(z, [0.12, 0.5, 0.88], axis=0)
    relative = np.clip((z - med) / (0.06 + hi - low), -5.0, 5.0)
    ranks = ((rankdata(z, method="average", axis=0) - 0.5) / len(z) - 0.5).astype("f")
    n = len(z)
    ix = np.arange(n)
    cols = [
        z,
        relative,
        ranks,
        features[:, 140:185],
        np.broadcast_to(low, z.shape),
        np.broadcast_to(hi, z.shape),
    ]
    # Heart rate and short-term waveform changes anchor resting and active reference bins.
    for score in [ranks[:, 0] + 0.45 * ranks[:, -17], ranks[:, 4] + 0.3 * ranks[:, 0]]:
        threshold = np.quantile(score, 0.20)
        cols.append(z - np.median(z[score <= threshold], axis=0))
    sums = np.vstack([np.zeros(z.shape[1]), np.cumsum(relative, axis=0, dtype=float)])

    def average(start, end):
        left = np.clip(ix + start, 0, n - 1)
        right = np.maximum(left + 1, np.clip(ix + end + 1, 1, n))
        return (sums[right] - sums[left]) / (right - left)[:, None]

    recent = average(-2, 2)
    cols.append(recent)
    for span in (4, 12, 30):
        cols.extend(
            [
                recent - average(-span, -max(1, span // 3)),
                average(max(1, span // 3), span) - recent,
            ]
        )
    return np.column_stack(cols).astype("f")
