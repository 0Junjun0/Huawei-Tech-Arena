"""Wearable classifier for complete offline recordings. Requires the challenge ADM_Base."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

try:
    from adm_base_class import ADM_Base
except ModuleNotFoundError as exc:
    if exc.name != "adm_base_class":
        raise

    class ADM_Base:
        """Standalone compatibility when running outside the challenge harness."""


from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, lfilter, sosfilt
from scipy.special import expit, softmax

LABELS = ["Rest", "Stress", "Activity"]

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
    adjacent = (np.diff(index) == 0) & (np.diff(times) == PERIOD_NS["BVP"])
    matching = np.bincount(index[1:][adjacent], minlength=n)
    complete = np.flatnonzero((counts == 640) & (matching == 639))
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


def physiology_windows(frame, spectrum_cache=None):
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans:
        raise ValueError("Expected a timestamp index without missing timestamps")
    if not len(idx):
        return (
            np.empty((0, PHYSIOLOGY_WIDTH), np.float32),
            np.empty(0, int),
            np.empty(0, np.int64),
        )
    original = idx.as_unit("ns").asi8
    order = None if idx.is_monotonic_increasing else np.argsort(original, kind="stable")
    times = original if order is None else original[order]
    bins = times // BIN_NS
    start = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
    clock = bins[start]
    rows = np.repeat(np.arange(len(clock)), np.diff(np.r_[start, len(times)]))
    n = len(clock)

    def sensor(key):
        if key not in frame:
            return np.full(len(times), np.nan)
        x = frame[key].to_numpy(dtype=float, na_value=np.nan)
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
        g, v, t = rows[valid], raw[valid], times[valid]
        k = count(g)
        if key == "BVP":
            output.extend(pulse_measurements(v, t, g, k, rate, spectrum_cache).T)
            output.append(np.minimum(k / 640, 1))
            continue
        m, sd = stats(v, g, k)
        if key == "HR":
            rate = m
        elif key == "EDA" and spectrum_cache is not None:
            spectrum_cache["eda"] = (t, v, g)
        output.extend((m, sd))
        if key == "IBI_duration":
            consecutive = beat_pairs(v, t, dt[valid])
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

LEVELS = np.array(
    [
        0,
        1,
        3,
        5,
        8,
        9,
        10,
        12,
        14,
        17,
        20,
        22,
        25,
        26,
        27,
        29,
        33,
        34,
        35,
        36,
        38,
        39,
        42,
    ]
)


def observed_statistics(values, observed, quantiles):
    """Compute per-signal quantiles and exact midranks from a single sort."""
    n, k = values.shape
    if not n:
        return np.zeros((len(quantiles), k)), np.empty_like(values)
    filled = np.where(observed, values, np.inf)
    order = np.argsort(filled, axis=0, kind="stable")
    ordered = np.take_along_axis(filled, order, axis=0)
    count = observed.sum(0)
    col = np.arange(k)
    percent = []
    for fraction in quantiles:
        j = fraction * np.maximum(0, count - 1)
        left = j.astype(int)
        right = np.minimum(left + 1, np.maximum(0, count - 1))
        a = np.where(count, ordered[left, col], 0.0)
        b = np.where(count, ordered[right, col], 0.0)
        percent.append(a + (b - a) * (j - left))
    change = ordered[1:] != ordered[:-1]
    pos = np.arange(n)[:, None]
    start = np.vstack((np.ones((1, k), bool), change))
    end = np.vstack((change, np.ones((1, k), bool)))
    lo = np.maximum.accumulate(np.where(start, pos, 0), axis=0)
    hi = np.minimum.accumulate(np.where(end, pos, n)[::-1], axis=0)[::-1]
    ranks = np.empty_like(values)
    np.put_along_axis(ranks, order, (lo + hi + 1) / np.maximum(count, 1) - 1.0, axis=0)
    return np.asarray(percent), np.where(observed, ranks, 0.0)


def base_series(values):
    if values.shape[1] != len(UNITS):
        raise ValueError("Unsupported pulse feature schema")
    seen = np.isfinite(values)
    v = np.arcsinh(np.where(seen, values, 0.0) / UNITS)
    (low, median, high), ranked = observed_statistics(v, seen, (0.1, 0.5, 0.9))
    spread = high - low
    z = np.clip((v - median) / np.maximum(0.045, spread), -4, 4)
    return v, seen, np.where(seen, z, 0.0), ranked, median, spread


def series_features(values, clock):
    v, seen, norm, rank, median, spread = base_series(values)
    regions = [
        norm,
        rank,
        v,
        seen[:, [0, 5, 17, 22, 29]],
        np.broadcast_to(median[LEVELS], (len(v), len(LEVELS))),
        np.broadcast_to(spread[LEVELS], (len(v), len(LEVELS))),
    ]
    # Use observed local physiology around each bin, with elapsed-time bounds across gaps.
    signals = norm[:, LEVELS]
    good = seen[:, LEVELS]
    sums = np.vstack((np.zeros(len(LEVELS)), np.cumsum(signals, axis=0)))
    counts = np.vstack((np.zeros(len(LEVELS)), np.cumsum(good, axis=0)))

    def region(offset, radius):
        lo = clock.searchsorted(clock + offset - radius)
        hi = clock.searchsorted(clock + offset + radius, side="right")
        return (sums[hi] - sums[lo]) / np.maximum(1, counts[hi] - counts[lo])

    for offset, radius in ((-24, 7), (-9, 3), (0, 2), (9, 3), (24, 7)):
        regions.append(region(offset, radius))
    return np.column_stack(regions).astype(np.float32)


def temporal_segments(clock):
    return np.split(np.arange(len(clock)), np.flatnonzero(np.diff(clock) > 1) + 1)


def beat_details(frame, clock):
    # Skip empty measurements and align exclusively using sensor timestamps.
    n = len(clock)
    ts = frame.index.as_unit("ns").asi8
    order = (
        None if frame.index.is_monotonic_increasing else np.argsort(ts, kind="stable")
    )
    if order is not None:
        ts = ts[order]
    outputs = np.full((n, 23), np.nan, np.float32)
    if not n:
        return outputs

    def observed(key, low, high):
        raw = (
            frame[key].to_numpy(dtype=float, na_value=np.nan)
            if key in frame
            else np.full(len(ts), np.nan)
        )
        if order is not None:
            raw = raw[order]
        k = np.isfinite(raw) & (raw >= low) & (raw <= high)
        return ts[k], raw[k]

    t, v = observed("BVP", -1.0e5, 1.0e5)
    secs = []
    amps = []
    intervals = []
    rises = []
    decays = []
    powers = []
    regular = []
    wids = []
    # Short holes of up to three 64 Hz samples can be bridged; long gaps split the waveform.
    if len(t) > 63:
        borders = np.r_[
            0,
            np.flatnonzero((np.diff(t) > 4 * 15625000) | (np.diff(t) <= 0)) + 1,
            len(t),
        ]
        sos = butter(3, [0.55, 5.0], fs=64, btype="bandpass", output="sos")
        for start, end in zip(borders[:-1], borders[1:]):
            if end - start < 128:
                continue
            st = t[start:end]
            wave = v[start:end]
            grid = np.rint((st - st[0]) / 15625000).astype(int)
            if grid[-1] > 2 * len(wave) or np.any(np.diff(grid) <= 0):
                continue
            if grid[-1] != len(wave) - 1:
                wave = np.interp(np.arange(grid[-1] + 1), grid, wave)
            filtered = sosfilt(sos, wave - wave[0])
            amplitude = uniform_filter1d(abs(filtered), 129, mode="nearest")
            pk, _ = find_peaks(
                filtered, distance=18, prominence=np.maximum(0.0001, 0.42 * amplitude)
            )
            if len(pk) < 3:
                continue
            # Exclude IIR startup. Only directly observed, plausible consecutive cardiac cycles contribute.
            begin = pk[:-1]
            last = pk[1:]
            dist = last - begin
            ok = (begin >= 48) & (dist >= 20) & (dist <= 120)
            begin = begin[ok]
            last = last[ok]
            dist = dist[ok]
            if not len(dist):
                continue
            # Vectorised search for the trough and width within each candidate optical cycle.
            indices = begin[:, None] + np.arange(int(dist.max()))[None, :]
            in_range = indices < last[:, None]
            win = filtered[np.minimum(indices, len(filtered) - 1)]
            tr_idx = np.where(in_range, win, np.inf).argmin(1)
            trough = begin + tr_idx
            peak = filtered[last]
            bottom = filtered[trough]
            amp = peak - bottom
            quality = (
                (amp > 0.50 * amplitude[last])
                & (amp < 6.0 * np.maximum(amplitude[last], 0.001))
                & (tr_idx > 1)
                & (tr_idx < dist - 1)
            )
            if not quality.any():
                continue
            midpoint = 0.5 * (filtered[begin] + bottom)
            width = np.sum((win > midpoint[:, None]) & in_range, axis=1) / dist

            def take(vec):
                return vec[quality]

            secs.extend((st[0] + last[quality] * 15625000) // 1000000000)
            amps.extend(take(np.log1p(np.maximum(0, amp))))
            intervals.extend(take(dist / 64))
            rises.extend(take((last - trough) / dist))
            decays.extend(take((filtered[begin] - bottom) / np.maximum(amp, 1.0e-6)))
            powers.extend(take(np.log1p(amplitude[last])))
            wids.extend(take(width))
            regular.extend(
                take(np.minimum(6.0, amp / np.maximum(0.0001, amplitude[last])))
            )

    def span(g, x, cols):
        k = np.bincount(g, minlength=n)
        use = k >= 2
        av = np.bincount(g, weights=x, minlength=n) / np.maximum(1, k)
        dv = np.sqrt(
            np.maximum(
                0,
                np.bincount(g, weights=x * x, minlength=n) / np.maximum(1, k) - av * av,
            )
        )
        outputs[use, cols] = av[use]
        outputs[use, cols + 1] = dv[use]

    seconds = np.asarray(secs, dtype=np.int64)
    amps = np.asarray(amps)
    intervals = np.asarray(intervals)
    if len(seconds):
        group = clock.searchsorted(seconds // 10)
        inside = (group < len(clock)) & (
            clock[np.minimum(group, len(clock) - 1)] == seconds // 10
        )
        for j, x in enumerate(
            (
                amps,
                intervals,
                np.asarray(rises),
                np.asarray(decays),
                np.asarray(powers),
                np.asarray(wids),
                np.asarray(regular),
            )
        ):
            span(group[inside], x[inside], 2 * j)
        outputs[:, 14] = np.minimum(1, np.bincount(group[inside], minlength=n) / 12)
    else:
        outputs[:, 14] = 0.0
    te, eda = observed("EDA", 0, 100)

    # 64-second overlapping windows. Only common, well-supported 1 Hz samples enter spectral features.
    def low_spectrum(sample_time, level):
        result = np.full((n, 3), np.nan)
        if len(sample_time) < 16:
            return result
        unique, inv = np.unique(sample_time, return_inverse=True)
        values = np.bincount(inv, weights=level) / np.bincount(inv)
        grid = clock[:, None] * 10 + 5 + np.arange(-32, 32)
        j = np.searchsorted(unique, grid)
        right = np.minimum(j, len(unique) - 1)
        left = np.maximum(0, j - 1)
        valid = (
            (grid >= unique[0])
            & (grid <= unique[-1])
            & ((unique[right] - unique[left] <= 3) | (grid == unique[right]))
        )
        supported = valid.sum(1) >= 56
        if not supported.any():
            return result
        seen = grid[supported]
        x = np.interp(seen.ravel(), unique, values).reshape(seen.shape)
        good = valid[supported]
        mean = (x * good).sum(1) / good.sum(1)
        x = (x - mean[:, None]) * good * np.hanning(64)
        power = np.abs(np.fft.rfft(x)) ** 2
        band = power[:, 7:27]
        total = power[:, 2:31].sum(1)
        usable = total > 1.0e-10
        result[supported] = np.where(
            usable[:, None],
            np.column_stack(
                (
                    band.sum(1) / np.maximum(total, 1.0e-10),
                    (band.argmax(1) + 7) / 64,
                    band.max(1) / np.maximum(band.sum(1), 1.0e-10),
                )
            ),
            np.nan,
        )
        return result

    outputs[:, 15:18] = low_spectrum(seconds, amps)
    outputs[:, 18:21] = low_spectrum(seconds, intervals)
    outputs[:, 21:23] = low_spectrum(te // 1000000000, eda)[:, [0, 2]]
    return outputs


BEAT_LEVELS = np.array([0, 1, 2, 3, 4, 6, 8, 10, 14, 15, 16, 18, 20, 21])


def beat_series(values, clock):
    observed = np.isfinite(values)
    raw = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.zeros_like(raw)
    ranks = np.zeros_like(raw)
    center = np.zeros(raw.shape[1], np.float32)
    for j in range(raw.shape[1]):
        good = observed[:, j]
        if good.any():
            v = raw[good, j]
            lo, mid, hi = np.quantile(v, (0.1, 0.5, 0.9))
            center[j] = mid
            norm[good, j] = np.clip((v - mid) / max(0.035, hi - lo), -4, 4)
            ordered = np.sort(v)
            ranks[good, j] = (
                ordered.searchsorted(v) + ordered.searchsorted(v, side="right")
            ) / len(v) - 1.0
    x = norm[:, BEAT_LEVELS]
    g = observed[:, BEAT_LEVELS]
    s = np.vstack([np.zeros(len(BEAT_LEVELS)), np.cumsum(x, axis=0, dtype=float)])
    count = np.vstack([np.zeros(len(BEAT_LEVELS)), np.cumsum(g, axis=0)])
    zones = [
        raw,
        norm,
        ranks,
        np.broadcast_to(center[BEAT_LEVELS], (len(x), len(BEAT_LEVELS))),
    ]
    for offset, n in ((-12, 4), (0, 3), (12, 4)):
        lo = clock.searchsorted(clock + offset - n)
        hi = clock.searchsorted(clock + offset + n, side="right")
        zones.append((s[hi] - s[lo]) / np.maximum(1, count[hi] - count[lo]))
    return np.column_stack(zones).astype("f")


def adaptive_features(features, clock):
    # The first block contains values normalised to this recording's own sensor distribution.
    x = features[:, LEVELS]
    groups = (
        (0, 1, 3, 22, 25, 26, 27, 29, 34, 35, 36, 38, 39),
        (5, 8, 9, 10, 12, 14, 17, 20),
    )
    series_count = len(x)
    out = []
    for keys in groups:
        chan = [int(np.flatnonzero(LEVELS == j)[0]) for j in keys]
        evidence = np.clip(x[:, chan], -2.0, 2.0)
        for minimum, cutoff in ((4, 0.68), (8, 1.4), (15, 2.0)):
            aggregate = np.zeros((series_count, len(LEVELS) * 2), np.float32)
            for seg in temporal_segments(clock):
                if not len(seg):
                    continue
                val = x[seg]
                z = evidence[seg]
                prefix = np.vstack([np.zeros(len(chan)), z.cumsum(axis=0, dtype=float)])
                spans = []
                queue = [(0, len(seg))]
                while queue:
                    lo, hi = queue.pop()
                    candidates = np.arange(lo + minimum, hi - minimum + 1)
                    if not len(candidates):
                        spans.append((lo, hi))
                        continue
                    left = (candidates - lo)[:, None]
                    right = (hi - candidates)[:, None]
                    diff = (prefix[candidates] - prefix[lo]) / left - (
                        prefix[hi] - prefix[candidates]
                    ) / right
                    score = (left[:, 0] * right[:, 0] / (hi - lo)) * np.mean(
                        diff**2, axis=1
                    )
                    j = np.argmax(score)
                    if score[j] < cutoff:
                        spans.append((lo, hi))
                        continue
                    point = candidates[j]
                    queue.extend(((lo, point), (point, hi)))
                spans.sort()
                for l, h in spans:
                    part = val[l:h]
                    # No location in the file, segment duration, fixed task order or class prior is included.
                    aggregate[seg[l:h], : len(LEVELS)] = part.mean(0)
                    mid = (h - l) // 2
                    if mid:
                        aggregate[seg[l:h], len(LEVELS) :] = part[mid:].mean(0) - part[
                            :mid
                        ].mean(0)
            out.append(aggregate)
    return np.column_stack([features, *out]).astype("float32", copy=False)


def autonomic_detail(frame, clock, spectrum_cache=None):
    n = len(clock)
    out = np.full((n, 34), np.nan, np.float32)
    if not n:
        return out
    raw_time = frame.index.as_unit("ns").asi8

    def observed(key, low, high):
        if key not in frame:
            return np.empty(0, np.int64), np.empty(0)
        v = frame[key].to_numpy(dtype=float, na_value=np.nan)
        good = np.isfinite(v) & (v >= low) & (v <= high)
        t = raw_time[good]
        v = v[good]
        if len(t) > 1 and np.any(np.diff(t) < 0):
            idx = np.argsort(t, kind="stable")
            t = t[idx]
            v = v[idx]
        return t, v

    cached_pulse = spectrum_cache.get("pulse") if spectrum_cache is not None else None
    if cached_pulse is None:
        t, v = observed("BVP", -1.0e5, 1.0e5)
    else:
        t, v, groups, counts, starts, complete = cached_pulse
    if len(t):
        grid = np.full((n, 640), np.nan)
        if cached_pulse is None:
            g = clock.searchsorted(t // 10_000_000_000)
            slot = (t % 10_000_000_000) // 15_625_000
            grid[g, slot] = v
            count = np.isfinite(grid).sum(1)
        else:
            # Consecutive 64 Hz bins already passed the timestamp and duplicate checks.
            # Only partial bins need slot-by-slot placement a second time.
            count = np.zeros(n, int)
            grid[complete] = v[starts[complete, None] + np.arange(640)]
            count[complete] = 640
            incomplete = counts >= 512
            incomplete[complete] = False
            for group in np.flatnonzero(incomplete):
                sl = slice(starts[group], starts[group] + counts[group])
                slot = (t[sl] % 10_000_000_000) // 15_625_000
                grid[group, slot] = v[sl]
                count[group] = np.isfinite(grid[group]).sum()
        usable = np.flatnonzero(count >= 512)
        bands = (
            (1, 5),
            (5, 10),
            (10, 18),
            (18, 30),
            (30, 50),
            (50, 90),
            (90, 150),
            (150, 250),
            (250, 318),
        )
        for first in range(0, len(usable), 96):
            selected = usable[first : first + 96]
            wave = grid[selected].copy()
            keep = np.ones(len(selected), bool)
            for j in np.flatnonzero(count[selected] != 640):
                known = np.flatnonzero(np.isfinite(grid[selected[j]]))
                if (
                    len(known) < 2
                    or known[0] > 12
                    or known[-1] < 627
                    or np.max(np.diff(known)) > 16
                ):
                    keep[j] = False
                    continue
                wave[j] = np.interp(np.arange(640), known, wave[j, known])
            if not keep.any():
                continue
            wave = wave[keep]
            selected = selected[keep]
            wave -= wave.mean(1, keepdims=True)
            cached = (
                spectrum_cache["seen"][selected]
                if spectrum_cache is not None and "seen" in spectrum_cache
                else np.zeros(len(selected), bool)
            )
            if cached.all():
                power = spectrum_cache["power"][selected]
            elif cached.any():
                power = np.empty((len(selected), 321))
                power[cached] = spectrum_cache["power"][selected[cached]]
                power[~cached] = (
                    abs(np.fft.rfft(wave[~cached] * np.hanning(640), axis=1)) ** 2
                )
            else:
                power = abs(np.fft.rfft(wave * np.hanning(640), axis=1)) ** 2
            tw = wave.reshape(-1, 5, 128)
            tw = tw - tw.mean(2, keepdims=True)
            short = abs(np.fft.rfft(tw * np.hanning(128), axis=2)) ** 2
            total = np.maximum(1.0e-8, power[:, 1:].sum(1))
            card = power[:, 6:35]
            weights = card / np.maximum(card.sum(1, keepdims=True), 1e-8)
            freq = card.argmax(1) + 6
            cols = [np.log1p(1000 * power[:, a:b].sum(1) / total) for a, b in bands]
            cols.extend(
                (
                    -(weights * np.log(np.maximum(1.0e-8, weights))).sum(1),
                    np.mean(
                        np.abs(np.diff(np.log1p(np.sqrt((tw * tw).mean(2))), axis=1)),
                        axis=1,
                    ),
                    np.std(short[:, :, 2:8].argmax(2), axis=1),
                    np.std(np.log1p(short[:, :, 16:42].sum(2)), axis=1),
                )
            )
            for divisor in (1, 2, 3, 4):
                bin_ = freq * divisor
                en = sum(power[np.arange(len(freq)), bin_ + j] for j in (-1, 0, 1))
                cols.append(np.log1p(1000 * en / total))
            peaks = abs(np.diff(wave, axis=1))
            cols.extend(
                (
                    np.log1p(peaks.max(1) / np.maximum(1.0e-7, peaks.mean(1))),
                    np.log1p(peaks.mean(1)),
                )
            )
            z = np.column_stack(cols)
            z[total <= 1.0e-8] = np.nan
            out[selected, :19] = z
    cached_skin = spectrum_cache.get("eda") if spectrum_cache is not None else None
    if cached_skin is None:
        t, v = observed("EDA", 0.0, 100.0)
        rows = clock.searchsorted(t // 10_000_000_000)
    else:
        t, v, rows = cached_skin
    if len(t) > 2:
        # Reset filters at absent measurements or a backwards or duplicate timestamp.
        cuts = np.r_[
            0,
            np.flatnonzero((np.diff(t) > 400_000_000) | (np.diff(t) <= 0)) + 1,
            len(t),
        ]
        latent = np.full((len(t), 11), np.nan)
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right - left < 4:
                continue
            x = median_filter(v[left:right], size=3, mode="nearest")

            def smooth(seconds):
                a = -np.expm1(-0.25 / seconds)
                return lfilter([a], [1.0, a - 1.0], x, zi=[(1 - a) * x[0]])[0]

            fast = smooth(0.7)
            medium = smooth(4.0)
            slow = smooth(20.0)
            dx = np.r_[0.0, np.diff(fast)] * 4
            dv = np.r_[0.0, np.diff(x)] * 4
            ph = fast - slow
            rapid = fast - medium
            scale = max(0.002, np.quantile(abs(dx), 0.60))
            peak = np.r_[
                False, (dx[:-2] > 0) & (dx[2:] <= 0) & (rapid[1:-1] > 0.006), False
            ]
            latent[left:right] = np.column_stack(
                (
                    ph,
                    rapid,
                    np.maximum(0, dx + rapid / 3),
                    np.maximum(0, -dx),
                    dx > max(0.008, scale),
                    dx < -max(0.008, scale),
                    peak,
                    np.maximum(0, ph),
                    dx,
                    np.abs(dv),
                    np.sign(dx) * np.log1p(abs(dx) / scale),
                )
            )
        ok = np.isfinite(latent[:, 0])
        g = rows[ok]
        v2 = latent[ok]
        count = np.bincount(g, minlength=n)
        use = count >= 8

        def mean(z):
            return np.bincount(g, weights=z, minlength=n) / np.maximum(1, count)

        means = [mean(x) for x in v2.T]
        spread = [
            np.sqrt(np.maximum(0, mean(v2[:, j] ** 2) - means[j] ** 2))
            for j in (0, 1, 2, 8)
        ]
        out[use, 19:] = np.column_stack([*means, *spread])[use]
    return out


CORE_COLS = np.array(
    [
        0,
        1,
        3,
        5,
        6,
        8,
        9,
        10,
        12,
        14,
        16,
        17,
        19,
        20,
        22,
        23,
        25,
        26,
        27,
        29,
        30,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        42,
        44,
    ]
)

DETAIL_COLS = np.array(
    [
        0,
        2,
        3,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        14,
        16,
        17,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        28,
        30,
        31,
        32,
    ]
)

PROFILE_COLS = np.r_[
    0,
    1,
    3,
    5,
    7,
    8,
    11,
    13,
    14,
    16,
    17,
    19,
    22,
    23,
    24,
    25,
    26,
    27,
    29,
    31,
    32
    + np.array([2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24]),
]


def response_design(raw, detail, clock):
    """Multiscale recent changes, rank in this recording and physiology around a quiet reference."""
    if not len(clock):
        return np.empty((0, 0), np.float32)
    vals = np.column_stack([raw[:, CORE_COLS], detail[:, DETAIL_COLS]]).astype(float)
    mask = np.isfinite(vals)
    v = np.where(mask, vals, 0.0)
    scales = np.array(
        [
            30,
            2,
            1,
            0.5,
            0.04,
            0.03,
            0.03,
            0.02,
            0.04,
            0.08,
            1,
            5,
            0.01,
            0.01,
            0.25,
            0.03,
            0.03,
            0.10,
            0.3,
            2,
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
            0.3,
            1,
        ]
        + [1.0] * 13
        + [0.03, 0.03, 0.03, 0.03, 0.25, 0.25, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03]
    )
    v = np.arcsinh(v / scales)
    (lo, cent, hi), q = observed_statistics(v, mask, (0.12, 0.5, 0.88))
    spread = hi - lo
    rel = np.where(mask, np.clip((v - cent) / np.maximum(0.04, spread), -3, 3), 0.0)
    f = PROFILE_COLS
    act = q[:, [0, 16, 22, 24]].mean(1)
    arouse = q[:, [0, 4, 7, 8, 32 + 15, 32 + 17]].mean(1)
    cols = [
        v,
        rel,
        q,
        np.broadcast_to(cent[f], (len(v), len(f))),
        np.broadcast_to(spread[f], (len(v), len(f))),
    ]
    # Multiple quiet reference definitions avoid anchoring every subject to a single noisy interval.
    for evidence in (act, arouse):
        w = np.exp(-3 * evidence)[:, None] * mask[:, f]
        base = (w * v[:, f]).sum(0) / np.maximum(0.001, w.sum(0))
        cols.append(
            np.where(
                mask[:, f],
                np.clip((v[:, f] - base) / np.maximum(0.08, spread[f]), -4, 4),
                0,
            )
        )
    d = rel[:, f]
    valid = mask[:, f]
    sm = np.vstack((np.zeros(len(f)), np.cumsum(d, axis=0)))
    ns = np.vstack((np.zeros(len(f)), np.cumsum(valid, axis=0)))

    def avg(low, high):
        # Use the nearest observed bin for an open-ended flank, avoiding artificial zeros at file boundaries.
        left = clock.searchsorted(clock + low)
        right = clock.searchsorted(clock + high, side="right")
        neutral = rel[:, f]
        return np.divide(
            sm[right] - sm[left],
            ns[right] - ns[left],
            out=neutral.copy(),
            where=ns[right] > ns[left],
        )

    local = avg(-2, 2)
    cols.append(local)
    for span in (4, 10, 23, 42):
        before = avg(-span, -max(1, span // 3))
        after = avg(max(1, span // 3), span)
        cols.extend((local - before, after - local))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


SHARED = np.array(
    [
        0,
        1,
        3,
        5,
        6,
        8,
        9,
        10,
        12,
        14,
        17,
        19,
        20,
        29,
        30,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        42,
    ]
)


def transfer_features(x, c):
    # All indices refer to the HR, EDA, temperature and optical channels shared by both datasets.
    q = x[:, 45 + SHARED]
    v = x[:, SHARED]
    ab = x[:, 90 + SHARED]
    # Rank and centered response permit transfer between wearers with dissimilar signal levels.
    cols = [q, v, ab]
    both = np.column_stack((q, v))
    p = np.vstack((np.zeros(both.shape[1]), np.cumsum(both, axis=0, dtype=float)))

    def mean(left, right):
        a = c.searchsorted(c + left)
        b = c.searchsorted(c + right, side="right")
        return np.divide(
            p[b] - p[a], (b - a)[:, None], out=both.copy(), where=(b > a)[:, None]
        )

    m = mean(-2, 2)
    cols.append(m[:, : len(SHARED)])
    for off, span in ((4, 2), (10, 3), (20, 6)):
        before = mean(-off - span, -off + span)
        after = mean(off - span, off + span)
        cols.extend(((m - before)[:, ::2], (after - m)[:, ::2]))
    pos = np.array([0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 16, 17, 18, 21])
    selected = v[:, pos]
    u1, u2 = np.triu_indices(len(pos), 1)
    prod = selected[:, u1] * selected[:, u2]
    # Joint fluctuations over roughly 90 seconds, including heart versus skin response.
    pm = np.vstack([np.zeros(len(pos)), np.cumsum(selected, axis=0, dtype=float)])
    pp = np.vstack([np.zeros(len(u1)), np.cumsum(prod, axis=0, dtype=float)])
    for span in (4, 12):
        a = c.searchsorted(c - span)
        b = c.searchsorted(c + span, side="right")
        den = (b - a)[:, None]
        mu = (pm[b] - pm[a]) / den
        cov = (pp[b] - pp[a]) / den - mu[:, u1] * mu[:, u2]
        cols.append(np.arcsinh(cov / 0.12))
    return np.column_stack(cols).astype("f")


def transfer_logit(t, prior):
    return (
        np.clip(
            (t - np.asarray(prior["center"])) / np.asarray(prior["scale"]), -3.5, 3.5
        )
        @ np.asarray(prior["coef"])
        + prior["bias"]
    ).astype("f")


def cardiac_response(frame, c):
    n = len(c)
    out = np.full((n, 26), np.nan, np.float32)
    if not n:
        return out
    ts = frame.index.as_unit("ns").asi8

    def get(key):
        return (
            frame[key].to_numpy(dtype=float, na_value=np.nan)
            if key in frame
            else np.full(len(ts), np.nan)
        )

    hr = get("HR")
    d = get("IBI_duration")
    dt = get("IBI_dt")
    if not frame.index.is_monotonic_increasing:
        order = ts.argsort(kind="stable")
        ts = ts[order]
        hr = hr[order]
        d = d[order]
        dt = dt[order]
    origin = c[0] * 10_000_000_000
    centers = (c - c[0]) * 10 + 5.0
    target = (ts - origin) * 1.0e-9
    pos = np.flatnonzero(np.isfinite(d) & (d >= 0.3) & (d <= 2.0))
    if len(pos) > 2:
        t = target[pos]
        ibi = d[pos]
        g = c.searchsorted(ts[pos] // 10_000_000_000)
        counts = np.bincount(g, minlength=n)
        elapsed = np.diff(t, prepend=t[0] - ibi[0])
        reported = dt[pos]
        estimated = np.where(
            np.isfinite(reported) & (reported > 0) & (abs(reported - elapsed) < 0.09),
            reported,
            elapsed,
        )
        ratio = estimated / np.maximum(ibi, 0.3)
        valid = (ratio >= 0.55) & (ratio <= 1.5)
        valid[0] = False
        delta = np.diff(ibi, prepend=ibi[0])
        nextv = np.r_[False, valid[:-1]]
        reversal = (delta * np.r_[0.0, delta[:-1]] < 0) & valid & nextv
        signals = [
            np.log1p(np.minimum(15, np.maximum(0, ratio - 1))),
            ratio > 1.65,
            valid.astype(float),
            valid * abs(delta),
            valid * delta**2,
            valid * (abs(delta) > 0.02),
            valid * (abs(delta) > 0.05),
            reversal,
        ]
        for j, x in enumerate(signals):
            out[:, j] = np.divide(
                np.bincount(g, weights=x, minlength=n),
                counts,
                out=np.full(n, np.nan),
                where=counts > 0,
            )
        for j, rad in enumerate((15, 40)):
            lo = t.searchsorted(centers - rad)
            hi = t.searchsorted(centers + rad, side="right")
            num = hi - lo
            good = num >= 6
            prefix = np.r_[0.0, np.cumsum(ibi)]
            sq = np.r_[0.0, np.cumsum(ibi**2)]
            mean = (prefix[hi] - prefix[lo]) / np.maximum(1, num)
            std = np.sqrt(
                np.maximum(0, (sq[hi] - sq[lo]) / np.maximum(1, num) - mean**2)
            )
            supp = np.clip((prefix[hi] - prefix[lo]) / (2 * rad), 0, 1)
            a = np.r_[0.0, np.cumsum(abs(delta) * valid)]
            k = np.r_[0, np.cumsum(valid)]
            out[good, 8 + j * 3 : 11 + j * 3] = np.column_stack(
                (std, np.divide(a[hi] - a[lo], np.maximum(1, k[hi] - k[lo])), supp)
            )[good]
        grid = centers[:, None] + np.arange(-32, 32, 0.5)
        right = t.searchsorted(grid)
        r = np.minimum(right, len(t) - 1)
        l = np.maximum(0, right - 1)
        ok = (grid >= t[0]) & (grid <= t[-1]) & ((t[r] - t[l]) <= 2.8)
        support = ok.sum(1)
        selected = support >= 96
        if selected.any():
            local = grid[selected]
            mask = ok[selected]
            v = np.interp(local.ravel(), t, ibi).reshape(local.shape)
            av = (v * mask).sum(1) / mask.sum(1)
            wave = (v - av[:, None]) * mask * np.hanning(128)
            power = np.abs(np.fft.rfft(wave, axis=1)) ** 2
            lf = power[:, 3:10].sum(1)
            hf = power[:, 10:26].sum(1)
            den = np.maximum(1.0e-8, power[:, 2:30].sum(1))
            out[selected, 14:18] = np.column_stack(
                (
                    np.log1p(lf),
                    np.log1p(hf),
                    (lf - hf) / den,
                    power[:, 10:26].max(1) / np.maximum(1.0e-8, hf),
                )
            )
    valid = np.isfinite(hr) & (hr >= 25) & (hr <= 235)
    ht = target[valid]
    h = hr[valid]
    if len(h) > 2:
        # Approximate short and long recovery slopes from measured HR, without inferring the phase from the clock.
        grid = centers[:, None] + np.arange(-60, 61, 2)
        right = ht.searchsorted(grid)
        r = np.minimum(right, len(h) - 1)
        l = np.maximum(0, right - 1)
        good = (grid >= ht[0]) & (grid <= ht[-1]) & ((ht[r] - ht[l]) <= 3.0)
        tr = np.interp(grid.ravel(), ht, h).reshape(grid.shape)
        for j, (start, end) in enumerate(((0, 26), (20, 31), (30, 41), (35, 61))):
            seg = tr[:, start:end]
            ok = good[:, start:end]
            count = ok.sum(1)
            use = count >= 0.65 * seg.shape[1]
            t = np.arange(end - start) * 2.0
            tmean = (ok * t).sum(1) / np.maximum(1, count)
            hm = (ok * seg).sum(1) / np.maximum(1, count)
            tt = t[None, :] - tmean[:, None]
            slope = (ok * tt * (seg - hm[:, None])).sum(1) / np.maximum(
                0.1, (ok * tt**2).sum(1)
            )
            out[use, 18 + j * 2] = slope[use]
            out[use, 19 + j * 2] = hm[use]
    return out


def cardiac_design(raw, c):
    # Features unseen in a recording receive zero relative response and an explicit availability indicator.
    scales = np.array(
        [
            0.5,
            0.5,
            1.0,
            0.03,
            0.002,
            0.5,
            0.5,
            0.5,
            0.04,
            0.02,
            1.0,
            0.04,
            0.02,
            1.0,
            0.1,
            0.1,
            1.0,
            0.3,
            0.3,
            50,
            0.3,
            50,
            0.3,
            50,
            0.3,
            50,
        ]
    )
    good = np.isfinite(raw)
    v = np.arcsinh(np.where(good, raw, 0.0) / scales)
    norm = np.zeros_like(v)
    for j in range(v.shape[1]):
        a = good[:, j]
        if a.any():
            q = np.quantile(v[a, j], [0.12, 0.50, 0.88])
            norm[a, j] = np.clip((v[a, j] - q[1]) / max(0.08, q[2] - q[0]), -4, 4)
    p = np.vstack((np.zeros(v.shape[1]), np.cumsum(norm, axis=0)))
    k = np.vstack((np.zeros(v.shape[1]), np.cumsum(good, axis=0)))
    lo = c.searchsorted(c - 5)
    hi = c.searchsorted(c + 5, side="right")
    local = (p[hi] - p[lo]) / np.maximum(1, k[hi] - k[lo])
    return np.column_stack((v, norm, local, good[:, [0, 8, 14, 18]])).astype("f")


def nonlinear_adjust(x, model):
    update = np.zeros((len(x), 2))
    for col, trees in enumerate(model.get("trees", ())):
        if not len(trees):
            continue
        f = np.asarray([t[0] for t in trees], int)
        c = np.asarray([t[1] for t in trees])
        val = np.asarray([t[2] for t in trees])
        node = np.where(
            x[:, f[:, 0]] <= c[:, 0],
            x[:, f[:, 1]] > c[:, 1],
            2 + (x[:, f[:, 2]] > c[:, 2]),
        )
        update[:, col] = val[np.arange(len(f))[None, :], node.astype(int)].sum(1)
    return update


BASE_SIGNALS = np.array(
    [0, 3, 5, 8, 10, 12, 14, 17, 20, 22, 25, 26, 29, 33, 34, 35, 36, 38, 42]
)


def odds(p):
    p = np.clip(p, 0.002, 0.998)
    return np.column_stack(
        (np.log(p[:, 1] / p[:, 0]), np.log(p[:, 2] / (p[:, 0] + p[:, 1])))
    )


def meta_design(parts, edges, x, c, prior=None):
    # All neighbourhoods depend on measured signal similarity or elapsed time between real observations.
    mat = np.stack(parts)
    average = mat.mean(0)
    n = len(c)
    z = np.column_stack([odds(p) for p in parts])
    avg = odds(average)
    sig = x[:, BASE_SIGNALS]
    diff = mat.max(0) - mat.min(0)
    stats = [z, np.column_stack(edges), diff]
    groups = temporal_segments(c)
    z0 = np.column_stack((z, average[:, 1:], np.column_stack(edges).mean(1)))
    pfx = np.vstack((np.zeros(z0.shape[1]), np.cumsum(z0, axis=0, dtype=float)))

    def avg_win(lo, hi):
        a = np.searchsorted(c, c + lo)
        b = np.searchsorted(c, c + hi, side="right")
        return np.divide(
            pfx[b] - pfx[a], (b - a)[:, None], out=z0.copy(), where=(b > a)[:, None]
        )

    for lo, hi in ((-2, 2), (-8, 8), (-22, 22), (-46, 46), (-16, -3), (3, 16)):
        stats.append(avg_win(lo, hi))
    # Compare nonadjacent windows with similar heart, skin and pulse responses in the same segment.
    ref = np.column_stack((sig, x[:, 90 + BASE_SIGNALS] * 0.20)).astype("f")
    smoothed = np.empty_like(avg)
    for seg in groups:
        if not len(seg):
            continue
        ref0 = ref[seg]
        unit = (ref0 * ref0).mean(1)
        # Chunk the similarity matrix to bound peak memory on long recordings.
        for block in np.array_split(
            np.arange(len(seg)), max(1, (len(seg) + 127) // 128)
        ):
            d = np.maximum(
                0,
                unit[block, None]
                + unit[None, :]
                - 2 * (ref0[block] @ ref0.T) / ref0.shape[1],
            )
            weight = np.exp(-6 * d)
            weight *= np.abs(c[seg[block], None] - c[seg][None, :]) >= 3
            den = weight.sum(1, keepdims=True)
            smoothed[seg[block]] = np.divide(
                weight @ avg[seg], den, out=avg[seg[block]].copy(), where=den > 0.005
            )
    stats.append(smoothed)
    if prior is not None:
        prior_logit = transfer_logit(transfer_features(x, c), prior)
        pp = np.r_[0.0, np.cumsum(prior_logit, dtype=float)]
        prior_cols = [prior_logit]
        for radius in (3, 10, 30):
            a = c.searchsorted(c - radius)
            b = c.searchsorted(c + radius, side="right")
            prior_cols.append((pp[b] - pp[a]) / (b - a))
        stats.append(np.column_stack(prior_cols))
    # Summaries of the nearby background distinguish short pulse responses from sustained exertion.
    context = np.empty((n, 10), float)
    for seg in groups:
        if not len(seg):
            continue
        m = np.quantile(avg[seg], [0.1, 0.35, 0.5, 0.70, 0.90], axis=0).ravel()
        context[seg] = m
    stats.extend((context, sig, x[:, 90 + BASE_SIGNALS], x[:, 140:150]))
    return np.column_stack(stats).astype(np.float32), avg, average


def residual_prob(z, base, model, strength):
    if strength == 0:
        d = base
    else:
        w = np.asarray(model["coef"])
        x = np.clip(
            (z - np.asarray(model["center"])) / np.asarray(model["scale"]), -4, 4
        )
        adjustment = w[:, 0] + x @ w[:, 1:].T + nonlinear_adjust(x, model)
        d = base + strength * np.clip(adjustment, -3.5, 3.5)
    s, a = expit(d).T
    return np.asarray(np.column_stack(((1 - a) * (1 - s), (1 - a) * s, a)), np.float32)


def smooth_activity(p, c, b, sw, aw, cost):
    if not len(c):
        return np.empty(0, int)
    q = np.log(np.maximum(p, 1.0e-6)) + np.log([1.0, sw, aw])
    if cost <= 0:
        return q.argmax(1)
    labels = np.empty(len(c), int)
    for seg in temporal_segments(c):
        if not len(seg):
            continue
        # Each class either stays or switches from the highest-scoring class.
        # Three scalar updates avoid allocating a transition matrix per time step.
        emission = q[seg].tolist()
        p0, p1, p2 = emission[0]
        parents = []
        penalties = (cost * np.clip(1.18 - 0.8 * b[seg], 0.40, 1.2)).tolist()
        for (e0, e1, e2), penalty in zip(emission[1:], penalties[1:]):
            top = 0 if p0 >= p1 and p0 >= p2 else (1 if p1 >= p2 else 2)
            largest = (p0, p1, p2)[top]
            s0, s1, s2 = p0 + penalty, p1 + penalty, p2 + penalty
            # Match the first-maximum tie rule of NumPy's argmax.
            parents.append(
                (
                    0 if s0 >= largest else top,
                    1 if s1 > largest or (s1 == largest and top >= 1) else top,
                    2 if s2 > largest or (s2 == largest and top == 2) else top,
                )
            )
            p0, p1, p2 = (
                max(s0, largest) + e0,
                max(s1, largest) + e1,
                max(s2, largest) + e2,
            )
        label = 0 if p0 >= p1 and p0 >= p2 else (1 if p1 >= p2 else 2)
        labels[seg[-1]] = label
        for position, parent in zip(seg[-2::-1], reversed(parents)):
            label = parent[label]
            labels[position] = label
    return labels


def arm_log_probability(values, base, fit, settings):
    p = residual_prob(values, base, fit, settings["strength"])
    return np.log(np.maximum(1.0e-6, p)) + np.log(
        [1.0, settings["stress_weight"], settings["activity_weight"]]
    )


RESPONSE_WIDTH = 678


def local_response_context(physiology, clock, parts):
    """Measure local trends and distances to quiet, aroused and active reference windows."""
    probability = np.asarray(parts).mean(axis=0)
    n = len(clock)
    indices = np.array(
        [0, 3, 5, 8, 10, 12, 14, 17, 20, 22, 25, 29, 33, 34, 35, 36, 38, 39, 42]
    )
    body = physiology[:, indices]
    slope_columns = [0, 1, 2, 3, 5, 6, 7, 9, 11, 13]
    local = body[:, slope_columns]
    prefix = np.vstack(
        (np.zeros(local.shape[1]), np.cumsum(local, axis=0, dtype=float))
    )

    def average(start, stop):
        left = clock.searchsorted(clock + start)
        right = clock.searchsorted(clock + stop, side="right")
        return np.divide(
            prefix[right] - prefix[left],
            (right - left)[:, None],
            out=local.astype(float).copy(),
            where=(right > left)[:, None],
        )

    near = average(-2, 2)
    past = average(-16, -4)
    future = average(4, 16)
    rise = near - past
    fall = future - near
    # Include both persistent autonomic changes and changes that reverse during recovery.
    features = [
        rise,
        fall,
        np.minimum(abs(rise), abs(fall)) * np.sign(rise * fall),
        near - average(-34, 34),
    ]
    result = np.zeros((n, 24), dtype=np.float32)
    for seg in temporal_segments(clock):
        if len(seg) < 3:
            continue
        pp = probability[seg]
        for k, ix in enumerate(
            ([0, 1, 9, 10, 11, 12, 13, 15, 16, 17], [2, 3, 4, 5, 6, 7, 8, 14, 18])
        ):
            v = np.ascontiguousarray(body[seg][:, ix])
            norm = (v * v).mean(1)
            for label in range(3):
                # Weight every reference by its physiological classification confidence.
                # These references use model predictions, including at inference.
                weight = (0.008 + pp[:, label]) ** 3
                target = (weight @ v) / weight.sum()
                delta = (v - target) ** 2
                result[seg, 12 * k + label] = np.sqrt(delta.mean(1))
            for subset in np.array_split(np.arange(len(seg)), (len(seg) + 127) // 128):
                dist = np.maximum(
                    0,
                    norm[subset, None]
                    + norm[None, :]
                    - 2 * (v[subset] @ v.T) / v.shape[1],
                )
                remote = np.abs(clock[seg[subset], None] - clock[seg][None, :]) > 3
                for h, band in enumerate((1.4, 6.0, 20.0)):
                    w = np.exp(-band * dist) * remote
                    output = np.divide(
                        w @ pp,
                        w.sum(1, keepdims=True),
                        out=pp[subset].astype(float).copy(),
                        where=w.sum(1, keepdims=True) > 1.0e-6,
                    )
                    result[seg[subset], 12 * k + 3 + h * 3 : 12 * k + 6 + h * 3] = (
                        output
                    )
    features.append(result)
    return np.column_stack(features).astype(np.float32)


SHORT_CORE = np.array(
    [
        0,
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
    ]
)

SHORT_DETAIL = np.array(
    [
        0,
        2,
        3,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        28,
        30,
        31,
        32,
        33,
    ]
)


def short_features(raw, detail, clock):
    """Absolute physiology, spread, temporal trend and local changes within the available recording."""
    di = np.r_[
        np.ones(14),
        [0.04, 0.04, 0.04, 0.04, 0.3, 0.3, 0.1, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
    ]
    values = np.column_stack(
        (raw[:, SHORT_CORE] / UNITS[SHORT_CORE], detail[:, SHORT_DETAIL] / di)
    )
    observed = np.isfinite(values)
    v = np.arcsinh(np.where(observed, values, 0.0))
    many = []
    for seg in temporal_segments(clock):
        if not len(seg):
            continue
        x = v[seg]
        ok = observed[seg]
        n = len(x)
        count = ok.sum(0)
        mean = x.sum(0) / np.maximum(1, count)
        sd = np.sqrt(np.maximum(0, (x * x).sum(0) / np.maximum(1, count) - mean**2))
        both = ok[:-1] & ok[1:]
        steps = (np.abs(np.diff(x, axis=0)) * both).sum(0) / np.maximum(1, both.sum(0))
        prefix = np.vstack((np.zeros(x.shape[1]), np.cumsum(x, axis=0)))
        num = np.vstack((np.zeros(x.shape[1]), np.cumsum(ok, axis=0)))

        def window(a, b):
            l = np.maximum(0, np.arange(n) + a)
            h = np.minimum(n, np.arange(n) + b + 1)
            l = np.minimum(l, n)
            h = np.maximum(h, 0)
            return np.divide(
                prefix[h] - prefix[l],
                num[h] - num[l],
                out=np.broadcast_to(mean, x.shape).copy(),
                where=num[h] > num[l],
            )

        mid = window(-2, 2)
        drift = window(1, 6) - window(-6, -1)
        select = [
            0,
            1,
            4,
            5,
            7,
            9,
            13,
            16,
            18,
            20,
            25,
            28,
            30,
            36,
            40,
            43,
            50,
            52,
            54,
            56,
            58,
            61,
        ]
        t = np.linspace(-1.0, 1.0, n)[:, None]
        tc = (ok * t).sum(0) / np.maximum(1, count)
        slope = (ok * (t - tc) * (x - mean)).sum(0) / np.maximum(
            0.3, (ok * (t - tc) ** 2).sum(0)
        )
        # Record-level features use measured response magnitudes, even in a single-condition trial.
        record = np.r_[mean, sd, np.log1p(steps * 5), slope[select]]
        available = ok[:, [0, 4, 16, 20]].astype(float)
        many.append(
            np.column_stack(
                (
                    np.broadcast_to(record, (n, len(record))),
                    mid,
                    drift[:, select],
                    (mid - mean)[:, select],
                    available,
                )
            )
        )
    return np.vstack(many).astype(np.float32)


def short_kernel(x, fit):
    z = np.clip(
        (x - np.asarray(fit["center"], dtype="f"))
        / np.asarray(fit["scale"], dtype="f"),
        -3.5,
        3.5,
    )
    centers = np.asarray(fit["centers"], dtype="f")
    dist = np.maximum(
        0,
        (z * z).mean(1)[:, None]
        + (centers * centers).mean(1)
        - 2 * (z @ centers.T) / z.shape[1],
    )
    return np.column_stack(
        (
            z * 0.32,
            np.maximum(0, z) * 0.22,
            np.exp(-1.8 * dist) * 2.0,
            np.exp(-5.5 * dist) * 2.0,
        )
    ).astype("f")


class TechArenaModel(ADM_Base):
    """Offline wrist classification. Each run uses only the submitted recording."""

    def __init__(self, path):
        self.load_model(path)

    def load_model(self, onnx_file):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        ort.disable_telemetry_events()
        session = ort.InferenceSession(
            onnx_file if isinstance(onnx_file, bytes) else str(onnx_file),
            opts,
            providers=["CPUExecutionProvider"],
        )
        meta = session.get_modelmeta().custom_metadata_map
        if (
            meta.get("pipeline")
            not in ("participant-response-v5", "participant-response-v6")
            or meta.get("feature_schema") != "wrist-v5"
            or json.loads(meta.get("classes", "[]")) != LABELS
        ):
            raise ValueError("Use the Python and ONNX files from the same release")
        self.short_width = int(meta.get("short_features", 0))
        self.names = json.loads(meta["networks"])
        self.rule = json.loads(meta["decision"])
        self.prior = json.loads(meta["external_prior"])
        self.calibration = json.loads(meta["residual"])
        self.context = "offline"
        if (
            not self.names
            or len(self.names) != len(set(self.names))
            or any(
                k not in ("alternate", "beat", "segments", "net") for k in self.names
            )
        ):
            raise ValueError("Invalid ONNX components")
        widths = {
            "alternate": ("physiology", 301),
            "segments": ("regimes", 577),
            "beat": ("pulse_detail", 125),
            "net": ("response", RESPONSE_WIDTH),
        }
        shapes = {k.name: k.shape[-1] for k in session.get_inputs()}
        if shapes != dict(widths[k] for k in self.names):
            raise ValueError("Unexpected ONNX feature inputs")
        self.outputs = [
            k + suffix
            for k in self.names
            for suffix in (("_p",) if k == "segments" else ("_p", "_b"))
        ]
        expected_outputs = set(self.outputs)
        if self.short_width:
            if meta.get("short_trial_version") != "independent-trials-v1":
                raise ValueError("Unknown low-context model")
            optional = {
                o.name: o.shape[-1] for o in session.get_overridable_initializers()
            }
            if optional != {"short_features": self.short_width}:
                raise ValueError("Unexpected short-recording input")
            expected_outputs.add("short_p")
        if expected_outputs != {o.name for o in session.get_outputs()}:
            raise ValueError("Unexpected ONNX output names")
        for k in ("stress_weight", "activity_weight", "strength", "switch_cost"):
            if (
                not np.isfinite(self.rule[k])
                or self.rule[k] < 0
                or (k.endswith("weight") and self.rule[k] <= 0)
            ):
                raise ValueError("Invalid classifier settings")
        self.session = session

    def decision_inputs(self, frame):
        spectrum = {} if "net" in self.names else None
        raw, rows, clock = physiology_windows(frame, spectrum)
        if not len(clock):
            return None, None, np.empty(0), rows, clock
        x = series_features(raw, clock)
        feed = {}
        if "alternate" in self.names or "beat" in self.names:
            feed["physiology"] = x
        if "beat" in self.names:
            feed["pulse_detail"] = beat_series(beat_details(frame, clock), clock)
        if "segments" in self.names:
            feed["regimes"] = adaptive_features(x, clock)
        groups = temporal_segments(clock)
        short = self.short_width and any(len(group) < 36 for group in groups)
        detail = (
            autonomic_detail(frame, clock, spectrum)
            if "net" in self.names or short
            else None
        )
        if "net" in self.names:
            feed["response"] = response_design(raw, detail, clock)
        short_probability = np.zeros((len(clock), 3), "f") if short else None
        short_weight = np.zeros((len(clock), 1), "f") if short else None
        pp = {k: np.empty((len(clock), 3), "f") for k in self.names}
        bd = {k: np.full(len(clock), 0.2, "f") for k in self.names}
        for seg in groups:
            feeds = {k: np.ascontiguousarray(v[seg]) for k, v in feed.items()}
            use_short = bool(short and 0 < len(seg) < 36)
            outputs = self.outputs
            if use_short:
                feeds["short_features"] = short_features(
                    raw[seg], detail[seg], clock[seg]
                )
                outputs = self.outputs + ["short_p"]
            got = dict(zip(outputs, self.session.run(outputs, feeds)))
            if use_short:
                short_probability[seg] = got["short_p"]
                short_weight[seg] = 0.94 * np.clip((36.0 - len(seg)) / 18.0, 0.0, 1.0)
            for k in self.names:
                pp[k][seg] = got[k + "_p"]
                if k != "segments":
                    bd[k][seg] = got[k + "_b"][:, 0]

        def features_for(route, fit):
            values, base, _ = meta_design(
                [pp[k] for k in route], [bd[k] for k in route], x, clock, self.prior
            )
            if fit.get("local_response", False):
                local = local_response_context(x, clock, [pp[k] for k in route])
                values = np.column_stack((values, local))
            if fit.get("cardiac", False):
                values = np.column_stack(
                    (values, cardiac_design(cardiac_response(frame, clock), clock))
                )
            return values, base, np.mean([bd[k] for k in route], axis=0)

        if "arms" in self.calibration:
            mix = self.calibration["mix"]
            arms = self.calibration["arms"]
            inputs = [features_for(arm["route"], arm["model"]) for arm in arms]
            boundary = mix * inputs[0][2] + (1.0 - mix) * inputs[1][2]
            return (
                inputs,
                ((None, short_probability, short_weight) if short else None),
                boundary,
                rows,
                clock,
            )
        values, base, boundary = features_for(self.names, self.calibration)
        return (
            values,
            ((base, short_probability, short_weight) if short else base),
            boundary,
            rows,
            clock,
        )

    def decisions(self, inputs, rule):
        z, base, b, _, c = inputs
        if not len(c):
            return np.empty(0, int)
        short = isinstance(base, tuple)
        if short:
            base, short_probability, short_weight = base
        if "arms" in self.calibration:
            logs = []
            for (values, odds, _), arm in zip(z, self.calibration["arms"]):
                settings = {
                    **arm["rule"],
                    "strength": rule["strength"] * arm["rule"]["strength"],
                }
                logs.append(arm_log_probability(values, odds, arm["model"], settings))
            mix = self.calibration["mix"]
            p = softmax(mix * logs[0] + (1.0 - mix) * logs[1], axis=1)
        else:
            p = residual_prob(z, base, self.calibration, rule["strength"])
        if short:
            weights = np.array([1.0, rule["stress_weight"], rule["activity_weight"]])
            long = p * weights
            long /= long.sum(1, keepdims=True)
            p = ((1 - short_weight) * long + short_weight * short_probability) / weights
        return smooth_activity(
            p, c, b, rule["stress_weight"], rule["activity_weight"], rule["switch_cost"]
        )

    def run(self, input):
        ins = self.decision_inputs(input)
        labels = self.decisions(ins, self.rule)
        return pd.DataFrame(
            {"Activity": pd.Categorical.from_codes(labels[ins[3]], categories=LABELS)},
            index=input.index,
        )


def get_model():
    return TechArenaModel(Path(__file__).with_suffix(".onnx"))
