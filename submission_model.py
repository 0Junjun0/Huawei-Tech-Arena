"""Wearable classifier for complete offline recordings. Requires the challenge ADM_Base."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import onnxruntime as ort

try:
    from adm_base_class import ADM_Base
except ModuleNotFoundError as exc:
    if exc.name != "adm_base_class":
        raise

    class ADM_Base:
        """Standalone compatibility when running outside the challenge harness."""


LABELS = ["Rest", "Stress", "Activity"]
SENSORS = ["HR", "EDA", "Temperature", "BVP", "IBI_duration"]


def regional_features(columns, series, bounds):
    """Compare current physiology with adjacent periods and the person's distribution."""
    good = np.isfinite(series)
    if not good.any():
        columns.extend([np.zeros(len(series))] * 8)
        return
    valid = series[good]
    q10, q50, q90 = np.quantile(valid, [0.1, 0.5, 0.9])
    scale = max(q90 - q10, 1e-6)
    standardized = np.where(good, np.clip((series - q50) / scale, -5, 5), 0.0)
    prefix = np.r_[0.0, standardized.cumsum()]
    n = np.r_[0, good.cumsum()]
    i = np.arange(len(series))
    columns.extend([standardized, np.full(len(i), q90 - q10)])
    for lo, hi in bounds:
        columns.append((prefix[i + 1] - prefix[lo]) / np.maximum(n[i + 1] - n[lo], 1))
        columns.append((prefix[hi] - prefix[i]) / np.maximum(n[hi] - n[i], 1))


def relative_trajectory(columns, series, bounds):
    """Three-bin means at offsets relative to now, plus each level's personal rank."""
    present = np.isfinite(series)
    prefix = np.r_[0.0, np.where(present, series, 0.0).cumsum()]
    counts = np.r_[0, present.cumsum()]
    sorted_values = np.sort(series[present])
    for left, right in bounds:
        n = counts[right] - counts[left]
        local = (prefix[right] - prefix[left]) / np.maximum(n, 1)
        percentile = sorted_values.searchsorted(local) / max(1, len(sorted_values))
        columns.extend((local, np.where(n, percentile, 0.5)))


def window_features(frame, context="long"):
    """Return physiology features and original-row positions in the occupied 10 s bins."""
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans:
        raise ValueError("Expected a timestamp index without missing timestamps")
    times = idx.as_unit("ns").asi8
    order = None
    if not idx.is_monotonic_increasing:
        order = np.argsort(times, kind="stable")
        times = times[order]
    bins = times // 10_000_000_000
    starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
    clock = bins[starts]
    positions = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(times)]))
    if context not in ("long", "short"):
        raise ValueError("Unknown feature context")
    radii = (3, 15) if context == "long" else (1, 4)
    widths = (6, 18) if context == "long" else (2, 6)

    def bounds(left, right):
        return clock.searchsorted(clock + left), clock.searchsorted(
            clock + right, side="right"
        )

    local_bounds = [bounds(-r, r) for r in radii]
    trend_bounds = [bounds(-w, w) for w in widths]
    regional_bounds = [bounds(-w, w) for w in (3, 12, 36)]
    trajectory_bounds = [bounds(k - 1, k + 1) for k in (-18, -9, -4, 0, 4, 9, 18)]
    split = np.arange(len(clock))
    columns = []
    for name in SENSORS:
        if name in frame:
            vals = frame[name].to_numpy(dtype=float, na_value=np.nan)
            if order is not None:
                vals = vals[order]
        else:
            vals = np.full(len(times), np.nan)
        good = np.isfinite(vals)
        v = vals[good]
        b = positions[good]
        count = np.bincount(b, minlength=len(clock))

        def average(values, groups, weights):
            return np.bincount(
                groups, weights=values, minlength=len(clock)
            ) / np.maximum(weights, 1)

        mean = average(v, b, count)
        sd = np.sqrt(np.maximum(average(v * v, b, count) - mean * mean, 0))
        # Actual successive samples only, within one physical bin.
        same = np.diff(b) == 0
        if name == "BVP":
            pulse_times = times[good]
            # Official BVP is 64 Hz. Half a sample of jitter is tolerated.
            period_ns = 15_625_000
            same &= np.abs(np.diff(pulse_times) - period_ns) < period_ns // 2
        g = b[1:][same]
        n = np.bincount(g, minlength=len(clock))
        change = average(np.abs(np.diff(v)[same]), g, n)
        mean[count == 0] = np.nan
        sd[count == 0] = np.nan
        change[n == 0] = np.nan
        if name in ("HR", "EDA", "Temperature", "IBI_duration"):
            regional_features(columns, mean, regional_bounds)
        if name == "EDA":
            # Positive conductance increments complement the unsigned change feature.
            rise = average(np.maximum(np.diff(v)[same], 0.0), g, n)
            rise[n == 0] = np.nan
            regional_features(columns, rise, regional_bounds)
        if name == "BVP":
            relative_trajectory(columns, sd, trajectory_bounds)
        if name in ("HR", "EDA", "Temperature", "IBI_duration"):
            relative_trajectory(columns, mean, trajectory_bounds)
        if name == "IBI_duration":
            # Differences of original beat events, avoiding long dropout intervals.
            dt = np.diff(times[good])
            pair = same & (dt > 0) & (dt <= 2_500_000_000)
            dest = b[1:][pair]
            pair_count = np.bincount(dest, minlength=len(clock))
            squared = np.bincount(
                dest, weights=np.diff(v)[pair] ** 2, minlength=len(clock)
            )
            left = clock.searchsorted(clock - 3)
            right = clock.searchsorted(clock + 3, side="right")
            total = np.r_[0.0, squared.cumsum()]
            nums = np.r_[0, pair_count.cumsum()]
            amount = nums[right] - nums[left]
            rmssd = np.sqrt((total[right] - total[left]) / np.maximum(1, amount))
            rmssd[amount < 5] = np.nan
            regional_features(columns, rmssd, regional_bounds)
        for series in (mean, sd, change):
            present = np.isfinite(series)
            observed = np.where(present, series, 0.0)
            columns.append(observed)
            # Centred summaries: complete recording is available at inference.
            sums = np.r_[0.0, observed.cumsum()]
            nums = np.r_[0, present.cumsum()]
            for left, right in local_bounds:
                local = (sums[right] - sums[left]) / np.maximum(
                    1, nums[right] - nums[left]
                )
                columns.append(local)
            middle = np.median(series[present]) if present.any() else 0.0
            columns.append(np.where(present, series - middle, 0.0))
        # Compare recent physiology with the preceding minute and the recording's
        # lower quartile. No phase, recording duration or participant ID is used.
        for series in (mean, sd):
            valid = np.isfinite(series)
            cumul = np.r_[0.0, np.where(valid, series, 0.0).cumsum()]
            nums = np.r_[0, valid.cumsum()]
            for lo, hi in trend_bounds:
                before = (cumul[split] - cumul[lo]) / np.maximum(
                    1, nums[split] - nums[lo]
                )
                after = (cumul[hi] - cumul[split]) / np.maximum(
                    1, nums[hi] - nums[split]
                )
                columns.append(
                    np.where(
                        (nums[split] > nums[lo]) & (nums[hi] > nums[split]),
                        after - before,
                        0.0,
                    )
                )
            base = np.quantile(series[valid], 0.25) if valid.any() else 0.0
            columns.append(np.where(valid, series - base, 0.0))
        if name == "BVP":
            # Local pulse-wave irregularity at two lags on the challenge BVP stream.
            # Require both valid samples to belong to the same ten-second bin.
            for lag in (8, 32):
                pair = b[lag:] == b[:-lag]
                span = pulse_times[lag:] - pulse_times[:-lag]
                pair &= np.abs(span - lag * period_ns) < period_ns // 2
                dest = b[lag:][pair]
                energy = (v[lag:][pair] - v[:-lag][pair]) ** 2
                counts = np.bincount(dest, minlength=len(clock))
                columns.append(
                    average(energy, dest, counts) / (1 + np.nan_to_num(sd) ** 2)
                )
        columns.append(np.log1p(count))
    if order is not None:
        restored = np.empty_like(positions)
        restored[order] = positions
        positions = restored
    return np.column_stack(columns).astype("float32"), positions, clock


def decisions(prob, clock, radius, stress_weight):
    sums = np.vstack([np.zeros(3), prob.cumsum(axis=0, dtype=float)])
    scores = (
        sums[clock.searchsorted(clock + radius, side="right")]
        - sums[clock.searchsorted(clock - radius)]
    )
    scores[:, 1] *= stress_weight
    return scores.argmax(axis=1)


class TechArenaModel(ADM_Base):
    def __init__(self, path):
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.rule = json.loads(
            self.session.get_modelmeta().custom_metadata_map["decision"]
        )

        self.context = self.rule.pop("context")

    def run(self, input):
        if len(input) == 0:
            return pd.DataFrame({"Activity": pd.Series(index=input.index, dtype=str)})
        x, rows, clock = window_features(input, self.context)
        p = self.session.run(["probabilities"], {"features": x})[0]
        final = decisions(p, clock, **self.rule)
        labels = pd.Categorical.from_codes(final[rows], categories=LABELS)
        return pd.DataFrame({"Activity": labels}, index=input.index)


def get_model():
    return TechArenaModel(Path(__file__).with_suffix(".onnx"))
