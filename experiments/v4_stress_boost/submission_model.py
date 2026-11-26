# Inference model implementation with synchronized 96-feature extraction.
import pathlib
import sys
import numpy as np
import pandas as pd
import onnxruntime as ort

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
for p in (SCRIPT_DIR, ROOT_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from adm_base_class import ADM_Base
except ImportError:
    class ADM_Base:
        def run(self, input: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError

LABELS = ["Rest", "Stress", "Activity"]
CLASS_WEIGHTS = [1.08, 0.85, 1.08]


def smooth_window_probs(probs: np.ndarray, sigma: float = 1.8) -> np.ndarray:
    """Apply Gaussian temporal smoothing across consecutive window probabilities."""
    n = len(probs)
    if n <= 1:
        return probs.copy()
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        padded = np.pad(probs[:, c], (radius, radius), mode="edge")
        smoothed[:, c] = np.convolve(padded, kernel, mode="valid")

    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return smoothed / row_sums


def compute_window_features(df: pd.DataFrame, win_sec: int = 10) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Extracts time-series window features (Strictly 96 features):
    1. Motion Magnitude (ACC_mag) & HR/EDA Baseline Normalization
    2. Ratios: Motion_Stress_Ratio & HR_Motion_Ratio
    3. Rolling Cooldown Trends (120s HR, 180s EDA)
    4. Window Summary Aggregations + HRV RMSSD
    
    Returns:
        (win_timestamps, bin_ids, X_win)
        - win_timestamps: DatetimeIndex of window start times
        - bin_ids: per-sample integer bin indices matching len(df)
        - X_win: (n_windows, 96) feature matrix
    """
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    start_time = data.index[0]
    total_seconds = (data.index - start_time).total_seconds().to_numpy()
    raw_bin_ids = (total_seconds // win_sec).astype(int)
    unique_bins = np.unique(raw_bin_ids)
    bin_ids = np.searchsorted(unique_bins, raw_bin_ids)

    # Base sensor fallback initialization
    for col in ["HR", "EDA", "BVP", "TEMP", "ACC_x", "ACC_y", "ACC_z"]:
        if col not in data.columns:
            data[col] = 0.0

    # Motion Magnitude
    data["ACC_mag"] = np.sqrt(data["ACC_x"] ** 2 + data["ACC_y"] ** 2 + data["ACC_z"] ** 2)

    # Session Baseline Percentiles
    hr_series = data["HR"]
    eda_series = data["EDA"]
    hr_floor = float(hr_series.quantile(0.05)) if len(hr_series) > 0 else 60.0
    hr_q95 = float(hr_series.quantile(0.95)) if len(hr_series) > 0 else 140.0
    hr_iqr = max(hr_q95 - hr_floor, 1.0)
    data["HR_rel_floor"] = (hr_series - hr_floor) / hr_iqr

    eda_floor = float(eda_series.quantile(0.05)) if len(eda_series) > 0 else 0.01
    eda_q95 = float(eda_series.quantile(0.95)) if len(eda_series) > 0 else 5.0
    eda_iqr = max(eda_q95 - eda_floor, 1e-4)
    data["EDA_rel_floor"] = (eda_series - eda_floor) / eda_iqr

    # Physiological Ratios for Stress vs Exercise
    data["Motion_Stress_Ratio"] = (data["EDA_rel_floor"] + 1e-3) / (data["ACC_mag"] + 1e-3)
    data["HR_Motion_Ratio"] = (data["HR_rel_floor"] + 1e-3) / (data["ACC_mag"] + 1e-3)

    # Extended Trends
    data["HR_mean_120s"] = hr_series.rolling(window=120, min_periods=1).mean()
    data["HR_slope_120s"] = hr_series.diff(120).fillna(0.0) / 120.0
    data["EDA_mean_180s"] = eda_series.rolling(window=180, min_periods=1).mean()
    data["EDA_trend_180s"] = eda_series - data["EDA_mean_180s"]

    # Core Channels for Aggregation (12 channels -> 60 window stats)
    core_cols = [
        "HR", "EDA", "BVP", "TEMP", "ACC_mag",
        "HR_rel_floor", "EDA_rel_floor", "Motion_Stress_Ratio",
        "HR_Motion_Ratio", "HR_mean_120s", "HR_slope_120s", "EDA_trend_180s"
    ]

    grouped_core = data[core_cols].groupby(bin_ids)
    c_mean = grouped_core.mean().add_suffix("_mean")
    c_std = grouped_core.std().fillna(0.0).add_suffix("_std")
    c_min = grouped_core.min().add_suffix("_min")
    c_max = grouped_core.max().add_suffix("_max")
    c_ptp = (grouped_core.max() - grouped_core.min()).add_suffix("_ptp")

    # Derivative Channels (7 channels -> 35 derivative window stats)
    diff_cols = ["HR", "EDA", "ACC_mag", "Motion_Stress_Ratio", "HR_Motion_Ratio", "EDA_rel_floor", "HR_rel_floor"]
    deriv_data = data[diff_cols].diff().fillna(0.0)
    grouped_diff = deriv_data.groupby(bin_ids)
    d_mean = grouped_diff.mean().add_suffix("_diff_mean")
    d_std = grouped_diff.std().fillna(0.0).add_suffix("_diff_std")
    d_min = grouped_diff.min().add_suffix("_diff_min")
    d_max = grouped_diff.max().add_suffix("_diff_max")
    d_ptp = (grouped_diff.max() - grouped_diff.min()).add_suffix("_diff_ptp")

    X_mat = pd.concat([c_mean, c_std, c_min, c_max, c_ptp, d_mean, d_std, d_min, d_max, d_ptp], axis=1)

    # IBI RMSSD Proxy (1 feature) -> Total = 60 + 35 + 1 = 96 features
    if "IBI_duration" in data.columns:
        rmssd_vals = []
        ibi_series = data["IBI_duration"]
        for idx_b in range(len(unique_bins)):
            sub_ibi = ibi_series.loc[bin_ids == idx_b].dropna()
            if len(sub_ibi) > 1:
                diffs = sub_ibi.diff().dropna()
                val = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
            else:
                val = 0.0
            rmssd_vals.append(val)
        X_mat["IBI_rmssd"] = rmssd_vals
    else:
        X_mat["IBI_rmssd"] = 0.0

    X_win = X_mat.to_numpy(dtype=np.float32)
    win_timestamps = pd.DatetimeIndex([start_time + pd.Timedelta(seconds=int(b * win_sec)) for b in unique_bins])

    return win_timestamps, bin_ids, X_win


class TechArenaModel(ADM_Base):
    def __init__(self):
        self.onnx_path = SCRIPT_DIR / "submission_model.onnx"
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        if input.empty:
            return pd.DataFrame(columns=["Activity"])

        win_timestamps, bin_ids, X_win = compute_window_features(input, win_sec=10)
        if len(X_win) == 0:
            return pd.DataFrame(columns=["Activity"])

        raw_probs = np.asarray(self.session.run(None, {self.input_name: X_win})[1], dtype=np.float32)
        smoothed_probs = smooth_window_probs(raw_probs, sigma=1.8)
        weighted_probs = smoothed_probs * np.array(CLASS_WEIGHTS, dtype=np.float32)
        pred_indices = np.argmax(weighted_probs, axis=1)

        preds = [LABELS[idx] for idx in pred_indices]
        return pd.DataFrame({"Activity": preds}, index=win_timestamps)


def get_model():
    return TechArenaModel()
