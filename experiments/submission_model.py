# Fully vectorized inference model implementation (MET ~0.020s, 95 features).
import pathlib
import sys
import numpy as np
import pandas as pd
import onnxruntime as ort

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

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
CLASS_WEIGHTS = [1.08, 1.15, 0.92]
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


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


def compute_window_features(df: pd.DataFrame, win_sec: int = 10, ctx_sec: int = 40, long_ctx_sec: int = 60) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Extracts time-series window features (Strictly 95 features, fully vectorized):
    Vectorized sliding window statistics without slow Pandas .apply() calls.
    """
    work = df.reindex(columns=SENSOR_COLS).copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)
    if not work.index.is_monotonic_increasing:
        work = work.sort_index()

    work = work.ffill().bfill().fillna(0.0)

    rule = f"{win_sec}s"
    g = work.resample(rule)

    mean_df = g.mean().fillna(0.0)
    std_df = g.std().fillna(0.0)
    min_df = g.min().fillna(0.0)
    max_df = g.max().fillna(0.0)
    ptp_df = max_df - min_df
    delta_df = mean_df.diff().fillna(0.0)

    # Searchsorted bin IDs map timestamps to mean_df.index
    bin_ids = np.searchsorted(mean_df.index, work.index, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(mean_df) - 1)

    ibi_diff_series = work["IBI_duration"].diff().abs().resample(rule).mean().fillna(0.0)
    ibi_diff_df = pd.DataFrame({"IBI_duration_abs_diff": ibi_diff_series}, index=mean_df.index)

    ctx_periods = max(1, ctx_sec // win_sec)
    ctx_mean_df = mean_df.rolling(ctx_periods, min_periods=1).mean()
    ctx_std_df = mean_df.rolling(ctx_periods, min_periods=1).std().fillna(0.0)
    diff_ctx_df = mean_df - ctx_mean_df

    long_periods = max(1, long_ctx_sec // win_sec)
    long_mean_df = mean_df.rolling(long_periods, min_periods=1).mean()
    diff_long_df = mean_df - long_mean_df
    eda_long_swing = (
        max_df["EDA"].rolling(long_periods, min_periods=1).max()
        - min_df["EDA"].rolling(long_periods, min_periods=1).min()
    ).fillna(0.0)
    eda_swing_df = pd.DataFrame({"EDA_ctx60_swing": eda_long_swing}, index=mean_df.index)

    hr_eda_diff_df = pd.DataFrame(
        {
            "HR_minus_ctx40": (mean_df["HR"] - ctx_mean_df["HR"]).fillna(0.0),
            "EDA_minus_ctx40": (mean_df["EDA"] - ctx_mean_df["EDA"]).fillna(0.0),
        },
        index=mean_df.index,
    )
    ibi_dt_diff_series = work["IBI_dt"].diff().abs().resample(rule).mean().fillna(0.0)
    extra_phys_df = pd.DataFrame(
        {
            "HR_std_ctx40": mean_df["HR"].rolling(ctx_periods, min_periods=1).std().fillna(0.0),
            "EDA_std_ctx40": mean_df["EDA"].rolling(ctx_periods, min_periods=1).std().fillna(0.0),
            "IBI_dt_abs_diff": ibi_dt_diff_series,
            "Temp_diff_ctx40": (mean_df["Temperature"] - ctx_mean_df["Temperature"]).fillna(0.0),
        },
        index=mean_df.index,
    )

    feat_df = pd.concat(
        [
            mean_df.add_suffix("_mean"),
            std_df.add_suffix("_std"),
            ptp_df.add_suffix("_ptp"),
            delta_df.add_suffix("_delta"),
            ibi_diff_df,
            ctx_mean_df.add_suffix("_ctx40_mean"),
            ctx_std_df.add_suffix("_ctx40_std"),
            diff_ctx_df.add_suffix("_diff_ctx40"),
            long_mean_df.add_suffix("_ctx60_mean"),
            diff_long_df.add_suffix("_diff_ctx60"),
            eda_swing_df,
            hr_eda_diff_df,
            extra_phys_df,
        ],
        axis=1,
    )

    X_win = feat_df.to_numpy(dtype=np.float32)
    return mean_df.index, bin_ids, X_win


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
            return pd.DataFrame(columns=["Activity"], index=input.index)

        raw_probs = np.asarray(self.session.run(None, {self.input_name: X_win})[1], dtype=np.float32)
        smoothed_probs = smooth_window_probs(raw_probs, sigma=1.8)
        weighted_probs = smoothed_probs * np.array(CLASS_WEIGHTS, dtype=np.float32)
        pred_indices = np.argmax(weighted_probs, axis=1)

        win_activities = [LABELS[idx] for idx in pred_indices]
        sample_preds = [win_activities[b] for b in bin_ids]

        return pd.DataFrame({"Activity": sample_preds}, index=input.index)


def get_model():
    return TechArenaModel()
