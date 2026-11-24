import os
import pathlib
import numpy as np
import pandas as pd
from adm_base_class import ADM_Base

# Enforce single-threaded CPU execution
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

LABELS = ["Rest", "Stress", "Activity"]
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def _compute_window_features(
    df: pd.DataFrame, win_sec: int = 10, ctx_sec: int = 40, long_ctx_sec: int = 60
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    work = df[SENSOR_COLS].copy()
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

    ibi_diff_series = work["IBI_duration"].diff().abs().resample(rule).mean().fillna(0.0)
    ibi_diff_df = pd.DataFrame({"IBI_duration_abs_diff": ibi_diff_series}, index=mean_df.index)

    ctx_periods = max(1, ctx_sec // win_sec)
    ctx_mean_df = mean_df.rolling(ctx_periods, min_periods=1).mean()
    diff_ctx_df = mean_df - ctx_mean_df

    long_periods = max(1, long_ctx_sec // win_sec)
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

    feat_df = pd.concat(
        [
            mean_df.add_suffix("_mean"),
            std_df.add_suffix("_std"),
            ptp_df.add_suffix("_ptp"),
            delta_df.add_suffix("_delta"),
            ibi_diff_df,
            ctx_mean_df.add_suffix("_ctx40_mean"),
            diff_ctx_df.add_suffix("_diff_ctx40"),
            eda_swing_df,
            hr_eda_diff_df,
        ],
        axis=1,
    )
    return feat_df.index, feat_df.to_numpy(dtype=np.float32)


class TechArenaModel(ADM_Base):
    def __init__(self, onnx_file: pathlib.Path) -> None:
        self.load_model(onnx_file)
        self.input_name = self.session.get_inputs()[0].name

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        if input.empty:
            return pd.DataFrame({"Activity": pd.Series(dtype=str)}, index=input.index)

        win_sec = 10
        win_idx, feats = _compute_window_features(input, win_sec=win_sec, ctx_sec=40, long_ctx_sec=60)

        ort_outs = self.session.run(None, {self.input_name: feats})
        if len(ort_outs) > 1:
            raw_probs = ort_outs[1]
            if isinstance(raw_probs, list) and len(raw_probs) > 0 and isinstance(raw_probs[0], dict):
                probs = np.zeros((len(raw_probs), len(LABELS)), dtype=np.float32)
                for row_idx, prob_dict in enumerate(raw_probs):
                    for cls_key, prob_val in prob_dict.items():
                        cls_i = int(cls_key)
                        if 0 <= cls_i < len(LABELS):
                            probs[row_idx, cls_i] = float(prob_val)
            else:
                probs = np.asarray(raw_probs, dtype=np.float32)

            if probs.ndim == 2 and probs.shape[1] == len(LABELS):
                class_weights = np.array([0.9, 1.35, 0.9], dtype=np.float32)
                pred_ids = np.argmax(probs * class_weights, axis=-1)
            else:
                pred_ids = np.asarray(ort_outs[0]).reshape(-1).astype(int)
        else:
            first_out = np.asarray(ort_outs[0])
            if first_out.ndim >= 2 and first_out.shape[-1] == len(LABELS):
                pred_ids = np.argmax(first_out, axis=-1)
            else:
                pred_ids = first_out.reshape(-1).astype(int)

        win_activities = [LABELS[int(np.clip(i, 0, len(LABELS) - 1))] for i in pred_ids]

        # Smooth across 3 consecutive 10-second windows (30s local vote)
        smoothed = []
        n_w = len(win_activities)
        for idx in range(n_w):
            left = max(0, idx - 1)
            right = min(n_w, idx + 2)
            chunk = win_activities[left:right]
            counts = {lbl: chunk.count(lbl) for lbl in set(chunk)}
            best_lbl = max(
                counts.keys(),
                key=lambda lbl: (counts[lbl], 1 if lbl == win_activities[idx] else 0),
            )
            smoothed.append(best_lbl)

        win_series = pd.Series(smoothed, index=win_idx)

        dt_index = pd.to_datetime(input.index)
        floor_idx = dt_index.floor(f"{win_sec}s")
        mapped = win_series.reindex(floor_idx).fillna("Rest").to_numpy()

        return pd.DataFrame({"Activity": mapped}, index=input.index)


def get_model() -> ADM_Base:
    onnx_path = pathlib.Path(__file__).with_suffix(".onnx")
    return TechArenaModel(onnx_path)