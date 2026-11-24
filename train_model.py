import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

LABELS = ["Rest", "Stress", "Activity"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

STRESS_PHASES = {"stroop", "tmct", "real opinion", "opposite opinion", "subtraction"}
ACTIVITY_PHASES = {"warm-up", "sprint", "cooldown"}
REST_PHASES = {"baseline", "rest"}

SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def map_phase_to_activity(phase_val) -> str | None:
    if pd.isna(phase_val):
        return None
    low = str(phase_val).strip().lower()
    if low in STRESS_PHASES:
        return "Stress"
    if low in REST_PHASES:
        return "Rest"
    if low in ACTIVITY_PHASES or "rpm" in low:
        return "Activity"
    return None


def extract_window_features(
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
            diff_ctx_df.add_suffix("_diff_ctx40"),
            eda_swing_df,
            hr_eda_diff_df,
            extra_phys_df,
        ],
        axis=1,
    )
    return feat_df.index, feat_df.to_numpy(dtype=np.float32)


def build_dataset(root_dir: pathlib.Path, win_sec: int = 10):
    X_list, y_list, groups_list = [], [], []
    data_dir = root_dir / "data"

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        subject_id = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        for col in SENSOR_COLS:
            if col not in df.columns:
                df[col] = np.nan

        win_idx, X_win = extract_window_features(df, win_sec=win_sec)

        if "Phase" not in df.columns:
            continue

        mapped_phase = df["Phase"].map(map_phase_to_activity)
        valid_mask = mapped_phase.notna()
        if not valid_mask.any():
            continue

        bin_labels = pd.Series(index=win_idx, dtype=object)
        grouped = mapped_phase[valid_mask].groupby(pd.Grouper(freq=f"{win_sec}s"))
        for ts, series in grouped:
            if ts in bin_labels.index and len(series) > 0:
                mode_vals = series.mode()
                if len(mode_vals) > 0:
                    bin_labels.loc[ts] = mode_vals.iloc[0]

        keep = bin_labels.notna().to_numpy()
        if not np.any(keep):
            continue

        y_win = np.array([LABEL_TO_IDX[lbl] for lbl in bin_labels.iloc[keep]], dtype=np.int64)
        X_list.append(X_win[keep])
        y_list.append(y_win)
        groups_list.append(np.full(len(y_win), subject_id))

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)


def main():
    root_dir = pathlib.Path("TechArena-Finland-2026")
    X, y, groups = build_dataset(root_dir, win_sec=10)
    print(f"Dataset shape: X={X.shape}, y={y.shape}, unique_subjects={len(np.unique(groups))}")

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), 3), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=18,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42 + fold,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        clf.fit(X[tr_idx], y[tr_idx])

        raw_probs = clf.predict_proba(X[va_idx])
        mapped_probs = np.zeros((len(va_idx), 3), dtype=np.float32)
        for col_idx, cls_val in enumerate(clf.classes_):
            mapped_probs[:, int(cls_val)] = raw_probs[:, col_idx]

        oof_probs[va_idx] = mapped_probs
        oof_preds[va_idx] = np.argmax(mapped_probs, axis=1)

    macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    print(f"OOF Macro-F1: {macro_f1:.4f} | Estimated Challenge Score: {macro_f1 * 80:.2f} / 80")
    print("\nPer-class breakdown:")
    print(classification_report(y, oof_preds, target_names=LABELS, digits=4, zero_division=0))

    best_f1 = macro_f1
    best_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for w_stress in np.arange(1.00, 1.85, 0.10):
        for w_rest in [0.95, 1.00, 1.05]:
            for w_act in [0.85, 0.90, 0.95, 1.00]:
                weights = np.array([w_rest, w_stress, w_act], dtype=np.float32)
                cand_preds = np.argmax(oof_probs * weights, axis=1)
                cand_f1 = f1_score(y, cand_preds, average="macro", zero_division=0)
                if cand_f1 > best_f1:
                    best_f1 = cand_f1
                    best_weights = weights

    print(
        f"Tuned OOF Macro-F1: {best_f1:.4f} "
        f"(Est. {best_f1 * 80:.2f} / 80) | "
        f"Best weights [Rest, Stress, Activity]: {[round(float(w), 4) for w in best_weights]}"
    )

    final_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=18,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_model.fit(X, y)

    rf_step = final_model.named_steps["rf"]
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        final_model,
        initial_types=initial_type,
        options={id(rf_step): {"zipmap": False}},
        target_opset=15,
    )
    out_onnx = pathlib.Path("submission_model.onnx")
    out_onnx.write_bytes(onnx_model.SerializeToString())
    print(f"Saved updated ONNX model to {out_onnx} ({out_onnx.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()