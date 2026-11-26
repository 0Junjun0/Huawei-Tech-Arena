import os
import pathlib
import sys
import warnings
import numpy as np
import pandas as pd

# Enforce single-threaded CPU execution for reproducible benchmarks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

warnings.filterwarnings("ignore")

# Define path anchors relative to repository layout
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026" / "data"
if not DATA_DIR.exists():
    DATA_DIR = ROOT_DIR / "data"

SUBMISSION_DIR = ROOT_DIR / "submission"
NOTES_FILE = ROOT_DIR / "notes" / "experiment_notes.md"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
try:
    from log_results import log_safe
except ImportError:
    def log_safe(*args, **kwargs):
        pass

SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
LABELS = ["Rest", "Stress", "Activity"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

STRESS_PHASES = {"stroop", "tmct", "real opinion", "opposite opinion", "subtraction"}
ACTIVITY_PHASES = {"warm-up", "sprint", "cooldown"}
REST_PHASES = {"baseline", "rest"}


def map_phase_to_activity(phase_val) -> str | None:
    """Map raw phase strings to target activities using set rules and 'rpm' substring matching."""
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
    """
    Deduplicated 42-feature extraction pipeline matching submission_model.py exactly.
    """
    work = df.reindex(columns=SENSOR_COLS).copy()

    if isinstance(work.index, pd.MultiIndex):
        work = work.reset_index(drop=True)

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
    ptp_df = (max_df - min_df).fillna(0.0)
    delta_df = mean_df.diff().fillna(0.0)

    ibi_diff_series = (
        work["IBI_duration"].diff().abs().resample(rule).mean().fillna(0.0)
    )
    ibi_diff_df = pd.DataFrame(
        {"IBI_duration_abs_diff": ibi_diff_series}, index=mean_df.index
    )

    ctx_periods = max(1, ctx_sec // win_sec)
    ctx_mean_df = mean_df.rolling(ctx_periods, min_periods=1).mean().fillna(0.0)
    diff_ctx_df = (mean_df - ctx_mean_df).fillna(0.0)

    long_periods = max(1, long_ctx_sec // win_sec)
    eda_long_swing = (
        max_df["EDA"].rolling(long_periods, min_periods=1).max()
        - min_df["EDA"].rolling(long_periods, min_periods=1).min()
    ).fillna(0.0)
    eda_swing_df = pd.DataFrame(
        {"EDA_ctx60_swing": eda_long_swing}, index=mean_df.index
    )

    ibi_dt_diff_series = (
        work["IBI_dt"].diff().abs().resample(rule).mean().fillna(0.0)
    )
    extra_phys_df = pd.DataFrame(
        {
            "HR_std_ctx40": mean_df["HR"]
            .rolling(ctx_periods, min_periods=1)
            .std()
            .fillna(0.0),
            "EDA_std_ctx40": mean_df["EDA"]
            .rolling(ctx_periods, min_periods=1)
            .std()
            .fillna(0.0),
            "IBI_dt_abs_diff": ibi_dt_diff_series,
            "Temp_diff_ctx40": (
                mean_df["Temperature"] - ctx_mean_df["Temperature"]
            ).fillna(0.0),
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
            extra_phys_df,
        ],
        axis=1,
    )

    return feat_df.index, feat_df.to_numpy(dtype=np.float32)


def build_dataset(
    data_dir: pathlib.Path, win_sec: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Traverse protocol directories, extract features, and map window modes to target activities.
    """
    # Exclude virtual environments (.venv) and hidden directories when discovering dataset files
    parquet_files = [
        p
        for p in sorted(data_dir.rglob("*.parquet"))
        if ".venv" not in p.parts and not p.name.startswith(".")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No valid .parquet files found in {data_dir}")

    X_list, y_list, groups_list = [], [], []

    for parquet_path in parquet_files:
        subject_id = parquet_path.stem
        df = pd.read_parquet(parquet_path)

        if "Phase" not in df.columns:
            continue

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(drop=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        win_idx, X_win = extract_window_features(df, win_sec=win_sec)

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

        y_win = np.array(
            [LABEL_TO_IDX[lbl] for lbl in bin_labels.iloc[keep]], dtype=np.int64
        )
        X_list.append(X_win[keep])
        y_list.append(y_win)
        groups_list.append(np.full(len(y_win), subject_id))

    if not X_list:
        raise ValueError(
            f"Failed to extract samples from {data_dir}. Check if parquet files contain valid Phase entries."
        )

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)


def search_class_weights(
    oof_probs: np.ndarray, y_true: np.ndarray
) -> tuple[list[float], float]:
    """Search multiplier grid for out-of-fold probability scaling to maximize Macro-F1."""
    best_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    best_f1 = f1_score(y_true, np.argmax(oof_probs, axis=1), average="macro", zero_division=0)

    for w_stress in np.arange(1.00, 1.95, 0.05):
        for w_rest in [0.90, 0.95, 1.00, 1.05]:
            for w_act in [0.80, 0.85, 0.90, 0.95, 1.00]:
                weights = np.array([w_rest, w_stress, w_act], dtype=np.float32)
                preds = np.argmax(oof_probs * weights, axis=1)
                macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_weights = weights

    return [float(w) for w in best_weights], float(best_f1)


def train_and_export() -> None:
    """Execute subject-grouped CV training, optimize class weights, and export ONNX model."""
    print("=== Training Pipeline: 42-Feature Random Forest ===")
    X, y, groups = build_dataset(DATA_DIR, win_sec=10)
    n_samples, n_features = X.shape
    unique_subjects = len(np.unique(groups))
    print(
        f"Features: {n_features} | Samples: {n_samples} | Subjects: {unique_subjects}"
    )

    n_splits = min(5, unique_subjects)
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((n_samples, len(LABELS)), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model_fold = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
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
        ])

        model_fold.fit(X[train_idx], y[train_idx])
        raw_probs = model_fold.predict_proba(X[val_idx])
        mapped_probs = np.zeros((len(val_idx), 3), dtype=np.float32)
        rf_clf = model_fold.named_steps["rf"]
        for col_idx, cls_val in enumerate(rf_clf.classes_):
            mapped_probs[:, int(cls_val)] = raw_probs[:, col_idx]

        oof_probs[val_idx] = mapped_probs

    # 1. Raw OOF Performance
    raw_oof_preds = np.argmax(oof_probs, axis=1)
    raw_macro_f1 = f1_score(y, raw_oof_preds, average="macro", zero_division=0)
    raw_quality_score = raw_macro_f1 * 80.0

    print("\n--- Raw Out-Of-Fold Classification Report ---")
    print(
        classification_report(
            y, raw_oof_preds, target_names=LABELS, digits=4, zero_division=0
        )
    )
    print(
        f"Raw OOF Macro-F1: {raw_macro_f1:.4f} (Quality Score:"
        f" {raw_quality_score:.2f} / 80)"
    )

    # 2. Probability Weight Tuning
    best_weights, tuned_macro_f1 = search_class_weights(oof_probs, y)
    tuned_quality_score = tuned_macro_f1 * 80.0
    print(
        f"\nTuned OOF Macro-F1: {tuned_macro_f1:.4f} (Quality Score:"
        f" {tuned_quality_score:.2f} / 80)"
    )
    print(
        "Optimized Class Multipliers [Rest, Stress, Activity]:"
        f" {best_weights}"
    )

    # 3. Fit Final Production Model on Full Dataset
    print("\nFitting final model on full dataset...")
    final_pipeline = Pipeline([
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
    ])
    final_pipeline.fit(X, y)

    # 4. Export ONNX Model to submission/ directory and repository root
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path_sub = SUBMISSION_DIR / "submission_model.onnx"
    onnx_path_root = ROOT_DIR / "submission_model.onnx"

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    rf_step = final_pipeline.named_steps["rf"]
    onnx_model = convert_sklearn(
        final_pipeline,
        initial_types=initial_type,
        options={id(rf_step): {"zipmap": False}},
        target_opset=15,
    )

    model_bytes = onnx_model.SerializeToString()
    onnx_path_sub.write_bytes(model_bytes)
    onnx_path_root.write_bytes(model_bytes)

    onnx_size_mb = len(model_bytes) / (1024 * 1024)
    print(
        f"ONNX Model successfully exported to: {onnx_path_sub}"
        f" ({onnx_size_mb:.2f} MB)"
    )

    # 5. Log Run to experiment_notes.md
    summary_text = "\n".join([
        f"Features: {n_features} | Samples: {n_samples} | Subjects: {unique_subjects}",
        f"Raw OOF Macro-F1: {raw_macro_f1:.4f} ({raw_quality_score:.2f}/80)",
        f"Tuned OOF Macro-F1: {tuned_macro_f1:.4f} ({tuned_quality_score:.2f}/80) | Weights: {best_weights}",
        f"ONNX size: {onnx_size_mb:.2f} MB",
    ])
    log_safe(
        run_name="RandomForest (42 features)",
        summary_text=summary_text,
        notes_file=NOTES_FILE,
    )


if __name__ == "__main__":
    train_and_export()
