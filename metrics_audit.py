import pathlib
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import submission_model
import train_model


def build_dataset_with_protocol(root_dir: pathlib.Path, win_sec: int = 10):
    X_list, y_list, groups_list, proto_list = [], [], [], []
    data_dir = root_dir / "data"

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        subject_id = parquet_path.stem
        protocol = parquet_path.parent.name if parquet_path.parent != data_dir else "UNKNOWN"
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        for col in train_model.SENSOR_COLS:
            if col not in df.columns:
                df[col] = np.nan

        win_idx, X_win = train_model.extract_window_features(df, win_sec=win_sec)
        if "Phase" not in df.columns:
            continue

        mapped_phase = df["Phase"].map(train_model.map_phase_to_activity)
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
            [train_model.LABEL_TO_IDX[lbl] for lbl in bin_labels.iloc[keep]],
            dtype=np.int64,
        )
        X_list.append(X_win[keep])
        y_list.append(y_win)
        groups_list.append(np.full(len(y_win), subject_id))
        proto_list.append(np.full(len(y_win), protocol))

    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list),
        np.concatenate(proto_list),
    )


def benchmark_submission_model(root_dir: pathlib.Path, n_runs: int = 20):
    onnx_path = pathlib.Path("submission_model.onnx")
    if not onnx_path.exists():
        print("\n[Speed Check] Skipped: submission_model.onnx not found.")
        return

    sample_files = sorted((root_dir / "data").rglob("*.parquet"))[:3]
    if not sample_files:
        print("\n[Speed Check] Skipped: no sample parquet files found.")
        return

    dfs = [pd.read_parquet(p)[train_model.SENSOR_COLS] for p in sample_files]
    model = submission_model.get_model()

    # Warmup
    _ = model.run(dfs[0])

    times = []
    for i in range(n_runs):
        df_in = dfs[i % len(dfs)]
        t0 = time.perf_counter()
        _ = model.run(df_in)
        times.append(time.perf_counter() - t0)

    met = float(np.mean(times))
    speed_score = max(0.0, min(20.0, (1.0 - met / 0.3) * 20.0))
    print("\n=== Submission Model Runtime Check ===")
    print(
        f"Runs: {n_runs} | MET: {met:.4f} s | Min: {min(times):.4f} s | "
        f"Max: {max(times):.4f} s | Est. Speed Score: {speed_score:.2f} / 20.00"
    )


def main():
    root_dir = pathlib.Path("TechArena-Finland-2026")
    X, y, groups, protocols = build_dataset_with_protocol(root_dir, win_sec=10)
    unique_subjects = np.unique(groups)

    print(f"Dataset: X={X.shape}, y={y.shape}, subjects={len(unique_subjects)}")
    class_counts = {lbl: int(np.sum(y == i)) for i, lbl in enumerate(train_model.LABELS)}
    print(f"Class distribution: {class_counts}")

    n_splits = min(5, len(unique_subjects))
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y), dtype=int)

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
        oof_preds[va_idx] = clf.predict(X[va_idx])

    macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y, oof_preds)
    qual_score = macro_f1 * 80.0

    print("\n=== Overall OOF Metrics ===")
    print(f"Macro-F1:          {macro_f1:.4f} (Est. Quality: {qual_score:.2f} / 80.00)")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred) [Rest, Stress, Activity]:")
    print(confusion_matrix(y, oof_preds, labels=[0, 1, 2]))
    print("\nPer-Class Report:")
    print(classification_report(y, oof_preds, target_names=train_model.LABELS, digits=4, zero_division=0))

    print("=== By Protocol ===")
    for proto in sorted(np.unique(protocols)):
        mask = protocols == proto
        p_f1 = f1_score(y[mask], oof_preds[mask], average="macro", zero_division=0)
        p_counts = {lbl: int(np.sum(y[mask] == i)) for i, lbl in enumerate(train_model.LABELS)}
        print(f"{proto:12s} | n={int(np.sum(mask)):4d} | Macro-F1={p_f1:.4f} | counts={p_counts}")

    print("\n=== Lowest-Scoring Subjects (Bottom 10 by Macro-F1) ===")
    subj_rows = []
    for subj in unique_subjects:
        mask = groups == subj
        s_f1 = f1_score(y[mask], oof_preds[mask], average="macro", zero_division=0)
        subj_rows.append((subj, int(np.sum(mask)), s_f1))
    subj_rows.sort(key=lambda r: r[2])
    for subj, n_w, s_f1 in subj_rows[:10]:
        print(f"Subject {subj:6s} | windows={n_w:4d} | Macro-F1={s_f1:.4f}")

    benchmark_submission_model(root_dir, n_runs=20)


if __name__ == "__main__":
    main()