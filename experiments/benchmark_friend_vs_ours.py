import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import importlib.util
import pathlib
import sys
import time
import tracemalloc
import numpy as np
import onnxruntime as ort
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

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
SUBMISSION_DIR = ROOT_DIR / "submission"
FRIEND_DIR = ROOT_DIR / "experiments" / "friend_benchmark"

for p in (ROOT_DIR, ROOT_DIR / "src", SUBMISSION_DIR, DATA_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

VALID_LABELS = {"Rest", "Stress", "Activity"}
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def load_module_from_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_friend_dataset_with_protocol(friend_tm, friend_sm, data_root: pathlib.Path):
    """Uses friend's exact build_dataset logic while also tracking protocol labels."""
    X_list, y_list, groups_list, purity_list, proto_list, file_slices = [], [], [], [], [], []
    data_dir = data_root / "data"
    offset = 0

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        subject_id = parquet_path.stem
        protocol = parquet_path.parent.name if parquet_path.parent != data_dir else "UNKNOWN"
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        for col in friend_sm.SENSOR_COLS:
            if col not in df.columns:
                df[col] = np.nan

        if "Phase" not in df.columns:
            continue

        _, bin_ids, X_win = friend_sm.compute_window_features(df, win_sec=10)
        n_bins = len(X_win)

        mapped_phase = df["Phase"].map(friend_tm.map_phase_to_activity)
        valid_mask = mapped_phase.notna().to_numpy()
        if not np.any(valid_mask):
            continue

        v_bins = bin_ids[valid_mask]
        v_labels = mapped_phase.iloc[valid_mask].map(friend_tm.LABEL_TO_IDX).to_numpy(dtype=np.int64)

        counts = np.zeros((n_bins, 3), dtype=np.int64)
        np.add.at(counts, (v_bins, v_labels), 1)
        totals = counts.sum(axis=1)
        has_label = totals > 0
        if not np.any(has_label):
            continue

        valid_counts = counts[has_label]
        y_win = np.argmax(valid_counts, axis=1)
        purity = np.max(valid_counts, axis=1) / totals[has_label]

        X_list.append(X_win[has_label])
        y_list.append(y_win)
        groups_list.append(np.full(len(y_win), subject_id, dtype=object))
        purity_list.append(purity)
        proto_list.append(np.full(len(y_win), protocol, dtype=object))

        file_slices.append(slice(offset, offset + len(y_win)))
        offset += len(y_win)

    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list),
        np.concatenate(purity_list),
        np.concatenate(proto_list),
        file_slices,
    )


def report_oof_breakdown(title: str, y: np.ndarray, preds: np.ndarray, groups: np.ndarray, protocols: np.ndarray, labels: list[str]) -> float:
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y, preds))
    qual_score = macro_f1 * 80.0

    print(f"\n==================== {title} ====================")
    print(f"Macro-F1:          {macro_f1:.4f} (Est. Quality: {qual_score:.2f} / 80.00)")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred) [Rest, Stress, Activity]:")
    print(confusion_matrix(y, preds, labels=[0, 1, 2]))
    print("\nPer-Class Report:")
    print(classification_report(y, preds, target_names=labels, digits=4, zero_division=0))

    print("=== By Protocol ===")
    for proto in sorted(np.unique(protocols)):
        mask = protocols == proto
        p_f1 = f1_score(y[mask], preds[mask], average="macro", zero_division=0)
        p_counts = {lbl: int(np.sum(y[mask] == i)) for i, lbl in enumerate(labels)}
        print(f"{proto:12s} | n={int(np.sum(mask)):4d} | Macro-F1={p_f1:.4f} | counts={p_counts}")

    print("\n=== Lowest-Scoring Subjects (Bottom 10 by Macro-F1) ===")
    subj_rows = []
    for subj in np.unique(groups):
        mask = groups == subj
        s_f1 = f1_score(y[mask], preds[mask], average="macro", zero_division=0)
        subj_rows.append((subj, int(np.sum(mask)), s_f1))
    subj_rows.sort(key=lambda r: r[2])
    for subj, n_w, s_f1 in subj_rows[:10]:
        print(f"Subject {subj:6s} | windows={n_w:4d} | Macro-F1={s_f1:.4f}")
    return macro_f1


def validate_output_df(out: pd.DataFrame, input_df: pd.DataFrame) -> None:
    assert isinstance(out, pd.DataFrame), "run() must return pd.DataFrame"
    assert list(out.columns) == ["Activity"], f"Expected ['Activity'], got {list(out.columns)}"
    assert len(out) == len(input_df), f"Expected {len(input_df)} rows, got {len(out)}"
    assert not out["Activity"].isna().any(), "Output contains NaNs"
    assert set(out["Activity"].unique()).issubset(VALID_LABELS), f"Invalid labels: {set(out['Activity'].unique()) - VALID_LABELS}"


def run_edge_cases_and_speed(label: str, model, onnx_path: pathlib.Path, sample_dfs: list[pd.DataFrame]) -> dict:
    print(f"\n==================== RUNTIME & ROBUSTNESS: {label} ====================")
    df0 = sample_dfs[0]

    # 1. Determinism
    out_a = model.run(df0)
    out_b = model.run(df0)
    validate_output_df(out_a, df0)
    validate_output_df(out_b, df0)
    pd.testing.assert_frame_equal(out_a, out_b)
    print("Determinism (repeated call): PASS")

    # 2. Reversed columns
    shuffled_cols = [c for c in reversed(SENSOR_COLS) if c in df0.columns] + [
        c for c in df0.columns if c not in SENSOR_COLS
    ]
    out_shuf = model.run(df0[shuffled_cols])
    validate_output_df(out_shuf, df0)
    shuf_match = bool((out_a["Activity"].to_numpy() == out_shuf["Activity"].to_numpy()).all())
    print(f"Reversed column order: PASS (exact_match={shuf_match})")

    # 3. Injected NaN
    df_nan = df0.copy()
    for col in SENSOR_COLS:
        if col in df_nan.columns:
            df_nan.iloc[::7, df_nan.columns.get_loc(col)] = np.nan
    out_nan = model.run(df_nan)
    validate_output_df(out_nan, df_nan)
    print("Injected NaN robustness: PASS")

    # 4. Unsorted index (EXP-OPT-09 check)
    rng = np.random.default_rng(42)
    df_unsorted = df0.iloc[rng.permutation(len(df0))]
    try:
        out_unsorted = model.run(df_unsorted)
        validate_output_df(out_unsorted, df_unsorted)
        sorted_aligned = out_a["Activity"].loc[out_unsorted.index].to_numpy()
        unsort_match = bool((sorted_aligned == out_unsorted["Activity"].to_numpy()).all())
        print(f"Unsorted DatetimeIndex: ran (exact_match_after_realign={unsort_match})")
    except Exception as exc:
        print(f"Unsorted DatetimeIndex: FAIL ({type(exc).__name__}: {exc})")

    # 5. String index (EXP-OPT-09 check)
    df_str = df0.copy()
    df_str.index = df_str.index.astype(str)
    try:
        out_str = model.run(df_str)
        validate_output_df(out_str, df_str)
        str_match = bool((out_a["Activity"].to_numpy() == out_str["Activity"].to_numpy()).all())
        print(f"String index: PASS (exact_match={str_match})")
    except Exception as exc:
        print(f"String index: FAIL ({type(exc).__name__}: {exc})")

    # 6. Missing column
    df_missing = df0.drop(columns=["IBI_dt"])
    try:
        out_miss = model.run(df_missing)
        validate_output_df(out_miss, df_missing)
        print("Missing sensor column (IBI_dt dropped): PASS")
    except Exception as exc:
        print(f"Missing sensor column (IBI_dt dropped): FAIL ({type(exc).__name__}: {exc})")

    # 7. 50-run single-core MET + separate tracemalloc peak RAM (check_42_speed_and_parity style)
    for df in sample_dfs[:3]:
        _ = model.run(df)

    latencies = []
    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]
        t0 = time.perf_counter()
        out = model.run(df)
        latencies.append(time.perf_counter() - t0)
        validate_output_df(out, df)

    tracemalloc.start()
    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]
        _ = model.run(df)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr = np.asarray(latencies, dtype=np.float64)
    met = float(np.mean(arr))
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    std = float(np.std(arr, ddof=1))
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    onnx_mb = onnx_path.stat().st_size / (1024.0 * 1024.0)
    speed_score = max(0.0, min(20.0, (1.0 - met / 0.30) * 20.0))

    print(
        f"50-Run Single-Core MET: {met:.4f} s | Median: {med:.4f} s | P95: {p95:.4f} s | "
        f"Std: {std:.4f} s | Peak RAM: {peak_mb:.2f} MB | ONNX: {onnx_mb:.2f} MB | Speed: {speed_score:.2f}/20"
    )
    return {
        "met": met,
        "median": med,
        "p95": p95,
        "std": std,
        "peak_mb": peak_mb,
        "onnx_mb": onnx_mb,
        "speed_score": speed_score,
    }


def main() -> None:
    # Load baseline modules
    import metrics_audit as ma
    import train_model as base_tm
    import submission_model as base_sm

    # Load friend modules from isolated folder
    # Ensure friend_sm is registered as 'submission_model' temporarily when importing friend_tm
    saved_sm = sys.modules.pop("submission_model", None)
    friend_sm = load_module_from_path("submission_model", FRIEND_DIR / "submission_model.py")
    friend_tm = load_module_from_path("friend_train_model", FRIEND_DIR / "train_model.py")
    if saved_sm is not None:
        sys.modules["submission_model"] = saved_sm

    # ---- PART A: YOUR BASELINE (RF-42) ----
    print(">>> Evaluating YOUR CURRENT BASELINE (RF-42)...")
    X_b, y_b, groups_b, protos_b = ma.build_dataset_with_protocol(DATA_DIR, win_sec=10)
    print(f"Baseline Dataset: X={X_b.shape}, y={y_b.shape}, subjects={len(np.unique(groups_b))}")

    gkf_b = GroupKFold(n_splits=min(5, len(np.unique(groups_b))))
    oof_probs_b = np.zeros((len(y_b), 3), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(gkf_b.split(X_b, y_b, groups_b)):
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=18, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=42 + fold, n_jobs=1,
            )),
        ])
        clf.fit(X_b[tr_idx], y_b[tr_idx])
        probs = clf.predict_proba(X_b[va_idx])
        mapped = np.zeros((len(va_idx), 3), dtype=np.float32)
        for c_i, c_val in enumerate(clf.classes_):
            mapped[:, int(c_val)] = probs[:, c_i]
        oof_probs_b[va_idx] = mapped

    f1_b_raw = report_oof_breakdown(
        "YOUR BASELINE (RF-42, Raw OOF)",
        y_b, np.argmax(oof_probs_b, axis=1), groups_b, protos_b, base_tm.LABELS,
    )
    w_b = np.array([0.95, 1.80, 0.85], dtype=np.float32)
    f1_b_weighted = report_oof_breakdown(
        "YOUR BASELINE (RF-42, Weighted [0.95, 1.80, 0.85])",
        y_b, np.argmax(oof_probs_b * w_b, axis=1), groups_b, protos_b, base_tm.LABELS,
    )

    # ---- PART B: FRIEND'S CODE (77 features) ----
    print("\n>>> Evaluating FRIEND'S CODE (77 features)...")
    X_f, y_f, groups_f, purity_f, protos_f, slices_f = build_friend_dataset_with_protocol(
        friend_tm, friend_sm, DATA_DIR
    )
    print(f"Friend Dataset: X={X_f.shape}, y={y_f.shape}, subjects={len(np.unique(groups_f))}")

    gkf_f = GroupKFold(n_splits=min(5, len(np.unique(groups_f))))
    folds_f = list(gkf_f.split(X_f, y_f, groups_f))

    # Determine which mode actually exported to ONNX
    onnx_f_path = FRIEND_DIR / "submission_model.onnx"
    assert onnx_f_path.exists(), f"Missing {onnx_f_path}; run friend train_model.py first."

    # Evaluate BOTH hgb_vote (what their CV runs) and gb_vote (what actually exported to ONNX)
    for mode_name, mode_label in [
        ("hgb_vote", "FRIEND CV MODE ('hgb_vote': HGB+ET+RF)"),
        ("gb_vote", "FRIEND ACTUAL ONNX MODE ('gb_vote': GB+ET+RF)"),
    ]:
        oof_probs_f = np.zeros((len(y_f), 3), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(folds_f):
            clean_tr = tr_idx[(purity_f[tr_idx] >= 0.75) & (X_f[tr_idx, 6] > 1e-5)]
            if len(clean_tr) < 100:
                clean_tr = tr_idx
            clf = friend_tm.make_ensemble_pipeline(seed=42 + fold, mode=mode_name)
            clf.fit(X_f[clean_tr], y_f[clean_tr])
            oof_probs_f[va_idx] = clf.predict_proba(X_f[va_idx]).astype(np.float32)

        smoothed_f = np.zeros_like(oof_probs_f)
        for sl in slices_f:
            smoothed_f[sl] = friend_sm.smooth_window_probs(oof_probs_f[sl])

        report_oof_breakdown(
            f"{mode_label} — Unsmoothed Raw",
            y_f, np.argmax(oof_probs_f, axis=1), groups_f, protos_f, friend_sm.LABELS,
        )
        report_oof_breakdown(
            f"{mode_label} — Smoothed Unweighted",
            y_f, np.argmax(smoothed_f, axis=1), groups_f, protos_f, friend_sm.LABELS,
        )

        # Nested tuned weights
        nested_preds_f = np.zeros_like(y_f)
        for tr_idx, va_idx in folds_f:
            w_fold = friend_tm.search_best_weights(smoothed_f[tr_idx], y_f[tr_idx])
            nested_preds_f[va_idx] = np.argmax(smoothed_f[va_idx] * w_fold, axis=1)
        report_oof_breakdown(
            f"{mode_label} — Smoothed + Nested Tuned Weights",
            y_f, nested_preds_f, groups_f, protos_f, friend_sm.LABELS,
        )

        # Also evaluate with the actual CLASS_WEIGHTS written into friend's submission_model.py
        w_deployed = np.array(friend_sm.CLASS_WEIGHTS, dtype=np.float32)
        report_oof_breakdown(
            f"{mode_label} — Smoothed + Deployed CLASS_WEIGHTS {list(friend_sm.CLASS_WEIGHTS)}",
            y_f, np.argmax(smoothed_f * w_deployed, axis=1), groups_f, protos_f, friend_sm.LABELS,
        )

    # ---- PART C: RUNTIME, ROBUSTNESS & ONNX PARITY ----
    parquet_files = sorted((DATA_DIR / "data").rglob("*.parquet"))
    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]

    base_model = base_sm.get_model()
    friend_model = friend_sm.TechArenaModel(onnx_f_path)

    base_perf = run_edge_cases_and_speed("YOUR BASELINE (RF-42)", base_model, SUBMISSION_DIR / "submission_model.onnx", sample_dfs)
    friend_perf = run_edge_cases_and_speed("FRIEND'S MODEL (77 features)", friend_model, onnx_f_path, sample_dfs)

    print("\n=== SUMMARY SPEED & SIZE ===")
    print(f"Baseline: Quality(raw/weighted)={f1_b_raw*80:.2f}/{f1_b_weighted*80:.2f} | Speed={base_perf['speed_score']:.2f}/20 | MET={base_perf['met']:.4f}s | ONNX={base_perf['onnx_mb']:.2f}MB | RAM={base_perf['peak_mb']:.2f}MB")
    print(f"Friend  : Speed={friend_perf['speed_score']:.2f}/20 | MET={friend_perf['met']:.4f}s | ONNX={friend_perf['onnx_mb']:.2f}MB | RAM={friend_perf['peak_mb']:.2f}MB")


if __name__ == "__main__":
    main()
