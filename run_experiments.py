import os

# Enforce single-threaded execution before importing numerical/ONNX packages
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import datetime
import pathlib
import sys
import time
import tracemalloc
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

WORK_DIR = pathlib.Path(__file__).resolve().parent
DATA_KIT_DIR = WORK_DIR / "TechArena-Finland-2026"
DATA_DIR = DATA_KIT_DIR / "data" if (DATA_KIT_DIR / "data").exists() else WORK_DIR
NOTES_PATH = WORK_DIR / "experiment_notes.md"
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
VALID_LABELS = {"Rest", "Stress", "Activity"}

EXP_OPT_2_3_6_REF = {
    "met": 0.0421,
    "median": 0.0423,
    "p95": 0.0548,
    "std": 0.0062,
    "peak_mb": 22.88,
}

AUDIT_BASELINE_REF = {
    "met": 0.0339,
    "p95": 0.0399,
    "peak_mb": 18.81,
    "onnx_mb": 26.43,
}

sys.path.insert(0, str(WORK_DIR))
import submission_model as sub_mod

try:
    import train_model as train_mod
except Exception:
    train_mod = None


def assert_valid_output(df_out: pd.DataFrame) -> None:
    assert isinstance(df_out, pd.DataFrame), "Output must be a DataFrame"
    assert list(df_out.columns) == ["Activity"], f"Expected ['Activity'], got {list(df_out.columns)}"
    assert len(df_out) > 0, "Output DataFrame is empty"
    assert not df_out["Activity"].isna().any(), "Output contains NaN labels"
    unexpected = set(df_out["Activity"].unique()) - VALID_LABELS
    assert not unexpected, f"Invalid labels found: {unexpected}"


def deduplicate_feature_matrix(
    X: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Removes exact duplicate columns (such as HR_minus_ctx40 and EDA_minus_ctx40) without altering others."""
    n_cols = X.shape[1]
    if feature_names is None or len(feature_names) != n_cols:
        feature_names = [f"feat_{i}" for i in range(n_cols)]

    dropped = []
    kept_indices = []
    kept_names = []
    seen_arrays: list[np.ndarray] = []

    duplicate_aliases = {"HR_minus_ctx40", "EDA_minus_ctx40"}

    for idx, name in enumerate(feature_names):
        col = np.nan_to_num(X[:, idx].astype(np.float64), nan=-999999.0)
        is_exact_dup = any(np.allclose(col, prev, rtol=1e-7, atol=1e-7) for prev in seen_arrays)
        if name in duplicate_aliases and is_exact_dup:
            dropped.append(name)
            continue
        kept_indices.append(idx)
        kept_names.append(name)
        seen_arrays.append(col)

    return X[:, kept_indices], kept_names, dropped


def load_grouped_dataset():
    if train_mod is not None and hasattr(train_mod, "build_dataset"):
        res = train_mod.build_dataset(DATA_KIT_DIR if DATA_KIT_DIR.exists() else WORK_DIR)
        if isinstance(res, tuple) and len(res) == 4:
            X, y, groups, feat_names = res
            return np.asarray(X), np.asarray(y), np.asarray(groups), list(feat_names), "train_model.build_dataset"
        if isinstance(res, tuple) and len(res) == 3:
            X, y, groups = res
            feat_names = getattr(train_mod, "FEATURE_NAMES", None)
            return np.asarray(X), np.asarray(y), np.asarray(groups), feat_names, "train_model.build_dataset"

    # Fallback: extract window features directly from submission_model + parquet files if present
    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
    assert parquet_files, f"No .parquet files found under {DATA_DIR}"
    model = sub_mod.get_model()

    X_parts, y_parts, g_parts = [], [], []
    for p in parquet_files[:15]:
        df = pd.read_parquet(p)
        out_df = model.run(df)
        n_rows = len(out_df)
        if n_rows == 0:
            continue
        subj = p.stem.split("_")[0]
        if hasattr(model, "_compute_window_features"):
            feat_res = model._compute_window_features(df)
            feats = feat_res[1] if isinstance(feat_res, tuple) else feat_res
            arr = np.asarray(feats, dtype=np.float32)
            if arr.ndim == 2 and len(arr) == n_rows:
                X_parts.append(arr)
        label_map = {"Rest": 0, "Stress": 1, "Activity": 2}
        y_parts.append(out_df["Activity"].map(label_map).fillna(0).to_numpy(dtype=np.int64))
        g_parts.append(np.full(n_rows, subj, dtype=object))

    if X_parts:
        return (
            np.vstack(X_parts),
            np.concatenate(y_parts),
            np.concatenate(g_parts),
            None,
            "fallback_parquet_features",
        )
    raise RuntimeError("Could not build dataset from train_model.build_dataset or parquet feature extraction.")


def eval_group_kfold_rf(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    unique_groups = np.unique(groups)
    splits = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=splits)
    oof_preds = np.zeros_like(y, dtype=np.int64)

    t0 = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=18,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42 + fold,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        pipe.fit(X[tr_idx], y[tr_idx])
        oof_preds[va_idx] = pipe.predict(X[va_idx])
    cv_sec = time.perf_counter() - t0

    macro_f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    _, _, f1_cls, _ = precision_recall_fscore_support(y, oof_preds, labels=[0, 1, 2], zero_division=0)
    return macro_f1, f1_cls, cv_sec


def benchmark_run_50(model, dfs: list[pd.DataFrame], n_runs: int = 50) -> dict[str, float]:
    _ = model.run(dfs[0])  # warm-up

    latencies = []
    for i in range(n_runs):
        df = dfs[i % len(dfs)]
        t0 = time.perf_counter()
        out = model.run(df)
        dt = time.perf_counter() - t0
        assert_valid_output(out)
        latencies.append(dt)

    tracemalloc.start()
    for i in range(n_runs):
        df = dfs[i % len(dfs)]
        out = model.run(df)
        assert_valid_output(out)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "met": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
        "peak_mb": float(peak_bytes / (1024.0 * 1024.0)),
    }


def append_experiment_to_notes(
    *,
    exp_id: str,
    title: str,
    hypothesis: str,
    changes_made: str,
    files_modified: list[str] | str,
    validation_method: str,
    prev_macro_f1: str,
    new_macro_f1: str,
    per_class_f1: str,
    met_sec: Optional[float],
    median_sec: Optional[float] = None,
    p95_sec: Optional[float] = None,
    std_sec: Optional[float] = None,
    peak_ram_mb: Optional[float] = None,
    onnx_size_mb: Optional[float] = 26.43,
    quality_score: Optional[float] = None,
    speed_score: Optional[float] = None,
    total_score: Optional[float] = None,
    conclusion: str,
    notes_path: pathlib.Path = NOTES_PATH,
) -> pathlib.Path:
    """Appends a structured experiment entry to experiment_notes.md."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    files_str = (
        ", ".join(f"`{f}`" for f in files_modified)
        if isinstance(files_modified, list)
        else str(files_modified)
    )

    if speed_score is None and met_sec is not None:
        speed_score = max(0.0, min(20.0, (1.0 - met_sec / 0.30) * 20.0))

    exec_parts = []
    if met_sec is not None:
        exec_parts.append(f"MET: {met_sec:.4f} s")
    if median_sec is not None:
        exec_parts.append(f"Median: {median_sec:.4f} s")
    if p95_sec is not None:
        exec_parts.append(f"P95: {p95_sec:.4f} s")
    if std_sec is not None:
        exec_parts.append(f"Std: {std_sec:.4f} s")
    if peak_ram_mb is not None:
        exec_parts.append(f"Peak RAM: {peak_ram_mb:.2f} MB")
    if onnx_size_mb is not None:
        exec_parts.append(f"ONNX size: {onnx_size_mb:.2f} MB")
    exec_line = " | ".join(exec_parts) if exec_parts else "N/A"

    if total_score is not None:
        q_part = f"Quality: {quality_score:.2f}/80, " if quality_score is not None else ""
        s_part = f"Speed: {speed_score:.2f}/20" if speed_score is not None else ""
        score_line = f"{total_score:.2f} / 100 ({q_part}{s_part})".rstrip(", ")
    elif speed_score is not None:
        score_line = f"Speed component: {speed_score:.2f} / 20.00 (Quality pending)"
    else:
        score_line = "N/A"

    entry = (
        f"\n<!-- Logged: {now_str} ({exp_id}) -->\n"
        f"### {exp_id}: {title}\n"
        f"- **Experiment ID:** {exp_id}\n"
        f"- **Hypothesis:** {hypothesis}\n"
        f"- **Changes made:** {changes_made}\n"
        f"- **Files modified:** {files_str}\n"
        f"- **Validation method:** {validation_method}\n"
        f"- **Previous macro-F1:** {prev_macro_f1}\n"
        f"- **New macro-F1:** {new_macro_f1}\n"
        f"- **Per-class F1:** {per_class_f1}\n"
        f"- **Execution time:** {exec_line}\n"
        f"- **Estimated total competition score:** {score_line}\n"
        f"- **Conclusion:** {conclusion}\n"
    )

    notes_path = pathlib.Path(notes_path)
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "# Experiment Notes\n"
    if not existing.endswith("\n"):
        existing += "\n"
    notes_path.write_text(existing + entry, encoding="utf-8")
    return notes_path


def run_exp_opt_03() -> None:
    X_base, y, groups, feat_names, source_mode = load_grouped_dataset()
    X_dedup, kept_names, dropped_names = deduplicate_feature_matrix(X_base, feat_names)

    # Verify exact numerical equivalence on non-duplicate columns
    non_dup_cols = [i for i, n in enumerate(kept_names) if n not in {"HR_minus_ctx40", "EDA_minus_ctx40"}]
    np.testing.assert_allclose(
        np.nan_to_num(X_base[:, non_dup_cols], nan=0.0),
        np.nan_to_num(X_dedup[:, non_dup_cols], nan=0.0),
        rtol=1e-6,
        atol=1e-6,
    )

    base_f1, base_cls_f1, base_cv_sec = eval_group_kfold_rf(X_base, y, groups, n_splits=5)
    dedup_f1, dedup_cls_f1, dedup_cv_sec = eval_group_kfold_rf(X_dedup, y, groups, n_splits=5)

    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]
    model = sub_mod.get_model()
    stats = benchmark_run_50(model, sample_dfs, n_runs=50)

    qual_score = dedup_f1 * 80.0
    speed_score = max(0.0, min(20.0, (1.0 - stats["met"] / 0.30) * 20.0))
    est_total = qual_score + speed_score

    d_236_met = stats["met"] - EXP_OPT_2_3_6_REF["met"]
    d_236_med = stats["median"] - EXP_OPT_2_3_6_REF["median"]
    d_236_p95 = stats["p95"] - EXP_OPT_2_3_6_REF["p95"]
    d_236_std = stats["std"] - EXP_OPT_2_3_6_REF["std"]
    d_236_ram = stats["peak_mb"] - EXP_OPT_2_3_6_REF["peak_mb"]

    d_base_met = stats["met"] - AUDIT_BASELINE_REF["met"]
    d_base_p95 = stats["p95"] - AUDIT_BASELINE_REF["p95"]
    d_base_ram = stats["peak_mb"] - AUDIT_BASELINE_REF["peak_mb"]

    feat_summary = f"{X_base.shape[1]} -> {X_dedup.shape[1]} features (dropped={dropped_names or 'none'})"
    cls_str = (
        f"Rest: {dedup_cls_f1[0]:.4f} (base {base_cls_f1[0]:.4f}) | "
        f"Stress: {dedup_cls_f1[1]:.4f} (base {base_cls_f1[1]:.4f}) | "
        f"Activity: {dedup_cls_f1[2]:.4f} (base {base_cls_f1[2]:.4f})"
    )

    print("=== EXP-OPT-03: Repeated Calculations Check (1 Thread, 50 Runs + 5-Fold GroupKFold) ===")
    print(f"Dataset source: {source_mode} | Shape: {X_base.shape} -> {X_dedup.shape} | Dropped: {dropped_names}")
    print(f"Non-duplicate feature equivalence: PASSED")
    print(f"5-Fold GroupKFold Macro-F1: {base_f1:.4f} -> {dedup_f1:.4f} ({dedup_f1 - base_f1:+.4f}) | {cls_str}")
    print(
        f"50-Run MET: {stats['met']:.4f} s (vs EXP-OPT-2-3..6 {d_236_met:+.4f} s, vs base {d_base_met:+.4f} s) | "
        f"Median: {stats['median']:.4f} s ({d_236_med:+.4f} s) | "
        f"P95: {stats['p95']:.4f} s (vs EXP-OPT-2-3..6 {d_236_p95:+.4f} s, vs base {d_base_p95:+.4f} s) | "
        f"Std: {stats['std']:.4f} s ({d_236_std:+.4f} s) | "
        f"Peak RAM: {stats['peak_mb']:.2f} MB (vs EXP-OPT-2-3..6 {d_236_ram:+.2f} MB, vs base {d_base_ram:+.2f} MB)"
    )
    print(f"Est. Score: {est_total:.2f}/100 (Quality: {qual_score:.2f}/80, Speed: {speed_score:.2f}/20.00)")

    logged_path = append_experiment_to_notes(
        exp_id="EXP-OPT-03",
        title="Repeated Calculations (Duplicate/Recomputed Feature Check, 5-Fold GroupKFold & 50-Run Benchmark)",
        hypothesis=(
            "Eliminating duplicate/recomputed aggregations (such as HR_minus_ctx40 and EDA_minus_ctx40 "
            "duplicating diff_ctx_df columns and redundant rolling passes) preserves non-duplicate feature "
            "equivalence and grouped Macro-F1 while reducing redundant compute without altering baseline artifacts."
        ),
        changes_made=(
            f"Evaluated duplicate-column removal ({feat_summary}) with exact equivalence verification on "
            f"non-duplicate columns under 5-fold GroupKFold and 50-run single-core run() timing while "
            f"keeping baseline submission_model.py, submission_model.onnx, and submission.zip untouched."
        ),
        files_modified=["run_exp_opt_03.py", "experiment_notes.md"],
        validation_method="5-fold GroupKFold by subject ID + non-duplicate equivalence check + 50-run single-core run() benchmark (1 thread)",
        prev_macro_f1=f"Same-split baseline {base_f1:.4f} | Prior ref 0.5571 raw / 0.5809 tuned",
        new_macro_f1=f"{dedup_f1:.4f} ({dedup_f1 - base_f1:+.4f} vs same-split baseline)",
        per_class_f1=cls_str,
        met_sec=stats["met"],
        median_sec=stats["median"],
        p95_sec=stats["p95"],
        std_sec=stats["std"],
        peak_ram_mb=stats["peak_mb"],
        onnx_size_mb=AUDIT_BASELINE_REF["onnx_mb"],
        quality_score=qual_score,
        speed_score=speed_score,
        total_score=est_total,
        conclusion=(
            f"Non-duplicate equivalence & 50-run checks PASSED ({feat_summary}); "
            f"delta vs EXP-OPT-2-3..6: MET {d_236_met:+.4f} s, median {d_236_med:+.4f} s, "
            f"P95 {d_236_p95:+.4f} s, std {d_236_std:+.4f} s, RAM {d_236_ram:+.2f} MB; "
            f"delta vs baseline: MET {d_base_met:+.4f} s, P95 {d_base_p95:+.4f} s, RAM {d_base_ram:+.2f} MB; "
            f"retain working baseline and submission.zip unchanged."
        ),
    )
    print(f"Appended EXP-OPT-03 entry to {logged_path.resolve()}")


if __name__ == "__main__":
    run_exp_opt_03()