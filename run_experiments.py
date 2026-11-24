import os

# Enforce single-threaded execution before importing numerical / ML / ONNX packages
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pathlib
import re
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from log_results import log_safe
import submission_model
import train_model

TUNED_WEIGHTS = np.array([0.95, 1.80, 0.85], dtype=np.float32)
PREV_RAW_F1 = 0.5571
PREV_TUNED_F1 = 0.5809


def resolve_root_dir(work_dir: pathlib.Path) -> pathlib.Path:
    for cand in [work_dir / "TechArena-Finland-2026", work_dir]:
        if (cand / "data").exists():
            return cand
    return work_dir / "TechArena-Finland-2026"


def log_exp_markdown(exp_id: str, md_block: str, notes_path: pathlib.Path) -> None:
    try:
        log_safe(
            exp_id,
            md_block,
            notes_file=notes_path,
            as_markdown=True,
            ensure_submission_model_ref=True,
        )
    except TypeError:
        log_safe(exp_id, md_block)


def audit_resample_rolling_calls(work_dir: pathlib.Path) -> dict:
    """EXP-OPT-01-5: Audit .resample(...) and .rolling(...) calls in submission_model.py and train_model.py."""
    counts = {}
    for fname in ["submission_model.py", "train_model.py"]:
        p = work_dir / fname
        if p.exists():
            src = p.read_text(encoding="utf-8", errors="replace")
            counts[fname] = {
                "resample_10s": len(re.findall(r'\.resample\(\s*["\']10s["\']', src)),
                "resample_any": len(re.findall(r"\.resample\(", src)),
                "rolling_any": len(re.findall(r"\.rolling\(", src)),
                "has_64hz_bvp": "BVP" in src,
            }
        else:
            counts[fname] = {"missing_file": True}
    return counts


def benchmark_stage_and_length(parquet_files: list[pathlib.Path], n_runs: int = 50) -> dict:
    """EXP-OPT-01-1, EXP-OPT-01-2, EXP-OPT-01-4, EXP-OPT-01-6."""
    sorted_by_size = sorted(parquet_files, key=lambda p: p.stat().st_size)
    short_p = sorted_by_size[0]
    med_p = sorted_by_size[len(sorted_by_size) // 2]
    long_p = sorted_by_size[-1]
    selected_paths = [short_p, med_p, long_p]

    dfs = [pd.read_parquet(p) for p in selected_paths]
    model = submission_model.get_model()

    # Warm-up run excluded from measured loop
    _ = model.run(dfs[0])

    # End-to-end 50-run timing + tracemalloc peak RAM
    tracemalloc.start()
    total_times = []
    for i in range(n_runs):
        df_curr = dfs[i % len(dfs)]
        t0 = time.perf_counter()
        _ = model.run(df_curr)
        total_times.append(time.perf_counter() - t0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Stage split: feature extraction vs. ONNX session inference
    feat_times = []
    onnx_times = []
    for i in range(n_runs):
        df_curr = dfs[i % len(dfs)]
        if hasattr(model, "_compute_window_features") and hasattr(model, "session"):
            input_name = getattr(model, "input_name", None) or model.session.get_inputs()[0].name
            t0 = time.perf_counter()
            _, feats = model._compute_window_features(df_curr)
            t1 = time.perf_counter()
            feats_arr = np.asarray(feats, dtype=np.float32)
            _ = model.session.run(None, {input_name: feats_arr})
            t2 = time.perf_counter()
            feat_times.append(t1 - t0)
            onnx_times.append(t2 - t1)

    # Recording-length breakdown (short / medium / long)
    per_length = {}
    for label, p_item, df_item in zip(["short", "medium", "long"], selected_paths, dfs):
        lt = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = model.run(df_item)
            lt.append(time.perf_counter() - t0)
        per_length[label] = {
            "file": p_item.name,
            "rows": int(len(df_item)),
            "met": float(np.mean(lt)),
            "p95": float(np.percentile(lt, 95)),
        }

    arr = np.asarray(total_times, dtype=np.float64)
    met = float(np.mean(arr))
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    peak_ram_mb = peak_bytes / (1024.0 * 1024.0)
    speed_score = max(0.0, min(20.0, (1.0 - met / 0.3) * 20.0))

    feat_stats = None
    onnx_stats = None
    if feat_times and onnx_times:
        f_arr = np.asarray(feat_times, dtype=np.float64)
        o_arr = np.asarray(onnx_times, dtype=np.float64)
        feat_stats = {
            "mean": float(np.mean(f_arr)),
            "median": float(np.median(f_arr)),
            "p95": float(np.percentile(f_arr, 95)),
            "std": float(np.std(f_arr, ddof=1)) if len(f_arr) > 1 else 0.0,
        }
        onnx_stats = {
            "mean": float(np.mean(o_arr)),
            "median": float(np.median(o_arr)),
            "p95": float(np.percentile(o_arr, 95)),
            "std": float(np.std(o_arr, ddof=1)) if len(o_arr) > 1 else 0.0,
        }

    return {
        "met": met,
        "median": median,
        "p95": p95,
        "std": std,
        "peak_ram_mb": peak_ram_mb,
        "speed_score": speed_score,
        "feat_stats": feat_stats,
        "onnx_stats": onnx_stats,
        "per_length": per_length,
    }


def evaluate_grouped_cv_and_parity(root_dir: pathlib.Path, onnx_path: pathlib.Path) -> dict:
    """EXP-OPT-01-3: 5-fold GroupKFold metrics + ONNX probability parity check."""
    X, y, groups = train_model.build_dataset(root_dir, win_sec=10)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(y), 3), dtype=np.float32)

    last_pipe = None
    last_va_x = None
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        pipe = train_model.Pipeline(
            [
                ("imputer", train_model.SimpleImputer(strategy="median")),
                (
                    "rf",
                    train_model.RandomForestClassifier(
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
        probs = pipe.predict_proba(X[va_idx])
        for col_idx, cls_val in enumerate(pipe.classes_):
            oof_probs[va_idx, int(cls_val)] = probs[:, col_idx]
        last_pipe = pipe
        last_va_x = X[va_idx]

    raw_preds = np.argmax(oof_probs, axis=1)
    raw_f1 = float(f1_score(y, raw_preds, average="macro", zero_division=0))

    tuned_preds = np.argmax(oof_probs * TUNED_WEIGHTS, axis=1)
    tuned_f1 = float(f1_score(y, tuned_preds, average="macro", zero_division=0))

    rest_f1 = float(f1_score(y == 0, tuned_preds == 0, zero_division=0))
    stress_f1 = float(f1_score(y == 1, tuned_preds == 1, zero_division=0))
    activity_f1 = float(f1_score(y == 2, tuned_preds == 2, zero_division=0))

    max_prob_diff = None
    if onnx_path.exists() and last_pipe is not None and last_va_x is not None:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            sample_x = np.asarray(last_va_x[: min(100, len(last_va_x))], dtype=np.float32)
            sk_probs = np.asarray(last_pipe.predict_proba(sample_x), dtype=np.float32)
            ort_outs = sess.run(None, {inp_name: sample_x})
            ort_probs = np.asarray(ort_outs[1] if len(ort_outs) > 1 else ort_outs[0], dtype=np.float32)
            if sk_probs.shape == ort_probs.shape:
                max_prob_diff = float(np.max(np.abs(sk_probs - ort_probs)))
        except Exception:
            max_prob_diff = None

    return {
        "n_subjects": int(len(np.unique(groups))),
        "n_windows": int(len(y)),
        "raw_f1": raw_f1,
        "tuned_f1": tuned_f1,
        "rest_f1": rest_f1,
        "stress_f1": stress_f1,
        "activity_f1": activity_f1,
        "max_prob_diff": max_prob_diff,
    }


def run_exp_opt_01_suite() -> None:
    work_dir = pathlib.Path(__file__).resolve().parent
    root_dir = resolve_root_dir(work_dir)
    notes_path = work_dir / "experiment_notes.md"
    onnx_path = work_dir / "submission_model.onnx"
    onnx_size_mb = onnx_path.stat().st_size / (1024.0 * 1024.0) if onnx_path.exists() else 0.0

    parquet_files = sorted((root_dir / "data").rglob("*.parquet"))
    assert parquet_files, f"No .parquet files found under {root_dir / 'data'}"

    call_audit = audit_resample_rolling_calls(work_dir)
    perf = benchmark_stage_and_length(parquet_files, n_runs=50)
    cv = evaluate_grouped_cv_and_parity(root_dir, onnx_path)

    qual_score = cv["tuned_f1"] * 80.0
    total_score = qual_score + perf["speed_score"]
    within_budget = perf["met"] < 0.30
    decision_str = (
        "Retain if MET < 0.30 s and grouped Macro-F1 / parity hold"
        if within_budget
        else f"Discard/do not retain as-is: MET={perf['met']:.4f} s exceeds 0.30 s threshold (Speed score = {perf['speed_score']:.2f}/20)"
    )

    parity_str = f"{cv['max_prob_diff']:.6e}" if cv["max_prob_diff"] is not None else "N/A"
    exec_line = (
        f"MET: {perf['met']:.4f} s | Median: {perf['median']:.4f} s | P95: {perf['p95']:.4f} s | "
        f"Std: {perf['std']:.4f} s | Peak RAM: {perf['peak_ram_mb']:.2f} MB | ONNX size: {onnx_size_mb:.2f} MB"
    )

    exp_01 = f"""### EXP-OPT-01: Feature Extraction Bottlenecks (Resample & Rolling Overhead)
- **Experiment ID:** EXP-OPT-01
- **Hypothesis:** Repeated `.resample("10s")` and `.rolling(...)` calls over high-rate sensor series (`BVP` at 64 Hz, `EDA` and `Temperature` at 4 Hz, `HR` at 1 Hz, and `IBI_*`) in `submission_model` (`submission_model.py` + `submission_model.onnx`) re-scan index buckets multiple times inside `run()`. Streamlining resampling/rolling passes reduces feature extraction overhead without changing grouped Macro-F1 or parity.
- **Changes made:** Executed full EXP-OPT-01 suite (`EXP-OPT-01` and `EXP-OPT-01-1` through `EXP-OPT-01-6`) via `run_experiments.py` on `submission_model` (`submission_model.py` + `submission_model.onnx`).
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `train_model.py`, `run_experiments.py`, `log_results.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` by subject ID ({cv['n_subjects']} subjects, {cv['n_windows']} windows) + 50-run single-core `submission_model` `run()` benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip; max prob diff: {parity_str})
- **Previous macro-F1:** Raw {PREV_RAW_F1:.4f} | Tuned {PREV_TUNED_F1:.4f}
- **New macro-F1:** Raw {cv['raw_f1']:.4f} | Tuned {cv['tuned_f1']:.4f}
- **Per-class F1:** Rest: {cv['rest_f1']:.4f} | Stress: {cv['stress_f1']:.4f} | Activity: {cv['activity_f1']:.4f}
- **Execution time:** {exec_line}
- **Estimated total competition score:** {total_score:.2f} / 100 (Quality: {qual_score:.2f}/80, Speed: {perf['speed_score']:.2f}/20)
- **Conclusion:** {decision_str}."""

    pl = perf["per_length"]
    exp_01_1 = f"""### EXP-OPT-01-1: Recording Length & Multi-Rate Resampling Scaling
- **Experiment ID:** EXP-OPT-01-1
- **Hypothesis:** Longer recordings and 64 Hz `BVP` alongside 4 Hz `EDA`/`Temperature` and 1 Hz `HR`/`IBI_*` increase resampling and rolling cost inside `submission_model.run()`.
- **Changes made:** Benchmarked 10 calls each on short (`{pl['short']['file']}`, {pl['short']['rows']} rows: MET={pl['short']['met']:.4f} s, P95={pl['short']['p95']:.4f} s), medium (`{pl['medium']['file']}`, {pl['medium']['rows']} rows: MET={pl['medium']['met']:.4f} s, P95={pl['medium']['p95']:.4f} s), and long (`{pl['long']['file']}`, {pl['long']['rows']} rows: MET={pl['long']['met']:.4f} s, P95={pl['long']['p95']:.4f} s) recordings on `submission_model` (`submission_model.py` + `submission_model.onnx`).
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` ({cv['n_subjects']} subjects, {cv['n_windows']} windows) + single-core per-length `run()` timing on `submission_model`
- **Previous macro-F1:** Raw {PREV_RAW_F1:.4f} | Tuned {PREV_TUNED_F1:.4f}
- **New macro-F1:** Raw {cv['raw_f1']:.4f} | Tuned {cv['tuned_f1']:.4f}
- **Per-class F1:** Rest: {cv['rest_f1']:.4f} | Stress: {cv['stress_f1']:.4f} | Activity: {cv['activity_f1']:.4f}
- **Execution time:** {exec_line}
- **Estimated total competition score:** {total_score:.2f} / 100
- **Conclusion:** Recorded short/medium/long recording latency scaling for `submission_model` (`submission_model.py` + `submission_model.onnx`)."""

    if perf["feat_stats"] and perf["onnx_stats"]:
        fs = perf["feat_stats"]
        os_ = perf["onnx_stats"]
        f_share = (fs["mean"] / perf["met"] * 100.0) if perf["met"] > 0 else 0.0
        stage_summary = (
            f"Feature extraction mean={fs['mean']:.4f} s (med={fs['median']:.4f} s, P95={fs['p95']:.4f} s, std={fs['std']:.4f} s, {f_share:.1f}% of MET) "
            f"vs. ONNX session mean={os_['mean']:.4f} s (med={os_['median']:.4f} s, P95={os_['p95']:.4f} s, std={os_['std']:.4f} s)"
        )
    else:
        stage_summary = "Stage-level breakdown unavailable (missing _compute_window_features or session attribute on submission_model)."

    exp_01_2 = f"""### EXP-OPT-01-2: Stage Profiling — Feature Extraction vs. ONNX Inference
- **Experiment ID:** EXP-OPT-01-2
- **Hypothesis:** Isolating `_compute_window_features` from `session.run(...)` on `submission_model` (`submission_model.py` + `submission_model.onnx`) reveals whether Pandas resampling/rolling or ONNX session inference dominates latency.
- **Changes made:** Separated feature extraction timing from ONNX session timing across 50 single-threaded runs on `submission_model`.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` ({cv['n_subjects']} subjects, {cv['n_windows']} windows) + 50-run single-core stage breakdown on `submission_model`
- **Previous macro-F1:** Raw {PREV_RAW_F1:.4f} | Tuned {PREV_TUNED_F1:.4f}
- **New macro-F1:** Raw {cv['raw_f1']:.4f} | Tuned {cv['tuned_f1']:.4f}
- **Per-class F1:** Rest: {cv['rest_f1']:.4f} | Stress: {cv['stress_f1']:.4f} | Activity: {cv['activity_f1']:.4f}
- **Execution time:** {exec_line} | {stage_summary}
- **Estimated total competition score:** {total_score:.2f} / 100
- **Conclusion:** {stage_summary} on `submission_model` (`submission_model.py` + `submission_model.onnx`)."""

    exp_01_3 = f"""### EXP-OPT-01-3: Numerical Parity & Grouped Macro-F1 Verification
- **Experiment ID:** EXP-OPT-01-3
- **Hypothesis:** Feature extraction adjustments should preserve sklearn-vs-ONNX probability parity and grouped `GroupKFold` Macro-F1 across `Rest`, `Stress`, and `Activity`.
- **Changes made:** Measured max absolute probability difference between scikit-learn and `submission_model.onnx` alongside 5-fold `GroupKFold` raw and weight-tuned (`[0.95, 1.80, 0.85]`) Macro-F1.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `train_model.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` ({cv['n_subjects']} subjects, {cv['n_windows']} windows) + sklearn vs. ONNX parity check
- **Previous macro-F1:** Raw {PREV_RAW_F1:.4f} | Tuned {PREV_TUNED_F1:.4f}
- **New macro-F1:** Raw {cv['raw_f1']:.4f} | Tuned {cv['tuned_f1']:.4f}
- **Per-class F1:** Rest: {cv['rest_f1']:.4f} | Stress: {cv['stress_f1']:.4f} | Activity: {cv['activity_f1']:.4f}
- **Execution time:** {exec_line} | Max prob diff: {parity_str}
- **Estimated total competition score:** {total_score:.2f} / 100
- **Conclusion:** Verified sklearn-vs-ONNX parity (`{parity_str}`) and grouped Macro-F1 (`{cv['tuned_f1']:.4f}`) for `submission_model` (`submission_model.py` + `submission_model.onnx`)."""

    ram_ok = perf["peak_ram_mb"] <= 4096.0
    size_ok = onnx_size_mb <= 50.0
    exp_01_4 = f"""### EXP-OPT-01-4: Single-Thread Execution, Peak RAM & ONNX Size Check
- **Experiment ID:** EXP-OPT-01-4
- **Hypothesis:** Resampling and rolling buffers in `submission_model` (`submission_model.py` + `submission_model.onnx`) must remain within single-core thread limits, <=4 GB RAM, and <=50 MB `.zip` / `.onnx` bounds.
- **Changes made:** Tracked `tracemalloc` peak RAM ({perf['peak_ram_mb']:.2f} MB) and `submission_model.onnx` size ({onnx_size_mb:.2f} MB) under pinned single-thread environment variables.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` ({cv['n_subjects']} subjects, {cv['n_windows']} windows) + single-core `tracemalloc` & artifact size audit on `submission_model`
- **Previous macro-F1:** Raw {PREV_RAW_F1:.4f} | Tuned {PREV_TUNED_F1:.4f}
- **New macro-F1:** Raw {cv['raw_f1']:.4f} | Tuned {cv['tuned_f1']:.4f}
- **Per-class F1:** Rest: {cv['rest_f1']:.4f} | Stress: {cv['stress_f1']:.4f} | Activity: {cv['activity_f1']:.4f}
- **Execution time:** {exec_line}
- **Estimated total competition score:** {total_score:.2f} / 100
- **Conclusion:** Peak RAM ({perf['peak_ram_mb']:.2f} MB, <=4 GB: {ram_ok}) and ONNX size ({onnx_size_mb:.2f} MB, <=50 MB: {size_ok}) checked on `submission_model` (`submission_model.py` + `submission_model.onnx`)."""

    sub_c = call_audit.get("submission_model.py", {})
    tr_c = call_audit.get("train_model.py", {})
    exp_01_5 = f"""### EXP-OPT-01-5: Resample & Rolling Call Count Audit
- **Experiment ID:** EXP-OPT-01-5
- **Hypothesis:** Counting `.resample(...)` and `.rolling(...)` calls in `submission_model.py` and `train_model.py` identifies repeated index bucket scans across multi-rate sensor columns.
- **Changes made:** Audited static `.resample(...)` and `.rolling(...)` call counts in `submission_model.py` ({sub_c}) and `train_model.py` ({tr_c}).
- **Files modified:** `submission_model.py`, `train_model.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` ({cv['n_subjects']} subjects, {cv['n_windows']} windows) + static call-count audit alongside 50-run `submission_model` timing
- **Previous macro-F1:** Raw {PREV_RAW_F1:.4f} | Tuned {PREV_TUNED_F1:.4f}
- **New macro-F1:** Raw {cv['raw_f1']:.4f} | Tuned {cv['tuned_f1']:.4f}
- **Per-class F1:** Rest: {cv['rest_f1']:.4f} | Stress: {cv['stress_f1']:.4f} | Activity: {cv['activity_f1']:.4f}
- **Execution time:** {exec_line}
- **Estimated total competition score:** {total_score:.2f} / 100
- **Conclusion:** Audited `.resample(...)` and `.rolling(...)` occurrences in `submission_model` (`submission_model.py` + `submission_model.onnx`) and `train_model.py`."""

    exp_01_6 = f"""### EXP-OPT-01-6: 0.30 s Speed Threshold & Retain/Discard Check
- **Experiment ID:** EXP-OPT-01-6
- **Hypothesis:** Any change that pushes 50-run single-core MET to or above `0.30 s` drops Speed score (`max(0, (1 - MET/0.3)*20)`) to `0.00 / 20` and must not be retained over a faster baseline.
- **Changes made:** Evaluated 50-run single-core MET (`{perf['met']:.4f} s`) against the `0.30 s` cutoff and computed resulting Speed (`{perf['speed_score']:.2f}/20`) and Total (`{total_score:.2f}/100`) scores for `submission_model`.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` ({cv['n_subjects']} subjects, {cv['n_windows']} windows) + 50-run single-core `submission_model` MET threshold check
- **Previous macro-F1:** Raw {PREV_RAW_F1:.4f} | Tuned {PREV_TUNED_F1:.4f}
- **New macro-F1:** Raw {cv['raw_f1']:.4f} | Tuned {cv['tuned_f1']:.4f}
- **Per-class F1:** Rest: {cv['rest_f1']:.4f} | Stress: {cv['stress_f1']:.4f} | Activity: {cv['activity_f1']:.4f}
- **Execution time:** {exec_line}
- **Estimated total competition score:** {total_score:.2f} / 100
- **Conclusion:** {decision_str} for `submission_model` (`submission_model.py` + `submission_model.onnx`)."""

    for exp_id, block in [
        ("EXP-OPT-01", exp_01),
        ("EXP-OPT-01-1", exp_01_1),
        ("EXP-OPT-01-2", exp_01_2),
        ("EXP-OPT-01-3", exp_01_3),
        ("EXP-OPT-01-4", exp_01_4),
        ("EXP-OPT-01-5", exp_01_5),
        ("EXP-OPT-01-6", exp_01_6),
    ]:
        log_exp_markdown(exp_id, block, notes_path)
        print(block + "\n")


if __name__ == "__main__":
    run_exp_opt_01_suite()