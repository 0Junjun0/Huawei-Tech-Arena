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
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
SUBMISSION_DIR = ROOT_DIR / "submission"
FRIEND_ORIG_DIR = ROOT_DIR / "experiments" / "friend_benchmark"
FRIEND_FIXED_DIR = ROOT_DIR / "experiments" / "friend_fixed"
MERGED_DIR = ROOT_DIR / "experiments" / "merged_candidate"

VALID_LABELS = {"Rest", "Stress", "Activity"}
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def load_mod(unique_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_output_df(out: pd.DataFrame, input_df: pd.DataFrame) -> None:
    assert isinstance(out, pd.DataFrame), "run() must return pd.DataFrame"
    assert list(out.columns) == ["Activity"], f"Expected ['Activity'], got {list(out.columns)}"
    assert len(out) == len(input_df), f"Expected {len(input_df)} rows, got {len(out)}"
    assert not out["Activity"].isna().any(), "Output contains NaNs"
    assert set(out["Activity"].unique()).issubset(VALID_LABELS), "Invalid labels produced"


def benchmark_runtime(label: str, model, onnx_path: pathlib.Path, sample_dfs: list[pd.DataFrame]) -> dict:
    print(f"\n==================== RUNTIME & ROBUSTNESS: {label} ====================")
    df0 = sample_dfs[0]

    # 1. Determinism
    out_a = model.run(df0)
    out_b = model.run(df0)
    validate_output_df(out_a, df0)
    validate_output_df(out_b, df0)
    pd.testing.assert_frame_equal(out_a, out_b)
    det_ok = True
    print("1. Determinism (repeated call): PASS")

    # 2. Reversed columns
    shuffled_cols = [c for c in reversed(SENSOR_COLS) if c in df0.columns] + [
        c for c in df0.columns if c not in SENSOR_COLS
    ]
    out_shuf = model.run(df0[shuffled_cols])
    validate_output_df(out_shuf, df0)
    shuf_match = bool((out_a["Activity"].to_numpy() == out_shuf["Activity"].to_numpy()).all())
    print(f"2. Reversed column order: PASS (exact_match={shuf_match})")

    # 3. Injected NaN
    df_nan = df0.copy()
    for col in SENSOR_COLS:
        if col in df_nan.columns:
            df_nan.iloc[::7, df_nan.columns.get_loc(col)] = np.nan
    out_nan = model.run(df_nan)
    validate_output_df(out_nan, df_nan)
    nan_ok = True
    print("3. Injected NaN robustness: PASS")

    # 4. Unsorted DatetimeIndex
    rng = np.random.default_rng(42)
    df_unsorted = df0.iloc[rng.permutation(len(df0))]
    try:
        out_unsorted = model.run(df_unsorted)
        validate_output_df(out_unsorted, df_unsorted)
        sorted_aligned = out_a["Activity"].loc[out_unsorted.index].to_numpy()
        unsort_match = bool((sorted_aligned == out_unsorted["Activity"].to_numpy()).all())
        print(f"4. Unsorted DatetimeIndex: {'PASS' if unsort_match else 'FAIL'} (exact_match={unsort_match})")
    except Exception as exc:
        unsort_match = False
        print(f"4. Unsorted DatetimeIndex: FAIL ({type(exc).__name__}: {exc})")

    # 5. String index
    df_str = df0.copy()
    df_str.index = df_str.index.astype(str)
    try:
        out_str = model.run(df_str)
        validate_output_df(out_str, df_str)
        str_match = bool((out_a["Activity"].to_numpy() == out_str["Activity"].to_numpy()).all())
        print(f"5. String index: {'PASS' if str_match else 'FAIL'} (exact_match={str_match})")
    except Exception as exc:
        str_match = False
        print(f"5. String index: FAIL ({type(exc).__name__}: {exc})")

    # 6. Missing sensor column
    df_missing = df0.drop(columns=["IBI_dt"])
    try:
        out_miss = model.run(df_missing)
        validate_output_df(out_miss, df_missing)
        miss_ok = True
        print("6. Missing sensor column (IBI_dt dropped): PASS")
    except Exception as exc:
        miss_ok = False
        print(f"6. Missing sensor column (IBI_dt dropped): FAIL ({type(exc).__name__}: {exc})")

    # 50-run single-core MET + separate tracemalloc peak RAM
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

    # ONNX session contract check
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    in_shape = sess.get_inputs()[0].shape

    passed_count = sum([det_ok, shuf_match, nan_ok, unsort_match, str_match, miss_ok])
    print(
        f"ONNX Shape: {in_shape} | Size: {onnx_mb:.2f} MB | Edge Checks: {passed_count}/6\n"
        f"50-Run MET: {met:.4f} s | Median: {med:.4f} s | P95: {p95:.4f} s | "
        f"Std: {std:.4f} s | Peak RAM: {peak_mb:.2f} MB | Speed: {speed_score:.2f}/20"
    )
    return {
        "label": label,
        "edge_passed": passed_count,
        "met": met,
        "median": med,
        "p95": p95,
        "peak_mb": peak_mb,
        "onnx_mb": onnx_mb,
        "speed_score": speed_score,
        "n_feats": in_shape[1],
    }


def report_cached_oof(title: str, cache_path: pathlib.Path) -> dict:
    data = np.load(cache_path, allow_pickle=True)
    y = data["y"]
    groups = data["groups"]
    raw_preds = np.argmax(data["raw_oof"], axis=1)
    sm_preds = np.argmax(data["smoothed_oof"], axis=1)
    nested_preds = data["nested_preds"]
    dep_preds = data["deployed_preds"]

    print(f"\n==================== CV BREAKDOWN: {title} ====================")
    for tag, preds in [
        ("Unsmoothed Raw", raw_preds),
        ("Smoothed Unweighted", sm_preds),
        ("Smoothed + Nested Tuned", nested_preds),
        ("Smoothed + Deployed Weights", dep_preds),
    ]:
        mf1 = float(f1_score(y, preds, average="macro", zero_division=0))
        ba = float(balanced_accuracy_score(y, preds))
        pf1 = f1_score(y, preds, average=None, labels=[0, 1, 2], zero_division=0)
        print(
            f"{tag:28s} | Macro-F1={mf1:.4f} (Qual={mf1*80:.2f}/80) | "
            f"BalAcc={ba:.4f} | F1[R/S/A]={pf1[0]:.4f}/{pf1[1]:.4f}/{pf1[2]:.4f}"
        )

    print("\nConfusion Matrix (Deployed Weights) [Rest, Stress, Activity]:")
    print(confusion_matrix(y, dep_preds, labels=[0, 1, 2]))
    print("\nPer-Class Report (Deployed Weights):")
    print(classification_report(y, dep_preds, target_names=["Rest", "Stress", "Activity"], digits=4, zero_division=0))

    print("=== Lowest-Scoring Subjects (Bottom 10, Deployed Weights) ===")
    subj_rows = []
    for subj in np.unique(groups):
        mask = groups == subj
        s_f1 = f1_score(y[mask], dep_preds[mask], average="macro", zero_division=0)
        subj_rows.append((subj, int(np.sum(mask)), s_f1))
    subj_rows.sort(key=lambda r: r[2])
    for subj, n_w, s_f1 in subj_rows[:10]:
        print(f"Subject {subj:6s} | windows={n_w:4d} | Macro-F1={s_f1:.4f}")

    dep_f1 = float(f1_score(y, dep_preds, average="macro", zero_division=0))
    nest_f1 = float(f1_score(y, nested_preds, average="macro", zero_division=0))
    return {"nested_f1": nest_f1, "deployed_f1": dep_f1}


def main() -> None:
    sys.path.insert(0, str(ROOT_DIR))
    sys.path.insert(0, str(SUBMISSION_DIR))
    parquet_files = sorted((DATA_DIR / "data").rglob("*.parquet"))
    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]

    base_sm = load_mod("base_sm", SUBMISSION_DIR / "submission_model.py")
    friend_orig_sm = load_mod("friend_orig_sm", FRIEND_ORIG_DIR / "submission_model.py")
    friend_fixed_sm = load_mod("friend_fixed_sm", FRIEND_FIXED_DIR / "submission_model.py")
    merged_sm = load_mod("merged_sm", MERGED_DIR / "submission_model.py")

    results = []
    for lbl, mod, onnx_p in [
        ("1. Baseline (RF-42)", base_sm, SUBMISSION_DIR / "submission_model.onnx"),
        ("2. Friend Original (gb_vote, 77f)", friend_orig_sm, FRIEND_ORIG_DIR / "submission_model.onnx"),
        ("3. Friend Fixed (gb_vote+bal, 77f)", friend_fixed_sm, FRIEND_FIXED_DIR / "submission_model.onnx"),
        ("4. Merged Candidate (95f)", merged_sm, MERGED_DIR / "submission_model.onnx"),
    ]:
        model = mod.TechArenaModel(onnx_p)
        results.append(benchmark_runtime(lbl, model, onnx_p, sample_dfs))

    if (MERGED_DIR / "oof_cache.npz").exists():
        report_cached_oof("MERGED CANDIDATE (95f)", MERGED_DIR / "oof_cache.npz")

    print("\n=== RUNTIME SUMMARY TABLE ===")
    for r in results:
        print(
            f"{r['label']:36s} | feats={r['n_feats']} | edge={r['edge_passed']}/6 | "
            f"MET={r['met']:.4f}s | P95={r['p95']:.4f}s | RAM={r['peak_mb']:.2f}MB | "
            f"ONNX={r['onnx_mb']:.2f}MB | Speed={r['speed_score']:.2f}/20"
        )


if __name__ == "__main__":
    main()
