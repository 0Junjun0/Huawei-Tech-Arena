import os

# Enforce single-core / single-thread execution before importing numerical/ONNX libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pathlib
import sys
import tempfile
import time
import tracemalloc
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
OUT_DIR = ROOT_DIR / "experiments" / "artifacts_42_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "submission", DATA_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_model as tm
import submission_model as sm


def make_single_thread_ort_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def test_onnx_export_and_parity(X: np.ndarray, y: np.ndarray) -> pathlib.Path:
    assert X.shape[1] == 42, f"Expected 42 features, got {X.shape[1]}"

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )),
    ])
    pipe.fit(X, y)

    rf_step = pipe.named_steps["rf"]
    init_types = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        pipe,
        initial_types=init_types,
        options={id(rf_step): {"zipmap": False}},
        target_opset=15,
    )

    cand_onnx = OUT_DIR / "cand_rf_42.onnx"
    cand_onnx.write_bytes(onnx_model.SerializeToString())
    onnx_mb = cand_onnx.stat().st_size / (1024.0 * 1024.0)

    sample_x32 = np.asarray(X[: min(200, len(X))], dtype=np.float32)
    sk_probs = pipe.predict_proba(sample_x32).astype(np.float32)

    sess = ort.InferenceSession(
        str(cand_onnx),
        sess_options=make_single_thread_ort_options(),
        providers=["CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    ort_outs = sess.run(None, {in_name: sample_x32})
    ort_probs = np.asarray(ort_outs[1], dtype=np.float32)

    max_prob_diff = float(np.max(np.abs(sk_probs - ort_probs)))
    label_mismatches = int(np.sum(np.argmax(sk_probs, axis=1) != np.argmax(ort_probs, axis=1)))

    print("=== 1. 42-Feature ONNX Export & Parity Check ===")
    print(f"Exported file:    {cand_onnx} ({onnx_mb:.2f} MB)")
    print(f"ONNX Input:       {in_name} {sess.get_inputs()[0].shape}")
    print(f"Max Prob Diff:    {max_prob_diff:.2e}")
    print(f"Label Mismatches: {label_mismatches}")
    assert onnx_mb <= 50.0, f"ONNX file exceeds 50 MB limit: {onnx_mb:.2f} MB"
    assert max_prob_diff < 1e-4, f"Probability difference too large: {max_prob_diff}"
    assert label_mismatches == 0, f"Found {label_mismatches} argmax label mismatches"
    return cand_onnx


def test_quiet_50_run_speed() -> None:
    parquet_files = sorted((DATA_DIR / "data").rglob("*.parquet"))
    assert parquet_files, "No .parquet files found under data/"

    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]
    model = sm.get_model()

    # Warm-up calls outside timing
    for df in sample_dfs[:3]:
        _ = model.run(df)

    latencies = []
    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]
        t0 = time.perf_counter()
        out = model.run(df)
        latencies.append(time.perf_counter() - t0)
        assert isinstance(out, pd.DataFrame) and list(out.columns) == ["Activity"]
        assert len(out) == len(df) and not out["Activity"].isna().any()

    # Measure peak memory separately so tracemalloc does not inflate latency
    tracemalloc.start()
    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]
        _ = model.run(df)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr = np.asarray(latencies, dtype=np.float64)
    met = float(np.mean(arr))
    speed_score = max(0.0, min(20.0, (1.0 - met / 0.30) * 20.0))

    print("\n=== 2. Quiet 50-Run Single-Core Speed Check ===")
    print(f"MET:          {met:.4f} s (target < 0.3000 s)")
    print(f"Median:       {np.median(arr):.4f} s")
    print(f"P95:          {np.percentile(arr, 95):.4f} s")
    print(f"Std:          {np.std(arr, ddof=1):.4f} s")
    print(f"Peak RAM:     {peak_bytes / (1024 * 1024):.2f} MB")
    print(f"Speed score:  {speed_score:.2f} / 20.00")


def main() -> None:
    X, y, _ = tm.build_dataset(DATA_DIR, win_sec=10)
    test_onnx_export_and_parity(X, y)
    test_quiet_50_run_speed()


if __name__ == "__main__":
    main()