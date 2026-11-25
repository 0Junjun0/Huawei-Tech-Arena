import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pathlib
import sys
import time
import tracemalloc
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026" / "data"
VALID_LABELS = {"Rest", "Stress", "Activity"}

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "submission", ROOT_DIR / "TechArena-Finland-2026"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import submission_model as sub_mod


def check_onnx_parity(model, onnx_path: pathlib.Path, sample_dfs: list[pd.DataFrame]) -> None:
    assert onnx_path.exists(), f"Missing ONNX file: {onnx_path}"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    assert inp.shape[1] == 42, f"Expected 42 ONNX input features, got {inp.shape}"

    max_prob_diffs = []
    label_mismatches = 0

    for df in sample_dfs[:5]:
        _, feats = sub_mod._compute_window_features(df, win_sec=10)
        x32 = np.asarray(feats, dtype=np.float32)
        assert x32.ndim == 2 and x32.shape[1] == 42

        ort_outs = sess.run(None, {inp.name: x32})
        if len(ort_outs) > 1 and hasattr(model, "session"):
            mod_outs = model.session.run(None, {model.input_name: x32})
            p_ort = np.asarray(ort_outs[1], dtype=np.float32)
            p_mod = np.asarray(mod_outs[1], dtype=np.float32)
            max_prob_diffs.append(float(np.max(np.abs(p_ort - p_mod))))
            l_ort = np.asarray(ort_outs[0]).reshape(-1)
            l_mod = np.asarray(mod_outs[0]).reshape(-1)
            label_mismatches += int(np.sum(l_ort != l_mod))

    max_diff = max(max_prob_diffs) if max_prob_diffs else 0.0
    print(f"ONNX Input:       {inp.name} {inp.shape} ({inp.type})")
    print(f"Max Prob Diff:    {max_diff:.2e}")
    print(f"Label Mismatches: {label_mismatches}")
    assert max_diff < 1e-5, f"Probability diff too high: {max_diff}"
    assert label_mismatches == 0, f"Found {label_mismatches} ONNX label mismatches"


def main() -> None:
    search_dir = DATA_DIR if DATA_DIR.exists() else ROOT_DIR
    parquet_files = sorted(search_dir.rglob("*.parquet"))
    assert parquet_files, f"No .parquet files found under {search_dir}"

    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]
    model = sub_mod.get_model()

    onnx_path = pathlib.Path(sub_mod.__file__).with_suffix(".onnx")
    check_onnx_parity(model, onnx_path, sample_dfs)

    # Warm-up before timing
    _ = model.run(sample_dfs[0])

    latencies = []
    tracemalloc.start()
    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]
        t0 = time.perf_counter()
        out = model.run(df)
        dt = time.perf_counter() - t0
        assert isinstance(out, pd.DataFrame) and list(out.columns) == ["Activity"]
        assert len(out) == len(df) and not out["Activity"].isna().any()
        assert set(out["Activity"].unique()).issubset(VALID_LABELS)
        latencies.append(dt)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr = np.asarray(latencies, dtype=np.float64)
    met = float(np.mean(arr))
    speed_score = max(0.0, min(20.0, (1.0 - met / 0.30) * 20.0))

    print("=== 42-Feature Single-Core E2E Speed Check (50 Runs) ===")
    print(f"MET:          {met:.4f} s (target < 0.3000 s)")
    print(f"Median:       {np.median(arr):.4f} s")
    print(f"P95:          {np.percentile(arr, 95):.4f} s")
    print(f"Peak RAM:     {peak_bytes / (1024 * 1024):.2f} MB")
    print(f"Speed score:  {speed_score:.2f} / 20.00")


if __name__ == "__main__":
    main()