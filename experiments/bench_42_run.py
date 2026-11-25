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
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026" / "data"
SUBMISSION_DIR = ROOT_DIR / "submission"
VALID_LABELS = {"Rest", "Stress", "Activity"}

for p in (ROOT_DIR, SUBMISSION_DIR, ROOT_DIR / "TechArena-Finland-2026"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import submission_model as sub_mod


def main() -> None:
    search_dir = DATA_DIR if DATA_DIR.exists() else ROOT_DIR
    parquet_files = sorted(search_dir.rglob("*.parquet"))
    assert parquet_files, f"No .parquet files found under {search_dir}"

    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]
    model = sub_mod.get_model()

    # Warm-up run outside timing loop
    _ = model.run(sample_dfs[0])

    latencies = []
    tracemalloc.start()
    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]
        t0 = time.perf_counter()
        out = model.run(df)
        dt = time.perf_counter() - t0
        assert isinstance(out, pd.DataFrame) and list(out.columns) == ["Activity"]
        assert len(out) > 0 and not out["Activity"].isna().any()
        assert set(out["Activity"].unique()).issubset(VALID_LABELS)
        latencies.append(dt)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr = np.asarray(latencies, dtype=np.float64)
    met = float(np.mean(arr))
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    std = float(np.std(arr, ddof=1))
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    speed_score = max(0.0, min(20.0, (1.0 - met / 0.30) * 20.0))

    onnx_path = SUBMISSION_DIR / "submission_model.onnx"
    onnx_mb = (onnx_path.stat().st_size / (1024.0 * 1024.0)) if onnx_path.exists() else float("nan")

    print("=== 50-Run Single-Core run() Speed Check ===")
    print(f"MET:          {met:.4f} s")
    print(f"Median:       {median:.4f} s")
    print(f"P95:          {p95:.4f} s")
    print(f"Std:          {std:.4f} s")
    print(f"Peak RAM:     {peak_mb:.2f} MB")
    print(f"ONNX size:    {onnx_mb:.2f} MB")
    print(f"Speed score:  {speed_score:.2f} / 20.00")


if __name__ == "__main__":
    main()