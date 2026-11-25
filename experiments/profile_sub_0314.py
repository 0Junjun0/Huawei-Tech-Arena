import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cProfile
import io
import pathlib
import pstats
import sys
import time
import numpy as np
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026" / "data"
SUBMISSION_DIR = ROOT_DIR / "submission"

for p in (ROOT_DIR, SUBMISSION_DIR, ROOT_DIR / "TechArena-Finland-2026"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import submission_model as sub_mod


def stats_line(label: str, times: list[float]) -> None:
    arr = np.asarray(times, dtype=np.float64)
    print(
        f"{label:28s} | mean={np.mean(arr):.4f}s | median={np.median(arr):.4f}s | "
        f"p95={np.percentile(arr, 95):.4f}s | max={np.max(arr):.4f}s"
    )


def main() -> None:
    search_dir = DATA_DIR if DATA_DIR.exists() else ROOT_DIR
    parquet_files = sorted(search_dir.rglob("*.parquet"))
    assert parquet_files, f"No .parquet files found under {search_dir}"

    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]
    model = sub_mod.get_model()

    # Warm-up
    _ = model.run(sample_dfs[0])

    total_times = []
    feat_times = []
    onnx_times = []

    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]

        t0 = time.perf_counter()
        out = model.run(df)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)
        assert len(out) > 0

        if hasattr(model, "_compute_window_features") and hasattr(model, "session"):
            f0 = time.perf_counter()
            _, feats = model._compute_window_features(df)
            f1 = time.perf_counter()
            x_in = np.asarray(feats, dtype=np.float32)
            in_name = getattr(model, "input_name", None) or model.session.get_inputs()[0].name
            _ = model.session.run(None, {in_name: x_in})
            f2 = time.perf_counter()
            feat_times.append(f1 - f0)
            onnx_times.append(f2 - f1)

    print("=== Stage Timing Breakdown (50 Single-Core Runs) ===")
    stats_line("Full model.run(df)", total_times)
    if feat_times and onnx_times:
        stats_line("_compute_window_features", feat_times)
        stats_line("ONNX session.run()", onnx_times)

    # Top functions inside model.run() on one sample DataFrame
    pr = cProfile.Profile()
    pr.enable()
    _ = model.run(sample_dfs[0])
    pr.disable()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(15)
    print("\n=== Top 15 Functions by Own Time (tottime) on 1 Call ===")
    print(buf.getvalue())


if __name__ == "__main__":
    main()