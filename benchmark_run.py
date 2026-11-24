import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pathlib
import sys
import time
import pandas as pd

sys.path.insert(0, "TechArena-Finland-2026")
import submission_model as m


def main() -> None:
    model = m.get_model()
    files = sorted(pathlib.Path("TechArena-Finland-2026/data").rglob("*.parquet"))
    sample_dfs = [pd.read_parquet(f) for f in files[:10]]

    # Warm-up run
    _ = model.run(sample_dfs[0])

    times = []
    for i in range(50):
        df = sample_dfs[i % len(sample_dfs)]
        t0 = time.perf_counter()
        _ = model.run(df)
        times.append(time.perf_counter() - t0)

    met = sum(times) / len(times)
    speed_score = max(0.0, (1.0 - met / 0.3) * 20.0)
    print(f"50-run MET: {met:.4f} s | Min: {min(times):.4f} s | Max: {max(times):.4f} s")
    print(f"Estimated Speed Score: {speed_score:.2f} / 20.00")


if __name__ == "__main__":
    main()