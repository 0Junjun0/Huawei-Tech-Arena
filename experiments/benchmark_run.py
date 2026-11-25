import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
NOTES_PATH = ROOT_DIR / "notes" / "experiment_notes.md"
SUBMISSION_DIR = ROOT_DIR / "submission"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "experiments", DATA_DIR, SUBMISSION_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import pandas as pd
import submission_model as m


def main() -> None:
    model = m.get_model()
    data_sub_dir = DATA_DIR / "data" if (DATA_DIR / "data").exists() else DATA_DIR
    files = sorted(data_sub_dir.rglob("*.parquet"))
    sample_dfs = [pd.read_parquet(f) for f in files[:10]]

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