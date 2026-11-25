"""Run on a single core: python benchmark_run.py --data path/to/data"""

import argparse
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from submission_model import get_model


def samples(files):
    ranked = sorted((pq.ParquetFile(f).metadata.num_rows, f) for f in files)
    ranked = [(size, f) for size, f in ranked if size]
    if not ranked:
        raise ValueError("No nonempty recordings")
    return [
        pd.read_parquet(ranked[i][1])
        for i in np.linspace(0, len(ranked) - 1, min(10, len(ranked)), dtype=int)
    ]


def measure(model, frames, runs=50):
    model.run(frames[0])
    times = []
    for i in range(runs):
        frame = frames[i % len(frames)]
        start = perf_counter()
        output = model.run(frame)
        times.append(perf_counter() - start)
        assert output.index.equals(frame.index)
    return np.array(times)


def speed(met):
    return max(0.0, 20 * (1 - met / 0.3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("TechArena-Finland-2026/data")
    )
    args = parser.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    times = measure(get_model(), samples(args.data.rglob("*.parquet")))
    print(
        f"50 runs: mean {times.mean() * 1000:.2f} ms, min {times.min() * 1000:.2f}, max {times.max() * 1000:.2f}; speed {speed(times.mean()):.2f}/20"
    )
