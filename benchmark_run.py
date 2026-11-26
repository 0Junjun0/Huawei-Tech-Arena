"""Measure the deployed run() method on one core, excluding input I/O and model loading."""

import argparse
import hashlib
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from submission_model import LABELS, TechArenaModel

INPUT_COLUMNS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
DEFAULT_MODEL = Path(__file__).with_name("submission_model.onnx")


@contextmanager
def one_core():
    prior = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    try:
        if prior:
            os.sched_setaffinity(0, {min(prior)})
        else:
            warnings.warn("CPU affinity unavailable. Single-core timing is unverified.")
        yield
    finally:
        if prior:
            os.sched_setaffinity(0, prior)


def read_sensors(path):
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [name for name in INPUT_COLUMNS if name in available]
    return pd.read_parquet(path, columns=columns).reindex(columns=INPUT_COLUMNS)


def samples(files, count=10):
    if count < 1:
        raise ValueError("Timing requires at least one recording")
    ranked = sorted((pq.ParquetFile(f).metadata.num_rows, f) for f in files)
    ranked = [(size, f) for size, f in ranked if size]
    if not ranked:
        raise ValueError("No nonempty recordings")
    return [
        read_sensors(ranked[i][1])
        for i in np.linspace(0, len(ranked) - 1, min(count, len(ranked)), dtype=int)
    ]


def validate_output(output, frame):
    if (
        not isinstance(output, pd.DataFrame)
        or list(output.columns) != ["Activity"]
        or not output.index.equals(frame.index)
        or not output.Activity.isin(LABELS).all()
    ):
        raise ValueError("Expected one valid Activity label per original timestamp")


def measure(model, frames, runs=50):
    if not frames or runs < 1:
        raise ValueError("Timing requires recordings and a positive run count")
    validate_output(model.run(frames[0]), frames[0])
    times = []
    for i in range(runs):
        frame = frames[i % len(frames)]
        start = perf_counter()
        output = model.run(frame)
        times.append(perf_counter() - start)
        validate_output(output, frame)
        del output
    return np.array(times)


def speed(met):
    return max(0.0, 20 * (1 - met / 0.3))


def load_timed_model(path):
    # The graph and its printed identity come from the same read even during a retrain.
    artifact = path.read_bytes()
    return TechArenaModel(artifact), hashlib.sha256(artifact).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("TechArena-Finland-2026/data")
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--recordings", type=int, default=10)
    args = parser.parse_args()
    with one_core():
        frames = samples(args.data.rglob("*.parquet"), args.recordings)
        model, fingerprint = load_timed_model(args.model)
        times = measure(model, frames)
    print(f"Model SHA-256 {fingerprint}")
    print(
        f"{len(times)} runs on {len(frames)} recordings, "
        f"mean {1000 * times.mean():.2f} ms, p95 {1000 * np.quantile(times, 0.95):.2f} ms, "
        f"speed {speed(times.mean()):.3f}/20"
    )
