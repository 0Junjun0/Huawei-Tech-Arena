import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pathlib
import sys
import time
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026" / "data"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "submission", ROOT_DIR / "TechArena-Finland-2026"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import submission_model as sm


def main() -> None:
    search_dir = DATA_DIR if DATA_DIR.exists() else ROOT_DIR
    files = sorted(search_dir.rglob("*.parquet"))
    assert files, f"No .parquet files found under {search_dir}"

    model = sm.get_model()
    _ = model.run(pd.read_parquet(files[0]))

    records = []
    for path in files:
        df = pd.read_parquet(path)
        t0 = time.perf_counter()
        _ = model.run(df)
        dt = time.perf_counter() - t0

        idx = pd.to_datetime(df.index)
        span_sec = float((idx.max() - idx.min()).total_seconds()) if len(idx) > 1 else 0.0
        records.append({
            "file": str(path.relative_to(search_dir)),
            "rows": len(df),
            "span_s": span_sec,
            "run_s": dt,
        })

    by_time = sorted(records, key=lambda r: r["run_s"], reverse=True)
    by_rows = sorted(records, key=lambda r: r["rows"], reverse=True)

    print("=== Top 10 Slowest Single-Core run() Calls ===")
    for r in by_time[:10]:
        print(f"  {r['file']:32s} | rows={r['rows']:7d} | span={r['span_s']:7.1f}s | run={r['run_s']:.4f}s")

    print("\n=== Top 10 Longest Files by Row Count ===")
    for r in by_rows[:10]:
        print(f"  {r['file']:32s} | rows={r['rows']:7d} | span={r['span_s']:7.1f}s | run={r['run_s']:.4f}s")


if __name__ == "__main__":
    main()