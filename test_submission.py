# This section enforces single-threaded execution to replicate the evaluation container.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ast
import pathlib
import time
import tracemalloc
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from adm_base_class import ADM_Base
import submission_model as m
from train_model import find_dataset_root, map_phase_to_activity


def verify_no_threading_ast(py_files: list[str]) -> None:
    """This subprogram inspects Python AST imports to confirm no threading or multiprocessing modules are used."""
    forbidden = {"threading", "multiprocessing", "concurrent", "joblib", "_thread"}
    for py_path in py_files:
        p = pathlib.Path(py_path)
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=py_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"Forbidden import '{alias.name}' in {py_path}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"Forbidden import '{node.module}' in {py_path}"
    print(f"[PASS] AST threading check passed for: {py_files}")


def verify_zip_package(zip_path: str = "submission.zip") -> None:
    """This subprogram verifies that submission.zip is under 50 MB and contains only matching .py and .onnx files."""
    zp = pathlib.Path(zip_path)
    if not zp.exists():
        print(f"[SKIP] {zip_path} not created yet.")
        return
    size_mb = zp.stat().st_size / (1024 * 1024)
    assert size_mb < 50.0, f"Zip archive exceeds 50 MB limit ({size_mb:.2f} MB)"
    with zipfile.ZipFile(zp, "r") as zf:
        names = set(zf.namelist())
        assert names == {"submission_model.py", "submission_model.onnx"}, f"Invalid zip contents: {names}"
    print(f"[PASS] Zip check: {size_mb:.2f} MB | Contents: {sorted(names)}")


def run_full_benchmark() -> None:
    """This subprogram benchmarks contract compliance, RAM, 50-run MET, and Estimated Total Score (/ 100)."""
    root_dir = find_dataset_root()
    all_files = sorted((root_dir / "data").rglob("*.parquet"))
    assert len(all_files) > 0, "No .parquet files found under data/"

    # Select balanced recordings across STRESS, AEROBIC, and ANAEROBIC for timing runs
    timing_files = []
    for proto in ["STRESS", "AEROBIC", "ANAEROBIC"]:
        timing_files.extend(sorted((root_dir / "data" / proto).glob("*.parquet"))[:4])
    if not timing_files:
        timing_files = all_files[:12]

    timing_inputs = [pd.read_parquet(f)[m.SENSOR_COLS].copy() for f in timing_files]

    tracemalloc.start()
    model = m.get_model()
    assert isinstance(model, ADM_Base), "get_model() must return an ADM_Base instance"

    # Warm-up inference and output contract verification
    out_df = model.run(timing_inputs[0])
    assert isinstance(out_df, pd.DataFrame), "run() must return a pandas DataFrame"
    assert list(out_df.columns) == ["Activity"], f"Invalid output columns: {out_df.columns}"
    assert out_df.index.equals(timing_inputs[0].index), "Output DatetimeIndex does not match input index"
    assert set(out_df["Activity"].unique()).issubset({"Rest", "Stress", "Activity"}), "Invalid activity labels"

    # Measure Mean Execution Time (MET) over 50 runs across all 3 protocols
    timings = []
    for i in range(50):
        df_in = timing_inputs[i % len(timing_inputs)]
        t0 = time.perf_counter()
        _ = model.run(df_in)
        timings.append(time.perf_counter() - t0)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    met = float(np.mean(timings))
    speed_score = max(0.0, (1.0 - met / 0.3) * 20.0)
    print(f"[PASS] ADM_Base & output schema valid.")
    print(f"[INFO] Peak Python RAM: {peak_bytes / (1024 * 1024):.2f} MB (Limit: 4096 MB)")
    print(f"[INFO] 50-run MET: {met * 1000:.2f} ms (Min: {min(timings)*1000:.2f} ms, Max: {max(timings)*1000:.2f} ms)")

    # Evaluate 10-second majority-vote Macro-F1 across all recordings in data/
    all_true, all_pred = [], []
    for pq_path in all_files:
        raw_df = pd.read_parquet(pq_path)
        if "Phase" not in raw_df.columns:
            continue
        pred_df = model.run(raw_df[m.SENSOR_COLS])
        true_series = raw_df["Phase"].map(map_phase_to_activity)
        valid = true_series.notna()
        if not valid.any():
            continue

        win_bins = pd.to_datetime(raw_df.index[valid]).floor("10s")
        t_win = true_series[valid].groupby(win_bins).agg(lambda s: s.mode().iloc[0])
        p_win = pred_df.loc[valid, "Activity"].groupby(win_bins).agg(lambda s: s.mode().iloc[0])
        all_true.extend(t_win.tolist())
        all_pred.extend(p_win.tolist())

    if all_true:
        labels = ["Rest", "Stress", "Activity"]
        per_cls = f1_score(all_true, all_pred, labels=labels, average=None, zero_division=0)
        macro_f1 = float(np.mean(per_cls))
        quality_score = macro_f1 * 80.0
        total_score = quality_score + speed_score

        print(f"[INFO] Per-Class F1 -> Rest: {per_cls[0]:.4f} | Stress: {per_cls[1]:.4f} | Activity: {per_cls[2]:.4f}")
        print(f"[INFO] Full Dataset Macro-F1: {macro_f1:.4f}")
        print(f"[INFO] Estimated Quality Score: {quality_score:.2f} / 80.00")
        print(f"[INFO] Estimated Speed Score:   {speed_score:.2f} / 20.00")
        print(f"[INFO] Estimated Total Score:   {total_score:.2f} / 100.00")
        print("       (Note: Above Total Score uses full-dataset predictions; compare with Nested OOF Quality + Speed Score for unseen-subject estimate.)")


if __name__ == "__main__":
    verify_no_threading_ast(["submission_model.py", "train_model.py", "test_submission.py"])
    run_full_benchmark()
    verify_zip_package("submission.zip")