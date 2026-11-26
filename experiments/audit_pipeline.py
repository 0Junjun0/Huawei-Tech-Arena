import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
NOTES_PATH = ROOT_DIR / "notes" / "experiment_notes.md"
SUBMISSION_DIR = ROOT_DIR / "submission"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "experiments", DATA_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os

# Enforce single-threaded execution before importing numpy/scipy/sklearn/onnxruntime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import inspect
import tempfile
import time
import tracemalloc
import zipfile
import numpy as np
import onnxruntime as ort
import pandas as pd

SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
VALID_LABELS = {"Stress", "Activity", "Rest"}


def check_files_and_zip() -> pathlib.Path:
    print("=== 1. File & ZIP Audit ===")
    py_path = SUBMISSION_DIR / "submission_model.py"
    onnx_path = SUBMISSION_DIR / "submission_model.onnx"
    base_path = DATA_DIR / "adm_base_class.py"
    if not base_path.exists() and (ROOT_DIR / "adm_base_class.py").exists():
        base_path = ROOT_DIR / "adm_base_class.py"

    assert py_path.exists(), f"Missing {py_path}"
    assert onnx_path.exists(), f"Missing {onnx_path}"
    assert base_path.exists(), f"Missing {base_path}"
    assert py_path.stem == onnx_path.stem, "Python and ONNX base names must match"

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Found {py_path.name} and {onnx_path.name} ({onnx_size_mb:.2f} MB)")

    zip_candidates = sorted(list(SUBMISSION_DIR.glob("*.zip")) + list(ROOT_DIR.glob("*.zip")))
    assert zip_candidates, "No submission .zip found in submission/ or root directory"
    zip_path = zip_candidates[0]
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    assert zip_size_mb <= 50.0, f"ZIP exceeds 50 MB limit: {zip_size_mb:.2f} MB"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    print(f"ZIP {zip_path.name} ({zip_size_mb:.2f} MB) contains: {names}")
    assert f"{py_path.stem}.py" in names, f"Missing {py_path.stem}.py at ZIP root"
    assert f"{onnx_path.stem}.onnx" in names, f"Missing {onnx_path.stem}.onnx at ZIP root"
    return zip_path


def check_onnx_contract(onnx_path: pathlib.Path) -> None:
    print("=== 2. ONNX Model Schema & Single-Thread Check ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"ONNX Inputs: {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
    print(f"ONNX Outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")

def resolve_base_class_path() -> pathlib.Path:
    candidates = [
        DATA_DIR / "adm_base_class.py",
        ROOT_DIR / "adm_base_class.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list(ROOT_DIR.rglob("adm_base_class.py"))
    assert found, "Could not locate adm_base_class.py anywhere under techarena_work"
    return found[0]


def check_package_load_and_adm_base(zip_path: pathlib.Path, base_path: pathlib.Path):
    print("\n=== 3. Clean Package Load & ADM_Base Check ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)
        print(f"Extracted {zip_path.name} to temporary folder: OK")

        target_base = tmp_path / "adm_base_class.py"
        target_base.write_text(base_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Loaded ADM_Base definition from {base_path}: OK")

        sys.path.insert(0, str(tmp_path))
        try:
            for mod_name in ["adm_base_class", "submission_model"]:
                sys.modules.pop(mod_name, None)

            import adm_base_class
            import submission_model as sub_mod

            assert hasattr(sub_mod, "get_model") and callable(sub_mod.get_model), "get_model() missing"
            print("Found callable get_model() in submission_model.py: OK")

            model = sub_mod.get_model()
            assert isinstance(model, adm_base_class.ADM_Base), "get_model() must return an ADM_Base instance"
            assert hasattr(model, "run") and callable(model.run), "Model must implement run()"
            print(f"get_model() returned {type(model).__name__} (ADM_Base subclass): OK")

            src = inspect.getsource(sub_mod)
            flagged = [
                token
                for token in ["TechArena-Finland-2026", "phases_to_activity", "subject-info", "requests", "urllib"]
                if token in src
            ]
            if flagged:
                print(f"[WARN] Found references in submission_model.py: {flagged}")
            else:
                print("No external dataset paths or network helpers found in submission_model.py: OK")

            return model
        finally:
            if str(tmp_path) in sys.path:
                sys.path.remove(str(tmp_path))


def validate_output_df(out: pd.DataFrame, input_df: pd.DataFrame) -> None:
    assert isinstance(out, pd.DataFrame), "run() must return a pd.DataFrame"
    assert list(out.columns) == ["Activity"], f"Expected ['Activity'], got {list(out.columns)}"
    assert len(out) == len(input_df), f"Expected {len(input_df)} rows, got {len(out)}"
    assert not out["Activity"].isna().any(), "Output contains NaN labels"
    unique_vals = set(out["Activity"].unique())
    assert unique_vals.issubset(VALID_LABELS), f"Invalid labels produced: {unique_vals - VALID_LABELS}"


def run_end_to_end_tests_and_benchmark(model) -> None:
    print("\n=== 4. End-to-End Tests, Column/NaN Robustness & 50-Run MET ===")
    data_sub_dir = DATA_DIR / "data" if (DATA_DIR / "data").exists() else DATA_DIR
    parquet_files = sorted(data_sub_dir.rglob("*.parquet")) if data_sub_dir.exists() else []
    assert parquet_files, f"No .parquet files found under {data_sub_dir}"

    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]
    df0 = sample_dfs[0]

    out_a = model.run(df0)
    out_b = model.run(df0)
    validate_output_df(out_a, df0)
    validate_output_df(out_b, df0)
    pd.testing.assert_frame_equal(out_a, out_b)

    shuffled_cols = [c for c in reversed(SENSOR_COLS) if c in df0.columns] + [
        c for c in df0.columns if c not in SENSOR_COLS
    ]
    out_shuf = model.run(df0[shuffled_cols])
    validate_output_df(out_shuf, df0)
    print("Repeated calls and reversed sensor column order: PASSED")

    df_nan = df0.copy()
    for col in SENSOR_COLS:
        if col in df_nan.columns:
            df_nan.iloc[::7, df_nan.columns.get_loc(col)] = np.nan
    out_nan = model.run(df_nan)
    validate_output_df(out_nan, df_nan)
    print("Injected NaN robustness test: PASSED")

    _ = model.run(sample_dfs[0])

    times = []
    for i in range(50):
        df_curr = sample_dfs[i % len(sample_dfs)]
        t0 = time.perf_counter()
        out_curr = model.run(df_curr)
        dt = time.perf_counter() - t0
        validate_output_df(out_curr, df_curr)
        times.append(dt)

    arr = np.array(times, dtype=np.float64)
    met = float(np.mean(arr))
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    std = float(np.std(arr))
    speed_pts = max(0.0, min(20.0, (1.0 - met / 0.3) * 20.0))

    print(
        f"50-Run End-to-End MET: {met:.4f} s | Median: {med:.4f} s | "
        f"P95: {p95:.4f} s | Std: {std:.4f} s"
    )
    print(f"Estimated Speed Score (0.3s cap): {speed_pts:.2f} / 20.00")


def check_validation_and_efficiency(model) -> None:
    print("\n=== 5. Validation & Single-Core Efficiency Gate ===")
    data_sub_dir = DATA_DIR / "data" if (DATA_DIR / "data").exists() else DATA_DIR
    parquet_files = sorted(data_sub_dir.rglob("*.parquet")) if data_sub_dir.exists() else []
    assert parquet_files, "No .parquet files found for efficiency check"

    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:5]]
    _ = model.run(sample_dfs[0])

    latencies = []
    for i in range(50):
        df_in = sample_dfs[i % len(sample_dfs)]
        t0 = time.perf_counter()
        out = model.run(df_in)
        latencies.append(time.perf_counter() - t0)
        validate_output_df(out, df_in)

    tracemalloc.start()
    for i in range(50):
        df_in = sample_dfs[i % len(sample_dfs)]
        out = model.run(df_in)
        validate_output_df(out, df_in)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr = np.asarray(latencies, dtype=np.float64)
    met = float(np.mean(arr))
    p95 = float(np.percentile(arr, 95))
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    onnx_path = SUBMISSION_DIR / "submission_model.onnx"
    onnx_mb = onnx_path.stat().st_size / (1024.0 * 1024.0) if onnx_path.exists() else 0.0
    speed_pts = max(0.0, min(20.0, (1.0 - met / 0.3) * 20.0))

    print(f"50-Run MET: {met:.4f} s | P95: {p95:.4f} s | Peak RAM: {peak_mb:.2f} MB | ONNX: {onnx_mb:.2f} MB")
    print(f"Speed Score: {speed_pts:.2f} / 20.00")
    assert met < 0.30, f"MET {met:.4f} s >= 0.30 s threshold"
    assert peak_mb <= 4096.0, f"Peak RAM {peak_mb:.2f} MB > 4096 MB limit"
    assert onnx_mb <= 50.0, f"ONNX size {onnx_mb:.2f} MB > 50 MB limit"


def main() -> None:
    zip_path = check_files_and_zip()
    check_onnx_contract(SUBMISSION_DIR / "submission_model.onnx")
    base_path = resolve_base_class_path()
    model = check_package_load_and_adm_base(zip_path, base_path)
    run_end_to_end_tests_and_benchmark(model)
    check_validation_and_efficiency(model)
    print("\nALL AUDIT & END-TO-END CHECKS PASSED.")


if __name__ == "__main__":
    main()
