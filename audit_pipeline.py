import os

# Enforce single-threaded execution before importing numpy/scipy/sklearn/onnxruntime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import inspect
import pathlib
import sys
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd
import onnxruntime as ort

WORK_DIR = pathlib.Path(__file__).resolve().parent
DATA_KIT_DIR = WORK_DIR / "TechArena-Finland-2026"
SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
VALID_LABELS = {"Stress", "Activity", "Rest"}


def check_files_and_zip() -> pathlib.Path:
    print("=== 1. File & ZIP Audit ===")
    py_path = WORK_DIR / "submission_model.py"
    onnx_path = WORK_DIR / "submission_model.onnx"
    base_path = DATA_KIT_DIR / "adm_base_class.py"
    if not base_path.exists() and (WORK_DIR / "adm_base_class.py").exists():
        base_path = WORK_DIR / "adm_base_class.py"

    assert py_path.exists(), "Missing submission_model.py"
    assert onnx_path.exists(), "Missing submission_model.onnx"
    assert base_path.exists(), "Missing adm_base_class.py"
    assert py_path.stem == onnx_path.stem, "Python and ONNX base names must match"

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Found {py_path.name} and {onnx_path.name} ({onnx_size_mb:.2f} MB)")

    zip_candidates = sorted(WORK_DIR.glob("*.zip"))
    assert zip_candidates, "No submission .zip found in working directory"
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
    print("\n=== 2. ONNX Model Schema & Single-Thread Check ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    assert len(inputs) == 1, f"Expected 1 ONNX input, got {len(inputs)}"
    inp = inputs[0]
    print(f"ONNX Input: name={inp.name}, shape={inp.shape}, type={inp.type}")
    print(f"ONNX Outputs: {[(o.name, o.shape, o.type) for o in outputs]}")
    assert "float" in inp.type.lower(), f"Expected float input tensor, got {inp.type}"

    n_feats = inp.shape[1] if (len(inp.shape) == 2 and isinstance(inp.shape[1], int)) else 44
    dummy = np.zeros((5, n_feats), dtype=np.float32)
    res = sess.run(None, {inp.name: dummy})
    assert len(res) >= 1 and len(res[0]) == 5, "Unexpected ONNX output shape"

def resolve_base_class_path() -> pathlib.Path:
    candidates = [
        DATA_KIT_DIR / "adm_base_class.py",
        WORK_DIR / "adm_base_class.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list(WORK_DIR.rglob("adm_base_class.py"))
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
    assert len(out) > 0, "Output DataFrame is empty"
    assert not out["Activity"].isna().any(), "Output contains NaN labels"
    unique_vals = set(out["Activity"].unique())
    assert unique_vals.issubset(VALID_LABELS), f"Invalid labels produced: {unique_vals - VALID_LABELS}"


def run_end_to_end_tests_and_benchmark(model) -> None:
    print("\n=== 4. End-to-End Tests, Column/NaN Robustness & 50-Run MET ===")
    parquet_files = sorted((DATA_KIT_DIR / "data").rglob("*.parquet")) if (DATA_KIT_DIR / "data").exists() else []
    assert parquet_files, "No .parquet files found under TechArena-Finland-2026/data"

    sample_dfs = [pd.read_parquet(p) for p in parquet_files[:10]]
    df0 = sample_dfs[0]

    # Repeated calls & shuffled column order check
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

    # Missing value stress test (injecting NaNs into sensor columns)
    df_nan = df0.copy()
    for col in SENSOR_COLS:
        if col in df_nan.columns:
            df_nan.iloc[::7, df_nan.columns.get_loc(col)] = np.nan
    out_nan = model.run(df_nan)
    validate_output_df(out_nan, df_nan)
    print("Injected NaN robustness test: PASSED")

    # Warm-up run (includes feature extraction + ONNX inference)
    _ = model.run(sample_dfs[0])

    # True end-to-end run() timing over 50 calls (includes Python feature extraction)
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


def main() -> None:
    zip_path = check_files_and_zip()
    check_onnx_contract(WORK_DIR / "submission_model.onnx")
    base_path = resolve_base_class_path()
    model = check_package_load_and_adm_base(zip_path, base_path)
    run_end_to_end_tests_and_benchmark(model)
    print("\nALL AUDIT & END-TO-END CHECKS PASSED.")


if __name__ == "__main__":
    main()