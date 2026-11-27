"""MET on whole sessions from the zip (50 runs over the first 10 files), plus feature/ONNX split if available."""
import os, sys, time, zipfile, tempfile, importlib.util, pathlib, warnings
os.environ["OMP_NUM_THREADS"] = "1"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "TechArena-Finland-2026")); sys.path.insert(0, str(root / "lab"))
from data import files
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
fs = files(); sel = fs[::9][:10]
sample = [pd.read_parquet(f)[COLS] for f in sel]
print("mean span min", np.mean([(d.index[-1] - d.index[0]).total_seconds() / 60 for d in sample]).round(1))
for name in sys.argv[1:]:
    tmp = pathlib.Path(tempfile.mkdtemp()); zipfile.ZipFile(root / "submissions" / name / f"{name}.zip").extractall(tmp)
    spec = importlib.util.spec_from_file_location(name, tmp / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.get_model(); m.run(sample[0]); ts = []
    for rep in range(3):
        for i in range(50):
            t = time.perf_counter(); m.run(sample[i % 10]); ts.append(time.perf_counter() - t)
    ts = np.array(ts).reshape(3, 50).mean(1)
    print(f"{name:20s} MET {ts.min()*1000:6.1f} ms (best of 3x50; all {np.round(ts*1000,1)}) -> speed {(1-ts.min()/0.3)*20:.2f}")
