import os, sys, time, zipfile, tempfile, importlib.util, pathlib, warnings
os.environ["OMP_NUM_THREADS"] = "1"; warnings.filterwarnings("ignore")
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "TechArena-Finland-2026")); sys.path.insert(0, str(root / "lab"))
import numpy as np, pandas as pd
from data import files
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
dfs = [pd.read_parquet(f)[COLS].sort_index() for f in files()[:10]]
def load(name):
    tmp = pathlib.Path(tempfile.mkdtemp()); zipfile.ZipFile(root / "submissions" / name / f"{name}.zip").extractall(tmp)
    spec = importlib.util.spec_from_file_location(name, tmp / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.get_model()
for name in sys.argv[1:]:
    model = load(name); res = []
    for L in (None, 600, 300, 120):
        ins = [d if L is None else d.iloc[: np.searchsorted(d.index.values, d.index.values[0] + np.timedelta64(L, "s"))].copy() for d in dfs]
        model.run(ins[0]); ts = []
        for i in range(50):
            t = time.perf_counter(); model.run(ins[i % 10]); ts.append(time.perf_counter() - t)
        met = np.mean(ts); res.append(f"{'whole' if L is None else str(L)+'s'}: {met*1000:.0f} ms -> {max(0,(1-met/0.3)*20):.1f}")
    print(f"{name:14s}", " | ".join(res), flush=True)
