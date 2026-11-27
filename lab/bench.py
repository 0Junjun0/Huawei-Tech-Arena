"""MET like the grader's benchmark: first 10 parquet files (all columns), 50 run() calls. usage: bench.py <dir> <name>"""
import os, sys, time, pathlib, importlib.util
os.environ["OMP_NUM_THREADS"] = "1"
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "TechArena-Finland-2026"))
import numpy as np, pandas as pd
d, name = pathlib.Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location(name, d / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.get_model()
dfs = [pd.read_parquet(f) for f in sorted((root / "TechArena-Finland-2026" / "data").rglob("*.parquet"))[:10]]
m.run(dfs[0]); ts = []
for rep in range(int(sys.argv[3]) if len(sys.argv) > 3 else 1):
    for i in range(50):
        t = time.perf_counter(); m.run(dfs[i % 10]); ts.append(time.perf_counter() - t)
met = np.median([np.mean(ts[j:j+50]) for j in range(0, len(ts), 50)])
print(f"{name}: MET {met*1000:.1f} ms -> speed {max(0,(1-met/0.3)*20):.2f}/20")
