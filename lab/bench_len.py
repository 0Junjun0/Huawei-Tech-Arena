"""per-call time vs input length for a submission. usage: bench_len.py <dir> <name>"""
import os, sys, time, pathlib, importlib.util
os.environ["OMP_NUM_THREADS"] = "1"
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "TechArena-Finland-2026"))
import numpy as np, pandas as pd
d, name = pathlib.Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location(name, d / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.get_model()
df = pd.read_parquet(sorted((root / "TechArena-Finland-2026" / "data").rglob("*.parquet"))[0])
res = []
for sec in (120, 300, 600, 1200, 99999):
    x = df.iloc[: sec * 64]; m.run(x); ts = []
    for _ in range(30):
        t = time.perf_counter(); m.run(x); ts.append(time.perf_counter() - t)
    res.append(f"{min(sec, len(df)//64)}s:{np.median(ts)*1000:.1f}ms")
print(name, " ".join(res))
