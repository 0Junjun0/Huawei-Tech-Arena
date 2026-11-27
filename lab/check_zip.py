"""Grader-style check using only the zip: load from a clean folder, run edge cases, 300 s chunks, speed."""
import os, sys, time, zipfile, tempfile, importlib.util, pathlib, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
name = sys.argv[1]
root = pathlib.Path(__file__).resolve().parent.parent
tmp = pathlib.Path(tempfile.mkdtemp())
zipfile.ZipFile(root / "submissions" / name / f"{name}.zip").extractall(tmp)
print("zip contents:", sorted(p.name for p in tmp.iterdir()))
sys.path.insert(0, str(root / "TechArena-Finland-2026"))
spec = importlib.util.spec_from_file_location(name, tmp / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, str(root / "lab"))
from data import files, window_labels
model = mod.get_model()
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
dfs = [pd.read_parquet(f).sort_index() for f in files()]
ok = lambda o, i: o.index.equals(i.index) and set(pd.unique(o["Activity"])) <= {"Rest", "Stress", "Activity"}
d0 = dfs[0][COLS]
cases = {"1 row": d0.iloc[:1], "10 s": d0.iloc[:640], "30 s": d0.iloc[:1920], "90 s": d0.iloc[:5760],
         "all NaN": d0.iloc[:6400] * np.nan, "shuffled": d0.sample(frac=1, random_state=0), "BVP only": d0[["BVP"]]}
for k, d in cases.items():
    try: o = model.run(d); print(f"  {k:9s}", "OK" if ok(o, d) else "BAD OUTPUT")
    except Exception as e: print(f"  {k:9s} CRASH {type(e).__name__}: {e}")
# in-sample plumbing check, whole sessions and 300 s chunks
for chunk in (None, 300):
    ys, ps = [], []
    for df in dfs:
        inp = df[COLS]
        if chunk is None: out = model.run(inp)["Activity"].to_numpy()
        else:
            t = inp.index.values.astype("datetime64[ns]").view(np.int64)
            edges = np.searchsorted(t, np.arange(t[0], t[-1] + chunk * 10**9, chunk * 10**9))
            out = np.concatenate([model.run(inp.iloc[a:b])["Activity"].to_numpy() for a, b in zip(edges[:-1], edges[1:]) if b > a])
        y = window_labels(df)
        pr = window_labels(pd.DataFrame({"Phase": pd.Series(out).replace({"Activity": "sprint", "Stress": "stroop", "Rest": "rest"}).values}, index=df.index))
        m = y >= 0; ys.append(y[m]); ps.append(pr[m])
    print(f"in-sample macro-F1 ({'whole' if chunk is None else f'{chunk}s chunks'}; plumbing only): {f1_score(np.concatenate(ys), np.concatenate(ps), average='macro'):.3f}")
sample = [d[COLS] for d in dfs[:10]]
model.run(sample[0]); ts = []
for i in range(50):
    t = time.perf_counter(); model.run(sample[i % 10]); ts.append(time.perf_counter() - t)
met = np.mean(ts); print(f"MET {met*1000:.1f} ms (whole sessions) -> speed {max(0,(1-met/0.3)*20):.2f}/20")
