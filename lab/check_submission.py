"""Load a submission like the grader does (only zip contents + adm_base_class), check parity, scoring path and speed."""
import os, sys, time, zipfile, tempfile, importlib.util, pathlib
os.environ["OMP_NUM_THREADS"] = "1"
name = sys.argv[1]
root = pathlib.Path(__file__).resolve().parent.parent
tmp = pathlib.Path(tempfile.mkdtemp())
zipfile.ZipFile(root / "submissions" / name / f"{name}.zip").extractall(tmp)
sys.path.insert(0, str(root / "TechArena-Finland-2026"))
spec = importlib.util.spec_from_file_location(name, tmp / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sys.path.insert(0, str(root / "lab"))
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score
from data import files, window_labels
model = mod.get_model()
lgbm = joblib.load(root / "submissions" / name / "lgbm.joblib")
fs = files()
dfs = [pd.read_parquet(f).sort_index() for f in fs]
# 1) parity + 2) grader-style scoring (in-sample: checks plumbing, not generalisation)
maxdiff, ys, ps = 0.0, [], []
for df in dfs:
    inp = df[["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]]
    out = model.run(inp)
    assert out.index.equals(inp.index) and set(out["Activity"].unique()) <= {"Rest", "Stress", "Activity"}
    _, X, _ = mod.compute(inp, use_time=mod.USE_TIME)
    P_onnx = model.session.run(None, {model.input_name: X})[model.prob_idx]
    maxdiff = max(maxdiff, float(np.abs(P_onnx - lgbm.predict_proba(X)).max()))
    y = window_labels(df)
    pr = window_labels(pd.DataFrame({"Phase": out["Activity"].replace({"Activity": "sprint", "Stress": "stroop", "Rest": "rest"})}, index=df.index))
    m = y >= 0; ys.append(y[m]); ps.append(pr[m])
print(f"ONNX vs LightGBM max prob diff: {maxdiff:.2e}")
print(f"grader-style in-sample macro-F1 (plumbing check only): {f1_score(np.concatenate(ys), np.concatenate(ps), average='macro'):.4f}")
# shuffled-order + DataFrame with extra cols robustness
inp = dfs[0][["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]]
a = model.run(inp); b = model.run(inp.sample(frac=1, random_state=0)).reindex(inp.index)
print("order-invariant:", bool((a["Activity"] == b["Activity"]).all()))
# 3) speed, same as benchmark_run.py (first 10 files, 50 runs)
sample = [d[["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]] for d in dfs[:10]]
model.run(sample[0]); ts = []
for i in range(50):
    t = time.perf_counter(); model.run(sample[i % 10]); ts.append(time.perf_counter() - t)
met = np.mean(ts)
print(f"MET {met*1000:.1f} ms  max {max(ts)*1000:.1f} ms  -> speed score {max(0,(1-met/0.3)*20):.2f}/20")
big = max(dfs, key=len)[["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]]
t = time.perf_counter(); model.run(big); print(f"longest recording ({len(big)} rows): {(time.perf_counter()-t)*1000:.1f} ms")
