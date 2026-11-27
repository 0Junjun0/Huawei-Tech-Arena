import os, sys, zipfile, tempfile, importlib.util, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
root = pathlib.Path(".").resolve()
sys.path.insert(0, str(root / "TechArena-Finland-2026")); sys.path.insert(0, str(root / "lab"))
from data import files, window_labels
def load(name):
    tmp = pathlib.Path(tempfile.mkdtemp()); zipfile.ZipFile(root / "submissions" / name / f"{name}.zip").extractall(tmp)
    spec = importlib.util.spec_from_file_location(name, tmp / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod.get_model()
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
fs = files()[::6]
dfs = [pd.read_parquet(f) for f in fs]
print("pandas", pd.__version__, "index dtype", dfs[0].index.dtype, "monotonic", dfs[0].index.is_monotonic_increasing, "unique", dfs[0].index.is_unique, "cols", list(dfs[0].columns), dfs[0].dtypes.to_dict())
def variants(df):
    inp = df[COLS]
    yield "normal", inp
    x = inp.copy(); x.index = x.index.as_unit("us"); yield "index[us]", x
    x = inp.copy(); x.index = x.index.as_unit("ms"); yield "index[ms]", x
    x = inp.copy(); x.index = x.index.tz_localize("UTC"); yield "tz-aware", x
    yield "with Phase col", df
    x = inp.reset_index(); yield "index as column", x
    x = inp.astype("float32"); yield "float32", x
    yield "labelled rows only", inp[df["Phase"].map(lambda p: str(p).lower() not in ("undefined","nan","none")).values]
for name in sys.argv[1:]:
    m = load(name); res = {}
    for df in dfs:
        y = window_labels(df)
        for vn, inp in variants(df):
            try:
                out = m.run(inp)["Activity"].to_numpy()
                if len(out) != len(df):
                    # map back by position is impossible; score on its own grid
                    d2 = df.loc[inp.index] if not isinstance(inp.index, pd.RangeIndex) else df
                    yy = window_labels(d2)
                    pr = window_labels(pd.DataFrame({"Phase": pd.Series(out).replace({"Activity": "sprint", "Stress": "stroop", "Rest": "rest"}).values}, index=d2.index))
                else:
                    yy = y
                    pr = window_labels(pd.DataFrame({"Phase": pd.Series(out).replace({"Activity": "sprint", "Stress": "stroop", "Rest": "rest"}).values}, index=df.index))
                mm = yy >= 0; res.setdefault(vn, ([], []))[0].append(yy[mm]); res[vn][1].append(pr[mm])
            except Exception as e:
                res.setdefault(vn, ([], [])); res.setdefault("ERR " + vn, str(type(e).__name__) + ": " + str(e)[:80])
    for vn, v in res.items():
        if isinstance(v, str): print(f"  {name:14s} {vn}: {v}")
        elif v[0]: print(f"  {name:14s} {vn:20s} in-sample F1 {f1_score(np.concatenate(v[0]), np.concatenate(v[1]), average='macro'):.3f}")
