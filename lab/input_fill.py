"""In-sample F1 of shipped zips when the grader preprocesses missing values (ffill, interpolate, fillna(0), dropna...)."""
import sys, zipfile, tempfile, importlib.util, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
root = pathlib.Path(".").resolve(); sys.path.insert(0, str(root / "TechArena-Finland-2026")); sys.path.insert(0, str(root / "lab"))
from data import files, window_labels
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
def load(name):
    tmp = pathlib.Path(tempfile.mkdtemp()); zipfile.ZipFile(root / "submissions" / name / f"{name}.zip").extractall(tmp)
    s = importlib.util.spec_from_file_location(name, tmp / f"{name}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.get_model()
dfs = [pd.read_parquet(f).sort_index() for f in files()[::5]]
V = {"as-is": lambda d: d, "ffill": lambda d: d.ffill(), "ffill+bfill": lambda d: d.ffill().bfill(), "interpolate": lambda d: d.interpolate(limit_direction="both"),
     "fillna0": lambda d: d.fillna(0), "dropna(any)": lambda d: d.dropna(), "resample1s": lambda d: d.resample("1s").mean(),
     "resample4Hz": lambda d: d.resample("250ms").mean()}
for name in sys.argv[1:]:
    m = load(name); out = []
    for vn, fn in V.items():
        Y, P = [], []
        try:
            for df in dfs:
                inp = fn(df[COLS])
                if len(inp) == 0: continue
                lab = m.run(inp)["Activity"]
                ph = df["Phase"].reindex(inp.index) if vn.startswith(("resample", "dropna")) else df["Phase"]
                ref = pd.DataFrame({"Phase": ph.values}, index=inp.index)
                if vn.startswith("resample"):   # majority phase within each resampled bin
                    ref = pd.DataFrame({"Phase": df["Phase"].resample(inp.index.freq).agg(lambda x: x.value_counts().index[0] if len(x) else None).reindex(inp.index).values}, index=inp.index)
                y = window_labels(ref)
                pr = window_labels(pd.DataFrame({"Phase": lab.replace({"Activity": "sprint", "Stress": "stroop", "Rest": "rest"}).values}, index=inp.index))
                k = min(len(y), len(pr)); mm = (y[:k] >= 0) & (pr[:k] >= 0); Y.append(y[:k][mm]); P.append(pr[:k][mm])
            out.append(f"{vn}={f1_score(np.concatenate(Y), np.concatenate(P), average='macro'):.3f}")
        except Exception as e:
            out.append(f"{vn}=ERR({type(e).__name__})")
    print(f"{name:16s}", " ".join(out), flush=True)
