import pathlib, pandas as pd, numpy as np
root = pathlib.Path("TechArena-Finland-2026/data")
files = sorted(root.rglob("*.parquet"))
df = pd.read_parquet(files[0])
print(files[0]); print(df.dtypes); print(df.head(3)); print(df.index[:3], df.index.dtype)
print(df.notna().mean())
for f in [files[0], root/"STRESS"/"S05.parquet", root/"STRESS"/"f05.parquet"]:
    d = pd.read_parquet(f)
    ph = d["Phase"].fillna("NA")
    runs = (ph != ph.shift()).cumsum()
    seq = d.groupby(runs).agg(phase=("Phase","first"), start=("Phase", lambda s: s.index[0]), n=("Phase","size"))
    seq["dur_s"] = seq["n"]/64
    print("\n", f, "start", d.index[0], "dur min", (d.index[-1]-d.index[0]).total_seconds()/60)
    print(seq[["phase","dur_s"]].to_string())
