import os, sys, time; os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"TechArena-Finland-2026"); sys.path.insert(0,"submissions/bebe_time")
import numpy as np, pandas as pd, bebe_time as m
from data import files
sys.path.insert(0,"lab")
mod = m.get_model()
dfs=[pd.read_parquet(f)[["BVP","EDA","HR","IBI_dt","IBI_duration","Temperature"]] for f in files()[:10]]
T={"compute":0,"onnx":0,"rest":0}
for i in range(50):
    df=dfs[i%10]; t0=time.perf_counter(); _,X,_=m.compute(df,use_time=True); t1=time.perf_counter()
    mod.session.run(None,{mod.input_name:X}); t2=time.perf_counter(); mod.run(df); t3=time.perf_counter()
    T["compute"]+=t1-t0; T["onnx"]+=t2-t1; T["rest"]+=(t3-t2)-(t2-t0)
print({k: round(v/50*1000,1) for k,v in T.items()}, "ms per run")
import cProfile, pstats
cProfile.run("for i in range(20): m.compute(dfs[i%10], use_time=True)", "lab/cache/prof")
pstats.Stats("lab/cache/prof").sort_stats("tottime").print_stats(12)
