"""End-to-end check: build a submission excluding fold-0 subjects, run its real run() on held-out chunks, compare to sim.
usage: e2e.py '<config json>'"""
import sys, json, subprocess, importlib.util, pathlib, os
sys.path.insert(0, "lab"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np
from sklearn.metrics import f1_score
from data import load_all
from sim import chunks
from sim3 import win_major, folds_subject, EVAL_SC
recs = load_all()
g = folds_subject(recs, 5, 0)
held = [i for i in range(len(recs)) if g[i] == 0]
cfg = json.loads(sys.argv[1]); cfg["exclude_subjects"] = sorted({recs[i]["subject"] for i in held})
subprocess.run([sys.executable, "lab/build_submission.py", "_e2e", json.dumps(cfg)], check=True, env=dict(os.environ, OMP_NUM_THREADS="4"))
spec = importlib.util.spec_from_file_location("e2e", "submissions/_e2e/_e2e.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.get_model()
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
for sc in EVAL_SC:
    Y, P = [], []
    for i in held:
        df = recs[i]["df"][COLS]; t = df.index.values.astype("datetime64[ns]").view(np.int64)
        lab = np.empty(len(df), dtype=object)
        for a, b in chunks(len(df), t, sc, np.random.RandomState(1000 + i)):
            lab[a:b] = m.run(df.iloc[a:b])["Activity"].to_numpy()
        row = np.select([lab == "Rest", lab == "Stress", lab == "Activity"], [0, 1, 2], -1)
        pw = win_major(t, row); yt = recs[i]["y"]; n = min(len(pw), len(yt)); k = (yt[:n] >= 0) & (pw[:n] >= 0)
        Y.append(yt[:n][k]); P.append(pw[:n][k])
    print(f"e2e fold0 {sc}: {f1_score(np.concatenate(Y), np.concatenate(P), average='macro'):.3f}", flush=True)
