"""Build fold-0-excluded submission, then feed each held-out subject's sessions concatenated (one run() call)."""
import sys, json, subprocess, importlib.util, os
sys.path.insert(0, "lab"); sys.path.insert(0, "TechArena-Finland-2026")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from data import load_all
from sim3 import win_major, folds_subject
recs = load_all()
g = folds_subject(recs, 5, 0)
held = [i for i in range(len(recs)) if g[i] == 0]
cfg = json.loads(sys.argv[1]); cfg["exclude_subjects"] = sorted({recs[i]["subject"] for i in held}); tag = sys.argv[2]
subprocess.run([sys.executable, "lab/build_submission.py", f"_e2e{tag}", json.dumps(cfg)], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
spec = importlib.util.spec_from_file_location("e", f"submissions/_e2e{tag}/_e2e{tag}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.get_model()
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
for gap_min in (0, 30):
    Y, P = [], []
    for s in sorted({recs[i]["subject"] for i in held}):
        idx = [i for i in held if recs[i]["subject"] == s]
        parts, off, first = [], pd.Timedelta(0), recs[idx[0]]["df"].index[0]
        spans = []
        for i in idx:
            d = recs[i]["df"][COLS].copy()
            d.index = d.index - d.index[0] + first + off
            parts.append(d); spans.append((i, len(d)))
            off = d.index[-1] - first + pd.Timedelta(minutes=gap_min) + pd.Timedelta(seconds=10)
        big = pd.concat(parts)
        lab = m.run(big)["Activity"].to_numpy()
        pos = 0
        for i, n in spans:
            l = lab[pos:pos + n]; pos += n
            row = np.select([l == "Rest", l == "Stress", l == "Activity"], [0, 1, 2], -1)
            t = recs[i]["df"].index.values.astype("datetime64[ns]").view(np.int64)
            pw = win_major(t, row); yt = recs[i]["y"]; k = min(len(pw), len(yt)); mm = (yt[:k] >= 0) & (pw[:k] >= 0)
            Y.append(yt[:k][mm]); P.append(pw[:k][mm])
    print(tag, f"concat gap{gap_min}min: {f1_score(np.concatenate(Y), np.concatenate(P), average='macro'):.3f}", flush=True)
