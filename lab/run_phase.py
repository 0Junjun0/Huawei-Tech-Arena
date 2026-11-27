"""Per (protocol, phase) accuracy of OOF whole-session predictions: where does each model family fail?"""
import sys, pickle, pathlib
sys.path.insert(0, "lab")
import numpy as np, pandas as pd
from data import load_all, files
from holdout import smooth_probs
recs = load_all(); C = pathlib.Path("lab/cache")
PER = pickle.loads((C / "scoring_per.pkl").read_bytes())
POST = {"rf": (3, (0.95, 1.8, 0.85)), "fares1": (7, (1, 1.4, 1)), "nxe": (5, (1, 1.4, 1)), "bebe_time": (3, (1, 1, 1)), "bebe_notime": (3, (1, 1, 1))}
pp = C / "phases.pkl"
if pp.exists(): PH = pickle.loads(pp.read_bytes())
else:
    PH = []
    for f in files():
        df = pd.read_parquet(f, columns=["Phase"]).sort_index()
        t = df.index.values.astype("datetime64[ns]").view(np.int64); wid = t // 10**10 - t[0] // 10**10
        ph = df["Phase"].astype(str).str.strip().str.lower().to_numpy()
        s = pd.Series(ph).groupby(wid).agg(lambda x: x.value_counts().index[0])
        out = np.full(int(wid[-1]) + 1, "none", dtype=object); out[s.index.to_numpy()] = s.to_numpy(); PH.append(out)
    pp.write_bytes(pickle.dumps(PH))
rows = []
for i, r in enumerate(recs):
    y = r["y"]; n = len(y)
    d = {"proto": r["protocol"], "stage": r["subject"][0], "phase": PH[i][:n], "y": y}
    for m, (sm, w) in POST.items():
        yy, P = PER[m][i]; q = (smooth_probs(P, sm) * np.asarray(w)).argmax(1); z = np.full(n, -1); z[:len(q)] = q; d[m] = z
    rows.append(pd.DataFrame(d))
D = pd.concat(rows); D = D[D.y >= 0]
D["phase"] = D.phase.str.replace(r"\d+ ?rpm", "rpm", regex=True)
pd.set_option("display.width", 250)
g = D.groupby(["proto", "stage", "phase"])
T = g.size().to_frame("n")
for m in POST:
    T[m] = g.apply(lambda x: (x[m] == x.y).mean()).round(2)
    for c, nm in ((0, "R"), (1, "S"), (2, "A")):
        if m in ("rf", "nxe"): T[f"{m}>{nm}"] = g.apply(lambda x: (x[m] == c).mean()).round(2)
print(T.to_string())
D.to_pickle(C / "phase_df.pkl")
