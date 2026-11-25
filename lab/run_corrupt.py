"""Robustness to messy recordings (connection loss, dropouts, placement). RF (exact) vs a built submission config.
usage: run_corrupt.py '<config json>' tag   (5 subject folds, real run() of the built submission)"""
import sys, json, subprocess, importlib.util, os, pickle
sys.path.insert(0, "lab"); sys.path.insert(0, "TechArena-Finland-2026"); sys.path.insert(0, "submission")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data import load_all
from sim import chunks
from sim3 import win_major, folds_subject, chunk_feats
from holdout import vote3
import submission_model as tm
recs = load_all(); COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
cfg0 = json.loads(sys.argv[1]); tag = sys.argv[2]
g = folds_subject(recs, 5, 0)
RFCF = chunk_feats(recs, "rf", None, ["full"])

def corrupt(df, kind, rng):
    d = df.copy(); n = len(d); t = d.index
    def block():
        L = int(rng.uniform(180, 480) * 64); a = rng.randint(0, max(1, n - L)); return a, a + L
    if kind == "clean": return d
    if kind == "conn_loss":           # rows missing (device disconnected) twice
        keep = np.ones(n, bool)
        for _ in range(2): a, b = block(); keep[a:b] = False
        return d[keep]
    if kind == "nan_block":           # rows present, all sensors NaN twice
        for _ in range(2): a, b = block(); d.iloc[a:b] = np.nan
        return d
    if kind == "eda_temp_drop":       # EDA + Temperature missing for the 2nd half
        d.iloc[n // 2:, [1, 5]] = np.nan; return d
    if kind == "ibi_sparse":          # 80 % of beats lost (poor contact)
        m = d["IBI_duration"].notna().to_numpy(); idx = np.flatnonzero(m); drop = rng.rand(len(idx)) < 0.8
        d.iloc[idx[drop], [3, 4]] = np.nan; return d
    if kind == "placement":           # loose / wrong placement: BVP x3 + noise, EDA x0.3
        d["BVP"] = d["BVP"] * 3 + rng.randn(n) * 30; d["EDA"] = d["EDA"] * 0.3; return d
    if kind == "bvp_x3":
        d["BVP"] = d["BVP"] * 3; return d
    if kind == "bvp_x03":
        d["BVP"] = d["BVP"] * 0.3; return d
    if kind == "bvp_noise":
        d["BVP"] = d["BVP"] + rng.randn(n) * 30; return d
    if kind == "eda_x03":
        d["EDA"] = d["EDA"] * 0.3; return d
    if kind == "eda_x3":
        d["EDA"] = d["EDA"] * 3; return d
    if kind == "hr_shift":            # different baseline physiology
        d["HR"] = d["HR"] + 12; d["IBI_duration"] = d["IBI_duration"] * 0.88; return d
    raise ValueError(kind)

KINDS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["clean", "conn_loss", "nan_block", "eda_temp_drop", "ibi_sparse", "placement", "hr_shift"]
SCS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["full", "chunk300"]
res = {(m, k, s): ([], []) for m in ("RF", tag) for k in KINDS for s in SCS}
for fold in range(5):
    held = [i for i in range(len(recs)) if g[i] == fold]
    tr = [i for i in range(len(recs)) if g[i] != fold]
    cfg = dict(cfg0, exclude_subjects=sorted({recs[i]["subject"] for i in held}))
    name = f"_c{tag}{fold}"
    subprocess.run([sys.executable, "lab/build_submission.py", name, json.dumps(cfg)], check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
    spec = importlib.util.spec_from_file_location(name, f"submissions/{name}/{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ours = mod.get_model()
    X = np.vstack([RFCF["full"][i][0][3] for i in tr]); y = np.concatenate([recs[i]["y"][:len(RFCF["full"][i][0][3])] for i in tr]); mm = y >= 0
    rf = Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=2))]).fit(X[mm], y[mm])
    def rf_run(d):
        idx, F = tm._compute_window_features(d)
        P = np.zeros((len(F), 3)); P[:, list(rf.classes_)] = rf.predict_proba(F)
        pw = vote3((P * np.array([0.95, 1.8, 0.85])).argmax(1))
        s = pd.Series(pw, index=idx); return s.reindex(pd.DatetimeIndex(d.index).floor("10s")).fillna(0).to_numpy().astype(int)
    def our_run(d):
        l = ours.run(d)["Activity"].to_numpy(); return np.select([l == "Rest", l == "Stress", l == "Activity"], [0, 1, 2], -1)
    for i in held:
        for kind in KINDS:
            rng = np.random.RandomState(7 + i)
            dfc = corrupt(recs[i]["df"][COLS], kind, rng)
            tc = dfc.index.values.astype("datetime64[ns]").view(np.int64)
            t_full = recs[i]["df"].index.values.astype("datetime64[ns]").view(np.int64)
            for sc in SCS:
                for mname, fn in (("RF", rf_run), (tag, our_run)):
                    row = np.full(len(dfc), -1)
                    for a, b in chunks(len(dfc), tc, sc, np.random.RandomState(1000 + i)):
                        row[a:b] = fn(dfc.iloc[a:b])
                    # score on the original window grid (missing rows just have no prediction)
                    full_row = np.full(len(t_full), -1); pos = np.searchsorted(t_full, tc); full_row[pos] = row
                    pw = win_major(t_full, full_row); yt = recs[i]["y"]; n = min(len(pw), len(yt)); k = (yt[:n] >= 0) & (pw[:n] >= 0)
                    res[(mname, kind, sc)][0].append(yt[:n][k]); res[(mname, kind, sc)][1].append(pw[:n][k])
    subprocess.run(["rm", "-rf", f"submissions/{name}"])
    print("fold", fold, "done", flush=True)
for sc in SCS:
    for mname in ("RF", tag):
        print(sc, f"{mname:10s}", " ".join(f"{k}={f1_score(np.concatenate(res[(mname,k,sc)][0]), np.concatenate(res[(mname,k,sc)][1]), average='macro'):.3f}" for k in KINDS), flush=True)
