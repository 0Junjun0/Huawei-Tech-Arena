"""Short / degenerate input tests for a feature fn or a submission model. usage: imported."""
import numpy as np, pandas as pd
COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
def cases(df):
    t0 = df.index[0] + pd.Timedelta(minutes=10)
    out = {"1row": df.iloc[[5000]]}
    for s in (10, 30, 90):
        out[f"{s}s"] = df[(df.index >= t0) & (df.index < t0 + pd.Timedelta(seconds=s))]
    nan = df.iloc[20000:30000].copy(); nan[["EDA", "Temperature", "HR", "IBI_dt", "IBI_duration"]] = np.nan
    out["90s_only_bvp"] = nan.iloc[:64 * 90]
    allnan = df.iloc[:640].copy(); allnan[:] = np.nan; out["allnan"] = allnan
    out["shuffled"] = df.iloc[:20000].sample(frac=1, random_state=0)
    g = df.iloc[:60000]; out["gap"] = pd.concat([g.iloc[:20000], g.iloc[40000:]])  # 5 min connection loss
    out["dup_ts"] = pd.concat([df.iloc[:3000], df.iloc[2000:4000]]).sort_index()
    two = df.iloc[:30000].copy(); two2 = df.iloc[30000:60000].copy(); two2.index = two2.index + pd.Timedelta(hours=2)
    out["two_sessions_gap"] = pd.concat([two, two2])
    out["str_index"] = df.iloc[:5000].copy(); out["str_index"].index = out["str_index"].index.astype(str)
    return out
def run_model(model, df):
    for k, d in cases(df[COLS]).items():
        o = model.run(d)
        assert len(o) == len(d) and o.index.equals(d.index), k
        assert set(o["Activity"].unique()) <= {"Rest", "Stress", "Activity"}, (k, o["Activity"].unique())
        assert o["Activity"].notna().all(), k
    return True
