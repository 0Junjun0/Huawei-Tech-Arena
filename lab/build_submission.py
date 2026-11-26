"""Train on all recordings and write a self-contained submission (<name>.py + <name>.onnx + <name>.zip).

usage: python lab/build_submission.py <name> --time|--notime
"""
import sys, pathlib, zipfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import lightgbm as lgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from data import load_all
from feats import compute

LAB = pathlib.Path(__file__).resolve().parent
OUT = LAB.parent / "submissions"

CONFIG = {
    True: dict(weights=(1.0, 1.0, 1.0), smooth=3),
    False: dict(weights=(1.0, 1.4, 0.8), smooth=3),
}

TEMPLATE = '''import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pathlib
import numpy as np
import pandas as pd
import onnxruntime as ort
from adm_base_class import ADM_Base

# ---- feature extraction (copied from lab/feats.py) ----
{feats}
# ---- model ----
LABELS = np.array(["Rest", "Stress", "Activity"], dtype=object)
FEAT_KW = {feat_kw}
USE_TIME = FEAT_KW.get("use_time", False)
CLASS_WEIGHTS = np.array({weights}, dtype=np.float32)
SMOOTH = {smooth}
SPLIT_GAP_NS = 600 * 1_000_000_000
GATE = {gate}  # (ratio, alpha): input-level protocol gate, or None
PROTO = {proto}  # logistic regression: P(input is an exercise session) from input aggregates, or None


FARES_DIM = {fares_dim}  # >0: append Fares' ver3 features (isolated module below)
_FARES = None
#__FARES_SRC__


def _fares_feats(tt, cols):
    """Fares' feature_matrix per contiguous segment, laid on our 10 s grid (gaps -> zeros)."""
    global _FARES
    if _FARES is None:
        import types
        _FARES = types.ModuleType("_fares")
        exec(compile(_FARES_SRC, "_fares", "exec"), _FARES.__dict__)
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(tt.view("datetime64[ns]")))
    n_win = int(tt[-1] // WIN_NS - tt[0] // WIN_NS) + 1
    out = np.zeros((n_win, FARES_DIM), dtype=np.float32)
    for rows, ids, start, F in getattr(_FARES, _FARES_FN)(df):
        off = int(start - tt[0] // WIN_NS)
        m = min(len(F), n_win - off)
        if m > 0:
            out[off:off + m] = F[:m]
    return out


def _proto_agg(X):
    B = X[:, :17].astype(np.float64)
    B[:, :4] = np.log1p(np.abs(B[:, :4])); B[:, 13] = np.log1p(np.abs(B[:, 13]))
    return np.concatenate([B.mean(0), B.std(0), np.percentile(B, 10, 0), np.percentile(B, 90, 0), [np.log1p(len(X))]])


def _proto_pe(X):
    z = (_proto_agg(X) - np.asarray(PROTO["mean"])) / np.asarray(PROTO["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(PROTO["coef"]) + PROTO["intercept"])))


def _smooth(P, k):
    if k <= 1:
        return P
    h = k // 2
    n = len(P)
    c = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


class TechArenaModel(ADM_Base):
    def load_model(self, onnx_file):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.prob_idx = [i for i, o in enumerate(self.session.get_outputs()) if "prob" in o.name.lower()][0]

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        n = len(input)
        if n == 0:
            return pd.DataFrame({{"Activity": pd.Series(dtype=object)}}, index=input.index)
        try:
            labels = self._predict(input)
        except Exception:
            labels = np.full(n, "Rest", dtype=object)
        return pd.DataFrame({{"Activity": pd.Series(labels, index=input.index, dtype=object, copy=False)}})

    def _segment(self, tt, cols):
        _, X, _ = compute(_Frame(tt, cols), **FEAT_KW)
        if FARES_DIM:
            Xf = np.zeros((len(X), FARES_DIM), dtype=np.float32)
            try:
                F = _fares_feats(tt, cols)
                n = min(len(F), len(X))
                Xf[:n] = F[:n]
            except Exception:
                pass
            X = np.hstack([X, Xf])
        P = np.asarray(self.session.run(None, {{self.input_name: X}})[self.prob_idx], dtype=np.float64)
        if PROTO is not None:
            pe = _proto_pe(X)
            P = P * np.array([1.0, (1.0 - pe) ** PROTO["gamma"], pe ** PROTO["gamma"]])
        if GATE is not None:
            # whole input looks like an exercise session -> no Stress, else no Activity
            if P[:, 2].mean() > GATE[0] * P[:, 1].mean():
                P[:, 1] *= GATE[1]
            else:
                P[:, 2] *= GATE[1]
        pred = (_smooth(P, SMOOTH) * CLASS_WEIGHTS).argmax(1)
        return LABELS[pred[tt // WIN_NS - tt[0] // WIN_NS]]

    def _predict(self, input):
        idx = input.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        t = np.asarray(idx, dtype="datetime64[ns]").view(np.int64)
        out = np.full(len(t), "Rest", dtype=object)
        cols = {{}}
        for c in ("BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"):
            if c in input.columns:
                col = input[c]
                if col.dtype.kind not in "fiub":
                    col = pd.to_numeric(col, errors="coerce")
                cols[c] = col.to_numpy(dtype=np.float64, na_value=np.nan)
        ok = t != np.iinfo(np.int64).min  # NaT
        if ok.all() and (t[1:] >= t[:-1]).all():
            pos, tt = None, t
        else:
            if not ok.any():
                return out
            pos = np.flatnonzero(ok)
            o = np.argsort(t[pos], kind="stable"); pos = pos[o]; tt = t[pos]
            cols = {{c: v[pos] for c, v in cols.items()}}
        # inputs holding several sessions separated by long gaps: process each segment on its own
        cuts = np.flatnonzero(np.diff(tt) > SPLIT_GAP_NS) + 1
        bounds = np.r_[0, cuts, len(tt)]
        lab = np.empty(len(tt), dtype=object)
        for a, b in zip(bounds[:-1], bounds[1:]):
            lab[a:b] = self._segment(tt[a:b], {{c: v[a:b] for c, v in cols.items()}} if len(bounds) > 2 else cols)
        if pos is None:
            return lab
        out[pos] = lab
        return out


class _Col:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def to_numpy(self, dtype=None):
        return self.a


class _Frame:
    """minimal DataFrame stand-in for compute(): .index (datetime64[ns]), .columns, [col].to_numpy()"""

    def __init__(self, t, cols):
        self.index = t.view("datetime64[ns]")
        self.columns = cols
        self._c = cols

    def __getitem__(self, c):
        return _Col(self._c[c])


def get_model() -> ADM_Base:
    m = TechArenaModel()
    m.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return m
'''


def main():
    """usage: build_submission.py <name> '<json config>'  config keys: feat_kw, weights, smooth, train_sc, lgb"""
    import json
    name = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].startswith("{"):
        cfg = json.loads(sys.argv[2])
    else:  # legacy flags
        use_time = "--time" in sys.argv
        cfg = dict(feat_kw=dict(use_time=use_time), **CONFIG[use_time])
    feat_kw = cfg["feat_kw"]
    recs = load_all()
    from sim3 import chunk_feats, train_xy
    keep = [i for i, r in enumerate(recs) if r["subject"] not in set(cfg.get("exclude_subjects", []))]
    train_sc = cfg.get("train_sc", ["full"])
    key = cfg.get("cache_key")
    fares_dim = 0
    if key:
        CF = chunk_feats(recs, key, lambda d: compute(d, **feat_kw)[1], train_sc)
        if cfg.get("fares"):
            import fares_feats as _ff
            CFf = chunk_feats(recs, cfg["fares"], _ff.fares_light if cfg.get("fares_light") else _ff.fares_feats, train_sc)
            fares_dim = CFf[train_sc[0]][0][0][3].shape[1]
            CF = {sc: [[(a, b, w0, np.hstack([X1, X2[:len(X1)]])) for (a, b, w0, X1), (_, _, _, X2) in zip(CF[sc][i], CFf[sc][i])]
                       for i in range(len(recs))] for sc in train_sc}
        sw = cfg.get("sc_weight")
        if sw:
            X, y, sample_w = train_xy(recs, CF, keep, train_sc, sw)
        else:
            X, y = train_xy(recs, CF, keep, train_sc); sample_w = None
    else:
        Xs, ys = [], []
        for r in [recs[i] for i in keep]:
            _, X, _ = compute(r["df"], **feat_kw)
            nn = min(len(X), len(r["y"]))
            Xs.append(X[:nn]); ys.append(r["y"][:nn])
        X = np.vstack(Xs); y = np.concatenate(ys); mm = y >= 0; X, y = X[mm], y[mm]; sample_w = None
    m = np.ones(len(y), bool)
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0,
                  verbose=-1, n_jobs=8, random_state=0)
    params.update(cfg.get("lgb", {}))
    proto = None
    if cfg.get("proto"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        exec(TEMPLATE.split("def _proto_agg")[1].split("def _proto_pe")[0].replace("{{", "{").replace("}}", "}").join(["def _proto_agg", ""]), globals())
        Z, yz = [], []
        for sc in train_sc:
            for i in keep:
                for a, b, w0, Xc in CF[sc][i]:
                    Z.append(_proto_agg(Xc)); yz.append(recs[i]["protocol"] != "STRESS")
        Z = np.array(Z); sc_ = StandardScaler().fit(Z); lr = LogisticRegression(C=cfg["proto"].get("C", 0.3), max_iter=5000).fit(sc_.transform(Z), yz)
        proto = dict(mean=sc_.mean_.tolist(), scale=sc_.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), gamma=cfg["proto"]["gamma"])
        print("proto LR train acc", lr.score(sc_.transform(Z), yz))
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, sample_weight=sample_w)
    onx = onnxmltools.convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
                                       zipmap=False, target_opset=15)
    # converter declares fixed batch dims on outputs -> make them dynamic (avoids ORT shape warnings)
    for o in onx.graph.output:
        dims = o.type.tensor_type.shape.dim
        if len(dims):
            dims[0].ClearField("dim_value"); dims[0].dim_param = "N"
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.onnx").write_bytes(onx.SerializeToString())
    feats_src = (LAB / "feats.py").read_text()
    feats_src = feats_src.split('"""', 2)[2]  # drop module docstring
    feats_src = "\n".join(l for l in feats_src.splitlines() if not l.startswith(("import numpy", "import pandas")))
    src = TEMPLATE.format(feats=feats_src, feat_kw=repr(feat_kw), weights=list(cfg["weights"]), fares_dim=fares_dim,
                          smooth=cfg["smooth"], gate=repr(tuple(cfg["gate"]) if cfg.get("gate") else None), proto=repr(proto))
    fsrc = ""
    if fares_dim:
        full = (LAB / "teammates" / "fares3.py").read_text()
        body = full.split("\nclass TechArenaModel")[0].replace("from adm_base_class import ADM_Base\n", "")
        body = body.replace("import onnxruntime as ort\n", "")
        if cfg.get("fares_light"):
            ff = (LAB / "fares_feats.py").read_text()
            light = ff[ff.index("def fares_light_matrix"):ff.index("def fares_light(")].replace("F.", "")
            body += "\n\n" + light + '''
def light_segments(frame):
    for rows, ids, start, section in recording_segments(frame):
        grid = (section.index.as_unit("ns").asi8, start, ids)
        yield rows, ids, start, fares_light_matrix(section, _grid=grid)
'''
        fsrc = "_FARES_SRC = " + repr(body) + "\n_FARES_FN = " + repr("light_segments" if cfg.get("fares_light") else "feature_segments")
    (d / f"{name}.py").write_text(src.replace("#__FARES_SRC__", fsrc))
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    with zipfile.ZipFile(d / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(d / f"{name}.py", f"{name}.py")
        z.write(d / f"{name}.onnx", f"{name}.onnx")
    np.save(d / "train_X_sample.npy", X[:2000])
    import joblib; joblib.dump(model, d / "lgbm.joblib")
    print(f"wrote {d}  features={X.shape[1]}  onnx={(d / f'{name}.onnx').stat().st_size/1024:.0f} KB  zip={(d / f'{name}.zip').stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
