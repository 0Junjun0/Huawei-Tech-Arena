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
USE_TIME = {use_time}
CLASS_WEIGHTS = np.array({weights}, dtype=np.float32)
SMOOTH = {smooth}


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
        if len(input) == 0:
            return pd.DataFrame({{"Activity": pd.Series(dtype=object)}}, index=input.index)
        df = input
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        order = None
        if not df.index.is_monotonic_increasing:
            order = np.argsort(df.index.values, kind="stable")
            df = df.iloc[order]
        _, X, _ = compute(df, use_time=USE_TIME)
        P = np.asarray(self.session.run(None, {{self.input_name: X}})[self.prob_idx], dtype=np.float64)
        pred = (_smooth(P, SMOOTH) * CLASS_WEIGHTS).argmax(1)
        _, wid, _, _ = window_grid(df.index)
        labels = LABELS[pred[wid]]
        if order is not None:
            out = np.empty_like(labels)
            out[order] = labels
            labels = out
        return pd.DataFrame({{"Activity": labels}}, index=input.index)


def get_model() -> ADM_Base:
    m = TechArenaModel()
    m.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return m
'''


def main():
    name = sys.argv[1]
    use_time = "--time" in sys.argv
    cfg = CONFIG[use_time]
    recs = load_all()
    Xs, ys = [], []
    for r in recs:
        _, X, names = compute(r["df"], use_time=use_time)
        n = min(len(X), len(r["y"]))
        Xs.append(X[:n]); ys.append(r["y"][:n])
    X = np.vstack(Xs); y = np.concatenate(ys); m = y >= 0
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", reg_lambda=1.0,
                               verbose=-1, n_jobs=8, random_state=0)
    model.fit(X[m], y[m])
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
    (d / f"{name}.py").write_text(TEMPLATE.format(feats=feats_src, use_time=use_time, weights=list(cfg["weights"]),
                                                  smooth=cfg["smooth"]))
    with zipfile.ZipFile(d / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(d / f"{name}.py", f"{name}.py")
        z.write(d / f"{name}.onnx", f"{name}.onnx")
    np.save(d / "train_X_sample.npy", X[:2000])
    import joblib; joblib.dump(model, d / "lgbm.joblib")
    print(f"wrote {d}  features={X.shape[1]}  onnx={(d / f'{name}.onnx').stat().st_size/1024:.0f} KB  zip={(d / f'{name}.zip').stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
