"""Package an ensemble of the three shipped team models into ONE .py + ONE .onnx (+ zip).

Members (each used exactly as submitted, trained on all data):
  o_ = ours   submissions/robust_nxe_pg   (LightGBM, chunk-robust, protocol gate)  LB 73 (nxe_p)
  r_ = RF     claudes-bebe submission/    (teammate RF)                            LB 73
  f_ = Fares  fares_ver1                  (HistGradientBoosting)                   LB 74 (most likely)

  l_ = local  submissions/robust_lx_tuned (LightGBM on the purely local subset of our features) LB 74
               (its 121 features are columns of ours, so it only costs one more ONNX pass; needs w_ours > 0)

usage: python lab/build_ensemble.py <name> <w_ours> <w_rf> <w_fares> [<w_local>]
"""
import sys, pathlib, zipfile
import onnx
from onnx import compose, helper

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMBERS = {
    "o": (ROOT / "submissions/robust_nxe_pg/robust_nxe_pg.py", ROOT / "submissions/robust_nxe_pg/robust_nxe_pg.onnx"),
    "r": (ROOT / "lab/teammates/rf/submission_model.py", ROOT / "lab/teammates/rf/submission_model.onnx"),
    "f": (ROOT / "lab/teammates/fares1/submission_model.py", ROOT / "lab/teammates/fares1/submission_model.onnx"),
    "l": (None, ROOT / "submissions/robust_lx_tuned/robust_lx_tuned.onnx"),
}
LOCAL_KW = dict(use_time=False, local=True, extras=True)
LOCAL_W, LOCAL_SMOOTH = [1, 1.8, 1.2], 9

TEMPLATE = r'''"""Team ensemble: ours (robust_nxe_pg) + teammate RF + Fares ver1, probability average per 10 s window.
Each member's code is embedded unchanged and executed in its own namespace."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pathlib
import numpy as np
import pandas as pd
import onnxruntime as ort
from adm_base_class import ADM_Base

_SRC = {src}
_NS = {{}}
for _k, _s in _SRC.items():
    _g = {{"__name__": "member_" + _k}}
    exec(compile(_s, "member_" + _k, "exec"), _g)
    _NS[_k] = _g

WEIGHTS = {weights}
L_NAMES = {lnames}
L_W, L_SMOOTH = np.array({lw}, np.float64), {lsmooth}
LABELS = np.array(["Rest", "Stress", "Activity"], dtype=object)
W = 10_000_000_000


def _norm(Q):
    s = Q.sum(1, keepdims=True)
    return np.where(s > 0, Q / np.where(s > 0, s, 1), 1.0 / 3)


def _smooth(P, k):
    if k <= 1 or len(P) == 0:
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
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.inputs = {{i.name: i.shape[-1] for i in self.session.get_inputs()}}
        self.l_idx = None
        if "o" not in _NS:
            return
        o = _NS["o"]
        self.o_cw, self.o_smooth, self.o_proto = np.asarray(o["CLASS_WEIGHTS"], np.float64), o["SMOOTH"], o["PROTO"]

    # ---- members: each returns (n_win, 3) normalised scores on the 10 s grid of tt ----
    def _ours(self, df, tt, n_win, feeds):
        o = _NS["o"]
        _, X, names = o["compute"](df, **o["FEAT_KW"])
        feeds["o_input"] = X[:n_win].astype(np.float32)
        if WEIGHTS.get("l", 0) > 0:
            if self.l_idx is None:
                names = list(names)
                self.l_idx = np.array([names.index(n) for n in L_NAMES])
            feeds["l_input"] = feeds["o_input"][:, self.l_idx]
        def post(P):
            if self.o_proto is not None:
                pe = o["_proto_pe"](X[:n_win])
                P = P * np.array([1.0, (1.0 - pe) ** self.o_proto["gamma"], pe ** self.o_proto["gamma"]])
            return _norm(_smooth(P, self.o_smooth) * self.o_cw)
        return post

    def _rf(self, df, tt, n_win, feeds):
        r = _NS["r"]
        _, X = r["_compute_window_features"](df)
        X = np.asarray(X, np.float32)[:n_win]
        feeds["r_float_input"] = X
        return lambda P: _norm(_smooth(P, 3) * np.array([0.95, 1.8, 0.85]))

    def _fares(self, df, tt, n_win, feeds):
        f = _NS["f"]
        x, rows, clock = f["window_features"](df, "long")
        feeds["f_features"] = np.asarray(x, np.float32)
        off = clock - (tt[0] // W)
        def post(P):
            sums = np.vstack([np.zeros(3), np.cumsum(P, axis=0, dtype=float)])
            S = sums[clock.searchsorted(clock + 3, side="right")] - sums[clock.searchsorted(clock - 3)]
            S[:, 1] *= 1.4
            out = np.full((n_win, 3), 1.0 / 3)
            ok = (off >= 0) & (off < n_win)
            out[off[ok]] = _norm(S)[ok]
            return out
        return post

    def _segment(self, df, tt):
        n_win = int(tt[-1] // W - tt[0] // W) + 1
        feeds, posts = {{}}, {{}}
        for key, fn in (("o", self._ours), ("r", self._rf), ("f", self._fares)):
            if WEIGHTS.get(key, 0) <= 0:
                continue
            try:
                posts[key] = fn(df, tt, n_win, feeds)
            except Exception:
                pass
        if not posts:
            return np.zeros(n_win, dtype=int)
        for name, dim in self.inputs.items():  # members that failed still need a (dummy) feed
            if name not in feeds:
                feeds[name] = np.zeros((1, dim), np.float32)
        if "l_input" in feeds and "o" in posts:
            posts["l"] = lambda P: _norm(_smooth(P, L_SMOOTH) * L_W)
        present = [k for k in ("o", "r", "f", "l") if WEIGHTS.get(k, 0) > 0]
        outs = self.session.run([k + "_probabilities" for k in present], feeds)
        total = np.zeros((n_win, 3))
        for key, P in zip(present, outs):
            if key in posts:
                try:
                    Q = posts[key](np.asarray(P, np.float64))
                    if Q.shape == (n_win, 3):
                        total += WEIGHTS[key] * Q
                except Exception:
                    pass
        return total.argmax(1)

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        n = len(input)
        if n == 0:
            return pd.DataFrame({{"Activity": pd.Series(dtype=object)}}, index=input.index)
        try:
            df = input
            idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert(None)
            if not idx.equals(df.index):
                df = df.copy(); df.index = idx
            order = None
            if not idx.is_monotonic_increasing:
                order = np.argsort(np.asarray(idx, dtype="datetime64[ns]").view(np.int64), kind="stable")
                df = df.iloc[order]
            tt = np.asarray(df.index, dtype="datetime64[ns]").view(np.int64)
            # several sessions separated by long gaps -> each on its own
            cuts = np.r_[0, np.flatnonzero(np.diff(tt) > 600 * 1_000_000_000) + 1, len(tt)]
            lab = np.empty(len(tt), dtype=object)
            for a, b in zip(cuts[:-1], cuts[1:]):
                seg_t = tt[a:b]
                pred = self._segment(df.iloc[a:b], seg_t)
                lab[a:b] = LABELS[pred[np.clip(seg_t // W - seg_t[0] // W, 0, len(pred) - 1)]]
            if order is not None:
                out = np.empty_like(lab); out[order] = lab; lab = out
        except Exception:
            lab = np.full(n, "Rest", dtype=object)
        return pd.DataFrame({{"Activity": pd.Series(lab, index=input.index, dtype=object, copy=False)}})


def get_model() -> ADM_Base:
    m = TechArenaModel()
    m.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return m
'''


def merged_onnx(keys):
    graphs, opsets = [], {}
    for key, (_, op) in MEMBERS.items():
        if key not in keys:
            continue
        m = onnx.load(str(op))
        m = compose.add_prefix(m, prefix=key + "_")
        graphs.append(m.graph)
        for o in m.opset_import:
            opsets[o.domain] = max(opsets.get(o.domain, 0), o.version)
    opsets[""] = max(opsets.get("", 0), 15)
    g = helper.make_graph(
        nodes=[n for gr in graphs for n in gr.node], name="team_ensemble",
        inputs=[i for gr in graphs for i in gr.input], outputs=[o for gr in graphs for o in gr.output],
        initializer=[t for gr in graphs for t in gr.initializer])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main():
    name = sys.argv[1]
    wo, wr, wf = (float(x) for x in sys.argv[2:5])
    w = {"o": wo, "r": wr, "f": wf, "l": float(sys.argv[5]) if len(sys.argv) > 5 else 0.0}
    keys = [k for k in MEMBERS if w[k] > 0]
    assert w["l"] <= 0 or w["o"] > 0, "the local member reuses our feature matrix"
    src = {k: p.read_text() for k, (p, _) in MEMBERS.items() if k in keys and p is not None}
    lnames = []
    if w["l"] > 0:
        sys.path.insert(0, str(ROOT / "lab"))
        import numpy as np, pandas as pd
        from feats import compute
        d = pd.DataFrame({c: [0.0] for c in ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]}, index=pd.to_datetime(["2020-01-01"]))
        lnames = list(compute(d, **LOCAL_KW)[2])
    # members import their base class themselves; the grader provides adm_base_class
    d = ROOT / "submissions" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.py").write_text(TEMPLATE.format(src=repr(src), weights=repr(w), lnames=repr(lnames), lw=repr(LOCAL_W), lsmooth=LOCAL_SMOOTH))
    onnx.save(merged_onnx(keys), str(d / f"{name}.onnx"))
    with zipfile.ZipFile(d / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(d / f"{name}.py", f"{name}.py"); z.write(d / f"{name}.onnx", f"{name}.onnx")
    print(f"wrote {d}: onnx {(d / f'{name}.onnx').stat().st_size / 1e6:.1f} MB, zip {(d / f'{name}.zip').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
