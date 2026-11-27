"""One fast model that copies the blend: LightGBM on our features trained on the blend's out-of-fold probabilities
(soft labels, as weighted duplicated rows), optionally mixed with the true labels.
usage: run_distill.py            -> CV comparison
       run_distill.py build NAME ALPHA -> train on all recordings, write submissions/NAME (uses robust_nxe_pg's code)"""
import sys, json, pickle, pathlib, zipfile
sys.path.insert(0, "lab")
import numpy as np, lightgbm as lgb
from sklearn.metrics import f1_score
from data import load_all
from holdout import folds_subject, smooth_probs
recs = load_all(); C = pathlib.Path("lab/cache")
X = [p[0][3] for p in pickle.loads((C / "cf_e_nxe_full.pkl").read_bytes())]
PER = pickle.loads((C / "scoring_per.pkl").read_bytes())
POST = {"lx": (9, (1, 1.8, 1.2)), "nxe": (5, (1, 1.4, 1)), "fares1": (7, (1, 1.4, 1))}
def q(name, i):
    y, P = PER[name][i]; sm, w = POST[name]; Q = smooth_probs(P, sm) * np.asarray(w); return Q / Q.sum(1, keepdims=True)
N = [min(len(recs[i]["y"]), len(X[i]), *(len(PER[m][i][0]) for m in POST)) for i in range(len(recs))]
T = [sum(q(m, i)[:N[i]] for m in POST) / 3 for i in range(len(recs))]          # teacher = OOF blend
Y = [recs[i]["y"][:N[i]] for i in range(len(recs))]
def mk():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, n_jobs=8)
def fit(idx, alpha):
    """alpha = share of the true label in the target (1 = ordinary training, 0 = pure copy of the blend)."""
    Xt = np.vstack([X[i][:N[i]] for i in idx]); yt = np.concatenate([Y[i] for i in idx]); Tt = np.vstack([T[i] for i in idx]); m = yt >= 0
    Xt, yt, Tt = Xt[m], yt[m], Tt[m]
    tgt = alpha * np.eye(3)[yt] + (1 - alpha) * Tt
    tgt = tgt * (len(tgt) / (3 * tgt.sum(0)))                                    # balance classes
    Xd = np.vstack([Xt] * 3); yd = np.repeat([0, 1, 2], len(Xt)); wd = tgt.T.reshape(-1)
    k = wd > 1e-4
    return mk().fit(Xd[k], yd[k], sample_weight=wd[k])
def score(P, sm, w):
    a, b = [], []
    for i in P:
        pr = (smooth_probs(P[i], sm) * np.asarray(w)).argmax(1); m = Y[i] >= 0; a.append(Y[i][m]); b.append(pr[m])
    return f1_score(np.concatenate(a), np.concatenate(b), average="macro")
if len(sys.argv) > 1 and sys.argv[1] == "build":
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    name, alpha = sys.argv[2], float(sys.argv[3]); sm, w = int(sys.argv[4]), json.loads(sys.argv[5])
    model = fit(range(len(recs)), alpha)
    onx = onnxmltools.convert_lightgbm(model, initial_types=[("input", FloatTensorType([None, X[0].shape[1]]))], zipmap=False, target_opset=15)
    for o in onx.graph.output:
        if o.type.tensor_type.shape.dim: o.type.tensor_type.shape.dim[0].dim_param = "N"
    d = pathlib.Path("submissions") / name; d.mkdir(exist_ok=True)
    (d / f"{name}.onnx").write_bytes(onx.SerializeToString())
    src = open("submissions/robust_nxe_pg/robust_nxe_pg.py", encoding="utf-8").read()
    import re
    src, n1 = re.subn(r"^CLASS_WEIGHTS = .*$", f"CLASS_WEIGHTS = np.array({w}, dtype=np.float32)", src, flags=re.M)
    src, n2 = re.subn(r"^SMOOTH = .*$", f"SMOOTH = {sm}", src, flags=re.M)
    src, n3 = re.subn(r"^PROTO = .*$", "PROTO = None  # the student already learnt the gated blend", src, flags=re.M)
    assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3)
    (d / f"{name}.py").write_text(src, encoding="utf-8")
    with zipfile.ZipFile(d / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(d / f"{name}.py", f"{name}.py"); z.write(d / f"{name}.onnx", f"{name}.onnx")
    print("wrote", d); sys.exit()
g = folds_subject(recs, 5, 0)
print("teacher (OOF blend of lx + nxe + fares1):", round(score({i: T[i] for i in range(len(recs))}, 1, (1, 1, 1)), 3))
for alpha in (1.0, 0.5, 0.25, 0.0):
    P = {}
    for k in np.unique(g):
        mod = fit(np.flatnonzero(g != k), alpha)
        for i in np.flatnonzero(g == k): P[i] = mod.predict_proba(X[i][:N[i]])
    res = {(sm, w): score(P, sm, w) for sm in (1, 3, 5, 9) for w in ((1, 1, 1), (1, 1.4, 1), (1, 1.8, 1))}
    best = max(res, key=res.get)
    print(f"student alpha={alpha}: smooth5 w(1,1.4,1) {res[(5, (1, 1.4, 1))]:.3f} | smooth1 w1 {res[(1, (1, 1, 1))]:.3f} | best {best} {res[best]:.3f}", flush=True)
