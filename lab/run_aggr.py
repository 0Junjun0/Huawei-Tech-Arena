"""Alternative score aggregations: per-recording mean macro-F1, per-protocol, per-subject, etc."""
import sys
sys.path.insert(0, "lab")
import numpy as np, pickle
from sklearn.metrics import f1_score
from data import load_all
from holdout import predict, CACHE
recs = load_all()
CFG = {"RF": dict(weights=(0.95, 1.8, 0.85), vote=True), "bebe_time": dict(smooth=3), "bebe_notime": dict(weights=(1, 1.4, 0.8), smooth=3), "local": dict(smooth=3)}
for name, pp in CFG.items():
    p = CACHE / f"oof_{name}_recording0.pkl"
    if not p.exists(): continue
    P, g = pickle.loads(p.read_bytes())
    pr = [predict(x, **pp) for x in P]
    yy = [r["y"][:len(x)] for r, x in zip(recs, pr)]
    def f(idx, labels=None):
        y = np.concatenate([yy[i] for i in idx]); q = np.concatenate([pr[i] for i in idx]); m = y >= 0
        return f1_score(y[m], q[m], average="macro", labels=labels)
    perrec = [f([i]) for i in range(len(recs))]
    perrec3 = [f([i], labels=[0, 1, 2]) for i in range(len(recs))]
    prot = {pt: f([i for i, r in enumerate(recs) if r["protocol"] == pt]) for pt in ("STRESS", "AEROBIC", "ANAEROBIC")}
    acc = np.mean(np.concatenate([(yy[i] == pr[i])[yy[i] >= 0] for i in range(len(recs))]))
    print(f"{name:12s} pooled {f(range(len(recs))):.3f} per-rec {np.mean(perrec):.3f} per-rec(3lab) {np.mean(perrec3):.3f} acc {acc:.3f} prot " + " ".join(f"{k}={v:.3f}" for k, v in prot.items()))
    print("   worst recs:", sorted([(round(perrec[i], 2), recs[i]["protocol"][:3] + recs[i]["subject"]) for i in range(len(recs))])[:8])
