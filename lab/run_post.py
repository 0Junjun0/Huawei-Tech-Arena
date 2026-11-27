"""input-level protocol prior post-processing. usage: run_post.py <ocp cache name> <key>"""
import sys, pickle, itertools
sys.path.insert(0, "lab")
import numpy as np
from data import load_all
import sim3
from sim3 import *
recs = load_all()
name, key = sys.argv[1], sys.argv[2]
OP = pickle.loads((CACHE / f"ocp_{name}_subject0.pkl").read_bytes())
sim3._CFREF.update(chunk_feats(recs, key, None, EVAL_SC))
def mk(r, alpha, mode):
    def post(P):
        sa, ss = P[:, 2].mean(), P[:, 1].mean()
        Q = P.copy()
        if mode == "hard":
            if sa > r * ss: Q[:, 1] *= alpha
            else: Q[:, 2] *= alpha
        else:  # soft: odds of exercise session
            z = sa / (sa + r * ss + 1e-9)          # in [0,1], 1 = exercise
            Q[:, 1] *= alpha ** z; Q[:, 2] *= alpha ** (1 - z)
        return Q
    return post
res = []
for sm, ws, wa in [(9, 1.4, 0.8), (9, 1.8, 1.2), (5, 1.4, 1.0)]:
    res.append((round(score_probs_fast(recs, OP, (1, ws, wa), sm)["blend"], 4), "none", sm, ws, wa))
    for mode, r, alpha in itertools.product(("hard", "soft"), (0.5, 1, 2, 4), (0.0, 0.1, 0.3)):
        res.append((round(score_probs_fast(recs, OP, (1, ws, wa), sm, post=mk(r, alpha, mode))["blend"], 4), mode, r, alpha, sm, ws, wa))
res.sort(key=lambda x: -x[0])
for x in res[:8]: print(x)
print([x for x in res if x[1] == "none"])
b = res[0]
if b[1] != "none":
    print("row-level best:", fmt(score_probs(recs, OP, (1, b[5], b[6]), b[4], post=mk(b[2], b[3], b[1]))))
