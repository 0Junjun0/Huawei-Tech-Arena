import sys, pickle, itertools
sys.path.insert(0, "lab")
import numpy as np
from data import load_all
import sim3
from sim3 import *
recs = load_all()
sim3._CFREF.update(chunk_feats(recs, "nxe", None, EVAL_SC))
L = lambda n: pickle.loads((CACHE / f"ocp_{n}_subject0.pkl").read_bytes())
A, B, C = L("x_nxe_sw07"), L("x_nxte_sw07"), L("lxbfe_light")
def mix(ops, ws):
    return {sc: [[(a, b, sum(w * o[sc][i][j][2] for o, w in zip(ops, ws))) for j, (a, b, _) in enumerate(ops[0][sc][i])] for i in range(len(recs))] for sc in EVAL_SC}
print("nxe      ", fmt(score_probs_fast(recs, A, (1, 1.4, 1.0), 5)))
print("nxe+nxte ", fmt(score_probs_fast(recs, mix([A, B], [.5, .5]), (1, 1.4, 1.0), 5)))
print("nxe+lxbfe", fmt(score_probs_fast(recs, mix([A, C], [.6, .4]), (1, 1.4, 1.0), 5)))
print("all3     ", fmt(score_probs_fast(recs, mix([A, B, C], [.4, .3, .3]), (1, 1.4, 1.0), 5)))
def viterbi_post(pen, w):
    lw = np.log(np.asarray(w))
    def post(P):
        L = np.log(P + 1e-9) + lw
        n = len(L); D = L[0].copy(); bp = np.zeros((n, 3), int)
        T = np.full((3, 3), -pen); np.fill_diagonal(T, 0)
        for t in range(1, n):
            M = D[:, None] + T; bp[t] = M.argmax(0); D = M.max(0) + L[t]
        s = np.zeros(n, int); s[-1] = D.argmax()
        for t in range(n - 1, 0, -1): s[t - 1] = bp[t, s[t]]
        return np.eye(3)[s]
    return post
for pen in (1, 2, 4, 8):
    for w in ((1, 1.4, 1.0), (1, 1.8, 1.0)):
        print("viterbi", pen, w, fmt(score_probs_fast(recs, A, (1, 1, 1), 1, post=viterbi_post(pen, w))))
