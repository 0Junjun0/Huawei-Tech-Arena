import sys, pickle
sys.path.insert(0, "lab")
import numpy as np
from data import load_all
import sim3
from sim3 import chunk_feats, score_probs_fast, EVAL_SC, TRAIN_SC, CACHE
recs = load_all()
SC = sorted(set(EVAL_SC) | set(TRAIN_SC))
CF = chunk_feats(recs, "c_nxe", None, SC); sim3._CFREF.update(CF)
for seed in (0, 1):
    A = pickle.loads((CACHE / f"ocp_cmp_oursnxe_{seed}_subject{seed}.pkl").read_bytes())
    B = pickle.loads((CACHE / f"ocp_cmp_fares3_{seed}_subject{seed}.pkl").read_bytes())
    C = pickle.loads((CACHE / f"ocp_cmp_both_{seed}_subject{seed}.pkl").read_bytes())
    for wa in (0.3, 0.5):
        M = {sc: [[(a, b, wa * Pa + (1 - wa) * Pb) for (a, b, Pa), (_, _, Pb) in zip(A[sc][i], B[sc][i])] for i in range(len(recs))] for sc in EVAL_SC}
        print(f"seed{seed} avg ours*{wa}+fares*{1-wa}: blend={score_probs_fast(recs, M, (1, 1.4, 1), 5)['blend']:.3f}", flush=True)
    M = {sc: [[(a, b, (Pa + Pb + Pc) / 3) for (a, b, Pa), (_, _, Pb), (_, _, Pc) in zip(A[sc][i], B[sc][i], C[sc][i])] for i in range(len(recs))] for sc in EVAL_SC}
    r = score_probs_fast(recs, M, (1, 1.4, 1), 5)
    print(f"seed{seed} avg all three: blend={r['blend']:.3f} chunk300={r['chunk300']:.3f} chunk600={r['chunk600']:.3f}", flush=True)
