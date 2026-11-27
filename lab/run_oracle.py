import sys, pickle
sys.path.insert(0, "lab")
import numpy as np
from data import load_all
import sim3
from sim3 import *
recs = load_all()
OP = pickle.loads((CACHE / "ocp_x_nxe_sw07_subject0.pkl").read_bytes())
sim3._CFREF.update(chunk_feats(recs, "nxe", None, EVAL_SC))
# oracle: zero the impossible class per recording
OP2 = {sc: [[(a, b, P * (np.array([1, 0, 1]) if recs[i]["protocol"] != "STRESS" else np.array([1, 1, 0]))) for a, b, P in parts] for i, parts in enumerate(OP[sc])] for sc in OP}
print("oracle protocol", fmt(score_probs_fast(recs, OP2, (1, 1.4, 1.0), 5)))
# how often does the hard rule pick the wrong protocol, per scenario
for sc in EVAL_SC:
    wrong = tot = 0
    for i, parts in enumerate(OP[sc]):
        for a, b, P in parts:
            ex = P[:, 2].mean() > 2 * P[:, 1].mean()
            wrong += ex != (recs[i]["protocol"] != "STRESS"); tot += 1
    print(sc, f"protocol error {wrong}/{tot}")
