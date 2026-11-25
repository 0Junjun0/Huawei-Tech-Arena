"""Cache OOF probabilities (recording folds, 7 folds, seed s) per model config."""
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from holdout import *
def oof_cached(recs, name, X, mk, seed=0, split="recording"):
    p = CACHE / f"oof_{name}_{split}{seed}.pkl"
    if p.exists(): return pickle.loads(p.read_bytes())
    g = folds_recording(recs, 7, seed) if split == "recording" else folds_subject(recs, 5, seed)
    P = oof_probs(recs, X, mk, g)
    p.write_bytes(pickle.dumps((P, g)))
    return P, g
