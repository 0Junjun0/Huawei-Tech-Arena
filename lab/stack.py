"""Stage-2 features built from stage-1 class probabilities over the recording timeline."""
import numpy as np

KS = (3, 7, 19, 61)


def _cum(P):
    return np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])


def _range_mean(c, lo, hi):
    n = np.maximum(hi - lo, 1)
    return (c[hi] - c[lo]) / n[:, None]


def stage2_features(P, t_extra=None):
    """P: (n_win, 3) stage-1 probabilities. Returns (n_win, F)."""
    n = len(P)
    c = _cum(P)
    i = np.arange(n)
    out = [P]
    for k in KS:
        h = k // 2
        out.append(_range_mean(c, np.clip(i - h, 0, n), np.clip(i + h + 1, 0, n)))
    for k in (6, 18, 60):  # one-sided: what came before / after
        out.append(_range_mean(c, np.clip(i - k, 0, n), i.clip(0, n)))
        out.append(_range_mean(c, np.clip(i + 1, 0, n), np.clip(i + 1 + k, 0, n)))
    # whole-recording context
    out.append(np.repeat(P.mean(0, keepdims=True), n, 0))
    out.append(np.repeat(np.percentile(P, 90, axis=0, keepdims=True), n, 0))
    # everything before / after this point
    out.append(_range_mean(c, np.zeros(n, int), i))
    out.append(_range_mean(c, i + 1, np.full(n, n)))
    pos = np.column_stack([i * 10.0, i / max(n - 1, 1), np.full(n, n * 10.0)])
    out.append(pos)
    if t_extra is not None:
        out.append(t_extra)
    return np.hstack(out).astype(np.float32)
