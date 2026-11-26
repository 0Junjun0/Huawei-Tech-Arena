"""Fit the Huawei wrist model and calibrate it on participant-held-out predictions.
The training data include CATSA v2 by Mohammod Abdul Motin, Md Santo Ali,
Sapnil Sarker Bipro, Meftahul Jannat Nuha, Abid Hasan and Sumaiya Kabir.
Baseline and Stroop signals were converted to model features and fitted parameters.
https://doi.org/10.6084/m9.figshare.28690988.v2
https://creativecommons.org/licenses/by/4.0/
Scoring exported models needs the Huawei runtime requirements. Fitting also requires CPU PyTorch, onnx and skl2onnx.
"""

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from submission_model import (
    BIN_NS,
    TechArenaModel,
    adaptive_features,
    autonomic_detail,
    beat_details,
    beat_series,
    cardiac_design,
    cardiac_response,
    local_response_context,
    meta_design,
    physiology_windows,
    residual_prob,
    response_design,
    series_features,
    short_features,
    short_kernel,
    smooth_activity,
    temporal_segments,
    transfer_features,
)

PHASES = {
    "baseline": 0,
    "rest": 0,
    "stroop": 1,
    "tmct": 1,
    "real opinion": 1,
    "opposite opinion": 1,
    "subtraction": 1,
    "warm-up": 2,
    "sprint": 2,
    "cooldown": 2,
}

CHANNELS = ("BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature")

PAIRED_OUTAGES = {
    "both_ibi": ("IBI_duration", "IBI_dt"),
    "optical_loss": ("HR", "BVP", "IBI_duration", "IBI_dt"),
    "eda_temperature": ("EDA", "Temperature"),
    "hr_and_ibi": ("HR", "IBI_duration", "IBI_dt"),
}

WHOLE_OUTAGES = tuple(("without_" + name for name in CHANNELS))

PARTIAL_OUTAGES = tuple(("partial_" + name for name in CHANNELS))

TEST_OUTAGES = (
    *WHOLE_OUTAGES,
    *PARTIAL_OUTAGES,
    *PAIRED_OUTAGES,
    "scattered_BVP",
    "rotating_gaps",
)


def recording_rng(seed, recording, scenario):
    key = f"{seed}/{recording}/{scenario}".encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    )


def time_gaps(clock, rng, short, long):
    """Two disjoint time regions when the recording length permits it."""
    selected = np.zeros(len(clock), dtype=bool)
    for part in np.array_split(np.arange(len(clock)), 2):
        if len(part):
            width = min(len(part), int(rng.integers(short, long + 1)))
            j = int(rng.integers(0, max(1, len(part) - width + 1)))
            start = clock[part[j]]
            selected |= (clock >= start) & (clock < start + width)
    return selected


def perturb(frame, scenario, rng):
    """Apply sensor loss in raw time, returning the bins targeted by the simulation."""
    raw = frame.index.as_unit("ns").asi8 // 10000000000
    clock, groups = np.unique(raw, return_inverse=True)
    touched = np.ones(len(clock), dtype=bool)
    if scenario.startswith("without_") or scenario in PAIRED_OUTAGES:
        names = (
            PAIRED_OUTAGES[scenario] if scenario in PAIRED_OUTAGES else (scenario[8:],)
        )
        return (frame.assign(**{name: np.nan for name in names}), touched)
    result = frame.copy(deep=False)
    if scenario.startswith("partial_"):
        name = scenario[8:]
        touched = time_gaps(clock, rng, 8, 18)
        result[name] = frame[name].mask(touched[groups])
    elif scenario == "rotating_gaps":
        touched = np.zeros(len(clock), dtype=bool)
        for name in CHANNELS:
            block = time_gaps(clock, rng, 5, 16)
            result[name] = frame[name].mask(block[groups])
            touched |= block
    elif scenario == "scattered_BVP":
        result["BVP"] = frame.BVP.mask(rng.random(len(frame)) < 0.06)
    else:
        raise ValueError(f"Unknown sensor outage: {scenario}")
    return (result, touched)


def labelled_windows(frame, positions, bins):
    phase = frame.Phase.astype("string").str.strip().str.lower()
    labels = phase.map(PHASES).mask(phase.str.contains("rpm", na=False), 2)
    known = labels.notna().to_numpy()
    counts = np.bincount(
        3 * positions[known] + labels[known].to_numpy(dtype=int), minlength=bins * 3
    ).reshape(-1, 3)
    y = counts.argmax(axis=1)
    y[counts.sum(1) == 0] = -1
    return y


def training_id(files):
    digests = []
    for file in files:
        with file.open("rb") as stream:
            digests.append(hashlib.file_digest(stream, "sha256").digest())
    return hashlib.sha256(b"".join(sorted(digests))).hexdigest()


FINE_CLASSES = (
    "quiet_baseline",
    "rest",
    "rest_after_effort",
    "tmct",
    "stroop",
    "speech",
    "subtraction",
    "warm_up",
    "cycling",
    "sprint",
    "cooldown",
)

FINE_TO_COARSE = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


def fine_targets(frame, row, clock, y):
    """Use task annotations for auxiliary training targets consistent with the coarse window label."""
    phase = frame.Phase.astype("string").str.strip().str.lower()
    lookup = {
        "baseline": 0,
        "rest": 1,
        "tmct": 3,
        "stroop": 4,
        "real opinion": 5,
        "opposite opinion": 5,
        "subtraction": 6,
        "warm-up": 7,
        "sprint": 9,
        "cooldown": 10,
    }
    codes = phase.map(lookup).mask(phase.str.contains("rpm", na=False), 8)
    valid = codes.notna().to_numpy()
    n = len(clock)
    hist = np.bincount(
        len(FINE_CLASSES) * row[valid] + codes[valid].to_numpy("int64"),
        minlength=n * len(FINE_CLASSES),
    ).reshape(n, -1)
    hist = np.where(FINE_TO_COARSE[None, :] == y[:, None], hist, 0)
    fine = hist.argmax(1)
    fine[hist.max(1) == 0] = -1
    exercise = np.flatnonzero(y == 2)
    if len(exercise):
        j = np.searchsorted(clock[exercise], clock, side="right") - 1
        recent = (j >= 0) & (clock - clock[exercise[np.maximum(j, 0)]] <= 60)
        fine[(fine == 1) & recent] = 2
    return fine


def scaler(items, indices):
    y = np.concatenate([items[i]["x"][0] for i in indices])
    center = np.median(y, axis=0)
    lo, hi = np.quantile(y, [0.1, 0.9], axis=0)
    return (center.astype("f"), np.maximum(hi - lo, 0.15).astype("f"))


def reference_model(width_in, mode):
    # Training-only packages stay optional for replaying the benchmark with exported folds.
    import torch
    from torch import nn
    from torch.nn import functional as F

    class TemporalBlock(nn.Module):
        def __init__(self, width, dilation, kernel):
            super().__init__()
            self.pad = dilation * (kernel // 2)
            self.filter = nn.Conv1d(width, width, kernel, dilation=dilation)
            self.point = nn.Conv1d(width, width, 1)
            self.drop = nn.Dropout(0.13)

        def forward(self, x):
            a = self.filter(F.pad(x, (self.pad, self.pad), mode="replicate"))
            return x + self.point(self.drop(torch.relu(a))) * 0.5

    class ReferenceNet(nn.Module):
        def __init__(self, width_in, mode):
            super().__init__()
            self.mode = mode
            w = 52
            self.input = nn.Conv1d(width_in, w, 1)
            self.before = nn.ModuleList([TemporalBlock(w, d, 3) for d in (1, 3, 7)])
            self.attention = nn.Conv1d(w, 5, 1)
            self.values = nn.Conv1d(w, 14, 1)
            self.combine = nn.Conv1d(w + 70, w, 1)
            self.after = nn.ModuleList([TemporalBlock(w, d, 3) for d in (1, 4, 9, 18)])
            self.classes = nn.Conv1d(w, 3, 1)
            self.subtasks = nn.Conv1d(w, 11, 1)
            self.changes = nn.Conv1d(w, 1, 1)
            self.drop = nn.Dropout(0.09)

        def forward(self, x, mask):
            h = torch.relu(self.input(self.drop(x)))
            for layer in self.before:
                h = layer(h)
            att = self.attention(h).masked_fill(~mask[:, None, :], -10000.0).softmax(2)
            anchors = torch.bmm(self.values(h), att.transpose(1, 2)).flatten(1)
            h = h + 0.5 * torch.relu(
                self.combine(
                    torch.cat((h, anchors[:, :, None].expand(-1, -1, h.shape[-1])), 1)
                )
            )
            for layer in self.after:
                h = layer(h)
            return (self.classes(h), self.subtasks(h), self.changes(h)[:, 0])

    return ReferenceNet(width_in, mode)


def hard_examples(items, indices):
    """Assign extra loss weight to rest and stress errors made on withheld people."""
    ids = np.asarray(indices)
    groups = [items[i]["subject"] for i in ids]
    result = {}
    for a, b in GroupKFold(3).split(ids, groups=groups):
        tr = ids[a]
        va = ids[b]
        center, spread = scaler(items, tr)

        def transformed(i):
            return np.clip((items[i]["x"][0] - center) / spread, -4, 4)

        xx = np.concatenate([transformed(i)[items[i]["y"] >= 0] for i in tr])
        yy = np.concatenate([items[i]["y"][items[i]["y"] >= 0] for i in tr])
        head = LogisticRegression(C=0.025, class_weight="balanced", max_iter=240)
        head.fit(xx, yy)
        for i in va:
            p = head.predict_proba(transformed(i))
            y = items[i]["y"]
            hard = np.where(y == 0, p[:, 1], np.where(y == 1, p[:, 0], 0.3 * p[:, 0]))
            w = 1.0 + 0.95 * hard
            result[int(i)] = (w / w[y >= 0].mean()).astype("f")
    return result


def train_reference(items, indices, mode, seed, epochs, hard=None):
    """Learn three activity classes, eleven supervised subtasks and local class changes."""
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    rgen = np.random.default_rng(seed)
    torch.manual_seed(seed)
    indices = np.asarray(indices)
    center, scale = scaler(items, indices)
    model = reference_model(len(center), mode)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0017, weight_decay=0.025)
    pool = []
    mass = np.zeros(3)
    smallmass = np.zeros(11)
    for i in indices:
        r = items[i]
        candidates = temporal_segments(r["c"])
        for seg in candidates:
            yy = r["y"][seg]
            valid = yy >= 0
            if len(seg) < 4 or not valid.any():
                continue
            pool.append((i, seg))
            mass += np.bincount(yy[valid], minlength=3)
            fine = r["f"][seg]
            smallmass += np.bincount(fine[fine >= 0], minlength=11)
    weights = torch.from_numpy(
        ((mass.sum() / (3 * np.maximum(1, mass))) ** 0.62).astype("f")
    )
    fineweights = torch.from_numpy(
        np.clip(
            (smallmass.sum() / (11 * np.maximum(1, smallmass))) ** 0.37, 0.35, 4
        ).astype("f")
    )
    sampling = np.array(
        [1.3 if np.any(items[i]["y"][seg] == 1) else 1.0 for i, seg in pool]
    )
    sampling /= sampling.sum()
    steps = max(7, round(len(indices) / 5))
    n_batch = 6
    kept = 0
    averaged = None
    for epoch in range(epochs):
        gain = 0.5 + 0.5 * np.cos(epoch * np.pi / epochs)
        opt.param_groups[0]["lr"] = 0.00016 + 0.00155 * gain
        for step in range(steps):
            inp = []
            labels = []
            fine = []
            hardness = []
            transitions = []
            for j in rgen.choice(len(pool), size=n_batch, p=sampling):
                i, seg = pool[j]
                r = items[i]
                m = len(seg)
                if rgen.random() < 0.34 and m > 60:
                    size = int(rgen.uniform(0.48, 0.94) * m)
                    pos = rgen.integers(0, m - size + 1)
                    seg = seg[pos : pos + size]
                variant = (
                    0 if rgen.random() < 0.72 else int(rgen.integers(1, len(r["x"])))
                )
                vals = np.clip((r["x"][variant, seg] - center) / scale, -5, 5)
                # Warp crops so the model cannot rely on a fixed laboratory task duration.
                factor = rgen.uniform(0.73, 1.32) if rgen.random() < 0.72 else 1.0
                t = np.linspace(0, len(seg) - 1, max(2, int(len(seg) / factor)))
                lower = t.astype(int)
                upper = np.minimum(len(seg) - 1, lower + 1)
                u = (t - lower).astype("f")[:, None]
                a = vals[lower] + (vals[upper] - vals[lower]) * u
                a *= rgen.binomial(1, 0.96, size=(1, a.shape[1])).astype("f")
                y = r["y"][seg[np.rint(t).astype(int)]]
                sub = r["f"][seg[np.rint(t).astype(int)]]
                hw = (
                    hard[i][seg[np.rint(t).astype(int)]]
                    if hard
                    else np.ones(len(t), np.float32)
                )
                boundary = (
                    np.flatnonzero((y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)) + 1
                )
                target = np.zeros(len(t), np.float32)
                for b in boundary:
                    target[max(0, b - 1) : min(len(t), b + 2)] = 1.0
                inp.append(a)
                labels.append(y)
                fine.append(sub)
                hardness.append(hw)
                transitions.append(target)
            n = max((len(v) for v in inp))
            xb = np.stack(
                [np.pad(v, ((0, n - len(v)), (0, 0)), mode="edge").T for v in inp]
            )
            mask = np.stack([np.arange(n) < len(v) for v in inp])
            yb = np.stack(
                [np.pad(v, (0, n - len(v)), constant_values=-1) for v in labels]
            )
            fb = np.stack(
                [np.pad(v, (0, n - len(v)), constant_values=-1) for v in fine]
            )
            hw = np.stack([np.pad(v, (0, n - len(v))) for v in hardness])
            bd = np.stack([np.pad(v, (0, n - len(v))) for v in transitions])
            x = torch.from_numpy(xb)
            y = torch.from_numpy(yb)
            f = torch.from_numpy(fb)
            valid = y >= 0
            fv = f >= 0
            p, q, b = model(x, torch.from_numpy(mask))
            loss = F.cross_entropy(
                p,
                y,
                weight=weights,
                ignore_index=-1,
                reduction="none",
                label_smoothing=0.025,
            )
            loss = (loss * torch.from_numpy(hw))[valid].mean()
            if fv.any():
                loss = loss + 0.42 * F.cross_entropy(
                    q, f, weight=fineweights, ignore_index=-1, label_smoothing=0.02
                )
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
                b[valid], torch.from_numpy(bd)[valid], pos_weight=torch.tensor(4.0)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        # Average late checkpoints to reduce variance across people.
        if epoch >= int(epochs * 0.68) and (epoch % 5 == 0 or epoch == epochs - 1):
            ss = model.state_dict()
            if averaged is None:
                averaged = {k: t.clone() for k, t in ss.items()}
            else:
                for k, t in ss.items():
                    averaged[k] += t
            kept += 1
    model.load_state_dict({k: v / kept for k, v in averaged.items()})
    model.eval()
    return (model, center, scale)


def predict_ref(fit, features, clock):
    import torch

    model, center, scale = fit
    x = np.clip((features - center) / scale, -5, 5)
    p = np.empty((len(x), 3), np.float32)
    b = np.empty(len(x), np.float32)
    with torch.no_grad():
        for seg in temporal_segments(clock):
            if not len(seg):
                continue
            a = torch.from_numpy(np.ascontiguousarray(x[seg].T)[None])
            coarse, fine, boundary = model(a, torch.ones(1, len(seg), dtype=torch.bool))
            prob = coarse[0].softmax(0).T.numpy()
            q = fine[0].softmax(0).T.numpy()
            relatives = np.column_stack(
                [q[:, FINE_TO_COARSE == j].sum(1) for j in range(3)]
            )
            p[seg] = 0.72 * prob + 0.28 * relatives
            b[seg] = boundary[0].sigmoid().numpy()
    return (p, b)


def train_segment(data, indices):
    """Fit the complementary tree on summaries of sensor-defined segments and corrupted signals."""
    rng = np.random.default_rng(64)
    xx = []
    yy = []
    weights = []
    hard = hard_examples(data, indices)
    for i in indices:
        r = data[i]
        known = r["y"] >= 0
        for v, aug in enumerate(r["a"]):
            chosen = known if v == 0 else known & (rng.random(len(known)) < 0.077)
            if chosen.any():
                xx.append(aug[chosen])
                yy.append(r["y"][chosen])
                weights.append((1.0 if v == 0 else 0.53) * hard[i][chosen])
    x = np.concatenate(xx)
    y = np.concatenate(yy)
    w = np.concatenate(weights)
    masses = np.bincount(y, weights=w)
    w *= np.power(masses.sum() / (3 * masses), 0.82)[y]
    m = HistGradientBoostingClassifier(
        max_iter=160,
        min_samples_leaf=35,
        max_leaf_nodes=13,
        learning_rate=0.065,
        l2_regularization=7,
        early_stopping=False,
        random_state=334,
    )
    m.fit(x, y, sample_weight=w)
    return m


TRAIN_OUTAGES = (
    "without_HR",
    "without_IBI_duration",
    "without_EDA",
    "without_Temperature",
    "without_BVP",
    "both_ibi",
    "optical_loss",
    "rotating_gaps",
    "scattered_BVP",
    "partial_BVP",
    "partial_Temperature",
    "eda_temperature",
    "hr_and_ibi",
    "without_IBI_dt",
)

VALIDATION_MODES = (0, 1, 3, 5, 6, 7, 8, 10, 11)


def base_graph(packs, rule, persistence, corpus, dev_score):
    """Export a variable-length, single-recording graph and embed its calibrated decision rule."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    weights = []
    outs = []
    serial = 0

    def const(name, v):
        weights.append(numpy_helper.from_array(np.ascontiguousarray(v), name=name))
        return name

    def op(operator, *args, name=None, **attrs):
        nonlocal serial
        serial += 1
        output = name or f"t{serial}"
        nodes.append(helper.make_node(operator, list(args), [output], **attrs))
        return output

    newaxis = const("batch_axis", np.array([0], np.int64))
    anchoraxis = const("anchor_axis", np.array([2], np.int64))
    squeeze = const("squeeze", np.array([0], np.int64))
    cliplo = const("cliplo", np.array(-5, np.float32))
    cliphi = const("cliphi", np.array(5, np.float32))
    half = const("half", np.array(0.5, np.float32))
    classmix = const("coarse_mix", np.array(0.72, np.float32))
    finemix = const("fine_mix", np.array(0.28, np.float32))
    matrix = const("task_to_class", np.eye(3, dtype=np.float32)[FINE_TO_COARSE])
    ins = [
        helper.make_tensor_value_info("physiology", TensorProto.FLOAT, ["time", 301])
    ]
    combined = None
    if any((p["variant"] == "beat" for p in packs.values())):
        ins.append(
            helper.make_tensor_value_info(
                "pulse_detail", TensorProto.FLOAT, ["time", 125]
            )
        )
        combined = op("Concat", "physiology", "pulse_detail", axis=1)
    additional_versions = {}
    for tag, packed in packs.items():
        if tag == "segments":
            from onnx.compose import add_prefix_graph
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            estimator = packed["estimator"]
            child = convert_sklearn(
                estimator,
                initial_types=[("regimes", FloatTensorType([None, 577]))],
                target_opset=17,
                options={id(estimator): {"zipmap": False}},
            )
            cg = add_prefix_graph(child.graph, "seg_", rename_inputs=False)
            nodes.extend(cg.node)
            weights.extend(cg.initializer)
            op("Identity", "seg_probabilities", name="segments_p")
            for ops in child.opset_import:
                additional_versions[ops.domain] = max(
                    additional_versions.get(ops.domain, 0), ops.version
                )
            outs.append(
                helper.make_tensor_value_info(
                    "segments_p", TensorProto.FLOAT, ["time", 3]
                )
            )
            ins.append(
                helper.make_tensor_value_info(
                    "regimes", TensorProto.FLOAT, ["time", 577]
                )
            )
            continue
        prefix = tag + "_"

        def w(name, v):
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return const(prefix + name, v)

        state = packed["state"]

        def conv(x, name, d=1):
            return op(
                "Conv",
                x,
                w(name + ".W", state[name + ".weight"]),
                w(name + ".b", state[name + ".bias"]),
                dilations=[d],
            )

        def edgepad(x, pad):
            return op(
                "Pad",
                x,
                w(
                    "pad" + str(len(weights)),
                    np.array([0, 0, pad, 0, 0, pad], np.int64),
                ),
                mode="edge",
            )

        def resid(h, base, d):
            fil = op("Relu", conv(edgepad(h, d), base + ".filter", d))
            return op("Add", h, op("Mul", half, conv(fil, base + ".point")))

        column = combined if packed["variant"] == "beat" else "physiology"
        z = op(
            "Clip",
            op(
                "Div",
                op("Sub", column, w("center", packed["center"])),
                w("scale", packed["scale"]),
            ),
            cliplo,
            cliphi,
        )
        x = op("Unsqueeze", op("Transpose", z, perm=[1, 0]), newaxis)

        def as_rows(k):
            return op("Transpose", op("Squeeze", k, squeeze), perm=[1, 0])

        h = op("Relu", conv(x, "input"))
        for j, d in enumerate((1, 3, 7)):
            h = resid(h, "before." + str(j), d)
        att = op("Softmax", conv(h, "attention"), axis=2)
        anchors = op(
            "Flatten",
            op("MatMul", conv(h, "values"), op("Transpose", att, perm=[0, 2, 1])),
            axis=1,
        )
        anchor_col = op("Unsqueeze", anchors, anchoraxis)
        zeros = op(
            "Mul",
            op("ReduceSum", h, w("channel_axis", np.array([1], np.int64)), keepdims=1),
            w("zero", np.array(0.0, np.float32)),
        )
        z = op("Concat", h, op("Add", anchor_col, zeros), axis=1)
        h = op("Add", h, op("Mul", half, op("Relu", conv(z, "combine"))))
        ds = (1, 4, 9, 18)
        for j, d in enumerate(ds):
            h = resid(h, "after." + str(j), d)
        coarse = op("Softmax", as_rows(conv(h, "classes")), axis=1)
        fine = op("MatMul", op("Softmax", as_rows(conv(h, "subtasks")), axis=1), matrix)
        op(
            "Add",
            op("Mul", classmix, coarse),
            op("Mul", finemix, fine),
            name=tag + "_p",
        )
        op("Sigmoid", as_rows(conv(h, "changes")), name=tag + "_b")
        outs.extend(
            [
                helper.make_tensor_value_info(
                    tag + "_p", TensorProto.FLOAT, ["time", 3]
                ),
                helper.make_tensor_value_info(
                    tag + "_b", TensorProto.FLOAT, ["time", 1]
                ),
            ]
        )
    graph = helper.make_graph(
        nodes, "physiological_sequence_classifier", ins, outs, initializer=weights
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            *[
                helper.make_opsetid(key, value)
                for key, value in additional_versions.items()
                if key
            ],
        ],
    )
    model.ir_version = 9
    helper.set_model_props(
        model,
        {
            "pipeline": "pulse-reference-v4",
            "feature_schema": "physiology45-beat23-v1",
            "estimator": "reference_and_transition_models",
            "training_files_sha256": corpus,
            "development_macro_f1": str(dev_score),
            "decision": json.dumps(rule),
            "persistence": json.dumps(persistence.tolist()),
            "classes": json.dumps(["Rest", "Stress", "Activity"]),
            "networks": json.dumps(list(packs)),
        },
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


PIPELINE = "participant-response-v6"


def state_targets(y, clock):
    """Return the nearest observed type of active episode for auxiliary supervision."""
    known = np.flatnonzero(y > 0)
    if not len(known):
        return np.full(len(y), -1)
    k = clock[known].searchsorted(clock)
    l = known[np.maximum(0, k - 1)]
    r = known[np.minimum(k, len(known) - 1)]
    closest = np.where(abs(clock - clock[l]) <= abs(clock - clock[r]), l, r)
    return np.where(
        abs(clock - clock[closest]) <= 180, (y[closest] == 2).astype(int), -1
    )


def pair_network(width):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
            self.conv = nn.Conv1d(42, 42, 3, dilation=d)
            self.mix = nn.Conv1d(42, 42, 1)

        def forward(self, x):
            h = F.pad(x, (self.d, self.d), mode="replicate")
            return x + 0.45 * self.mix(
                F.dropout(F.silu(self.conv(h)), 0.12, self.training)
            )

    class PairNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.ch = nn.Conv1d(width, 42, 1)
            self.first = nn.ModuleList([Block(d) for d in (1, 4, 11)])
            self.q = nn.Conv1d(42, 16, 1)
            self.k = nn.Conv1d(42, 16, 1)
            self.value = nn.Conv1d(42, 22, 1)
            self.fuse = nn.Conv1d(86, 42, 1)
            self.last = nn.ModuleList([Block(d) for d in (2, 7, 18)])
            self.coarse = nn.Conv1d(42, 3, 1)
            self.head = nn.Conv1d(42, 5, 1)
            self.fine = nn.Conv1d(42, 11, 1)

        def forward(self, x, valid):
            h = F.silu(self.ch(F.dropout(x, 0.075, self.training)))
            for block in self.first:
                h = block(h)
            q = self.q(h)
            k = self.k(h)
            sim = torch.bmm(q.transpose(1, 2), k) * 0.25
            sim = sim.masked_fill(~valid[:, None, :], -1e4)
            opp = (-torch.bmm(q.transpose(1, 2), k) * 0.25).masked_fill(
                ~valid[:, None, :], -1e4
            )
            v = self.value(h)
            near = torch.bmm(v, sim.softmax(2).transpose(1, 2))
            far = torch.bmm(v, opp.softmax(2).transpose(1, 2))
            h = h + 0.5 * F.silu(self.fuse(torch.cat((h, near, near - far), 1)))
            for block in self.last:
                h = block(h)
            return self.coarse(h), self.head(h), self.fine(h)

    return PairNetwork()


def fit_pair(data, ids, external=(), epochs=50, seed=3241):
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xx = np.concatenate([data[i]["z"][0, ::2] for i in ids])
    center = np.median(xx, 0).astype("f")
    a, b = np.quantile(xx, [0.12, 0.88], axis=0)
    scale = np.maximum(0.18, b - a).astype("f")
    items = [(data[i], False) for i in ids] + [(r, True) for r in external]
    model = pair_network(len(center))
    opt = torch.optim.AdamW(model.parameters(), lr=0.0016, weight_decay=0.035)
    pool = []
    weight = []
    for i, (r, ext) in enumerate(items):
        ctx = np.zeros(len(r["y"]), int) if ext else state_targets(r["y"], r["c"])
        # External baseline and Stroop come from separately recorded periods. Compare them by physiology, never by sample order.
        for seg in temporal_segments(r["c"]):
            if len(seg) > 3 and np.any(r["y"][seg] >= 0):
                pool.append((i, seg, ctx, ext))
                weight.append(
                    0.13 if ext else (1.60 if np.any(r["y"][seg] == 1) else 1.0)
                )
    wt = np.array(weight)
    wt /= wt.sum()
    cls = torch.tensor([0.95, 1.85, 0.80])
    keep = 0
    avg = None
    for ep in range(epochs):
        opt.param_groups[0]["lr"] = 0.00020 + 0.00165 * (
            0.5 + 0.5 * np.cos(np.pi * ep / epochs)
        )
        for _ in range(max(7, round(len(ids) / 5))):
            xs = []
            ys = []
            cs = []
            fs = []
            bound = []
            strength = []
            for ix in rng.choice(len(pool), size=7, p=wt):
                i, seg, ctx, ext = pool[ix]
                r = items[i][0]
                if len(seg) > 60 and rng.random() < 0.46:
                    width = int(min(len(seg), rng.integers(48, 145)))
                    j = int(rng.integers(len(seg) - width + 1))
                    seg = seg[j : j + width]
                y = r["y"][seg]
                v = (
                    0
                    if ext or rng.random() < 0.73
                    else int(rng.integers(1, len(r["z"])))
                )
                x = np.clip((r["z"][v, seg] - center) / scale, -5, 5)
                factor = rng.uniform(0.76, 1.24) if rng.random() < 0.68 else 1.0
                tt = np.linspace(0, len(seg) - 1, max(3, round(len(seg) / factor)))
                l = tt.astype(int)
                h = np.minimum(len(seg) - 1, l + 1)
                x = x[l] + (x[h] - x[l]) * (tt - l).astype("f")[:, None]
                x *= rng.binomial(1, 0.97, size=(1, x.shape[1])).astype("f")
                jj = seg[np.rint(tt).astype(int)]
                y = r["y"][jj]
                c = ctx[jj]
                fine = r["f"][jj]
                e = np.zeros(len(y), "f")
                sz = (
                    np.flatnonzero((y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)) + 1
                )
                for k in sz:
                    e[max(0, k - 1) : min(len(y), k + 2)] = 1
                xs.append(x)
                ys.append(y)
                cs.append(c)
                fs.append(fine)
                bound.append(e)
                strength.append(0.66 if ext else 1.0)
            n = max(map(len, xs))
            X = torch.from_numpy(
                np.stack(
                    [np.pad(v, ((0, n - len(v)), (0, 0)), mode="edge").T for v in xs]
                )
            )
            valid = torch.from_numpy(np.stack([np.arange(n) < len(v) for v in xs]))

            def pad(v, pad_value):
                return torch.from_numpy(
                    np.stack(
                        [
                            np.pad(x, (0, n - len(x)), constant_values=pad_value)
                            for x in v
                        ]
                    )
                )

            Y = pad(ys, -1)
            C = pad(cs, -1)
            Z = pad(fs, -1)
            B = pad(bound, 0.0)
            W = torch.tensor(strength)[:, None]
            known = Y >= 0
            coarse, head, fine = model(X, valid)
            loss = (
                W
                * F.cross_entropy(
                    coarse,
                    Y,
                    weight=cls,
                    reduction="none",
                    ignore_index=-1,
                    label_smoothing=0.025,
                )
            )[known].mean()

            def binary(logit, target, mask, pos):
                if not mask.any():
                    return logit.sum() * 0
                a = F.binary_cross_entropy_with_logits(
                    logit,
                    target.float(),
                    pos_weight=torch.tensor(float(pos)),
                    reduction="none",
                )
                return (a * W)[mask].mean()

            mental = head[:, 0]
            effort = head[:, 1]
            gate = head[:, 2]
            e = head[:, 3]
            loss = (
                loss
                + 0.50 * binary(mental, Y == 1, known & (C == 0), 1.35)
                + 0.34 * binary(effort, Y == 2, known & (C == 1), 0.88)
            )
            loss = (
                loss
                + 0.23 * binary(gate, C == 1, known & (C >= 0), 1.0)
                + 0.10 * binary(e, B, known, 3.0)
            )
            if (Z >= 0).any():
                loss = loss + 0.26 * F.cross_entropy(
                    fine, Z, ignore_index=-1, label_smoothing=0.03
                )
            ranking = []
            for j in range(len(xs)):
                hi = mental[j, Y[j] == 1][::3]
                lo = mental[j, (Y[j] == 0) & (C[j] == 0)][::4]
                if len(hi) and len(lo):
                    ranking.append(F.softplus(0.5 - hi[:, None] + lo[None, :]).mean())
            if ranking:
                loss = loss + 0.23 * torch.stack(ranking).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.2)
            opt.step()
        if ep >= int(0.65 * epochs) and (ep % 4 == 0 or ep == epochs - 1):
            now = model.state_dict()
            if avg is None:
                avg = {k: v.clone() for k, v in now.items()}
            else:
                for k, v in now.items():
                    avg[k] += v
            keep += 1
    model.load_state_dict({k: v / keep for k, v in avg.items()})
    model.eval()
    return model, center, scale


def paired_predict(fit, x, clock):
    import torch

    m, center, scale = fit
    z = np.clip((x - center) / scale, -5, 5)
    p = np.empty((len(x), 3), "f")
    b = np.empty(len(x), "f")
    with torch.no_grad():
        for seg in temporal_segments(clock):
            if not len(seg):
                continue
            v = torch.from_numpy(np.ascontiguousarray(z[seg].T)[None])
            coarse, head, _ = m(v, torch.ones(1, len(seg), dtype=torch.bool))
            coarse = coarse[0].softmax(0).T.numpy()
            head = head[0].sigmoid().numpy()
            stress = head[0] * (1 - head[2])
            act = head[1] * head[2]
            p[seg] = 0.52 * coarse + 0.48 * np.column_stack(
                (1 - stress - act, stress, act)
            )
            b[seg] = head[3]
    return p, b


def external_people_id(records):
    people = sorted(str(r["subject"]) for r in records)
    return hashlib.sha256(
        json.dumps(people, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def fit_external_prior(records):
    from sklearn.linear_model import LogisticRegression

    X = []
    Y = []
    subject = []
    for j, r in enumerate(records):
        if not np.any(r["y"] == 1) or not np.any(r["y"] == 0):
            continue
        feature = transfer_features(r["x"][0], r["c"])
        keep = r["y"] >= 0
        X.append(feature[keep])
        Y.append(r["y"][keep])
        subject.extend([j] * keep.sum())
    x = np.concatenate(X)
    y = np.concatenate(Y)
    s = np.asarray(subject)
    center = np.median(x, axis=0)
    lo, hi = np.quantile(x, [0.12, 0.88], axis=0)
    scale = np.maximum(0.22, hi - lo).astype("f")
    z = np.clip((x - center) / scale, -3.5, 3.5)
    rng = np.random.default_rng(728)
    diff = []
    labels = []
    for uid in np.unique(s):
        a = np.flatnonzero((s == uid) & (y == 1))
        b = np.flatnonzero((s == uid) & (y == 0))
        step = z[rng.choice(a, 60)] - z[rng.choice(b, 60)]
        diff.extend((step, -step))
        labels.extend((np.ones(len(step)), np.zeros(len(step))))
    clf = LogisticRegression(C=0.004, fit_intercept=False, max_iter=260)
    clf.fit(np.concatenate(diff), np.concatenate(labels))
    # Map the ranking score to a balanced probability without assuming stress prevalence in the target recording.
    beta = clf.coef_[0]
    r = z @ beta
    threshold = 0.5 * (np.median(r[y == 0]) + np.median(r[y == 1]))
    return {
        "people_sha256": external_people_id(records),
        "people_count": len(records),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coef": beta.tolist(),
        "bias": -float(threshold),
    }


def external_frames(archive, folders=("baseline", "stroop")):
    """Read requested independent CATSA wrist trials. By default training uses baseline and Stroop."""
    from io import BytesIO

    ticks = {
        "BVP": 15_625_000,
        "EDA": 250_000_000,
        "Temperature": 250_000_000,
        "HR": 1_000_000_000,
    }
    with zipfile.ZipFile(archive) as bundle:
        members = {name.lower(): name for name in bundle.namelist()}
        subjects = sorted(
            {
                name.split("/")[1]
                for name in members
                if re.match(r"^catsa/sub\d+/(baseline|stroop)/bvp\.csv$", name)
            }
        )
        for subject in subjects:
            chunks = []
            for j, folder in enumerate(folders):
                signals = {}
                for key, rate in ticks.items():
                    labelname = "temp" if key == "Temperature" else key.lower()
                    path = members.get(f"catsa/{subject}/{folder}/{labelname}.csv")
                    if not path:
                        continue
                    table = pd.read_csv(BytesIO(bundle.read(path)))
                    if table.shape[1] != 1:
                        raise ValueError(f"Unexpected CATSA sensor format: {path}")
                    values = pd.to_numeric(table.iloc[:, 0], errors="coerce").to_numpy(
                        float
                    )
                    # Each source file starts at the beginning of its own labelled trial. Its original absolute start is unavailable.
                    observed = pd.to_datetime(
                        pd.Timestamp("2000-01-01").value
                        + j * 7_200_000_000_000
                        + np.arange(len(values), dtype=np.int64) * rate,
                        unit="ns",
                    )
                    signals[key] = pd.Series(values, index=observed)
                if len(signals) < 3:
                    break
                frame = pd.DataFrame(signals).sort_index()
                frame["Phase"] = folder
                chunks.append(frame)
            if len(chunks) == len(folders):
                yield "CATSA/" + subject, pd.concat(chunks)


def fit_residual(rows, ids, amount=0.18):
    z = np.concatenate([rows[i][0] for i in ids])
    base = np.concatenate([rows[i][1] for i in ids])
    y = np.concatenate([rows[i][2] for i in ids])
    weights = np.concatenate([rows[i][3] for i in ids])
    ctr = np.average(z, axis=0, weights=weights)
    spread = np.maximum(
        0.18, np.sqrt(np.average((z - ctr) ** 2, axis=0, weights=weights))
    )
    x = np.ascontiguousarray(np.clip((z - ctr) / spread, -4, 4), dtype=float)
    models = []
    for col in range(2):
        valid = (y >= 0) & ((y <= 1) if col == 0 else True)
        target = (y[valid] == col + 1).astype(float)
        X = x[valid]
        b = base[valid, col]
        w = weights[valid].copy()
        w *= np.where(target > 0, 2.1 if col == 0 else 1.0, 1.0)
        w /= w.sum()

        def loss(theta):
            eta = b + theta[0] + X @ theta[1:]
            e = w * (expit(eta) - target)
            return w @ (np.logaddexp(0, eta) - target * eta) + 0.5 * amount * (
                theta[1:] @ theta[1:]
            ) + 0.018 * theta[0] ** 2, np.r_[
                e.sum() + 0.036 * theta[0], X.T @ e + amount * theta[1:]
            ]

        opt = minimize(
            loss,
            np.zeros(x.shape[1] + 1),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=70),
        )
        models.append(opt.x)
    return {
        "center": ctr.tolist(),
        "scale": spread.tolist(),
        "coef": np.asarray(models).tolist(),
    }


def summarize(cm):
    f1 = 2 * cm.diagonal() / np.maximum(1, cm.sum(0) + cm.sum(1))
    return dict(
        macro=float(f1.mean()),
        stress=float(f1[1]),
        activity=float(f1[2]),
        rest=float(f1[0]),
        fp=int(cm[:, 1].sum() - cm[1, 1]),
        fn=int(cm[1].sum() - cm[1, 1]),
        warm=int(cm[2, 0]),
        cm=cm.tolist(),
    )


def get_meta(data, indices, pg, bg, mode, prior):
    store = {}
    for i in indices:
        r = data[i]
        x = r["x"][(0, 1, 3, 5, 6, 7, 8, 10, 11)[mode]]
        c = r["c"]
        z, b0, _ = meta_design(
            [p[i][mode] for p in pg], [o[i][mode] for o in bg], x, c, prior
        )
        store[int(i)] = (
            z,
            b0,
            r["y"],
            r["c"],
            np.mean([o[i][mode] for o in bg], axis=0),
        )
    return store


def pool_rows(data, ids, dsets):
    rows = {}
    weights = {}
    for i in ids:
        weights[data[i]["subject"]] = weights.get(data[i]["subject"], 0) + int(
            (data[i]["y"] >= 0).sum()
        )
    for i in ids:
        sizes = weights[data[i]["subject"]]
        zs = []
        bs = []
        ys = []
        ws = []
        for v, (mode, d) in enumerate(dsets.items()):
            x, b, y, c, edge = d[i]
            choose = np.arange(len(y))
            if mode:
                choose = choose[(choose + int(i)) % (3 if mode == 6 else 5) == 0]
            zs.append(x[choose])
            bs.append(b[choose])
            ys.append(y[choose])
            ws.append(
                np.full(len(choose), (1.0 if mode == 0 else 0.35) / np.sqrt(sizes))
            )
        rows[int(i)] = (
            np.concatenate(zs),
            np.concatenate(bs),
            np.concatenate(ys),
            np.concatenate(ws),
        )
    return rows


def assemble_dev(data, ids, bankprobs, bankedges, route, prior):
    sel = [bankprobs[k] for k in route]
    bed = [bankedges[k] for k in route]
    return {mode: get_meta(data, ids, sel, bed, mode, prior) for mode in (0, 2, 3, 6)}


def make_response_graph(
    packs, prior, calibration, settings, corpus, development_score, base_export
):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    old = {k: v for k, v in packs.items() if k != "net"}
    if old:
        rule = dict(
            blend={k: 1 / len(old) for k in old},
            stress_weight=1.0,
            activity_weight=1.0,
            switch_cost=0.0,
            gate=0.0,
            duration=0.0,
        )
        model = onnx.load_model_from_string(
            base_export(old, rule, np.ones((3, 14)) / 14, corpus, development_score)
        )
    else:
        graph = helper.make_graph([], "wrist_response", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
    g = model.graph
    if "net" in packs:
        pack = packs["net"]
        state = pack["state"]
        serial = 0

        def const(name, val):
            if hasattr(val, "detach"):
                val = val.detach().numpy()
            g.initializer.append(
                numpy_helper.from_array(np.asarray(val), name="response_" + name)
            )
            return "response_" + name

        def op(name, *inputs, out=None, **kw):
            nonlocal serial
            serial += 1
            dest = out or "response_v" + str(serial)
            g.node.append(helper.make_node(name, list(inputs), [dest], **kw))
            return dest

        c = const("center", pack["center"])
        s = const("scale", pack["scale"])
        g.input.append(
            helper.make_tensor_value_info(
                "response", TensorProto.FLOAT, ["time", len(pack["center"])]
            )
        )
        x = op(
            "Clip",
            op("Div", op("Sub", "response", c), s),
            const("lo", np.float32(-5)),
            const("hi", np.float32(5)),
        )
        axis = const("axis0", np.array([0], np.int64))
        x = op("Unsqueeze", op("Transpose", x, perm=[1, 0]), axis)

        def conv(x, name, d=1):
            return op(
                "Conv",
                x,
                const(name + "_W", state[name + ".weight"]),
                const(name + "_b", state[name + ".bias"]),
                dilations=[d],
            )

        def silu(x):
            return op("Mul", x, op("Sigmoid", x))

        def residual(x, name, d):
            zz = op(
                "Pad",
                x,
                const(name + "_pad", np.array([0, 0, d, 0, 0, d], np.int64)),
                mode="edge",
            )
            return op(
                "Add",
                x,
                op(
                    "Mul",
                    const(name + "_fraction", np.float32(0.45)),
                    conv(silu(conv(zz, name + ".conv", d)), name + ".mix"),
                ),
            )

        h = silu(conv(x, "ch"))
        for j, d in enumerate((1, 4, 11)):
            h = residual(h, "first." + str(j), d)
        q = conv(h, "q")
        k = conv(h, "k")
        v = conv(h, "value")
        sim = op(
            "Mul",
            op("MatMul", op("Transpose", q, perm=[0, 2, 1]), k),
            const("temperature", np.float32(0.25)),
        )
        near = op(
            "MatMul", v, op("Transpose", op("Softmax", sim, axis=2), perm=[0, 2, 1])
        )
        far = op(
            "MatMul",
            v,
            op("Transpose", op("Softmax", op("Neg", sim), axis=2), perm=[0, 2, 1]),
        )
        feat = op("Concat", h, near, op("Sub", near, far), axis=1)
        h = op(
            "Add", h, op("Mul", const("mix", np.float32(0.5)), silu(conv(feat, "fuse")))
        )
        for j, d in enumerate((2, 7, 18)):
            h = residual(h, "last." + str(j), d)

        def rows(a):
            return op("Transpose", op("Squeeze", a, axis), perm=[1, 0])

        direct = op("Softmax", rows(conv(h, "coarse")), axis=1)
        heads = op("Sigmoid", rows(conv(h, "head")))

        def col(k):
            return op(
                "Gather", heads, const("col" + str(k), np.array([k], np.int64)), axis=1
            )

        m, a, gate = col(0), col(1), col(2)
        one = const("one", np.float32(1.0))
        stress = op("Mul", m, op("Sub", one, gate))
        effort = op("Mul", a, gate)
        quiet = op("Sub", op("Sub", one, stress), effort)
        three = op("Concat", quiet, stress, effort, axis=1)
        op(
            "Add",
            op("Mul", direct, const("coarse_fraction", np.float32(0.52))),
            op("Mul", three, const("conditional_fraction", np.float32(0.48))),
            out="net_p",
        )
        op("Identity", col(3), out="net_b")
        g.output.extend(
            [
                helper.make_tensor_value_info("net_p", TensorProto.FLOAT, ["time", 3]),
                helper.make_tensor_value_info("net_b", TensorProto.FLOAT, ["time", 1]),
            ]
        )
    meta = dict(
        pipeline="participant-response-v6",
        estimator="participant_relative_stress_and_effort",
        feature_schema="wrist-v5",
        classes=json.dumps(["Rest", "Stress", "Activity"]),
        networks=json.dumps(list(packs)),
        decision=json.dumps(settings),
        external_prior=json.dumps(prior),
        external_training_fingerprint=prior.get("people_sha256", ""),
        residual=json.dumps(calibration),
        development_macro_f1=str(development_score),
        training_files_sha256=corpus,
        training_recipe="participant-calibration-v1",
        external_sha256="8c17d8cec3b9fbae1ef2727357e75f60c837b4ff4e308446ce47435049e4a02a",
        external_attribution="CATSA v2, Md. Santo Ali and coauthors, doi:10.6084/m9.figshare.28690988.v2, CC BY 4.0. Baseline and Stroop wrist signals transformed into fitted parameters.",
    )
    helper.set_model_props(model, meta)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def collect_short_trials(frame, truth, clock, external=False):
    """Create label-independent short input segments and recompute their sensor features."""
    sensors = frame.reindex(columns=CHANNELS)
    ticks = sensors.index.as_unit("ns").asi8 // BIN_NS
    if np.any(np.diff(ticks) < 0):
        raise ValueError("Expected training rows in chronological order")
    xx = []
    yy = []
    ww = []
    for contiguous in temporal_segments(clock):
        if not len(contiguous):
            continue
        cuts = ((0, 18),) if external else ((0, 18), (7, 25))
        for offset, width in cuts:
            for start in range(offset, len(contiguous), width):
                seg = contiguous[start : start + width]
                if len(seg) < 5 or not np.any(truth[seg] >= 0):
                    continue
                a = ticks.searchsorted(clock[seg[0]])
                b = ticks.searchsorted(clock[seg[-1]], side="right")
                part = sensors.iloc[a:b]
                raw, _, c = physiology_windows(part)
                detail = autonomic_detail(part, c)
                if not np.array_equal(c, clock[seg]):
                    raise ValueError("Short training and label clocks disagree")
                xx.append(short_features(raw, detail, c))
                yy.append(truth[seg])
                ww.append(np.ones(len(seg)))
                # Missing shared channels are never specific to a labelled class.
                mask = raw.copy()
                md = detail.copy()
                group = (start // width + offset) % 4
                if group == 0:
                    mask[:, :5] = np.nan
                elif group == 1:
                    mask[:, 5:17] = np.nan
                    md[:, 19:] = np.nan
                elif group == 2:
                    mask[:, 17:22] = np.nan
                else:
                    mask[:, 29:] = np.nan
                    md[:, :19] = np.nan
                xx.append(short_features(mask, md, c))
                yy.append(truth[seg])
                ww.append(np.full(len(seg), 0.23))
    if not xx:
        return None
    return (np.concatenate(xx), np.concatenate(yy), np.concatenate(ww))


def fit_short_response(huawei, external):
    data = [
        (r["short"], r["subject"], i >= len(huawei))
        for i, r in enumerate([*huawei, *external])
        if r.get("short") is not None
    ]
    x = np.concatenate([r[0] for r, uid, ext in data])
    y = np.concatenate([r[1] for r, uid, ext in data])
    weight = np.concatenate([r[2] for r, uid, ext in data])
    subject = np.concatenate([np.repeat(uid, len(r[1])) for r, uid, ext in data])
    ext = np.concatenate([np.full(len(r[1]), ex) for r, uid, ex in data])
    keep = y >= 0
    x = x[keep]
    y = y[keep]
    weight = weight[keep]
    subject = subject[keep]
    ext = ext[keep]
    rng = np.random.default_rng(3127)
    # Each population contributes equally to estimating the feature scale.
    ids = np.concatenate(
        [
            rng.choice(k, 2400, replace=len(k) < 2400)
            for k in (np.flatnonzero(~ext), np.flatnonzero(ext))
        ]
    )
    low, center, high = np.quantile(x[ids], [0.1, 0.5, 0.9], axis=0)
    scale = np.maximum(0.07, high - low)
    z = np.asarray(np.clip((x - center) / scale, -3.5, 3.5), dtype="f")
    choices = [
        np.flatnonzero((y == label) & (ext == domain))
        for domain in (False, True)
        for label in (0, 1, 2)
    ]
    centers = np.concatenate(
        [rng.choice(k, 36, replace=len(k) < 36) for k in choices if len(k)]
    )
    fit = dict(
        center=center.astype("f").tolist(),
        scale=scale.astype("f").tolist(),
        centers=z[centers].tolist(),
    )
    features = short_kernel(x, fit)
    betas = []
    bias = []
    for out in range(2):
        use = (y < 2) if out == 0 else np.ones(len(y), bool)
        xf = features[use]
        target = (y[use] == out + 1).astype(np.int32)
        person = subject[use]
        source = ext[use]
        w = weight[use].copy()
        # Balance people, not the many repeated windows from a long exercise recording.
        for uid in np.unique(person):
            k = person == uid
            w[k] /= max(1, w[k].sum()) ** 0.67
        for domain, total in (
            (False, 0.48 if out == 0 else 0.84),
            (True, 0.52 if out == 0 else 0.16),
        ):
            for label in (0, 1):
                k = (source == domain) & (target == label)
                if not k.any():
                    continue
                share = (
                    (0.56 if label == 1 else 0.44)
                    if out == 0
                    else ((0.50 if label == 1 else 0.50) if not domain else 1.0)
                )
                w[k] *= (total * share) / w[k].sum()
        w *= 3000 / w.sum()
        learner = LogisticRegression(C=0.055, max_iter=200, tol=3.0e-5)
        learner.fit(xf, target, sample_weight=w)
        betas.append(learner.coef_[0].astype("f").tolist())
        bias.append(float(learner.intercept_[0]))
    fit.update(coef=betas, bias=bias)
    return fit


def add_short_graph(blob, head):
    """Add the low-context classifier as a second output of the deployed ONNX graph."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load_model_from_string(blob)
    graph = model.graph

    def constant(name, x):
        name = "short_" + name
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(x, dtype=np.float32), name=name)
        )
        return name

    seq = 0

    def op(kind, values, out=None, **kwargs):
        nonlocal seq
        seq += 1
        key = out or f"short_node_{seq}"
        graph.node.append(helper.make_node(kind, values, [key], **kwargs))
        return key

    center = np.asarray(head["center"], "f")
    basis = np.asarray(head["centers"], "f")
    dim = len(center)
    source = "short_features"
    # An initializer supplies a default input on long recordings and for callers
    # requesting only the existing network outputs.
    graph.initializer.append(
        numpy_helper.from_array(np.zeros((1, dim), np.float32), source)
    )
    graph.input.append(
        helper.make_tensor_value_info(source, TensorProto.FLOAT, ["short_time", dim])
    )
    z = op(
        "Clip",
        [
            op(
                "Div",
                [
                    op("Sub", [source, constant("center", center)]),
                    constant("scale", head["scale"]),
                ],
            ),
            constant("lower", -3.5),
            constant("upper", 3.5),
        ],
    )
    z2 = op("ReduceMean", [op("Mul", [z, z])], axes=[1], keepdims=1)
    squared = op("Add", [z2, constant("norm", (basis * basis).mean(1))])
    cross = op("MatMul", [z, constant("basis", basis.T)])
    dist = op(
        "Relu",
        [op("Sub", [squared, op("Mul", [cross, constant("factor", 2.0 / dim)])])],
    )
    two = constant("two", 2.0)
    feats = [
        op("Mul", [z, constant("linear", 0.32)]),
        op("Mul", [op("Relu", [z]), constant("hinge", 0.22)]),
    ]
    for k, width in enumerate((-1.8, -5.5)):
        feats.append(
            op(
                "Mul",
                [two, op("Exp", [op("Mul", [dist, constant("rbf" + str(k), width)])])],
            )
        )
    xx = op("Concat", feats, axis=1)
    logits = op(
        "Add",
        [
            op("MatMul", [xx, constant("coeff", np.asarray(head["coef"]).T)]),
            constant("bias", head["bias"]),
        ],
    )
    out = op("Sigmoid", [logits])
    ss, aa = "short_stress", "short_activity"
    graph.node.append(helper.make_node("Split", [out], [ss, aa], axis=1))
    one = constant("one", 1.0)
    res = op("Sub", [one, aa])
    quiet = op("Mul", [res, op("Sub", [one, ss])])
    stress = op("Mul", [res, ss])
    op("Concat", [quiet, stress, aa], out="short_p", axis=1)
    graph.output.append(
        helper.make_tensor_value_info("short_p", TensorProto.FLOAT, ["short_time", 3])
    )
    props = {m.key: m.value for m in model.metadata_props}
    props.update(short_trial_version="independent-trials-v1", short_features=str(dim))
    helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def load_variants(
    data, augment=True, catsa_zip=None, compact=False, work_dir=None, progress=None
):
    """Build training records in file order. A cached benchmark only needs labels and person IDs."""
    records = []
    items = []
    feature_store = None
    if augment:
        if work_dir is not None:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
        feature_store = tempfile.TemporaryDirectory(
            prefix="wrist-features-", dir=work_dir
        )
    paths = sorted(data.rglob("*.parquet"))
    for file_number, path in enumerate(paths, 1):
        frame = pd.read_parquet(path, columns=None if augment else ["Phase", "user_id"])
        if frame.empty:
            continue
        subjects = frame.user_id.dropna().unique()
        if len(subjects) != 1:
            raise ValueError(f"Expected one participant in {path}")
        subject = str(subjects[0])
        clock, positions = np.unique(
            frame.index.as_unit("ns").asi8 // BIN_NS, return_inverse=True
        )
        y = labelled_windows(frame, positions, len(clock))
        records.append((None, y, clock, subject, path))
        if not augment:
            continue
        fine = fine_targets(frame, positions, clock, y)
        sensors = frame.reindex(columns=CHANNELS)
        relative = path.relative_to(data).as_posix()
        features = []
        morphology = []
        regimes = []
        designs = []
        for scenario in (None, *TRAIN_OUTAGES):
            damaged = (
                sensors
                if scenario is None
                else perturb(
                    sensors, scenario, recording_rng(9245, relative, scenario)
                )[0]
            )
            raw, _, grid = physiology_windows(damaged)
            if not np.array_equal(grid, clock):
                raise ValueError(f"Augmentation changed the recording clock in {path}")
            x = series_features(raw, grid)
            if not compact:
                if scenario is None:
                    clean_details = beat_details(sensors, grid)
                    detail = clean_details
                elif scenario in ("without_EDA", "eda_temperature"):
                    detail = clean_details.copy()
                    detail[:, 21:] = np.nan
                elif scenario in ("without_BVP", "optical_loss"):
                    detail = clean_details.copy()
                    detail[:, :21] = np.nan
                    detail[:, 14] = 0.0
                elif scenario in ("partial_BVP", "rotating_gaps", "scattered_BVP"):
                    detail = beat_details(damaged, grid)
                else:
                    detail = clean_details
                if scenario is None:
                    cardiac_clean = cardiac_design(
                        cardiac_response(sensors, grid), grid
                    )
                    cardiac_views = {
                        0: cardiac_clean,
                        2: cardiac_clean,
                        3: cardiac_clean,
                    }
                elif scenario == "rotating_gaps":
                    cardiac_views[6] = cardiac_design(
                        cardiac_response(damaged, grid), grid
                    )
            if scenario is None:
                clean_response = autonomic_detail(sensors, grid)
                comparison = clean_response
            elif scenario in ("without_BVP", "optical_loss"):
                comparison = clean_response.copy()
                comparison[:, :19] = np.nan
            elif scenario in ("without_EDA", "eda_temperature"):
                comparison = clean_response.copy()
                comparison[:, 19:] = np.nan
            elif scenario in ("rotating_gaps", "scattered_BVP", "partial_BVP"):
                comparison = autonomic_detail(damaged, grid)
            else:
                comparison = clean_response
            designs.append(response_design(raw, comparison, grid))
            features.append(x)
            if not compact:
                morphology.append(beat_series(detail, grid))
            regimes.append(adaptive_features(x, grid))
        x = np.stack(features)
        row = dict(
            x=x,
            z=np.stack(designs),
            a=np.stack(regimes),
            y=y,
            c=clock,
            f=fine,
            subject=subject,
        )
        row["short"] = collect_short_trials(sensors, y, clock)
        if not compact:
            row.update(
                cardiac=cardiac_views,
                extended=np.concatenate((x, np.stack(morphology)), axis=2),
            )
        # Retain only read-only mappings between recordings. The training matrix
        # otherwise grows with every sensor-outage variant across every person.
        keys = ("x", "z", "a") if compact else ("x", "z", "a", "extended")
        for key in keys:
            destination = Path(feature_store.name) / f"{file_number}-{key}.npy"
            np.save(destination, row[key], allow_pickle=False)
            row[key] = np.load(destination, mmap_mode="r", allow_pickle=False)
        items.append(row)
        if progress is not None and (
            file_number % 10 == 0 or file_number == len(paths)
        ):
            progress(f"Prepared recording {file_number}/{len(paths)}")
    if not records:
        raise ValueError(f"No nonempty Parquet recordings in {data}")
    return {
        "long": records,
        "items": items,
        "external": load_external(catsa_zip) if augment else [],
        "_feature_store": feature_store,
    }


def train_component(items, indices, kind):
    if kind == "segments":
        model = train_segment(items, indices)
        return {"variant": kind, "estimator": model}, model
    selected = (
        [dict(row, x=row["extended"]) for row in items] if kind == "beat" else items
    )
    fit = train_reference(
        selected,
        indices,
        kind,
        787 if kind == "beat" else 239,
        75,
        hard_examples(selected, indices),
    )
    model, center, scale = fit
    pack = {
        "variant": kind,
        "state": model.state_dict(),
        "center": center,
        "scale": scale,
    }
    return pack, fit


def component_predictions(kind, fit, row, variants):
    if kind == "segments":
        probs = np.stack(
            [fit.predict_proba(row["a"][v]).astype(np.float32) for v in variants]
        )
        return probs, np.full(probs.shape[:2], 0.2, np.float32)
    x = row["extended"] if kind == "beat" else row["x"]
    predictions = [predict_ref(fit, x[v], row["c"]) for v in variants]
    return tuple(np.stack(part) for part in zip(*predictions))


CATSA_SHA256 = "8c17d8cec3b9fbae1ef2727357e75f60c837b4ff4e308446ce47435049e4a02a"
CATSA_URL = "https://ndownloader.figshare.com/files/53301287"


def external_records(archive):
    archive = Path(archive)
    if (
        not archive.is_file()
        or hashlib.sha256(archive.read_bytes()).hexdigest() != CATSA_SHA256
    ):
        raise ValueError(
            "Use CATSA.zip version 2 from doi:10.6084/m9.figshare.28690988.v2"
        )
    records = []
    for subject, frame in external_frames(archive):
        sensors = frame.reindex(columns=CHANNELS)
        raw, rows, clock = physiology_windows(sensors)
        y = labelled_windows(frame, rows, len(clock))
        records.append(
            dict(
                x=series_features(raw, clock)[None],
                z=response_design(raw, autonomic_detail(sensors, clock), clock)[None],
                y=y,
                short=collect_short_trials(sensors, y, clock, external=True),
                c=clock,
                f=np.where(y == 0, 0, np.where(y == 1, 4, -1)),
                subject=subject,
            )
        )
    if len(records) != 50:
        raise ValueError("Expected 50 independent CATSA participants")
    return records


def load_external(archive=None):
    """Read the official public archive, or fetch the fixed version for a fresh training run."""
    if archive is not None:
        return external_records(archive)
    import shutil
    import urllib.request

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "CATSA.zip"
        with (
            urllib.request.urlopen(CATSA_URL, timeout=90) as source,
            path.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        return external_records(path)


def fit_component(items, ids, kind, external):
    if kind == "net":
        fit = fit_pair(items, ids, external)
        return dict(
            variant="net", state=fit[0].state_dict(), center=fit[1], scale=fit[2]
        ), fit
    return train_component(items, ids, kind)


def predict_component(kind, fit, row, modes):
    if kind != "net":
        return component_predictions(kind, fit, row, modes)
    outputs = [paired_predict(fit, row["z"][mode], row["c"]) for mode in modes]
    return tuple(np.stack(values) for values in zip(*outputs))


COHORT_ROUTES = (("alternate", "net"), ("alternate", "segments", "net"))

COHORT_TIMES = {("alternate", "net"): 27.0, ("alternate", "segments", "net"): 32.0}

COHORT_MODES = (0, 2, 3, 6)


def fit_balanced_response(rows, indices, people, regularization):
    z = np.concatenate([rows[i][0] for i in indices])
    base = np.concatenate([rows[i][1] for i in indices])
    labels = np.concatenate([rows[i][2] for i in indices])
    w0 = np.concatenate([rows[i][3] for i in indices])
    person = np.concatenate([np.full(len(rows[i][2]), people[i], int) for i in indices])
    center = np.average(z, axis=0, weights=w0)
    scale = np.maximum(0.19, np.sqrt(np.average((z - center) ** 2, axis=0, weights=w0)))
    x = np.ascontiguousarray(np.clip((z - center) / scale, -4, 4), dtype=float)
    coef = []
    for j in range(2):
        keep = (labels >= 0) & ((labels < 2) if j == 0 else True)
        yy = (labels[keep] == j + 1).astype(float)
        who = person[keep]
        xx = x[keep]
        offset = base[keep, j]
        weight = w0[keep].copy()
        # Balance positive and negative evidence within each person, with shrinkage for
        # short tasks and people who contribute only one state.
        within = np.bincount(
            2 * who + yy.astype(int),
            weights=weight,
            minlength=2 * (int(person.max()) + 1),
        ).reshape(-1, 2)
        masses = within[who, yy.astype(int)]
        typical = np.median(masses[masses > 0])
        weight *= np.clip((typical + 3.0) / (masses + 3.0), 0.4, 3.0) ** 0.55
        weight *= np.where(yy == 1, 1.65 if j == 0 else 0.94, 1.0)
        weight /= weight.sum()
        rng = np.random.default_rng(9107 + j)
        plus = []
        minus = []
        for uid in np.unique(who):
            positive = np.flatnonzero((who == uid) & (yy == 1))
            negative = np.flatnonzero((who == uid) & (yy == 0))
            if min(len(positive), len(negative)) < 4:
                continue
            wp = 0.50 + expit(-offset[positive])
            wn = 0.50 + expit(offset[negative])
            plus.extend(rng.choice(positive, 36, p=wp / wp.sum()))
            minus.extend(rng.choice(negative, 36, p=wn / wn.sum()))
        plus = np.array(plus, int)
        minus = np.array(minus, int)
        xd = xx[plus] - xx[minus]
        bd = offset[plus] - offset[minus]

        def objective(theta):
            eta = offset + theta[0] + xx @ theta[1:]
            prob = expit(eta)
            e = weight * (prob - yy)
            error = weight @ (np.logaddexp(0, eta) - yy * eta)
            loss = (
                error
                + 0.5 * regularization * (theta[1:] @ theta[1:])
                + 0.025 * theta[0] ** 2
            )
            grad = np.r_[
                e.sum() + 0.05 * theta[0], xx.T @ e + regularization * theta[1:]
            ]
            if len(bd):
                contrast = 0.35 - bd - xd @ theta[1:]
                mix = 0.26 if j == 0 else 0.14
                loss += mix * np.logaddexp(0, contrast).mean()
                grad[1:] -= mix * (expit(contrast) @ xd) / len(bd)
            return loss, grad

        fitted = minimize(
            objective,
            np.zeros(x.shape[1] + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 65},
        )
        coef.append(fitted.x)
    return dict(
        center=center.tolist(),
        scale=scale.tolist(),
        coef=np.asarray(coef).tolist(),
        local_response=True,
    )


def cohort_value(cm, person_weighted, latency):
    def macro(a):
        return (2 * np.diag(a) / np.maximum(1.0e-9, a.sum(0) + a.sum(1))).mean()

    return float(
        80 * (0.88 * macro(cm) + 0.12 * macro(person_weighted))
        + 20 * (1 - latency / 300.0)
    )


def build_route_views(data, ids, probabilities, edges, route, prior):
    plain = assemble_dev(data, ids, probabilities, edges, route, prior)
    extended = {}
    for mode, records in plain.items():
        rows = {}
        for i, (z, base, y, c, edge) in records.items():
            raw = data[i]["x"][VALIDATION_MODES[mode]]
            local = local_response_context(
                raw, c, [probabilities[k][i][mode] for k in route]
            )
            rows[i] = (np.column_stack((z, local)), base, y, c, edge)
        extended[mode] = rows
    return plain, extended


def select_cohort_calibration(
    data,
    indices,
    bank,
    edges,
    prior,
    trials=None,
    allowed_routes=COHORT_ROUTES,
    ordinary_only=False,
):
    ids = np.asarray(indices, int)
    people = np.unique([r["subject"] for r in data], return_inverse=True)[1]
    splits = list(GroupKFold(4).split(ids, groups=people[ids]))
    best_value = -np.inf
    champion = None
    for route in allowed_routes:
        ordinary, context = build_route_views(data, ids, bank, edges, route, prior)
        for method, views, alpha in [
            ("ordinary", ordinary, 0.12),
            ("ordinary", ordinary, 0.45),
            ("balanced", context, 0.14),
            ("balanced", context, 0.38),
        ]:
            if ordinary_only and method != "ordinary":
                continue
            rows = pool_rows(data, ids, views)

            def fit(group):
                return (
                    fit_residual(rows, group, alpha)
                    if method == "ordinary"
                    else fit_balanced_response(rows, group, people, alpha)
                )

            levels = (0.55, 1.0, 1.45)
            raw = {v: {} for v in levels}
            masked = {}
            for a, b in splits:
                model = fit(ids[a])
                for i in ids[b]:
                    z, base, *rest = views[0][i]
                    for level in levels:
                        raw[level][i] = residual_prob(z, base, model, level)
                    z, base, *rest = views[6][i]
                    masked[i] = residual_prob(z, base, model, 1.0)
            actual = np.concatenate([data[i]["y"] for i in ids])
            good = actual >= 0
            person = np.concatenate(
                [np.full(len(data[i]["y"]), people[i], int) for i in ids]
            )[good]
            sizes = np.bincount(person)[person]
            weights = sizes.mean() / sizes
            y = actual[good]
            latency = COHORT_TIMES[route] + (2.0 if method == "balanced" else 0.0)
            person_den = {
                p: sum((data[i]["y"] >= 0).sum() for i in ids if people[i] == p)
                for p in np.unique(people[ids])
            }
            scale = {i: np.mean(sizes) / person_den[people[i]] for i in ids}
            selected = None
            for level in levels:
                p = np.concatenate([raw[level][i] for i in ids])[good]
                cand = []
                for sw in (0.85, 1.05, 1.25, 1.48, 1.75):
                    for aw in (0.85, 1.0, 1.18):
                        pred = (p * np.array([1.0, sw, aw])).argmax(1)
                        enc = 3 * y + pred
                        cm = np.bincount(enc, minlength=9).reshape(3, 3)
                        balanced = np.bincount(
                            enc, weights=weights, minlength=9
                        ).reshape(3, 3)
                        cand.append((cohort_value(cm, balanced, latency), sw, aw))
                for _, sw, aw in sorted(cand, reverse=True)[:3]:
                    for switch in (0.0, 2.0, 4.0):
                        cm = np.zeros((3, 3), int)
                        bal = np.zeros((3, 3))
                        mg = np.zeros((3, 3), int)
                        for i in ids:
                            _, _, y0, c, e = views[0][i]
                            val = y0 >= 0
                            g = smooth_activity(raw[level][i], c, e, sw, aw, switch)
                            m = np.bincount(3 * y0[val] + g[val], minlength=9).reshape(
                                3, 3
                            )
                            cm += m
                            bal += m * scale[i]
                            yh = (masked[i] * [1, sw, aw]).argmax(1)
                            mg += np.bincount(
                                3 * y0[val] + yh[val], minlength=9
                            ).reshape(3, 3)
                        score = 0.96 * cohort_value(
                            cm, bal, latency + (0.25 if switch else 0.0)
                        ) + 0.04 * cohort_value(mg, mg, latency)
                        summary = summarize(cm)
                        rec = dict(
                            route="+".join(route),
                            method=method,
                            regularization=alpha,
                            strength=level,
                            stress_weight=sw,
                            activity_weight=aw,
                            switch_cost=switch,
                            selection_score=score,
                            macro=summary["macro"],
                            stress=summary["stress"],
                        )
                        if trials is not None:
                            trials.append(rec)
                        if selected is None or score > selected["selection_score"]:
                            selected = rec
            if selected["selection_score"] > best_value:
                best_value = selected["selection_score"]
                champion = (route, fit(ids), selected)
    return champion


def average_component_weights(full, replicas, original_fraction=0.60):
    """Average same-seed networks fit on different training participants.

    Express each input projection in the full fit's normalization before
    combining its parameters. Runtime still evaluates one network per component.
    """
    import torch

    kind = full["variant"]
    if kind == "segments" or not replicas:
        return full
    key = "ch" if kind == "net" else "input"
    name = key + ".weight"
    bias = key + ".bias"
    center = torch.as_tensor(full["center"], dtype=torch.float64)
    scale = torch.as_tensor(full["scale"], dtype=torch.float64)
    pieces = [full, *replicas]
    weights = [original_fraction] + [(1.0 - original_fraction) / len(replicas)] * len(
        replicas
    )
    sums = {}
    for piece, mass in zip(pieces, weights):
        state = {k: v.detach().double() for k, v in piece["state"].items()}
        other_center = torch.as_tensor(piece["center"], dtype=torch.float64)
        other_scale = torch.as_tensor(piece["scale"], dtype=torch.float64)
        offset = (center - other_center) / other_scale
        projection = state[name][:, :, 0]
        state[bias] = state[bias] + projection @ offset
        state[name] = state[name] * (scale / other_scale)[None, :, None]
        for k, parameter in state.items():
            if k in sums:
                sums[k].add_(parameter, alpha=mass)
            else:
                sums[k] = parameter * mass
    return {**full, "state": {k: v.to(full["state"][k].dtype) for k, v in sums.items()}}


def stamp_profile(blob, profile):
    import onnx

    graph = onnx.load_model_from_string(blob)
    fields = {entry.key: entry.value for entry in graph.metadata_props}
    fields["cohort_profile"] = profile
    onnx.helper.set_model_props(graph, fields)
    return graph.SerializeToString()


DEFAULT_PROFILE = "cohort"


def select_and_fit(
    variants,
    indices,
    path,
    trials=None,
    progress=None,
    calibration_peers=(),
    external_indices=None,
    profile=DEFAULT_PROFILE,
):
    """Fit on every supplied training person. All calibration probabilities come from withheld people.

    During leave-one-person-out replay, the other members of the person's reference
    fold serve as an extra inner holdout. This permits safe model reuse across folds.
    """
    if profile not in ("cohort", "triplet", "average", "average_triplet"):
        raise ValueError("Unknown training profile")
    use_average = profile.startswith("average")
    use_triplet = profile.endswith("triplet")
    data = variants["items"]
    indices = np.asarray(list(indices), dtype=int)
    if len(set(data[i]["subject"] for i in indices)) < 6:
        raise ValueError("At least six Huawei participants are required")
    ext = (
        variants["external"]
        if external_indices is None
        else [variants["external"][j] for j in external_indices]
    )
    prior = fit_external_prior(ext)
    peers = np.asarray(list(calibration_peers), dtype=int)
    if len(peers) and not set(peers).issubset(indices):
        raise ValueError("Calibration peers must belong to the training set")
    if len(peers):
        peer_people = {data[i]["subject"] for i in peers}
        if set(peers) != {int(i) for i in indices if data[i]["subject"] in peer_people}:
            raise ValueError("A calibration participant must enter with all recordings")
    core = np.setdiff1d(indices, peers)
    groups = [data[i]["subject"] for i in core]
    parts = [(core[a], core[b]) for a, b in GroupKFold(3).split(core, groups=groups)]
    if len(peers):
        parts.append((core, peers))
    kinds = ("alternate", "segments", "net")
    bank = {k: {} for k in kinds}
    edge = {k: {} for k in kinds}
    replicas = {k: [] for k in kinds}
    for fold, (train, valid) in enumerate(parts, 1):
        if {data[i]["subject"] for i in train} & {data[i]["subject"] for i in valid}:
            raise ValueError("Calibration participant overlap")
        for k in kinds:
            if progress is not None:
                progress(f"Fitting {k} on calibration split {fold}/{len(parts)}")
            pack, fitted = fit_component(data, train, k, ext)
            if use_average and k != "segments":
                replicas[k].append(pack)
            for i in valid:
                p, b = predict_component(k, fitted, data[i], VALIDATION_MODES)
                bank[k][int(i)] = p
                edge[k][int(i)] = b
    if progress is not None:
        progress("Selecting calibration using held-out training participants")
    route, cal, select = select_cohort_calibration(
        data,
        indices,
        bank,
        edge,
        prior,
        trials,
        allowed_routes=(COHORT_ROUTES[1],) if use_triplet else COHORT_ROUTES,
        ordinary_only=use_triplet,
    )
    rule = {
        k: select[k]
        for k in ("strength", "stress_weight", "activity_weight", "switch_cost")
    }
    models = {}
    probe = data[indices[0]]
    expected = {}
    bound = {}
    for kind in route:
        if progress is not None:
            progress(f"Fitting final {kind} model")
        models[kind], fit = fit_component(data, indices, kind, ext)
        if use_average and kind != "segments":
            models[kind] = average_component_weights(models[kind], replicas[kind])
            fit[0].load_state_dict(models[kind]["state"])
        expected[kind], bound[kind] = predict_component(kind, fit, probe, (0, 3))
    corpus = training_id([variants["long"][i][4] for i in indices])
    blob = make_response_graph(
        models, prior, cal, rule, corpus, select["macro"], base_graph
    )
    short_head = fit_short_response([data[i] for i in indices], ext)
    blob = add_short_graph(blob, short_head)
    blob = stamp_profile(blob, profile)
    model = TechArenaModel(blob)
    for pos, v in enumerate((0, 3)):
        for seg in temporal_segments(probe["c"]):
            feed = {}
            if "alternate" in route or "beat" in route:
                feed["physiology"] = probe["x"][v, seg]
            if "segments" in route:
                feed["regimes"] = probe["a"][v, seg]
            if "beat" in route:
                feed["pulse_detail"] = probe["extended"][v, seg, 301:]
            if "net" in route:
                feed["response"] = probe["z"][v, seg]
            result = dict(zip(model.outputs, model.session.run(model.outputs, feed)))
            for k in route:
                np.testing.assert_allclose(
                    result[k + "_p"], expected[k][pos, seg], atol=4.0e-5, rtol=1.0e-4
                )
                if k != "segments":
                    np.testing.assert_allclose(
                        result[k + "_b"][:, 0],
                        bound[k][pos, seg],
                        atol=4.0e-5,
                        rtol=1.0e-4,
                    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        tmp = Path(directory) / "submission.onnx"
        tmp.write_bytes(blob)
        tmp.replace(path)
    return rule, select["macro"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("TechArena-Finland-2026/data")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("submission_model.onnx")
    )
    parser.add_argument(
        "--catsa-zip",
        type=Path,
        help="Optional local CATSA v2 archive for fully offline training",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        help="Save participant-grouped training comparisons to CSV",
    )
    parser.add_argument(
        "--parameter-checks",
        type=Path,
        help="Save the nested development settings and metrics to CSV",
    )
    parser.add_argument(
        "--profile",
        choices=["cohort", "triplet", "average", "average_triplet"],
        default=DEFAULT_PROFILE,
        help="Participant-tested model and calibration recipe",
    )
    args = parser.parse_args()
    print("Preparing sensor features and training examples", flush=True)

    def show_progress(message):
        print(message, flush=True)

    ds = load_variants(
        args.data,
        catsa_zip=args.catsa_zip,
        compact=True,
        work_dir=args.output.parent,
        progress=show_progress,
    )
    trials = []

    rule, score = select_and_fit(
        ds,
        range(len(ds["long"])),
        args.output,
        trials,
        progress=show_progress,
        profile=args.profile,
    )
    for f in (args.trials, args.parameter_checks):
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(trials).to_csv(f, index=False)
    print("Saved", args.output, "inner macro F1", score, "settings", rule)


if __name__ == "__main__":
    main()
