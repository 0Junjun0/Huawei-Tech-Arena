"""Tune on participant-separated folds and refit all participants.
Example: python train_model.py --data TechArena-Finland-2026/data
Requires scikit-learn, protobuf>=6.31,<7, and skl2onnx in addition to the challenge runtime.
"""

import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupKFold
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from submission_model import window_features, decisions, TechArenaModel

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


def dataset(data, context):
    records = []
    for path in sorted(data.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        subjects = frame.user_id.dropna().unique()
        if len(subjects) != 1:
            raise ValueError(f"Expected a single participant in {path}")
        x, rows, clock = window_features(frame, context)
        phase = frame.Phase.astype("string").str.strip().str.lower()
        labels = phase.map(PHASES).mask(phase.str.contains("rpm", na=False), 2)
        keep = labels.notna().to_numpy()
        counts = np.bincount(
            3 * rows[keep] + labels[keep].to_numpy(dtype=int), minlength=len(x) * 3
        ).reshape(-1, 3)
        y = counts.argmax(axis=1)
        y[counts.sum(axis=1) == 0] = -1
        records.append((x, y, clock, str(subjects[0]), path))
    return records


def fit(records, kind):
    x = np.concatenate([r[0] for r in records])
    y = np.concatenate([r[1] for r in records])
    valid = y >= 0
    if len(np.unique(y[valid])) != 3:
        raise ValueError("A training fold lacks a class")
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=4,
    )
    if kind == "boost":
        model = HistGradientBoostingClassifier(
            max_iter=130,
            max_leaf_nodes=15,
            learning_rate=0.075,
            l2_regularization=5,
            class_weight="balanced",
            early_stopping=False,
            random_state=42,
        )
    else:
        model = forest
    return model.fit(x[valid], y[valid])


def write_model(model, rule, path):
    graph = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, model.n_features_in_]))],
        target_opset=15,
        options={id(model): {"zipmap": False}},
    )
    graph.metadata_props.add(key="decision", value=json.dumps(rule))
    path.write_bytes(graph.SerializeToString())


def score(predictions, rule):
    cm = np.zeros((3, 3), int)
    for prob, y, clock in predictions:
        pred = decisions(prob, clock, **rule)
        valid = y >= 0
        cm += np.bincount(3 * y[valid] + pred[valid], minlength=9).reshape(3, 3)
    f1 = np.divide(2 * cm.diagonal(), np.maximum(cm.sum(axis=0) + cm.sum(axis=1), 1))
    return f1.mean(), cm


def select_and_fit(variants, indices, path):
    """Use only development participants for all context and decision tuning."""
    groups = [variants["long"][i][3] for i in indices]
    best = -1.0
    for kind in ("forest", "boost"):
        context = "long"
        records = variants[context]
        pool = [records[i] for i in indices]
        predictions = []
        for tr, va in GroupKFold(3).split(pool, groups=groups):
            model = fit([pool[i] for i in tr], kind)
            for i in va:
                x, y, clock, _, _ = pool[i]
                predictions.append((model.predict_proba(x), y, clock))
        for radius in (0, 1, 3, 6):
            for weight in (1.0, 0.85, 1.15, 1.4):
                rule = dict(radius=radius, stress_weight=weight)
                f1, _ = score(predictions, rule)
                if f1 > best:
                    best, chosen, decision = f1, kind, rule
    rule = dict(context="long", **decision)
    model = fit([variants["long"][i] for i in indices], chosen)
    write_model(model, rule, path)
    return {**rule, "estimator": chosen}, best


def load_variants(data):
    return {"long": dataset(data, "long")}


def main(data, output):
    variants = load_variants(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        path = Path(tmp) / "candidate.onnx"
        rule, development = select_and_fit(variants, range(len(variants["long"])), path)
        model = TechArenaModel(path)
        model.run(pd.read_parquet(variants["long"][0][4]))
        path.replace(output)
    print("Final settings", rule, "selection macro-F1", development)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("TechArena-Finland-2026/data"))
    p.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("submission_model.onnx")
    )
    args = p.parse_args()
    main(args.data, args.output)
