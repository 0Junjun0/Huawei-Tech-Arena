import pathlib
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

import train_model

OUTLIER_SUBJECTS = {"f17", "f14", "S03", "S06", "S07"}


def tune_class_weights(y_true: np.ndarray, oof_probs: np.ndarray, base_f1: float):
    best_f1 = base_f1
    best_w = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for w_stress in np.arange(1.0, 2.1, 0.1):
        for w_rest in [0.9, 0.95, 1.0, 1.05]:
            for w_act in [0.8, 0.85, 0.9, 0.95, 1.0]:
                w = np.array([w_rest, w_stress, w_act], dtype=np.float32)
                preds = np.argmax(oof_probs * w, axis=1)
                score = f1_score(y_true, preds, average="macro", zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_w = w
    return best_f1, best_w


def make_estimator(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def measure_onnx_speed(onnx_bytes: bytes, sample_feats: np.ndarray, n_runs: int = 30):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_bytes, sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    x_in = sample_feats.astype(np.float32)

    _ = sess.run(None, {in_name: x_in})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {in_name: x_in})
        times.append(time.perf_counter() - t0)

    met = float(np.mean(times))
    speed_pts = max(0.0, min(20.0, (1.0 - met / 0.3) * 20.0))
    return met, speed_pts


def evaluate_variant(name: str, kind: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, drop_outliers: bool):
    unique_subjects = np.unique(groups)
    gkf = GroupKFold(n_splits=min(5, len(unique_subjects)))
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), 3), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        if drop_outliers:
            keep_tr = np.array([groups[i] not in OUTLIER_SUBJECTS for i in tr_idx])
            tr_fit_idx = tr_idx[keep_tr]
        else:
            tr_fit_idx = tr_idx

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", make_estimator(kind, seed=42 + fold)),
        ])
        pipe.fit(X[tr_fit_idx], y[tr_fit_idx])

        raw_probs = pipe.predict_proba(X[va_idx])
        mapped = np.zeros((len(va_idx), 3), dtype=np.float32)
        for col_i, cls_val in enumerate(pipe.classes_):
            mapped[:, int(cls_val)] = raw_probs[:, col_i]

        oof_probs[va_idx] = mapped
        oof_preds[va_idx] = np.argmax(mapped, axis=1)

    raw_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    tuned_f1, best_w = tune_class_weights(y, oof_probs, raw_f1)

    # Fit full model & export ONNX in memory to check size and speed
    if drop_outliers:
        full_mask = np.array([g not in OUTLIER_SUBJECTS for g in groups])
        X_full, y_full = X[full_mask], y[full_mask]
    else:
        X_full, y_full = X, y

    full_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", make_estimator(kind, seed=42)),
    ])
    full_pipe.fit(X_full, y_full)

    clf_step = full_pipe.named_steps["clf"]
    init_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    opts_map = {id(clf_step): {"zipmap": False}} if kind in {"rf", "et"} else {}
    onnx_model = convert_sklearn(full_pipe, initial_types=init_type, options=opts_map, target_opset=15)
    onnx_bytes = onnx_model.SerializeToString()
    size_kb = len(onnx_bytes) / 1024.0
    met, speed_pts = measure_onnx_speed(onnx_bytes, X[:250])

    print(f"\n--- {name} ---")
    print(f"Raw OOF Macro-F1:   {raw_f1:.4f} (Est. {raw_f1 * 80:.2f}/80)")
    print(
        f"Tuned OOF Macro-F1: {tuned_f1:.4f} (Est. {tuned_f1 * 80:.2f}/80) | "
        f"Weights: {[round(float(v), 2) for v in best_w]}"
    )
    print(f"ONNX Size: {size_kb:.1f} KB | Classifier-only MET: {met:.4f} s (Est. Speed: {speed_pts:.2f}/20)")
    print(classification_report(y, oof_preds, target_names=train_model.LABELS, digits=4, zero_division=0))


def main():
    root_dir = pathlib.Path("TechArena-Finland-2026")
    X, y, groups = train_model.build_dataset(root_dir, win_sec=10)
    print(f"Loaded X={X.shape}, y={y.shape}, subjects={len(np.unique(groups))}")

    evaluate_variant("rf_44_baseline", "rf", X, y, groups, drop_outliers=False)
    evaluate_variant("hgb_44", "hgb", X, y, groups, drop_outliers=False)
    evaluate_variant("et_44", "et", X, y, groups, drop_outliers=False)
    evaluate_variant("rf_44_drop_outliers", "rf", X, y, groups, drop_outliers=True)


if __name__ == "__main__":
    main()