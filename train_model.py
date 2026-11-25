# This section disables multi-core worker pools before importing training and ONNX export libraries.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pathlib
import re
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from submission_model import LABELS, SENSOR_COLS, compute_window_features, smooth_window_probs

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
STRESS_PHASES = {"stroop", "tmct", "real opinion", "opposite opinion", "subtraction"}
ACTIVITY_PHASES = {"warm-up", "sprint", "cooldown"}
REST_PHASES = {"baseline", "rest"}


def find_dataset_root() -> pathlib.Path:
    """This subprogram locates the directory containing the competition data/ folder."""
    for candidate in [pathlib.Path("."), pathlib.Path("TechArena-Finland-2026")]:
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find 'data/' folder in current directory or 'TechArena-Finland-2026/'.")


def map_phase_to_activity(phase_val) -> str | None:
    """This subprogram maps protocol Phase strings to Rest, Stress, Activity, or None using phases_to_activity.txt rules."""
    if pd.isna(phase_val):
        return None
    low = str(phase_val).strip().lower()
    if low in STRESS_PHASES:
        return "Stress"
    if low in REST_PHASES:
        return "Rest"
    if low in ACTIVITY_PHASES or "rpm" in low:
        return "Activity"
    return None


def build_dataset(root_dir: pathlib.Path, win_sec: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[slice]]:
    """This subprogram loads all participant Parquet files, builds 10s windows, and records window purity and recording slices."""
    X_list, y_list, groups_list, purity_list, file_slices = [], [], [], [], []
    data_dir = root_dir / "data"
    offset = 0

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        subject_id = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        for col in SENSOR_COLS:
            if col not in df.columns:
                df[col] = np.nan

        if "Phase" not in df.columns:
            continue

        _, bin_ids, X_win = compute_window_features(df, win_sec=win_sec)
        n_bins = len(X_win)

        mapped_phase = df["Phase"].map(map_phase_to_activity)
        valid_mask = mapped_phase.notna().to_numpy()
        if not np.any(valid_mask):
            continue

        v_bins = bin_ids[valid_mask]
        v_labels = mapped_phase.iloc[valid_mask].map(LABEL_TO_IDX).to_numpy(dtype=np.int64)

        counts = np.zeros((n_bins, 3), dtype=np.int64)
        np.add.at(counts, (v_bins, v_labels), 1)
        totals = counts.sum(axis=1)
        has_label = totals > 0
        if not np.any(has_label):
            continue

        valid_counts = counts[has_label]
        y_win = np.argmax(valid_counts, axis=1)
        purity = np.max(valid_counts, axis=1) / totals[has_label]

        X_list.append(X_win[has_label])
        y_list.append(y_win)
        groups_list.append(np.full(len(y_win), subject_id, dtype=object))
        purity_list.append(purity)

        file_slices.append(slice(offset, offset + len(y_win)))
        offset += len(y_win)

    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list),
        np.concatenate(purity_list),
        file_slices,
    )


def make_ensemble_pipeline(seed: int = 42, mode: str = "hgb_vote") -> Pipeline:
    """This subprogram builds a single-threaded classifier pipeline compatible with skl2onnx export."""
    if mode == "hgb_vote":
        ensemble = VotingClassifier(
            estimators=[
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        max_depth=6,
                        learning_rate=0.05,
                        l2_regularization=1.5,
                        min_samples_leaf=15,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        max_depth=16,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=180,
                        max_depth=15,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=seed + 100,
                    ),
                ),
            ],
            voting="soft",
            weights=np.array([2.0, 1.0, 1.0], dtype=np.float64),
            flatten_transform=False,
            n_jobs=1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("ens", ensemble)])

    if mode == "gb_vote":
        ensemble = VotingClassifier(
            estimators=[
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=120,
                        max_depth=4,
                        learning_rate=0.06,
                        subsample=0.8,
                        random_state=seed,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        max_depth=16,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=180,
                        max_depth=15,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=seed + 100,
                    ),
                ),
            ],
            voting="soft",
            weights=np.array([2.0, 1.0, 1.0], dtype=np.float64),
            flatten_transform=False,
            n_jobs=1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("ens", ensemble)])

    # Standalone ExtraTrees fallback if VotingClassifier fails ONNX conversion
    clf = ExtraTreesClassifier(
        n_estimators=260,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])


def search_best_weights(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """This subprogram finds regularized class probability multipliers maximizing Macro-F1."""
    best_score = -1.0
    best_w = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for w_stress in np.arange(0.85, 1.45, 0.05):
        for w_rest in [0.85, 0.92, 1.00, 1.08]:
            for w_act in [0.85, 0.92, 1.00, 1.08]:
                w = np.array([w_rest, w_stress, w_act], dtype=np.float32)
                preds = np.argmax(probs * w, axis=1)
                reg_penalty = 0.002 * float(np.sum((w - 1.0) ** 2))
                score = f1_score(y_true, preds, average="macro") - reg_penalty
                if score > best_score:
                    best_score = score
                    best_w = w
    return best_w


def update_submission_weights(sub_py_path: pathlib.Path, weights: np.ndarray) -> None:
    """This subprogram writes the tuned CLASS_WEIGHTS list directly into submission_model.py."""
    if not sub_py_path.exists():
        return
    text = sub_py_path.read_text(encoding="utf-8")
    new_line = f"CLASS_WEIGHTS = {[round(float(w), 4) for w in weights]}"
    updated = re.sub(r"^CLASS_WEIGHTS\s*=.*$", new_line, text, flags=re.MULTILINE)
    sub_py_path.write_text(updated, encoding="utf-8")


def main() -> None:
    """This subprogram runs leak-free nested subject-grouped CV, exports the ONNX model, and verifies parity."""
    root_dir = find_dataset_root()
    X, y, groups, purity, file_slices = build_dataset(root_dir, win_sec=10)
    print(f"Dataset shape: X={X.shape}, y={y.shape}, unique_subjects={len(np.unique(groups))}")

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(y), 3), dtype=np.float32)
    fold_indices = list(gkf.split(X, y, groups))

    # Train ensemble across subject-grouped folds
    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        clean_tr = tr_idx[(purity[tr_idx] >= 0.75) & (X[tr_idx, 6] > 1e-5)]
        if len(clean_tr) < 100:
            clean_tr = tr_idx

        clf = make_ensemble_pipeline(seed=42 + fold, mode="hgb_vote")
        clf.fit(X[clean_tr], y[clean_tr])
        oof_probs[va_idx] = clf.predict_proba(X[va_idx]).astype(np.float32)

    # Apply per-recording Gaussian probability smoothing without crossing recording boundaries
    smoothed_oof = np.zeros_like(oof_probs)
    for sl in file_slices:
        smoothed_oof[sl] = smooth_window_probs(oof_probs[sl])

    raw_f1 = f1_score(y, np.argmax(smoothed_oof, axis=1), average="macro")
    print(f"Smoothed OOF Macro-F1 (unweighted): {raw_f1:.4f} | Quality: {raw_f1 * 80:.2f} / 80.0")

    # Nested out-of-fold evaluation of threshold multipliers to prevent overfitting
    nested_preds = np.zeros_like(y)
    for tr_idx, va_idx in fold_indices:
        w_fold = search_best_weights(smoothed_oof[tr_idx], y[tr_idx])
        nested_preds[va_idx] = np.argmax(smoothed_oof[va_idx] * w_fold, axis=1)

    nested_f1 = f1_score(y, nested_preds, average="macro")
    print(f"Nested Tuned OOF Macro-F1 (honest estimate): {nested_f1:.4f} | Quality: {nested_f1 * 80:.2f} / 80.0")
    print("\nNested Per-Class Breakdown:")
    print(classification_report(y, nested_preds, target_names=LABELS, digits=4))

    final_weights = search_best_weights(smoothed_oof, y)
    print(f"Deployment weights [Rest, Stress, Activity]: {[round(float(w), 4) for w in final_weights]}")
    update_submission_weights(pathlib.Path("submission_model.py"), final_weights)

    # Fit final model on clean windows across all subjects and export to ONNX with automatic fallbacks
    all_clean = (purity >= 0.75) & (X[:, 6] > 1e-5)
    X_fit, y_fit = (X[all_clean], y[all_clean]) if np.sum(all_clean) >= 100 else (X, y)

    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    out_onnx = pathlib.Path("submission_model.onnx")
    onnx_model = None
    final_model = None

    for mode, opt_cls in [
        ("hgb_vote", VotingClassifier),
        ("gb_vote", VotingClassifier),
        ("et_single", ExtraTreesClassifier),
    ]:
        try:
            candidate = make_ensemble_pipeline(seed=42, mode=mode)
            candidate.fit(X_fit, y_fit)
            onnx_model = convert_sklearn(
                candidate,
                initial_types=initial_type,
                options={opt_cls: {"zipmap": False}},
                target_opset=15,
            )
            final_model = candidate
            print(f"Successfully converted model mode='{mode}' to ONNX.")
            break
        except Exception as exc:
            print(f"ONNX export skipped mode='{mode}' ({exc}); trying next fallback...")

    if onnx_model is None or final_model is None:
        raise RuntimeError("All ONNX export modes failed.")

    out_onnx.write_bytes(onnx_model.SerializeToString())

    # Verify ONNX Runtime probabilities match scikit-learn probabilities
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(out_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    onnx_probs = np.asarray(sess.run(None, {"float_input": X[:200]})[1], dtype=np.float32)
    sk_probs = final_model.predict_proba(X[:200]).astype(np.float32)
    assert np.allclose(sk_probs, onnx_probs, atol=1e-4), "ONNX probability parity mismatch!"
    print(f"Verified & saved {out_onnx} ({out_onnx.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()