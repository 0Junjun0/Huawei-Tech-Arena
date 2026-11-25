import os
import pathlib
import sys
import time
import tracemalloc
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
NOTES_PATH = ROOT_DIR / "notes" / "experiment_notes.md"
SUBMISSION_DIR = ROOT_DIR / "submission"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "experiments", DATA_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_model as tm
from auto_log_experiment import append_experiment_to_notes


def map_to_3_classes(classes, raw_probs: np.ndarray) -> np.ndarray:
    mapped = np.zeros((len(raw_probs), len(tm.LABELS)), dtype=np.float32)
    for col_i, cls_val in enumerate(classes):
        mapped[:, int(cls_val)] = raw_probs[:, col_i]
    return mapped


def eval_oof(name: str, y: np.ndarray, probs: np.ndarray) -> tuple[float, float, np.ndarray, str]:
    raw_preds = np.argmax(probs, axis=1)
    raw_f1 = float(f1_score(y, raw_preds, average="macro", zero_division=0))

    best_f1 = raw_f1
    best_w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for w_stress in np.arange(0.90, 1.95, 0.05):
        for w_rest in [0.90, 0.95, 1.00, 1.05]:
            for w_act in [0.85, 0.90, 0.95, 1.00]:
                w = np.array([w_rest, w_stress, w_act], dtype=np.float32)
                preds_w = np.argmax(probs * w, axis=1)
                f1_w = float(f1_score(y, preds_w, average="macro", zero_division=0))
                if f1_w > best_f1:
                    best_f1 = f1_w
                    best_w = w

    per_class = f1_score(y, raw_preds, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_str = (
        f"Rest: {per_class[0]:.4f} | Stress: {per_class[1]:.4f} | Activity: {per_class[2]:.4f}"
    )

    print(f"\n=== {name} ===")
    print(f"Raw OOF Macro-F1:   {raw_f1:.4f} ({raw_f1 * 80:.2f}/80)")
    print(
        f"Tuned OOF Macro-F1: {best_f1:.4f} ({best_f1 * 80:.2f}/80) | "
        f"Best weights [Rest, Stress, Activity]: {[round(float(x), 4) for x in best_w]}"
    )
    print(classification_report(y, raw_preds, target_names=tm.LABELS, digits=4, zero_division=0))
    return raw_f1, best_f1, best_w, per_class_str


def main() -> None:
    tracemalloc.start()
    data_root = DATA_DIR if DATA_DIR.exists() else ROOT_DIR

    X, y, groups = tm.build_dataset(data_root, win_sec=10)
    print(f"Loaded dataset: X={X.shape}, y={y.shape}, subjects={len(np.unique(groups))}")

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rf_probs = np.zeros((len(y), len(tm.LABELS)), dtype=np.float32)
    hgb_probs = np.zeros((len(y), len(tm.LABELS)), dtype=np.float32)

    t0 = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        rf_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=18,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42 + fold,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        rf_pipe.fit(X[tr_idx], y[tr_idx])
        rf_probs[va_idx] = map_to_3_classes(
            rf_pipe.named_steps["rf"].classes_, rf_pipe.predict_proba(X[va_idx])
        )

        hgb_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=25,
                        l2_regularization=2.0,
                        class_weight="balanced",
                        random_state=42 + fold,
                    ),
                ),
            ]
        )
        hgb_pipe.fit(X[tr_idx], y[tr_idx])
        hgb_probs[va_idx] = map_to_3_classes(
            hgb_pipe.named_steps["hgb"].classes_, hgb_pipe.predict_proba(X[va_idx])
        )

    cv_sec = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / (1024 * 1024)

    print(f"\n5-Fold CV completed in {cv_sec:.2f}s | Peak traced RAM: {peak_ram_mb:.2f} MB")

    eval_oof("RF (42 features)", y, rf_probs)
    eval_oof("HGB (42 features)", y, hgb_probs)
    blend_probs = 0.5 * rf_probs + 0.5 * hgb_probs
    blend_raw, blend_tuned, blend_w, blend_per_class = eval_oof(
        "Blend 0.5 RF + 0.5 HGB (42 features)", y, blend_probs
    )

    onnx_path = SUBMISSION_DIR / "submission_model.onnx"
    onnx_size_mb = (onnx_path.stat().st_size / (1024 * 1024)) if onnx_path.exists() else None
    w_list = [round(float(x), 4) for x in blend_w]

    append_experiment_to_notes(
        exp_id="EXP-BLEND-42",
        title="0.5 RF + 0.5 HGB Probability Blend on 42 Features (Isolated Folder)",
        hypothesis=(
            "Averaging calibrated/balanced class probabilities from RandomForestClassifier and "
            "HistGradientBoostingClassifier on the 42-feature pipeline improves grouped OOF "
            "Macro-F1 over each standalone model."
        ),
        changes_made=(
            "Created isolated blend_exp/run_blend_42.py running 5-fold GroupKFold on the 42-feature "
            "dataset with 0.5 RF + 0.5 HGB probability blending while keeping submission/ untouched."
        ),
        files_modified=["blend_exp/__init__.py", "blend_exp/run_blend_42.py", "notes/experiment_notes.md"],
        validation_method="5-fold GroupKFold across 35 subjects (14,890 windows, 42 features, 1 thread)",
        prev_macro_f1="RF-42: Raw 0.5504 / Tuned 0.5859 | HGB-42: Raw 0.5665 / Tuned 0.5777",
        new_macro_f1=f"Raw {blend_raw:.4f} | Tuned {blend_tuned:.4f} (weights {w_list})",
        per_class_f1=blend_per_class,
        met_sec=None,
        peak_ram_mb=peak_ram_mb,
        onnx_size_mb=onnx_size_mb,
        quality_score=blend_tuned * 80.0,
        conclusion=(
            f"Blend reached Raw {blend_raw:.4f} and Tuned {blend_tuned:.4f} (5-fold fit {cv_sec:.2f}s); "
            "HGB still faces skl2onnx export constraints and dual-model inference adds latency overhead, "
            "so keep submission/ baseline untouched until ONNX/speed trade-offs are checked."
        ),
        notes_path=NOTES_PATH,
    )


if __name__ == "__main__":
    main()