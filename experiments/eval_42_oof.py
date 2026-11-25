import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pathlib
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "TechArena-Finland-2026"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_model as tm


def oof_probs_3(clf, X_val: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(X_val)
    out = np.zeros((len(X_val), 3), dtype=np.float32)
    classes = getattr(clf, "classes_", None)
    if classes is None and hasattr(clf, "named_steps"):
        last = list(clf.named_steps.values())[-1]
        classes = getattr(last, "classes_", np.arange(3))
    for col_idx, cls_id in enumerate(classes):
        out[:, int(cls_id)] = probs[:, col_idx]
    return out


def main() -> None:
    X, y, groups = tm.build_dataset(DATA_DIR, win_sec=10)
    assert X.shape[1] == 42, f"Expected 42 features, got {X.shape[1]}"

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rf_oof = np.zeros((len(y), 3), dtype=np.float32)
    hgb_oof = np.zeros((len(y), 3), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        rf_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42 + fold,
                n_jobs=1,
            )),
        ])
        rf_pipe.fit(X[tr_idx], y[tr_idx])
        rf_oof[va_idx] = oof_probs_3(rf_pipe, X[va_idx])

        hgb_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42 + fold,
            )),
        ])
        hgb_pipe.fit(X[tr_idx], y[tr_idx])
        hgb_oof[va_idx] = oof_probs_3(hgb_pipe, X[va_idx])

    blend_oof = 0.5 * rf_oof + 0.5 * hgb_oof
    weights = np.array([0.95, 1.80, 0.85], dtype=np.float32)

    for label, probs in [("RF-42", rf_oof), ("HGB-42", hgb_oof), ("Blend 0.5/0.5 (raw)", blend_oof), ("Blend 0.5/0.5 (weighted)", blend_oof * weights)]:
        preds = np.argmax(probs, axis=1)
        mf1 = float(f1_score(y, preds, average="macro", zero_division=0))
        _, _, cf1, _ = precision_recall_fscore_support(y, preds, labels=[0, 1, 2], zero_division=0)
        print(
            f"{label:26s} | Macro-F1={mf1:.4f} (Qual={mf1*80:.2f}/80) | "
            f"F1 [Rest/Stress/Act]={cf1[0]:.4f}/{cf1[1]:.4f}/{cf1[2]:.4f}"
        )


if __name__ == "__main__":
    main()