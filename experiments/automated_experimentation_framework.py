# Automated Experimentation & Optimization Framework
# Huawei 2026 Finland Tech Arena Competition

import os
import pathlib
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "TechArena-Finland-2026").exists() else SCRIPT_DIR

sys.path.insert(0, str(SCRIPT_DIR))

from submission_model import LABELS, compute_window_features, smooth_window_probs, get_model
from train_model import find_dataset_root, build_dataset, make_ensemble_pipeline, search_best_weights


class CompetitionExperimentHarness:
    def __init__(self):
        self.root_dir = find_dataset_root()
        print(f"[HARNESS] Found dataset root at: {self.root_dir}")

    def run_ablation_study(self):
        print("\n" + "=" * 80)
        print("          AUTOMATED RESEARCH & EXPERIMENTATION FRAMEWORK RUN")
        print("=" * 80)
        X, y, groups, purity, file_slices = build_dataset(self.root_dir, win_sec=10)
        n_samples, n_feats = X.shape
        n_subjects = len(np.unique(groups))
        print(f"[DATASET] Loaded {n_samples} 10s windows across {n_subjects} subjects | Feature dimension: {n_feats}")

        gkf = GroupKFold(n_splits=5)
        oof_probs = np.zeros((len(y), 3), dtype=np.float32)
        fold_indices = list(gkf.split(X, y, groups))

        print("\n[CV] Executing 5-Fold Subject-Grouped Cross Validation...")
        start_t = time.time()
        for fold, (tr_idx, va_idx) in enumerate(fold_indices):
            clean_tr = tr_idx[purity[tr_idx] >= 0.75]
            if len(clean_tr) < 100:
                clean_tr = tr_idx

            sw_tr = compute_sample_weight("balanced", y[clean_tr])
            clf = make_ensemble_pipeline(seed=42 + fold)
            clf.fit(X[clean_tr], y[clean_tr], ens__sample_weight=sw_tr)
            oof_probs[va_idx] = clf.predict_proba(X[va_idx]).astype(np.float32)
            print(f"  └─ Fold {fold+1}/5 Complete")

        cv_dur = time.time() - start_t
        print(f"[CV] Completed in {cv_dur:.2f} seconds")

        smoothed_oof = np.zeros_like(oof_probs)
        for sl in file_slices:
            smoothed_oof[sl] = smooth_window_probs(oof_probs[sl])

        raw_f1 = f1_score(y, np.argmax(smoothed_oof, axis=1), average="macro")
        quality_raw = raw_f1 * 80.0

        nested_preds = np.zeros_like(y)
        for tr_idx, va_idx in fold_indices:
            w_fold = search_best_weights(smoothed_oof[tr_idx], y[tr_idx])
            nested_preds[va_idx] = np.argmax(smoothed_oof[va_idx] * w_fold, axis=1)

        nested_f1 = f1_score(y, nested_preds, average="macro")
        quality_nested = nested_f1 * 80.0

        final_weights = search_best_weights(smoothed_oof, y)
        deployed_preds = np.argmax(smoothed_oof * final_weights, axis=1)
        deployed_f1 = f1_score(y, deployed_preds, average="macro")
        quality_deployed = deployed_f1 * 80.0

        print("\n" + "-" * 50)
        print("          CROSS-VALIDATION EVALUATION METRICS")
        print("-" * 50)
        print(f"1. Unweighted Smoothed Macro-F1:  {raw_f1:.4f}  | Quality Score: {quality_raw:.2f} / 80.0")
        print(f"2. Nested Tuned Macro-F1 (Honest): {nested_f1:.4f}  | Quality Score: {quality_nested:.2f} / 80.0")
        print(f"3. Deployed Weights Macro-F1:     {deployed_f1:.4f}  | Quality Score: {quality_deployed:.2f} / 80.0")
        print(f"   Optimal Deployment Weights [R/S/A]: {[round(float(w), 4) for w in final_weights]}")

        print("\n[CLASSIFICATION REPORT - HONEST NESTED PREDICTIONS]")
        print(classification_report(y, nested_preds, target_names=LABELS, digits=4))

        print("\n[PER-SUBJECT PERFORMANCE AUDIT (BOTTOM 5 SUBJECTS)]")
        sub_scores = []
        for s in np.unique(groups):
            mask = (groups == s)
            s_f1 = f1_score(y[mask], deployed_preds[mask], average="macro", zero_division=0)
            sub_scores.append((s, np.sum(mask), s_f1))
        sub_scores.sort(key=lambda x: x[2])
        for s, n_w, score in sub_scores[:5]:
            print(f"  Subject {s:5s} | windows={n_w:4d} | Deployed Macro-F1={score:.4f}")

        return quality_deployed


if __name__ == "__main__":
    harness = CompetitionExperimentHarness()
    harness.run_ablation_study()
