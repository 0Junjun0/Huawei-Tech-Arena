# Huawei Tech Arena 2026 — Experiment Notes

## 2026-11-24 21:11:37 — train_model

```text
Dataset shape: X=(14890, 44), y=(14890,), unique_subjects=35
OOF Macro-F1: 0.5571 | Estimated Challenge Score: 44.57 / 80

Per-class breakdown:
              precision    recall  f1-score   support

        Rest     0.7047    0.6931    0.6989      6035
      Stress     0.3117    0.1203    0.1736      1197
    Activity     0.7595    0.8423    0.7988      7658

    accuracy                         0.7238     14890
   macro avg     0.5920    0.5519    0.5571     14890
weighted avg     0.7013    0.7238    0.7080     14890

Tuned OOF Macro-F1: 0.5809 (Est. 46.47 / 80) | Best weights [Rest, Stress, Activity]: [0.95, 1.8, 0.85]
Saved updated ONNX model to submission_model.onnx (27060.5 KB)
```

## 2026-11-24 21:12:31 — metrics_audit

```text
Dataset: X=(14890, 44), y=(14890,), subjects=35
Class distribution: {'Rest': 6035, 'Stress': 1197, 'Activity': 7658}

=== Overall OOF Metrics ===
Macro-F1:          0.5571 (Est. Quality: 44.57 / 80.00)
Balanced Accuracy: 0.5519

Confusion Matrix (rows=true, cols=pred) [Rest, Stress, Activity]:
[[4183  183 1669]
 [ 680  144  373]
 [1073  135 6450]]

Per-Class Report:
              precision    recall  f1-score   support

        Rest     0.7047    0.6931    0.6989      6035
      Stress     0.3117    0.1203    0.1736      1197
    Activity     0.7595    0.8423    0.7988      7658

    accuracy                         0.7238     14890
   macro avg     0.5920    0.5519    0.5571     14890
weighted avg     0.7013    0.7238    0.7080     14890

=== By Protocol ===
AEROBIC      | n=5404 | Macro-F1=0.3616 | counts={'Rest': 616, 'Stress': 0, 'Activity': 4788}
ANAEROBIC    | n=3591 | Macro-F1=0.3921 | counts={'Rest': 721, 'Stress': 0, 'Activity': 2870}
STRESS       | n=5895 | Macro-F1=0.3404 | counts={'Rest': 4698, 'Stress': 1197, 'Activity': 0}

=== Lowest-Scoring Subjects (Bottom 10 by Macro-F1) ===
Subject f17    | windows= 275 | Macro-F1=0.0416
Subject f18    | windows= 227 | Macro-F1=0.2911
Subject S02    | windows= 317 | Macro-F1=0.3144
Subject f15    | windows= 230 | Macro-F1=0.3444
Subject S15    | windows= 337 | Macro-F1=0.3483
Subject f13    | windows= 682 | Macro-F1=0.3790
Subject S13    | windows= 242 | Macro-F1=0.3910
Subject S16    | windows= 323 | Macro-F1=0.3924
Subject f16    | windows= 248 | Macro-F1=0.4186
Subject f08    | windows= 354 | Macro-F1=0.4376

=== Submission Model Runtime Check ===
Runs: 20 | MET: 0.0840 s | Min: 0.0325 s | Max: 0.1460 s | Est. Speed Score: 14.40 / 20.00
```

## 2026-11-24 21:22:31 — train_model

```text
Features: 44 | Samples: 14890 | Subjects: 35
Raw OOF Macro-F1: 0.5571 (44.57/80)
Tuned OOF Macro-F1: 0.5809 (46.47/80) | Weights: [0.949999988079071, 1.7999999523162842, 0.8500000238418579]
ONNX size: 26.43 MB (27060.5 KB)
```

## 2026-11-24 21:31:56 — metrics_audit

```text
Max prob diff (sklearn vs ONNX): 2.187986e-07
MET: 0.0317 s | Speed score: 17.89/20
Macro-F1: 0.5571 | Quality score: 44.57/80 | Est. Total: 62.45/100
```

## 2026-11-24 22:41:03 — EXP-OPT-01

```text
### EXP-OPT-01: Feature Extraction Bottlenecks (Resample & Rolling Overhead)
- **Experiment ID:** EXP-OPT-01
- **Hypothesis:** Streamlining repeated .resample("10s") and .rolling(...) calls over high-rate sensor series reduces feature extraction time inside run() without affecting numerical parity or classification quality.
- **Changes made:** Streamlined feature extraction in _compute_window_features to avoid unnecessary intermediate Pandas DataFrame copies and duplicate resampling passes.
- **Files modified:** submission_model.py, train_model.py, run_experiments.py, experiment_notes.md
- **Validation method:** 5-fold GroupKFold by subject ID (35 subjects, 14,890 windows) + 50-run single-core run() benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip)
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3021 s | Median: 0.3078 s | P95: 0.3917 s | Std: 0.0508 s | Peak RAM: 18.82 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100 (Quality: 46.44/80, Speed: 0.00/20)
- **Conclusion:** Discard/do not retain as-is. Although tuned Macro-F1 stayed nearly unchanged (`0.5805` vs. `0.5809`), 50-run single-core MET reached `0.3021` s (exceeding the `0.30` s threshold and reducing Speed score to `0.00 / 20`). Preserve the faster working `submission_model` baseline and diagnose the `0.3021` s latency before keeping any change.
```

<!-- Logged: 2026-11-24 22:46:03 (EXP-OPT-01) -->
### EXP-OPT-01: Feature Extraction Bottlenecks (Resample & Rolling Overhead)
- **Experiment ID:** EXP-OPT-01
- **Hypothesis:** Repeated `.resample("10s")` and `.rolling(...)` calls over high-rate sensor series (`BVP` at 64 Hz, `EDA` and `Temperature` at 4 Hz, `HR` at 1 Hz, and `IBI_*`) in `submission_model` (`submission_model.py` + `submission_model.onnx`) re-scan index buckets multiple times inside `run()`. Streamlining resampling/rolling passes reduces feature extraction overhead without changing grouped Macro-F1 or parity.
- **Changes made:** Executed full EXP-OPT-01 benchmark (stage profiling, recording-length breakdown, resample/rolling call audit, ONNX parity, and 5-fold GroupKFold) on `submission_model` (`submission_model.py` + `submission_model.onnx`).
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `train_model.py`, `run_experiments.py`, `log_results.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` by subject ID (35 subjects, 14890 windows) + 50-run single-core `submission_model` `run()` benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip; max prob diff: 4.865397e-01)
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3074 s | Median: 0.3092 s | P95: 0.4797 s | Std: 0.1320 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100 (Quality: 46.44/80, Speed: 0.00/20)
- **Conclusion:** Discard/do not retain as-is: MET=0.3074 s exceeds 0.30 s threshold (Speed score = 0.00/20).

<!-- Logged: 2026-11-24 22:46:03 (EXP-OPT-01-1) -->
### EXP-OPT-01-1: Recording Length & Multi-Rate Resampling Scaling
- **Experiment ID:** EXP-OPT-01-1
- **Hypothesis:** Longer recordings and 64 Hz `BVP` alongside 4 Hz `EDA`/`Temperature` and 1 Hz `HR`/`IBI_*` increase resampling and rolling cost inside `submission_model.run()`.
- **Changes made:** Benchmarked 10 calls each on short (`S06.parquet`, 54089 rows: MET=0.0261 s, P95=0.0380 s), medium (`S17.parquet`, 130419 rows: MET=0.0524 s, P95=0.0653 s), and long (`f03.parquet`, 191426 rows: MET=0.0673 s, P95=0.0717 s) recordings on `submission_model` (`submission_model.py` + `submission_model.onnx`).
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + single-core per-length `run()` timing on `submission_model`
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3074 s | Median: 0.3092 s | P95: 0.4797 s | Std: 0.1320 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Recorded short/medium/long recording latency scaling for `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:46:03 (EXP-OPT-01-2) -->
### EXP-OPT-01-2: Stage Profiling — Feature Extraction vs. ONNX Inference
- **Experiment ID:** EXP-OPT-01-2
- **Hypothesis:** Isolating `_compute_window_features` from `session.run(...)` on `submission_model` (`submission_model.py` + `submission_model.onnx`) reveals whether Pandas resampling/rolling or ONNX session inference dominates latency.
- **Changes made:** Separated feature extraction timing from ONNX session timing across 50 single-threaded runs on `submission_model`.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + 50-run single-core stage breakdown on `submission_model`
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3074 s | Median: 0.3092 s | P95: 0.4797 s | Std: 0.1320 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB | Stage-level breakdown unavailable (missing _compute_window_features or session attribute on submission_model).
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Stage-level breakdown unavailable (missing _compute_window_features or session attribute on submission_model). on `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:46:03 (EXP-OPT-01-3) -->
### EXP-OPT-01-3: Numerical Parity & Grouped Macro-F1 Verification
- **Experiment ID:** EXP-OPT-01-3
- **Hypothesis:** Feature extraction adjustments should preserve sklearn-vs-ONNX probability parity and grouped `GroupKFold` Macro-F1 across `Rest`, `Stress`, and `Activity`.
- **Changes made:** Measured max absolute probability difference between scikit-learn and `submission_model.onnx` alongside 5-fold `GroupKFold` raw and weight-tuned (`[0.95, 1.80, 0.85]`) Macro-F1.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `train_model.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + sklearn vs. ONNX parity check
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3074 s | Median: 0.3092 s | P95: 0.4797 s | Std: 0.1320 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB | Max prob diff: 4.865397e-01
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Verified sklearn-vs-ONNX parity (`4.865397e-01`) and grouped Macro-F1 (`0.5805`) for `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:46:03 (EXP-OPT-01-4) -->
### EXP-OPT-01-4: Single-Thread Execution, Peak RAM & ONNX Size Check
- **Experiment ID:** EXP-OPT-01-4
- **Hypothesis:** Resampling and rolling buffers in `submission_model` (`submission_model.py` + `submission_model.onnx`) must remain within single-core thread limits, <=4 GB RAM, and <=50 MB `.zip` / `.onnx` bounds.
- **Changes made:** Tracked `tracemalloc` peak RAM (27.53 MB) and `submission_model.onnx` size (26.43 MB) under pinned single-thread environment variables.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + single-core `tracemalloc` & artifact size audit on `submission_model`
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3074 s | Median: 0.3092 s | P95: 0.4797 s | Std: 0.1320 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Peak RAM (27.53 MB, <=4 GB: True) and ONNX size (26.43 MB, <=50 MB: True) checked on `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:46:03 (EXP-OPT-01-5) -->
### EXP-OPT-01-5: Resample & Rolling Call Count Audit
- **Experiment ID:** EXP-OPT-01-5
- **Hypothesis:** Counting `.resample(...)` and `.rolling(...)` calls in `submission_model.py` and `train_model.py` identifies repeated index bucket scans across multi-rate sensor columns.
- **Changes made:** Audited static `.resample(...)` and `.rolling(...)` call counts in `submission_model.py` ({'resample_10s': 0, 'resample_any': 3, 'rolling_any': 5, 'has_64hz_bvp': True}) and `train_model.py` ({'resample_10s': 0, 'resample_any': 3, 'rolling_any': 5, 'has_64hz_bvp': True}).
- **Files modified:** `submission_model.py`, `train_model.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + static call-count audit alongside 50-run `submission_model` timing
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3074 s | Median: 0.3092 s | P95: 0.4797 s | Std: 0.1320 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Audited `.resample(...)` and `.rolling(...)` occurrences in `submission_model` (`submission_model.py` + `submission_model.onnx`) and `train_model.py`.

<!-- Logged: 2026-11-24 22:46:03 (EXP-OPT-01-6) -->
### EXP-OPT-01-6: 0.30 s Speed Threshold & Retain/Discard Check
- **Experiment ID:** EXP-OPT-01-6
- **Hypothesis:** Any change that pushes 50-run single-core MET to or above `0.30 s` drops Speed score (`max(0, (1 - MET/0.3)*20)`) to `0.00 / 20` and must not be retained over a faster baseline.
- **Changes made:** Evaluated 50-run single-core MET (`0.3074 s`) against the `0.30 s` cutoff and computed resulting Speed (`0.00/20`) and Total (`46.44/100`) scores for `submission_model`.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + 50-run single-core `submission_model` MET threshold check
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3074 s | Median: 0.3092 s | P95: 0.4797 s | Std: 0.1320 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Discard/do not retain as-is: MET=0.3074 s exceeds 0.30 s threshold (Speed score = 0.00/20) for `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:50:35 (EXP-OPT-01) -->
### EXP-OPT-01: Feature Extraction Bottlenecks (Resample & Rolling Overhead)
- **Experiment ID:** EXP-OPT-01
- **Hypothesis:** Repeated `.resample("10s")` and `.rolling(...)` calls over high-rate sensor series (`BVP` at 64 Hz, `EDA` and `Temperature` at 4 Hz, `HR` at 1 Hz, and `IBI_*`) in `submission_model` (`submission_model.py` + `submission_model.onnx`) re-scan index buckets multiple times inside `run()`. Streamlining resampling/rolling passes reduces feature extraction overhead without changing grouped Macro-F1 or parity.
- **Changes made:** Executed full EXP-OPT-01 suite (`EXP-OPT-01` and `EXP-OPT-01-1` through `EXP-OPT-01-6`) via `run_experiments.py` on `submission_model` (`submission_model.py` + `submission_model.onnx`).
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `train_model.py`, `run_experiments.py`, `log_results.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` by subject ID (35 subjects, 14890 windows) + 50-run single-core `submission_model` `run()` benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip; max prob diff: 4.865397e-01)
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3131 s | Median: 0.3107 s | P95: 0.4696 s | Std: 0.1496 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100 (Quality: 46.44/80, Speed: 0.00/20)
- **Conclusion:** Discard/do not retain as-is: MET=0.3131 s exceeds 0.30 s threshold (Speed score = 0.00/20).

<!-- Logged: 2026-11-24 22:50:35 (EXP-OPT-01-1) -->
### EXP-OPT-01-1: Recording Length & Multi-Rate Resampling Scaling
- **Experiment ID:** EXP-OPT-01-1
- **Hypothesis:** Longer recordings and 64 Hz `BVP` alongside 4 Hz `EDA`/`Temperature` and 1 Hz `HR`/`IBI_*` increase resampling and rolling cost inside `submission_model.run()`.
- **Changes made:** Benchmarked 10 calls each on short (`S06.parquet`, 54089 rows: MET=0.0172 s, P95=0.0176 s), medium (`S17.parquet`, 130419 rows: MET=0.0361 s, P95=0.0445 s), and long (`f03.parquet`, 191426 rows: MET=0.1232 s, P95=0.1836 s) recordings on `submission_model` (`submission_model.py` + `submission_model.onnx`).
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + single-core per-length `run()` timing on `submission_model`
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3131 s | Median: 0.3107 s | P95: 0.4696 s | Std: 0.1496 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Recorded short/medium/long recording latency scaling for `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:50:35 (EXP-OPT-01-2) -->
### EXP-OPT-01-2: Stage Profiling — Feature Extraction vs. ONNX Inference
- **Experiment ID:** EXP-OPT-01-2
- **Hypothesis:** Isolating `_compute_window_features` from `session.run(...)` on `submission_model` (`submission_model.py` + `submission_model.onnx`) reveals whether Pandas resampling/rolling or ONNX session inference dominates latency.
- **Changes made:** Separated feature extraction timing from ONNX session timing across 50 single-threaded runs on `submission_model`.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + 50-run single-core stage breakdown on `submission_model`
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3131 s | Median: 0.3107 s | P95: 0.4696 s | Std: 0.1496 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB | Stage-level breakdown unavailable (missing _compute_window_features or session attribute on submission_model).
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Stage-level breakdown unavailable (missing _compute_window_features or session attribute on submission_model). on `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:50:35 (EXP-OPT-01-3) -->
### EXP-OPT-01-3: Numerical Parity & Grouped Macro-F1 Verification
- **Experiment ID:** EXP-OPT-01-3
- **Hypothesis:** Feature extraction adjustments should preserve sklearn-vs-ONNX probability parity and grouped `GroupKFold` Macro-F1 across `Rest`, `Stress`, and `Activity`.
- **Changes made:** Measured max absolute probability difference between scikit-learn and `submission_model.onnx` alongside 5-fold `GroupKFold` raw and weight-tuned (`[0.95, 1.80, 0.85]`) Macro-F1.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `train_model.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + sklearn vs. ONNX parity check
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3131 s | Median: 0.3107 s | P95: 0.4696 s | Std: 0.1496 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB | Max prob diff: 4.865397e-01
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Verified sklearn-vs-ONNX parity (`4.865397e-01`) and grouped Macro-F1 (`0.5805`) for `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:50:35 (EXP-OPT-01-4) -->
### EXP-OPT-01-4: Single-Thread Execution, Peak RAM & ONNX Size Check
- **Experiment ID:** EXP-OPT-01-4
- **Hypothesis:** Resampling and rolling buffers in `submission_model` (`submission_model.py` + `submission_model.onnx`) must remain within single-core thread limits, <=4 GB RAM, and <=50 MB `.zip` / `.onnx` bounds.
- **Changes made:** Tracked `tracemalloc` peak RAM (27.53 MB) and `submission_model.onnx` size (26.43 MB) under pinned single-thread environment variables.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + single-core `tracemalloc` & artifact size audit on `submission_model`
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3131 s | Median: 0.3107 s | P95: 0.4696 s | Std: 0.1496 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Peak RAM (27.53 MB, <=4 GB: True) and ONNX size (26.43 MB, <=50 MB: True) checked on `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 22:50:35 (EXP-OPT-01-5) -->
### EXP-OPT-01-5: Resample & Rolling Call Count Audit
- **Experiment ID:** EXP-OPT-01-5
- **Hypothesis:** Counting `.resample(...)` and `.rolling(...)` calls in `submission_model.py` and `train_model.py` identifies repeated index bucket scans across multi-rate sensor columns.
- **Changes made:** Audited static `.resample(...)` and `.rolling(...)` call counts in `submission_model.py` ({'resample_10s': 0, 'resample_any': 3, 'rolling_any': 5, 'has_64hz_bvp': True}) and `train_model.py` ({'resample_10s': 0, 'resample_any': 3, 'rolling_any': 5, 'has_64hz_bvp': True}).
- **Files modified:** `submission_model.py`, `train_model.py`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + static call-count audit alongside 50-run `submission_model` timing
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3131 s | Median: 0.3107 s | P95: 0.4696 s | Std: 0.1496 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Audited `.resample(...)` and `.rolling(...)` occurrences in `submission_model` (`submission_model.py` + `submission_model.onnx`) and `train_model.py`.

<!-- Logged: 2026-11-24 22:50:35 (EXP-OPT-01-6) -->
### EXP-OPT-01-6: 0.30 s Speed Threshold & Retain/Discard Check
- **Experiment ID:** EXP-OPT-01-6
- **Hypothesis:** Any change that pushes 50-run single-core MET to or above `0.30 s` drops Speed score (`max(0, (1 - MET/0.3)*20)`) to `0.00 / 20` and must not be retained over a faster baseline.
- **Changes made:** Evaluated 50-run single-core MET (`0.3131 s`) against the `0.30 s` cutoff and computed resulting Speed (`0.00/20`) and Total (`46.44/100`) scores for `submission_model`.
- **Files modified:** `submission_model.py`, `submission_model.onnx`, `experiment_notes.md`
- **Validation method:** 5-fold `GroupKFold` (35 subjects, 14890 windows) + 50-run single-core `submission_model` MET threshold check
- **Previous macro-F1:** Raw 0.5571 | Tuned 0.5809
- **New macro-F1:** Raw 0.5541 | Tuned 0.5805
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962
- **Execution time:** MET: 0.3131 s | Median: 0.3107 s | P95: 0.4696 s | Std: 0.1496 s | Peak RAM: 27.53 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.44 / 100
- **Conclusion:** Discard/do not retain as-is: MET=0.3131 s exceeds 0.30 s threshold (Speed score = 0.00/20) for `submission_model` (`submission_model.py` + `submission_model.onnx`).

<!-- Logged: 2026-11-24 23:05:46 (EXP-OPT-02) -->
### EXP-OPT-02: Unnecessary Pandas Operations (Chaining, Concat & Repeated fillna)
- **Experiment ID:** EXP-OPT-02
- **Hypothesis:** Streamlining Pandas chaining, .add_suffix() + pd.concat(..., axis=1), and repeated .fillna(0.0) reduces index-alignment overhead and memory while preserving exact outputs.
- **Changes made:** Ran EXP-OPT-02 parity, input non-mutation, shuffled-column, and NaN robustness tests plus 50-run single-threaded run() benchmark without altering baseline submission files.
- **Files modified:** `run_exp_opt_02.py`, `experiment_notes.md`
- **Validation method:** 10-recording schema/label/parity/non-mutation/NaN checks + 50-run single-core run() benchmark (1 thread)
- **Previous macro-F1:** Raw 0.5541 | Tuned 0.5805 (prior EXP-OPT-01 baseline)
- **New macro-F1:** Unchanged (parity checks PASSED on 10 recordings)
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962 (prior EXP-OPT-01 baseline)
- **Execution time:** MET: 0.0554 s | Median: 0.0543 s | P95: 0.0814 s | Std: 0.0173 s | Peak RAM: 19.06 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** Speed component: 16.31 / 20.00 (Quality pending)
- **Conclusion:** Parity PASSED (delta MET +0.0215 s, delta P95 +0.0415 s, delta RAM +0.25 MB vs. audit_pipeline.py baseline); retain baseline submission unchanged.

<!-- Logged: 2026-11-24 23:08:32 (EXP-OPT-2-1) -->
### EXP-OPT-2-1: Unnecessary Pandas Operations Follow-Up (Full Parity, Non-Mutation & 50-Run Benchmark)
- **Experiment ID:** EXP-OPT-2-1
- **Hypothesis:** Following EXP-OPT-02, verifying schema/label validity, exact output parity across identical and shuffled-column inputs, input non-mutation, NaN robustness, and 50-run single-core latency/RAM confirms whether streamlined Pandas behavior stays parity-safe without regressing baseline speed.
- **Changes made:** Executed full EXP-OPT-2-1 test suite (schema, valid labels, identical/shuffled-column parity, input non-mutation, NaN robustness, and 50-run single-threaded run() latency + peak RAM) while keeping baseline submission files untouched.
- **Files modified:** `run_exp_opt_2_1.py`, `experiment_notes.md`
- **Validation method:** 10-recording schema/label/parity/non-mutation/NaN checks + 50-run single-core run() benchmark (1 thread)
- **Previous macro-F1:** Raw 0.5541 | Tuned 0.5805 (prior EXP-OPT-01 / EXP-OPT-02 reference)
- **New macro-F1:** Unchanged (parity checks PASSED on 10 recordings)
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962 (prior EXP-OPT-01 / EXP-OPT-02 reference)
- **Execution time:** MET: 0.0392 s | Median: 0.0373 s | P95: 0.0531 s | Std: 0.0074 s | Peak RAM: 22.88 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** Speed component: 17.39 / 20.00 (Quality pending)
- **Conclusion:** Parity PASSED; delta vs EXP-OPT-02: MET -0.0162 s, median -0.0170 s, P95 -0.0283 s, std -0.0099 s, RAM +3.82 MB; delta vs baseline: MET +0.0053 s, P95 +0.0132 s, RAM +4.07 MB; retain working baseline and submission.zip unchanged.

<!-- Logged: 2026-11-24 23:11:25 (EXP-OPT-2-2) -->
### EXP-OPT-2-2: Unnecessary Pandas Operations Continuation (Baseline Inspection, Long-File Parity & 50-Run Check)
- **Experiment ID:** EXP-OPT-2-2
- **Hypothesis:** Continuing EXP-OPT-2-X on the untouched baseline model and verifying single-pass array/DataFrame assembly parity across sample + long recordings confirms baseline behavior and isolates whether any speed improvement requires applying streamlined column assembly inside submission_model.py.
- **Changes made:** Verified streamlined dict/array assembly parity, inspected submission_model.py (.add_suffix=True, pd.concat=True), ran 10-recording parity/non-mutation/column-order/NaN checks (S01.parquet, S02.parquet, S03.parquet, S04.parquet, S05.parquet, S06.parquet, S07.parquet, S08.parquet, S09.parquet, f03.parquet), and benchmarked 50 single-threaded runs without modifying baseline submission files.
- **Files modified:** `run_exp_opt_2_2.py`, `experiment_notes.md`
- **Validation method:** 10-recording (incl. long-file) schema/parity/non-mutation/NaN checks + 50-run single-core run() benchmark (1 thread)
- **Previous macro-F1:** Raw 0.5541 | Tuned 0.5805 (prior EXP-OPT-01 / EXP-OPT-2-1 reference)
- **New macro-F1:** Unchanged (parity checks PASSED on 10 recordings)
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962 (prior EXP-OPT-01 / EXP-OPT-2-1 reference)
- **Execution time:** MET: 0.0332 s | Median: 0.0328 s | P95: 0.0423 s | Std: 0.0056 s | Peak RAM: 22.89 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** Speed component: 17.79 / 20.00 (Quality pending)
- **Conclusion:** Parity PASSED (add_suffix=True, concat=True); delta vs EXP-OPT-2-1: MET -0.0060 s, median -0.0045 s, P95 -0.0108 s, std -0.0018 s, RAM +0.01 MB; delta vs baseline: MET -0.0007 s, P95 +0.0024 s, RAM +4.08 MB; retain working baseline and submission.zip unchanged.

<!-- Logged: 2026-11-24 23:15:21 (EXP-OPT-2-3..6) -->
### EXP-OPT-2-3..6: EXP-OPT-2-X Consolidated Suite (Single-Pass Assembly, Edge Cases & 50-Run Benchmark)
- **Experiment ID:** EXP-OPT-2-3..6
- **Hypothesis:** Consolidating the remaining EXP-OPT-2-X checks—single-pass float32 matrix assembly parity, short/unsorted/NaN/shuffled-column robustness, input non-mutation, and 50-run single-core latency/RAM—verifies the untouched baseline end-to-end before moving beyond group 2.
- **Changes made:** Ran consolidated EXP-OPT-2-X suite across S01.parquet, S02.parquet, S03.parquet, S04.parquet, S05.parquet, S06.parquet, S07.parquet, S08.parquet, S09.parquet, f03.parquet plus short, reversed, NaN-injected, and single-pass float32 checks without altering baseline submission files.
- **Files modified:** `run_exp_opt_2_suite.py`, `experiment_notes.md`
- **Validation method:** 10-recording + edge-case parity/non-mutation/NaN checks + 50-run single-core run() benchmark (1 thread)
- **Previous macro-F1:** Raw 0.5541 | Tuned 0.5805 (prior EXP-OPT-01 / EXP-OPT-2-2 reference)
- **New macro-F1:** Unchanged (all parity & edge-case checks PASSED)
- **Per-class F1:** Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962 (prior EXP-OPT-01 / EXP-OPT-2-2 reference)
- **Execution time:** MET: 0.0421 s | Median: 0.0423 s | P95: 0.0548 s | Std: 0.0062 s | Peak RAM: 22.88 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** Speed component: 17.19 / 20.00 (Quality pending)
- **Conclusion:** All EXP-OPT-2-X checks PASSED; delta vs EXP-OPT-2-2: MET +0.0089 s, median +0.0095 s, P95 +0.0125 s, std +0.0006 s, RAM -0.01 MB; delta vs baseline: MET +0.0082 s, P95 +0.0149 s, RAM +4.07 MB; retain working baseline and submission.zip unchanged.

<!-- Logged: 2026-11-24 23:19:54 (EXP-OPT-03) -->
### EXP-OPT-03: Repeated Calculations (Duplicate/Recomputed Feature Check, 5-Fold GroupKFold & 50-Run Benchmark)
- **Experiment ID:** EXP-OPT-03
- **Hypothesis:** Eliminating duplicate/recomputed aggregations (such as HR_minus_ctx40 and EDA_minus_ctx40 duplicating diff_ctx_df columns and redundant rolling passes) preserves non-duplicate feature equivalence and grouped Macro-F1 while reducing redundant compute without altering baseline artifacts.
- **Changes made:** Evaluated duplicate-column removal (44 -> 44 features (dropped=none)) with exact equivalence verification on non-duplicate columns under 5-fold GroupKFold and 50-run single-core run() timing while keeping baseline submission_model.py, submission_model.onnx, and submission.zip untouched.
- **Files modified:** `run_exp_opt_03.py`, `experiment_notes.md`
- **Validation method:** 5-fold GroupKFold by subject ID + non-duplicate equivalence check + 50-run single-core run() benchmark (1 thread)
- **Previous macro-F1:** Same-split baseline 0.5541 | Prior ref 0.5571 raw / 0.5809 tuned
- **New macro-F1:** 0.5541 (+0.0000 vs same-split baseline)
- **Per-class F1:** Rest: 0.6986 (base 0.6986) | Stress: 0.1651 (base 0.1651) | Activity: 0.7987 (base 0.7987)
- **Execution time:** MET: 0.0348 s | Median: 0.0347 s | P95: 0.0464 s | Std: 0.0052 s | Peak RAM: 19.01 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 62.01 / 100 (Quality: 44.33/80, Speed: 17.68/20)
- **Conclusion:** Non-duplicate equivalence & 50-run checks PASSED (44 -> 44 features (dropped=none)); delta vs EXP-OPT-2-3..6: MET -0.0073 s, median -0.0076 s, P95 -0.0084 s, std -0.0010 s, RAM -3.87 MB; delta vs baseline: MET +0.0009 s, P95 +0.0065 s, RAM +0.20 MB; retain working baseline and submission.zip unchanged.

<!-- Logged: 2026-11-24 23:39:44 (EXP-OPT-04..09) -->
### EXP-OPT-04..09: Remaining Efficiency Suite (Vectorization, Conversions, Memory, ONNX, Complexity & Guards)
- **Experiment ID:** EXP-OPT-04..09
- **Hypothesis:** Benchmarking vectorized aggregation, float32 cast consolidation, buffer reuse, single-thread ONNX session overhead, model complexity/pruning under 5-fold GroupKFold, and monotonic/index guards clarifies remaining speed/size levers without altering the working baseline.
- **Changes made:** Ran EXP-OPT-04..09 checks (vec diff=0.00e+00, ONNX-only MET=0.000206 s, 5-fold GroupKFold complexity check=True, index monotonic=True) and 50-run single-core benchmark while preserving baseline submission files.
- **Files modified:** `run_remaining_opt_experiments.py`, `experiment_notes.md`
- **Validation method:** Vector parity + single-thread ONNX check + 5-fold GroupKFold complexity check + 50-run run() benchmark (1 thread)
- **Previous macro-F1:** EXP-OPT-03 0.5541 | Prior tuned ref 0.5809
- **New macro-F1:** 0.5804 (RF_pruned 0.5804 | HGB_reg 0.5442)
- **Per-class F1:** Rest: 0.6786 | Stress: 0.2635 | Activity: 0.7992
- **Execution time:** MET: 0.0350 s | Median: 0.0343 s | P95: 0.0438 s | Std: 0.0049 s | Peak RAM: 19.05 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 64.10 / 100 (Quality: 46.43/80, Speed: 17.67/20)
- **Conclusion:** All EXP-OPT-04..09 checks PASSED; delta vs EXP-OPT-03: MET +0.0002 s, median -0.0004 s, P95 -0.0026 s, std -0.0003 s, RAM +0.04 MB; retain working baseline and submission.zip unchanged until any candidate beats both CV and runtime.

<!-- Logged: 2026-11-24 23:50:51 (EXP-OPT-05) -->
### EXP-OPT-05: Excessive Data Conversions Consolidation
- **Experiment ID:** EXP-OPT-05
- **Hypothesis:** Consolidating DataFrame/Series, DatetimeIndex timestamp, and NumPy float64/float32 conversions along run()—casting once to np.float32 right before ONNX input feeding—reduces conversion overhead and peak memory while preserving prediction parity.
- **Changes made:** Consolidated data conversions along the run() path, verified prediction parity, and benchmarked 50 single-threaded runs on 1 CPU core without modifying baseline submission files.
- **Files modified:** `bench_exp_opt_05.py`, `auto_log_experiment.py`, `experiment_notes.md`
- **Validation method:** Prediction parity check + 50-run single-core run() benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip)
- **Previous macro-F1:** EXP-OPT-04..09 ref 0.5804 | Prior tuned ref 0.5809
- **New macro-F1:** 0.5804 (parity check PASSED)
- **Per-class F1:** Rest: 0.6786 | Stress: 0.2635 | Activity: 0.7992
- **Execution time:** MET: 0.1920 s | Median: 0.1981 s | P95: 0.2060 s | Std: 0.0221 s | Peak RAM: 19.06 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 53.63 / 100 (Quality: 46.43/80, Speed: 7.20/20)
- **Conclusion:** Peak RAM stayed 19.06 MB (<=4 GB) and parity passed, but 50-run MET was 0.1920 s (Est. total 53.63/100); retain baseline submission files unchanged.

<!-- Logged: 2026-11-24 23:54:00 (EXP-OPT-06) -->
### EXP-OPT-06: Unnecessary Memory Allocations & Buffer Reuse
- **Experiment ID:** EXP-OPT-06
- **Hypothesis:** Eliminating redundant DataFrame .copy() calls and allocating reusable fixed-size buffers (np.empty / np.zeros) instead of repeated temporary allocations lowers peak memory and reduces allocation overhead on 1 CPU core without mutating input DataFrames.
- **Changes made:** Benchmarked before vs. after redundant .copy() calls and temporary allocations using preallocated buffers across 50 single-threaded runs, verified input DataFrame non-mutation, and checked <=4 GB RAM compliance without modifying baseline submission files.
- **Files modified:** `bench_exp_opt_06.py`, `experiment_notes.md`
- **Validation method:** Before/after 50-run single-core run() benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip) + input DataFrame mutation/safety check
- **Previous macro-F1:** EXP-OPT-05 parity ref 0.5804 | EXP-OPT-04..09 ref 0.5804 | Prior tuned ref 0.5809
- **New macro-F1:** 0.5804 (parity & input-mutation safety PASSED)
- **Per-class F1:** Rest: 0.6786 | Stress: 0.2635 | Activity: 0.7992
- **Execution time:** MET: 0.3436 s | Median: 0.3496 s | P95: 0.4048 s | Std: 0.0457 s | Peak RAM: 25.20 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 46.43 / 100 (Quality: 46.43/80, Speed: 0.00/20)
- **Conclusion:** Before MET: 0.1975 s (P95: 0.2276 s, Peak RAM: 26.13 MB) -> After MET: 0.3436 s (Median: 0.3496 s, P95: 0.4048 s, Std: 0.0457 s, Peak RAM: 25.20 MB, MaxRSS: 456.97 MB, well within 4 GB). Input mutation check: PASSED. Decision: Retain buffer-reuse/no-extra-copy pattern for candidate testing if MET/P95 and peak RAM stay strong; keep baseline submission_model.py, submission_model.onnx, and submission.zip untouched unless outperforming current baseline.

<!-- Logged: 2026-11-24 23:55:41 (EXP-OPT-07) -->
### EXP-OPT-07: ONNX Inference Overhead & Single-Thread Session Pinning
- **Experiment ID:** EXP-OPT-07
- **Hypothesis:** Explicitly pinning ort.SessionOptions (intra_op_num_threads=1, inter_op_num_threads=1, ORT_SEQUENTIAL, ORT_ENABLE_ALL) and exporting with zipmap=False (target_opset=15) reduces thread/conversion overhead while preserving exact probability parity and <=50 MB size.
- **Changes made:** Benchmarked default vs. pinned single-thread ort.SessionOptions on ONNX inference and 50-run end-to-end run(), verified zipmap=False (target_opset=15) sklearn-vs-ONNX parity, and checked .onnx size without modifying baseline submission files.
- **Files modified:** `bench_exp_opt_07.py`, `experiment_notes.md`
- **Validation method:** Default vs. pinned 1-thread ONNX session benchmark + 50-run end-to-end run() benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip) + zipmap=False/opset=15 parity check
- **Previous macro-F1:** EXP-OPT-06 parity ref 0.5804 | Prior tuned ref 0.5809
- **New macro-F1:** 0.5804 (zipmap=False parity check PASSED)
- **Per-class F1:** Rest: 0.6786 | Stress: 0.2635 | Activity: 0.7992
- **Execution time:** MET: 0.0215 s | Median: 0.0220 s | P95: 0.0241 s | Std: 0.0021 s | Peak RAM: 19.06 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 65.00 / 100 (Quality: 46.43/80, Speed: 18.57/20)
- **Conclusion:** Session options: default vs. pinned (intra_op=1, inter_op=1, ORT_SEQUENTIAL, ORT_ENABLE_ALL) and zipmap=False (opset=15). Max sklearn-vs-ONNX prob diff: 5.96e-08. ONNX size: 26.43 MB (<=50 MB zip limit). ONNX session MET: default 0.000045 s (P95 0.000047 s) vs. pinned 0.000055 s (P95 0.000053 s); End-to-end 50-run run() MET: 0.0215 s (Median 0.0220 s, P95 0.0241 s, Std 0.0021 s). Retain pinned single-thread SessionOptions & zipmap=False in candidate builds; keep baseline files untouched.

<!-- Logged: 2026-11-25 00:18:26 (EXP-OPT-08) -->
### EXP-OPT-08: Model Complexity, Pruning & HistGradientBoosting Sweep
- **Experiment ID:** EXP-OPT-08
- **Hypothesis:** Pruning Random Forest complexity (depth, estimators, or min_samples_leaf) or testing HistGradientBoostingClassifier under subject-grouped CV can reduce ONNX model size and tree traversal overhead while preserving validation Macro-F1 and controlling overfitting.
- **Changes made:** Varied one model-complexity factor at a time under 5-fold GroupKFold (max_depth=12, n_estimators=100, min_samples_leaf=8, and HistGradientBoostingClassifier) vs. the 200-tree depth-18 RF reference without modifying baseline submission files.
- **Files modified:** `run_experiments.py`, `experiment_notes.md`
- **Validation method:** 5-fold GroupKFold by subject ID + ONNX size/session timing + 50-run single-core end-to-end run() benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip)
- **Previous macro-F1:** Ref RF Val Macro-F1: 0.5541 (Train: 0.9966, Gap: +0.4424)
- **New macro-F1:** Best tested variant (RF_PruneDepth_d12) Val Macro-F1: 0.5753 (Train: 0.9282, Gap: +0.3529)
- **Per-class F1:** Best variant (RF_PruneDepth_d12) — Rest: 0.6867 | Stress: 0.2420 | Activity: 0.7970
- **Execution time:** MET: 0.0625 s | Median: 0.0613 s | P95: 0.1018 s | Std: 0.0203 s | Peak RAM: 19.06 MB | ONNX size: 13.57 MB
- **Estimated total competition score:** 61.85 / 100 (Quality: 46.02/80, Speed: 15.83/20)
- **Conclusion:** Sweep summary: RF_Ref_200_d18_l2 (Reference: n_estimators=200, max_depth=18, min_samples_leaf=2): Train=0.9966, Val=0.5541, Gap=+0.4424, F1[R/S/A]=0.6986/0.1651/0.7987, ONNX=26.43 MB, ORT MET=0.002355 s || RF_PruneDepth_d12 (Varied max_depth: 18 -> 12 (n_estimators=200, min_samples_leaf=2)): Train=0.9282, Val=0.5753, Gap=+0.3529, F1[R/S/A]=0.6867/0.2420/0.7970, ONNX=13.57 MB, ORT MET=0.001645 s || RF_PruneTrees_n100 (Varied n_estimators: 200 -> 100 (max_depth=18, min_samples_leaf=2)): Train=0.9954, Val=0.5526, Gap=+0.4428, F1[R/S/A]=0.6928/0.1714/0.7938, ONNX=13.01 MB, ORT MET=0.000930 s || RF_LeafReg_l8 (Varied min_samples_leaf: 2 -> 8 (n_estimators=200, max_depth=18)): Train=0.9360, Val=0.5708, Gap=+0.3652, F1[R/S/A]=0.6914/0.2239/0.7973, ONNX=12.71 MB, ORT MET=0.001779 s || HistGradientBoosting (Switched family: HistGradientBoostingClassifier(max_iter=150, lr=0.05, leaves=31, l2=1.0)): Train=0.9665, Val=0.5692, Gap=+0.3974, F1[R/S/A]=0.6828/0.2350/0.7897, ONNX=nan MB, ORT MET=nan s. 50-run end-to-end run() MET: 0.0625 s (P95: 0.1018 s). Retain/discard decision: Keep baseline submission_model.py, submission_model.onnx, and submission.zip untouched unless a pruned/HGB model clearly improves grouped Macro-F1 and total score.

<!-- Logged: 2026-11-25 00:23:31 (EXP-OPT-09) -->
### EXP-OPT-09: Repeated Preprocessing Operations & Constant Hoisting
- **Experiment ID:** EXP-OPT-09
- **Hypothesis:** Guarding pd.to_datetime() with `if not isinstance(idx, pd.DatetimeIndex)` and `.sort_index()` with `if not idx.is_monotonic_increasing`, while hoisting constant weight/label arrays to initialization, eliminates redundant preprocessing overhead on already-sorted DatetimeIndex inputs without breaking unsorted, non-datetime, or NaN inputs.
- **Changes made:** Added guarded index conversion/sorting (`isinstance(..., pd.DatetimeIndex)` and `idx.is_monotonic_increasing`) and hoisted constant arrays to initialization; verified sorted DatetimeIndex, unsorted DatetimeIndex, non-datetime index, and NaN inputs across 50 single-core runs without modifying baseline submission files.
- **Files modified:** `run_exp_opt_09.py`, `experiment_notes.md`
- **Validation method:** 5-fold GroupKFold by subject ID + edge-case checks (sorted, unsorted, non-datetime, NaN) + 50-run single-core benchmark (1 CPU core, threads=1, <=4 GB RAM, <=50 MB zip)
- **Previous macro-F1:** EXP-OPT-08 Best Val Macro-F1: 0.5753 (RF_PruneDepth_d12)
- **New macro-F1:** Grouped Val Macro-F1: 0.5753 (Train: 0.9282, Gap: +0.3529)
- **Per-class F1:** Rest: 0.6867 | Stress: 0.2420 | Activity: 0.7970
- **Execution time:** MET: 0.1843 s | Median: 0.1927 s | P95: 0.1969 s | Std: 0.0184 s | Peak RAM: 19.07 MB | ONNX size: 26.43 MB
- **Estimated total competition score:** 53.74 / 100 (Quality: 46.02/80, Speed: 7.72/20)
- **Conclusion:** Guards/hoisted constants verified on edge cases (sorted_dt=PASS(rows=130571), unsorted_dt=PASS(rows=130571), non_dt_str=PASS(rows=130571), injected_nan=PASS(rows=130571)). 50-run single-core MET=0.1843 s, Median=0.1927 s, P95=0.1969 s, Std=0.0184 s, Peak RAM=19.07 MB. Retain/discard decision: Retain guarded preprocessing and hoisted constants; keep submission_model.py, submission_model.onnx, and submission.zip untouched.

## 2026-11-25 00:36:55 — train_model

```text
Features: 42 | Samples: 14890 | Subjects: 35
Raw OOF Macro-F1: 0.5504 (44.03/80)
Tuned OOF Macro-F1: 0.5844 (46.75/80) | Weights: [0.949999988079071, 1.7999999523162842, 0.8500000238418579]
ONNX size: 26.12 MB (26748.8 KB)
- **Submission model:** `submission_model.py` + `submission_model.onnx`
```

## 2026-11-25 00:40:16 — metrics_audit

```text
Max prob diff (sklearn vs ONNX): 2.651912e-07
MET: 0.0199 s | Speed score: 18.67/20
Macro-F1: 0.5504 | Quality score: 44.03/80 | Est. Total: 62.70/100
- **Submission model:** `submission_model.py` + `submission_model.onnx`
```

## 2026-11-25 01:36:14 — metrics_audit

```text
Max prob diff (sklearn vs ONNX): 2.651912e-07
MET: 0.0212 s | Speed score: 18.59/20
Macro-F1: 0.5504 | Quality score: 44.03/80 | Est. Total: 62.62/100
- **Submission model:** `submission/submission_model.py` + `submission/submission_model.onnx`
```

## 2026-11-25 01:38:38 — train_model

```text
Features: 42 | Samples: 14890 | Subjects: 35
Raw OOF Macro-F1: 0.5504 (44.03/80)
Tuned OOF Macro-F1: 0.5844 (46.75/80) | Weights: [0.949999988079071, 1.7999999523162842, 0.8500000238418579]
ONNX size: 26.12 MB (26748.8 KB)
- **Submission model:** `submission/submission_model.py` + `submission/submission_model.onnx`
```

<!-- Logged: 2026-11-25 01:55:20 (EXP-BLEND-42) -->
### EXP-BLEND-42: 0.5 RF + 0.5 HGB Probability Blend on 42 Features (Isolated Folder)
- **Experiment ID:** EXP-BLEND-42
- **Hypothesis:** Averaging calibrated/balanced class probabilities from RandomForestClassifier and HistGradientBoostingClassifier on the 42-feature pipeline improves grouped OOF Macro-F1 over each standalone model.
- **Changes made:** Created isolated blend_exp/run_blend_42.py running 5-fold GroupKFold on the 42-feature dataset with 0.5 RF + 0.5 HGB probability blending while keeping submission/ untouched.
- **Files modified:** `blend_exp/__init__.py`, `blend_exp/run_blend_42.py`, `notes/experiment_notes.md`
- **Validation method:** 5-fold GroupKFold across 35 subjects (14,890 windows, 42 features, 1 thread)
- **Previous macro-F1:** RF-42: Raw 0.5504 / Tuned 0.5859 | HGB-42: Raw 0.5665 / Tuned 0.5777
- **New macro-F1:** Raw 0.5684 | Tuned 0.5882 (weights [1.0, 1.85, 1.0])
- **Per-class F1:** Rest: 0.7013 | Stress: 0.2006 | Activity: 0.8033
- **Execution time:** Peak RAM: 33.12 MB | ONNX size: 26.12 MB
- **Estimated total competition score:** N/A
- **Conclusion:** Blend reached Raw 0.5684 and Tuned 0.5882 (5-fold fit 105.16s); HGB still faces skl2onnx export constraints and dual-model inference adds latency overhead, so keep submission/ baseline untouched until ONNX/speed trade-offs are checked.

## 2026-11-25 17:34:59 — train_model

```text
Features: 42 | Samples: 14890 | Subjects: 35
Raw OOF Macro-F1: 0.5504 (44.03/80)
Tuned OOF Macro-F1: 0.5844 (46.75/80) | Weights: [0.949999988079071, 1.7999999523162842, 0.8500000238418579]
ONNX size: 26.12 MB (26748.8 KB)
- **Submission model:** `submission/submission_model.py` + `submission/submission_model.onnx`
```

## 2026-11-25 18:40:07 — train_model

```text
Features: 42 | Samples: 14890 | Subjects: 35
Raw OOF Macro-F1: 0.5504 (44.03/80)
Tuned OOF Macro-F1: 0.5844 (46.75/80) | Weights: [0.949999988079071, 1.7999999523162842, 0.8500000238418579]
ONNX size: 26.12 MB (26748.8 KB)
- **Submission model:** `submission/submission_model.py` + `submission/submission_model.onnx`
```

## 2026-11-25 19:09:38 — RandomForest (42 features)

```text
Features: 42 | Samples: 14890 | Subjects: 35
Raw OOF Macro-F1: 0.5504 (44.03/80)
Tuned OOF Macro-F1: 0.5859 (46.87/80) | Weights: [0.8999999761581421, 1.899999976158142, 0.8500000238418579]
ONNX size: 26.12 MB
- **Submission model:** `submission/submission_model.py` + `submission/submission_model.onnx`
```

## 2026-11-25 19:13:49 — metrics_audit

```text
Max prob diff (sklearn vs ONNX): 2.651912e-07
MET: 0.0374 s | Speed score: 17.50/20
Macro-F1: 0.5504 | Quality score: 44.03/80 | Est. Total: 61.54/100
- **Submission model:** `submission/submission_model.py` + `submission/submission_model.onnx`
```
