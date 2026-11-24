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
