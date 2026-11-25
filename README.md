# Huawei Tech Arena 2026 - Project & Experimentation README

*Last updated:* `2026-11-25 19:05:20`

## 1. Task & Competition Overview
- **Target States:** `Rest/recovery` (`Rest`), `Acute Stress` (`Stress`), and `Physical Activity` (`Activity`).
- **Input Signals:** `BVP`, `EDA`, `HR`, `IBI_dt`, `IBI_duration`, and `Temperature`.
- **Evaluation Window:** 10-second windows, with the most frequent physiological state in each window determining the ground-truth label.
- **Scoring & Constraints:**
  - Quality score: `Macro-F1 * 80` (0-80 points), averaging `Stress`, `Activity`, and `Rest` equally.
  - Speed score: `max(0, (1 - MET / 0.3) * 20)` (0-20 points), measured over 50 runs on 1 CPU core.
  - Runtime environment: Python 3.13, 1 CPU core, multi-threading disabled, at most 4 GB RAM, and submission `.zip` at most 50 MB containing matching `<name>.py` (exposing `get_model()` returning an `ADM_Base` instance) and `<name>.onnx`.
  - Submission rules: up to 3 submissions per team per day; best valid submission counts; update via **View your participations** -> **Model (.zip)**.

## 2. Current Local Files & Artifacts
- `src/train_model.py` (11.5 KB)
- `src/log_results.py` (2.6 KB)
- `src/auto_log_experiment.py` (5.0 KB)
- `src/summarize_for_chat.py` (3.3 KB)
- `src/update_readme.py` (6.4 KB)
- `experiments/run_experiments.py` (9.4 KB)
- `experiments/metrics_audit.py` (8.7 KB)
- `experiments/audit_pipeline.py` (10.2 KB)
- `experiments/benchmark_run.py` (1.3 KB)
- `submission/submission_model.py` (5.4 KB)
- `submission/submission_model.onnx` (26748.8 KB)
- `notes/experiment_notes.md` (46.4 KB)
- `submission/submission.zip` (3.40 MB ZIP)

## 3. Dataset & Subject Metadata Snapshot
### Phases-to-Activity Mapping (`phases_to_activity.txt`)
```text
Stress: Stroop, TMCT, Real opinion, Opposite opinion, Subtraction
Activity: Warm-up, any column with rpm in it, Sprint, Cooldown
Rest: Baseline, Rest
```

### Data Constraints (`data_constraints.txt`)
```text
AEROBIC
- S03: Ended the protocol early, only reaching 90 rpm.
- S07: Ended the protocol early, only reaching 95 rpm.

ANAEROBIC
- S06: Ended the protocol early (the last sprint is missing).
```

### Subject Metadata (`subject-info.csv`)
```text
Info ,Gender,Age,Height (cm),Weight (kg),Does physical activity regularly?,Protocol,Stress Inducement,Aerobic Exercise,Anaerobic Exercise
S01,m,21,192,84,Yes,V1,Yes,Yes,Yes****
S02,m,20,185,95,No,V1,Yes****,Yes,Yes
S03,m,20,175,62,Yes,V1,Yes,Yes***,Yes
S04,m,21,174,70,Yes,V1,Yes,Yes,Yes
S05,m,21,173,72,Yes,V1,Yes,Yes,Yes
S06,m,21,172,70,Yes,V1,Yes,Yes,Yes***
S07,m,19,184,88,Yes,V1,Yes,Yes***,Yes
S08,m,20,174,67,Yes,V1,Yes,Yes,Yes
S09,m,19,174,63,Yes,V1,Yes,Yes,Yes
S10,m,21,180,80,Yes,V1,Yes,Yes,Yes
S11,m,21,183,64,Yes,V1,Yes,Yes**,Yes
S12,m,18,176,79,Yes,V1,Yes,-,Yes
S13,m,21,175,65,Yes,V1,Yes,Yes,Yes
S14,m,19,182,85,Yes,V1,Yes,Yes,Yes
S15,m,21,176,77,Yes,V1,Yes,Yes,Yes
S16,m,20,168,61,Yes,V1,Yes,Yes,Yes**
S17,m,22,173,78,Yes,V1,Yes,Yes,Yes
S18,m,21,183,80,Yes,V1,Yes,Yes,Yes
f01,f,25,152,61,No,V2,Yes,Yes,Yes
f02,f,29,164,80,No,V2,Yes,Yes,Yes
f03,f,26,160,61,Yes,V2,Yes,Yes,Yes
f04,f,29,168,56,No,V2,Yes,Yes,Yes
f05,f,21,165,55,Yes,V2,Yes,Yes,Yes
f06,f,21,169,58,No,V2,Yes,Yes,Yes
f07,f,21,163,47,No,V2,Yes*,Yes,Yes
f08,f,22,158,50,No,V2,Yes,Yes,Yes
f09,f,21,170,56,Yes,V2,Yes,Yes,Yes
f10,f,21,172,65,Yes,V2,Yes,Yes,Yes
f11,f,31,170,84,No,V2,Yes,Yes,Yes
f12,f,30,158,97,No,V2,Yes,Yes,Yes
f13,f,29,154,56,No,V2,Yes,Yes,Yes
f14,f,-,-,-,-,V2,Yes**,-,-
f15,f,-,-,-,-,V2,Yes,-,-
f16,f,-,-,-,-,V2,Yes,-,-
f17,f,-,-,-,-,V2,Yes,-,-
f18,m,-,-,-,-,V2,Yes,-,-
,,,,,,,,,
References:,,,,,,,,,
m:male,,,,,,,,,
f: female,,,,,,,,,
V1: First version of the protocol,,,,,,,,,
V2: Second version of the protocol,,,,,,,,,
* Wrong wristband placement.,,,,,,,,,
** Connection Loss ,,,,,,,,,
*** Protocol was not fully completed,,,,,,,,,
**** Files with Annotations,,,,,,,,,
```

## 4. What Is Being Tested & How
1. **Dataset & Signal Audit:** Dataset structure, subject/session/window counts, class balance, missing values/NaNs, sampling intervals, corrupted/outlier signals, subject baseline differences, subject/protocol dominance, causal temporal dependencies, and protocol-specific patterns.
2. **Validation Framework (No Leakage):**
   - Random stratified split (diagnostic baseline only - never used to optimize models).
   - Subject-grouped cross-validation (`GroupKFold` by subject ID) and Leave-One-Subject-Out (`LOSO`) where feasible.
   - Protocol-grouped validation where dataset structure supports it.
   - Preprocessing and feature fitting restricted to training splits without subject leakage.
3. **Metrics Reported:** Overall Macro-F1, per-class precision/recall/F1 (`Rest`, `Stress`, `Activity`), confusion matrices, per-subject and per-protocol performance, and mean/standard deviation across validation folds.
4. **Controlled Experimentation & Inference Profiling:**
   - Single-variable changes while preserving the working baseline and logging runs to `notes/experiment_notes.md`.
   - End-to-end `run()` timing (`mean`, `median`, `P95`, `std`), peak memory, `.onnx` file size, and `sklearn` vs. ONNX parity checks under single-thread settings.

## 5. Recent Experiment Notes Summary
```text
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
- **Changes made:** Streamlined feature extraction in _compute_window_features to avoid u
... [truncated]
```
