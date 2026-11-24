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
