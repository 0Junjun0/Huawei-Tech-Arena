"""Independent score check using final row outputs and participant-held-out ONNX models.
python benchmark_check.py --data path/to/data --reports checks
Add --historical-split for the earlier seven-person split. Default: five outer folds.
Optional --reference-windows reads prior out-of-fold predictions from identical folds.
"""

import argparse
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score
from submission_model import TechArenaModel, LABELS
from train_model import PHASES, load_variants, select_and_fit
from benchmark_run import measure, samples, speed


def window_results(frame, model):
    output = model.run(frame)
    if not output.index.equals(frame.index) or not output.Activity.isin(LABELS).all():
        raise ValueError(
            "Output labels or timestamps do not match the expected interface"
        )
    bins = frame.index.floor("10s")
    phase = frame.Phase.astype("string").str.lower().str.strip()
    truth = phase.map(PHASES).mask(phase.str.contains("rpm", na=False), 2)

    # Independently reduce final row predictions and true labels to window modes.
    def votes(values):
        return pd.crosstab(bins, values.to_numpy()).reindex(
            columns=[0, 1, 2], fill_value=0
        )

    labelled = votes(truth)
    predicted = votes(output.Activity.astype("string").map(dict(zip(LABELS, range(3)))))
    detail = (
        pd.DataFrame(
            {"actual": labelled.idxmax(axis=1), "predicted": predicted.idxmax(axis=1)}
        )
        .dropna()
        .astype(int)
    )
    # Distance to an observed task change; ignore unknown phases and interrupted segments.
    timeline = pd.DataFrame(
        {"phase": phase.to_numpy(), "known": truth.notna().to_numpy()},
        index=frame.index,
    ).sort_index()
    changed = (
        timeline.known
        & timeline.index.to_series().diff().le(pd.Timedelta(seconds=1))
        & timeline.known.shift(fill_value=False)
        & timeline.phase.ne(timeline.phase.shift()).fillna(False)
    )
    boundaries = timeline.index[changed].as_unit("ns").asi8 / 1e9
    centers = detail.index.as_unit("ns").asi8 / 1e9 + 5
    distance = np.full(len(detail), np.inf)
    if len(boundaries):
        j = boundaries.searchsorted(centers)
        distance = np.minimum(
            abs(centers - boundaries[np.maximum(0, j - 1)]),
            abs(centers - boundaries[np.minimum(j, len(boundaries) - 1)]),
        )
    detail["distance_seconds"] = distance
    detail["transition_band"] = np.where(
        np.isinf(distance),
        "unknown",
        np.where(distance <= 30, "within_30s", "beyond_30s"),
    )
    known = timeline.loc[timeline.known, "phase"]
    task_counts = pd.crosstab(known.index.floor("10s"), known.to_numpy())
    detail["task"] = task_counts.idxmax(axis=1).reindex(detail.index)
    return detail


def check(data, reports, historical, reference_windows):
    variants = load_variants(data)
    records = variants["long"]
    groups = np.array([r[3] for r in records])
    folds = (
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        if historical
        else GroupKFold(5, shuffle=True, random_state=2026)
    )
    all_windows = []
    reports.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        for fold, (tr, te) in enumerate(folds.split(records, groups=groups), 1):
            assert set(groups[tr]).isdisjoint(groups[te])
            path = Path(folder) / "corrected.onnx"
            rule, _ = select_and_fit(variants, tr, path)
            models = {"corrected": TechArenaModel(path)}
            for i in te:
                *_, subject, file = records[i]
                frame = pd.read_parquet(file)
                for version, model in models.items():
                    rows = window_results(frame, model)
                    rows = rows.assign(
                        fold=fold,
                        participant=subject,
                        recording=file.relative_to(data).as_posix(),
                        version=version,
                    )
                    all_windows.append(rows)
            print("Finished fold", fold, rule, flush=True)
    detail = pd.concat(all_windows)
    if reference_windows:
        keys = ["fold", "participant", "recording", "window", "actual"]
        current = detail.rename_axis("window").reset_index()
        ref = pd.read_csv(reference_windows, parse_dates=["window"])
        ref = ref[ref.version == "corrected"]
        pairs = current.merge(
            ref[keys + ["predicted"]],
            on=keys,
            how="outer",
            indicator=True,
            validate="one_to_one",
            suffixes=("", "_ref"),
        )
        if not (pairs._merge == "both").all():
            raise ValueError("Reference records or folds differ from this run")
        print(
            "Reference macro-F1",
            f1_score(
                pairs.actual, pairs.predicted_ref, labels=[0, 1, 2], average="macro"
            ),
        )
    detail["error"] = detail.actual != detail.predicted
    detail.to_csv(reports / "windows.csv", index_label="window")
    detail.groupby(
        ["version", "participant", "task", "transition_band", "actual", "predicted"]
    ).size().rename("windows").to_csv(reports / "error_counts.csv")
    for version, rows in detail.groupby("version"):
        print(
            version,
            "macro-F1",
            f1_score(rows.actual, rows.predicted, labels=[0, 1, 2], average="macro"),
            "accuracy",
            1 - rows.error.mean(),
            "confusion",
            confusion_matrix(rows.actual, rows.predicted, labels=[0, 1, 2]),
            flush=True,
        )
    per_person = []
    for (version, subject), rows in detail.groupby(["version", "participant"]):
        matrix = confusion_matrix(rows.actual, rows.predicted, labels=[0, 1, 2])
        per_person.append(
            {
                "version": version,
                "participant": subject,
                "windows": len(rows),
                "accuracy": float(np.trace(matrix) / len(rows)),
                "rest_as_activity": int(matrix[0, 2]),
                "stress_as_rest": int(matrix[1, 0]),
                "stress_as_activity": int(matrix[1, 2]),
            }
        )
    pd.DataFrame(per_person).to_csv(reports / "participant_errors.csv", index=False)
    # Time shipped all-data model on this host; quality above always uses cross-fitted models.
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    model = TechArenaModel(Path(__file__).with_name("submission_model.onnx"))
    times = measure(model, samples(data.rglob("*.parquet")))
    rows = detail[detail.version == "corrected"]
    quality = 80 * f1_score(
        rows.actual, rows.predicted, average="macro", labels=[0, 1, 2]
    )
    print(
        "50-run MET",
        times.mean(),
        "quality",
        quality,
        "speed",
        speed(times.mean()),
        "combined",
        quality + speed(times.mean()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("TechArena-Finland-2026/data"))
    p.add_argument("--reports", type=Path, default=Path("benchmark_reports"))
    p.add_argument("--historical-split", action="store_true")
    p.add_argument("--reference-windows", type=Path)
    args = p.parse_args()
    check(args.data, args.reports, args.historical_split, args.reference_windows)
