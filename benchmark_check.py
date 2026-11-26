"""Independent score check using final row outputs and participant-held-out ONNX models.
python benchmark_check.py --data path/to/data --reports checks
Add --historical-split for the earlier seven-person split. Default: five outer folds.
Use --all-participants for every Huawei person held out individually. Optional CATSA, WESAD
and PPG-DaLiA archives add separate external-study results. --reference-windows compares public folds.
"""

import argparse
import codecs
import hashlib
import json
import pickle
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_run import (
    DEFAULT_MODEL,
    INPUT_COLUMNS,
    load_timed_model,
    measure,
    one_core,
    samples,
    speed,
    validate_output,
)
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from submission_model import (
    LABELS,
    TechArenaModel,
)
from train_model import (
    CATSA_SHA256,
    PHASES,
    PIPELINE,
    TEST_OUTAGES,
    external_frames,
    external_people_id,
    labelled_windows,
    load_variants,
    perturb,
    recording_rng,
    select_and_fit,
    training_id,
)


def window_results(frame, model):
    sensors = frame.reindex(columns=INPUT_COLUMNS)
    output = model.run(sensors)
    validate_output(output, frame)

    # Absolute time avoids ambiguity during a clock change in a timezone-aware index.
    def floor_bins(index):
        if index.tz is None:
            return index.floor("10s")
        return index.tz_convert("UTC").floor("10s").tz_convert(index.tz)

    bins = floor_bins(frame.index)
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
    if len(known):
        task_counts = pd.crosstab(floor_bins(known.index), known.to_numpy())
        detail["task"] = task_counts.idxmax(axis=1).reindex(detail.index)
    else:
        detail["task"] = pd.Series(index=detail.index, dtype="string")
    return detail


def failure_checks(frame, truth, model, recording):
    """Score actual row outputs. A temporary outage is graded inside its targeted bins."""
    sensors = frame.reindex(columns=INPUT_COLUMNS)
    bins, rows = np.unique(
        frame.index.as_unit("ns").asi8 // 10_000_000_000, return_inverse=True
    )
    expected = truth.index.as_unit("ns").asi8 // 10_000_000_000
    positions = np.searchsorted(bins, expected)
    if not np.array_equal(bins[positions], expected):
        raise ValueError("Label and prediction clocks disagree")
    actual = truth.actual.to_numpy(int)
    results = []

    def count(setting, scope, labels, guesses):
        cm = np.bincount(3 * labels + guesses, minlength=9).reshape(3, 3)
        return {
            "setting": setting,
            "scope": scope,
            **{f"n{i}{j}": int(cm[i, j]) for i in range(3) for j in range(3)},
        }

    clean_inputs = model.decision_inputs(sensors)
    clock = clean_inputs[-1]
    if not np.array_equal(clock, bins):
        raise ValueError("Runtime and independent benchmark timestamps differ")
    clean = model.decisions(clean_inputs, model.rule)[positions]
    if not np.array_equal(clean, truth.predicted):
        raise ValueError("Window decisions disagree with the scored row outputs")
    results.append(count("deployed", "whole", actual, clean))
    for key, neutral in (
        ("stress_weight", 1.0),
        ("activity_weight", 1.0),
        ("switch_cost", 0.0),
        ("strength", 0.0),
    ):
        guesses = model.decisions(clean_inputs, {**model.rule, key: neutral})[positions]
        results.append(count(key + "_neutral", "whole", actual, guesses))
    for scenario in TEST_OUTAGES:
        rng = recording_rng(18083, recording, scenario)
        damaged, affected = perturb(sensors, scenario, rng)
        output = model.run(damaged)
        validate_output(output, damaged)
        codes = pd.Categorical(output.Activity, categories=LABELS).codes
        votes = np.bincount(
            3 * rows + codes.astype(int), minlength=3 * len(bins)
        ).reshape(-1, 3)
        pred = votes.argmax(1)[positions]
        for scope, keep in (
            ("affected", affected[positions]),
            ("whole", np.ones(len(actual), bool)),
        ):
            results.append(count(scenario, scope, actual[keep], pred[keep]))
    return results


def check(
    data,
    reports,
    historical,
    reference_windows,
    model_path=DEFAULT_MODEL,
    seed=2026,
    timing_recordings=50,
    audit=False,
    fold_models=None,
    catsa_zip=None,
):
    count = 1 if historical else 5
    cache_complete = fold_models is not None and all(
        (fold_models / f"fold-{n}.onnx").is_file() for n in range(1, count + 1)
    )
    variants = load_variants(
        data,
        augment=not cache_complete,
        catsa_zip=catsa_zip,
        compact=True,
        work_dir=reports,
    )
    records = variants["long"]
    deployed, fingerprint = load_timed_model(model_path)
    raw_id = training_id([r[4] for r in records])
    corpus_matches = (
        deployed.session.get_modelmeta().custom_metadata_map.get(
            "training_files_sha256"
        )
        == raw_id
    )
    if not corpus_matches:
        print(
            "The timed artifact has a different training corpus. Quality and latency are reported separately."
        )
    groups = np.array([r[3] for r in records])
    folds = (
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        if historical
        else GroupKFold(5, shuffle=True, random_state=seed)
    )
    all_windows = []
    reports.mkdir(parents=True, exist_ok=True)
    tested = set()
    selections = []
    sensitivity = []
    with tempfile.TemporaryDirectory() as folder:
        for fold, (tr, te) in enumerate(folds.split(records, groups=groups), 1):
            held = set(groups[te])
            if held.intersection(groups[tr]) or tested.intersection(held):
                raise ValueError("Participant overlap in the test folds")
            tested.update(held)
            base = fold_models if fold_models is not None else Path(folder)
            base.mkdir(parents=True, exist_ok=True)
            path = base / f"fold-{fold}.onnx"
            trials = []
            if fold_models is not None and path.is_file():
                model = TechArenaModel(path.read_bytes())
                meta = model.session.get_modelmeta().custom_metadata_map
                if (
                    meta.get("short_trial_version")
                    != deployed.session.get_modelmeta().custom_metadata_map.get(
                        "short_trial_version"
                    )
                    or meta.get("cohort_profile")
                    != deployed.session.get_modelmeta().custom_metadata_map.get(
                        "cohort_profile"
                    )
                    or meta.get("external_training_fingerprint")
                    != deployed.session.get_modelmeta().custom_metadata_map.get(
                        "external_training_fingerprint"
                    )
                    or meta.get("pipeline") != PIPELINE
                    or meta.get("training_files_sha256")
                    != training_id([records[i][4] for i in tr])
                ):
                    raise ValueError(
                        f"Fold {fold} model does not match its training participants and recipe"
                    )
                rule = dict(
                    context=model.context, **model.rule, estimator=meta["estimator"]
                )
                dev_f1 = float(meta["development_macro_f1"])
            else:
                rule, dev_f1 = select_and_fit(
                    variants,
                    tr,
                    path,
                    trials,
                    profile=deployed.session.get_modelmeta().custom_metadata_map[
                        "cohort_profile"
                    ],
                )
                model = TechArenaModel(path.read_bytes())
            selections.append(dict(fold=fold, **rule, development_macro_f1=dev_f1))
            if trials:
                pd.DataFrame(trials).to_csv(
                    reports / f"development_fold_{fold}.csv", index=False
                )
            for i in te:
                _, train_y, _, subject, file = records[i]
                frame = pd.read_parquet(file)
                rows = window_results(frame, model)
                if audit:
                    sensitivity.extend(
                        dict(fold=fold, participant=subject, **row)
                        for row in failure_checks(
                            frame, rows, model, file.relative_to(data).as_posix()
                        )
                    )
                if int((train_y >= 0).sum()) != len(rows):
                    raise ValueError(
                        f"Trainer and row evaluator disagree on labelled bins: {file}"
                    )
                all_windows.append(
                    rows.assign(
                        fold=fold,
                        participant=subject,
                        recording=file.relative_to(data).as_posix(),
                        version="corrected",
                    )
                )
            print("Finished fold", fold, rule, flush=True)
    if not historical and tested != set(groups):
        raise ValueError("Some participants received no held-out prediction")
    if audit:
        pd.DataFrame(sensitivity).groupby(
            ["fold", "participant", "setting", "scope"]
        ).sum().to_csv(reports / "sensitivity.csv")
    pd.DataFrame(selections).to_csv(reports / "selected_folds.csv", index=False)
    detail = pd.concat(all_windows)
    if detail.empty:
        raise ValueError("No labelled test windows")
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
    # Validate artifact identity and time the nominated all-data model.
    with one_core():
        model = deployed
        frames = samples(data.rglob("*.parquet"), timing_recordings)
        times = measure(model, frames)
    rows = detail[detail.version == "corrected"]
    quality = 80 * f1_score(
        rows.actual, rows.predicted, average="macro", labels=[0, 1, 2]
    )
    result = {
        "participants": int(rows.participant.nunique()),
        "windows": len(rows),
        "outer_seed": None if historical else seed,
        "model_sha256": fingerprint,
        "model_rule": dict(context=model.context, **model.rule),
        "estimator": model.session.get_modelmeta().custom_metadata_map["estimator"],
        "mean_seconds": float(times.mean()),
        "p95_seconds": float(np.quantile(times, 0.95)),
        "timing_recordings": len(frames),
        "timing_runs": len(times),
        "quality": quality,
        "speed": speed(times.mean()),
        "matching_training_corpus": corpus_matches,
        "combined": quality + speed(times.mean()) if corpus_matches else None,
    }
    (reports / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def study_person_files(archive):
    files = {}
    pattern = re.compile(r"(?:^|/)(S[0-9]+)/\1\.pkl$")
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            match = pattern.search(name)
            if match:
                if match[1] in files:
                    raise ValueError("Duplicate synchronized participant file")
                files[match[1]] = name
    if not files:
        raise ValueError(
            "Use the original study ZIP containing one synchronized pickle per participant (for DaLiA use data.zip)."
        )
    return sorted(files.items(), key=lambda pair: int(pair[0][1:]))


class WristStudyReader(pickle.Unpickler):
    """Only deserialize the NumPy array records used by the official archives."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "dtype"): np.dtype,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            (
                "numpy._core.multiarray",
                "_reconstruct",
            ): np._core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("_codecs", "encode"): codecs.encode,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported study data type {module}.{name}")
        return allowed[module, name]


def study_arrays(archive, study, member):
    with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(member)
        if info.file_size > 3_500_000_000:
            raise ValueError("Unexpected synchronized file size")
        path = Path(folder) / "synchronized.pkl"
        with path.open("wb") as dst, bundle.open(info) as src:
            shutil.copyfileobj(src, dst, length=4 * 1024**2)
        with path.open("rb") as src:
            data = WristStudyReader(src, encoding="latin1").load()
    wrist = data["signal"]["wrist"]
    out = {
        key: np.asarray(wrist[raw], dtype=np.float32).reshape(-1).copy()
        for raw, key in [("BVP", "BVP"), ("EDA", "EDA"), ("TEMP", "Temperature")]
    }
    if study == "WESAD":
        truth = np.asarray(data["label"]).reshape(-1)
        hz = 700
        labels = {1: 0, 2: 1}
    elif study == "PPG-DaLiA":
        truth = np.asarray(data["activity"]).reshape(-1)
        hz = 4
        labels = {1: 0, 2: 2, 3: 2, 4: 2, 7: 2}
    else:
        raise ValueError("Unsupported independent study")
    if abs(len(out["BVP"]) / 64 - len(truth) / hz) > 3.0:
        raise ValueError("Signal and ground-truth clocks disagree")
    # Score only bins with an unambiguous majority of a mapped activity. Other
    # activities stay in the input so the model sees the entire recording.
    n = (len(out["BVP"]) + 639) // 640
    votes = np.zeros((n, 3), int)
    for key, code in labels.items():
        idx = np.flatnonzero(truth == key) // (10 * hz)
        idx = idx[idx < n]
        votes[:, code] += np.bincount(idx, minlength=n)
    support = np.clip(len(truth) - np.arange(n) * 10 * hz, 0, 10 * hz)
    y = votes.argmax(1)
    y[2 * votes.max(1) <= support] = -1
    return out, y


def field_frame(arrays):
    n = len(arrays["BVP"])
    idx = pd.date_range("2000-01-01", freq="15625000ns", periods=n)
    data = pd.DataFrame(index=idx)
    data["BVP"] = arrays["BVP"]
    for key in ("EDA", "Temperature"):
        full = np.full(n, np.nan, dtype="f")
        x = arrays[key]
        k = min(len(x), (n + 15) // 16)
        full[np.arange(k) * 16] = x[:k]
        data[key] = full
    return data.reindex(
        columns=["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
    )


def participant_folds(records, seed):
    groups = np.asarray([row[3] for row in records])
    ids = np.arange(len(records))
    people = sorted(set(groups))
    if len(people) < 8:
        raise ValueError(
            "Leave-one-person-out validation needs at least eight participants for nested calibration."
        )
    reference = list(
        GroupKFold(min(5, len(people)), shuffle=True, random_state=seed).split(
            ids, groups=groups
        )
    )
    for j, person in enumerate(people, 1):
        held = ids[groups == person]
        train = ids[groups != person]
        pool = next(test for _, test in reference if np.isin(held, test).all())
        peers = pool[groups[pool] != person]
        yield f"person{j:02}", person, train, held, peers


def study_metrics(cm, expected_classes):
    actual = cm.sum(1)
    called = cm.sum(0)
    tp = np.diag(cm)
    den = actual + called
    scores = np.divide(2 * tp, den, out=np.zeros(3, dtype=float), where=den > 0)
    available = np.flatnonzero(actual)
    # Every prediction made for a labelled window is retained, including calls
    # of a third class in a two-class study.
    supported = np.array(expected_classes, dtype=int)
    metrics = dict(
        windows=int(cm.sum()),
        accuracy=float(tp.sum() / cm.sum()) if cm.sum() else None,
        classes_present=",".join(LABELS[i] for i in available),
        macro_f1_observed=float(scores[available].mean()) if len(available) else None,
        macro_f1_study=float(scores[supported].mean()) if cm.sum() else None,
    )
    for j, label in enumerate(LABELS):
        metrics["true_" + label.lower()] = int(actual[j])
        metrics["predicted_" + label.lower()] = int(called[j])
        metrics["f1_" + label.lower()] = float(scores[j]) if actual[j] else None
    metrics.update({f"n{i}{j}": int(cm[i, j]) for i in range(3) for j in range(3)})
    return metrics


def score_input(model, sensors, expected_clock, labels):
    """Independent majority vote on every raw output row, checked against internal window decisions."""
    sensors = sensors.reindex(columns=INPUT_COLUMNS)
    out = model.run(sensors)
    validate_output(out, sensors)
    clock, rows = np.unique(
        sensors.index.as_unit("ns").asi8 // 10_000_000_000, return_inverse=True
    )
    codes = pd.Categorical(out["Activity"], categories=LABELS).codes.astype(np.int64)
    votes = np.bincount(3 * rows + codes, minlength=3 * len(clock)).reshape(-1, 3)
    pred = votes.argmax(1)
    ins = model.decision_inputs(sensors)
    if not np.array_equal(clock, ins[-1]) or not np.array_equal(
        pred, model.decisions(ins, model.rule)
    ):
        raise ValueError(
            "Raw-output majority votes disagree with native ten-second predictions"
        )
    if not np.array_equal(expected_clock, clock):
        raise ValueError("Input and prediction clocks differ")
    if labels is None:
        return None, len(sensors), len(pred), pred
    if len(labels) != len(pred):
        raise ValueError("Label and prediction counts differ")
    keep = labels >= 0
    cm = np.bincount(3 * labels[keep] + pred[keep], minlength=9).reshape(3, 3)
    return cm, len(sensors), len(pred), pred


def huawei_record(model, frame):
    scored = window_results(frame, model)
    clock = np.unique(frame.index.as_unit("ns").asi8 // 10_000_000_000)
    labels = np.full(len(clock), -1, int)
    positions = clock.searchsorted(scored.index.as_unit("ns").asi8 // 10_000_000_000)
    labels[positions] = scored.actual.to_numpy(int)
    cm, rows, bins, pred = score_input(model, frame, clock, labels)
    if not np.array_equal(pred[positions], scored.predicted.to_numpy(int)):
        raise ValueError(
            "Independent Huawei label aggregation disagrees with model output."
        )
    return cm, rows, bins, scored


def expected_candidate(model, profile, train_hash, external_hash):
    meta = model.session.get_modelmeta().custom_metadata_map
    if (
        meta.get("short_trial_version") != "independent-trials-v1"
        or meta.get("pipeline") != PIPELINE
        or meta.get("cohort_profile") != profile
        or meta.get("training_files_sha256") != train_hash
        or meta.get("external_training_fingerprint") != external_hash
    ):
        raise ValueError(
            "Model fingerprint differs from the requested training people, external training people or training recipe."
        )


def run_participant_check(
    data,
    reports,
    model_path,
    fold_models,
    catsa_zip=None,
    wesad_zip=None,
    dalia_zip=None,
    seed=2026,
    timing_recordings=50,
    with_audit=False,
    include_huawei=True,
    field_cache=None,
):
    reports.mkdir(parents=True, exist_ok=True)
    records = load_variants(data, augment=False)["long"]
    all_ids = np.arange(len(records))
    deployed, fingerprint = load_timed_model(model_path)
    current = deployed.session.get_modelmeta().custom_metadata_map
    profile = current.get("cohort_profile")
    if not profile:
        raise ValueError(
            "The all-participant benchmark needs the released all-participant model."
        )
    corpus = training_id([r[4] for r in records])
    if corpus != current.get("training_files_sha256"):
        raise ValueError(
            "The deployed model and the requested Huawei training collection do not match."
        )
    ext_frames = dict(external_frames(catsa_zip)) if catsa_zip is not None else {}
    if catsa_zip is not None:
        with Path(catsa_zip).open("rb") as inp:
            if hashlib.file_digest(inp, "sha256").hexdigest() != CATSA_SHA256:
                raise ValueError("Unexpected CATSA input archive")
        if current.get("external_training_fingerprint") != external_people_id(
            [{"subject": s} for s in sorted(ext_frames)]
        ):
            raise ValueError(
                "Deployment and external training cohort fingerprints differ."
            )
    variants = None
    people = []
    sens = []
    by_task = []

    def train_or_open(root, name, train, peers, ext_train, expected_ext):
        nonlocal variants
        root.mkdir(parents=True, exist_ok=True)
        path = root / (name + ".onnx")
        if not path.exists():
            if variants is None:
                variants = load_variants(
                    data, catsa_zip=catsa_zip, compact=True, work_dir=reports
                )
            select_and_fit(
                variants,
                train,
                path,
                calibration_peers=peers,
                external_indices=ext_train,
                profile=profile,
            )
        model = TechArenaModel(path)
        expected_candidate(
            model, profile, training_id([records[i][4] for i in train]), expected_ext
        )
        return model

    def collect(study, participant, fold, cm, classes, rows, bins, nrec):
        people.append(
            dict(
                study=study,
                participant=participant,
                test_fold=fold,
                recordings=nrec,
                input_rows=int(rows),
                observed_windows=int(bins),
                **study_metrics(cm, classes),
            )
        )

    with tempfile.TemporaryDirectory() as fallback:
        root = fold_models or Path(fallback)
        if include_huawei:
            identities = set()
            for name, person, train, held, peers in participant_folds(records, seed):
                train_people = {records[i][3] for i in train}
                if person in train_people or person in identities:
                    raise ValueError("Invalid leave-one-participant-out split.")
                identities.add(person)
                m = train_or_open(
                    root / "leave_one_out",
                    name,
                    train,
                    peers,
                    None,
                    current["external_training_fingerprint"],
                )
                cm = np.zeros((3, 3), int)
                rows = bins = 0
                for i in held:
                    f = records[i][4]
                    frame = pd.read_parquet(f)
                    cc, n, k, windows = huawei_record(m, frame)
                    cm += cc
                    rows += n
                    bins += k
                    if with_audit:
                        sens.extend(
                            dict(fold=name, **z)
                            for z in failure_checks(
                                frame, windows, m, f.relative_to(data).as_posix()
                            )
                        )
                    for (task, y), q in windows.groupby(["task", "actual"]):
                        cnt = np.bincount(q.predicted, minlength=3)
                        by_task.append(
                            dict(
                                task=task,
                                actual=int(y),
                                true_windows=len(q),
                                rest=int(cnt[0]),
                                stress=int(cnt[1]),
                                activity=int(cnt[2]),
                            )
                        )
                collect("Huawei", person, name, cm, (0, 1, 2), rows, bins, len(held))
                print("Scored Huawei participant", len(identities), flush=True)
            if len(identities) != len(set(r[3] for r in records)):
                raise AssertionError("Incomplete Huawei coverage")
        if ext_frames:
            task_rows = []
            participant_models = {}
            subjects = sorted(ext_frames)
            external_ids = np.arange(len(subjects))
            seen = set()
            for j, (tr, te) in enumerate(
                GroupKFold(5, shuffle=True, random_state=519).split(
                    external_ids, groups=subjects
                ),
                1,
            ):
                key = f"ext{j}"
                model = train_or_open(
                    root / "catsa",
                    key,
                    all_ids,
                    [],
                    tr,
                    external_people_id([{"subject": subjects[i]} for i in tr]),
                )
                for i in te:
                    subject = subjects[i]
                    frame = ext_frames[subject]
                    if subject in seen:
                        raise ValueError("Duplicate external test participant")
                    seen.add(subject)
                    participant_models[subject] = (key, model)
                    cm = np.zeros((3, 3), int)
                    rows = bins = 0
                    count = 0
                    # Separate source trials are separate model calls. The model never
                    # sees a paired, labelled baseline for the test participant.
                    boundary = np.r_[
                        0,
                        np.flatnonzero(
                            np.diff(frame.index.as_unit("ns").asi8) > 60_000_000_000
                        )
                        + 1,
                        len(frame),
                    ]
                    for a, b in zip(boundary[:-1], boundary[1:]):
                        part = frame.iloc[a:b]
                        clock, which = np.unique(
                            part.index.as_unit("ns").asi8 // 10_000_000_000,
                            return_inverse=True,
                        )
                        yy = labelled_windows(part, which, len(clock))
                        cc, n, k, _ = score_input(model, part, clock, yy)
                        counts = cc.sum(axis=0)
                        task_rows.append(
                            dict(
                                participant=subject,
                                fold=key,
                                trial=str(part.Phase.iloc[0]),
                                rest=int(counts[0]),
                                stress=int(counts[1]),
                                activity=int(counts[2]),
                            )
                        )
                        cm += cc
                        rows += n
                        bins += k
                        count += 1
                    collect("CATSA", subject, key, cm, (0, 1), rows, bins, count)
            if seen != set(subjects):
                raise AssertionError("Incomplete CATSA coverage")
            # These two cognitive tasks lack a direct acute-stress reference. Save
            # the held-out predictions and keep them outside the baseline/Stroop grade.
            extra_people = set()
            for subject, trials in external_frames(catsa_zip, ("logic", "sudoku")):
                key, model = participant_models[subject]
                extra_people.add(subject)
                for task, part in trials.groupby("Phase", sort=False):
                    clock = np.unique(part.index.as_unit("ns").asi8 // 10_000_000_000)
                    _, _, _, predictions = score_input(model, part, clock, None)
                    counts = np.bincount(predictions, minlength=3)
                    task_rows.append(
                        dict(
                            participant=subject,
                            fold=key,
                            trial=str(task),
                            rest=int(counts[0]),
                            stress=int(counts[1]),
                            activity=int(counts[2]),
                        )
                    )
            if extra_people != seen:
                raise AssertionError("Missing CATSA task recordings")
            pd.DataFrame(task_rows).sort_values(["participant", "trial"]).to_csv(
                reports / "catsa_tasks.csv", index=False
            )
        for study, archive, tag, classes in [
            ("WESAD", wesad_zip, "wesad", (0, 1)),
            ("PPG-DaLiA", dalia_zip, "dalia", (0, 2)),
        ]:
            if archive is None:
                continue
            manifest = {}
            if field_cache is not None:
                cache_root = Path(field_cache) / tag
                cache_root.mkdir(parents=True, exist_ok=True)
                manifest_file = cache_root / "manifest.json"
                with Path(archive).open("rb") as source:
                    archive_id = hashlib.file_digest(source, "sha256").hexdigest()
                if manifest_file.exists():
                    manifest = json.loads(manifest_file.read_text())
                    if (
                        manifest.get("archive_sha256") != archive_id
                        or manifest.get("schema") != 1
                    ):
                        raise ValueError(
                            "Independent study cache has a different archive fingerprint or labelling scheme"
                        )
                else:
                    manifest = {"archive_sha256": archive_id, "schema": 1, "files": {}}
            for person, member in study_person_files(archive):
                cache = None if field_cache is None else cache_root / (person + ".npz")
                if cache is not None and person in manifest["files"]:
                    with cache.open("rb") as cached:
                        if (
                            hashlib.file_digest(cached, "sha256").hexdigest()
                            != manifest["files"][person]
                        ):
                            raise ValueError(
                                "Study participant cache fingerprint mismatch"
                            )
                    with np.load(cache, allow_pickle=False) as d:
                        arrays = {k: d[k] for k in ("BVP", "EDA", "Temperature")}
                        yy = d["truth"]
                else:
                    arrays, yy = study_arrays(archive, study, member)
                    if cache is not None:
                        with tempfile.NamedTemporaryFile(
                            dir=cache_root, suffix=".tmp", delete=False
                        ) as f:
                            tmp = Path(f.name)
                            np.savez_compressed(f, **arrays, truth=yy)
                        tmp.replace(cache)
                        with cache.open("rb") as cached:
                            manifest["files"][person] = hashlib.file_digest(
                                cached, "sha256"
                            ).hexdigest()
                        with tempfile.NamedTemporaryFile(
                            dir=cache_root, mode="w", delete=False
                        ) as f:
                            tmp = Path(f.name)
                            json.dump(manifest, f)
                        tmp.replace(manifest_file)
                frame = field_frame(arrays)
                clock = np.unique(frame.index.as_unit("ns").asi8 // 10_000_000_000)
                cc, n, k, _ = score_input(deployed, frame, clock, yy)
                collect(study, person, "external", cc, classes, n, k, 1)
    p = pd.DataFrame(people)
    if p.empty:
        raise ValueError("No evaluation cohorts requested")
    if p.duplicated(["study", "participant"]).any():
        raise ValueError("A participant was counted more than once within a study")
    p.to_csv(reports / "participants.csv", index=False)
    if by_task:
        pd.DataFrame(by_task).groupby(["task", "actual"]).sum().to_csv(
            reports / "huawei_tasks.csv"
        )
    if sens:
        pd.DataFrame(sens).groupby(["fold", "setting", "scope"]).sum().to_csv(
            reports / "leave_one_out_sensors.csv"
        )
    with one_core():
        times = measure(deployed, samples(data.rglob("*.parquet"), timing_recordings))
    summary = {
        "deployed_sha256": fingerprint,
        "profile": profile,
        "outer_seed": seed,
        "mean_seconds": float(times.mean()),
        "speed": speed(times.mean()),
        "runs": len(times),
        "cohorts": {},
    }
    cols = [f"n{i}{j}" for i in range(3) for j in range(3)]
    for name, g in p.groupby("study"):
        classes = {
            "Huawei": (0, 1, 2),
            "CATSA": (0, 1),
            "WESAD": (0, 1),
            "PPG-DaLiA": (0, 2),
        }[name]
        cm = g[cols].sum(0).to_numpy(int).reshape(3, 3)
        m = study_metrics(cm, classes)
        rec = dict(
            participants=len(g),
            recordings=int(g.recordings.sum()),
            input_rows=int(g.input_rows.sum()),
            observed_windows=int(g.observed_windows.sum()),
            **m,
            equal_person_mean_observed_macro_f1=float(g.macro_f1_observed.mean()),
            median_person_observed_macro_f1=float(g.macro_f1_observed.median()),
            confusion=cm.tolist(),
        )
        if name == "Huawei":
            rec.update(
                quality=80 * m["macro_f1_study"],
                combined=80 * m["macro_f1_study"] + summary["speed"],
            )
        summary["cohorts"][name] = rec
    summary["participants"] = len(p)
    summary["labeled_windows"] = int(p.windows.sum())
    (reports / "population_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("TechArena-Finland-2026/data"))
    p.add_argument("--reports", type=Path, default=Path("benchmark_reports"))
    p.add_argument("--historical-split", action="store_true")
    p.add_argument("--reference-windows", type=Path)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed for the five-fold participant split",
    )
    p.add_argument("--timing-recordings", type=int, default=50)
    p.add_argument(
        "--audit",
        action="store_true",
        help="Check every sensor and decision setting on held-out people",
    )
    p.add_argument(
        "--fold-models",
        type=Path,
        help="Save or replay models with verified training-corpus fingerprints",
    )
    p.add_argument(
        "--catsa-zip",
        type=Path,
        help="Optional local CATSA v2 archive for fitting missing fold models",
    )
    p.add_argument(
        "--all-participants",
        action="store_true",
        help="Leave each Huawei person out and evaluate every participant in each supplied external study",
    )
    p.add_argument(
        "--wesad-zip",
        type=Path,
        help="Optional original WESAD study archive for independent evaluation",
    )
    p.add_argument(
        "--dalia-zip",
        type=Path,
        help="Optional original PPG-DaLiA data.zip for independent evaluation",
    )
    p.add_argument(
        "--study-cache",
        type=Path,
        help="Optional verified cache created by this benchmark for independent studies",
    )
    args = p.parse_args()
    if args.all_participants:
        if args.historical_split:
            p.error("--historical-split and --all-participants select different tests")
        out = run_participant_check(
            args.data,
            args.reports,
            args.model,
            args.fold_models,
            args.catsa_zip,
            args.wesad_zip,
            args.dalia_zip,
            args.seed,
            args.timing_recordings,
            args.audit,
            field_cache=args.study_cache,
        )
        print(json.dumps(out, indent=2))
        raise SystemExit(0)
    if args.wesad_zip or args.dalia_zip or args.study_cache:
        p.error("Independent study inputs require --all-participants")

    check(
        args.data,
        args.reports,
        args.historical_split,
        args.reference_windows,
        args.model,
        args.seed,
        args.timing_recordings,
        args.audit,
        args.fold_models,
        args.catsa_zip,
    )
