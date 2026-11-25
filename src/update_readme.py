import datetime
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
NOTES_PATH = ROOT_DIR / "notes" / "experiment_notes.md"
SUBMISSION_DIR = ROOT_DIR / "submission"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "experiments", DATA_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def read_if_exists(path: pathlib.Path | None, max_chars: int = 4000) -> str:
    if path is None or not path.exists():
        return "_Not found locally._"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return "_Empty file._"
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text


def find_first(candidates: list[pathlib.Path]) -> pathlib.Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    work_dir = ROOT_DIR
    kit_dir = DATA_DIR

    subj_info_path = find_first([
        work_dir / "subject-info.csv",
        kit_dir / "subject-info.csv",
        work_dir / "notes" / "subject-info.csv",
    ])
    constraints_path = find_first([
        work_dir / "data_constraints.txt",
        kit_dir / "data_constraints.txt",
        work_dir / "notes" / "data_constraints.txt",
    ])
    phases_path = find_first([
        work_dir / "phases_to_activity.txt",
        kit_dir / "phases_to_activity.txt",
        work_dir / "notes" / "phases_to_activity.txt",
    ])
    notes_path = find_first([
        NOTES_PATH,
        work_dir / "experiment_notes.md",
        kit_dir / "experiment_notes.md",
    ])

    tracked_files = [
        "src/train_model.py",
        "src/log_results.py",
        "src/auto_log_experiment.py",
        "src/summarize_for_chat.py",
        "src/update_readme.py",
        "experiments/run_experiments.py",
        "experiments/metrics_audit.py",
        "experiments/audit_pipeline.py",
        "experiments/benchmark_run.py",
        "submission/submission_model.py",
        "submission/submission_model.onnx",
        "notes/experiment_notes.md",
    ]
    present_files = []
    for rel_name in tracked_files:
        p = find_first([work_dir / rel_name, kit_dir / pathlib.Path(rel_name).name])
        if p is not None:
            size_kb = p.stat().st_size / 1024.0
            present_files.append(f"- `{rel_name}` ({size_kb:.1f} KB)")

    for zp in sorted(list(work_dir.glob("*.zip")) + list(SUBMISSION_DIR.glob("*.zip"))):
        size_mb = zp.stat().st_size / (1024 * 1024)
        present_files.append(f"- `{zp.relative_to(work_dir)}` ({size_mb:.2f} MB ZIP)")

    subj_info_txt = read_if_exists(subj_info_path, 2500)
    constraints_txt = read_if_exists(constraints_path, 2500)
    phases_txt = read_if_exists(phases_path, 1500)
    notes_txt = read_if_exists(notes_path, 3500)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    files_block = "\n".join(present_files) if present_files else "- _No tracked project files detected in project folder._"

    lines = [
        "# Huawei Tech Arena 2026 - Project & Experimentation README",
        "",
        f"*Last updated:* `{now_str}`",
        "",
        "## 1. Task & Competition Overview",
        "- **Target States:** `Rest/recovery` (`Rest`), `Acute Stress` (`Stress`), and `Physical Activity` (`Activity`).",
        "- **Input Signals:** `BVP`, `EDA`, `HR`, `IBI_dt`, `IBI_duration`, and `Temperature`.",
        "- **Evaluation Window:** 10-second windows, with the most frequent physiological state in each window determining the ground-truth label.",
        "- **Scoring & Constraints:**",
        "  - Quality score: `Macro-F1 * 80` (0-80 points), averaging `Stress`, `Activity`, and `Rest` equally.",
        "  - Speed score: `max(0, (1 - MET / 0.3) * 20)` (0-20 points), measured over 50 runs on 1 CPU core.",
        "  - Runtime environment: Python 3.13, 1 CPU core, multi-threading disabled, at most 4 GB RAM, and submission `.zip` at most 50 MB containing matching `<name>.py` (exposing `get_model()` returning an `ADM_Base` instance) and `<name>.onnx`.",
        "  - Submission rules: up to 3 submissions per team per day; best valid submission counts; update via **View your participations** -> **Model (.zip)**.",
        "",
        "## 2. Current Local Files & Artifacts",
        files_block,
        "",
        "## 3. Dataset & Subject Metadata Snapshot",
        "### Phases-to-Activity Mapping (`phases_to_activity.txt`)",
        "```text",
        phases_txt,
        "```",
        "",
        "### Data Constraints (`data_constraints.txt`)",
        "```text",
        constraints_txt,
        "```",
        "",
        "### Subject Metadata (`subject-info.csv`)",
        "```text",
        subj_info_txt,
        "```",
        "",
        "## 4. What Is Being Tested & How",
        "1. **Dataset & Signal Audit:** Dataset structure, subject/session/window counts, class balance, missing values/NaNs, sampling intervals, corrupted/outlier signals, subject baseline differences, subject/protocol dominance, causal temporal dependencies, and protocol-specific patterns.",
        "2. **Validation Framework (No Leakage):**",
        "   - Random stratified split (diagnostic baseline only - never used to optimize models).",
        "   - Subject-grouped cross-validation (`GroupKFold` by subject ID) and Leave-One-Subject-Out (`LOSO`) where feasible.",
        "   - Protocol-grouped validation where dataset structure supports it.",
        "   - Preprocessing and feature fitting restricted to training splits without subject leakage.",
        "3. **Metrics Reported:** Overall Macro-F1, per-class precision/recall/F1 (`Rest`, `Stress`, `Activity`), confusion matrices, per-subject and per-protocol performance, and mean/standard deviation across validation folds.",
        "4. **Controlled Experimentation & Inference Profiling:**",
        "   - Single-variable changes while preserving the working baseline and logging runs to `notes/experiment_notes.md`.",
        "   - End-to-end `run()` timing (`mean`, `median`, `P95`, `std`), peak memory, `.onnx` file size, and `sklearn` vs. ONNX parity checks under single-thread settings.",
        "",
        "## 5. Recent Experiment Notes Summary",
        "```text",
        notes_txt,
        "```",
    ]

    out_path = work_dir / "README.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {out_path.resolve()}")


if __name__ == "__main__":
    main()