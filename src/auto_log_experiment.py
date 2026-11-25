import datetime
import pathlib
import sys
from typing import Optional

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
NOTES_PATH = ROOT_DIR / "notes" / "experiment_notes.md"
SUBMISSION_DIR = ROOT_DIR / "submission"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "experiments", DATA_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def append_experiment_to_notes(
    *,
    exp_id: str,
    title: str,
    hypothesis: str,
    changes_made: str,
    files_modified: list[str] | str,
    validation_method: str,
    prev_macro_f1: str,
    new_macro_f1: str,
    per_class_f1: str,
    met_sec: Optional[float],
    median_sec: Optional[float] = None,
    p95_sec: Optional[float] = None,
    std_sec: Optional[float] = None,
    peak_ram_mb: Optional[float] = None,
    onnx_size_mb: Optional[float] = 26.43,
    quality_score: Optional[float] = None,
    speed_score: Optional[float] = None,
    total_score: Optional[float] = None,
    conclusion: str,
    notes_path: pathlib.Path = NOTES_PATH,
) -> pathlib.Path:
    """Appends a structured experiment entry to experiment_notes.md."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    files_str = (
        ", ".join(f"`{f}`" for f in files_modified)
        if isinstance(files_modified, list)
        else str(files_modified)
    )

    if speed_score is None and met_sec is not None:
        speed_score = max(0.0, min(20.0, (1.0 - met_sec / 0.30) * 20.0))

    exec_parts = []
    if met_sec is not None:
        exec_parts.append(f"MET: {met_sec:.4f} s")
    if median_sec is not None:
        exec_parts.append(f"Median: {median_sec:.4f} s")
    if p95_sec is not None:
        exec_parts.append(f"P95: {p95_sec:.4f} s")
    if std_sec is not None:
        exec_parts.append(f"Std: {std_sec:.4f} s")
    if peak_ram_mb is not None:
        exec_parts.append(f"Peak RAM: {peak_ram_mb:.2f} MB")
    if onnx_size_mb is not None:
        exec_parts.append(f"ONNX size: {onnx_size_mb:.2f} MB")
    exec_line = " | ".join(exec_parts) if exec_parts else "N/A"

    if total_score is not None:
        q_part = f"Quality: {quality_score:.2f}/80, " if quality_score is not None else ""
        s_part = f"Speed: {speed_score:.2f}/20" if speed_score is not None else ""
        score_line = f"{total_score:.2f} / 100 ({q_part}{s_part})".rstrip(", ")
    elif speed_score is not None:
        score_line = f"Speed component: {speed_score:.2f} / 20.00 (Quality pending)"
    else:
        score_line = "N/A"

    entry = (
        f"\n<!-- Logged: {now_str} ({exp_id}) -->\n"
        f"### {exp_id}: {title}\n"
        f"- **Experiment ID:** {exp_id}\n"
        f"- **Hypothesis:** {hypothesis}\n"
        f"- **Changes made:** {changes_made}\n"
        f"- **Files modified:** {files_str}\n"
        f"- **Validation method:** {validation_method}\n"
        f"- **Previous macro-F1:** {prev_macro_f1}\n"
        f"- **New macro-F1:** {new_macro_f1}\n"
        f"- **Per-class F1:** {per_class_f1}\n"
        f"- **Execution time:** {exec_line}\n"
        f"- **Estimated total competition score:** {score_line}\n"
        f"- **Conclusion:** {conclusion}\n"
    )

    notes_path = pathlib.Path(notes_path)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "# Experiment Notes\n"
    if not existing.endswith("\n"):
        existing += "\n"
    notes_path.write_text(existing + entry, encoding="utf-8")
    return notes_path


if __name__ == "__main__":
    out_file = append_experiment_to_notes(
        exp_id="EXP-OPT-02",
        title="Unnecessary Pandas Operations (Chaining, Concat & Repeated fillna)",
        hypothesis=(
            "Streamlining Pandas chaining, .add_suffix() + pd.concat(..., axis=1), and repeated "
            ".fillna(0.0) reduces index-alignment overhead while preserving exact predictions."
        ),
        changes_made=(
            "Benchmarked EXP-OPT-02 parity and 50-run single-threaded run() latency on 10 sample "
            "recordings without modifying baseline submission artifacts."
        ),
        files_modified=["experiments/run_experiments.py", "notes/experiment_notes.md"],
        validation_method="10-recording schema/label/shuffled-column parity + 50-run single-core run() benchmark (1 thread)",
        prev_macro_f1="Raw 0.5541 | Tuned 0.5805 (from EXP-OPT-01)",
        new_macro_f1="Unchanged (parity check PASSED on 10 recordings)",
        per_class_f1="Rest: 0.6751 | Stress: 0.2700 | Activity: 0.7962 (prior EXP-OPT-01 baseline)",
        met_sec=0.0654,
        median_sec=0.0562,
        p95_sec=0.1197,
        std_sec=0.0308,
        peak_ram_mb=19.06,
        onnx_size_mb=26.43,
        conclusion=(
            "Discard/do not overwrite baseline with this run: parity PASSED and peak RAM stayed low (19.06 MB), "
            "but MET rose to 0.0654 s (P95 0.1197 s, Est. Speed 15.64/20) vs. ~0.034-0.038 s in audit_pipeline.py."
        ),
    )
    print(f"Appended EXP-OPT-02 to {out_file.resolve()}")
