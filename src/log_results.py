import datetime
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TechArena-Finland-2026"
NOTES_PATH = ROOT_DIR / "notes" / "experiment_notes.md"
SUBMISSION_DIR = ROOT_DIR / "submission"

for p in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "experiments", DATA_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _clean_summary_text(summary_text: str) -> str:
    clean_lines = [
        line
        for line in summary_text.splitlines()
        if "'nodes_values'" not in line and "np.uint32(" not in line
    ]
    return "\n".join(clean_lines).strip()


def append_experiment_note(
    run_name: str,
    summary_text: str,
    notes_file: pathlib.Path = NOTES_PATH,
    as_markdown: bool = False,
    ensure_submission_model_ref: bool = True,
) -> None:
    notes_file = pathlib.Path(notes_file)
    notes_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cleaned_body = _clean_summary_text(summary_text)

    if ensure_submission_model_ref and "submission_model" not in cleaned_body:
        cleaned_body += (
            "\n- **Submission model:** `submission/submission_model.py` + `submission/submission_model.onnx`"
        )

    if as_markdown:
        entry = f"\n<!-- Logged: {timestamp} ({run_name}) -->\n{cleaned_body}\n"
    else:
        entry = (
            f"\n## {timestamp} — {run_name}\n\n"
            f"```text\n{cleaned_body}\n```\n"
        )

    if not notes_file.exists():
        header = "# Huawei Tech Arena 2026 — Experiment Notes\n"
        notes_file.write_text(header + entry, encoding="utf-8")
    else:
        with notes_file.open("a", encoding="utf-8") as f:
            f.write(entry)

    print(f"Saved results to {notes_file.resolve()}")


def log_safe(
    run_name: str,
    summary_text: str,
    notes_file: pathlib.Path = NOTES_PATH,
    as_markdown: bool = False,
    ensure_submission_model_ref: bool = True,
) -> None:
    try:
        append_experiment_note(
            run_name=run_name,
            summary_text=summary_text,
            notes_file=notes_file,
            as_markdown=as_markdown,
            ensure_submission_model_ref=ensure_submission_model_ref,
        )
    except Exception as exc:
        print(f"[warning] Could not write to {notes_file}: {exc}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "manual-run"
    md_mode = "--md" in sys.argv[2:]
    pasted = sys.stdin.read()
    append_experiment_note(label, pasted, as_markdown=md_mode)