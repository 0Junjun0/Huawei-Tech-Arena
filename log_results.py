import datetime
import pathlib


def append_experiment_note(
    run_name: str,
    summary_text: str,
    notes_file: pathlib.Path = pathlib.Path("experiment_notes.md"),
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_lines = [
        line
        for line in summary_text.splitlines()
        if "'nodes_values'" not in line and "np.uint32(" not in line
    ]
    cleaned_body = "\n".join(clean_lines).strip()

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
    notes_file: pathlib.Path = pathlib.Path("experiment_notes.md"),
) -> None:
    try:
        append_experiment_note(
            run_name=run_name,
            summary_text=summary_text,
            notes_file=notes_file,
        )
    except Exception as exc:
        print(f"[warning] Could not write to {notes_file}: {exc}")


if __name__ == "__main__":
    import sys

    label = sys.argv[1] if len(sys.argv) > 1 else "manual-run"
    pasted = sys.stdin.read()
    append_experiment_note(label, pasted)