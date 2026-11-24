import json
import pathlib
import numpy as np
import pandas as pd

SENSOR_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


def main() -> None:
    data_dir = pathlib.Path("TechArena-Finland-2026/data")
    out_dir = pathlib.Path("chat_summaries")
    out_dir.mkdir(exist_ok=True)

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    rows = []

    for p in parquet_files:
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            df = df.sort_index()
            dt_sec = df.index.to_series().diff().dt.total_seconds().dropna()
            median_dt = float(dt_sec.median()) if len(dt_sec) else np.nan
            n_windows_10s = int(
                ((df.index.max() - df.index.min()).total_seconds() // 10) + 1
            )
        else:
            median_dt = np.nan
            n_windows_10s = int(np.ceil(len(df) / 10))

        rec = {
            "subject": p.stem,
            "protocol": p.parent.name if p.parent != data_dir else "root",
            "n_rows": int(len(df)),
            "approx_10s_windows": n_windows_10s,
            "median_dt_sec": median_dt,
            "has_phase": "Phase" in df.columns,
        }
        if "Phase" in df.columns:
            vc = df["Phase"].fillna("MISSING").value_counts().to_dict()
            rec["top_phases"] = json.dumps(dict(list(vc.items())[:8]))

        for col in SENSOR_COLS:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                rec[f"{col}_nan_pct"] = round(float(s.isna().mean() * 100.0), 2)
                rec[f"{col}_mean"] = round(float(s.mean()), 4) if s.notna().any() else None
                rec[f"{col}_std"] = round(float(s.std()), 4) if s.notna().any() else None
            else:
                rec[f"{col}_nan_pct"] = 100.0
                rec[f"{col}_mean"] = None
                rec[f"{col}_std"] = None
        rows.append(rec)

    summary_df = pd.DataFrame(rows)
    csv_path = out_dir / "file_summaries.csv"
    summary_df.to_csv(csv_path, index=False)

    overview = {
        "n_parquet_files": len(parquet_files),
        "unique_subjects": int(summary_df["subject"].nunique()) if len(summary_df) else 0,
        "protocols": summary_df["protocol"].value_counts().to_dict() if len(summary_df) else {},
        "total_rows": int(summary_df["n_rows"].sum()) if len(summary_df) else 0,
        "total_approx_10s_windows": int(summary_df["approx_10s_windows"].sum()) if len(summary_df) else 0,
    }
    (out_dir / "dataset_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(f"Wrote {overview} and {csv_path}")


if __name__ == "__main__":
    main()