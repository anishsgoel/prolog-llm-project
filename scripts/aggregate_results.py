#!/usr/bin/env python
"""Aggregate the per-method/ablation baseline CSVs into a single comparison.

Reads every ``<indir>/*.csv`` (skipping its own ``comparison_*`` outputs),
concatenates the rows (each already carries ``method``/``ablation`` columns from
batch_run), and writes:

  comparison_all_runs.csv – every row (one per method x ablation x config x model x run)
  comparison_summary.csv  – mean of each numeric metric per (config, model, method, ablation)
  comparison.md           – headline comparison table per (method, ablation),
                            averaged over configs/models/runs
  comparison.xlsx         – workbook with 'results' (all runs) + 'summary' sheets
                            (skipped with a warning if openpyxl is unavailable)

Reuses batch_run's helpers (``_summary_rows``, ``_write_xlsx`` etc.) — it does
NOT modify or re-implement them. Run standalone after the array finishes, or let
submit_baselines.sh schedule it as a dependent job:

  python scripts/aggregate_results.py --indir results/baselines --outdir results/baselines
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# --- make the project importable (mirrors batch_run's own path setup) ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse, don't duplicate: batch_run already knows how to group rows and which
# fields are numeric. Importing it does not run its main().
from experiment.batch_run import (  # noqa: E402
    _numeric_metric_fields,
    _result_fieldnames,
    _summary_fieldnames,
    _summary_rows,
    _write_xlsx,
)

# Honest headline metrics for the markdown table (plain `success` is ~always 1
# and uninformative, so it is shown last, for reference only).
HEADLINE = [
    "valid_success", "proof_matches_gt", "precision_hypothesized",
    "recall_omitted", "hallucination_rate", "llm_queries_total",
    "wall_time_s", "success",
]


def _load_rows(indir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(indir.glob("*.csv")):
        if path.name.startswith("comparison_") or path.name == "comparison.csv":
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
    return rows


def _coerce_numeric(rows: list[dict]) -> list[dict]:
    """CSV values are strings; turn numeric metric columns into floats so the
    means work. Non-numbers (e.g. the "NA" ground-truth sentinel or blanks) are
    left as-is and simply skipped when averaging."""
    numeric = set(_numeric_metric_fields())
    for row in rows:
        for key in numeric:
            val = row.get(key, "")
            if isinstance(val, str) and val not in ("", "NA"):
                try:
                    row[key] = float(val)
                except ValueError:
                    pass
    return rows


def _mean(values) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (sum(nums) / len(nums)) if nums else None


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, help="directory of per-task CSVs")
    ap.add_argument("--outdir", required=True, help="where to write the comparison files")
    args = ap.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _coerce_numeric(_load_rows(indir))
    if not rows:
        print(f"No per-task CSVs found in {indir} — nothing to aggregate.")
        return

    # 1) all runs concatenated ------------------------------------------------
    fieldnames = _result_fieldnames()
    for key in dict.fromkeys(k for r in rows for k in r):
        if key not in fieldnames:
            fieldnames.append(key)
    with open(outdir / "comparison_all_runs.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 2) per (config, model, method, ablation) means (reuse batch_run) --------
    summary = _summary_rows(rows)
    with open(outdir / "comparison_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    # 3) headline table per (method, ablation), averaged over configs ---------
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for row in rows:
        key = (row.get("method", ""), row.get("ablation", ""))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    lines = [
        "# Baseline comparison",
        "",
        f"Averaged over configs/models/runs ({len(rows)} rows across "
        f"{len(order)} method/ablation groups).",
        "",
        "Headline metrics -- **do not** read plain `success` as the headline "
        "(~always 1: the solver can force a proof by injecting a fact).",
        "",
        "| method | ablation | n | " + " | ".join(HEADLINE) + " |",
        "|" + "---|" * (3 + len(HEADLINE)),
    ]
    for key in order:
        method, ablation = key
        grp = groups[key]
        cells = [_mean([g.get(h) for g in grp]) for h in HEADLINE]
        lines.append(
            f"| {method or '-'} | {ablation or '-'} | {len(grp)} | "
            + " | ".join(_fmt(c) for c in cells) + " |"
        )
    (outdir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 4) xlsx workbook (results + summary sheets) — reuse batch_run's writer.
    wrote_xlsx = False
    try:
        _write_xlsx(rows, outdir / "comparison.xlsx", fieldnames)
        wrote_xlsx = True
    except SystemExit:
        print("comparison.xlsx skipped (openpyxl not installed).")
    except Exception as exc:  # keep the CSV/MD outputs even if xlsx fails
        print(f"comparison.xlsx skipped ({exc.__class__.__name__}: {exc}).")

    outputs = "comparison_all_runs.csv, comparison_summary.csv, comparison.md"
    if wrote_xlsx:
        outputs += ", comparison.xlsx"
    print(f"Wrote {outputs} to {outdir}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()