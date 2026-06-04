#!/usr/bin/env python
"""Batch Prolog-LLM experiment runner.

Runs multiple problem configs, logs every run to MLflow, and writes a
summary CSV.  The original run.py is unchanged and still available for
quick single-problem debugging.

Usage
-----
    # Run all configs once (default — '*' is the implicit default)
    python src/experiment/batch_run.py
    python src/experiment/batch_run.py --configs "*"

    # Run specific configs
    python src/experiment/batch_run.py --configs london_directed krebs_cycle

    # Three independent repetitions of every config
    python src/experiment/batch_run.py --runs 3 --output results.csv

    # Also export an Excel workbook (requires openpyxl)
    python src/experiment/batch_run.py --xlsx results.xlsx

    # Custom MLflow server
    python src/experiment/batch_run.py --tracking-uri http://localhost:5000

    # CSV only, no MLflow
    python src/experiment/batch_run.py --no-mlflow --output results.csv

MLflow UI
---------
    mlflow ui            # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---- path setup ----
_SRC_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in (str(_SRC_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlflow

from experiment.tracking import ALL_CONFIGS, RunMetrics, run_tracked

# Fields stored as MLflow *params* (categorical / non-numeric)
_PARAM_KEYS = {"config_name", "goal", "omit_fact_ids", "model", "run_index"}


def _log_to_mlflow(metrics: RunMetrics, run_index: int) -> None:
    """Log one RunMetrics into the active MLflow run."""
    d = metrics.to_dict()
    # Identity → params
    for k in ("config_name", "goal", "omit_fact_ids", "model"):
        mlflow.log_param(k, d[k])
    mlflow.log_param("run_index", run_index)
    # Everything numeric → metrics
    for k, v in d.items():
        if k in _PARAM_KEYS:
            continue
        try:
            mlflow.log_metric(k, float(v))
        except (ValueError, TypeError):
            mlflow.log_param(k, str(v))


def _collect_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Union of all keys across rows, preserving first-seen order."""
    seen: set[str] = set()
    fieldnames: List[str] = []
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    return fieldnames


def _write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    fieldnames = _collect_fieldnames(rows)
    with open(output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_xlsx(rows: List[Dict[str, Any]], output: Path) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise SystemExit(
            "Writing .xlsx requires the 'openpyxl' package. Install it with "
            "`pip install openpyxl`."
        ) from exc

    fieldnames = _collect_fieldnames(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(k) for k in fieldnames])
    wb.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Prolog-LLM runner: multi-config, MLflow logging, CSV export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs", nargs="+", default=["*"], metavar="NAME",
        help="Config YAML stems to run. Use '*' (or omit) to run all. "
             f"Available: {', '.join(ALL_CONFIGS)}",
    )
    parser.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help="Independent runs per config (default: 1). "
             "Useful for estimating variance over stochastic LLM responses.",
    )
    parser.add_argument(
        "--output", default="results.csv", metavar="PATH",
        help="Output CSV file path (default: results.csv).",
    )
    parser.add_argument(
        "--xlsx", default=None, metavar="PATH",
        help="Also write results to this .xlsx file (requires openpyxl). "
             "Omit to skip Excel output.",
    )
    parser.add_argument(
        "--mlflow-experiment", default="prolog-llm", metavar="NAME",
        help="MLflow experiment name (default: prolog-llm).",
    )
    parser.add_argument(
        "--tracking-uri", default=None, metavar="URI",
        help="MLflow tracking server URI. "
             "Omit to use the default local mlruns/ directory.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging; write CSV only.",
    )
    args = parser.parse_args()

    # Expand '*' (or a shell-escaped '*') to the full config list.
    if args.configs == ["*"] or "*" in args.configs:
        args.configs = ALL_CONFIGS

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    all_rows: List[Dict[str, Any]] = []
    total = len(args.configs) * args.runs

    for cfg_name in args.configs:
        for run_idx in range(args.runs):
            label = f"[{len(all_rows) + 1}/{total}]"
            print(f"\n{label} config={cfg_name}  run={run_idx + 1}/{args.runs}")
            try:
                if use_mlflow:
                    with mlflow.start_run(run_name=f"{cfg_name}_{run_idx}"):
                        metrics = run_tracked(cfg_name)
                        _log_to_mlflow(metrics, run_idx)
                else:
                    metrics = run_tracked(cfg_name)

                row = metrics.to_dict()
                row["run_index"] = run_idx
                all_rows.append(row)

                print(
                    f"  success={metrics.success}"
                    f"  conf={metrics.proof_confidence:.3f}"
                    f"  proof_depth={metrics.proof_depth}"
                    f"  gt_depth={metrics.ground_truth_proof_depth}"
                    f"  llm_queries={metrics.llm_queries_total}"
                    f"  hallucination_rate={metrics.hallucination_rate:.2f}"
                    f"  recall_omitted={metrics.recall_omitted:.2f}"
                    f"  time={metrics.wall_time_s:.1f}s"
                )

            except Exception as exc:
                print(f"  ERROR: {exc}")
                all_rows.append({
                    "config_name": cfg_name,
                    "run_index": run_idx,
                    "error": str(exc),
                })

    if all_rows:
        output_path = Path(args.output)
        _write_csv(all_rows, output_path)
        print(f"\nWrote {len(all_rows)} row(s) to {output_path}")
        if args.xlsx:
            xlsx_path = Path(args.xlsx)
            _write_xlsx(all_rows, xlsx_path)
            print(f"Wrote {len(all_rows)} row(s) to {xlsx_path}")
        if use_mlflow:
            print(f"MLflow experiment: '{args.mlflow_experiment}'  "
                  f"(run `mlflow ui` then open http://127.0.0.1:5000)")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
