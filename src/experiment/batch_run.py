#!/usr/bin/env python
"""Batch Prolog-LLM experiment runner.

Runs multiple problem configs, logs every run to MLflow, and writes a
summary CSV (optionally an .xlsx too).  Rows are flushed to disk as each
run finishes, so an interrupted batch (crash, Ctrl-C, server dying) keeps
every result completed so far.  The original run.py is unchanged and still
available for quick single-problem debugging.

Usage
-----
    # Run all configs once (default — '*' is the implicit default)
    python src/experiment/batch_run.py
    python src/experiment/batch_run.py --configs "*"

    # Run specific configs
    python src/experiment/batch_run.py --configs london_directed krebs_cycle

    # Three independent repetitions of every config
    python src/experiment/batch_run.py --runs 3 --output results.csv

    # Evaluate several LLM models (each config is run with each model)
    python src/experiment/batch_run.py --models gpt-oss:20b llama3:8b --xlsx results.xlsx

    # Give up on any single evaluation that runs longer than 30 minutes
    python src/experiment/batch_run.py --eval-timeout 1800
    # (default is 7200s = 2 hours; use --eval-timeout 0 to disable the limit)

    # Also export an Excel workbook (requires openpyxl). The workbook has a
    # 'results' sheet (one row per run) and a 'summary' sheet (means per
    # dataset x model).
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
import dataclasses
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- path setup ----
_SRC_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in (str(_SRC_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlflow

import config as _cfg_mod
from experiment.tracking import ALL_CONFIGS, RunMetrics, run_tracked
from experiment.timed_run import run_tracked_timed
from log_setup import ERROR_LOGGER, get_logger

_err_log = get_logger(ERROR_LOGGER)

# Fields stored as MLflow *params* (categorical / non-numeric). The proof
# strings are long free text, so they go to params too (the full text is always
# preserved in the CSV/xlsx; only the MLflow copy is length-capped).
_PARAM_KEYS = {"config_name", "goal", "omit_fact_ids", "model", "run_index",
               "proof", "ground_truth_proof"}
_MLFLOW_PARAM_MAXLEN = 480  # MLflow caps param value length; truncate to stay safe


def _log_to_mlflow(metrics: RunMetrics, run_index: int) -> None:
    """Log one RunMetrics into the active MLflow run."""
    d = metrics.to_dict()
    # Identity → params
    for k in ("config_name", "goal", "omit_fact_ids", "model"):
        mlflow.log_param(k, str(d[k])[:_MLFLOW_PARAM_MAXLEN])
    mlflow.log_param("run_index", run_index)
    # Proof text → params (truncated for MLflow only)
    for k in ("proof", "ground_truth_proof"):
        mlflow.log_param(k, str(d.get(k, ""))[:_MLFLOW_PARAM_MAXLEN])
    # Everything numeric → metrics
    for k, v in d.items():
        if k in _PARAM_KEYS:
            continue
        try:
            mlflow.log_metric(k, float(v))
        except (ValueError, TypeError):
            mlflow.log_param(k, str(v)[:_MLFLOW_PARAM_MAXLEN])


def _evaluate(config_name: str, model: Optional[str], eval_timeout: float):
    """Run one evaluation, optionally under a wall-clock limit.

    Returns ``(RunMetrics, None)`` on success, or ``(None, message)`` if the run
    timed out or failed. When ``eval_timeout`` is 0/negative the evaluation runs
    in-process with no limit (original behaviour).
    """
    try:
        if eval_timeout and eval_timeout > 0:
            res = run_tracked_timed(config_name, model, eval_timeout)
            if res["status"] == "ok":
                return RunMetrics(**res["metrics"]), None
            if res["status"] == "timeout":
                return None, f"timed out after {eval_timeout:g}s"
            return None, res.get("message", "evaluation worker error")
        return run_tracked(config_name, model=model), None
    except Exception as exc:
        _err_log.exception("_evaluate: evaluation failed: config=%s model=%s", config_name, model)
        return None, f"{exc.__class__.__name__}: {exc}"


def _result_fieldnames() -> List[str]:
    """Stable, complete column set known up front.

    Derived from the RunMetrics schema plus the extra keys added per row
    (``run_index``) and on failure (``error``).  Knowing all columns in
    advance lets us write the CSV header once and stream rows as they
    complete, instead of buffering every row to discover the columns.
    """
    names = [f.name for f in dataclasses.fields(RunMetrics)]
    names.append("run_index")
    names.append("error")
    return names


def _numeric_metric_fields() -> List[str]:
    """RunMetrics fields that hold a number (so they can be averaged)."""
    return [f.name for f in dataclasses.fields(RunMetrics) if f.type in ("int", "float")]


def _summary_fieldnames() -> List[str]:
    return ["config_name", "model", "n_runs", "n_errors"] + \
           ["mean_" + f for f in _numeric_metric_fields()]


def _summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate raw rows into one summary row per (dataset, model).

    The mean of each numeric metric is taken over the rows where that metric is
    actually a number, so error rows and NA ground-truth values are skipped
    rather than corrupting the average.
    """
    numeric = _numeric_metric_fields()
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    order: List[tuple] = []
    for row in rows:
        key = (row.get("config_name", ""), row.get("model", ""))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    summary: List[Dict[str, Any]] = []
    for key in order:
        cfg_name, model = key
        grp = groups[key]
        out: Dict[str, Any] = {
            "config_name": cfg_name,
            "model": model,
            "n_runs": len(grp),
            "n_errors": sum(1 for r in grp if r.get("error")),
        }
        for field in numeric:
            vals = [r[field] for r in grp
                    if isinstance(r.get(field), (int, float)) and not isinstance(r.get(field), bool)]
            out["mean_" + field] = (sum(vals) / len(vals)) if vals else None
        summary.append(out)
    return summary


def _write_xlsx(rows: List[Dict[str, Any]], output: Path, fieldnames: List[str]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        _err_log.exception("_write_xlsx: openpyxl not installed")
        raise SystemExit(
            "Writing .xlsx requires the 'openpyxl' package. Install it with "
            "`pip install openpyxl`."
        ) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(k) for k in fieldnames])

    # Per-(dataset, model) summary sheet.
    ws_summary = wb.create_sheet("summary")
    summary_fieldnames = _summary_fieldnames()
    ws_summary.append(summary_fieldnames)
    for srow in _summary_rows(rows):
        ws_summary.append([srow.get(k) for k in summary_fieldnames])

    wb.save(output)


class _ResultsWriter:
    """Persist result rows incrementally so a mid-run crash keeps what finished.

    The CSV is opened once, the header is written, and every appended row is
    flushed straight to disk.  If an ``.xlsx`` path is given it is rewritten
    atomically (temp file + ``os.replace``) after each row, so the workbook on
    disk is always complete and never half-written.
    """

    def __init__(self, csv_path: Path, xlsx_path: Optional[Path] = None,
                 fieldnames: Optional[List[str]] = None):
        self.csv_path = csv_path
        self.xlsx_path = xlsx_path
        self.fieldnames = fieldnames or _result_fieldnames()
        self.rows: List[Dict[str, Any]] = []

        self._fh = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()

    def append(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        if self.xlsx_path is not None:
            self._rewrite_xlsx()

    def _rewrite_xlsx(self) -> None:
        tmp = self.xlsx_path.with_name(self.xlsx_path.name + ".tmp")
        _write_xlsx(self.rows, tmp, self.fieldnames)
        os.replace(tmp, self.xlsx_path)

    def close(self) -> None:
        self._fh.close()


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
        "--models", nargs="+", default=None, metavar="MODEL",
        help="One or more LLM model names to evaluate (e.g. gpt-oss:20b llama3:8b). "
             "Each config is run with each model. Omit to use the model from the "
             "active configuration.",
    )
    parser.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help="Independent runs per config (default: 1). "
             "Useful for estimating variance over stochastic LLM responses.",
    )
    parser.add_argument(
        "--eval-timeout", type=float, default=7200, metavar="SECONDS",
        help="Per-evaluation wall-clock limit in seconds (default: 7200 = 2 hours). "
             "If a single config x model run exceeds it, the run is killed and "
             "recorded as a timeout. Set 0 to disable (run in-process, no limit).",
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

    # Resolve the model list. With no --models, use the configured default so
    # rows (and the summary) are still labelled with a concrete model name.
    if args.models:
        models = args.models
    else:
        _cfg_mod.init_config()
        models = [_cfg_mod.MODEL]

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    output_path = Path(args.output)
    xlsx_path = Path(args.xlsx) if args.xlsx else None
    total = len(models) * len(args.configs) * args.runs
    written = 0

    # The writer flushes each row to disk as soon as it is produced, so an
    # interrupted run (crash, Ctrl-C, server dying) keeps every finished row.
    writer = _ResultsWriter(output_path, xlsx_path)
    try:
        for model in models:
            for cfg_name in args.configs:
                for run_idx in range(args.runs):
                    label = f"[{written + 1}/{total}]"
                    print(f"\n{label} model={model}  config={cfg_name}  run={run_idx + 1}/{args.runs}")

                    metrics, err = _evaluate(cfg_name, model, args.eval_timeout)

                    if use_mlflow:
                        with mlflow.start_run(run_name=f"{model}_{cfg_name}_{run_idx}"):
                            if metrics is not None:
                                _log_to_mlflow(metrics, run_idx)
                            else:
                                mlflow.log_param("config_name", cfg_name)
                                mlflow.log_param("model", model)
                                mlflow.log_param("run_index", run_idx)
                                mlflow.log_param("error", str(err)[:_MLFLOW_PARAM_MAXLEN])

                    if metrics is not None:
                        row = metrics.to_dict()
                        row["run_index"] = run_idx
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
                    else:
                        print(f"  GAVE UP: {err}")
                        row = {
                            "config_name": cfg_name,
                            "model": model,
                            "run_index": run_idx,
                            "error": err,
                        }

                    writer.append(row)
                    written += 1
    finally:
        writer.close()

    if written:
        print(f"\nWrote {written} row(s) to {output_path}")
        if xlsx_path is not None:
            print(f"Wrote {written} row(s) to {xlsx_path} (sheets: 'results', 'summary')")
        if use_mlflow:
            print(f"MLflow experiment: '{args.mlflow_experiment}'  "
                  f"(run `mlflow ui` then open http://127.0.0.1:5000)")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
