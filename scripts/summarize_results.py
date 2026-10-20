#!/usr/bin/env python
"""Per-model (and per-tier, per-difficulty) summary of a results directory.

Complements ``scripts/aggregate_results.py``, which answers "which METHOD wins,
averaged over everything". This one answers the questions that average hides:

  * per model            – is the ranking of methods the same on 8b as on 120b?
  * per method x model   – the full matrix, so a method that only works on one
                           model cannot hide inside a pooled mean
  * per tier             – how does each method degrade as the KB grows from 25
                           to 270 stations (London benchmark only)
  * per difficulty       – by number of omitted edges, omission placement, and
                           shortest-path length
  * reliability          – error rate, timeout rate, and how often the ground
                           truth came back NA, per model; a method can only be
                           compared on rows where the ground truth exists

The difficulty columns come from ``benchmark_manifest.csv``, joined on
``config_name``. Rows from configs that are not in the manifest (bio_*,
bohemia_*, krebs_*, the small london_* ones) still appear in the per-model and
per-method tables; they are simply absent from the tier/difficulty ones.

Usage:
    python scripts/summarize_results.py --indir results/london --outdir results/london

Reads the same per-task CSVs as aggregate_results.py and writes:
    summary.md                     everything below, as one readable report
    summary_by_model.csv
    summary_by_method_model.csv
    summary_by_tier.csv            (London only)
    summary_by_difficulty.csv      (London only)
    summary_reliability.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse, don't duplicate: aggregate_results already knows how to find and
# numeric-coerce the per-task CSVs, and batch_run owns the metric schema.
from aggregate_results import _coerce_numeric, _fmt, _load_rows, _mean  # noqa: E402

# Honest headline metrics. Plain `success` is ~always 1 (the solver can force a
# proof by injecting a fact), so it is shown last and never leads a table.
HEADLINE = [
    "valid_success", "proof_matches_gt", "precision_hypothesized",
    "recall_omitted", "hallucination_rate", "llm_queries_total",
    "wall_time_s", "success",
]

MANIFEST_COLUMNS = [
    "tier", "n_omitted_edges", "placement", "shortest_path_hops",
    "depth_band", "crosses_junction", "shortcut_plausible", "gt_proof_depth",
]


def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    """config_name -> difficulty attributes, for the generated benchmark configs."""
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("config_name") or "").strip()
            if not name:
                continue
            out[name] = {c: row.get(c, "") for c in MANIFEST_COLUMNS}
    return out


def _join_manifest(rows: List[dict], manifest: Dict[str, Dict[str, str]]) -> int:
    matched = 0
    for row in rows:
        attrs = manifest.get(row.get("config_name", ""))
        if attrs:
            row.update(attrs)
            matched += 1
    return matched


def _method_label(row: dict) -> str:
    ablation = (row.get("ablation") or "").strip()
    method = (row.get("method") or "").strip() or "-"
    return f"{method}[{ablation}]" if ablation else method


def _is_error(row: dict) -> bool:
    return bool((row.get("error") or "").strip())


def _gt_is_na(row: dict) -> bool:
    """Ground truth unavailable, so proof_matches_gt / gt depth mean nothing.

    compute_metrics writes the NA sentinel when the ground-truth search exceeds
    its budget; it survives the CSV round-trip as the literal string "NA".
    """
    return str(row.get("ground_truth_proof_depth", "")).strip() in ("NA", "")


def _group(rows: Iterable[dict], key_fn) -> Tuple[Dict[Any, List[dict]], List[Any]]:
    groups: Dict[Any, List[dict]] = defaultdict(list)
    order: List[Any] = []
    for row in rows:
        key = key_fn(row)
        if key not in groups:
            order.append(key)
        groups[key].append(row)
    return groups, order


def _table(rows: List[dict], key_fn, key_names: Sequence[str],
           metrics: Sequence[str] = HEADLINE,
           sort: bool = True) -> Tuple[List[str], List[List[str]], List[dict]]:
    """Build one grouped mean table: (header, markdown cells, csv records)."""
    groups, order = _group(rows, key_fn)
    if sort:
        order = sorted(order, key=lambda k: tuple(str(x) for x in (k if isinstance(k, tuple) else (k,))))

    header = list(key_names) + ["n", "n_err"] + list(metrics)
    md_rows: List[List[str]] = []
    records: List[dict] = []
    for key in order:
        grp = groups[key]
        keys = list(key) if isinstance(key, tuple) else [key]
        n_err = sum(1 for g in grp if _is_error(g))
        means = [_mean([g.get(m) for g in grp]) for m in metrics]
        md_rows.append([str(k) for k in keys] + [str(len(grp)), str(n_err)]
                       + [_fmt(m) for m in means])
        rec = dict(zip(key_names, [str(k) for k in keys]))
        rec.update({"n": len(grp), "n_err": n_err})
        rec.update({m: means[i] for i, m in enumerate(metrics)})
        records.append(rec)
    return header, md_rows, records


def _md(title: str, note: str, header: Sequence[str], rows: List[List[str]]) -> List[str]:
    out = [f"## {title}", ""]
    if note:
        out += [note, ""]
    if not rows:
        return out + ["_(no rows)_", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")
    return out


def _write_csv(path: Path, records: List[dict], header: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: ("" if rec.get(k) is None else rec.get(k)) for k in header})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, help="directory of per-task CSVs")
    ap.add_argument("--outdir", required=True, help="where to write the summary files")
    ap.add_argument("--manifest", default=str(_PROJECT_ROOT / "benchmark_manifest.csv"),
                    help="benchmark_manifest.csv, for the tier/difficulty tables")
    args = ap.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _coerce_numeric(_load_rows(indir))
    if not rows:
        print(f"No per-task CSVs found in {indir} — nothing to summarize.")
        return

    manifest = _load_manifest(Path(args.manifest))
    matched = _join_manifest(rows, manifest)

    n_models = len({r.get("model", "") for r in rows})
    doc: List[str] = [
        "# Results summary",
        "",
        f"{len(rows)} runs, {n_models} model(s), "
        f"{len({_method_label(r) for r in rows})} method/ablation cell(s), "
        f"{len({r.get('config_name', '') for r in rows})} config(s).",
        "",
        "`valid_success` and `proof_matches_gt` are the honest headline metrics. "
        "**Do not** read plain `success`: the solver can always inject a fact to "
        "force a proof, so it is ~always 1 and is shown for reference only.",
        "",
    ]
    if matched:
        doc += [f"Joined {matched} of {len(rows)} rows against "
                f"`{Path(args.manifest).name}` for the tier/difficulty tables.", ""]
    else:
        doc += ["_No rows matched the benchmark manifest, so the tier and "
                "difficulty tables are omitted._", ""]

    # --- 1. per model --------------------------------------------------------
    header, md_rows, records = _table(rows, lambda r: r.get("model", "-"), ["model"])
    doc += _md("Per model", "Averaged over every method and config.", header, md_rows)
    _write_csv(outdir / "summary_by_model.csv", records, header)

    # --- 2. per method x model ----------------------------------------------
    header, md_rows, records = _table(
        rows, lambda r: (_method_label(r), r.get("model", "-")), ["method", "model"])
    doc += _md("Per method x model",
               "The full matrix. A method that only works on one model cannot "
               "hide inside a pooled mean here.", header, md_rows)
    _write_csv(outdir / "summary_by_method_model.csv", records, header)

    # --- 3. per tier ---------------------------------------------------------
    tier_rows = [r for r in rows if (r.get("tier") or "").strip()]
    if tier_rows:
        header, md_rows, records = _table(
            tier_rows, lambda r: (r.get("tier", "-"), _method_label(r)), ["tier", "method"])
        doc += _md("Per tier x method",
                   "How each method degrades as the KB grows from 25 stations "
                   "(T1) to 270 (T5). See docs/LONDON_BENCHMARK.md for the ladder.",
                   header, md_rows)
        _write_csv(outdir / "summary_by_tier.csv", records, header)

        header, md_rows, records = _table(
            tier_rows, lambda r: (r.get("tier", "-"), r.get("model", "-")), ["tier", "model"])
        doc += _md("Per tier x model", "", header, md_rows)

    # --- 4. per difficulty axis ---------------------------------------------
    if tier_rows:
        all_records: List[dict] = []
        for axis, label in [("n_omitted_edges", "omitted edges"),
                            ("placement", "omission placement"),
                            ("shortest_path_hops", "shortest-path hops"),
                            ("shortcut_plausible", "plausible direct shortcut"),
                            ("crosses_junction", "path crosses a junction")]:
            subset = [r for r in tier_rows if str(r.get(axis, "")).strip()]
            if not subset:
                continue
            header, md_rows, records = _table(subset, lambda r, a=axis: r.get(a, "-"), [axis])
            doc += _md(f"By {label}", "", header, md_rows)
            for rec in records:
                all_records.append({"axis": axis, "value": rec.get(axis, ""),
                                    **{k: v for k, v in rec.items() if k != axis}})
        if all_records:
            diff_header = ["axis", "value", "n", "n_err"] + list(HEADLINE)
            _write_csv(outdir / "summary_by_difficulty.csv", all_records, diff_header)

    # --- 5. reliability ------------------------------------------------------
    groups, order = _group(rows, lambda r: r.get("model", "-"))
    rel_header = ["model", "n", "n_err", "err_rate", "n_gt_na", "gt_na_rate",
                  "mean_wall_time_s", "mean_llm_queries_total"]
    rel_rows: List[List[str]] = []
    rel_records: List[dict] = []
    for key in sorted(order, key=str):
        grp = groups[key]
        n_err = sum(1 for g in grp if _is_error(g))
        n_na = sum(1 for g in grp if _gt_is_na(g))
        rec = {
            "model": key, "n": len(grp), "n_err": n_err,
            "err_rate": n_err / len(grp),
            "n_gt_na": n_na, "gt_na_rate": n_na / len(grp),
            "mean_wall_time_s": _mean([g.get("wall_time_s") for g in grp]),
            "mean_llm_queries_total": _mean([g.get("llm_queries_total") for g in grp]),
        }
        rel_records.append(rec)
        rel_rows.append([str(rec["model"]), str(rec["n"]), str(rec["n_err"]),
                         _fmt(rec["err_rate"]), str(rec["n_gt_na"]), _fmt(rec["gt_na_rate"]),
                         _fmt(rec["mean_wall_time_s"]), _fmt(rec["mean_llm_queries_total"])])
    doc += _md("Reliability per model",
               "`gt_na_rate` is the fraction of runs whose ground-truth search "
               "exceeded its budget. `proof_matches_gt` is only meaningful on the "
               "remaining rows, so a high rate here devalues that column above.",
               rel_header, rel_rows)
    _write_csv(outdir / "summary_reliability.csv", rel_records, rel_header)

    (outdir / "summary.md").write_text("\n".join(doc) + "\n", encoding="utf-8")
    written = ["summary.md", "summary_by_model.csv", "summary_by_method_model.csv",
               "summary_reliability.csv"]
    if tier_rows:
        written += ["summary_by_tier.csv", "summary_by_difficulty.csv"]
    print(f"Wrote {', '.join(written)} to {outdir}\n")
    print("\n".join(doc))


if __name__ == "__main__":
    main()
