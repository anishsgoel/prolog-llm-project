#!/usr/bin/env bash
# Dependent SLURM job: aggregate the per-method/ablation CSVs into a single
# comparison once the array has finished. Submitted by submit_baselines.sh with
# a `--dependency=afterany` on the array (so it also runs if some tasks failed,
# giving a partial comparison). CPU-only — no GPU, no ollama.
#
# Usage (normally via submit_baselines.sh): sbatch scripts/slurm_compare.sh <outdir>
#
#SBATCH --job-name=pllm-compare
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -ef -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/env_setup.sh
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

OUTDIR="${1:-${OUTDIR:-results/baselines}}"
echo "aggregating comparison from $OUTDIR"
python3 scripts/aggregate_results.py --indir "$OUTDIR" --outdir "$OUTDIR"

# Per-model / per-tier / per-difficulty breakdown. Run second and guarded, so a
# failure here still leaves the comparison files from the step above intact.
echo
echo "summarizing per model / tier / difficulty from $OUTDIR"
python3 scripts/summarize_results.py --indir "$OUTDIR" --outdir "$OUTDIR" \
  || echo "WARNING: summarize_results.py failed; comparison.* files are still valid." >&2