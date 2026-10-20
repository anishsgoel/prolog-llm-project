#!/usr/bin/env bash
# Submit the PLLM baseline suite as SLURM job arrays — EVERYTHING EXCEPT the
# London Underground benchmark tiers.
#
# The London tiers (london_t1_*..london_t5_*, 76 configs) are excluded here and
# submitted separately by scripts/submit_london.sh, because they are a different
# kind of workload: many more configs, much larger KBs, and their own ground-truth
# tractability limits. Mixing them in would make one array task run 76 + 17
# configs sequentially and would bury the per-tier signal in a pooled average.
#
# ONE array task = ONE (method|ablation) x ONE model, so:
#   * ollama loads a model exactly once (no evict/reload thrash between models),
#   * a model that cannot load fails only its own tasks,
#   * resources are sized PER MODEL,
#   * each cell writes its own CSV, so resubmitting one cell cannot clobber others.
#
# This is the ENTRY POINT — run it on the cluster login node:
#
#   bash scripts/submit_baselines.sh
#
# Override any knob via the environment, e.g.:
#   MODELS="gpt-oss:20b llama3:8b" CONFIGS="london_directed bio_grn_p53" \
#   RUNS=3 EVAL_TIMEOUT=3600 bash scripts/submit_baselines.sh
#
# To include the London tiers after all: EXCLUDE_CONFIGS="" bash scripts/submit_baselines.sh
set -euo pipefail

SUITE_NAME="${SUITE_NAME:-baselines}"
export CONFIGS="${CONFIGS:-*}"
# The one substantive difference from the London suite: drop the tier configs.
# A pattern, not a hand-written list, so regenerating the benchmark with a
# different number of configs cannot leave this stale.
export EXCLUDE_CONFIGS="${EXCLUDE_CONFIGS:-london_t*}"
export OUTDIR="${OUTDIR:-results/baselines}"

GPU_TASKS=()   # empty => submit_common.sh uses the standard table
CPU_TASKS=()

# shellcheck source=scripts/submit_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_common.sh"
