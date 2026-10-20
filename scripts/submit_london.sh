#!/usr/bin/env bash
# Submit ONLY the London Underground benchmark tiers as SLURM job arrays.
#
#   bash scripts/submit_london.sh
#
# Companion to scripts/submit_baselines.sh, which excludes these configs. Both
# share scripts/submit_common.sh, so the resource profiles, task table and
# aggregation wiring stay identical between the two suites.
#
# Selection is the pattern 'london_t*', expanded by batch_run against the
# configs directory, so it automatically tracks whatever
# tools/build_london_benchmark.py last generated (76 configs across T1..T5 at
# the time of writing) with nothing to keep in sync by hand.
#
# Useful restrictions:
#   CONFIGS="london_t1_* london_t2_*"  bash scripts/submit_london.sh   # small tiers only
#   CONFIGS="london_t5_*"              bash scripts/submit_london.sh   # full network only
#   CONFIGS="london_t*_1edge_*"        bash scripts/submit_london.sh   # single-omission only
#   MODELS="gpt-oss:20b" RUNS=3        bash scripts/submit_london.sh
#
# WHY a longer per-eval timeout than the baseline suite: proof-search cost on
# these KBs grows steeply with the hop count of the goal and with tier size (see
# docs/LONDON_BENCHMARK.md §7). The ground truth is bounded separately by
# GROUND_TRUTH_TIMEOUT_S; this budget covers the meta-solver's own search on the
# ABLATED KB, which is strictly harder than the ground-truth search.
set -euo pipefail

SUITE_NAME="${SUITE_NAME:-london}"
export CONFIGS="${CONFIGS:-london_t*}"
export EXCLUDE_CONFIGS="${EXCLUDE_CONFIGS:-}"     # nothing to exclude; this IS the London suite
export OUTDIR="${OUTDIR:-results/london}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-14400}"      # 4 h per (config x model) evaluation

# 76 configs per task instead of ~17, on much larger KBs, so give the arrays
# more wall-clock than the baseline suite's defaults.
TIME_BIG="${TIME_BIG:-72:00:00}"
TIME_STD="${TIME_STD:-48:00:00}"
TIME_SMALL="${TIME_SMALL:-24:00:00}"
CPU_TIME="${CPU_TIME:-24:00:00}"
CPU_MEM="${CPU_MEM:-32G}"

GPU_TASKS=()   # empty => submit_common.sh uses the standard table
CPU_TASKS=()

# shellcheck source=scripts/submit_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_common.sh"
