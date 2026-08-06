#!/usr/bin/env bash
# Submit the PLLM baseline suite as SLURM job arrays.
#
# ONE array task = ONE (method|ablation) x ONE model, so:
#   * ollama loads a model exactly once (no evict/reload thrash between models),
#   * a model that cannot load fails only its own tasks,
#   * resources are sized PER MODEL (gpt-oss:120b needs 2 GPUs; llama3:8b needs 1),
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
# It does NOT modify batch_run.py — each task shells out to it, and the
# aggregator only imports its read-only summary helpers.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- knobs (override via env) -----------------------------------------------
MODELS="${MODELS:-gpt-oss:20b gpt-oss:120b llama3:8b ministral-3}"
export CONFIGS="${CONFIGS:-*}"           # "*" = all configs, or "cfg_a cfg_b ..."
export RUNS="${RUNS:-1}"                 # independent repetitions per config
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-7200}"   # per-eval wall-clock cap (s), 0 disables
export OUTDIR="${OUTDIR:-results/baselines}"
export PROJECT_ROOT
export PULL_MODELS="${PULL_MODELS:-0}"   # 1 = `ollama pull` each model first

# --- partitions --------------------------------------------------------------
# Discover what your cluster actually offers with:
#     sinfo -o "%20P %10l %8m %10G %N"
# V100 nodes (n21-n32): 4 x 32 GiB, sm70.   A100 nodes (g01-g10): 4 x 40 GiB, sm80.
GPU_PARTITION="${GPU_PARTITION:-gpuextralong}"        # V100 partition: small models
GPU_PARTITION_BIG="${GPU_PARTITION_BIG:-$GPU_PARTITION}"  # <-- SET THIS to the A100 partition
CPU_PARTITION="${CPU_PARTITION:-cpuextralong}"
COMPARE_PARTITION="${COMPARE_PARTITION:-$CPU_PARTITION}"

# Max concurrently running array tasks. Each LLM task loads its model from
# $HOME/.ollama/models; 11 tasks pulling a 60 GiB blob off a network home dir at
# once will saturate the filesystem long before it saturates the GPUs.
THROTTLE="${THROTTLE:-4}"

# --- per-model resource profiles --------------------------------------------
# echoes: <partition> <gres> <mem> <cpus> <time>
#
# WHY gpu:2 for gpt-oss:120b — its weights are 59.8 GiB, which fits in NEITHER a
# 32 GiB V100 NOR a 40 GiB A100. On one GPU ollama commits a split plan (18/37
# layers on GPU, 31.5 GiB of weights on the host with mmap disabled) and the
# runner is then SIGKILLed by the memory cgroup during load — every /api/chat
# returns HTTP 500 and the whole model's results come out as silent zeros.
# 2 x A100 = 80 GiB holds the model entirely in VRAM.
#
# WHY --cpus-per-task at all — ollama sized its runner at NumThreads:36 while
# SLURM's default allocation is ONE cpu, i.e. ~36x oversubscription on any layer
# that stays on the host. Ask for the cores you intend ollama to use.
model_profile() {
  case "$1" in
    gpt-oss:120b) echo "$GPU_PARTITION_BIG gpu:80gb:1 128G 2 48:00:00" ;;
    gpt-oss:20b)  echo "$GPU_PARTITION     gpu:1 64G  2 24:00:00" ;;
    *)            echo "$GPU_PARTITION     gpu:1 64G  2 12:00:00" ;;
  esac
}

# --- the task table ----------------------------------------------------------
# Comment out lines to skip a method/ablation.
GPU_TASKS=(                 # need the LLM -> get a GPU, once per model
  "--method pllm"
  "--method llm_oneshot"
  "--method llm_qa"
  "--method logic_lm"
  "--ablation verify_on"
  "--ablation min_hypotheses"
  "--ablation no_order"
  "--ablation fixed_depth"
  "--ablation init_only"
  "--ablation backtrack_only"
  "--ablation self_consistency"
)
# Symbolic / no LLM. These NEVER call the model, so running them once per model
# just produces byte-identical duplicate rows (the 20260804 run did exactly that:
# 4 identical copies each, which over-weights them in any pooled average).
# They run ONCE, with a single model label to keep the CSV column populated.
CPU_TASKS=(
  "--method no_abduction"
  "--method random_edge"
  "--method minimal_alp"
)
SYMBOLIC_MODEL_LABEL="${SYMBOLIC_MODEL_LABEL:-${MODELS%% *}}"   # first model in the list

mkdir -p "$OUTDIR" slurm_logs

echo "Submitting baseline suite"
echo "  models : $MODELS"
echo "  configs: $CONFIGS"
echo "  runs   : $RUNS   eval-timeout: ${EVAL_TIMEOUT}s   throttle: %$THROTTLE"
echo "  outdir : $OUTDIR"
echo

DEPS=()

# --- one GPU array per model ------------------------------------------------
for MODEL in $MODELS; do
  read -r PART GRES MEM CPUS TIME <<< "$(model_profile "$MODEL")"
  MODEL_SAFE="${MODEL//[^A-Za-z0-9._-]/-}"
  TF="$OUTDIR/tasks_gpu_${MODEL_SAFE}.txt"
  : > "$TF"
  for T in "${GPU_TASKS[@]}"; do printf '%s %s\n' "$T" "$MODEL" >> "$TF"; done
  N=${#GPU_TASKS[@]}
  [ "$N" -gt 0 ] || continue

  if [ "$MODEL" = "gpt-oss:120b" ] && [ "$GPU_PARTITION_BIG" = "$GPU_PARTITION" ]; then
    echo "  !! WARNING: gpt-oss:120b is being sent to '$GPU_PARTITION' with $GRES."
    echo "     Its 59.8 GiB of weights need >= 2 GPUs (2xA100=80GiB). Set"
    echo "     GPU_PARTITION_BIG=<a100 partition> or it will be killed at load."
  fi

  JID=$(sbatch --parsable --array="0-$((N-1))%${THROTTLE}" \
      --job-name="pllm-${MODEL_SAFE}" \
      -p "$PART" --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --export=ALL,TASKS_FILE="$TF" \
      scripts/slurm_baselines.sh)
  echo "  -> $MODEL: array $JID  ($N tasks; $PART $GRES mem=$MEM cpus=$CPUS time=$TIME)"
  DEPS+=("$JID")
done

# --- one CPU array, model-independent ---------------------------------------
NC=${#CPU_TASKS[@]}
if [ "$NC" -gt 0 ]; then
  TF="$OUTDIR/tasks_cpu.txt"
  : > "$TF"
  for T in "${CPU_TASKS[@]}"; do printf '%s %s\n' "$T" "$SYMBOLIC_MODEL_LABEL" >> "$TF"; done
  JID=$(sbatch --parsable --array="0-$((NC-1))" \
      --job-name="pllm-symbolic" \
      -p "$CPU_PARTITION" --gres=none --mem=16G --cpus-per-task=4 --time=08:00:00 \
      --export=ALL,TASKS_FILE="$TF" \
      scripts/slurm_baselines.sh)
  echo "  -> symbolic: array $JID  ($NC tasks; $CPU_PARTITION, no GPU, label=$SYMBOLIC_MODEL_LABEL)"
  DEPS+=("$JID")
fi

# --- aggregation, after everything ------------------------------------------
# afterany: also runs if some tasks failed/timed out, so you still get a partial
# comparison. A real script (not --wrap) so it sources the same module/venv env.
DEP="afterany"
for J in "${DEPS[@]}"; do DEP="$DEP:$J"; done
CMP_JID=$(sbatch --parsable --dependency="$DEP" \
    -p "$COMPARE_PARTITION" --gres=none \
    scripts/slurm_compare.sh "$OUTDIR")
echo "  -> compare job: $CMP_JID (waits for all arrays)"

cat <<EOF

When done, see: $OUTDIR/comparison.md, comparison.xlsx, comparison_summary.csv, comparison_all_runs.csv
Per-cell outputs are $OUTDIR/<method>__<model>.{csv,xlsx} (one file per method x model).

Sanity checks after the run:
  # any task killed rather than finished?
  sacct -j <arrayjobid> -o JobID%20,JobName%18,State,ExitCode,MaxRSS,ReqMem,Elapsed
  # did any model fail its preflight?
  grep -l "SMOKE FAIL" slurm_logs/pllm-*_*.err
  # did an ollama runner get OOM-killed?
  grep -l "signal: killed" slurm_logs/ollama_*.log
EOF
