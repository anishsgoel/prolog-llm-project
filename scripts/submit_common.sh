# Shared submission logic for the PLLM SLURM suites — SOURCED, not executed.
#
# scripts/submit_baselines.sh (everything except the London tiers) and
# scripts/submit_london.sh (only the London tiers) differ solely in which
# configs they select and where they write. Everything else — the per-model
# resource profiles, the task table, the one-array-per-model layout, the
# dependent aggregation job — lives here so the two suites cannot drift apart.
#
# The caller sets its knobs, then sources this file. Anything already set in the
# environment wins, so every value below stays overridable per invocation.
#
# Expected from the caller (all optional except SUITE_NAME):
#   SUITE_NAME        short tag used in job names and messages   (required)
#   CONFIGS           config selection passed to --configs
#   EXCLUDE_CONFIGS   patterns passed to --exclude-configs ("" = none)
#   OUTDIR            results directory
#   GPU_TASKS[]       array of "--method X" / "--ablation X" needing the LLM
#   CPU_TASKS[]       array of symbolic (no-LLM) tasks
#
# It does NOT modify batch_run.py — each task shells out to it, and the
# aggregators only import its read-only summary helpers.

: "${SUITE_NAME:?submit_common.sh: SUITE_NAME must be set by the caller}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- knobs (override via env) -----------------------------------------------
MODELS="${MODELS:-gpt-oss:20b gpt-oss:120b llama3:8b ministral-3}"
export CONFIGS="${CONFIGS:-*}"                # "*" = all, or "cfg_a cfg_b", or "london_t*"
export EXCLUDE_CONFIGS="${EXCLUDE_CONFIGS:-}" # patterns to drop; "" = drop nothing
export RUNS="${RUNS:-1}"                      # independent repetitions per config
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-7200}"   # per-eval wall-clock cap (s), 0 disables
export OUTDIR="${OUTDIR:-results/${SUITE_NAME}}"
export PROJECT_ROOT
export PULL_MODELS="${PULL_MODELS:-0}"        # 1 = `ollama pull` each model first

# --- partitions --------------------------------------------------------------
# Discover what your cluster actually offers with:
#     sinfo -o "%20P %10l %8m %10G %N"
# V100 nodes (n21-n32): 4 x 32 GiB, sm70.   A100 nodes (g01-g10): 4 x 40 GiB, sm80.
GPU_PARTITION="${GPU_PARTITION:-gpuextralong}"
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
# WHY an 80 GiB card for gpt-oss:120b — its weights are 59.8 GiB, which fits in
# NEITHER a 32 GiB V100 NOR a 40 GiB A100. When it does not fit, ollama commits a
# split plan (18/37 layers on GPU, 31.5 GiB of weights on the host with mmap
# disabled) and the runner is then SIGKILLed by the memory cgroup during load —
# every /api/chat returns HTTP 500 and the whole model's results come out as
# silent zeros. One 80 GiB card holds it entirely in VRAM with ~20 GiB spare.
#
# WHY --cpus-per-task at all — ollama sizes its runner from the host core count
# regardless of the cgroup quota, while SLURM's default allocation is ONE cpu.
# With the model fully in VRAM the CPU is out of the token path and 2 is enough
# (ollama + batch_run parent + the spawned eval worker in timed_run.py).
model_profile() {
  case "$1" in
    gpt-oss:120b) echo "$GPU_PARTITION_BIG gpu:2 128G ${TASK_CPUS:-2} ${TIME_BIG:-48:00:00}" ;;
    gpt-oss:20b)  echo "$GPU_PARTITION     gpu:1 64G  ${TASK_CPUS:-2} ${TIME_STD:-24:00:00}" ;;
    *)            echo "$GPU_PARTITION     gpu:1 64G  ${TASK_CPUS:-2} ${TIME_SMALL:-12:00:00}" ;;
  esac
}

# --- default task table (caller may override before sourcing) ----------------
if [ "${#GPU_TASKS[@]}" -eq 0 ]; then
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
    "--ablation unify_target"
    "--ablation no_soft_facts_in_prompt"
    "--ablation self_consistency"
  )
fi
# Symbolic / no LLM. These NEVER call the model, so running them once per model
# just produces byte-identical duplicate rows, which over-weights them in any
# pooled average. They run ONCE, with a single model label to keep the column
# populated.
if [ "${#CPU_TASKS[@]}" -eq 0 ]; then
  CPU_TASKS=(
    "--method no_abduction"
    "--method random_edge"
    "--method minimal_alp"
  )
fi
SYMBOLIC_MODEL_LABEL="${SYMBOLIC_MODEL_LABEL:-${MODELS%% *}}"   # first model in the list

mkdir -p "$OUTDIR" slurm_logs

echo "Submitting suite: $SUITE_NAME"
echo "  models  : $MODELS"
echo "  configs : $CONFIGS"
[ -n "$EXCLUDE_CONFIGS" ] && echo "  exclude : $EXCLUDE_CONFIGS"
echo "  runs    : $RUNS   eval-timeout: ${EVAL_TIMEOUT}s   throttle: %$THROTTLE"
echo "  outdir  : $OUTDIR"
echo "  gpu tasks: ${#GPU_TASKS[@]}   cpu tasks: ${#CPU_TASKS[@]}"
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
    echo "     Its 59.8 GiB of weights need one >=64 GiB GPU (or 2 smaller ones);"
    echo "     the V100/A100-40GB partition has neither. Set GPU_PARTITION_BIG=<80GB"
    echo "     partition>, or the runner will be SIGKILLed at load and every row"
    echo "     for this model will come out as silent zeros."
  fi

  JID=$(sbatch --parsable --array="0-$((N-1))%${THROTTLE}" \
      --job-name="pllm-${SUITE_NAME}-${MODEL_SAFE}" \
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
      --job-name="pllm-${SUITE_NAME}-symbolic" \
      -p "$CPU_PARTITION" --gres=none --mem="${CPU_MEM:-16G}" \
      --cpus-per-task=4 --time="${CPU_TIME:-08:00:00}" \
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
    --job-name="pllm-${SUITE_NAME}-compare" \
    -p "$COMPARE_PARTITION" --gres=none \
    scripts/slurm_compare.sh "$OUTDIR")
echo "  -> compare job: $CMP_JID (waits for all arrays; writes comparison.md AND summary.md)"

cat <<EOF

When done, see:
  $OUTDIR/summary.md              per-model / per-tier / per-difficulty breakdown
  $OUTDIR/comparison.md           per-method headline table
  $OUTDIR/comparison.xlsx, comparison_summary.csv, comparison_all_runs.csv
Per-cell outputs are $OUTDIR/<method>__<model>.{csv,xlsx} (one file per method x model).

Sanity checks after the run:
  # any task killed rather than finished?
  sacct -j <arrayjobid> -o JobID%20,JobName%18,State,ExitCode,MaxRSS,ReqMem,Elapsed
  # did any model fail its preflight?
  grep -l "SMOKE FAIL" slurm_logs/pllm-*_*.err
  # did an ollama runner get OOM-killed?
  grep -l "signal: killed" slurm_logs/ollama_*.log
EOF
