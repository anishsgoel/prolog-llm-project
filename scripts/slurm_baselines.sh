#!/usr/bin/env bash
# SLURM array worker for the PLLM baseline suite.
# ONE array task = ONE (method|ablation) x ONE model.  Do not run this directly —
# submit it via scripts/submit_baselines.sh, which sets --array, the resources,
# and exports the knobs (TASKS_FILE, CONFIGS, ...).
#
# Each task that needs the LLM starts its OWN ollama server (Ollama is assumed
# installed but not running) on a task-unique port, loads exactly ONE model,
# smoke-tests it, then runs batch_run.py and writes one CSV + one XLSX.
# batch_run.py is unchanged; we only invoke it via `python3 -m experiment.batch_run`.
#
# WHY one model per task (changed 2026-08): with several models in one task,
# batch_run loops over them and ollama evicts + reloads the whole model on every
# switch ("model requires more gpu memory than is currently available, evicting
# a model to make space").  Worse, a model that fails to load poisons the rest of
# the task's rows.  One model per task keeps failures isolated and resubmittable.
#
#SBATCH --job-name=pllm-baseline
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
# NOTE: partition, --gres, --mem, --cpus-per-task and --time are intentionally
# NOT set here. They differ per model (gpt-oss:120b needs an order of magnitude
# more RAM/VRAM than llama3:8b), so submit_baselines.sh passes them per array on
# the sbatch command line. Running this script directly would inherit the
# cluster defaults — which is exactly the bug that SIGKILLed the 120b runner.

set -ef -o pipefail   # -f (noglob) keeps a literal '*' for --configs

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

# --- environment: Python module + venv (shared), then ollama module ----------
source scripts/env_setup.sh
ml "${PLLM_OLLAMA_MODULE:-ollama/0.11.10-GCCcore-14.3.0-CUDA-12.9.1}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/pllm/ollama/bin/ollama}"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
python3 --version

CONFIGS="${CONFIGS:-*}"
EXCLUDE_CONFIGS="${EXCLUDE_CONFIGS:-}"   # patterns dropped after --configs expands
RUNS="${RUNS:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-7200}"
OUTDIR="${OUTDIR:-results/baselines}"
TASKS_FILE="${TASKS_FILE:-$OUTDIR/tasks_gpu.txt}"   # submit_baselines.sh passes this per-array
PULL_MODELS="${PULL_MODELS:-0}"   # 0 = assume models already present
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# --- what did SLURM actually give us? ---------------------------------------
# Printed FIRST so every future log answers "was this an OOM?" without guessing.
echo "[env] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-unset}" \
     "mem_per_node=${SLURM_MEM_PER_NODE:-unset}MB mem_per_cpu=${SLURM_MEM_PER_CPU:-unset}MB" \
     "gres=${SLURM_JOB_GRES:-unset} time=${SLURM_TIMELIMIT:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
  || echo "[env] no nvidia-smi (CPU-only task?)"
if [ -r /sys/fs/cgroup/memory.max ]; then
  echo "[env] cgroup memory.max = $(cat /sys/fs/cgroup/memory.max)"
elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  echo "[env] cgroup memory.limit_in_bytes = $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
fi

# --- resolve this task ------------------------------------------------------
# Task line format: "<--method|--ablation> <name> <model>"   (3 fields)
TASK_LINE="$(sed -n "$((IDX + 1))p" "$TASKS_FILE")"
if [ -z "$TASK_LINE" ]; then echo "no task at index $IDX in $TASKS_FILE"; exit 0; fi
read -r FLAG NAME MODEL <<< "$TASK_LINE"
if [ -z "$MODEL" ]; then echo "ERROR: task line '$TASK_LINE' has no model field" >&2; exit 2; fi
if [ "$FLAG" = "--ablation" ]; then LABEL="ablation_${NAME}"; else LABEL="$NAME"; fi
# ':' and '/' are legal on Linux but break Windows/Dropbox sync of the results.
MODEL_SAFE="${MODEL//[^A-Za-z0-9._-]/-}"
echo "[task $IDX] '$TASK_LINE' -> label=$LABEL model=$MODEL"

# non-LLM methods don't need ollama at all
NEED_LLM=1
case "$NAME" in
  no_abduction|minimal_alp|random_edge) NEED_LLM=0 ;;
esac

# --- ollama lifecycle -------------------------------------------------------
OLLAMA_PID=""
cleanup() {
  [ -n "$OLLAMA_PID" ] || return 0
  # `ollama serve` forks per-model `ollama runner` children; killing only the
  # parent orphans them and they keep holding VRAM. setsid (below) puts the
  # server in its own process group, so we can signal the whole group.
  kill -TERM "-$OLLAMA_PID" 2>/dev/null || kill -TERM "$OLLAMA_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do kill -0 "$OLLAMA_PID" 2>/dev/null || return 0; sleep 1; done
  kill -KILL "-$OLLAMA_PID" 2>/dev/null || kill -KILL "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "$NEED_LLM" -eq 1 ]; then
  # A port nobody else is on. `ollama serve` binds $OLLAMA_HOST, and the ollama
  # python client used by batch_run reads the same variable, so no code change
  # is needed.
  #
  # WHY this is probed rather than computed as 11434+IDX: the array task index
  # restarts at 0 for every array, so two suites submitted concurrently
  # (submit_baselines.sh and submit_london.sh) produce the same index on
  # different arrays. If two such tasks land on the same node they pick the same
  # port, the second `ollama serve` fails to bind, and — because the readiness
  # check below only proves that SOMETHING answers — the second task would
  # happily drive the FIRST task's server. That server has a different model
  # loaded and OLLAMA_MAX_LOADED_MODELS=1, so the two tasks would evict and
  # reload each other's weights on every call, and whichever task finished first
  # would kill the server out from under the other, turning the rest of its rows
  # into silent zeros. Exactly the failure mode the preflight below exists to
  # catch, arriving by a different route.
  PORT="$(python3 - "$((11434 + IDX))" <<'PY'
import socket, sys
start = int(sys.argv[1])
for port in range(start, start + 512):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    break
else:
    sys.exit(1)
PY
)" || { echo "ERROR: no free port in [$((11434 + IDX)), $((11434 + IDX + 512)))" >&2; exit 1; }
  export OLLAMA_HOST="127.0.0.1:${PORT}"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
  # One model, one request at a time, and never unload mid-sweep: keeps a single
  # 60 GiB load instead of repeated evict/reload cycles.
  export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
  export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"

  OLLAMA_LOG="slurm_logs/ollama_${SLURM_JOB_ID:-x}_${IDX}_${MODEL_SAFE}.log"
  echo "starting ollama ($OLLAMA_BIN) on $OLLAMA_HOST -> $OLLAMA_LOG"
  setsid "$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &
  OLLAMA_PID=$!

  # Require OUR server to be the one answering: `ollama list` succeeding proves
  # only that something is listening on that port, which would mask a port
  # steal. Checking our own PID first turns that into a loud failure.
  READY=0
  for _ in $(seq 1 60); do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "ERROR: our 'ollama serve' (pid $OLLAMA_PID) exited during startup." >&2
      echo "       See $OLLAMA_LOG — a bind failure here means another task on" >&2
      echo "       $(hostname) already holds $OLLAMA_HOST." >&2
      exit 1
    fi
    if "$OLLAMA_BIN" list >/dev/null 2>&1; then READY=1; break; fi
    sleep 2
  done
  [ "$READY" -eq 1 ] || { echo "ERROR: ollama server not ready on $OLLAMA_HOST" >&2; exit 1; }
  echo "ollama ready on $OLLAMA_HOST (pid $OLLAMA_PID)"

  if [ "$PULL_MODELS" -eq 1 ]; then
    echo "pulling model $MODEL"; "$OLLAMA_BIN" pull "$MODEL"
  fi

  # --- PREFLIGHT: the server being up does NOT mean the model can load. ------
  # The 20260804 run silently produced 68 all-zero 120b rows because the runner
  # was SIGKILLed at load time, every /api/chat returned HTTP 500, and
  # LLMInterface.generate() swallows the exception and returns "".  We therefore
  # exercise the SAME client path here and FAIL THE TASK if it comes back empty,
  # so a broken cell shows up as a failed SLURM task instead of as data.
  echo "smoke-testing $MODEL through the same ollama client the harness uses..."
  if ! python3 - "$MODEL" <<'PY'
import sys
import ollama
model = sys.argv[1]
try:
    resp = ollama.Client().chat(
        model=model,
        messages=[{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
    )
except Exception as exc:                      # HTTP 500, connection refused, ...
    print(f"SMOKE FAIL: chat() raised {exc.__class__.__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
msg = resp["message"]
text = ((msg.get("content") or "") or (msg.get("thinking") or "")).strip()
if not text:
    print(f"SMOKE FAIL: empty content AND thinking. Raw: {msg!r}", file=sys.stderr)
    sys.exit(1)
print(f"SMOKE OK: {text[:120]!r}")
PY
  then
    echo "ERROR: model '$MODEL' cannot serve a single completion on this node." >&2
    echo "       See $OLLAMA_LOG — check for 'signal: killed' (memory cgroup)" >&2
    echo "       or 'evicting a model' / insufficient VRAM." >&2
    exit 3
  fi
fi

mkdir -p "$OUTDIR"
# Model is part of the filename: one file per (method, model) cell, so a
# resubmit of one cell cannot truncate another's results. aggregate_results.py
# globs <outdir>/*.csv, so extra files are picked up with no change needed.
OUT_CSV="$OUTDIR/${LABEL}__${MODEL_SAFE}.csv"
OUT_XLSX="$OUTDIR/${LABEL}__${MODEL_SAFE}.xlsx"
echo "running: batch_run $FLAG $NAME --models $MODEL --configs $CONFIGS" \
     "${EXCLUDE_CONFIGS:+--exclude-configs $EXCLUDE_CONFIGS} -> $OUT_CSV"

# $CONFIGS / $EXCLUDE_CONFIGS are intentionally unquoted so multi-token values
# split into separate argv items; noglob (set -f) keeps '*' literal.
python3 -m experiment.batch_run "$FLAG" "$NAME" \
  --models "$MODEL" \
  --configs $CONFIGS \
  ${EXCLUDE_CONFIGS:+--exclude-configs $EXCLUDE_CONFIGS} \
  --runs "$RUNS" \
  --eval-timeout "$EVAL_TIMEOUT" \
  --no-mlflow \
  --output "$OUT_CSV" \
  --xlsx "$OUT_XLSX"

echo "done: wrote $OUT_CSV and $OUT_XLSX"
