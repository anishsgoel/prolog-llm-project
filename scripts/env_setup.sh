# Cluster Python environment — SOURCED by the SLURM jobs (not executed).
# Edit here to match your cluster; the array workers and the compare job both
# source this file, so the module/venv setup lives in exactly one place.
#
# Mirrors the user's working setup:
#   ml Python/3.13.5-GCCcore-14.3.0
#   source ~/venvs_python/pllm/bin/activate
# Each line is overridable via the environment for portability.

ml "${PLLM_PYTHON_MODULE:-Python/3.13.5-GCCcore-14.3.0}"

# shellcheck disable=SC1090
source "${PLLM_VENV:-$HOME/venvs_python/pllm}/bin/activate"