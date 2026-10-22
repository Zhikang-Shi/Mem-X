#!/usr/bin/env bash
# Run a launcher command in the Mem-X conda environment.
# CONDA_ENV selects the environment name. The default is memx.

# Resolve a runtime built from upstream ETPNav plus the tracked Mem-X adapter.
memx_prepare_etpnav() {
  local repo="$1" source_root="$2" run_root="$3"
  python3 "${repo}/baselines/adapt/etpnav_runtime.py" \
    --source "${source_root}" --repo "${repo}" \
    --runtime-root "${run_root}/.runtime/etpnav"
}

memx_activate() {
  local env_name="${CONDA_ENV:-memx}"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" && -n "${CONDA_PREFIX:-}" ]]; then
    return 0
  fi
  local base="${CONDA_BASE:-}"
  if [[ -z "${base}" ]]; then
    base="$(conda info --base)"
  fi
  # shellcheck disable=SC1091
  source "${base}/etc/profile.d/conda.sh"
  conda activate "${env_name}"
}

# memx_run GPU WORKDIR -- command...
# Activates the conda environment and replaces the current process.
# Callers that need a log and a background job should redirect and append &.
memx_run() {
  local gpu="$1"
  local workdir="$2"
  shift 2
  local env_name="${CONDA_ENV:-memx}"
  local base="${CONDA_BASE:-}"
  if [[ -z "${base}" ]]; then
    base="$(conda info --base)"
  fi
  local saved_pythonpath="${PYTHONPATH:-}"
  # shellcheck disable=SC1091
  source "${base}/etc/profile.d/conda.sh"
  conda activate "${env_name}"
  if [[ -n "${saved_pythonpath}" ]]; then
    export PYTHONPATH="${saved_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  if [[ -n "${MEMX_ENV_FILE:-}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${MEMX_ENV_FILE}"
    set +a
  fi
  if [[ -n "${workdir}" ]]; then
    cd "${workdir}"
  fi
  exec "$@"
}

# memx_spawn GPU WORKDIR LOGFILE -- command...
# The command runs after `conda activate` and inherits the caller's environment.
memx_spawn() {
  local gpu="$1"
  local workdir="$2"
  local logfile="$3"
  shift 3
  local env_name="${CONDA_ENV:-memx}"
  local base="${CONDA_BASE:-}"
  if [[ -z "${base}" ]]; then
    base="$(conda info --base)"
  fi
  (
    local saved_pythonpath="${PYTHONPATH:-}"
    # shellcheck disable=SC1091
    source "${base}/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    if [[ -n "${saved_pythonpath}" ]]; then
      export PYTHONPATH="${saved_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    if [[ -n "${MEMX_ENV_FILE:-}" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "${MEMX_ENV_FILE}"
      set +a
    fi
    if [[ -n "${workdir}" ]]; then
      cd "${workdir}"
    fi
    exec "$@"
  ) >"${logfile}" 2>&1 &
  echo $!
}
