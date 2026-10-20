#!/usr/bin/env bash
# Level 3 +MemBridge. MODEL selects the policy. The schedule is the same for all four models.
#
# EP0 navigates and then builds episode-local memory. The step budget is 300.
# EP1 localizes while navigating and writes the aligned memory. The step budget is 300.
# EP2 is the memory task. After localization it explores for at most 100 steps, then runs the query. The step budget is 400.
# If EP1 did not lock, EP2 uses the policy from the first step and does not explore.
# If EP1 locked, EP2 explores for at most 100 steps. If it is still unlocked, the policy continues and localization continues.
# After a lock, instruction parsing and top-3 selection call the VLM live.
# A parse other than [1, 1, 2], or a candidate list whose length is not 3, is invalid.
# The two type-1 targets are walked on the scene graph. The type-2 suffix is given to the policy.
# A rejected top-3 selection does not skip exploration.
# EP2_UPDATE_MEMORY=full is the default. It updates the scene graph, the navigation graph, and the trajectory.
# Janus uses GPU_PAIRS. The other models use GPU_IDS.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLHN="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -ge 1 && "${1}" != -* ]]; then
  MODEL="$1"
  shift
fi
MODEL="${MODEL:-etpnav}"

die() { echo "[l3-membridge] ERROR: $*" >&2; exit 1; }
log() { echo "[l3-membridge] $*"; }

case "${MODEL}" in
  etpnav|ETPNav)
    MODEL_KEY="etpnav"
    MODEL_LABEL="ETPNav"
    ADAPT_NAME="ETPNav_adapt"
    RESULT_NAME="ETPNav"
    ;;
  janus|janusvln|JanusVLN)
    MODEL_KEY="janus"
    MODEL_LABEL="JanusVLN"
    ADAPT_NAME="JanusVLN_adapt"
    RESULT_NAME="JanusVLN"
    ;;
  cma|CMA)
    MODEL_KEY="cma"
    MODEL_LABEL="CMA"
    ADAPT_NAME="CMA_adapt"
    RESULT_NAME="CMA"
    ;;
  unilavira|uni-lavira|Uni-LaViRA|UniLaViRA)
    MODEL_KEY="unilavira"
    MODEL_LABEL="Uni-LaViRA"
    ADAPT_NAME="UniLaViRA_adapt"
    RESULT_NAME="UniLaViRA"
    ;;
  *)
    die "MODEL must be etpnav, janus, cma, or unilavira: ${MODEL}"
    ;;
esac

RUNNER="${LLHN}/baselines/run.py"

STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${LLHN}/experiments_results/${RESULT_NAME}/level3/runs/${RUN_TAG}}"
EP2_UPDATE_MEMORY="${EP2_UPDATE_MEMORY:-full}"
export MODEL MODEL_KEY RUN_TAG RUN_ROOT EP2_UPDATE_MEMORY

log "model=${MODEL_LABEL} run_root=${RUN_ROOT}"
[[ -f "${LLHN}/datasets/level3/level3.json" && -f "${LLHN}/datasets/level3/level3_gt.json" ]] \
  || die "missing ${LLHN}/datasets/level3/level3.json"
python3 "${LLHN}/scripts/materialize_habitat_splits.py" --root "${LLHN}" --level 3
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ -f "${RUNNER}" ]] || die "missing ${RUNNER}"

latest_output() {
  local link="$1"
  [[ -L "${link}" || -d "${link}" ]] || die "missing output link: ${link}"
  readlink -f -- "${link}"
}

run_role() {
  local role="$1"
  local output="${RUN_ROOT}/${role}"
  local dataset="/datasets/level3/habitat/level3_${role}.json.gz"
  mkdir -p "${output}/logs"
  log "${MODEL_LABEL} ${role}: one runner, level differences come from baselines/runners/stage.py"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips ${MODEL_LABEL} ${role}"
    return
  fi
  case "${MODEL_KEY}" in
    janus) [[ -n "${JANUS_ROOT:-}" && -d "${JANUS_ROOT}" ]] || die "Set JANUS_ROOT to a JanusVLN checkout" ;;
    cma) [[ -n "${CMA_ROOT:-}" && -d "${CMA_ROOT}" ]] || die "Set CMA_ROOT to a CMA checkout" ;;
    unilavira) [[ -n "${UNI_ROOT:-}" && -d "${UNI_ROOT}" ]] || die "Set UNI_ROOT or UNILAVIRA_ROOT" ;;
  esac
  local -a pids=()
  local shard=0 worker fail=0
  if [[ "${MODEL_KEY}" == "janus" ]]; then
    read -r -a WORKERS <<< "${GPU_PAIRS:-0,1}"
  else
    read -r -a WORKERS <<< "${GPU_IDS:-0}"
  fi
  for worker in "${WORKERS[@]}"; do
    local py="/opt/conda/envs/lavira/bin/python"
    local extra=""
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      py="/janus/envs/streamvln/bin/python"
      extra="--volume ${JANUS_ENV:?Set JANUS_ENV to the streamvln environment}:/janus/envs/streamvln:ro"
    fi
    docker run --rm --gpus "\"device=${worker}\"" --network host --shm-size 16g \
      --volume "${LLHN}:${LLHN}:ro" \
      --volume "${ETPNAV_ROOT:?Set ETPNAV_ROOT}:${ETPNAV_ROOT}:ro" \
      ${extra} \
      --volume "${LLHN}/datasets:/datasets:ro" \
      --volume "${output}:/workspace/output" \
      ${MEMX_IMAGE:-mem-x:latest} "${py}" "${RUNNER}" \
        --mode auto --level 3 --pipeline membridge --role "${role}" --nav-model "${MODEL_KEY}" \
        --dataset "${dataset}" --output-root /workspace/output \
        --num-shards ${#WORKERS[@]} --shard-index ${shard} \
      >"${output}/logs/shard$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "${MODEL_LABEL} ${role} failed"
}

run_ep0() { run_role ep0; }
run_ep1() { run_role ep1; }
run_ep2() { run_role ep2; }

case "${STAGE}" in
  all)
    run_ep0
    if [[ "${DRY_RUN}" == 1 ]]; then
      log "dry run stops after EP0; EP1 and EP2 need the M0 link from a real EP0"
      exit 0
    fi
    run_ep1
    run_ep2
    ;;
  ep0) run_ep0 ;;
  ep1) run_ep1 ;;
  ep2) run_ep2 ;;
  *) die "STAGE must be all, ep0, ep1, or ep2: ${STAGE}" ;;
esac
