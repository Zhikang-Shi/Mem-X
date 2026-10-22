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
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/conda_run.sh"
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
ETP_ROOT="${ETPNAV_ROOT:-${LLHN}/baselines/models/ETPNav}"
L3_HABITAT="${LLHN}/datasets/level3/habitat"

GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${LLHN}/experiments_results/${RESULT_NAME}/level3/runs/${RUN_TAG}}"
if [[ "${MODEL_KEY}" != "janus" ]]; then
  ETP_ROOT="$(memx_prepare_etpnav "${LLHN}" "${ETP_ROOT}" "${RUN_ROOT}")"
fi
ETP_DATA="$(readlink -f -- "${ETP_ROOT}/data")"
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

run_etpnav_ep0() {
  local output="${RUN_ROOT}/ep0"
  local m0="${RUN_ROOT}/ep0_m0"
  log "EP0: ETPNav navigation, then local M0, output=${output}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips ETPNav EP0"
    return
  fi
  mkdir -p "${output}/logs" "${m0}"
  local -a pids=()
  local shard=0 gpu fail=0
  read -r -a WORKERS <<< "${GPU_IDS}"
  for gpu in "${WORKERS[@]}"; do
    local logf="${output}/logs/shard$(printf '%02d' "${shard}").log"
    PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MAGNUM_LOG=quiet GLOG_minloglevel=2 \
    memx_spawn "${gpu}" "${ETP_ROOT}" "${logf}" \
      python run.py \
        --exp_name "l3_ep0_s${shard}" --run-type eval \
        --exp-config run_r2r/iter_level3.yaml \
        SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
        EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
        EVAL.SPLIT ep0 EVAL.EPISODE_COUNT "${EPISODE_COUNT:--1}" \
        EVAL.DUMP_MEMORY True EVAL.LOAD_GRAPHMAP False \
        EVAL.MEMORY_ROOT "${output}" EVAL.MEMORY_STAGE M0 \
        EVAL.SHARD_ID "${shard}" EVAL.NUM_SHARDS "${#WORKERS[@]}" \
        IL.back_algo control IL.is_requeue False
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "ETPNav EP0 failed"
  memx_spawn 0 "${LLHN}" "${m0}/build.log" \
    python "${RUNNER}" --mode build --level 3 --pipeline membridge \
      --ep0-rollouts "${output}" \
      --dataset "${L3_HABITAT}/level3_ep0.json.gz" \
      --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
      --output-root "${m0}" \
      --memory-module-root "${LLHN}/membridge" \
      --query-vocabulary "${LLHN}/membridge/semantic_scene_graph/config/level1_runtime_query_vocabulary_75.json" \
      --coordinate-frame ep0_local \
      --expected-count "${EXPECTED_COUNT:-157}" \
      --skip-existing --continue-on-error >/dev/null
  wait "$!" || die "M0 build failed"
  log "M0 ${m0}"
}

run_role() {
  local role="$1"
  local output="${RUN_ROOT}/${role}"
  local dataset="${L3_HABITAT}/level3_${role}.json.gz"
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
    read -r -a WORKERS <<< "${GPU_PAIRS}"
  else
    read -r -a WORKERS <<< "${GPU_IDS}"
  fi
  for worker in "${WORKERS[@]}"; do
    local py_path="${ETP_ROOT}:${LLHN}:${LLHN}/baselines/adapt:${LLHN}/membridge/localization:${LLHN}/membridge/runners"
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      py_path="${JANUS_ROOT}/src:${py_path}"
    fi
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      export JANUSVLN_VGGT_DEVICE=cuda:0 JANUSVLN_LANGUAGE_DEVICE=cuda:1
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    fi
    local -a extra=()
    [[ -n "${SESSION_ID:-}" ]] && extra+=(--session-id "${SESSION_ID}")
    if [[ "${role}" != "ep0" && -n "${LIMIT:-}" ]]; then
      extra+=(--limit "${LIMIT}")
    fi
    [[ -n "${M0_ROOT:-}" ]] && extra+=(--m0-root "${M0_ROOT}")
    if [[ "${role}" == "ep2" && -n "${M1_ROOT:-}" ]]; then
      extra+=(--m1-root "${M1_ROOT}")
    fi
    if [[ "${role}" == "ep2" ]]; then
      extra+=(--update-memory "${EP2_UPDATE_MEMORY}")
    fi
    MEMX_ENV_FILE="${VLM_ENV_FILE:-}" \
    INHERIT_SG_SAM3_CKPT="${MODELS_DIR:-}/facebook/sam3/sam3.pt" \
    INHERIT_SG_SAM3_SOURCE_ROOT="${MODELS_DIR:-}/source/sam3" \
    SAM3_SOURCE_ROOT="${MODELS_DIR:-}/source/sam3" \
    INHERIT_SG_DINOV3_MODEL="${MODELS_DIR:-}/facebook/dinov3" \
    INHERIT_SG_DINOV3_REPO="${MODELS_DIR:-}/source/dinov3" \
    INHERIT_SG_TEXT_ENCODER="${MODELS_DIR:-}/all-MiniLM-L6-v2" \
    PYTHONPATH="${py_path}${PYTHONPATH:+:${PYTHONPATH}}" \
    memx_spawn "${worker}" "${LLHN}" "${output}/logs/shard$(printf '%02d' "${shard}").log" \
      python "${RUNNER}" \
        --mode auto --level 3 --pipeline membridge --role "${role}" --nav-model "${MODEL_KEY}" \
        --dataset "${dataset}" --output-root "${output}" \
        --etpnav-root "${ETP_ROOT}" \
        --exp-config "${ETP_ROOT}/run_r2r/iter_level3.yaml" \
        --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
        --ckpt "${ETP_DATA}/logs/checkpoints/release_r2r/ckpt.iter12000.pth" \
        --waypoint-ckpt "${ETP_DATA}/wp_pred/check_cwp_bestdist_hfov90" \
        --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
        --num-shards "${#WORKERS[@]}" --shard-index "${shard}" \
        "${extra[@]}"
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "${MODEL_LABEL} ${role} failed"
}

run_ep0() {
  if [[ "${MODEL_KEY}" == "etpnav" ]]; then
    run_etpnav_ep0
  else
    run_role ep0
  fi
}
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
