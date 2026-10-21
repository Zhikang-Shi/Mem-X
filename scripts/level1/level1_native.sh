#!/usr/bin/env bash
# Level 1 native navigation for ETPNav, CMA, Uni-LaViRA, and JanusVLN.
#
# Trajectories stay in the Habitat world frame. The step budget is 300.
# ETPNav and CMA turn 15 degrees. JanusVLN and Uni-LaViRA turn 30 degrees.
# ETPNav, CMA, and Uni-LaViRA clear memory at each episode.
# JanusVLN keeps its visual cache for the session, from EP0 through EP4.
# Janus uses GPU_PAIRS. The other models use GPU_IDS.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLHN="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/conda_run.sh"
JANUS_MODEL="${JANUS_MODEL:-${MODELS_DIR:-${LLHN}/models}/JanusVLN_Extra}"
MEMX="${LLHN}"

if [[ $# -ge 1 && "${1}" != -* ]]; then
  MODEL="$1"
  shift
fi
MODEL="${MODEL:-etpnav}"

die() { echo "[l1-native] ERROR: $*" >&2; exit 1; }
log() { echo "[l1-native] $*"; }

case "${MODEL}" in
  etpnav|ETPNav)
    MODEL_KEY="etpnav"; MODEL_LABEL="ETPNav"; RESULT_NAME="ETPNav" ;;
  janus|janusvln|JanusVLN)
    MODEL_KEY="janus"; MODEL_LABEL="JanusVLN"; RESULT_NAME="JanusVLN" ;;
  cma|CMA)
    MODEL_KEY="cma"; MODEL_LABEL="CMA"; RESULT_NAME="CMA" ;;
  unilavira|uni-lavira|Uni-LaViRA|UniLaViRA)
    MODEL_KEY="unilavira"; MODEL_LABEL="Uni-LaViRA"; RESULT_NAME="UniLaViRA" ;;
  *)
    die "MODEL must be etpnav, janus, cma, or unilavira: ${MODEL}" ;;
esac

ETP_ROOT="${ETPNAV_ROOT:-${LLHN}/baselines/models/ETPNav}"
ETP_DATA="$(readlink -f -- "${ETP_ROOT}/data")"
L1_DATA="${LLHN}/datasets/level1/habitat"
RUNNER="${MEMX}/baselines/run.py --mode basic --level 1 --pipeline native"
JANUS_ROOT="${JANUS_ROOT:-${LLHN}/baselines/models/JanusVLN}"
CMA_ROOT="${CMA_ROOT:-${LLHN}/baselines/models/CMA}"
UNI_ROOT="${UNI_ROOT:-${UNILAVIRA_ROOT:-${LLHN}/baselines/models/Uni-LaViRA}}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
MAX_STEPS="${MAX_STEPS:-300}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${MEMX}/experiments_results/${RESULT_NAME}/level1/runs/${RUN_TAG}/native}"
VLM_ENV_FILE="${VLM_ENV_FILE:-${UNILAVIRA_ROOT-}/.env}"
ROLES=(ep0 ep1 ep2 ep3 ep4)

log "model=${MODEL_LABEL} run_root=${RUN_ROOT} coordinate_frame=habitat_world"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ "${STAGE}" == "all" || "${STAGE}" == "ep0" || "${STAGE}" == "ep1" || "${STAGE}" == "ep2" || "${STAGE}" == "ep3" || "${STAGE}" == "ep4" ]] \
  || die "STAGE must be all, ep0, ep1, ep2, ep3, or ep4"
[[ -f "${MEMX}/baselines/run.py" ]] || die "missing baselines/run.py"
[[ -f "${L1_DATA}/level1_ep0.json.gz" && -f "${L1_DATA}/level1_ep4.json.gz" ]] \
  || die "missing Level 1 habitat split under ${L1_DATA}"
[[ -f "${VLM_ENV_FILE}" ]] || die "missing VLM env ${VLM_ENV_FILE}"

if [[ "${MODEL_KEY}" == "janus" ]]; then
  read -r -a WORKERS <<< "${GPU_PAIRS}"
else
  read -r -a WORKERS <<< "${GPU_IDS}"
fi
((${#WORKERS[@]} > 0)) || die "no GPUs"

stage_name() {
  case "$1" in
    ep0) echo M0 ;;
    ep1) echo M1 ;;
    ep2) echo M2 ;;
    ep3) echo M3 ;;
    ep4) echo M4 ;;
  esac
}

run_etpnav_role() {
  local role="$1"
  local output="${RUN_ROOT}/${role}"
  local stage
  stage="$(stage_name "${role}")"
  log "ETPNav ${role}: empty GraphMap, habitat world, output=${output}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips ETPNav ${role}"
    return
  fi
  mkdir -p "${output}/logs"
  local -a pids=()
  local shard=0 gpu fail=0
  for gpu in "${WORKERS[@]}"; do
    PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MAGNUM_LOG=quiet GLOG_minloglevel=2 \
    memx_spawn "${gpu}" "${ETP_ROOT}" "${output}/logs/shard$(printf '%02d' "${shard}").log" \
      python run.py \
        --exp_name "l1_${role}_s${shard}" --run-type eval \
        --exp-config run_r2r/iter_level1.yaml \
        SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
        EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
        EVAL.SPLIT "${role}" EVAL.EPISODE_COUNT "${EPISODE_COUNT:--1}" \
        EVAL.DUMP_MEMORY True EVAL.LOAD_GRAPHMAP False \
        EVAL.MEMORY_ROOT "${output}" EVAL.MEMORY_STAGE "${stage}" \
        EVAL.SHARD_ID "${shard}" EVAL.NUM_SHARDS "${#WORKERS[@]}" \
        IL.back_algo control IL.is_requeue False
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "ETPNav ${role} failed"
  log "finished ${output}"
}

run_other_role() {
  local role="$1"
  local output="${RUN_ROOT}/${role}"
  local dataset="${LLHN}/datasets/level1/habitat/level1_${role}.json.gz"
  log "${MODEL_LABEL} ${role}: habitat world, output=${output}"
  mkdir -p "${output}/logs"
  launch_one() {
    local worker="$1" shard="$2" dry="$3"
    local turn=15
    if [[ "${MODEL_KEY}" == "janus" || "${MODEL_KEY}" == "unilavira" ]]; then
      turn=30
    fi
    local flags="--nav-model ${MODEL_KEY} --role ${role} --dataset ${dataset} --output-root ${output}"
    flags+=" --world-poses --turn-angle ${turn} --cache-root ${RUN_ROOT}"
    flags+=" --max-steps ${MAX_STEPS} --num-shards ${#WORKERS[@]} --shard-index ${shard}"
    [[ "${dry}" == 1 ]] && flags+=" --dry-run"
    [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
    local logf="${output}/logs/shard$(printf '%02d' "${shard}").log"
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      [[ "${worker}" =~ ^[0-9]+,[0-9]+$ ]] || die "GPU pair must be FIRST,SECOND: ${worker}"
      JANUSVLN_VGGT_DEVICE=cuda:0 JANUSVLN_LANGUAGE_DEVICE=cuda:1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="${JANUS_ROOT}/src:${MEMX}:${MEMX}/membridge/localization:${MEMX}/baselines/adapt${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_run "${worker}" "${JANUS_ROOT}" \
        python ${RUNNER} \
          --etpnav-root "${ETP_ROOT}" \
          --janus-root "${JANUS_ROOT}" \
          --janus-model "${JANUS_MODEL}" \
          --janus-config "${JANUS_ROOT}/config/vln_level1_protocol.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
          ${flags}
    elif [[ "${MODEL_KEY}" == "cma" ]]; then
      flags+=" --cma-root ${CMA_ROOT} --ckpt ${CMA_ROOT}/data/checkpoints/CMA_PM_DA_Aug.pth"
      flags+=" --r2r-train ${CMA_ROOT}/data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz"
      flags+=" --exp-config ${CMA_ROOT}/experiment_level3/cma_level3.yaml"
      MEMX_ENV_FILE="${VLM_ENV_FILE}" \
      PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}:${CMA_ROOT}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_run "${worker}" "${LLHN}" \
        python ${RUNNER} \
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level1.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
          ${flags}
    else
      flags+=" --unilavira-root ${UNI_ROOT} --exp-config ${ETP_ROOT}/run_r2r/iter_level1.yaml"
      MEMX_ENV_FILE="${VLM_ENV_FILE}" \
      PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}:${UNI_ROOT}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_run "${worker}" "${LLHN}" \
        python ${RUNNER} \
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level1.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
          ${flags}
    fi
  }
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
  local shard=0 worker pid fail=0
  for worker in "${WORKERS[@]}"; do
    launch_one "${worker}" "${shard}" 0 >"${output}/logs/shard$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "${MODEL_LABEL} ${role} failed"
  log "finished ${output}"
}

run_role() {
  local role="$1"
  if [[ "${MODEL_KEY}" == "etpnav" ]]; then
    run_etpnav_role "${role}"
  else
    run_other_role "${role}"
  fi
}

case "${STAGE}" in
  all)
    for role in "${ROLES[@]}"; do
      run_role "${role}"
    done
    ;;
  ep0|ep1|ep2|ep3|ep4) run_role "${STAGE}" ;;
esac
