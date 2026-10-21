#!/usr/bin/env bash
# Level 2 native navigation. MemBridge is not used.
#
# Poses are stored in the episode-local frame.
# ETPNav, CMA, and Uni-LaViRA clear memory at each episode.
# JanusVLN keeps its visual cache: EP0 writes it, and EP1 and EP2 read it.
# Janus uses GPU_PAIRS. The other models use GPU_IDS.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/conda_run.sh"
LLHN="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MEMX="${LLHN}"

if [[ $# -ge 1 && "${1}" != -* ]]; then
  MODEL="$1"
  shift
fi
MODEL="${MODEL:-etpnav}"

die() { echo "[l2-native] ERROR: $*" >&2; exit 1; }
log() { echo "[l2-native] $*"; }

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
L2_HABITAT="${LLHN}/datasets/level2/habitat"
RUNNER="${MEMX}/baselines/run.py --mode basic --level 2 --pipeline native"
JANUS_ROOT="${JANUS_ROOT:-${LLHN}/baselines/models/JanusVLN}"
JANUS_MODEL="${JANUS_MODEL:-${MODELS_DIR:-${LLHN}/models}/JanusVLN_Extra}"
CMA_ROOT="${CMA_ROOT:-${LLHN}/baselines/models/CMA}"
UNI_ROOT="${UNI_ROOT:-${UNILAVIRA_ROOT:-${LLHN}/baselines/models/Uni-LaViRA}}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
MAX_STEPS="${MAX_STEPS:-300}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${MEMX}/experiments_results/${RESULT_NAME}/level2/runs/${RUN_TAG}/native}"
VLM_ENV_FILE="${VLM_ENV_FILE:-${UNILAVIRA_ROOT-}/.env}"

log "model=${MODEL_LABEL} run_root=${RUN_ROOT}"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ "${STAGE}" == "all" || "${STAGE}" == "ep0" || "${STAGE}" == "ep1" || "${STAGE}" == "ep2" ]] \
  || die "STAGE must be all, ep0, ep1, or ep2"
[[ -f "${MEMX}/baselines/run.py" ]] || die "missing baselines/run.py"
[[ -f "${L2_HABITAT}/level2_ep0.json.gz" ]] || die "missing Level 2 habitat split"

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
  esac
}

run_etpnav_role() {
  local role="$1"
  local output="${RUN_ROOT}/${role}"
  local stage
  stage="$(stage_name "${role}")"
  log "ETPNav ${role}: empty GraphMap, output=${output}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips ETPNav ${role}"
    return
  fi
  mkdir -p "${output}/logs"
  local -a pids=()
  local shard=0 gpu fail=0
  for gpu in "${WORKERS[@]}"; do
    local logf="${output}/logs/shard$(printf '%02d' "${shard}").log"
    PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MAGNUM_LOG=quiet GLOG_minloglevel=2 \
    memx_spawn "${gpu}" "${ETP_ROOT}" "${logf}" \
      bash -lc "python run.py --exp_name l2_${role}_s${shard} --run-type eval --exp-config run_r2r/iter_level2.yaml SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth EVAL.SPLIT ${role} EVAL.EPISODE_COUNT ${EPISODE_COUNT:--1} TASK_CONFIG.DATASET.DATA_PATH ${LLHN}/datasets/level2/habitat/level2_{split}.json.gz TASK_CONFIG.TASK.NDTW.GT_PATH ${LLHN}/datasets/level2/habitat/level2_{split}_gt.json.gz TASK_CONFIG.TASK.SDTW.GT_PATH ${LLHN}/datasets/level2/habitat/level2_{split}_gt.json.gz EVAL.DUMP_MEMORY True EVAL.LOAD_GRAPHMAP False EVAL.MEMORY_ROOT ${output} EVAL.MEMORY_STAGE ${stage} EVAL.SHARD_ID ${shard} EVAL.NUM_SHARDS ${#WORKERS[@]} IL.back_algo control IL.is_requeue False && python ${RUNNER} --convert-root ${output}"
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
  local dataset="${L2_HABITAT}/level2_${role}.json.gz"
  [[ -f "${dataset}" ]] || die "missing ${dataset}"
  log "${MODEL_LABEL} ${role}: output=${output}"
  mkdir -p "${output}/logs"
  launch_one() {
    local worker="$1" shard="$2" dry="$3"
    local flags="--nav-model ${MODEL_KEY} --role ${role} --dataset ${dataset} --output-root ${output}"
    flags+=" --cache-root ${RUN_ROOT} --max-steps ${MAX_STEPS} --num-shards ${#WORKERS[@]} --shard-index ${shard}"
    [[ "${dry}" == 1 ]] && flags+=" --dry-run"
    [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      [[ "${worker}" =~ ^[0-9]+,[0-9]+$ ]] || die "GPU pair must be FIRST,SECOND: ${worker}"
      JANUSVLN_VGGT_DEVICE=cuda:0 JANUSVLN_LANGUAGE_DEVICE=cuda:1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="${JANUS_ROOT}/src:${MEMX}:${MEMX}/membridge/localization:${MEMX}/baselines/adapt${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_run "${worker}" "${JANUS_ROOT}" \
        python ${RUNNER} \
          --etpnav-root "${ETP_ROOT}" \
          --janus-root "${JANUS_ROOT}" \
          --janus-model "${JANUS_MODEL}" \
          --janus-config "${JANUS_ROOT}/config/vln_level2_protocol.yaml" \
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
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
          ${flags}
    else
      flags+=" --unilavira-root ${UNI_ROOT}"
      MEMX_ENV_FILE="${VLM_ENV_FILE}" \
      PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}:${UNI_ROOT}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_run "${worker}" "${LLHN}" \
        python ${RUNNER} \
          --exp-config "${ETP_ROOT}/run_r2r/iter_level2.yaml" \
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
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
    run_role ep0
    run_role ep1
    run_role ep2
    ;;
  ep0) run_role ep0 ;;
  ep1) run_role ep1 ;;
  ep2) run_role ep2 ;;
esac
