#!/usr/bin/env bash
# Level 2 +MemBridge. MODEL selects the policy. The schedule is the same for all four models.
#
# EP0 navigates and then builds episode-local memory. The step budget is 300.
# EP1 localizes against that memory, then updates the scene graph, navigation graph, and trajectory. The step budget is 300.
# EP2 reads both memory stages, localizes, and updates the same structures. The step budget is 300. There is no extra exploration.
# Each episode uses its own local frame. On a retrieval hit with a usable alignment, the scene graph walks in the aligned frame.
# The type-2 suffix is given to the policy. A miss or a failed alignment gives the full instruction to the policy.
# Janus uses GPU_PAIRS: FIRST,SECOND[,MEMORY]. The other models use GPU_IDS.

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

die() { echo "[l2-membridge] ERROR: $*" >&2; exit 1; }
log() { echo "[l2-membridge] $*"; }

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

LINK_DIR="${MEMX}/experiments_results/${RESULT_NAME}/level2/links"
EP0_ETP="${LLHN}/scripts/level2/launch_ep0_memory_local.sh"
RUNNER="${MEMX}/baselines/run.py --mode memory --level 2 --pipeline membridge"
EP0_RUNNER="${MEMX}/baselines/run.py --mode basic --level 2 --pipeline membridge"
M0_BUILDER="${MEMX}/baselines/run.py --mode build --level 2 --pipeline membridge"
ETP_ROOT="${ETPNAV_ROOT:-${LLHN}/baselines/models/ETPNav}"
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
RUN_ROOT="${RUN_ROOT:-${MEMX}/experiments_results/${RESULT_NAME}/level2/runs/${RUN_TAG}}"
if [[ "${MODEL_KEY}" != "janus" ]]; then
  ETP_ROOT="$(memx_prepare_etpnav "${LLHN}" "${ETP_ROOT}" "${RUN_ROOT}")"
fi
ETP_DATA="$(readlink -f -- "${ETP_ROOT}/data")"
VLM_ENV_FILE="${VLM_ENV_FILE:-${UNILAVIRA_ROOT-}/.env}"
FROZEN_M0="${FROZEN_M0:-}"
FROZEN_M1="${FROZEN_M1:-}"
FROZEN_EP0="${FROZEN_EP0:-}"
DATASET_EP0="${LLHN}/datasets/level2/habitat/level2_ep0.json.gz"
export ETPNAV_ROOT="${ETP_ROOT}"
export MODEL MODEL_KEY RUN_TAG RUN_ROOT

log "model=${MODEL_LABEL} run_root=${RUN_ROOT}"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ "${STAGE}" == "all" || "${STAGE}" == "ep0" || "${STAGE}" == "ep1" || "${STAGE}" == "ep2" ]] \
  || die "STAGE must be all, ep0, ep1, or ep2: ${STAGE}"
[[ -f "${MEMX}/baselines/run.py" && -f "${EP0_ETP}" ]] || die "missing level2 runner"
[[ -f "${DATASET_EP0}" ]] || die "missing ${DATASET_EP0}"
[[ -f "${VLM_ENV_FILE}" ]] || die "missing VLM env file ${VLM_ENV_FILE}"
mkdir -p "${LINK_DIR}"

if [[ "${MODEL_KEY}" == "janus" ]]; then
  read -r -a WORKERS <<< "${GPU_PAIRS}"
else
  read -r -a WORKERS <<< "${GPU_IDS}"
fi
((${#WORKERS[@]} > 0)) || die "no GPUs"

latest_output() {
  local link="$1"
  [[ -L "${link}" || -d "${link}" ]] || die "missing output link: ${link}"
  readlink -f -- "${link}"
}

resolve_memory() {
  local which="$1"
  local given="$2"
  local link="$3"
  local frozen="$4"
  if [[ -n "${given}" ]]; then
    readlink -f -- "${given}"
  elif [[ -L "${link}" || -d "${link}" ]]; then
    readlink -f -- "${link}"
  else
    echo "[l2-membridge] ${which} link missing, using frozen ${frozen}" >&2
    echo "${frozen}"
  fi
}

resolve_ep0_rollout() {
  local m0="$1"
  local sibling
  if [[ -n "${EP0_ROLLOUT_ROOT:-}" ]]; then
    readlink -f -- "${EP0_ROLLOUT_ROOT}"
    return
  fi
  sibling="$(readlink -f -- "${m0}/../ep0_native" 2>/dev/null || true)"
  if [[ -n "${sibling}" && -d "${sibling}/sessions" ]]; then
    echo "${sibling}"
    return
  fi
  echo "[l2-membridge] EP0 rollout link missing, using frozen ${FROZEN_EP0}" >&2
  echo "${FROZEN_EP0}"
}

build_local_m0() {
  local rollouts="$1"
  local output="$2"
  log "build local M0 from ${rollouts}"
  mkdir -p "${output}"
  ln -sfn "${output}" "${LINK_DIR}/ep0_memory_from_poses_local_latest"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips the M0 builder"
    return
  fi
  memx_spawn "${M0_GPU:-0}" "${LLHN}" "${output}/build.log" \
    python "${LLHN}/baselines/run.py" --mode build --level 2 --pipeline membridge \
    --ep0-rollouts "${rollouts}" \
    --dataset "${DATASET_EP0}" \
    --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
    --output-root "${output}" \
    --memory-module-root "${LLHN}/membridge" \
    --expected-count "${EXPECTED_COUNT:-238}" \
    --skip-existing --continue-on-error >/dev/null
  wait "$!" || die "M0 build failed"
  log "M0 ${output}"
}

run_discrete_ep0() {
  local native="${RUN_ROOT}/ep0_native"
  local output="${RUN_ROOT}/ep0_m0"
  log "EP0: ${MODEL_LABEL} navigation, then async local M0"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run: ${MODEL_LABEL} EP0 navigation, then local M0 from ${DATASET_EP0}"
    return
  fi
  case "${MODEL_KEY}" in
    janus) [[ -n "${JANUS_ROOT:-}" && -d "${JANUS_ROOT}" ]] || die "Set JANUS_ROOT to a JanusVLN checkout" ;;
    cma) [[ -n "${CMA_ROOT:-}" && -d "${CMA_ROOT}" ]] || die "Set CMA_ROOT to a CMA checkout" ;;
    unilavira) [[ -n "${UNI_ROOT:-}" && -d "${UNI_ROOT}" ]] || die "Set UNI_ROOT or UNILAVIRA_ROOT" ;;
  esac
  mkdir -p "${native}/logs"
  local -a pids=()
  local shard=0 worker fail=0
  for worker in "${WORKERS[@]}"; do
    launch_discrete_ep0_shard "${worker}" "${shard}" "${native}" \
      >"${native}/logs/shard$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "${MODEL_LABEL} EP0 navigation failed"
  build_local_m0 "${native}" "${output}"
}

launch_discrete_ep0_shard() {
  local worker="$1" shard="$2" native="$3"
  local flags="--nav-model ${MODEL_KEY} --dataset ${DATASET_EP0} --output-root ${native}"
  flags+=" --max-steps ${MAX_STEPS} --num-shards ${#WORKERS[@]} --shard-index ${shard}"
  [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
  if [[ "${MODEL_KEY}" == "janus" ]]; then
    [[ "${worker}" =~ ^[0-9]+,[0-9]+(,[0-9]+)?$ ]] || die "Janus GPUs must be FIRST,SECOND[,MEMORY]: ${worker}"
    JANUSVLN_VGGT_DEVICE=cuda:0 JANUSVLN_LANGUAGE_DEVICE=cuda:1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="${JANUS_ROOT}/src:${MEMX}:${MEMX}/baselines/adapt${PYTHONPATH:+:${PYTHONPATH}}" \
    memx_run "${worker}" "${JANUS_ROOT}" \
      python ${EP0_RUNNER} \
        --etpnav-root "${ETP_ROOT}" \
        --exp-config "${ETP_ROOT}/run_r2r/iter_level2.yaml" \
        --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
        --janus-root "${JANUS_ROOT}" \
        --janus-model "${JANUS_MODEL}" \
        --janus-config "${JANUS_ROOT}/config/vln_level2_protocol.yaml" \
        --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
        ${flags}
  elif [[ "${MODEL_KEY}" == "cma" ]]; then
    MEMX_ENV_FILE="${VLM_ENV_FILE}" \
    PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}:${CMA_ROOT}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/runners${PYTHONPATH:+:${PYTHONPATH}}" \
    memx_run "${worker}" "${LLHN}" \
      python ${EP0_RUNNER} \
        --etpnav-root "${ETP_ROOT}" \
        --exp-config "${CMA_ROOT}/vlnce_baselines/config/r2r_baselines/cma_pm_da_aug_tune.yaml" \
        --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
        --cma-root "${CMA_ROOT}" \
        --ckpt "${CMA_ROOT}/data/checkpoints/CMA_PM_DA_Aug.pth" \
        --r2r-train "${CMA_ROOT}/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz" \
        --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
        ${flags}
  else
    MEMX_ENV_FILE="${VLM_ENV_FILE}" \
    PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}:${UNI_ROOT}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/runners${PYTHONPATH:+:${PYTHONPATH}}" \
    memx_run "${worker}" "${LLHN}" \
      python ${EP0_RUNNER} \
        --etpnav-root "${ETP_ROOT}" \
        --exp-config "${ETP_ROOT}/run_r2r/iter_level2.yaml" \
        --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
        --unilavira-root "${UNI_ROOT}" \
        --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
        ${flags}
  fi
}

run_ep0() {
  if [[ "${MODEL_KEY}" == "etpnav" ]]; then
    log "EP0: ETPNav navigation, then async local M0"
    DRY_RUN="${DRY_RUN}" GPU_IDS="${GPU_IDS}" RUN_TAG="${RUN_TAG}" RUN_ROOT="${RUN_ROOT}" \
      bash "${EP0_ETP}"
    if [[ "${DRY_RUN}" != 1 ]]; then
      ln -sfn "${RUN_ROOT}/ep0_m0" "${LINK_DIR}/ep0_memory_from_poses_local_latest"
    fi
    return
  fi
  run_discrete_ep0
}

run_membridge_role() {
  local role="$1"
  local m0="$2"
  local m1="$3"
  local ep0_rollout="$4"
  local output="${RUN_ROOT}/${role}"
  local dataset="${LLHN}/datasets/level2/habitat/level2_${role}.json.gz"
  [[ -f "${dataset}" ]] || die "missing ${dataset}"
  [[ -d "${m0}/sessions" ]] || die "missing M0 ${m0}"
  [[ "${role}" == "ep1" || -d "${m1}/sessions" ]] || die "missing M1 ${m1}"
  [[ -d "${ep0_rollout}/sessions" ]] || die "missing EP0 rollout ${ep0_rollout}"
  mkdir -p "${output}/logs"
  if [[ "${role}" == "ep1" ]]; then
    ln -sfn "${output}" "${LINK_DIR}/ep1_local_loc_latest"
  fi
  log "${MODEL_LABEL} ${role}: output=${output}"
  log "M0=${m0}"
  [[ "${role}" == "ep2" ]] && log "M1=${m1}"
  log "ep0_rollout=${ep0_rollout}"
  launch_role_shard() {
    local worker="$1" shard="$2" dry="$3"
    local flags="--role ${role} --nav-model ${MODEL_KEY} --num-shards ${#WORKERS[@]} --shard-index ${shard} --max-steps ${MAX_STEPS}"
    flags+=" --m0-root ${m0} --m1-root ${m1} --ep0-rollout-root ${ep0_rollout}"
    [[ "${dry}" == 1 ]] && flags+=" --dry-run"
    [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
    if [[ "${MODEL_KEY}" == "cma" ]]; then
      flags+=" --cma-root ${CMA_ROOT} --r2r-train ${CMA_ROOT}/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
    elif [[ "${MODEL_KEY}" == "unilavira" ]]; then
      flags+=" --unilavira-root ${UNI_ROOT}"
    fi
    local exp_config="${ETP_ROOT}/run_r2r/iter_level2.yaml"
    local ckpt="${ETP_DATA}/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
    if [[ "${MODEL_KEY}" == "cma" ]]; then
      exp_config="${CMA_ROOT}/vlnce_baselines/config/r2r_baselines/cma_pm_da_aug_tune.yaml"
      ckpt="${CMA_ROOT}/data/checkpoints/CMA_PM_DA_Aug.pth"
    fi
    local lightglue_py=""
    if [[ -n "${LIGHTGLUE_ROOT:-}" ]]; then
      lightglue_py=":${LIGHTGLUE_ROOT}"
    fi
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      [[ "${worker}" =~ ^[0-9]+,[0-9]+(,[0-9]+)?$ ]] || die "Janus GPUs must be FIRST,SECOND[,MEMORY]: ${worker}"
      MEMX_ENV_FILE="${VLM_ENV_FILE}" \
      JANUSVLN_VGGT_DEVICE=cuda:0 JANUSVLN_LANGUAGE_DEVICE=cuda:1 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH="${JANUS_ROOT}/src:${MEMX}:${MEMX}/membridge/localization:${MEMX}/membridge/runners:${MEMX}/baselines/adapt${lightglue_py}${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_run "${worker}" "${JANUS_ROOT}" \
        python ${RUNNER} \
          --etpnav-root "${ETP_ROOT}" \
          --exp-config "${ETP_ROOT}/run_r2r/iter_level2.yaml" \
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
          --dataset "${dataset}" \
          --ckpt "${ETP_DATA}/logs/checkpoints/release_r2r/ckpt.iter12000.pth" \
          --waypoint-ckpt "${ETP_DATA}/wp_pred/check_cwp_bestdist_hfov90" \
          --output-root "${output}" --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
          --device cuda:0 --sim-gpu-id 0 \
          --janus-root "${JANUS_ROOT}" \
          --janus-model "${JANUS_MODEL}" \
          --janus-config "${JANUS_ROOT}/config/vln_level2_protocol.yaml" \
          ${flags}
    else
      local extra_py=""
      if [[ "${MODEL_KEY}" == "cma" ]]; then
        extra_py=":${CMA_ROOT}"
      elif [[ "${MODEL_KEY}" == "unilavira" ]]; then
        extra_py=":${UNI_ROOT}"
      fi
      MEMX_ENV_FILE="${VLM_ENV_FILE}" \
      INHERIT_SG_SAM3_CKPT="${MODELS_DIR:?Set MODELS_DIR}/facebook/sam3/sam3.pt" \
      INHERIT_SG_SAM3_SOURCE_ROOT="${MODELS_DIR}/source/sam3" \
      SAM3_SOURCE_ROOT="${MODELS_DIR}/source/sam3" \
      INHERIT_SG_DINOV3_MODEL="${MODELS_DIR}/facebook/dinov3" \
      INHERIT_SG_DINOV3_REPO="${MODELS_DIR}/source/dinov3" \
      INHERIT_SG_TEXT_ENCODER="${MODELS_DIR}/all-MiniLM-L6-v2" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${extra_py}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization:${MEMX}/membridge/runners${lightglue_py}${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_run "${worker}" "${LLHN}" \
        python ${RUNNER} \
          --etpnav-root "${ETP_ROOT}" \
          --exp-config "${exp_config}" \
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level3.yaml" \
          --dataset "${dataset}" \
          --ckpt "${ckpt}" \
          --waypoint-ckpt "${ETP_DATA}/wp_pred/check_cwp_bestdist_hfov90" \
          --output-root "${output}" --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
          --device cuda:0 --sim-gpu-id 0 \
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
    launch_role_shard "${worker}" "${shard}" 0 >"${output}/logs/shard$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "${MODEL_LABEL} ${role} failed"
  log "finished ${output}"
}

run_ep1() {
  local m0 ep0_rollout
  m0="$(resolve_memory M0 "${M0_ROOT:-}" "${LINK_DIR}/ep0_memory_from_poses_local_latest" "${FROZEN_M0}")"
  ep0_rollout="$(resolve_ep0_rollout "${m0}")"
  log "EP1: ${MODEL_LABEL} localization onto ${m0}"
  run_membridge_role ep1 "${m0}" "${m0}" "${ep0_rollout}"
}

run_ep2() {
  local m0 m1 ep0_rollout
  m0="$(resolve_memory M0 "${M0_ROOT:-}" "${LINK_DIR}/ep0_memory_from_poses_local_latest" "${FROZEN_M0}")"
  m1="$(resolve_memory M1 "${M1_ROOT:-}" "${LINK_DIR}/ep1_local_loc_latest" "${FROZEN_M1}")"
  ep0_rollout="$(resolve_ep0_rollout "${m0}")"
  log "EP2: ${MODEL_LABEL} localization onto ${m1}"
  run_membridge_role ep2 "${m0}" "${m1}" "${ep0_rollout}"
}

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
esac
