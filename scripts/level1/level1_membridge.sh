#!/usr/bin/env bash
# Level 1 +MemBridge. MODEL selects the policy. The schedule is the same for all four models.
#
# EP0 navigates from empty memory and builds memory in Habitat world coordinates. The step budget is 300.
# EP1-EP4 each read that memory and update only their own branch. The step budget is 300.
# Branches do not propagate to each other. There is no geometric localization and no SE(2) alignment.
# Instruction parsing and top-3 selection call the VLM live. A parse that is not a single type-1 target is a miss.
# On a hit, the scene graph walks to the reachable cell. ETPNav and CMA turn 15 degrees. JanusVLN and Uni-LaViRA turn 30 degrees.
# If the controller has no next action, exploration continues on the scene graph.
# If exploration cannot continue, or the parse misses, the full instruction is given to the policy.
# Janus uses GPU_PAIRS. The other models use GPU_IDS.
# ROUTE_SOURCE=live is the default. ROUTE_SOURCE=saved reads a previously published parse.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLHN="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MEMX="${LLHN}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/conda_run.sh"
JANUS_MODEL="${JANUS_MODEL:-${MODELS_DIR:-${LLHN}/models}/JanusVLN_Extra}"

if [[ $# -ge 1 && "${1}" != -* ]]; then
  MODEL="$1"
  shift
fi
MODEL="${MODEL:-etpnav}"

die() { echo "[l1-membridge] ERROR: $*" >&2; exit 1; }
log() { echo "[l1-membridge] $*"; }

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
L1_DATA="${LLHN}/datasets/level1/habitat"
RUNNER="${MEMX}/baselines/run.py --mode memory --level 1 --pipeline membridge"
EP0_RUNNER="${MEMX}/baselines/run.py --mode basic --level 1 --pipeline membridge"
M0_BUILDER="${MEMX}/baselines/run.py --mode build --level 1 --pipeline membridge"
JANUS_ROOT="${JANUS_ROOT:-${LLHN}/baselines/models/JanusVLN}"
CMA_ROOT="${CMA_ROOT:-${LLHN}/baselines/models/CMA}"
UNI_ROOT="${UNI_ROOT:-${UNILAVIRA_ROOT:-${LLHN}/baselines/models/Uni-LaViRA}}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
MAX_STEPS="${MAX_STEPS:-300}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
QUERY_ROLES="${QUERY_ROLES:-ep1 ep2 ep3 ep4}"
ROUTE_SOURCE="${ROUTE_SOURCE:-live}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${MEMX}/experiments_results/${RESULT_NAME}/level1/runs/${RUN_TAG}}"
if [[ "${MODEL_KEY}" != "janus" ]]; then
  ETP_ROOT="$(memx_prepare_etpnav "${LLHN}" "${ETP_ROOT}" "${RUN_ROOT}")"
fi
ETP_DATA="$(readlink -f -- "${ETP_ROOT}/data")"
VLM_ENV_FILE="${VLM_ENV_FILE:-${UNILAVIRA_ROOT-}/.env}"

log "model=${MODEL_LABEL} run_root=${RUN_ROOT} coordinate_frame=habitat_world route_source=${ROUTE_SOURCE}"
[[ "${ROUTE_SOURCE}" == "live" || "${ROUTE_SOURCE}" == "saved" ]] \
  || die "ROUTE_SOURCE must be live or saved"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ "${STAGE}" == "all" || "${STAGE}" == "ep0" || "${STAGE}" == "query" ]] \
  || die "STAGE must be all, ep0, or query"
[[ -f "${MEMX}/baselines/run.py" ]] || die "missing ${MEMX}/baselines/run.py"
[[ -f "${L1_DATA}/level1_ep0.json.gz" && -f "${L1_DATA}/level1_ep1.json.gz" ]] \
  || die "missing Level 1 habitat split under ${L1_DATA}"
[[ -f "${VLM_ENV_FILE}" ]] || die "missing VLM env ${VLM_ENV_FILE}"

if [[ "${MODEL_KEY}" == "janus" ]]; then
  read -r -a WORKERS <<< "${GPU_PAIRS}"
else
  read -r -a WORKERS <<< "${GPU_IDS}"
fi
((${#WORKERS[@]} > 0)) || die "no GPUs"

build_m0() {
  local rollouts="$1"
  local output="${RUN_ROOT}/ep0_m0"
  log "build global M0 from ${rollouts}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips the M0 builder"
    return
  fi
  mkdir -p "${output}"
  memx_spawn 0 "${LLHN}" "${output}/build.log" \
    python ${M0_BUILDER} \
    --ep0-rollouts "${rollouts}" \
    --dataset "${L1_DATA}/level1_ep0.json.gz" \
    --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
    --output-root "${output}" \
    --memory-module-root "${LLHN}/membridge" \
    --coordinate-frame habitat_world \
    --query-vocabulary "${LLHN}/membridge/semantic_scene_graph/config/level1_runtime_query_vocabulary_75.json" \
    --sam3-ckpt "${MODELS_DIR:?Set MODELS_DIR}/facebook/sam3/sam3.pt" \
    --sam3-source-root "${MODELS_DIR}/source/sam3" \
    --dinov3-model "${MODELS_DIR}/facebook/dinov3" \
    --dinov3-repo "${MODELS_DIR}/source/dinov3" \
    --text-encoder "${MODELS_DIR}/all-MiniLM-L6-v2" \
    --expected-count "${EXPECTED_COUNT:-315}" \
    --skip-existing --continue-on-error >/dev/null
  wait "$!" || die "M0 build failed"
  log "M0 ${output}"
}

run_etpnav_ep0() {
  local output="${RUN_ROOT}/ep0"
  log "EP0: ETPNav navigation in habitat world, then global M0"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips ETPNav EP0"
    return
  fi
  mkdir -p "${output}/logs"
  local -a pids=()
  local shard=0 gpu fail=0
  for gpu in "${WORKERS[@]}"; do
    PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    memx_spawn "${gpu}" "${ETP_ROOT}" "${output}/logs/shard$(printf '%02d' "${shard}").log" \
      python run.py \
        --exp_name "l1_ep0_s${shard}" --run-type eval \
        --exp-config run_r2r/iter_level1.yaml \
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
  build_m0 "${output}"
}

run_discrete_ep0() {
  local output="${RUN_ROOT}/ep0"
  log "EP0: ${MODEL_LABEL} navigation in habitat world, then global M0"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips ${MODEL_LABEL} EP0"
    return
  fi
  case "${MODEL_KEY}" in
    janus) [[ -n "${JANUS_ROOT:-}" && -d "${JANUS_ROOT}" ]] || die "Set JANUS_ROOT to a JanusVLN checkout" ;;
    cma) [[ -n "${CMA_ROOT:-}" && -d "${CMA_ROOT}" ]] || die "Set CMA_ROOT to a CMA checkout" ;;
    unilavira) [[ -n "${UNI_ROOT:-}" && -d "${UNI_ROOT}" ]] || die "Set UNI_ROOT or UNILAVIRA_ROOT" ;;
  esac
  mkdir -p "${output}/logs"
  local -a pids=()
  local shard=0 worker fail=0
  for worker in "${WORKERS[@]}"; do
    local flags="--nav-model ${MODEL_KEY} --role ep0 --world-poses --dataset ${L1_DATA}/level1_ep0.json.gz"
    flags+=" --output-root ${output} --max-steps ${MAX_STEPS} --num-shards ${#WORKERS[@]} --shard-index ${shard}"
    [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
    local logf="${output}/logs/shard$(printf '%02d' "${shard}").log"
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      JANUSVLN_VGGT_DEVICE=cuda:0 JANUSVLN_LANGUAGE_DEVICE=cuda:1 HF_HUB_OFFLINE=1 \
      PYTHONPATH="${JANUS_ROOT}/src:${MEMX}:${MEMX}/membridge/localization:${MEMX}/baselines/adapt${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_spawn "${worker}" "${JANUS_ROOT}" "${logf}" \
        python ${EP0_RUNNER} \
          --janus-root "${JANUS_ROOT}" --janus-model "${JANUS_MODEL}" \
          --janus-config "${JANUS_ROOT}/config/vln_level1_protocol.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 ${flags}
    else
      local extra_py="" exp="${ETP_ROOT}/run_r2r/iter_level1.yaml"
      if [[ "${MODEL_KEY}" == "cma" ]]; then
        extra_py=":${CMA_ROOT}"
        exp="${CMA_ROOT}/experiment_level3/cma_level3.yaml"
        flags+=" --cma-root ${CMA_ROOT} --ckpt ${CMA_ROOT}/data/checkpoints/CMA_PM_DA_Aug.pth --exp-config ${exp}"
      else
        extra_py=":${UNI_ROOT}"
        flags+=" --unilavira-root ${UNI_ROOT} --exp-config ${exp}"
      fi
      PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${extra_py}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_spawn "${worker}" "${LLHN}" "${logf}" \
        python ${EP0_RUNNER} \
          --etpnav-root "${ETP_ROOT}" \
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level1.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 ${flags}
    fi
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "${MODEL_LABEL} EP0 failed"
  build_m0 "${output}"
}

run_query_role() {
  local role="$1"
  local m0="$2"
  local output="${RUN_ROOT}/query/${role}"
  local dataset="${L1_DATA}/level1_${role}.json.gz"
  log "query ${role}: read frozen M0 ${m0}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    log "dry run skips query ${role}"
    return
  fi
  case "${MODEL_KEY}" in
    janus) [[ -n "${JANUS_ROOT:-}" && -d "${JANUS_ROOT}" ]] || die "Set JANUS_ROOT to a JanusVLN checkout" ;;
    cma) [[ -n "${CMA_ROOT:-}" && -d "${CMA_ROOT}" ]] || die "Set CMA_ROOT to a CMA checkout" ;;
    unilavira) [[ -n "${UNI_ROOT:-}" && -d "${UNI_ROOT}" ]] || die "Set UNI_ROOT or UNILAVIRA_ROOT" ;;
  esac
  [[ -d "${m0}/sessions" ]] || die "missing M0 ${m0}"
  mkdir -p "${output}/logs"
  local -a pids=()
  local shard=0 worker fail=0
  for worker in "${WORKERS[@]}"; do
    local flags="--nav-model ${MODEL_KEY} --role ${role} --dataset ${dataset} --m0-root ${m0}"
    flags+=" --route-source ${ROUTE_SOURCE}"
    flags+=" --output-root ${output} --max-steps ${MAX_STEPS} --num-shards ${#WORKERS[@]} --shard-index ${shard}"
    [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
    local logf="${output}/logs/shard$(printf '%02d' "${shard}").log"
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      MEMX_ENV_FILE="${VLM_ENV_FILE}" \
      JANUSVLN_VGGT_DEVICE=cuda:0 JANUSVLN_LANGUAGE_DEVICE=cuda:1 HF_HUB_OFFLINE=1 \
      PYTHONPATH="${JANUS_ROOT}/src:${MEMX}:${MEMX}/membridge/runners:${MEMX}/membridge/localization:${MEMX}/baselines/adapt${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_spawn "${worker}" "${JANUS_ROOT}" "${logf}" \
        python ${RUNNER} --janus-root "${JANUS_ROOT}" --janus-model "${JANUS_MODEL}" \
          --janus-config "${JANUS_ROOT}/config/vln_level1_protocol.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 ${flags}
    else
      local extra_py="" exp="${ETP_ROOT}/run_r2r/iter_level1.yaml" ckpt="${ETP_DATA}/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
      if [[ "${MODEL_KEY}" == "cma" ]]; then
        extra_py=":${CMA_ROOT}"
        exp="${CMA_ROOT}/experiment_level3/cma_level3.yaml"
        ckpt="${CMA_ROOT}/data/checkpoints/CMA_PM_DA_Aug.pth"
        flags+=" --cma-root ${CMA_ROOT} --r2r-train ${CMA_ROOT}/data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz"
      elif [[ "${MODEL_KEY}" == "unilavira" ]]; then
        extra_py=":${UNI_ROOT}"
        flags+=" --unilavira-root ${UNI_ROOT}"
      fi
      MEMX_ENV_FILE="${VLM_ENV_FILE}" \
      INHERIT_SG_SAM3_CKPT="${MODELS_DIR:?Set MODELS_DIR}/facebook/sam3/sam3.pt" \
      INHERIT_SG_SAM3_SOURCE_ROOT="${MODELS_DIR}/source/sam3" \
      SAM3_SOURCE_ROOT="${MODELS_DIR}/source/sam3" \
      INHERIT_SG_DINOV3_MODEL="${MODELS_DIR}/facebook/dinov3" \
      INHERIT_SG_DINOV3_REPO="${MODELS_DIR}/source/dinov3" \
      INHERIT_SG_TEXT_ENCODER="${MODELS_DIR}/all-MiniLM-L6-v2" \
      PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${extra_py}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/runners:${MEMX}/membridge/localization${PYTHONPATH:+:${PYTHONPATH}}" \
      memx_spawn "${worker}" "${LLHN}" "${logf}" \
        python ${RUNNER} --exp-config "${exp}" --ckpt "${ckpt}" \
          --etpnav-root "${ETP_ROOT}" \
          --waypoint-ckpt "${ETP_DATA}/wp_pred/check_cwp_bestdist_hfov90" \
          --task-config "${ETP_ROOT}/run_r2r/hm3d_level1.yaml" \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 ${flags}
    fi
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || die "${MODEL_LABEL} query ${role} failed"
  log "finished ${output}"
}

run_ep0() {
  if [[ "${MODEL_KEY}" == "etpnav" ]]; then
    run_etpnav_ep0
  else
    run_discrete_ep0
  fi
}

run_queries() {
  local m0="${M0_ROOT:-${RUN_ROOT}/ep0_m0}"
  m0="$(readlink -f -- "${m0}")"
  log "queries read M0 ${m0}"
  local role
  for role in ${QUERY_ROLES}; do
    run_query_role "${role}" "${m0}"
  done
}

case "${STAGE}" in
  all)
    run_ep0
    if [[ "${DRY_RUN}" == 1 ]]; then
      log "dry run stops after EP0; queries need the M0 from a real EP0"
      exit 0
    fi
    run_queries
    ;;
  ep0) run_ep0 ;;
  query) run_queries ;;
esac
