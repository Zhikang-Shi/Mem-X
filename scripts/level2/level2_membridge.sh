#!/usr/bin/env bash
# Level 2 +MemBridge. MODEL selects the policy. The schedule is the same for all four models.
#
# EP0 navigates and then builds episode-local memory. The step budget is 300.
# EP1 localizes against that memory, then updates the scene graph, navigation graph, and trajectory. The step budget is 300.
# EP2 reads both memory stages, localizes, and updates the same structures. The step budget is 300. There is no extra exploration.
# Each episode uses its own local frame. On a retrieval hit with a usable alignment, the scene graph walks in the aligned frame.
# The type-2 suffix is given to the policy. A miss or a failed alignment gives the full instruction to the policy.
# Janus uses GPU_PAIRS. The other models use GPU_IDS.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
ETP_DATA="$(readlink -f -- "${ETP_ROOT}/data")"
JANUS_ROOT="${JANUS_ROOT:-${LLHN}/baselines/models/JanusVLN}"
CMA_ROOT="${CMA_ROOT:-${LLHN}/baselines/models/CMA}"
UNI_ROOT="${UNI_ROOT:-${UNILAVIRA_ROOT:-${LLHN}/baselines/models/Uni-LaViRA}}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
MAX_STEPS="${MAX_STEPS:-300}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${MEMX}/experiments_results/${RESULT_NAME}/level2/runs/${RUN_TAG}}"
VLM_ENV_FILE="${VLM_ENV_FILE:-${UNILAVIRA_ROOT-}/.env}"
FROZEN_M0="${FROZEN_M0:-}"
FROZEN_M1="${FROZEN_M1:-}"
FROZEN_EP0="${FROZEN_EP0:-}"
DATASET_EP0="${LLHN}/datasets/level2/habitat/level2_ep0.json.gz"
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
  docker run --rm --gpus '"device=0"' --network host --shm-size 16g \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env http_proxy=http://127.0.0.1:7897 --env https_proxy=http://127.0.0.1:7897 \
    --volume "${LLHN}:/workspace/LifeLongHomeNav:ro" \
    --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
    --volume "${MODELS_DIR}:${MODELS_DIR}:ro" \
    --volume "${output}:/workspace/output" \
    --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} bash -lc "
      set -euo pipefail
      PY=/opt/conda/envs/lavira/bin/python
      \$PY -m pip install -q --disable-pip-version-check sentence-transformers scikit-learn iopath timm ftfy huggingface_hub einops psutil
      exec \$PY /workspace/LifeLongHomeNav/baselines/run.py --mode build --level 2 --pipeline membridge \
      --ep0-rollouts "${rollouts}" \
      --dataset "${DATASET_EP0}" \
      --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
      --output-root /workspace/output \
      --memory-module-root /workspace/LifeLongHomeNav/membridge \
      --expected-count 238 \
      --skip-existing --continue-on-error
    "
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
  local flags="--nav-model ${MODEL_KEY} --dataset ${DATASET_EP0} --output-root /workspace/output"
  flags+=" --max-steps ${MAX_STEPS} --num-shards ${#WORKERS[@]} --shard-index ${shard}"
  [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
  if [[ "${MODEL_KEY}" == "janus" ]]; then
    [[ "${worker}" =~ ^[0-9]+,[0-9]+$ ]] || die "GPU pair must be FIRST,SECOND: ${worker}"
    env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
      --network host --shm-size 16g \
      --env CUDA_VISIBLE_DEVICES=0,1 \
      --env JANUSVLN_VGGT_DEVICE=cuda:0 --env JANUSVLN_LANGUAGE_DEVICE=cuda:1 \
      --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
      --env TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch} --env HOME=/root \
      --workdir /janus/src/JanusVLN \
      --env PYTHONPATH="${MEMX}/baselines/models/JanusVLN/src:${MEMX}:${MEMX}/baselines/adapt:${MEMX}/baselines/adapt" \
      --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
      --volume "${JANUS_ENV:?Set JANUS_ENV to the streamvln environment}:/janus/envs/streamvln:ro" \
      --volume "${MEMX}:${MEMX}:ro" \
      --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
      --volume "${native}:/workspace/output" \
      --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} /janus/envs/streamvln/bin/python ${EP0_RUNNER} \
        --etpnav-root ${MEMX}/baselines/models/ETPNav \
        --exp-config /etpnav/run_r2r/iter_level2.yaml \
        --task-config /etpnav/run_r2r/hm3d_level3.yaml \
        --janus-root ${MEMX}/baselines/models/JanusVLN \
        --janus-model /janus/models/JanusVLN_Extra \
        --janus-config /janus/src/JanusVLN/config/vln_level2_protocol.yaml \
        --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
        ${flags}
  elif [[ "${MODEL_KEY}" == "cma" ]]; then
    env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
      --network host --shm-size 16g --env-file "${VLM_ENV_FILE}" \
      --env CUDA_VISIBLE_DEVICES=0 --env HOME=/root \
      --env PYTHONPATH="/root/habitat-lab:/etpnav:/cma:${MEMX}/baselines/adapt:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/runners" \
      --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
      --volume "${CMA_ROOT}:/cma:ro" \
      --volume "${MEMX}:${MEMX}:ro" \
      --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
      --volume "${native}:/workspace/output" \
      --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} /opt/conda/envs/lavira/bin/python ${EP0_RUNNER} \
        --etpnav-root ${MEMX}/baselines/models/ETPNav \
        --exp-config /cma/experiment_level3/cma_level3.yaml \
        --task-config /etpnav/run_r2r/hm3d_level3.yaml \
        --cma-root ${MEMX}/baselines/models/CMA \
        --ckpt ${MEMX}/baselines/models/CMA/data/checkpoints/CMA_PM_DA_Aug.pth \
        --r2r-train ${MEMX}/baselines/models/CMA/data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz \
        --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
        ${flags}
  else
    env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
      --network host --shm-size 16g --env-file "${VLM_ENV_FILE}" \
      --env CUDA_VISIBLE_DEVICES=0 --env HOME=/root \
      --env PYTHONPATH="/root/habitat-lab:/etpnav:/unilavira:${MEMX}/baselines/adapt:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/runners" \
      --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
      --volume "${UNI_ROOT}:/unilavira:ro" \
      --volume "${MEMX}:${MEMX}:ro" \
      --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
      --volume "${native}:/workspace/output" \
      --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} /opt/conda/envs/lavira/bin/python ${EP0_RUNNER} \
        --etpnav-root ${MEMX}/baselines/models/ETPNav \
        --exp-config /etpnav/run_r2r/iter_level2.yaml \
        --task-config /etpnav/run_r2r/hm3d_level3.yaml \
        --unilavira-root ${MEMX}/baselines/models/Uni-LaViRA \
        --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
        ${flags}
  fi
}

run_ep0() {
  if [[ "${MODEL_KEY}" == "etpnav" ]]; then
    log "EP0: ETPNav navigation, then async local M0"
    DRY_RUN="${DRY_RUN}" GPU_IDS="${GPU_IDS}" RUN_TAG="${RUN_TAG}" RUN_ROOT="${RUN_ROOT}" \
      bash "${EP0_ETP}"
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
      flags+=" --cma-root ${MEMX}/baselines/models/CMA --r2r-train ${MEMX}/baselines/models/CMA/data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz"
    elif [[ "${MODEL_KEY}" == "unilavira" ]]; then
      flags+=" --unilavira-root ${MEMX}/baselines/models/Uni-LaViRA"
    fi
    local exp_config="/etpnav/run_r2r/iter_level2.yaml"
    local ckpt="/etpnav/data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
    if [[ "${MODEL_KEY}" == "cma" ]]; then
      exp_config="/cma/experiment_level3/cma_level3.yaml"
      ckpt="/cma/data/checkpoints/CMA_PM_DA_Aug.pth"
    fi
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      [[ "${worker}" =~ ^[0-9]+,[0-9]+$ ]] || die "GPU pair must be FIRST,SECOND: ${worker}"
      env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
        --network host --shm-size 16g --env-file "${VLM_ENV_FILE}" \
        --env CUDA_VISIBLE_DEVICES=0,1 \
        --env JANUSVLN_VGGT_DEVICE=cuda:0 --env JANUSVLN_LANGUAGE_DEVICE=cuda:1 \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        --env http_proxy=http://127.0.0.1:7897 --env https_proxy=http://127.0.0.1:7897 \
        --env TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch} --env HOME=/root \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --env TOKENIZERS_PARALLELISM=false \
        --workdir /janus/src/JanusVLN \
        --env PYTHONPATH="${MEMX}/baselines/models/JanusVLN/src:${MEMX}:${MEMX}/membridge/localization:${MEMX}/membridge/runners:${MEMX}/baselines/adapt:${MEMX}/baselines/adapt:${LIGHTGLUE_ROOT}" \
        --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
        --volume "${JANUS_ENV:?Set JANUS_ENV to the streamvln environment}:/janus/envs/streamvln:ro" \
        --volume "${MEMX}:${MEMX}:ro" \
        --volume "${MODELS_DIR}:${MODELS_DIR}:ro" \
        --volume ${LIGHTGLUE_ROOT}:${LIGHTGLUE_ROOT}:ro \
        --volume "${HOST_CACHE:-${HOME}/.cache}:/root/.cache:rw" \
        --volume ${TORCH_HOME:-${HOME}/.cache/torch}:${TORCH_HOME:-${HOME}/.cache/torch}:rw \
        --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
        --volume "${output}:/workspace/output" \
        --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} bash -lc "
          set -euo pipefail
          PY=/janus/envs/streamvln/bin/python
          \$PY -m pip install -q --disable-pip-version-check sentence-transformers scikit-learn iopath timm ftfy huggingface_hub einops psutil
          exec \$PY ${RUNNER} \
          --etpnav-root ${MEMX}/baselines/models/ETPNav \
          --exp-config /etpnav/run_r2r/iter_level2.yaml \
          --task-config /etpnav/run_r2r/hm3d_level3.yaml \
          --dataset ${dataset} \
          --ckpt /etpnav/data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
          --waypoint-ckpt /etpnav/data/wp_pred/check_cwp_bestdist_hfov90 \
          --output-root /workspace/output --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
          --device cuda:0 --sim-gpu-id 0 \
          --janus-root ${MEMX}/baselines/models/JanusVLN \
          --janus-model /janus/models/JanusVLN_Extra \
          --janus-config /janus/src/JanusVLN/config/vln_level2_protocol.yaml \
          ${flags}
        "
    else
      local extra_vol="" extra_py=""
      if [[ "${MODEL_KEY}" == "cma" ]]; then
        extra_vol="--volume ${CMA_ROOT}:/cma:ro"
        extra_py=":/cma"
      elif [[ "${MODEL_KEY}" == "unilavira" ]]; then
        extra_vol="--volume ${UNI_ROOT}:/unilavira:ro"
        extra_py=":/unilavira"
      fi
      env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
        --network host --shm-size 16g --env-file "${VLM_ENV_FILE}" \
        --env CUDA_VISIBLE_DEVICES=0 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        --env http_proxy=http://127.0.0.1:7897 --env https_proxy=http://127.0.0.1:7897 \
        --env TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch} --env HOME=/root \
        --env PYTHONPATH="/root/habitat-lab:/etpnav${extra_py}:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization:${MEMX}/membridge/runners:${MEMX}/baselines/adapt:${LIGHTGLUE_ROOT}" \
        --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
        ${extra_vol} \
        --volume "${MEMX}:${MEMX}:ro" \
        --volume "${MODELS_DIR}:${MODELS_DIR}:ro" \
        --volume ${LIGHTGLUE_ROOT}:${LIGHTGLUE_ROOT}:ro \
        --volume "${HOST_CACHE:-${HOME}/.cache}:/root/.cache:rw" \
        --volume ${TORCH_HOME:-${HOME}/.cache/torch}:${TORCH_HOME:-${HOME}/.cache/torch}:ro \
        --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
        --volume "${output}:/workspace/output" \
        --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} bash -lc "
          set -euo pipefail
          PY=/opt/conda/envs/lavira/bin/python
          flock /etpnav/third_party/CLIP/.pip.lock \
            \$PY -m pip install -q --disable-pip-version-check -e /etpnav/third_party/CLIP jsonlines msgpack-numpy sentence-transformers scikit-learn iopath timm ftfy huggingface_hub einops psutil
          exec \$PY ${RUNNER} \
            --etpnav-root ${MEMX}/baselines/models/ETPNav \
            --exp-config ${exp_config} \
            --task-config /etpnav/run_r2r/hm3d_level3.yaml \
            --dataset ${dataset} \
            --ckpt ${ckpt} \
            --waypoint-ckpt /etpnav/data/wp_pred/check_cwp_bestdist_hfov90 \
            --output-root /workspace/output --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
            --device cuda:0 --sim-gpu-id 0 \
            ${flags}
        "
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
