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
    docker run --rm --gpus "device=${gpu}" --network host --shm-size 8g \
      --workdir /etpnav \
      --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false \
      --env MAGNUM_LOG=quiet --env GLOG_minloglevel=2 \
      --env PYTHONPATH=/root/habitat-lab:/etpnav \
      --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:rw" \
      --volume "${MEMX}:${MEMX}:ro" \
      --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
      --volume "${HOST_CACHE:-${HOME}/.cache}:/root/.cache:rw" \
      --volume "${MODELS_DIR}:${MODELS_DIR}:ro" \
      --volume "${output}:/workspace/output" \
      --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} bash -lc "
        set -euo pipefail
        PY=/opt/conda/envs/lavira/bin/python
        flock /etpnav/third_party/CLIP/.pip.lock \
          \$PY -m pip install -q --disable-pip-version-check -e /etpnav/third_party/CLIP jsonlines msgpack-numpy
        cd /etpnav
        exec \$PY run.py \
          --exp_name l1_${role}_s${shard} --run-type eval \
          --exp-config run_r2r/iter_level1.yaml \
          SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
          EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
          EVAL.SPLIT ${role} EVAL.EPISODE_COUNT ${EPISODE_COUNT:--1} \
          EVAL.DUMP_MEMORY True EVAL.LOAD_GRAPHMAP False \
          EVAL.MEMORY_ROOT /workspace/output EVAL.MEMORY_STAGE ${stage} \
          EVAL.SHARD_ID ${shard} EVAL.NUM_SHARDS ${#WORKERS[@]} \
          IL.back_algo control IL.is_requeue False
      " >"${output}/logs/shard$(printf '%02d' "${shard}").log" 2>&1 &
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
  local dataset="/datasets/level1/habitat/level1_${role}.json.gz"
  log "${MODEL_LABEL} ${role}: habitat world, output=${output}"
  mkdir -p "${output}/logs"
  launch_one() {
    local worker="$1" shard="$2" dry="$3"
    local turn=15
    if [[ "${MODEL_KEY}" == "janus" || "${MODEL_KEY}" == "unilavira" ]]; then
      turn=30
    fi
    local flags="--nav-model ${MODEL_KEY} --role ${role} --dataset ${dataset} --output-root /workspace/output"
    flags+=" --world-poses --turn-angle ${turn} --cache-root ${RUN_ROOT}"
    flags+=" --max-steps ${MAX_STEPS} --num-shards ${#WORKERS[@]} --shard-index ${shard}"
    [[ "${dry}" == 1 ]] && flags+=" --dry-run"
    [[ -n "${SESSION_ID:-}" ]] && flags+=" --session-id ${SESSION_ID}"
    if [[ "${MODEL_KEY}" == "janus" ]]; then
      [[ "${worker}" =~ ^[0-9]+,[0-9]+$ ]] || die "GPU pair must be FIRST,SECOND: ${worker}"
      env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
        --network host --shm-size 16g \
        --env CUDA_VISIBLE_DEVICES=0,1 \
        --env JANUSVLN_VGGT_DEVICE=cuda:0 --env JANUSVLN_LANGUAGE_DEVICE=cuda:1 \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HOME=/root \
        --workdir /janus/src/JanusVLN \
        --env PYTHONPATH="${MEMX}/baselines/models/JanusVLN/src:${MEMX}:${MEMX}/membridge/localization:${MEMX}/baselines/adapt:${MEMX}/baselines/adapt" \
        --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
        --volume "${JANUS_ENV:?Set JANUS_ENV to the streamvln environment}:/janus/envs/streamvln:ro" \
        --volume "${MEMX}:${MEMX}:ro" \
        --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
        --volume "${RUN_ROOT}:${RUN_ROOT}" \
        --volume "${output}:/workspace/output" \
        --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} /janus/envs/streamvln/bin/python ${RUNNER} \
          --etpnav-root ${MEMX}/baselines/models/ETPNav \
          --janus-root ${MEMX}/baselines/models/JanusVLN \
          --janus-model /janus/models/JanusVLN_Extra \
          --janus-config /janus/src/JanusVLN/config/vln_level1_protocol.yaml \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
          ${flags}
    elif [[ "${MODEL_KEY}" == "cma" ]]; then
      flags+=" --cma-root ${MEMX}/baselines/models/CMA --ckpt ${MEMX}/baselines/models/CMA/data/checkpoints/CMA_PM_DA_Aug.pth"
      flags+=" --r2r-train ${MEMX}/baselines/models/CMA/data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz"
      flags+=" --exp-config /cma/experiment_level3/cma_level3.yaml"
      env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
        --network host --shm-size 16g --env-file "${VLM_ENV_FILE}" \
        --env CUDA_VISIBLE_DEVICES=0 --env HOME=/root \
        --env PYTHONPATH="/root/habitat-lab:/etpnav:/cma:${MEMX}/baselines/adapt:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization" \
        --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
        --volume "${CMA_ROOT}:/cma:ro" \
        --volume "${MEMX}:${MEMX}:ro" \
        --volume ${CLIP_CACHE:-${HOME}/.cache/clip}:/root/.cache/clip:ro \
        --volume ${TORCH_HUB:-${HOME}/.cache/torch/hub/checkpoints}:/root/.cache/torch/hub/checkpoints:ro \
        --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
        --volume "${output}:/workspace/output" \
        --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} /opt/conda/envs/lavira/bin/python ${RUNNER} \
          --task-config /etpnav/run_r2r/hm3d_level1.yaml \
          --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" --sim-gpu-id 0 \
          ${flags}
    else
      flags+=" --unilavira-root ${MEMX}/baselines/models/Uni-LaViRA --exp-config /etpnav/run_r2r/iter_level1.yaml"
      env -u NVIDIA_VISIBLE_DEVICES docker run --rm --gpus "\"device=${worker}\"" \
        --network host --shm-size 16g --env-file "${VLM_ENV_FILE}" \
        --env CUDA_VISIBLE_DEVICES=0 --env HOME=/root \
        --env PYTHONPATH="/root/habitat-lab:/etpnav:/unilavira:${MEMX}/baselines/adapt:${MEMX}/baselines/adapt:${MEMX}:${MEMX}/membridge/localization" \
        --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
        --volume "${UNI_ROOT}:/unilavira:ro" \
        --volume "${MEMX}:${MEMX}:ro" \
        --volume ${CLIP_CACHE:-${HOME}/.cache/clip}:/root/.cache/clip:ro \
        --volume ${TORCH_HUB:-${HOME}/.cache/torch/hub/checkpoints}:/root/.cache/torch/hub/checkpoints:ro \
        --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
        --volume "${output}:/workspace/output" \
        --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} /opt/conda/envs/lavira/bin/python ${RUNNER} \
          --task-config /etpnav/run_r2r/hm3d_level1.yaml \
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
