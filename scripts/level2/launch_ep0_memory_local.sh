#!/usr/bin/env bash
# ETPNav Level-2 EP0: empty-GraphMap navigation, then local M0.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLHN="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${LLHN}/experiments_results/ETPNav/level2/runs/${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
NATIVE_ROOT="${NATIVE_ROOT:-${RUN_ROOT}/ep0_native}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/ep0_m0}"
DATASET="${LLHN}/datasets/level2/habitat/level2_ep0.json.gz"
[[ -f "${DATASET}" ]] || { echo "[etpnav-l2-ep0] missing ${DATASET}" >&2; exit 1; }
if [[ "${DRY_RUN}" == 1 && -z "${ROLLOUT_ROOT:-}" ]]; then
  echo "[etpnav-l2-ep0] dry run: native EP0, then local M0 from ${DATASET}"
  exit 0
fi
if [[ -z "${ROLLOUT_ROOT:-}" ]]; then
  mkdir -p "${NATIVE_ROOT}/logs"
  read -r -a GPUS <<< "${GPU_IDS}"
  declare -a PIDS=()
  shard=0
  for gpu in "${GPUS[@]}"; do
    GPU="${gpu}" SHARD_ID="${shard}" NUM_SHARDS="${#GPUS[@]}" MEMORY_ROOT="${NATIVE_ROOT}" \
      NAME="etpnav-l2-ep0-g${gpu}-s${shard}-${RUN_TAG}" \
      bash "${SCRIPT_DIR}/run_ep0_native.sh" \
      >"${NATIVE_ROOT}/logs/shard$(printf '%02d' "${shard}").log" 2>&1 &
    PIDS+=("$!")
    shard=$((shard + 1))
  done
  fail=0
  for pid in "${PIDS[@]}"; do wait "${pid}" || fail=1; done
  [[ "${fail}" -eq 0 ]] || { echo "[etpnav-l2-ep0] navigation failed" >&2; exit 1; }
  ROLLOUT_ROOT="${NATIVE_ROOT}"
fi
mkdir -p "${SCRIPT_DIR}/outputs" "${OUTPUT_ROOT}"
ln -sfn "${OUTPUT_ROOT}" "${SCRIPT_DIR}/outputs/ep0_memory_from_poses_local_latest"
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "[etpnav-l2-ep0] dry run builder from ${ROLLOUT_ROOT}"
  exit 0
fi
docker run --rm --gpus '"device=0"' --network host --shm-size 16g \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env http_proxy=http://127.0.0.1:7897 --env https_proxy=http://127.0.0.1:7897 \
  --volume "${LLHN}:/workspace/LifeLongHomeNav:ro" \
  --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
  --volume "${MODELS_DIR}:${MODELS_DIR}:ro" \
  --volume "${OUTPUT_ROOT}:/workspace/output" \
  --volume "${LLHN}/datasets:/datasets:ro" ${MEMX_IMAGE:-mem-x:latest} bash -lc "
    set -euo pipefail
    PY=/opt/conda/envs/lavira/bin/python
    \$PY -m pip install -q --disable-pip-version-check sentence-transformers scikit-learn iopath timm ftfy huggingface_hub einops psutil
    exec \$PY /workspace/LifeLongHomeNav/baselines/run.py --mode build --level 2 --pipeline membridge \
    --ep0-rollouts "${ROLLOUT_ROOT}" \
    --dataset "${DATASET}" \
    --scenes-dir "${SCENES_DIR:?Set SCENES_DIR}" \
    --output-root /workspace/output \
    --memory-module-root /workspace/LifeLongHomeNav/membridge \
    --expected-count 238 \
    --skip-existing --continue-on-error
  "
echo "[etpnav-l2-ep0] M0 ${OUTPUT_ROOT}"
