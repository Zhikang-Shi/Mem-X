#!/usr/bin/env bash
# One-shard native ETPNav Level-2 EP0. Empty GraphMap, world xyz+xyzw.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ETP_ROOT="${ETPNAV_ROOT:?Set ETPNAV_ROOT}"
ETP_DATA="${ETP_DATA:-/home/shizk/LifeLongHomeNav/baseline_experiment/ETPNav/data}"
L2_HABITAT="${L2_HABITAT:-${LLHN}/datasets/level2/habitat}"
[[ -f "${L2_HABITAT}/level2_ep0.json.gz" && -f "${L2_HABITAT}/level2_ep0_gt.json.gz" ]] || {
  echo "[etpnav-l2-ep0] missing Level 2 habitat split under ${L2_HABITAT}" >&2
  exit 1
}
IMAGE="${IMAGE:-${MEMX_IMAGE:-mem-x:latest}}"
GPU="${GPU:-0}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MEMORY_ROOT="${MEMORY_ROOT:?MEMORY_ROOT is required}"
NAME="${NAME:-etpnav-l2-ep0-g${GPU}-s${SHARD_ID}}"
CKPT="${CKPT:-data/logs/checkpoints/release_r2r/ckpt.iter12000.pth}"
WP="${WP:-data/wp_pred/check_cwp_bestdist_hfov90}"
mkdir -p "${MEMORY_ROOT}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run --rm --name "${NAME}" --gpus "device=${GPU}" --network host --shm-size 8g \
  --workdir /etpnav \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false \
  --env MAGNUM_LOG=quiet --env GLOG_minloglevel=2 \
  --env PYTHONPATH=/root/habitat-lab:/etpnav \
  --volume "${ETP_ROOT}:/etpnav:rw" --volume "${ETP_DATA}:/etpnav/data:ro" \
  --volume "${L2_HABITAT}:/l2habitat:ro" \
  --volume "${SCENES_DIR}:${SCENES_DIR}:ro" \
  --volume "${HOST_CACHE:-${HOME}/.cache}:/root/.cache:rw" \
  --volume "${MODELS_DIR}:${MODELS_DIR}:ro" \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    PY=/opt/conda/envs/lavira/bin/python
    flock /etpnav/third_party/CLIP/.pip.lock \
      \$PY -m pip install -q --disable-pip-version-check -e /etpnav/third_party/CLIP jsonlines msgpack-numpy
    cd /etpnav
    exec \$PY run.py \
      --exp_name l2_ep0_s${SHARD_ID} --run-type eval \
      --exp-config run_r2r/iter_level2.yaml \
      SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
      EVAL.CKPT_PATH_DIR ${CKPT} EVAL.SPLIT ep0 EVAL.EPISODE_COUNT -1 \
      DATASET.DATA_PATH /datasets/level2/r2r/level2_{split}.json.gz \
      TASK.NDTW.GT_PATH /datasets/level2/r2r/level2_{split}_gt.json.gz \
      TASK.SDTW.GT_PATH /datasets/level2/r2r/level2_{split}_gt.json.gz \
      EVAL.DUMP_MEMORY True EVAL.LOAD_GRAPHMAP False \
      EVAL.MEMORY_ROOT ${MEMORY_ROOT} EVAL.MEMORY_STAGE M0 \
      EVAL.SHARD_ID ${SHARD_ID} EVAL.NUM_SHARDS ${NUM_SHARDS} \
      IL.waypoint_ckpt ${WP} IL.back_algo control IL.is_requeue False
  "
