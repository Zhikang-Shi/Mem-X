#!/usr/bin/env bash
# One-shard native ETPNav Level-2 EP0. Empty GraphMap, world xyz+xyzw.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLHN="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ETP_ROOT="${ETPNAV_ROOT:-${LLHN}/baselines/models/ETPNav}"
L2_HABITAT="${L2_HABITAT:-${LLHN}/datasets/level2/habitat}"
[[ -f "${L2_HABITAT}/level2_ep0.json.gz" && -f "${L2_HABITAT}/level2_ep0_gt.json.gz" ]] || {
  echo "[etpnav-l2-ep0] missing Level 2 habitat split under ${L2_HABITAT}" >&2
  exit 1
}
GPU="${GPU:-0}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MEMORY_ROOT="${MEMORY_ROOT:?MEMORY_ROOT is required}"
NAME="${NAME:-etpnav-l2-ep0-g${GPU}-s${SHARD_ID}}"
CKPT="${CKPT:-data/logs/checkpoints/release_r2r/ckpt.iter12000.pth}"
mkdir -p "${MEMORY_ROOT}"
SCRIPT_LIB="$(cd -- "${SCRIPT_DIR}/../lib" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_LIB}/conda_run.sh"
ETP_ROOT="$(memx_prepare_etpnav "${LLHN}" "${ETP_ROOT}" "${MEMORY_ROOT}")"
export PYTHONPATH="${ETP_ROOT}/third_party/CLIP:${ETP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MAGNUM_LOG=quiet GLOG_minloglevel=2
memx_run "${GPU}" "${ETP_ROOT}" \
  python run.py \
    --exp_name "l2_ep0_s${SHARD_ID}" --run-type eval \
    --exp-config run_r2r/iter_level2.yaml \
    SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
    EVAL.CKPT_PATH_DIR "${CKPT}" EVAL.SPLIT ep0 EVAL.EPISODE_COUNT "${EPISODE_COUNT:--1}" \
    TASK_CONFIG.DATASET.DATA_PATH "${LLHN}/datasets/level2/habitat/level2_{split}.json.gz" \
    TASK_CONFIG.TASK.NDTW.GT_PATH "${LLHN}/datasets/level2/habitat/level2_{split}_gt.json.gz" \
    TASK_CONFIG.TASK.SDTW.GT_PATH "${LLHN}/datasets/level2/habitat/level2_{split}_gt.json.gz" \
    EVAL.DUMP_MEMORY True EVAL.LOAD_GRAPHMAP False \
    EVAL.MEMORY_ROOT "${MEMORY_ROOT}" EVAL.MEMORY_STAGE M0 \
    EVAL.SHARD_ID "${SHARD_ID}" EVAL.NUM_SHARDS "${NUM_SHARDS}" \
    IL.back_algo control IL.is_requeue False
