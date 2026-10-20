#!/usr/bin/env bash
# Level 1 entry. PIPELINE selects native or +MemBridge. MODEL selects the policy.
#
# membridge runs EP0, then EP1-EP4. EP0 builds memory. Each later episode reads that memory and updates only its own branch.
# native navigates without a scene graph. Coordinates stay in the Habitat world frame.
# ETPNav, CMA, and Uni-LaViRA clear memory at each episode. Janus keeps its visual cache from EP0 through EP4.
# Janus uses GPU_PAIRS. The other models use GPU_IDS.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${PIPELINE:-membridge}"

case "${PIPELINE}" in
  membridge|+membridge|plus)
    exec bash "${SCRIPT_DIR}/level1_membridge.sh" "$@"
    ;;
  native|original)
    exec bash "${SCRIPT_DIR}/level1_native.sh" "$@"
    ;;
  *)
    echo "[l1] ERROR: PIPELINE must be membridge or native: ${PIPELINE}" >&2
    exit 1
    ;;
esac
