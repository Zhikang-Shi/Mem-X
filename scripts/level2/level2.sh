#!/usr/bin/env bash
# Level 2 entry. PIPELINE selects native or +MemBridge. MODEL selects the policy.
#
# membridge runs EP0, EP1, and EP2 and updates memory online.
# native navigates without MemBridge. ETPNav, CMA, and Uni-LaViRA clear memory at each episode. Janus keeps its visual cache.
# Janus uses GPU_PAIRS. The other models use GPU_IDS.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${PIPELINE:-membridge}"

case "${PIPELINE}" in
  membridge|+membridge|plus)
    exec bash "${SCRIPT_DIR}/level2_membridge.sh" "$@"
    ;;
  native|original)
    exec bash "${SCRIPT_DIR}/level2_native.sh" "$@"
    ;;
  *)
    echo "[l2] ERROR: PIPELINE must be membridge or native: ${PIPELINE}" >&2
    exit 1
    ;;
esac
