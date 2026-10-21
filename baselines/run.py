#!/usr/bin/env python3
"""One entry for level1.sh, level2.sh, and level3.sh.

  python baselines/run.py --mode basic  --level 2 --pipeline native --role ep0 ...
  python baselines/run.py --mode memory --level 3 --pipeline membridge --role ep2 ...
  python baselines/run.py --mode build  --level 1 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runners"))
from bootstrap import install  # noqa: E402

install()
from stage import stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "basic", "memory", "build"), default="auto")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--pipeline", choices=("membridge", "native"), default="membridge")
    parser.add_argument("--role", default="ep0")
    parser.add_argument("--nav-model", default="")
    args, rest = parser.parse_known_args()
    chosen = stage(args.level, args.role, args.pipeline)
    mode = args.mode
    if mode == "auto":
        if chosen["kind"] == "basic" and chosen["localize"]:
            mode = "basic-localize"
        else:
            mode = chosen["kind"]
    elif mode == "basic" and chosen["localize"]:
        mode = "basic-localize"
    if mode == "memory" and chosen["explore_steps"] and "--explore-steps" not in rest:
        rest.extend(["--explore-steps", str(chosen["explore_steps"])])
    needs_role = mode == "basic" or (mode == "memory" and args.level in (1, 2))
    if needs_role and "--role" not in rest:
        rest.extend(["--role", args.role])
    forward_nav_model = mode in {"basic", "basic-localize"} or (
        mode == "memory" and args.level in (1, 2, 3)
    )
    if forward_nav_model and args.nav_model and "--nav-model" not in rest:
        rest.extend(["--nav-model", args.nav_model])
    if mode == "basic":
        from run_level2_native import main as entry
    elif mode == "basic-localize":
        from run_ep1_memory_etpnav_local_loc import main as entry
    elif mode == "build":
        from build_ep0_memory_from_poses_local import main as entry
    elif args.level == 1:
        from run_level1_membridge import main as entry
    elif args.level == 2:
        from run_level2_membridge import main as entry
    else:
        from run_ep2_explore_loc_sg import main as entry
    sys.argv = [str(Path(__file__).resolve()), *rest]
    code = entry()
    return 0 if code is None else int(code)


if __name__ == "__main__":
    raise SystemExit(main())
