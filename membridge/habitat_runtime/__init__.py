"""Cross-module Habitat runtime for the eight memory capabilities.

This package owns only framework orchestration: evaluator registration,
episode lifecycle, environment step interception, and warm-branch binding.
Algorithms remain in the eight capability packages under ``memory_module``.
"""

from .bootstrap import MEMORY_WARM_TRAINER, install_runtime

__all__ = ["MEMORY_WARM_TRAINER", "install_runtime"]
