"""Complete goal-directed exploration runtime owned by memory_module."""

from .exploration_patch import ExplorationConfig, install_goal_directed_exploration

__all__ = ["ExplorationConfig", "install_goal_directed_exploration"]
