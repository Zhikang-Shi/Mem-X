"""Uni-LaViRA navigator. One VLM decision becomes one Habitat action."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


class _SilentViz:
    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            return None

        return _noop


def _action_id(text: str) -> int:
    name = str(text or "").lower()
    if "stop" in name:
        return 0
    if "left" in name or "behind" in name:
        return 2
    if "right" in name:
        return 3
    return 1


class UniLaViRADiscretePolicy:
    discrete_actions = True
    nav_name = "unilavira"

    def __init__(self, args: Any) -> None:
        root = Path(
            getattr(
                args,
                "unilavira_root",
                str(Path(__file__).resolve().parents[1] / "models" / "Uni-LaViRA"),
            )
        ).resolve()
        # ETPNav is also named vlnce_baselines and is often earlier on PYTHONPATH.
        while str(root) in sys.path:
            sys.path.remove(str(root))
        sys.path.insert(0, str(root))
        loaded = sys.modules.get("vlnce_baselines")
        loaded_file = str(getattr(loaded, "__file__", "") or "")
        if loaded is not None and str(root) not in loaded_file:
            for key in list(sys.modules):
                if key == "vlnce_baselines" or key.startswith("vlnce_baselines."):
                    del sys.modules[key]
        pose_utils = root / "habitat_extensions" / "pose_utils.py"
        if pose_utils.is_file() and "habitat_extensions.pose_utils" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "habitat_extensions.pose_utils", pose_utils
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["habitat_extensions.pose_utils"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        from vlnce_baselines.agent import VLMReasoningAgent

        self.agent = VLMReasoningAgent(_SilentViz(), task_type="VLN")
        self.actor_steps = 0
        self._instruction: str | None = None

    def begin_episode(self, position: Sequence[float], heading: float, state: Any = None) -> None:
        del position, heading, state
        self.actor_steps = 0
        self._instruction = None
        self.agent.todo_list = None
        self.agent.guideline = ""

    def reset_episode(self, position: Sequence[float], heading: float) -> None:
        self.begin_episode(position, heading)

    def save_graphmap(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(
            json.dumps({"nav_model": self.nav_name, "actor_steps": int(self.actor_steps)}) + "\n",
            encoding="utf-8",
        )

    def _views(self, env: Any) -> tuple[Image.Image, list[dict]]:
        obs = env.sim.get_sensor_observations()
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)[..., :3]
        depth = np.asarray(obs["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        views = [
            {"rgb": rgb, "depth": depth, "angle": angle, "step": self.actor_steps}
            for angle in (0, 90, 180, 270)
        ]
        return Image.fromarray(rgb).convert("RGB"), views

    def observe_and_act(
        self,
        env: Any,
        instruction: str,
        position: Sequence[float],
        heading: float,
        *,
        act: bool,
    ) -> tuple[int | None, float | None, float | None]:
        del position, heading
        self._instruction = instruction
        if not act:
            return None, None, None
        frame, views = self._views(env)
        visited = [{
            "panorama_frames": views,
            "step": self.actor_steps,
            "memory_role": None,
        }]
        history = [{"step": self.actor_steps, "image": frame}]
        decision = self.agent.navigate_or_backtrack(
            instruction,
            visited,
            episode_id="ep",
            step=self.actor_steps,
            history_images=history,
        )
        self.actor_steps += 1
        return _action_id(str(decision)), None, None
