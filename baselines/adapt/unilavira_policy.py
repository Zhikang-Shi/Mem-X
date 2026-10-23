"""Uni-LaViRA navigator. One VLM decision becomes one Habitat action."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import quaternion
from PIL import Image


class _SilentViz:
    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            return None

        return _noop


def _action_id(text: Any) -> int:
    if isinstance(text, Mapping):
        stop = text.get("stop_signal", text.get("stop", False))
        if stop is True or (isinstance(stop, str) and stop.lower() == "true"):
            return 0
        action = str(text.get("action", "")).strip().lower()
        if action == "stop":
            return 0
        if action == "backtrack":
            raise NotImplementedError("Uni-LaViRA waypoint backtracking requires a waypoint controller")
        text = text.get("direction", action)
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
        state = env.sim.get_agent_state()
        views = []
        for angle in (0, 90, 180, 270):
            rotation = quaternion.from_rotation_vector(
                [0.0, float(np.deg2rad(angle)), 0.0]
            ) * state.rotation
            obs = env.sim.get_observations_at(
                position=state.position, rotation=rotation,
                keep_agent_at_new_pose=False,
            )
            if obs is None:
                raise RuntimeError(f"Could not render Uni-LaViRA panorama at {angle} degrees")
            rgb = np.asarray(obs["rgb"], dtype=np.uint8)[..., :3].copy()
            depth = np.asarray(obs["depth"], dtype=np.float32).copy()
            if depth.ndim == 3:
                depth = depth[..., 0]
            views.append({"rgb": rgb, "depth": depth, "angle": angle, "step": self.actor_steps})
        return Image.fromarray(views[0]["rgb"]).convert("RGB"), views

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
        action = _action_id(decision)
        print(f"[unilavira] step={self.actor_steps} action={action}", flush=True)
        return action, None, None
