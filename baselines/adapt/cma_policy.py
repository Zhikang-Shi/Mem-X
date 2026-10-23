"""CMA navigator. Loads CMAPolicy without importing CMA's trainer package."""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

WORD_RE = re.compile(r"[A-Za-z0-9]+|[^\s]")
CMA_RGB = 224
CMA_DEPTH = 256
MAX_DEPTH_M = 10.0


def _tokenize(text: str) -> list[str]:
    tokens = []
    for token in WORD_RE.findall(text.lower()):
        if token == ",":
            continue
        if token in {".", "?", "!", "'", '"', "(", ")", "-", "/"} or token[:1].isalnum():
            tokens.append(token)
            continue
        tokens.extend(char for char in token if char != ",")
    return tokens


def _tokens_to_ids(text: str, stoi: Mapping[str, int], pad_to: int = 200) -> np.ndarray:
    unk = int(stoi.get("<unk>", 1))
    ids = [int(stoi.get(token, unk)) for token in _tokenize(text)]
    if len(ids) < pad_to:
        ids.extend([0] * (pad_to - len(ids)))
    return np.asarray(ids[:pad_to], dtype=np.int64)


def _load_stoi(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    vocab = payload["instruction_vocab"]
    if "stoi" in vocab:
        return dict(vocab["stoi"])
    return {word: index for index, word in enumerate(vocab["word_list"])}


def _validate_vocab(stoi: Mapping[str, int], vocab_size: int) -> None:
    if set(stoi.values()) != set(range(vocab_size)):
        raise ValueError(
            f"CMA vocabulary does not match its {vocab_size}-row embedding table; "
            "use R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
        )


def _resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _resize_depth(depth: np.ndarray, size: int) -> np.ndarray:
    import cv2

    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    # The shared Habitat runner explicitly returns depth in meters, including
    # frames whose farthest pixel is closer than 1.5 m.
    image = np.clip(image / MAX_DEPTH_M, 0.0, 1.0)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32)[..., None]


def _use_cma_package(root: Path) -> None:
    """Point vlnce_baselines at CMA without running its trainer imports."""
    pkg = sys.modules.get("vlnce_baselines")
    if pkg is not None and Path(getattr(pkg, "__file__", "") or "").is_file():
        return
    package = types.ModuleType("vlnce_baselines")
    package.__path__ = [str(root / "vlnce_baselines")]
    package.__package__ = "vlnce_baselines"
    sys.modules["vlnce_baselines"] = package


class CMADiscretePolicy:
    discrete_actions = True
    nav_name = "cma"

    def __init__(self, args: Any) -> None:
        import torch
        from gym.spaces import Box, Dict, Discrete
        from habitat_baselines.utils.common import batch_obs

        root = Path(args.cma_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _use_cma_package(root)
        from vlnce_baselines.config.default import get_config
        from vlnce_baselines.models.cma_policy import CMAPolicy

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.batch_obs = batch_obs
        self.stoi = _load_stoi(Path(args.r2r_train))
        observation_space = Dict({
            "rgb": Box(low=0, high=255, shape=(CMA_RGB, CMA_RGB, 3), dtype=np.uint8),
            "depth": Box(low=0.0, high=1.0, shape=(CMA_DEPTH, CMA_DEPTH, 1), dtype=np.float32),
            "instruction": Box(low=0, high=2504, shape=(200,), dtype=np.int64),
        })
        action_space = Discrete(4)
        cwd = os.getcwd()
        os.chdir(root)
        try:
            config = get_config(str(Path(args.exp_config).resolve()))
            ddppo = Path(args.etpnav_root).resolve() / "data/ddppo-models/gibson-2plus-resnet50.pth"
            config.defrost()
            config.MODEL.DEPTH_ENCODER.ddppo_checkpoint = str(ddppo)
            # The evaluation checkpoint contains the complete embedding table.
            # Initialize its shape locally instead of requiring a second GloVe file.
            config.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = False
            config.freeze()
            self.policy = CMAPolicy.from_config(
                config, observation_space, action_space
            ).to(self.device)
        finally:
            os.chdir(cwd)
        ckpt = torch.load(str(args.ckpt), map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        cleaned = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
        self.policy.load_state_dict(cleaned, strict=True)
        vocab_size = self.policy.net.instruction_encoder.embedding_layer.num_embeddings
        _validate_vocab(self.stoi, vocab_size)
        print(f"[cma] loaded full checkpoint: {args.ckpt}", flush=True)
        self.policy.eval()
        hidden = int(config.MODEL.STATE_ENCODER.hidden_size)
        layers = int(self.policy.net.num_recurrent_layers)
        self.rnn_states = torch.zeros(1, layers, hidden, device=self.device)
        self.prev_actions = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.not_done = torch.zeros(1, 1, device=self.device, dtype=torch.uint8)
        self.instruction_tokens = _tokens_to_ids("", self.stoi)
        self.actor_steps = 0
        self._instruction: str | None = None

    def begin_episode(self, position: Sequence[float], heading: float, state: Any = None) -> None:
        del position, heading, state
        self._instruction = None
        self.reset("")

    def reset_episode(self, position: Sequence[float], heading: float) -> None:
        self.begin_episode(position, heading)

    def reset(self, instruction: str) -> None:
        self.instruction_tokens = _tokens_to_ids(instruction or "", self.stoi)
        self.rnn_states.zero_()
        self.prev_actions.zero_()
        self.not_done.zero_()
        self.actor_steps = 0

    def set_instruction(self, instruction: str) -> None:
        self.instruction_tokens = _tokens_to_ids(instruction or "", self.stoi)

    def save_graphmap(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(
            json.dumps({"nav_model": self.nav_name, "actor_steps": int(self.actor_steps)}) + "\n",
            encoding="utf-8",
        )

    def _forward(self, observations: Mapping[str, Any]):
        packed = {
            "rgb": _resize_rgb(observations["rgb"], CMA_RGB),
            "depth": _resize_depth(observations["depth"], CMA_DEPTH),
            "instruction": self.instruction_tokens,
        }
        batch = self.batch_obs([packed], device=self.device)
        with self.torch.no_grad():
            actions, self.rnn_states = self.policy.act(
                batch, self.rnn_states, self.prev_actions, self.not_done, deterministic=True
            )
        self.not_done.fill_(1)
        return actions

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
        text = str(instruction or "")
        if text != (self._instruction or "") or int(np.count_nonzero(self.instruction_tokens)) == 0:
            if self.actor_steps == 0:
                self.reset(text)
            else:
                self.set_instruction(text)
            self._instruction = text
        if int(np.count_nonzero(self.instruction_tokens)) == 0:
            raise RuntimeError(f"CMA received an empty instruction: {text!r}")
        if not act:
            return None, None, None
        observations = env.sim.get_sensor_observations()
        actions = self._forward(observations)
        self.prev_actions.copy_(actions)
        self.actor_steps += 1
        return int(actions[0][0].item()), None, None
