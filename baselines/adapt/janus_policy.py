"""JanusVLN navigator with the same calls ETPNavSuffixPolicy exposes.

ETPNav returns a waypoint. Janus returns one Habitat action. The MemBridge
loop checks ``discrete_actions`` and does not turn that action into a stop.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

NUM_HISTORY = 8
ACTIONS = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}


def _pose_key(position: Sequence[float], heading: float) -> tuple[float, float, float, float]:
    xyz = [float(value) for value in position]
    return (round(xyz[0], 3), round(xyz[1], 3), round(xyz[2], 3), round(float(heading), 3))


def _sample_history(history: list[Image.Image], maximum: int) -> list[Image.Image]:
    last = len(history) - 1
    if last <= maximum:
        return list(history)
    indices = np.linspace(0, last, maximum + 1, dtype=int)
    return [history[int(index)] for index in indices]


def _restore_gamma_parameters(model, pretrained: str) -> None:
    """Put VGGT LayerScale weights back after a transformers 4.46 load.

    That release renames every checkpoint key containing ``gamma`` to ``weight``.
    The 144 VGGT scales are then missing and stay at their init value.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors import safe_open

    index_path = Path(pretrained) / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    weight_map = json.loads(index_path.read_text())["weight_map"]
    params = dict(model.named_parameters())
    grouped: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if key.endswith(".gamma") and key in params:
            grouped.setdefault(filename, []).append(key)
    if not grouped:
        return
    with torch.no_grad():
        for filename, keys in grouped.items():
            with safe_open(str(Path(pretrained) / filename), framework="pt") as handle:
                for key in keys:
                    param = params[key]
                    value = handle.get_tensor(key)
                    param.copy_(value.to(device=param.device, dtype=param.dtype))


class JanusVLNRunner:
    """JanusVLN_Extra caller. Does not import evaluation.py, which needs a newer Habitat."""

    def __init__(self, pretrained: str, device: str = "cuda") -> None:
        import copy
        import torch
        from qwen_vl.model.configuration_qwen2_5_vl import Qwen2_5_VLConfig
        from qwen_vl.model.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGenerationForJanusVLN,
        )
        from qwen_vl.model.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.utils import is_flash_attn_2_available

        # lavira ships transformers 4.46, which has no qwen2_5_vl entry.
        # streamvln's 4.50 already has one, so register only when it is missing.
        if "qwen2_5_vl" not in CONFIG_MAPPING:
            AutoConfig.register("qwen2_5_vl", Qwen2_5_VLConfig)
            AutoProcessor.register(Qwen2_5_VLConfig, Qwen2_5_VLProcessor)

        self._torch = torch
        self._copy = copy
        vggt_device = torch.device(os.environ.get("JANUSVLN_VGGT_DEVICE", device))
        language_device = torch.device(os.environ.get("JANUSVLN_LANGUAGE_DEVICE", device))
        config = AutoConfig.from_pretrained(pretrained)
        last_error: Exception | None = None
        model = None
        # A failed flash-attn load leaves torch's default dtype at bfloat16.
        # lavira has no flash-attn, and torchvision then builds bfloat16 images.
        attn_order = (
            ("flash_attention_2", "sdpa", "eager")
            if is_flash_attn_2_available()
            else ("sdpa", "eager")
        )
        for attn in attn_order:
            try:
                model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
                    pretrained,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map={"": device},
                    attn_implementation=attn,
                    mode="evaluation",
                ).eval()
                break
            except Exception as exc:
                last_error = exc
                model = None
                torch.set_default_dtype(torch.float32)
        if model is None:
            raise RuntimeError(f"JanusVLN failed to load: {last_error}")
        from janus_attention import install_segmented_vision_sdpa

        install_segmented_vision_sdpa(model)
        if vggt_device != language_device:
            model.vggt.to(vggt_device)
            model.merger.to(vggt_device)
            model.visual.to(language_device)
            model.model.to(language_device)
            model.lm_head.to(language_device)
            if hasattr(model, "classifier"):
                model.classifier.to(language_device)
            # Upstream adds merger output to the language GPU's visual
            # embeddings. Move that output across the split-device boundary.
            model.merger.register_forward_hook(
                lambda _module, _inputs, output: output.to(language_device)
            )
            torch.cuda.empty_cache()
        _restore_gamma_parameters(model, pretrained)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=1605632,
            min_pixels=28 * 28,
            padding_side="left",
        )
        self.device = language_device
        self.vggt_device = vggt_device

    def reset_memory(self) -> None:
        self.model.past_key_values_vggt = None

    def call_model(self, observations, task: str, step_id: int):
        import torch
        from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
        from qwen_vl_utils import extract_vision_info

        del step_id
        message = [
            {
                "role": "system",
                "content": (
                    "You are a visual language navigation model, and your should go to the "
                    "locations to complete the given task. Compare the observation and "
                    "instruction to infer your current progress, and then select the correct "
                    "direction from the candidates to go to the target location and finish the task."
                ),
            }
        ]
        context = (
            "These images are your historical observations and your current observation.\n"
            f" Your task is to {task} \n You should take one of the following actions:\n"
            " MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        )
        visual = observations
        if isinstance(visual, Image.Image):
            message.append(
                {"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]}
            )
        else:
            image_content = [{"type": "image", "image": image} for image in visual]
            message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
        messages = [message]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        images_vggt = []
        image_inputs = []
        for item in messages:
            vision_info = extract_vision_info(item)
            current = []
            for index, ele in enumerate(vision_info):
                image = load_and_preprocess_images([ele["image"]])[0]
                if index == len(vision_info) - 1:
                    current.append(image)
                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image_inputs.append(image[:, :height, :width])
            images_vggt.append(torch.stack(current))
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        )
        inputs["images_vggt"] = [feat.to(self.vggt_device) for feat in images_vggt]
        inputs = inputs.to(self.device)
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
            max_new_tokens=24,
        )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class JanusDiscretePolicy:
    """One JanusVLN_Extra policy. Memory resets at every episode."""

    discrete_actions = True
    nav_name = "janus"

    def __init__(self, args: Any) -> None:
        janus_root = Path(args.janus_root).resolve()
        model_path = Path(args.janus_model).resolve()
        src = janus_root / "src"
        if not (src / "qwen_vl" / "model" / "modeling_qwen2_5_vl.py").is_file():
            raise FileNotFoundError(f"Janus model code missing under {src}")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        language_device = os.environ.get("JANUSVLN_LANGUAGE_DEVICE", "cuda:1")
        self.model = JanusVLNRunner(str(model_path), device=language_device)
        self.num_history = int(getattr(args, "num_history", NUM_HISTORY))
        self.actor_steps = 0
        self.history: list[Image.Image] = []
        self._key: tuple[float, float, float, float] | None = None

    def begin_episode(self, position: Sequence[float], heading: float, state: Any = None) -> None:
        del position, heading, state
        self.model.reset_memory()
        self.history = []
        self._key = None
        self.actor_steps = 0

    def reset_episode(self, position: Sequence[float], heading: float) -> None:
        self.begin_episode(position, heading)

    def save_graphmap(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nav_model": self.nav_name, "actor_steps": int(self.actor_steps)}
        note = path.with_suffix(".json")
        note.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def _current_image(self, env: Any) -> Image.Image:
        observations = env.sim.get_sensor_observations()
        rgb = np.asarray(observations["rgb"], dtype=np.uint8)[..., :3]
        return Image.fromarray(rgb).convert("RGB")

    def observe_and_act(
        self,
        env: Any,
        instruction: str,
        position: Sequence[float],
        heading: float,
        *,
        act: bool,
    ) -> tuple[int | None, float | None, float | None]:
        key = _pose_key(position, heading)
        if key != self._key:
            self.history.append(self._current_image(env))
            self._key = key
        if not act:
            return None, None, None
        self.actor_steps += 1
        images = _sample_history(self.history, self.num_history)
        answer = str(self.model.call_model(images, instruction, self.actor_steps)[0]).strip().upper()
        action = ACTIONS.get(answer, ACTIONS["STOP"])
        if self.actor_steps % 25 == 0:
            print(f"[janus] step={self.actor_steps} action={answer} history={len(images)}", flush=True)
        return int(action), None, None


def build_policy(args: Any, etp_policy_cls: type):
    name = getattr(args, "nav_model", "etpnav")
    if name == "etpnav":
        return etp_policy_cls(args)
    if name == "janus":
        return JanusDiscretePolicy(args)
    baselines = Path(__file__).resolve().parents[1]
    if name == "cma":
        sys.path.insert(0, str(baselines / "adapt"))
        from cma_policy import CMADiscretePolicy

        return CMADiscretePolicy(args)
    if name == "unilavira":
        sys.path.insert(0, str(baselines / "adapt"))
        from unilavira_policy import UniLaViRADiscretePolicy

        return UniLaViRADiscretePolicy(args)
    raise RuntimeError(f"unknown nav model: {name}")
