"""Load the self-contained semantic builder source used by Habitat runtime."""

import importlib.util
import os
import re
import sys
import types
from dataclasses import fields, is_dataclass


_DEFAULT_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "semantic_scene_graph",
    "legacy_builder_source",
)
_SOURCE_DIR = os.environ.get(
    "MEMORY_MODULE_SEMANTIC_SOURCE_DIR", _DEFAULT_SOURCE_DIR
)
_BUILDER_PATH = os.path.join(_SOURCE_DIR, "builder.py")

if not os.path.exists(_BUILDER_PATH):
    raise FileNotFoundError(
        "memory_module semantic builder.py not found at "
        f"{_BUILDER_PATH}"
    )

if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

with open(_BUILDER_PATH, "r", encoding="utf-8") as _f:
    _source = _f.read()

if "from __future__ import annotations" not in _source[:500]:
    _source = "from __future__ import annotations\n" + _source

_sam3_root = os.environ.get("SAM3_SOURCE_ROOT")
if _sam3_root:
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not getattr(finder.__class__, "__module__", "").startswith("__editable___sam3")
    ]
    if _sam3_root not in sys.path:
        sys.path.insert(0, _sam3_root)
    _source = _source.replace(
        'SAM3_SOURCE_ROOT = "/data2/shizk/sam3"',
        f'SAM3_SOURCE_ROOT = {repr(_sam3_root)}',
    )

_source = _source.replace(
    "from ram.models import ram_plus\nfrom ram import inference_ram as inference\nfrom ram import get_transform",
    "try:\n"
    "    from ram.models import ram_plus\n"
    "    from ram import inference_ram as inference\n"
    "    from ram import get_transform\n"
    "except Exception as _ram_import_error:\n"
    "    ram_plus = None\n"
    "    inference = None\n"
    "    get_transform = None\n"
    "    print(f\"[INHerit-SG] RAM++ import disabled: {_ram_import_error}\")",
)

_source = _source.replace(
    "                self.floor_tracker.update(current_pose)\n"
    "                self._online_mapping_and_room_step(i, rgb_path, depth_path, current_pose)",
    "                curr_room_node = self.room_topology_cache.get(self.current_room_id, {}).get(\"room_node_id\")\n"
    "                is_floor_switch = self.floor_tracker.update(current_pose, curr_room_node)\n"
    "                if is_floor_switch:\n"
    "                    print(f\"\\n>>> Floor Switch Detected! <<<\")\n"
    "                    global_update_start = time.time()\n"
    "                    self._perform_global_update(current_pose, i)\n"
    "                    print(f\"  [Global Update] Completed in {time.time() - global_update_start:.2f}s\")\n"
    "                    floor_start = time.time()\n"
    "                    self.floor_tracker.finalize_floor()\n"
    "                    print(f\"  [Floor Tracker] Finalized Floor {self.floor_tracker.current_floor_id} in {time.time() - floor_start:.2f}s\")\n"
    "                    switch_start = time.time()\n"
    "                    new_floor_id = self.floor_tracker.current_floor_id\n"
    "                    self.map_manager.switch_floor(new_floor_id)\n"
    "                    self.room_segmenter.switch_floor(new_floor_id)\n"
    "                    self.current_room_id = 0\n"
    "                    self.room_segmenter.current_room_id = 0\n"
    "                    print(f\">>> Switched Context to Floor {new_floor_id}\")\n"
    "                    print(f\"  [Floor Switch] Completed in {time.time() - switch_start:.2f}s\")\n"
    "                self._online_mapping_and_room_step(i, rgb_path, depth_path, current_pose)",
)


def _patch_streaming_floor_switch(match):
    indent = match.group("indent")
    lines = [
        f"{indent}curr_room_node = self.room_topology_cache.get(self.current_room_id, {{}}).get(\"room_node_id\")",
        f"{indent}is_floor_switch = self.floor_tracker.update(current_pose, curr_room_node)",
        f"{indent}if is_floor_switch:",
        f"{indent}    print(f\"\\n>>> Floor Switch Detected! <<<\")",
        f"{indent}    global_update_start = time.time()",
        f"{indent}    self._perform_global_update(current_pose, i)",
        f"{indent}    print(f\"  [Global Update] Completed in {{time.time() - global_update_start:.2f}}s\")",
        f"{indent}    floor_start = time.time()",
        f"{indent}    self.floor_tracker.finalize_floor()",
        f"{indent}    print(f\"  [Floor Tracker] Finalized Floor {{self.floor_tracker.current_floor_id}} in {{time.time() - floor_start:.2f}}s\")",
        f"{indent}    switch_start = time.time()",
        f"{indent}    new_floor_id = self.floor_tracker.current_floor_id",
        f"{indent}    self.map_manager.switch_floor(new_floor_id)",
        f"{indent}    self.room_segmenter.switch_floor(new_floor_id)",
        f"{indent}    self.current_room_id = 0",
        f"{indent}    self.room_segmenter.current_room_id = 0",
        f"{indent}    print(f\">>> Switched Context to Floor {{new_floor_id}}\")",
        f"{indent}    print(f\"  [Floor Switch] Completed in {{time.time() - switch_start:.2f}}s\")",
        f"{indent}self._online_mapping_and_room_step(i, rgb_path, depth_path, current_pose)",
    ]
    return "\n".join(lines)


_source, _floor_switch_patch_count = re.subn(
    r"(?m)^(?P<indent>[ \t]*)self\.floor_tracker\.update\(current_pose\)\n"
    r"(?P=indent)self\._online_mapping_and_room_step\(i, rgb_path, depth_path, current_pose\)",
    _patch_streaming_floor_switch,
    _source,
    count=1,
)
if (
    _floor_switch_patch_count != 1
    and "is_floor_switch = self.floor_tracker.update(current_pose, curr_room_node)" not in _source
):
    raise RuntimeError("Failed to patch INHerit-SG streaming floor-switch handling")


def _start_frame_observation_capture(match):
    indent = match.group("indent")
    return match.group(0) + f"{indent}    self._inherit_sg_frame_observations = []\n"


_source, _observation_capture_init_count = re.subn(
    r"(?m)^(?P<indent>[ \t]*)def _process_frame_results\([^\n]*\):\n",
    _start_frame_observation_capture,
    _source,
    count=1,
)

_current_frame_append = (
    '                current_frame_nodes.append({"node_id": final_node_id, '
    '"bbox": curr_obj[\'raw_obj\'][\'bounding_box_xyxy\']})'
)
_observation_capture_append = _current_frame_append + "\n" + (
    "                self._inherit_sg_frame_observations.append({\n"
    "                    \"node_id\": final_node_id,\n"
    "                    \"bbox\": curr_obj['raw_obj']['bounding_box_xyxy'],\n"
    "                    \"rgb_path\": current_rgb_path,\n"
    "                    \"object_position\": curr_obj.get('pos_3d'),\n"
    "                })"
)
_observation_capture_append_count = _source.count(_current_frame_append)
if _observation_capture_append_count == 1:
    _source = _source.replace(_current_frame_append, _observation_capture_append, 1)

if _observation_capture_init_count != 1 or _observation_capture_append_count != 1:
    raise RuntimeError(
        "Failed to patch INHerit-SG per-frame object observation capture: "
        f"init={_observation_capture_init_count}, append={_observation_capture_append_count}"
    )

_module = types.ModuleType("_external_inherit_sg_builder")
_module.__file__ = _BUILDER_PATH
_module.__package__ = ""
exec(compile(_source, _BUILDER_PATH, "exec"), _module.__dict__)


def _install_sam3_collator_compat():
    """Make SAM3's API collator work when dataclass annotations are strings."""
    try:
        import torch
        from sam3.train.data import collator as sam3_collator
    except Exception:
        return

    if getattr(sam3_collator, "_inherit_sg_convert_patched", False):
        return

    def convert_my_tensors_compat(obj):
        for field in fields(obj):
            value = getattr(obj, field.name)
            if is_dataclass(value):
                convert_my_tensors_compat(value)
                continue
            type_attr = field.name + "__type"
            if value is None or not hasattr(obj, type_attr):
                continue

            dtype = getattr(obj, type_attr)
            if isinstance(value, torch.Tensor):
                setattr(obj, field.name, value.to(dtype=dtype))
            elif isinstance(value, list) and len(value) and isinstance(value[0], torch.Tensor):
                stack_dim = 1 if field.name in ["input_boxes", "input_boxes_label"] else 0
                setattr(obj, field.name, torch.stack(value, dim=stack_dim).to(dtype))
            else:
                setattr(obj, field.name, torch.as_tensor(value, dtype=dtype))
        return obj

    sam3_collator.convert_my_tensors = convert_my_tensors_compat
    sam3_collator._inherit_sg_convert_patched = True


_install_sam3_collator_compat()

TrajectoryPipeline = _module.TrajectoryPipeline
Sam3LocalClient = _module.Sam3LocalClient
load_feature_extractor = _module.load_feature_extractor
extract_feature = _module.extract_feature
