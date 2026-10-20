"""Semantic scene-graph construction.

The caller supplies the API endpoint and key. This module does not store either value.
"""

import os
import sys
import json
import yaml
import base64
import time
import cv2
import numpy as np
import torch
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from datetime import datetime

class Logger:
    def __init__(self, log_file_path: str):
        self.terminal = sys.__stdout__ 
        
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
      
    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def __del__(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

def read_pose_matrix(pose_path: str) -> np.ndarray:
    """Read a 4x4 camera-to-world matrix."""
    try:
        with open(pose_path, 'r') as f:
            content = f.read().strip().split()
            values = [float(x) for x in content]
            return np.array(values).reshape(4, 4)
    except Exception as e:
        print(f"Error reading pose: {e}")
        return np.eye(4)

def pixel_to_world(u: int, v: int, d: float, pose: np.ndarray, intrinsics: Dict) -> List[float]:
    """Convert a pixel to world coordinates and correct the OpenCV frame into the HM3D frame."""
    if d <= 0: return None
    
    z_cam = d
    x_cam = (u - intrinsics['cx']) * z_cam / intrinsics['fx']
    y_cam = (v - intrinsics['cy']) * z_cam / intrinsics['fy']
    
    p_cam_cv = np.array([x_cam, y_cam, z_cam, 1.0])
    
    T_fix = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    p_cam_habitat = T_fix @ p_cam_cv
    
    p_world = pose @ p_cam_habitat
    
    return p_world[:3].tolist()



def extract_area_base_id(node_id):
    """Return the area id without its timestamp suffix."""
    match = re.match(r'(area_\d+_\d+)_\d+', node_id)
    if match:
        return match.group(1)
    return node_id

def process_nodes_file(input_path, output_path=None):
    """Deduplicate nodes.jsonl. Rooms, floors, and objects use the full id. Areas ignore the timestamp suffix."""
    if output_path is None:
        output_path = input_path.replace('.jsonl', '_final.jsonl')
    
    nodes_by_base_id = defaultdict(list)
    
    print(f"Reading nodes from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                node = json.loads(line)
                node_id = node.get('node_id')
                
                if not node_id:
                    print(f"Warning: Line {line_num} missing node_id")
                    continue
                
                base_id = extract_area_base_id(node_id)
                nodes_by_base_id[base_id].append(node)
                
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} - {e}")
                continue
    
    total_lines = sum(len(versions) for versions in nodes_by_base_id.values())
    print(f"\n=== Original Data ===")
    print(f"Total lines: {total_lines}")
    print(f"Unique base IDs: {len(nodes_by_base_id)}")
    
    duplicates = {k: v for k, v in nodes_by_base_id.items() if len(v) > 1}
    if duplicates:
        print(f"\nNodes with multiple versions: {len(duplicates)}")
        
        by_type = defaultdict(list)
        for base_id, versions in duplicates.items():
            if base_id.startswith('area_'):
                by_type['Area'].append((base_id, len(versions)))
            elif base_id.startswith('room_'):
                by_type['Room'].append((base_id, len(versions)))
            elif base_id.startswith('floor_'):
                by_type['Floor'].append((base_id, len(versions)))
            else:
                by_type['Object'].append((base_id, len(versions)))
        
        for type_name, items in sorted(by_type.items()):
            print(f"\n  {type_name} duplicates: {len(items)}")
            for base_id, count in sorted(items, key=lambda x: -x[1])[:3]:
                print(f"    {base_id}: {count} versions")
    
    final_nodes = {}
    
    for base_id, versions in nodes_by_base_id.items():
        if len(versions) == 1:
            final_nodes[base_id] = versions[0]
        else:
            sorted_versions = sorted(
                versions,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            
            final_node = sorted_versions[0]
            final_nodes[base_id] = final_node
            
            if base_id.startswith('area_') and len(final_nodes) <= 5:
                print(f"\n'{base_id}' merged {len(versions)} versions:")
                for v in versions:
                    print(f"  - {v['node_id']} ({v.get('timestamp', 'N/A')})")
                print(f"  ✅ Kept: {final_node['node_id']}")
    
    by_level = defaultdict(list)
    for node in final_nodes.values():
        level = node.get('node_level', -1)
        by_level[level].append(node)
    
    print(f"\n=== Writing Results ===")
    with open(output_path, 'w', encoding='utf-8') as f:
        for level in sorted(by_level.keys()):
            nodes = sorted(by_level[level], key=lambda x: x.get('node_id', ''))
            for node in nodes:
                f.write(json.dumps(node, ensure_ascii=False) + '\n')
    
    print(f"\n=== Final Statistics ===")
    level_names = {0: 'Floors', 1: 'Rooms', 2: 'Areas', 3: 'Objects'}
    for level in sorted(by_level.keys()):
        count = len(by_level[level])
        name = level_names.get(level, f'Level-{level}')
        print(f"  {name}: {count}")
    
    reduction = total_lines - len(final_nodes)
    print(f"\nRemoved {reduction} duplicates ({reduction/total_lines*100:.1f}%)")
    print(f"Output: {output_path}")
    
    return final_nodes


def analyze_area_quality(nodes_dict):
    """Report area clustering quality."""
    areas = [n for n in nodes_dict.values() if n.get('node_level') == 2]
    
    if not areas:
        return
    
    print(f"\n=== Area Quality Analysis ===")
    
    obj_counts = [len(a.get('contained_obj_nodes', [])) for a in areas]
    
    print(f"Total areas: {len(areas)}")
    print(f"Objects per area: min={min(obj_counts)}, max={max(obj_counts)}, avg={sum(obj_counts)/len(obj_counts):.1f}")
    
    singles = [a for a in areas if len(a.get('contained_obj_nodes', [])) == 1]
    if singles:
        pct = len(singles) / len(areas) * 100
        print(f"\n⚠️  Single-object areas: {len(singles)} ({pct:.1f}%)")
        if pct > 30:
            print(f"    Suggestion: Increase DIST_THRESH from 0.8 to 1.2-1.5 meters")
    
    by_room = defaultdict(list)
    for a in areas:
        room = a.get('parent_room_id', 'unknown')
        by_room[room].append(a)
    
    print(f"\nAreas per room:")
    for room, areas_list in sorted(by_room.items()):
        avg_obj = sum(len(a.get('contained_obj_nodes', [])) for a in areas_list) / len(areas_list)
        print(f"  {room}: {len(areas_list)} areas, avg {avg_obj:.1f} objects/area")




class LLMHandler:
    def __init__(self, llm_config: Dict[str, Any]):
        self.config = llm_config
        self.mode = llm_config.get('deployment_type', 'api') 
        self.clients = {}      
        self.model_names = {}  
        
        self.spatial_config = llm_config.get('api_settings', {})
        self.spatial_method = "vlm"
        
        print(f"\n--- Initializing LLM Handler (Mode: {self.mode.upper()}) ---")
        if self.mode == 'local':
            local_cfg = llm_config.get('local_settings', {})
            endpoints = local_cfg.get('endpoints', {})
            self.routing = local_cfg.get('routing', {})
            for name, cfg in endpoints.items():
                print(f"  - Loading Local Endpoint: {name} -> {cfg['model_name']}")
                self.clients[name] = OpenAI(api_key=cfg.get('api_key', "EMPTY"), base_url=cfg['api_base_url'])
                self.model_names[name] = cfg['model_name']
        else:
            api_cfg = llm_config.get('api_settings', {})
            print(f"  - Loading API Endpoint: {api_cfg['vision_model']}")
            client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['api_base_url'])
            self.clients['default'] = client
            self.api_vision_model = api_cfg['vision_model']
            self.api_text_model = api_cfg['text_model']
            self.temperature = api_cfg.get('temperature', 0.0)
        self.token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    def _update_token_stats(self, response):
        """Accumulate token usage from an API response."""
        try:
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.token_stats["prompt_tokens"] += usage.prompt_tokens
                self.token_stats["completion_tokens"] += usage.completion_tokens
                self.token_stats["total_tokens"] += usage.total_tokens
                
                # print(f"   [Token Cost] In: {usage.prompt_tokens}, Out: {usage.completion_tokens}")
        except Exception as e:
            print(f"[Warning] Failed to track tokens: {e}")

    def get_token_report(self):
        return self.token_stats
    def _get_client_and_model(self, task_type: str):
        vision_tasks = [
            'leaf_description', 
            'spatial_reasoning', 
            'cluster_summary', 
            'object_expansion', 
            'verification',
            'map_decision'
        ]
        
        text_tasks = [
            'tag_filtering',
            'query_parsing'  
        ]
        
        if self.mode == 'local':
            endpoint_name = self.routing.get(task_type)
            
            if not endpoint_name and task_type in vision_tasks:
                endpoint_name = self.routing.get('spatial_reasoning', self.routing.get('leaf_description'))
            
            if not endpoint_name and task_type in text_tasks:
                endpoint_name = 'primary'

            if not endpoint_name:
                endpoint_name = 'primary'
            
            if endpoint_name not in self.clients: 
                endpoint_name = list(self.clients.keys())[0]
                
            return (self.clients[endpoint_name], self.model_names[endpoint_name], 0.1)
        else:
            if task_type in vision_tasks:
                model_name = self.api_vision_model
            else:
                model_name = self.api_text_model
                
            if task_type == 'cluster_summary': temp = 0.3
            elif task_type == 'map_decision': temp = 0.1
            elif task_type == 'tag_filtering': temp = 0.0
            elif task_type == 'query_parsing': temp = 0.0
            else: temp = self.temperature
            
            return self.clients['default'], model_name, temp
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def filter_object_tags(self, raw_tags: List[str]) -> List[str]:
        """Keep concrete object nouns from the RAM++ tags."""
        if not raw_tags:
            return []

        client, model_name, temp = self._get_client_and_model('tag_filtering')
        
        tags_str = ", ".join(raw_tags)
        
        prompt_text = (
            f"Task: Filter image tags for object detection.\n"
            f"Input List: [{tags_str}]\n"
            f"Requirements:\n"
            f"1. KEEP specific physical objects (e.g., 'chair', 'sofa', 'lamp').\n"
            f"2. REMOVE scene/room names (e.g., 'living room', 'kitchen', 'bedroom', 'room', 'floor', 'ceiling', 'wall').\n"
            f"3. REMOVE actions, states, or lighting (e.g., 'burn', 'shiny', 'cluttered', 'bright').\n"
            f"4. REMOVE building structures unless removable (remove 'beam', 'pillar', 'hardwood').\n"
            f"5. MERGE synonyms (e.g., 'couch' & 'sofa' -> keep 'sofa'; 'appliance' & 'home appliance' -> keep 'appliance';'wood floor' & 'floor' -> remove).\n"
            f"6. Output strictly a JSON list of strings.\n"
            f"Example Input: ['living room', 'chair', 'seat', 'bright', 'floor','white']\n"
            f"Example Output: [\"chair\"]\n"
        )

        try:
            messages = [{"role": "user", "content": prompt_text}]
            
            response = client.chat.completions.create(
                model=model_name, 
                messages=messages, 
                max_tokens=200, 
                temperature=temp
            )
            
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            
            if not content.startswith("["): content = "[" + content
            if not content.endswith("]"): content = content + "]"
            
            filtered_tags = json.loads(content)
            
            final_tags = [str(t).lower() for t in filtered_tags if isinstance(t, str) and t.strip()]
            
            print(f"  [LLM Filter] {len(raw_tags)} -> {len(final_tags)} tags. Removed: {set(raw_tags) - set(final_tags)}")
            return final_tags

        except Exception as e:
            print(f"  [Error] Tag filtering failed: {e}. Using raw unique tags.")
            return list(set(raw_tags))
    
    def generate_leaf_node_semantics(self, image_path: str, tag: str, spatial_info: str = "") -> str:
        client, model_name, temp = self._get_client_and_model('leaf_description')
        base64_image = self._encode_image(image_path)
        prompt_text = f"任务：简要描述物体。\n物体：{tag}\n位置：{spatial_info}\n要求：包含颜色、材质、状态。50字以, 必须使用英文。"
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], max_tokens=100, temperature=temp)
            self._update_token_stats(response)
            return response.choices[0].message.content.strip()
        except Exception as e: return f"描述生成失败: {tag}"

    def detect_spatial_relationships(self, image_path: str, object_list: List[str], existing_relations: List[str], 
                                     object_positions: Dict[str, List[float]] = None, method: str = 'vlm') -> List[List[Any]]:
        """Select VLM or geometric spatial relations."""
        if method == 'geometric':
            if object_positions is None:
                print("[LLMHandler] Warning: Geometric mode selected but no positions provided.")
                return []
            return self._geometric_spatial_reasoning(object_list, object_positions)

        client, model_name, temp = self._get_client_and_model('spatial_reasoning')
        base64_image = self._encode_image(image_path)
        obj_str = ", ".join(object_list)
        history_str = "; ".join(existing_relations) if existing_relations else "None"
        
        prompt_text = f"任务：分析物理空间关系。\n物体列表：[{obj_str}]\n已知关系：[{history_str}]\n要求：\n1. 输出 JSON 列表: [Source, Target, Relation, Weight(0.0-1.0)], 且必须是列表的列表形式。\n2. 只允许使用以下关系词: on, under, next_to, inside, near, beside, above, below, left_of, right_of.\n3. 权重: >0.8(接触/支撑), 0.6-0.8(紧邻), <0.6(松散).\n4. 尽量全面, 但是不能强加关系, 如果距离较远的物体则不要添加关系; 重点关注标签不是object的物品, 但是标签是object的物体也要适当关注。"
        
        try:
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], max_tokens=600, temperature=temp)
            self._update_token_stats(response)
            content = response.choices[0].message.content.strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[0].strip()
            return json.loads(content)
        except Exception: return []
        
    def _geometric_spatial_reasoning(self, object_list: List[str], object_positions: Dict[str, np.ndarray]) -> List[List[Any]]:
        """Compute spatial relations from 3D centroids."""
        relationships = []
        if not object_positions:
            return relationships

        VERTICAL_THRESH = 0.15
        HORIZONTAL_ON_THRESH = 0.5
        NEAR_THRESH = 1.5

        ids = [oid for oid in object_list if oid in object_positions]
        
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j: continue
                
                u, v = ids[i], ids[j]
                pos_u = np.array(object_positions[u])
                pos_v = np.array(object_positions[v])
                
                diff = pos_u - pos_v
                dx, dy, dz = diff[0], diff[1], diff[2]
                
                dist_xz = np.sqrt(dx**2 + dz**2)
                dist_3d = np.linalg.norm(diff)
                
                rel = None
                weight = 0.0
                
                if dist_xz < HORIZONTAL_ON_THRESH:
                    if dy > VERTICAL_THRESH:
                        rel = "on" if dy < 1.0 else "above"
                        weight = 0.9 if rel == "on" else 0.7
                    elif dy < -VERTICAL_THRESH:
                        rel = "under" if abs(dy) < 1.0 else "below"
                        weight = 0.9 if rel == "under" else 0.7
                
                if rel is None:
                    if dist_3d < NEAR_THRESH and abs(dy) < VERTICAL_THRESH:
                        rel = "next_to"
                        weight = max(0.5, 1.0 - (dist_3d / NEAR_THRESH))
                
                if rel:
                    relationships.append([u, v, rel, float(f"{weight:.2f}")])

        return relationships
        
    def generate_spatial_cluster_summary(self, object_list: List[Dict], region_image_path: str = None) -> Dict:
        client, model_name, temp = self._get_client_and_model('cluster_summary')
        objs_text = "\n".join([f"- {item['tag']}: {item.get('desc', '')}" for item in object_list])
        prompt_text = f"任务：推断区域功能。\n包含物体：\n{objs_text}\n要求：\n1. Area Tag (如 reading_corner, dining_area).(不是判断房间，而是功能区，是房间节点的子节点)\n2. Summary (详细描述布局和功能, 必须使用英文).\n输出 JSON: {{\"area_tag\": \"...\", \"summary\": \"...\"}}"
        messages = []
        if region_image_path and os.path.exists(region_image_path):
            base64_img = self._encode_image(region_image_path)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]}]
        else:
            messages = [{"role": "user", "content": prompt_text}]
        try:
            response = client.chat.completions.create(model=model_name, messages=messages, max_tokens=500, temperature=temp)
            self._update_token_stats(response)
            content = response.choices[0].message.content.strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[0].strip()
            return json.loads(content)
        except Exception: return {"area_tag": "unknown", "summary": "Failed."}

    def parse_retrieval_query(self, user_query: str) -> Dict:
        client, model_name, temp = self._get_client_and_model('query_parsing')
        
        prompt_text = f"""
    You are an expert semantic parser for an embodied agent. 
    Your task is to extract structured information from a user's natural language query for object retrieval.

    ### Input Query:
    "{user_query}"

    ### Output Schema:
    You must return a single valid JSON object with the following fields:
    {{
        "_reasoning": "Briefly explain your logic here. Identify the target, reference, and any location constraints.",
        "target_floor": int or null, // e.g., 0, 1, 2. Extract specific floor numbers if mentioned.
        "target_object": "string", // The main object to find (noun only, singular).
        "target_object_description": "string or null", // Adjectives describing the target (color, shape, material).
        "target_object_description_negated": boolean, // True if the user says "NOT the red one" or "non-plastic".
        "spatial_relation": "string or null", // Prepositions like "on", "under", "near", "next to", "inside".
        "reference_object": "string or null", // The object used as an anchor (e.g., "on the TABLE").
        "reference_object_negated": boolean, // True if the user says "NOT near the table".
        "reference_object_description": "string or null", // Adjectives describing the reference object.
        "reference_object_description_negated": boolean, // True if user says "near a NOT white table".
        "area_hint": "string or null", // Functional zone (e.g., 'dining area', 'lounge area'). 
        "room_hint": "string or null", // Architectural room name (e.g., 'bedroom', 'kitchen', 'staircase').
        "room_hint_negated": boolean // True if the user says "NOT in the kitchen".
    }}

    ### Guidelines:
    1. **Target vs Reference**: In "Find the cup on the table", 'cup' is the target, 'table' is the reference.
    2. **Room vs Area**: 'Room' = architectural (Bedroom, Kitchen); 'Area' = functional (Dining area, Sofa area).
    3. **Negation Logic**: 
    - **Room Negation**: "Not in the kitchen" -> room_hint_negated: true.
    - **Attribute Negation**: If the user says "Find the cup that is NOT red", extract "red" as the description and set description_negated to true. Do not put "not red" in the description string.

    
    ### Response (JSON only):
    """
        
        try:    
            response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt_text}], temperature=0.0)
            self._update_token_stats(response)
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            
            result = json.loads(content)
            print(result)
            query_lower = user_query.lower()
            if result.get('room_hint') and result.get('room_hint').lower() in query_lower:
                room_val = result['room_hint'].lower()
                pattern = rf"(not|except)\s+(in\s+)?(the\s+)?{re.escape(room_val)}"
                if re.search(pattern, query_lower):
                    result['room_hint_negated'] = True
            
            
            if 'target_object_description_negated' not in result:
                result['target_object_description_negated'] = False
            if 'reference_object_description_negated' not in result:
                result['reference_object_description_negated'] = False

            # weights = self._calculate_llm_weights(user_query, result) 

            return result, {}
        


        except Exception as e:
            import traceback
            print("\n" + "!"*50)
            print(f"【严重错误】无法连接或解析 LLM 响应")
            print(f"当前使用的模型名: {model_name}")
            print(f"当前 Client Base URL: {client.base_url}")
            print("详细报错信息如下：")
            print(traceback.format_exc())
            print("!"*50 + "\n")
            
            return {}, {} 
      
    
    def _extract_entities(self, user_query: str) -> Dict:
        """Extract entities, attributes, and relations. Negation is handled separately."""
        client, model_name, temp = self._get_client_and_model('query_parsing')
        prompt_text = f"""
You are a semantic parsing assistant for a robot. 
Parse the user's natural language query into a structured JSON object.

### Extraction Rules:
1. **Target & Reference Objects**: 
   - Extract the main noun in **singular** form (e.g., "apples" -> "apple", "cups" -> "cup").
   - Extract adjectives (color, shape, material) into the `_description` fields. Do NOT include words like "not", "no", "non-".
   - Example: "red wooden chair" -> object: "chair", description: "red wooden".
2. **Floor Mapping**: 
   - Convert floor mentions to integers: "ground floor"->0, "1st floor"->1, "2nd floor"->2, "basement"->-1.
   - If no floor is mentioned, set to null.
3. **Locations**: 
   - `room_hint`: Architectural rooms (e.g., kitchen, bedroom, hallway).
   - `area_hint`: Functional zones (e.g., dining area, lounge area, sofa area).

### Output JSON Schema:
{{
    "_reasoning": "Briefly explain extraction logic.",
    "target_floor": int or null,
    "target_object": "string (noun, singular)",
    "target_object_description": "string or null (adjectives)",
    "spatial_relation": "string or null (e.g., on, under, next to)",
    "reference_object": "string or null (anchor object)",
    "reference_object_description": "string or null",
    "area_hint": "string or null",
    "room_hint": "string or null"
}}

### Few-Shot Examples:

User: "Find the black remote control on the sofa in the living room."
Output:
{{
    "_reasoning": "Target is 'remote control' (black), located 'on' the 'sofa' in the 'living room'.",
    "target_floor": null,
    "target_object": "remote control",
    "target_object_description": "black",
    "spatial_relation": "on",
    "reference_object": "sofa",
    "reference_object_description": null,
    "area_hint": null,
    "room_hint": "living room"
}}

User: "Go to the second floor and find the white plastic cup next to the coffee machine."
Output:
{{
    "_reasoning": "Floor is 2nd (2). Target is 'cup' (white plastic). Relation is 'next to' 'coffee machine'.",
    "target_floor": 2,
    "target_object": "cup",
    "target_object_description": "white plastic",
    "spatial_relation": "next to",
    "reference_object": "coffee machine",
    "reference_object_description": null,
    "area_hint": null,
    "room_hint": null
}}

User: "{user_query}"
Output:
"""

        try:
            response = client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": prompt_text}], 
                temperature=0.0
            )
            self._update_token_stats(response)
            content = response.choices[0].message.content.strip()
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            return json.loads(content)
        except Exception as e:
            print(f"Step 1 Extraction Error: {e}")
            return {}
    
    def _calculate_llm_weights(self, user_query: str, extracted_data: Dict) -> Dict:
        """Assign a retrieval weight to each field present in the parse."""
        field_mapping = {
            'target_object': 'name',
            'room_hint': 'room',
            'area_hint': 'area',
            'target_object_description': 't_desc',
            'reference_object': 'ref',
            'reference_object_description': 'r_desc',
            'spatial_relation': 'rel'
        }

        active_fields = []
        active_weights_keys = []

        for data_key, weight_key in field_mapping.items():
            if extracted_data.get(data_key):
                active_fields.append(f"- {weight_key} (from '{data_key}': '{extracted_data[data_key]}')")
                active_weights_keys.append(weight_key)
        
        if len(active_weights_keys) == 1:
            return {'name': 1.0, 'room': 0.0, 'area': 0.0, 't_desc': 0.0, 'ref': 0.0, 'r_desc': 0.0, 'rel': 0.0}

        client, model_name, temp = self._get_client_and_model('query_parsing')
        
        active_fields_str = "\n".join(active_fields)
        
        prompt_text = f"""
You are a retrieval strategy expert.
The user issued a query to a robot. We have extracted structured entities.
Your task is to assign **importance weights (0.0 to 1.0)** to the active fields to optimize the search algorithm.

### Context
User Query: "{user_query}"
Extracted Entities: {json.dumps(extracted_data)}

### Active Fields (You must distribute weights among these):
{active_fields_str}

### Weighting Strategy Rules:
1. **Core Anchor**: 'name' (Target Object) is usually the most important, but if the location is very specific, location weights should increase.
2. **Spatial Constraints**: If a room ('room') or reference object ('ref') is mentioned, they are hard constraints. Give them high weights.
3. **Visual vs Spatial**: 
   - If the query is "Red cup on table", 'ref' and 'rel' are critical.
   - If the query is "The very strange red round cup", 't_desc' is critical.
4. **Sum**: The sum of all weights should ideally be close to 1.0 (but I will normalize them later).

### Output JSON Schema:
Return a JSON object with float values ONLY for the active keys listed above.
Example Output: {{ "name": 0.3, "room": 0.5, "t_desc": 0.2 }}
"""
        try:
            response = client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": prompt_text}], 
                temperature=0.1 
            )
            self._update_token_stats(response)
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            
            raw_weights = json.loads(content)
            
            final_weights = {
                'name': 0.0, 'room': 0.0, 'area': 0.0, 
                't_desc': 0.0, 'ref': 0.0, 'r_desc': 0.0, 'rel': 0.0
            }
            
            total_score = 0.0
            
            for k in active_weights_keys:
                score = raw_weights.get(k, 0.0)
                final_weights[k] = float(score)
                total_score += final_weights[k]
            
            if total_score > 0:
                for k in final_weights:
                    final_weights[k] = final_weights[k] / total_score
            else:
                assert False, "Total weight score is zero."
                fallback_val = 1.0 / len(active_weights_keys)
                for k in active_weights_keys:
                    final_weights[k] = fallback_val

            return final_weights

        except Exception as e:
            print(f"Weight Calculation Error: {e}")
            assert False, "Weight calculation failed."
            return {'name': 1.0, 'room': 0.0, 'area': 0.0, 't_desc': 0.0, 'ref': 0.0, 'r_desc': 0.0, 'rel': 0.0}

    def _analyze_negation(self, user_query: str, extracted_data: Dict) -> Dict:
        """Detect negation in the parsed fields."""
        
        check_tasks = []
        
        if extracted_data.get('target_object_description'):
            check_tasks.append(f"- **Target Description**: '{extracted_data['target_object_description']}' (Check field: `target_object_description_negated`)")
            
        if extracted_data.get('reference_object') and extracted_data.get('spatial_relation'):
            check_tasks.append(f"- **Reference Object**: '{extracted_data['reference_object']}' with relation '{extracted_data['spatial_relation']}' (Check field: `reference_object_negated`)")
            
        if extracted_data.get('reference_object_description'):
            check_tasks.append(f"- **Reference Description**: '{extracted_data['reference_object_description']}' (Check field: `reference_object_description_negated`)")
            
        if extracted_data.get('room_hint'):
            check_tasks.append(f"- **Room**: '{extracted_data['room_hint']}' (Check field: `room_hint_negated`)")

        if not check_tasks:
            return self._merge_negation_defaults(extracted_data, {})

        client, model_name, temp = self._get_client_and_model('query_parsing')
        
        context_info = {
            "_reasoning": extracted_data.get('_reasoning'),
            "target_floor": extracted_data.get('target_floor'),
            "target_object": extracted_data.get('target_object')
        }

        task_str = "\n".join(check_tasks)
        
        prompt_text = f"""
You are a logic analysis assistant. 
Review the original query and the extracted entities to determine if there are **NEGATION** constraints.

### Context:
- **User Query**: "{user_query}"
- **Basic Extraction**: {json.dumps(context_info)}

### Your Task:
Analyze the following extracted fields for negation. 
Return a JSON object with boolean values (true/false) for the specified keys.
If the user says "not X", "except X", "non-X", or "anything but X", set to true.

{task_str}

### Example Logic:
- Query: "Find the apple that is not red." -> `target_object_description`: "red" -> **Result**: `target_object_description_negated`: true
- Query: "Find the cup, do not look in the kitchen." -> `room_hint`: "kitchen" -> **Result**: `room_hint_negated`: true
- Query: "Find the cup near the table." -> **Result**: `reference_object_negated`: false
- Query: "Find the cup, not on the table." -> **Result**: `reference_object_negated`: true

### Output JSON Only:
"""
        try:
            response = client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": prompt_text}], 
                temperature=0.01
            )
            self._update_token_stats(response)
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            
            negation_result = json.loads(content)
            
            return self._merge_negation_defaults(extracted_data, negation_result)
            
        except Exception as e:
            print(f"Step 2 Negation Analysis Error: {e}")
            assert False, "Negation analysis failed."
            return self._merge_negation_defaults(extracted_data, {})

    def _merge_negation_defaults(self, original_data: Dict, negation_data: Dict) -> Dict:
        """Merge the parse dictionaries and default missing negation flags to false."""
        merged = original_data.copy()
        merged.update(negation_data)
        
        keys_to_check = [
            "target_object_description_negated",
            "reference_object_negated",
            "reference_object_description_negated",
            "room_hint_negated"
        ]
        
        for key in keys_to_check:
            if key not in merged:
                merged[key] = False
                
        return merged




    def generate_multimodal_room_summary(self, image_path: str, area_summaries: List[Dict]) -> Dict:
        """Write a room label and description from the best view and its area summaries."""
        client, model_name, temp = self._get_client_and_model('cluster_summary') 
        
        areas_text = "\n".join([f"- Area {item.get('tag', 'unknown')}: {item.get('summary', '')}" for item in area_summaries])
        if not areas_text:
            areas_text = "No specific functional areas detected yet."

        prompt_text = f"""
Task: Analyze the provided image of a room and the list of functional areas within it to generate a final room definition.

Input Data:
1. **Image**: A representative view of the room.
2. **Contained Areas**:
{areas_text}

Requirements:
1. **Room Label**: Provide a short, specific semantic category for this room (e.g., "Bedroom", "Office", "Kitchen", "Living Room", "Corridor", "Bathroom").
2. **Description**: Write a concise summary (under 100 words) of the room's layout, function, and atmosphere based on both the visual visual style and the contained areas.

Output JSON only:
{{
    "room_label": "...",
    "description": "..."
}}
"""
        try:
            messages = []
            if image_path and os.path.exists(image_path):
                base64_img = self._encode_image(image_path)
                messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }]
            else:
                messages = [{"role": "user", "content": prompt_text}]

            response = client.chat.completions.create(
                model=model_name, 
                messages=messages, 
                max_tokens=300, 
                temperature=temp
            )
            self._update_token_stats(response)
            
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            
            return json.loads(content)
            
        except Exception as e:
            print(f"[LLM Error] Multimodal Room Summary failed: {e}")
            return {"room_label": "Unknown Room", "description": "Failed to generate description due to LLM error."}

    def detect_room_objects_vlm(self, image_path: str, existing_labels: List[str]) -> List[str]:
        """Add object tags that the segmenter missed in the room image."""
        client, model_name, temp = self._get_client_and_model('object_expansion')
        
        if not image_path or not os.path.exists(image_path):
            return []
        base64_img = self._encode_image(image_path)

        known_str = ", ".join(existing_labels[:50])
        
        prompt_text = f"""
Task: Analyze the image of the room and list visible objects that are NOT in the "Known Objects" list.

Known Objects: [{known_str}]

Requirements:
1. **Identify New Objects**: List distinct, physical objects visible in the image that are missing from the known list.
2. **Strict Exclusions**: 
   - DO NOT list structural elements: wall, floor, ceiling, window, door, door_frame, window_pane.
   - DO NOT list generic terms: item, object, thing, stuff, entity, unknown.
   - DO NOT list lighting unless it's a specific lamp (e.g., exclude "light", "lighting").
3. **Format**: Output ONLY a JSON list of strings (e.g., ["magazine", "remote control", "plant"]).
4. **Specificity**: Be specific (e.g., use "cushion" instead of "cloth").

Output JSON:
"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                max_tokens=300,
                temperature=0.1
            )
            self._update_token_stats(response)
            
            content = response.choices[0].message.content.strip()
            
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            
            new_objects = json.loads(content)
            if isinstance(new_objects, list):
                return [str(o).lower().replace(" ", "_") for o in new_objects]
            return []
            
        except Exception as e:
            print(f"[LLM Error] Object expansion failed: {e}")
            return []

    def verify_retrieval_candidate(self, image_path: str, user_query: str) -> Tuple[bool, str]:
        """Check whether the image matches the user query."""
        client, model_name, temp = self._get_client_and_model('spatial_reasoning')
        base64_image = self._encode_image(image_path)
        
        prompt_text = f"""
        Task: Verification.
        User Query: "{user_query}"
        
        Image: Focus on the object inside the RED bounding box.
        
        Question: Does the object inside the red box match the user's query description?
        - If the query implies a specific location (e.g. 'on the table'), check the context.
        - If the query specifies color or type, check strictly.
        
        Output JSON only:
        {{
            "is_match": boolean,
            "reason": "short explanation"
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=150,
                temperature=0.0 
            )
            self._update_token_stats(response)
            content = response.choices[0].message.content.strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[0].strip()
            
            res_json = json.loads(content)
            return res_json.get("is_match", False), res_json.get("reason", "No reason provided")
            
        except Exception as e:
            print(f"[LLM Error] Verification failed: {e}")
            return False, f"Error: {str(e)}"
        
    def analyze_trajectory_map_for_actions(self, map_image_path: str, current_floor_id: int) -> Dict[str, bool]:
        """Ask whether the current trajectory map shows enough new exploration."""
        client, model_name, temp = self._get_client_and_model('map_decision')
        
        if not map_image_path or not os.path.exists(map_image_path):
            return {"trigger_global_update": False}

        base64_img = self._encode_image(map_image_path)
        
        prompt_text = f"""
Task: Analyze the Robot's Exploration Progress on Floor {current_floor_id}.
Input Map Legend:
- **Colored Overlay**: Identified Room Regions (Masks).
- **Red Line**: Current robot trajectory.
- **Blue Wedges/Dots(The color will fade over time.)**: Locations where a "Global Update" was previously performed.
- **Grey Dots**: Location of objects seen by the robot
- **Empty Space**: Unexplored areas.
- **Colored areas**: Segment room areas.
Decision Goal: Should the robot trigger a "Global Memory Update" (Re-segment all rooms, expand object labels, summarize semantics) NOW?

Criteria for YES (Trigger Update):
1. **Initial Update**: If the map does not yet show blue wedge markers, and the robot's trajectory has already covered at least one room, must update.
2. **New Area**: The robot has explored at least one new room (The red trajectory crosses a colored area that is not yet covered by the blue wedge-shaped markers and the fan-shaped field of view.) since the last Blue Wedge marker. Note that a global update is required as soon as the trajectory covers a room.
3. **Loop Closure**: The trajectory shows the robot has circled back or fully outlined a room's shape. There are blue wedge-shaped markers in the area, but the color is very light or even nonexistent.

Output JSON format:
{{
    "trigger_global_update": true/false,
    "reasoning": "Explain based on the distance from last Blue Wedge and the area coverage."
}}
"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                max_tokens=150,
                temperature=temp
            )
            self._update_token_stats(response)
            
            content = response.choices[0].message.content.strip()
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: 
                content = content.split("```")[0].strip()
            
            result = json.loads(content)
            print(f"  [VLM Decision] Global Update: {result.get('trigger_global_update')} | Reason: {result.get('reasoning')}")
            return result
            
        except Exception as e:
            print(f"[LLM Error] Map decision failed: {e}")
            return {"trigger_global_update": False}
