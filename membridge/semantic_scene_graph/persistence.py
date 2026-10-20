"""Persistence for the refactored semantic graph artifact contract."""

from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

from .hierarchy import hierarchy_edges
from .models import ObjectNode, SceneGraphMemory, SpatialRelation


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


class SceneGraphStore:
    """Read/write JSON-first artifacts plus a compact spatial-graph pickle."""

    def save(self, memory: SceneGraphMemory, directory: str | os.PathLike[str]) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        objects = memory.object_dicts()
        _atomic_text(
            root / "memory_kb.json",
            json.dumps(objects, indent=2, ensure_ascii=False) + "\n",
        )
        nodes = [
            *[memory.floors[key] for key in sorted(memory.floors)],
            *[memory.rooms[key] for key in sorted(memory.rooms)],
            *[memory.areas[key] for key in sorted(memory.areas)],
            *objects,
        ]
        _atomic_text(
            root / "nodes.jsonl",
            "".join(json.dumps(node, ensure_ascii=False) + "\n" for node in nodes),
        )
        relations_path = root / "relations.csv"
        temporary = relations_path.with_suffix(relations_path.suffix + f".tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["child_node_id", "parent_node_id", "node_level"])
            writer.writerows(hierarchy_edges(memory))
        temporary.replace(relations_path)
        graph_payload = {
            "nodes": objects,
            "edges": [relation.to_dict() for relation in memory.spatial_relations],
            "schema_version": 1,
        }
        _atomic_bytes(
            root / "spatial_graph.pkl",
            pickle.dumps(graph_payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        _atomic_text(
            root / "memory_module_semantic_manifest.json",
            json.dumps({
                "schema_version": 1,
                "artifact_kind": "memory_module_semantic_scene_graph",
                "object_count": len(memory.objects),
            }, indent=2, sort_keys=True) + "\n",
        )
        return root

    def load(self, directory: str | os.PathLike[str]) -> SceneGraphMemory:
        root = Path(directory)
        with (root / "memory_kb.json").open("r", encoding="utf-8") as handle:
            objects_raw = json.load(handle)
        memory = SceneGraphMemory(
            objects={
                str(item["node_id"]): ObjectNode.from_dict(item) for item in objects_raw
            }
        )
        nodes_path = root / "nodes.jsonl"
        if nodes_path.is_file():
            for line in nodes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                node = json.loads(line)
                level = int(node.get("node_level", -1))
                node_id = str(node.get("node_id") or "")
                if level == 0:
                    memory.floors[node_id] = node
                elif level == 1:
                    memory.rooms[node_id] = node
                elif level == 2:
                    memory.areas[node_id] = node
        graph_path = root / "spatial_graph.pkl"
        if graph_path.is_file():
            raw: Mapping[str, Any] = pickle.loads(graph_path.read_bytes())
            memory.spatial_relations = [
                SpatialRelation(
                    str(edge["source"]),
                    str(edge["target"]),
                    str(edge["relation"]),
                    float(edge.get("weight", 0.0)),
                )
                for edge in raw.get("edges", [])
            ]
        return memory
