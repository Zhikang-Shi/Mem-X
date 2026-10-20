"""Deterministic rule/VLM merge policy for recall fields."""

from __future__ import annotations

from typing import Any, Mapping

from .models import InstructionSegment, RECALL_SCOPE_STANDARD, ResolvedRecallQuery


def normalize_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def resolve_recall_query(
    segment: InstructionSegment, *, label_aliases: Mapping[str, str] | None = None,
    rule_room: str | None = None, room_aliases: Mapping[str, str] | None = None,
) -> ResolvedRecallQuery:
    if not isinstance(segment, InstructionSegment) or segment.type != 1:
        raise ValueError("resolve_recall_query requires a type-1 segment")
    raw_label = normalize_name(segment.label)
    aliases = {normalize_name(k): normalize_name(v) for k, v in (label_aliases or {}).items()}
    label = aliases.get(raw_label, raw_label)
    normalized_rule_room = normalize_name(rule_room) or None
    vlm_room = normalize_name(segment.room) or None
    selected = normalized_rule_room or vlm_room
    source = "rule" if normalized_rule_room else ("vlm_fallback" if vlm_room else None)
    room_alias_map = {normalize_name(k): normalize_name(v) for k, v in (room_aliases or {}).items()}
    if selected:
        selected = room_alias_map.get(selected, selected)
    return ResolvedRecallQuery(
        label, selected, segment.recall_scope or RECALL_SCOPE_STANDARD,
        raw_label, normalized_rule_room, vlm_room, source,
    )
