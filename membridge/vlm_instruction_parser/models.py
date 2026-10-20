"""Public data contract for instruction routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

RECALL_SCOPE_STANDARD = "standard"
RECALL_SCOPE_BOTH_PRIOR_EPISODES = "both_prior_episodes"
VALID_RECALL_SCOPES = frozenset({RECALL_SCOPE_STANDARD, RECALL_SCOPE_BOTH_PRIOR_EPISODES})


@dataclass(frozen=True)
class InstructionSegment:
    type: int
    text: str
    label: str | None = None
    room: str | None = None
    recall_scope: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.type, bool) or not isinstance(self.type, int) or self.type not in {1, 2}:
            raise ValueError("type must be 1 or 2")
        scope = str(self.recall_scope or "").strip().lower() or None
        if self.type == 1:
            scope = scope or RECALL_SCOPE_STANDARD
            if not str(self.label or "").strip():
                raise ValueError("a type-1 segment requires a label")
            if scope not in VALID_RECALL_SCOPES:
                raise ValueError(f"invalid recall_scope: {scope}")
        elif any((self.label, self.room, scope)):
            raise ValueError("a type-2 segment requires null label, room and recall_scope")
        object.__setattr__(self, "recall_scope", scope)

    @property
    def kind(self) -> str:
        return "memory_recall" if self.type == 1 else "ordinary_vln"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type, "kind": self.kind, "text": self.text,
            "label": self.label, "room": self.room, "recall_scope": self.recall_scope,
        }


@dataclass(frozen=True)
class InstructionParseResult:
    instruction: str
    segments: Sequence[InstructionSegment]
    model: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    raw_response: str = field(default="", repr=False)

    @property
    def recall_segments(self) -> tuple[InstructionSegment, ...]:
        return tuple(segment for segment in self.segments if segment.type == 1)

    @property
    def ordinary_vln_segments(self) -> tuple[InstructionSegment, ...]:
        return tuple(segment for segment in self.segments if segment.type == 2)

    def format_lines(self) -> str:
        lines = []
        for segment in self.segments:
            if segment.type == 2:
                lines.append(f"2 {segment.text}")
                continue
            line = f"1 {segment.text}, {segment.label}"
            if segment.room:
                line += f", {segment.room}"
            if segment.recall_scope != RECALL_SCOPE_STANDARD:
                line += f", recall_scope={segment.recall_scope}"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self, *, include_raw_response: bool = False) -> dict[str, Any]:
        result = {
            "instruction": self.instruction,
            "segments": [segment.to_dict() for segment in self.segments],
            "formatted": self.format_lines(), "model": self.model,
            "usage": dict(self.usage), "request_id": self.request_id,
        }
        if include_raw_response:
            result["raw_response"] = self.raw_response
        return result


@dataclass(frozen=True)
class ResolvedRecallQuery:
    label: str
    room: str | None
    recall_scope: str
    raw_vlm_label: str
    rule_room: str | None
    vlm_room: str | None
    room_source: str | None

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()
