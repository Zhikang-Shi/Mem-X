"""Strict response validation for the instruction-routing VLM."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .client import VLMCompletion
from .models import (
    InstructionParseResult, InstructionSegment, RECALL_SCOPE_STANDARD,
    VALID_RECALL_SCOPES,
)
from .prompts import INSTRUCTION_ROUTER_SYSTEM_PROMPT


class InstructionParserError(RuntimeError):
    pass


class InvalidInstructionError(InstructionParserError):
    pass


class InstructionVLMCallError(InstructionParserError):
    pass


class InstructionVLMResponseError(InstructionParserError):
    pass


_TERMINAL_EN = re.compile(
    r"^\s*(?:(?:and\s+)?then|next|finally)?\s*,?\s*(?:stop|halt|wait|stand|remain|stay|position(?:\s+yourself)?)\b[^.!?;]*\b(?:it|there|that\s+(?:object|target|place)|the\s+(?:same\s+)?(?:object|target|place))\b\s*[.!?]*\s*$",
    re.I,
)
_TERMINAL_ZH = re.compile(
    r"^\s*(?:接着|然后|最后|并)?\s*[，,]?\s*(?:(?:停|站|等|留)(?:在|到|于|靠在)?[^。；;！？!?]*(?:它|那里|该物体|该目标)[^。；;！？!?]*|(?:在)?(?:它|那里|该物体|该目标)[^。；;！？!?]*(?:停下|停止|停住|等待))\s*[。！？.!?]*\s*$"
)


def extract_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.S | re.I)
    if fenced:
        value = fenced.group(1)
    else:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            raise InstructionVLMResponseError("VLM response contains no JSON object")
        value = value[start:end + 1]
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InstructionVLMResponseError(f"invalid VLM JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InstructionVLMResponseError("VLM JSON must be an object")
    return payload


def parse_segments(text: str) -> list[InstructionSegment]:
    raw_segments = extract_json_object(text).get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise InstructionVLMResponseError("segments must be a non-empty list")
    result = []
    for index, raw in enumerate(raw_segments, 1):
        if not isinstance(raw, Mapping):
            raise InstructionVLMResponseError(f"segment {index} must be an object")
        segment_type = raw.get("type")
        if isinstance(segment_type, bool) or not isinstance(segment_type, int) or segment_type not in {1, 2}:
            raise InstructionVLMResponseError(f"segment {index} type must be integer 1 or 2")
        segment_text = str(raw.get("text") or "").strip()
        label = str(raw.get("label") or "").strip().lower() or None
        room_value, scope_value = raw.get("room"), raw.get("recall_scope")
        if room_value is not None and not isinstance(room_value, str):
            raise InstructionVLMResponseError(f"segment {index} room must be string or null")
        if scope_value is not None and not isinstance(scope_value, str):
            raise InstructionVLMResponseError(f"segment {index} recall_scope must be string or null")
        room = str(room_value or "").strip().lower() or None
        scope = str(scope_value or "").strip().lower() or None
        if not segment_text:
            raise InstructionVLMResponseError(f"segment {index} text is empty")
        if segment_type == 1:
            scope = scope or RECALL_SCOPE_STANDARD
            if not label or scope not in VALID_RECALL_SCOPES:
                raise InstructionVLMResponseError(f"segment {index} has invalid recall fields")
        elif label or room or scope:
            raise InstructionVLMResponseError(f"type-2 segment {index} requires null recall fields")
        result.append(InstructionSegment(segment_type, segment_text, label, room, scope))
    return result


def coalesce_standard_terminal(
    instruction: str, segments: Sequence[InstructionSegment]
) -> list[InstructionSegment]:
    result, index = [], 0
    while index < len(segments):
        current = segments[index]
        following = segments[index + 1] if index + 1 < len(segments) else None
        merge = (
            following is not None and current.type == 1
            and current.recall_scope == RECALL_SCOPE_STANDARD and following.type == 2
            and bool(_TERMINAL_EN.fullmatch(following.text) or _TERMINAL_ZH.fullmatch(following.text))
        )
        if merge:
            text = str(instruction).strip() if len(segments) == 2 and not result else f"{current.text.rstrip()} {following.text.lstrip()}"
            result.append(InstructionSegment(1, text, current.label, current.room, current.recall_scope))
            index += 2
        else:
            result.append(current)
            index += 1
    return result


class VLMInstructionParser:
    def __init__(self, client: Any, *, max_tokens: int = 1024) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.client, self.max_tokens = client, int(max_tokens)

    @staticmethod
    def build_messages(instruction: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": INSTRUCTION_ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original navigation instruction:\n{instruction}\n\nReturn the required JSON object now."},
        ]

    def parse(self, instruction: str) -> InstructionParseResult:
        instruction = str(instruction or "").strip()
        if not instruction:
            raise InvalidInstructionError("navigation instruction is empty")
        try:
            completion = self.client.complete(
                self.build_messages(instruction), max_tokens=self.max_tokens, temperature=0.0
            )
        except Exception as exc:
            raise InstructionVLMCallError(f"instruction-routing VLM call failed: {exc}") from exc
        if isinstance(completion, str):
            completion = VLMCompletion(completion)
        if not isinstance(completion, VLMCompletion):
            raise InstructionVLMCallError("client.complete must return VLMCompletion or str")
        segments = coalesce_standard_terminal(instruction, parse_segments(completion.text))
        return InstructionParseResult(
            instruction, tuple(segments), completion.model, completion.usage,
            completion.request_id, completion.text,
        )
