"""One-call VLM selector with strict response and deterministic fallback."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from membridge.vlm_instruction_parser.client import VLMCompletion

from .images import CandidateImageError, CandidateImagePreparer, candidate_uid
from .models import PreparedCandidateImage, RerankResult
from .prompts import ALLOW_MULTIPLE_PROMPT, FORCE_SINGLE_PROMPT


class Top3RerankError(RuntimeError):
    pass


class InvalidCandidateError(Top3RerankError):
    pass


class VLMCallError(Top3RerankError):
    pass


class VLMResponseError(Top3RerankError):
    pass


class VLMNoMatchError(Top3RerankError):
    """The VLM intentionally rejected all retrieved SG candidates."""

    def __init__(
        self, reason: str, *, confidence: float, completion: VLMCompletion,
        prepared_images: Sequence[PreparedCandidateImage], raw_response: str,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.confidence = float(confidence)
        self.model = completion.model
        self.usage = completion.usage
        self.request_id = completion.request_id
        self.prepared_images = tuple(prepared_images)
        self.raw_response = raw_response

    def to_dict(self) -> dict[str, Any]:
        return {
            "no_match": True,
            "confidence": self.confidence,
            "reason": self.reason,
            "model": self.model,
            "usage": dict(self.usage),
            "request_id": self.request_id,
            "prepared_images": [item.audit_dict() for item in self.prepared_images],
            "raw_response": self.raw_response,
        }


def extract_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.S | re.I)
    if fenced:
        value = fenced.group(1)
    else:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            raise VLMResponseError("VLM response contains no JSON object")
        value = value[start:end + 1]
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise VLMResponseError(f"invalid VLM JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VLMResponseError("VLM JSON must be an object")
    return payload


class VLMTop3Reranker:
    def __init__(
        self, client: Any, *, image_preparer: CandidateImagePreparer | None = None,
        fallback_policy: str = "raise", selection_mode: str = "force_single",
        image_detail: str = "high", max_tokens: int = 256,
        system_prompt: str | None = None,
    ) -> None:
        if fallback_policy not in {"raise", "top1"} or selection_mode not in {"allow_multiple", "force_single"}:
            raise ValueError("invalid fallback_policy or selection_mode")
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("invalid image_detail")
        self.client, self.image_preparer = client, image_preparer or CandidateImagePreparer()
        self.fallback_policy, self.selection_mode = fallback_policy, selection_mode
        self.image_detail, self.max_tokens = image_detail, int(max_tokens)
        self.system_prompt = system_prompt or (FORCE_SINGLE_PROMPT if selection_mode == "force_single" else ALLOW_MULTIPLE_PROMPT)

    @staticmethod
    def _validate(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if len(candidates) != 3:
            raise InvalidCandidateError(f"exactly three candidates required, received {len(candidates)}")
        result = list(candidates)
        try:
            uids = [candidate_uid(candidate) for candidate in result]
        except CandidateImageError as exc:
            raise InvalidCandidateError(str(exc)) from exc
        if len(set(uids)) != 3:
            raise InvalidCandidateError(f"candidate UIDs must be unique: {uids}")
        return result

    def build_messages(
        self, instruction: str, candidates: Sequence[Mapping[str, Any]],
        images: Sequence[PreparedCandidateImage],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{
            "type": "text", "text": f"Navigation instruction:\n{instruction}\n\nCompare all three marked candidates."
        }]
        if images and all(image.image_mode == "episode_pair_frame_box" for image in images):
            content.append({"type": "text", "text": "Each panel shows ep0 left and ep1 right for one SG object UID."})
        for index, (candidate, image) in enumerate(zip(candidates, images), 1):
            content.extend([
                {"type": "text", "text": f"Candidate {index}: uid={candidate_uid(candidate)}; SG label={candidate.get('label') or 'unknown'}; room={candidate.get('room_label') or 'unknown'}; original SG rank={index}."},
                {"type": "image_url", "image_url": {"url": image.data_url, "detail": self.image_detail}},
            ])
        ending = (
            "Return exactly one compatible candidate, or no_match=true with an empty selection "
            "when all three are obvious visual mismatches; multiple is forbidden."
            if self.selection_mode == "force_single"
            else "Return all visually plausible candidates when tied."
        )
        content.append({"type": "text", "text": ending + " Return JSON now."})
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}]

    def _parse(self, text: str, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        payload = extract_json_object(text)
        no_match = payload.get("no_match", False)
        if not isinstance(no_match, bool):
            raise VLMResponseError("no_match must be boolean")
        raw_indices = payload.get("candidate_indices", [payload.get("candidate_index")])
        if not isinstance(raw_indices, list):
            raise VLMResponseError("candidate_indices must be a list")
        if no_match:
            if self.selection_mode != "force_single":
                raise VLMResponseError("no_match is supported only in force_single mode")
            if raw_indices or payload.get("selected_uids") not in (None, []):
                raise VLMResponseError("no_match requires empty indices and UIDs")
        elif not raw_indices:
            raise VLMResponseError("candidate_indices must be a non-empty list")
        if any(isinstance(index, bool) or not isinstance(index, int) or index not in {1, 2, 3} for index in raw_indices):
            raise VLMResponseError("candidate indices must be integers in [1,3]")
        if raw_indices != sorted(set(raw_indices)):
            raise VLMResponseError("candidate indices must be sorted and unique")
        if self.selection_mode == "force_single" and not no_match and len(raw_indices) != 1:
            raise VLMResponseError("force_single requires exactly one index")
        expected = [candidate_uid(candidates[index - 1]) for index in raw_indices]
        selected = payload.get("selected_uids")
        if selected is not None and selected != expected:
            raise VLMResponseError(f"indices/UIDs mismatch: expected {expected}, received {selected}")
        if selected is None and payload.get("selected_uid"):
            if len(expected) != 1 or str(payload["selected_uid"]).strip() != expected[0]:
                raise VLMResponseError("legacy selected_uid does not match candidate_index")
        try:
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise VLMResponseError("confidence is not numeric") from exc
        reason = str(payload.get("reason") or "").strip()
        if not math.isfinite(confidence) or not 0 <= confidence <= 1 or not reason:
            raise VLMResponseError("invalid confidence or empty reason")
        return {
            "indices": raw_indices,
            "confidence": confidence,
            "reason": reason,
            "no_match": no_match,
        }

    def rerank(self, instruction: str, candidates: Sequence[Mapping[str, Any]]) -> RerankResult:
        instruction = str(instruction or "").strip()
        if not instruction:
            raise InvalidCandidateError("instruction is empty")
        validated = self._validate(candidates)
        prepared = [self.image_preparer.prepare(candidate, index) for index, candidate in enumerate(validated, 1)]
        raw_response = ""
        try:
            completion = self.client.complete(self.build_messages(instruction, validated, prepared), max_tokens=self.max_tokens, temperature=0.0)
            completion = VLMCompletion(completion) if isinstance(completion, str) else completion
            if not isinstance(completion, VLMCompletion):
                raise VLMCallError("client.complete must return VLMCompletion or str")
            raw_response = completion.text
            parsed = self._parse(raw_response, validated)
            if parsed["no_match"]:
                raise VLMNoMatchError(
                    parsed["reason"],
                    confidence=parsed["confidence"],
                    completion=completion,
                    prepared_images=prepared,
                    raw_response=raw_response,
                )
        except VLMNoMatchError:
            # An intentional visual rejection is an SG miss, never a transport
            # error eligible for deterministic top-1 fallback.
            raise
        except Exception as exc:
            error = exc if isinstance(exc, Top3RerankError) else VLMCallError(f"VLM call failed: {exc}")
            if self.fallback_policy == "raise":
                raise error
            selected = validated[0]
            return RerankResult(
                1, candidate_uid(selected), selected, 0.0,
                "Deterministic SG top-1 fallback because VLM reranking failed",
                None, {}, None, prepared, raw_response, (1,), (candidate_uid(selected),),
                "fallback_sg_top1", True, str(error),
            )
        index = min(parsed["indices"])
        uids = tuple(candidate_uid(validated[value - 1]) for value in parsed["indices"])
        return RerankResult(
            index, candidate_uid(validated[index - 1]), validated[index - 1],
            parsed["confidence"], parsed["reason"], completion.model,
            completion.usage, completion.request_id, prepared, raw_response,
            tuple(parsed["indices"]), uids,
            "sg_top1_within_vlm_candidates" if len(uids) > 1 else "vlm_unique",
        )

    def prepare_only(self, instruction: str, candidates: Sequence[Mapping[str, Any]]):
        if not str(instruction or "").strip():
            raise InvalidCandidateError("instruction is empty")
        validated = self._validate(candidates)
        prepared = [self.image_preparer.prepare(candidate, index) for index, candidate in enumerate(validated, 1)]
        return self.build_messages(str(instruction).strip(), validated, prepared), prepared


def enrich_candidates_from_memory(
    candidates: Sequence[Mapping[str, Any]], memory_kb: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_uid = {candidate_uid(item): item for item in memory_kb}
    result = []
    for candidate in candidates:
        merged = copy.deepcopy(dict(by_uid.get(candidate_uid(candidate), {})))
        merged.update(copy.deepcopy(dict(candidate)))
        result.append(merged)
    return result
