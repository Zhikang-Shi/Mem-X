"""Level-3 dual-anchor recall orchestration owned by ``memory_module``.

This module deliberately contains no dataset or evaluator code.  It consumes
only the policy-visible instruction and the two-episode memory snapshot that
the Habitat runtime has already built.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from membridge.vlm_instruction_parser import (
    RECALL_SCOPE_BOTH_PRIOR_EPISODES,
    InstructionParseResult,
    resolve_recall_query,
)


REQUIRED_MEMORY_EPISODES = ("ep0", "ep1")
RECALL_POLICY_ANY_MEMORY = "any_memory"
RECALL_POLICY_BOTH_PRIOR = "both_prior_episodes"
RECALL_POLICY_ENV = "LIFELONG_SG_LEVEL3_RECALL_POLICY"


def validate_recall_policy(value: str) -> str:
    if value not in {RECALL_POLICY_ANY_MEMORY, RECALL_POLICY_BOTH_PRIOR}:
        raise ValueError("Unknown Level-3 recall policy: {!r}".format(value))
    return value


def level3_recall_policy_from_env() -> str:
    # Preserve already-running/legacy workers without the new explicit setting.
    # New formal launchers explicitly choose any_memory.
    return validate_recall_policy(
        os.environ.get(RECALL_POLICY_ENV, RECALL_POLICY_BOTH_PRIOR).strip()
    )


class InvalidLevel3Parse(ValueError):
    """The instruction did not satisfy the required [recall, recall, VLN] form."""


class InsufficientProvenanceTop3(RuntimeError):
    """The configured recall pool did not return exactly three candidates."""

    def __init__(
        self, anchor_index: int, candidates: Sequence[Mapping[str, Any]],
        recall_policy: str = RECALL_POLICY_BOTH_PRIOR,
    ):
        self.anchor_index = int(anchor_index)
        self.candidates = tuple(copy.deepcopy(list(candidates)))
        self.recall_policy = validate_recall_policy(recall_policy)
        super().__init__(
            "Anchor {} has {} candidates under {}; exactly 3 are required".format(
                anchor_index, len(candidates), recall_policy
            )
        )


@dataclass(frozen=True)
class AnchorRecallDecision:
    anchor_index: int
    segment: Mapping[str, Any]
    query: Mapping[str, Any]
    candidates: Sequence[Mapping[str, Any]]
    rerank: Mapping[str, Any]
    selected_candidate: Mapping[str, Any]
    recall_policy: str = RECALL_POLICY_BOTH_PRIOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_index": self.anchor_index,
            "segment": copy.deepcopy(dict(self.segment)),
            "query": copy.deepcopy(dict(self.query)),
            "recall_policy": self.recall_policy,
            "required_observed_episode_ids": (
                list(REQUIRED_MEMORY_EPISODES)
                if self.recall_policy == RECALL_POLICY_BOTH_PRIOR else []
            ),
            "candidate_image_mode": (
                "episode_pair_frame_box"
                if self.recall_policy == RECALL_POLICY_BOTH_PRIOR else "frame_box"
            ),
            "candidates": copy.deepcopy(list(self.candidates)),
            "rerank": copy.deepcopy(dict(self.rerank)),
            "selected_candidate": copy.deepcopy(dict(self.selected_candidate)),
        }


@dataclass(frozen=True)
class DualAnchorRecallPlan:
    instruction_parse: Mapping[str, Any]
    anchors: tuple[AnchorRecallDecision, AnchorRecallDecision]
    ordinary_vln_segment: Mapping[str, Any]

    @property
    def selected_targets(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return tuple(anchor.selected_candidate for anchor in self.anchors)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "memory_module_level3_dual_anchor_plan_v1",
            "instruction_parse": copy.deepcopy(dict(self.instruction_parse)),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "ordinary_vln_segment": copy.deepcopy(dict(self.ordinary_vln_segment)),
            "execution_contract": {
                "navigation_order": [1, 2],
                "ordinary_vln_suffix_after_anchor2": True,
                "candidate_switch_after_vlm": False,
            },
        }


@dataclass
class DualAnchorNavigationState:
    targets: Sequence[Mapping[str, Any]]
    phase: str = "pending_anchor1"
    active_index: Optional[int] = None
    reached_uids: list[str] = field(default_factory=list)
    failure: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if len(self.targets) != 2:
            raise ValueError("Dual-anchor navigation requires exactly two targets")
        uids = [
            str(target.get("uid") or target.get("node_id") or "")
            for target in self.targets
        ]
        if any(not uid for uid in uids):
            raise ValueError("Every dual-anchor target requires a UID")

    @property
    def done(self) -> bool:
        return self.phase in {"recall_complete", "failed"}

    @property
    def active_target(self) -> Optional[Mapping[str, Any]]:
        return None if self.active_index is None else self.targets[self.active_index]

    def next_target(self) -> Mapping[str, Any]:
        if self.phase == "pending_anchor1":
            self.active_index = 0
            self.phase = "navigating_anchor1"
        elif self.phase == "pending_anchor2":
            self.active_index = 1
            self.phase = "navigating_anchor2"
        else:
            raise RuntimeError("No target can start from phase {}".format(self.phase))
        return self.targets[self.active_index]

    def mark_reached(self, uid: str) -> Optional[Mapping[str, Any]]:
        if self.phase not in {"navigating_anchor1", "navigating_anchor2"}:
            raise RuntimeError("No active target in phase {}".format(self.phase))
        expected = str(
            self.active_target.get("uid") or self.active_target.get("node_id")
        )
        if str(uid) != expected:
            raise ValueError(
                "Reached UID {} does not match active UID {}".format(uid, expected)
            )
        self.reached_uids.append(expected)
        if self.active_index == 0:
            self.active_index = None
            self.phase = "pending_anchor2"
            return self.next_target()
        self.active_index = None
        self.phase = "recall_complete"
        return None

    def mark_failed(self, reason: str) -> None:
        if self.done:
            raise RuntimeError("Cannot fail a finished dual-anchor navigation")
        self.failure = {
            "phase": self.phase,
            "active_index": self.active_index,
            "reason": str(reason),
        }
        self.active_index = None
        self.phase = "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "active_index": self.active_index,
            "reached_uids": list(self.reached_uids),
            "failure": dict(self.failure) if self.failure else None,
            "ordinary_vln_suffix_pending": self.phase == "recall_complete",
        }


def validate_level3_parse(parsed: InstructionParseResult) -> None:
    types = [segment.type for segment in parsed.segments]
    if types != [1, 1, 2]:
        raise InvalidLevel3Parse(
            "Level-3 instruction must parse as [1, 1, 2], got {}".format(types)
        )
    for index, segment in enumerate(parsed.segments[:2], 1):
        if segment.recall_scope != RECALL_SCOPE_BOTH_PRIOR_EPISODES:
            raise InvalidLevel3Parse(
                "Recall anchor {} must use both_prior_episodes".format(index)
            )
        if not str(segment.label or "").strip():
            raise InvalidLevel3Parse("Recall anchor {} has no label".format(index))
    if not str(parsed.segments[2].text or "").strip():
        raise InvalidLevel3Parse("Level-3 ordinary VLN suffix is empty")


def _ascii_words(value: str) -> str:
    value = str(value or "").lower().replace("_", " ").replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def validate_instruction_authored_label(
    instruction: str,
    label: str,
    aliases: Mapping[str, str],
    *,
    canonical_labels: Optional[Sequence[str]] = None,
) -> None:
    canonical = _ascii_words(label)
    allowed = {_ascii_words(value) for value in (canonical_labels or ())}
    if allowed and canonical not in allowed:
        raise InvalidLevel3Parse(
            "Recall label {!r} is outside the runtime vocabulary".format(label)
        )
    authored = " {} ".format(_ascii_words(instruction))
    mentions = {canonical}
    mentions.update(
        _ascii_words(alias)
        for alias, target in aliases.items()
        if _ascii_words(target) == canonical
    )
    if not any(value and " {} ".format(value) in authored for value in mentions):
        raise InvalidLevel3Parse(
            "Recall label {!r} is not present in the instruction".format(label)
        )


class DualAnchorRecallPlanner:
    """Select two ordered memory targets without consuming hidden evaluator GT."""

    def __init__(
        self,
        adapter: Any,
        reranker: Any,
        *,
        label_aliases: Mapping[str, str] | None = None,
        room_aliases: Mapping[str, str] | None = None,
        canonical_labels: Sequence[str] | None = None,
        recall_policy: str = RECALL_POLICY_BOTH_PRIOR,
    ) -> None:
        if getattr(reranker, "selection_mode", None) != "force_single":
            raise ValueError("Level-3 reranker must use force_single")
        self.adapter = adapter
        self.reranker = reranker
        self.label_aliases = dict(label_aliases or {})
        self.room_aliases = dict(room_aliases or {})
        self.canonical_labels = tuple(canonical_labels or ())
        self.recall_policy = validate_recall_policy(recall_policy)

    def plan(self, parsed: InstructionParseResult) -> DualAnchorRecallPlan:
        validate_level3_parse(parsed)
        decisions = []
        for anchor_index, segment in enumerate(parsed.segments[:2], 1):
            query = resolve_recall_query(
                segment,
                label_aliases=self.label_aliases,
                room_aliases=self.room_aliases,
            )
            validate_instruction_authored_label(
                parsed.instruction,
                query.label,
                self.label_aliases,
                canonical_labels=self.canonical_labels,
            )
            query_args = dict(
                task_text=segment.text,
                room_hint=query.room,
                target_object=query.label,
                top_k=3,
            )
            if self.recall_policy == RECALL_POLICY_BOTH_PRIOR:
                query_args["required_observed_episode_ids"] = REQUIRED_MEMORY_EPISODES
            # The language's both-prior recall_scope describes the referent;
            # missing SG observations are not evidence the object is absent.
            candidates = self.adapter.query_two_stage(**query_args)
            if len(candidates) != 3:
                raise InsufficientProvenanceTop3(
                    anchor_index, candidates, self.recall_policy
                )
            required = (
                set(REQUIRED_MEMORY_EPISODES)
                if self.recall_policy == RECALL_POLICY_BOTH_PRIOR else set()
            )
            if any(
                not required.issubset(set(item.get("observed_episode_ids") or []))
                for item in candidates
            ):
                raise RuntimeError("A Top3 candidate lacks ep0+ep1 provenance")
            if self.recall_policy == RECALL_POLICY_ANY_MEMORY:
                # Match Level-1/2: the other anchor and navigation suffix must
                # not distract visual selection from this recalled object.
                provenance = "; ".join(
                    "{}: {}".format(
                        item.get("uid") or item.get("node_id"),
                        ",".join(item.get("observed_episode_ids") or []) or "unrecorded",
                    )
                    for item in candidates
                )
                rerank_instruction = (
                    "{}\n\nOptional SG observation metadata (not eligibility "
                    "requirements): {}"
                ).format(segment.text, provenance)
            else:
                rerank_instruction = (
                    "Full navigation instruction:\n{}\n\n"
                    "Current memory-recall target {} of 2:\n{}"
                ).format(parsed.instruction, anchor_index, segment.text)
            try:
                result = self.reranker.rerank(rerank_instruction, candidates)
            except Exception as exc:
                # Preserve the exact anchor and candidates for rejection audits.
                # Selecting a UID is not equivalent to physically reaching it.
                exc.anchor_index = anchor_index
                exc.candidates = tuple(copy.deepcopy(candidates))
                exc.recall_policy = self.recall_policy
                raise
            selected = candidates[result.selected_index - 1]
            if str(selected.get("uid") or selected.get("node_id")) != str(
                result.selected_uid
            ):
                raise RuntimeError("VLM selected UID does not match the candidate")
            decisions.append(
                AnchorRecallDecision(
                    anchor_index=anchor_index,
                    segment=segment.to_dict(),
                    query=query.to_dict(),
                    candidates=tuple(copy.deepcopy(candidates)),
                    rerank=result.to_dict(
                        include_candidate=False, include_raw_response=True
                    ),
                    selected_candidate=copy.deepcopy(selected),
                    recall_policy=self.recall_policy,
                )
            )
        return DualAnchorRecallPlan(
            instruction_parse=parsed.to_dict(include_raw_response=True),
            anchors=(decisions[0], decisions[1]),
            ordinary_vln_segment=parsed.segments[2].to_dict(),
        )
