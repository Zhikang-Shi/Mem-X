"""Memory-module-owned Level-3 prompt, retry, replay, and exploration adapters."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re


_INSTALL_MARKER = "_sg_dual_anchor_plan_replay_overlay_installed"
_INITIAL_PROBE_BRIDGE_MARKER = (
    "_sg_level3_initial_frontier_probe_bridge_installed"
)
_REPLAY_ENV = "LIFELONG_SG_DUAL_ANCHOR_REPLAY_PLAN"
_REPLAY_INDEX_ENV = "LIFELONG_SG_DUAL_ANCHOR_REPLAY_INDEX"
_REPLAY_ALLOW_DYNAMIC_ENV = (
    "LIFELONG_SG_DUAL_ANCHOR_REPLAY_ALLOW_DYNAMIC_MEMORY"
)
_TEMPLATE_TOP1_ENV = "LIFELONG_SG_DUAL_ANCHOR_TEMPLATE_TOP1_DEBUG"
_SCHEMA_RETRIES_ENV = "LIFELONG_SG_VLM_SCHEMA_RETRIES"
_DUAL_PROMPT_MARKER = "_level3_scope_aware_dual_prompt_installed"
_DUAL_RETRY_MARKER = "_level3_dual_schema_retry_installed"

_ANY_MEMORY_PROMPT_SUFFIX = """

Memory coverage can be incomplete. Each candidate shows its best available marked observation,
not a mandatory ep0/ep1 image pair. Prioritize matching the CURRENT recalled object by visual
type, appearance, context and stated spatial relations. If the clause says the object was seen
in both prior episodes, do not require the stored SG node to prove both observations. Missing
one episode (or missing provenance metadata) is not a visual mismatch and is not a reason to
reject a candidate. Recorded episode IDs are optional supporting metadata, not an eligibility
test and not a reason to prefer a visually wrong object. Do not guess evidence in unseen frames.
"""

_DUAL_EPISODE_FORCE_SINGLE_PROMPT = """You select one recalled SG object from exactly three panels.
Each panel contains two marked observations of the same SG object: ep0 and ep1. Compare all three
candidates and both frames in every panel. Use only instruction-supported visual evidence: object
type, appearance, room/context and stated relations. SG labels and SG rank are weak metadata, not
visual proof. Select exactly one candidate when at least one panel is visually compatible. If more
than one candidate remains plausible, still select the best one, use confidence <= 0.34 and explain
the ambiguity. Reject all three only when every panel is an obvious visual mismatch.
For a selection return JSON only: {"no_match":false,"candidate_indices":[1],
"selected_uids":["..."],"confidence":0.0,"reason":"..."}.
For rejection return JSON only: {"no_match":true,"candidate_indices":[],
"selected_uids":[],"confidence":0.0,"reason":"all candidates visibly mismatch because ..."}.
The two forms above are the complete response contract. Do not omit no_match and do not add prose.
"""

_TOP3_SCHEMA_RETRY_SUFFIX = """

Your previous response violated the required JSON contract: {error}
Try again. Return exactly one of the two JSON forms above. In particular, an empty
candidate_indices list is valid only with no_match=true and selected_uids=[].
"""

_LEVEL3_ROUTER_RETRY_SUFFIX = """

This is a Level-3 retry. Your previous response violated the Level-3 schema: {error}
The instruction contains two objects shared by both prior episodes followed by a current-navigation
suffix. Return exactly three ordered segments with types [1,1,2]. Both type-1 segments must use
recall_scope="both_prior_episodes"; preserve the final suffix as the type-2 segment. Return JSON only.
"""


def _enabled(name: str) -> bool:
    return str(os.environ.get(name, "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _schema_retry_count() -> int:
    raw = str(os.environ.get(_SCHEMA_RETRIES_ENV, "2") or "2").strip()
    try:
        retries = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "{} must be a non-negative integer".format(_SCHEMA_RETRIES_ENV)
        ) from exc
    if retries < 0:
        raise RuntimeError(
            "{} must be a non-negative integer".format(_SCHEMA_RETRIES_ENV)
        )
    return retries


def _top3_prompt_for_recall_scopes(scopes, *, recall_policy="both_prior_episodes"):
    from .level3_dual_anchor import (
        RECALL_POLICY_ANY_MEMORY,
        validate_recall_policy,
    )
    from membridge.vlm_instruction_parser.models import (
        RECALL_SCOPE_BOTH_PRIOR_EPISODES,
        RECALL_SCOPE_STANDARD,
    )
    from membridge.vlm_top3_selection.prompts import FORCE_SINGLE_PROMPT

    normalized = tuple(str(value or "").strip().lower() for value in scopes)
    validate_recall_policy(recall_policy)
    if recall_policy == RECALL_POLICY_ANY_MEMORY:
        if not normalized or any(value not in {
            RECALL_SCOPE_BOTH_PRIOR_EPISODES, RECALL_SCOPE_STANDARD
        } for value in normalized):
            raise ValueError("invalid recall scopes: {!r}".format(normalized))
        return FORCE_SINGLE_PROMPT + _ANY_MEMORY_PROMPT_SUFFIX, "any_memory_frame"
    if normalized and all(
        value == RECALL_SCOPE_BOTH_PRIOR_EPISODES for value in normalized
    ):
        return _DUAL_EPISODE_FORCE_SINGLE_PROMPT, "dual_episode_pair"
    if normalized and all(value == RECALL_SCOPE_STANDARD for value in normalized):
        return FORCE_SINGLE_PROMPT, "single_episode_frame"
    raise ValueError("mixed or empty recall scopes: {!r}".format(normalized))


def _install_scope_aware_top3_prompt(logger):
    from membridge.habitat_runtime.level3_dual_anchor import (
        DualAnchorRecallPlanner,
    )

    if getattr(DualAnchorRecallPlanner, _DUAL_PROMPT_MARKER, False):
        return
    original = DualAnchorRecallPlanner.plan

    def plan(self, parsed):
        recalls = tuple(
            segment for segment in parsed.segments if int(segment.type) == 1
        )
        prompt, prompt_mode = _top3_prompt_for_recall_scopes(
            [segment.recall_scope for segment in recalls],
            recall_policy=self.recall_policy,
        )
        self.reranker.system_prompt = prompt
        logger.info(
            "[LEVEL3-VLM-PROMPT] selected Top3 prompt from recall policy and scope: "
            "mode=%s, scopes=%s",
            prompt_mode,
            [segment.recall_scope for segment in recalls],
        )
        return original(self, parsed)

    DualAnchorRecallPlanner.plan = plan
    setattr(DualAnchorRecallPlanner, _DUAL_PROMPT_MARKER, True)


def _install_dual_top3_schema_retry(logger):
    from membridge.vlm_top3_selection import (
        EpisodePairCandidateImagePreparer,
        VLMResponseError,
        VLMTop3Reranker,
    )

    if getattr(VLMTop3Reranker, _DUAL_RETRY_MARKER, False):
        return
    original = VLMTop3Reranker.rerank

    def rerank(self, instruction, candidates):
        if not isinstance(
            self.image_preparer, EpisodePairCandidateImagePreparer
        ) and not getattr(self, "level3_schema_retry", False):
            return original(self, instruction, candidates)
        retries = _schema_retry_count()
        base_prompt = self.system_prompt
        last_error = None
        try:
            for attempt in range(retries + 1):
                if attempt:
                    self.system_prompt = base_prompt + _TOP3_SCHEMA_RETRY_SUFFIX.format(
                        error=str(last_error)
                    )
                try:
                    return original(self, instruction, candidates)
                except VLMResponseError as exc:
                    last_error = exc
                    if attempt >= retries:
                        raise
                    logger.warning(
                        "[LEVEL3-VLM-RETRY] Top3 response schema invalid; "
                        "retrying: retry=%d/%d, error=%s",
                        attempt + 1,
                        retries,
                        exc,
                    )
        finally:
            self.system_prompt = base_prompt

    VLMTop3Reranker.rerank = rerank
    setattr(VLMTop3Reranker, _DUAL_RETRY_MARKER, True)


def _prepare_live_level3_with_schema_retry(
    original,
    evaluator,
    adapter,
    full_instruction,
    logger,
):
    from membridge.habitat_runtime.level3_dual_anchor import (
        InvalidLevel3Parse,
    )
    from membridge.vlm_instruction_parser import parser as parser_module
    from membridge.vlm_instruction_parser.parser import (
        InstructionVLMResponseError,
    )

    retries = _schema_retry_count()
    base_prompt = parser_module.INSTRUCTION_ROUTER_SYSTEM_PROMPT
    last_error = None
    try:
        for attempt in range(retries + 1):
            if attempt:
                parser_module.INSTRUCTION_ROUTER_SYSTEM_PROMPT = (
                    base_prompt
                    + _LEVEL3_ROUTER_RETRY_SUFFIX.format(error=str(last_error))
                )
            try:
                return original(evaluator, adapter, full_instruction)
            except (InstructionVLMResponseError, InvalidLevel3Parse) as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                logger.warning(
                    "[LEVEL3-VLM-RETRY] instruction response schema invalid; "
                    "retrying: retry=%d/%d, error=%s",
                    attempt + 1,
                    retries,
                    exc,
                )
    finally:
        parser_module.INSTRUCTION_ROUTER_SYSTEM_PROMPT = base_prompt


def _deferred_directional_probe_approach(
    evaluator,
    candidate,
    current_world_xz,
    initial_failure,
):
    """Build the deferred exploration state with the memory-module controller.

    The full exploration controller already knows how to execute and audit a
    directional probe, but its initial post-hit path used to return ``None``
    before that state was installed when the selector reported
    ``start_not_clearance_safe``. Reuse its state builder here so the existing
    controller first rotates to collect fresh RGB-D observations, retries safe
    frontier selection, and then enters its bounded directional-probe branch.
    """

    from membridge.habitat_runtime.full_exploration import (
        exploration_patch as full_exploration,
    )

    approach = full_exploration._deferred_requery_approach(
        evaluator,
        candidate,
        current_world_xz,
        initial_failure,
        step=int(getattr(evaluator, "current_step", 0) or 0),
    )
    if approach is None:
        return None
    reference = approach["navigation_reference"]
    reference.update(
        {
            "approach_exploration_trigger": (
                "initial_frontier_start_not_clearance_safe"
            ),
            "approach_exploration_deferred_reason": (
                "initial_frontier_start_not_clearance_safe"
            ),
        }
    )
    return approach


def _is_exploration_connectivity_failure(payload):
    from membridge.habitat_runtime.full_exploration import (
        exploration_patch as full_exploration,
    )

    return full_exploration._is_connectivity_failure(payload)


def _install_initial_frontier_probe_bridge(evaluator_class, logger):
    """Route one previously terminal Level-3 selector result into probing."""

    if getattr(evaluator_class, _INITIAL_PROBE_BRIDGE_MARKER, False):
        return evaluator_class
    original = getattr(evaluator_class, "_sg_plan_v8_post_hit_approach", None)
    if original is None:
        return evaluator_class

    def approach(
        self,
        candidate,
        current_world_xz,
        trajectory_route,
        route_planning_anchors,
    ):
        result = original(
            self,
            candidate,
            current_world_xz,
            trajectory_route,
            route_planning_anchors,
        )
        if result is not None:
            return result
        selection = getattr(self, "_sg_exploration_last_selection", None)
        if not isinstance(selection, dict):
            return None
        status = str(selection.get("status") or "")
        if status != "start_not_clearance_safe":
            return None
        config = getattr(self, "_sg_exploration_config", None)
        if config is not None and not bool(
            getattr(config, "directional_probe", False)
        ):
            return None
        failure = copy.deepcopy(getattr(self, "_sg_last_theta_failure", None))
        if not _is_exploration_connectivity_failure(failure):
            return None
        deferred = _deferred_directional_probe_approach(
            self,
            candidate,
            current_world_xz,
            failure,
        )
        if deferred is None:
            return None
        logger.info(
            "[LEVEL3-EXPLORE-BRIDGE] initial frontier is not clearance-safe; "
            "entering deferred scan/directional probe: uid=%s, status=%s",
            candidate.get("uid") or candidate.get("node_id"),
            status,
        )
        return deferred

    evaluator_class._sg_plan_v8_post_hit_approach = approach
    setattr(evaluator_class, _INITIAL_PROBE_BRIDGE_MARKER, True)
    return evaluator_class


def _template_label_mentions(instruction, vocabulary):
    """Extract authored English labels without consulting private GT data."""

    terms = {
        str(label).strip().lower(): str(label).strip().lower()
        for label in vocabulary.get("labels") or ()
        if str(label).strip()
    }
    terms.update(
        {
            str(alias).strip().lower(): str(canonical).strip().lower()
            for alias, canonical in (vocabulary.get("aliases") or {}).items()
            if str(alias).strip() and str(canonical).strip()
        }
    )
    lowered = str(instruction).lower()
    hits = []
    for term in sorted(terms, key=lambda value: (-len(value), value)):
        pattern = r"(?<![a-z0-9]){}(?![a-z0-9])".format(re.escape(term))
        for match in re.finditer(pattern, lowered):
            hits.append((match.start(), match.end(), term, terms[term]))
    non_overlapping = []
    for hit in sorted(hits, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(
            not (hit[1] <= existing[0] or hit[0] >= existing[1])
            for existing in non_overlapping
        ):
            continue
        non_overlapping.append(hit)
    distinct = []
    seen_canonical = set()
    for hit in sorted(non_overlapping):
        if hit[3] in seen_canonical:
            continue
        seen_canonical.add(hit[3])
        distinct.append(hit)
    if len(distinct) < 2:
        raise ValueError(
            "Level-3 template debug parser found fewer than two distinct "
            "vocabulary labels"
        )
    return tuple(distinct[:2])


def _template_instruction_parse(instruction, vocabulary):
    """Parse the fixed public v3.4 three-sentence Level-3 template."""

    from membridge.vlm_instruction_parser import (
        RECALL_SCOPE_BOTH_PRIOR_EPISODES,
        InstructionParseResult,
        InstructionSegment,
    )

    instruction = str(instruction or "").strip()
    mentions = _template_label_mentions(instruction, vocabulary)
    sentences = [
        (match.start(), match.end(), match.group().strip())
        for match in re.finditer(r"[^。！？!?]+[。！？!?]?", instruction)
        if match.group().strip()
    ]
    if len(sentences) != 3:
        raise ValueError(
            "Level-3 template debug parser requires exactly three sentences"
        )

    sentence_indices = []
    for mention in mentions:
        index = next(
            (
                sentence_index
                for sentence_index, (start, end, _) in enumerate(sentences)
                if start <= mention[0] < end
            ),
            None,
        )
        sentence_indices.append(index)
    if sentence_indices[0] != 0 or sentence_indices[1] not in {1, 2}:
        raise ValueError(
            "Level-3 template labels do not follow the public v3.4 order"
        )

    first_text = sentences[0][2]
    if sentence_indices[1] == 1:
        second_text = sentences[1][2]
        ordinary_text = sentences[2][2]
    else:
        third_start, _, third_text = sentences[2]
        local_label_end = mentions[1][1] - third_start
        suffix = third_text[local_label_end:]
        boundary = re.search(r"后\s*[，,]?", suffix)
        if boundary is None:
            raise ValueError(
                "Level-3 template combined anchor/ordinary sentence has no 后 boundary"
            )
        split_at = local_label_end + boundary.end()
        second_text = "{} {}".format(
            sentences[1][2], third_text[:split_at].strip()
        ).strip()
        ordinary_text = third_text[split_at:].lstrip(" ，,").strip()
    if not ordinary_text:
        raise ValueError("Level-3 template ordinary-VLN suffix is empty")

    segments = (
        InstructionSegment(
            type=1,
            text=first_text,
            label=mentions[0][2],
            room=None,
            recall_scope=RECALL_SCOPE_BOTH_PRIOR_EPISODES,
        ),
        InstructionSegment(
            type=1,
            text=second_text,
            label=mentions[1][2],
            room=None,
            recall_scope=RECALL_SCOPE_BOTH_PRIOR_EPISODES,
        ),
        InstructionSegment(
            type=2,
            text=ordinary_text,
            label=None,
            room=None,
            recall_scope=None,
        ),
    )
    raw_response = json.dumps(
        {
            "debug_parser": "public_level3_v34_template_v1",
            "segments": [segment.to_dict() for segment in segments],
        },
        ensure_ascii=False,
    )
    return InstructionParseResult(
        instruction=instruction,
        segments=segments,
        model="deterministic_public_level3_v34_template_v1",
        usage={},
        request_id=None,
        raw_response=raw_response,
    )


class _DebugTop1Result:
    def __init__(self, candidate):
        self.selected_index = 1
        self.selected_uid = str(candidate.get("uid") or candidate.get("node_id") or "")
        if not self.selected_uid:
            raise ValueError("SG Top-1 debug candidate has no UID")

    def to_dict(self, *, include_candidate=False, include_raw_response=False):
        payload = {
            "selected_index": 1,
            "selected_uid": self.selected_uid,
            "confidence": 0.0,
            "reason": "exploration_debug_only_external_vlm_unavailable",
            "model": None,
            "usage": {},
            "request_id": None,
            "vlm_candidate_indices": [],
            "vlm_candidate_uids": [],
            "vlm_returned_multiple": False,
            "selection_policy": "debug_sg_top1_no_vlm",
            "used_fallback": True,
            "fallback_reason": "external_vlm_unavailable",
            "prepared_images": [],
        }
        if include_raw_response:
            payload["raw_response"] = ""
        return payload


class _DebugTop1Reranker:
    selection_mode = "force_single"

    @staticmethod
    def rerank(instruction, candidates):
        if len(candidates) != 3:
            raise ValueError("SG Top-1 debug reranker requires exactly three candidates")
        return _DebugTop1Result(candidates[0])


def _prepare_template_top1_plan(evaluator, adapter, full_instruction, logger):
    from membridge.habitat_runtime.level3_dual_anchor import (
        DualAnchorNavigationState,
        DualAnchorRecallPlanner,
        level3_recall_policy_from_env,
    )

    instruction = str(full_instruction or "").strip()
    if (
        evaluator._sg_dual_anchor_plan is not None
        and evaluator._sg_dual_anchor_instruction == instruction
    ):
        return evaluator._sg_dual_anchor_plan
    vocabulary = getattr(adapter, "runtime_query_vocabulary", None) or {}
    parsed = _template_instruction_parse(instruction, vocabulary)
    planner = DualAnchorRecallPlanner(
        adapter,
        _DebugTop1Reranker(),
        label_aliases=vocabulary.get("aliases") or {},
        room_aliases=vocabulary.get("room_aliases") or {},
        canonical_labels=vocabulary.get("labels") or (),
        recall_policy=level3_recall_policy_from_env(),
    )
    plan = planner.plan(parsed)
    state = DualAnchorNavigationState(plan.selected_targets)
    session_dir = str(getattr(adapter.pipeline, "run_dir", "") or "")
    if not session_dir:
        raise RuntimeError("Cannot persist template-debug plan without SG run_dir")
    payload = plan.to_dict()
    payload["navigation_state"] = state.to_dict()
    payload["debug_replay"] = {
        "mode": "public_level3_v34_template_plus_sg_top1",
        "external_vlm_called": False,
        "gt_anchor_identity_used": False,
        "gt_anchor_coordinates_used": False,
        "strict_vlm_evaluation": False,
    }
    plan_path = os.path.join(session_dir, "level3_dual_anchor_recall_plan.json")
    temporary = "{}.tmp-{}".format(plan_path, os.getpid())
    with open(temporary, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, plan_path)
    evaluator._sg_dual_anchor_plan = plan
    evaluator._sg_dual_anchor_state = state
    evaluator._sg_dual_anchor_instruction = instruction
    evaluator._sg_dual_anchor_plan_path = plan_path
    logger.warning(
        "[SG-EXPLORE-TEMPLATE-TOP1] debug-only offline plan: labels=%s, "
        "uids=%s, strict_vlm_evaluation=False",
        [anchor.query.get("label") for anchor in plan.anchors],
        [target.get("uid") for target in plan.selected_targets],
    )
    return plan


def _load_replay_payload(path: str, instruction: str):
    with open(path, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") not in {
        "level3_dual_anchor_recall_plan_v1",
        "memory_module_level3_dual_anchor_plan_v1",
    }:
        raise ValueError("Replay plan has an unsupported schema_version")
    parsed = payload.get("instruction_parse") or {}
    if str(parsed.get("instruction") or "").strip() != instruction.strip():
        raise ValueError("Replay plan instruction differs from the live instruction")
    anchors = payload.get("anchors") or []
    if len(anchors) != 2:
        raise ValueError("Replay plan must contain exactly two recall anchors")
    return payload, hashlib.sha256(raw).hexdigest()


def _resolve_indexed_replay(evaluator, index_path: str):
    """Resolve one audited plan for the persistent worker's current task."""

    with open(index_path, "rb") as handle:
        index_payload = json.loads(handle.read().decode("utf-8"))
    if (
        index_payload.get("schema_version")
        != "level3_dual_anchor_vlm_replay_index_v1"
    ):
        raise ValueError("Replay index has an unsupported schema_version")

    binding = getattr(evaluator, "_inherit_sg_current_memory_binding", None) or {}
    task_id = str(binding.get("task_id") or "").strip()
    if not task_id:
        raise RuntimeError(
            "Indexed dual-anchor replay requires a current Level-3 task binding"
        )

    matches = [
        (str(experiment_id), item)
        for experiment_id, item in (index_payload.get("plans") or {}).items()
        if str((item or {}).get("task_id") or "") == task_id
    ]
    if len(matches) != 1:
        raise KeyError(
            "Replay index expected exactly one plan for task {!r}, found {}".format(
                task_id, len(matches)
            )
        )
    experiment_id, item = matches[0]
    relative_path = str(item.get("path") or "").strip()
    expected_sha256 = str(item.get("sha256") or "").strip().lower()
    if not relative_path or not expected_sha256:
        raise ValueError("Replay index entry lacks path or sha256")

    index_root = os.path.realpath(os.path.dirname(index_path))
    replay_path = os.path.realpath(os.path.join(index_root, relative_path))
    if os.path.commonpath((index_root, replay_path)) != index_root:
        raise ValueError("Replay index plan path escapes its index directory")
    return replay_path, expected_sha256, experiment_id, task_id


def _validated_anchor(
    anchor_payload, expected_index, anchor_type, *, recall_policy="both_prior_episodes"
):
    from .level3_dual_anchor import RECALL_POLICY_BOTH_PRIOR, validate_recall_policy

    validate_recall_policy(recall_policy)
    saved_policy = anchor_payload.get("recall_policy", RECALL_POLICY_BOTH_PRIOR)
    if saved_policy != recall_policy:
        raise ValueError("Replay recall policy differs from the current experiment")
    candidates = list(anchor_payload.get("candidates") or [])
    selected = copy.deepcopy(anchor_payload.get("selected_candidate") or {})
    if len(candidates) != 3:
        raise ValueError("Replay anchor must preserve the original SG Top3")
    selected_uid = str(selected.get("uid") or selected.get("node_id") or "")
    candidate_uids = {
        str(candidate.get("uid") or candidate.get("node_id") or "")
        for candidate in candidates
    }
    if not selected_uid or selected_uid not in candidate_uids:
        raise ValueError("Replay selection is not a member of its SG Top3")
    for candidate in candidates:
        observed = set(candidate.get("observed_episode_ids") or [])
        if recall_policy == RECALL_POLICY_BOTH_PRIOR and not {"ep0", "ep1"}.issubset(observed):
            raise ValueError("Replay candidate lacks same-node ep0+ep1 provenance")
    return anchor_type(
        anchor_index=int(expected_index),
        segment=copy.deepcopy(anchor_payload.get("segment") or {}),
        query=copy.deepcopy(anchor_payload.get("query") or {}),
        candidates=tuple(copy.deepcopy(candidates)),
        rerank=copy.deepcopy(anchor_payload.get("rerank") or {}),
        selected_candidate=selected,
        recall_policy=recall_policy,
    )


def install_dual_anchor_plan_replay(evaluator_class, logger):
    """Replay a prior VLM selection only when explicitly enabled for a test."""

    _install_initial_frontier_probe_bridge(evaluator_class, logger)
    if evaluator_class.__module__ == "membridge.habitat_runtime.trainer":
        _install_scope_aware_top3_prompt(logger)
        _install_dual_top3_schema_retry(logger)
    if getattr(evaluator_class, _INSTALL_MARKER, False):
        return evaluator_class
    original = getattr(evaluator_class, "_sg_prepare_dual_anchor_plan", None)
    if original is None:
        # Janus MemoryNavigator only needs the exploration bridge above.
        return evaluator_class

    def prepare(self, adapter, full_instruction):
        replay_path = str(os.environ.get(_REPLAY_ENV, "") or "").strip()
        replay_index_path = str(
            os.environ.get(_REPLAY_INDEX_ENV, "") or ""
        ).strip()
        expected_sha256 = None
        replay_experiment_id = None
        replay_task_id = None
        if replay_path and replay_index_path:
            raise RuntimeError(
                "Set only one of direct replay plan and replay index"
            )
        if replay_index_path:
            (
                replay_path,
                expected_sha256,
                replay_experiment_id,
                replay_task_id,
            ) = _resolve_indexed_replay(self, replay_index_path)
        elif not replay_path:
            if _enabled(_TEMPLATE_TOP1_ENV):
                if not bool(getattr(adapter, "query_map_update_only", False)):
                    raise RuntimeError(
                        "Template Top-1 debug is restricted to query map-only runs"
                    )
                return _prepare_template_top1_plan(
                    self, adapter, full_instruction, logger
                )
            return _prepare_live_level3_with_schema_retry(
                original,
                self,
                adapter,
                full_instruction,
                logger,
            )
        if not bool(getattr(adapter, "query_map_update_only", False)) and not _enabled(
            _REPLAY_ALLOW_DYNAMIC_ENV
        ):
            raise RuntimeError(
                "Dual-anchor replay in a dynamically updated memory requires "
                "LIFELONG_SG_DUAL_ANCHOR_REPLAY_ALLOW_DYNAMIC_MEMORY=1"
            )

        instruction = str(full_instruction or "").strip()
        if (
            self._sg_dual_anchor_plan is not None
            and self._sg_dual_anchor_instruction == instruction
        ):
            return self._sg_dual_anchor_plan

        from membridge.habitat_runtime.level3_dual_anchor import (
            AnchorRecallDecision,
            DualAnchorNavigationState,
            DualAnchorRecallPlan,
            level3_recall_policy_from_env,
        )

        payload, replay_sha256 = _load_replay_payload(replay_path, instruction)
        if expected_sha256 is not None and replay_sha256 != expected_sha256:
            raise ValueError(
                "Replay plan SHA256 differs from the audited replay index"
            )
        anchors = tuple(
            _validated_anchor(
                item, index, AnchorRecallDecision,
                recall_policy=level3_recall_policy_from_env(),
            )
            for index, item in enumerate(payload["anchors"], 1)
        )
        plan = DualAnchorRecallPlan(
            instruction_parse=copy.deepcopy(payload["instruction_parse"]),
            anchors=anchors,
            ordinary_vln_segment=copy.deepcopy(
                payload.get("ordinary_vln_segment") or {}
            ),
        )
        state = DualAnchorNavigationState(plan.selected_targets)

        session_dir = str(getattr(adapter.pipeline, "run_dir", "") or "")
        if not session_dir:
            raise RuntimeError("Cannot persist replayed Level-3 plan without SG run_dir")
        output_payload = plan.to_dict()
        output_payload["navigation_state"] = state.to_dict()
        if payload.get("cached_vlm_replay") is not None:
            output_payload["cached_vlm_replay"] = copy.deepcopy(
                payload["cached_vlm_replay"]
            )
        output_payload["debug_replay"] = {
            "mode": "cached_vlm_top3_force_single",
            "source_path": replay_path,
            "source_sha256": replay_sha256,
            "source_index_path": replay_index_path or None,
            "source_experiment_id": replay_experiment_id,
            "source_task_id": replay_task_id,
            "external_vlm_called": False,
            "vlm_selection_replayed": True,
            "strict_vlm_evaluation": True,
            "old_navigation_state_ignored": True,
        }
        plan_path = os.path.join(session_dir, "level3_dual_anchor_recall_plan.json")
        temporary = "{}.tmp-{}".format(plan_path, os.getpid())
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, plan_path)

        self._sg_dual_anchor_plan = plan
        self._sg_dual_anchor_state = state
        self._sg_dual_anchor_instruction = instruction
        self._sg_dual_anchor_plan_path = plan_path
        logger.info(
            "[SG-EXPLORE-REPLAY] loaded audited dual-anchor selection: "
            "uids={}, sha256={}".format(
                [target.get("uid") for target in plan.selected_targets],
                replay_sha256,
            )
        )
        return plan

    evaluator_class._sg_prepare_dual_anchor_plan = prepare
    setattr(evaluator_class, _INSTALL_MARKER, True)
    return evaluator_class


__all__ = ["install_dual_anchor_plan_replay"]
