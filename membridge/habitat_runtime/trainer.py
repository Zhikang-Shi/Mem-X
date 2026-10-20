import copy
import hashlib
import os
import re
import traceback
import json
import heapq
from collections import deque

import cv2
import numpy as np
import torch
from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry
from PIL import Image, ImageDraw

from vlnce_baselines.ZS_Evaluator_mp import ZeroShotVlnEvaluatorMP
from .online_adapter import OnlineINHeritSGBuilder
from .post_hit_navigation import build_projected_approach_candidates
from .sg_exploration_adapter import HabitatSGExplorationAdapter


class _SGEnvProxy:
    """Delegate all env calls while streaming step observations to INHerit-SG."""

    def __init__(self, envs, adapter_getter, step_getter, observation_observer=None):
        self._envs = envs
        self._adapter_getter = adapter_getter
        self._step_getter = step_getter
        self._observation_observer = observation_observer

    def step(self, actions):
        outputs = self._envs.step(actions)
        adapter = self._adapter_getter()
        step = int(self._step_getter() or 0)
        for index, item in enumerate(outputs):
            try:
                obs, _, done, info = item
                action = actions[index] if index < len(actions) else None
                terminal_pose = None
                if self._observation_observer is not None:
                    terminal_pose = self._observation_observer(
                        observation=obs,
                        action=action,
                        done=done,
                        info=info,
                        step=step + 1,
                    )
                if done and adapter is not None and terminal_pose is not None:
                    adapter.record_terminal_pose(terminal_pose, step=step + 1)
                if adapter is not None and not done:
                    adapter.process_observation(obs, step)
            except Exception as exc:
                logger.warning(f"[INHerit-SG] env step hook failed: {exc}")
                if self._observation_observer is not None or getattr(
                    adapter, "fail_closed", False
                ):
                    raise
        return outputs

    def close(self):
        return self._envs.close()

    def __getattr__(self, name):
        return getattr(self._envs, name)


@baseline_registry.register_trainer(name="ZS-Evaluator-mp-memory-module-warm")
class INHeritSGZeroShotVlnEvaluatorMP(ZeroShotVlnEvaluatorMP):
    """Uni-LaViRA evaluator with online INHerit-SG graph building attached."""

    def __init__(self, config, r2r, segment_module=None, mapping_module=None) -> None:
        super().__init__(config, r2r, segment_module=segment_module, mapping_module=mapping_module)
        if os.environ.get("LAVIRA_SAVE_VISUALIZATIONS", "1") != "1":
            # The base evaluator invokes the visualizer before Semantic_Mapping
            # has initialized its map.  Warm runs intentionally save SG/query
            # artifacts rather than redundant combined-frame renderings.
            self.visualize = False
            self.visualizer.visualize = False
            self.visualizer.debug_log_dir = None
        exp_name = os.path.basename(config.EVAL_CKPT_PATH_DIR.rstrip("/")) or "run"
        root = os.environ.get("INHERIT_SG_OUTPUT_DIR")
        if not root:
            raise RuntimeError(
                "INHERIT_SG_OUTPUT_DIR is required; refusing an implicit output fallback"
            )
        device = self._sg_device_string()
        self.inherit_sg = OnlineINHeritSGBuilder(
            config,
            width=self.width,
            height=self.height,
            save_dir=root,
            device=device,
        )
        self._inherit_sg_output_root = os.path.abspath(root)
        self._inherit_sg_device = device
        self._inherit_sg_preload_root = os.environ.get(
            "INHERIT_SG_PRELOAD_ROOT", ""
        ).strip()
        self._inherit_sg_episode_memory_map = (
            self._load_inherit_sg_episode_memory_map()
        )
        self._inherit_sg_current_memory_binding = None
        self.sg_first_enabled = self._env_bool("LIFELONG_SG_FIRST", True)
        self.sg_stop_on_miss = self._env_bool("LIFELONG_SG_STOP_ON_MISS", False)
        self._sg_stop_on_miss_requested = False
        self.sg_first_hit_conf = float(os.environ.get("LIFELONG_SG_HIT_CONF", "0.8"))
        self.sg_nav_stall_steps = int(os.environ.get("LIFELONG_SG_STALL_STEPS", "10"))
        self.sg_nav_stall_eps_m = float(os.environ.get("LIFELONG_SG_STALL_EPS_M", "0.05"))
        self.sg_max_refinements = int(
            os.environ.get("LIFELONG_SG_MAX_REFINEMENTS", "3")
        )
        self.sg_local_action_radius_m = float(
            os.environ.get("LIFELONG_SG_LOCAL_ACTION_RADIUS_M", "2.5")
        )

        self.sg_local_action_max_states = int(
            os.environ.get("LIFELONG_SG_LOCAL_ACTION_MAX_STATES", "12000")
        )
        self.sg_local_action_goal_tolerance_m = float(
            os.environ.get("LIFELONG_SG_LOCAL_ACTION_GOAL_TOLERANCE_M", "0.15")
        )
        # Habitat can only translate in fixed FORWARD_STEP_SIZE increments.  A
        # world-map target therefore needs a controller-independent terminal
        # set: reaching the same physical pose must not depend on whether the
        # bounded local-action recovery happened to run first.  Keep the
        # default aligned with the local lattice's attainable-goal tolerance.
        self.sg_nav_goal_tolerance_m = float(
            os.environ.get(
                "LIFELONG_SG_NAV_GOAL_TOLERANCE_M",
                str(self.sg_local_action_goal_tolerance_m),
            )
        )
        self.sg_max_query_attempts = max(
            1, int(os.environ.get("LIFELONG_SG_MAX_QUERY_ATTEMPTS", "1"))
        )
        self.sg_path_lookahead_m = float(
            os.environ.get("LIFELONG_SG_PATH_LOOKAHEAD_M", "1.0")
        )
        self.sg_path_deviation_m = float(
            os.environ.get("LIFELONG_SG_PATH_DEVIATION_M", "0.35")
        )
        self.sg_path_turn_cost_cells = float(
            os.environ.get("LIFELONG_SG_PATH_TURN_COST_CELLS", "0.25")
        )
        self.sg_snap_radius_cells = int(os.environ.get("LIFELONG_SG_SNAP_RADIUS_CELLS", "40"))
        # A sofa's fused semantic centroid can lie under the seat or on the
        # far side of its footprint.  In the benchmark this makes the nearest
        # navigable point topologically wrong even though retrieval is
        # correct.  The best observation is an agent pose looking at the
        # object, so extend that viewing ray by a conservative sofa half-depth
        # before performing the existing navmesh/M0 terminal projection.
        # This is deliberately exact-label scoped: armchairs and every other
        # object keep the original semantic-centre objective.
        self.sg_sofa_observation_ray_center_enable = self._env_bool(
            "LIFELONG_SG_SOFA_OBSERVATION_RAY_CENTER_ENABLE", True
        )
        self.sg_sofa_center_half_depth_m = float(
            os.environ.get("LIFELONG_SG_SOFA_CENTER_HALF_DEPTH_M", "0.75")
        )
        self.sg_observation_ray_approach_enable = self._env_bool(
            "LIFELONG_SG_OBSERVATION_RAY_APPROACH_ENABLE", True
        )
        self.sg_object_approach_standoff_m = float(
            os.environ.get("LIFELONG_SG_OBJECT_APPROACH_STANDOFF_M", "0.75")
        )
        # Staircase objects are a special case for a 2-D world map: the
        # semantic centre may be correctly retrieved while an exact grid-cell
        # navmesh test rejects every nearby M0 terminal (for example when the
        # navmesh surface is on an adjacent stair tread).  If that displacement
        # is already large, retain the semantic-centre objective but allow a
        # bounded terminal snap no larger than the controller's goal tolerance.
        # The fallback is accepted only when the persisted acquisition route to
        # the object's best observation can be replayed end to end.
        self.sg_staircase_plant_waypoint_enable = self._env_bool(
            "LIFELONG_SG_STAIRCASE_PLANT_WAYPOINT_ENABLE", False
        )
        self.sg_staircase_plant_center_residual_trigger_m = float(
            os.environ.get(
                "LIFELONG_SG_STAIRCASE_PLANT_CENTER_RESIDUAL_TRIGGER_M",
                "1.0",
            )
        )
        self.sg_staircase_plant_navmesh_snap_tolerance_m = float(
            os.environ.get(
                "LIFELONG_SG_STAIRCASE_PLANT_NAVMESH_SNAP_TOLERANCE_M",
                str(self.sg_nav_goal_tolerance_m),
            )
        )
        self.sg_trajectory_corridor_enable = self._env_bool(
            "LIFELONG_SG_TRAJECTORY_CORRIDOR_ENABLE", True
        )
        self.sg_warm_start_trajectory_connect_m = float(
            os.environ.get("LIFELONG_SG_WARM_START_TRAJECTORY_CONNECT_M", "0.35")
        )
        # The acquisition route is a traversability certificate and recovery
        # backbone, not a mandatory temporal replay.  Prefer a shorter path
        # through ordinary observed-free M0 cells (plus only the bounded warm
        # terminal connector) when it saves at least one physical step.  The
        # full certified route remains the fail-safe fallback.
        self.sg_shortest_safe_path_enable = self._env_bool(
            "LIFELONG_SG_SHORTEST_SAFE_PATH_ENABLE", True
        )
        self.sg_shortcut_min_savings_m = float(
            os.environ.get("LIFELONG_SG_SHORTCUT_MIN_SAVINGS_M", "0.25")
        )
        # Global routing remains map/certificate based.  The one-step
        # PathFinder transition model is enabled only to reproduce Habitat's
        # local collision sliding for a candidate MOVE_FORWARD; it never asks
        # for a route, snaps the SG target, or changes semantic retrieval.
        # Without this transition model the nominal 0.25-m endpoint can overlap
        # furniture even when Habitat safely slides along the certified
        # acquisition path (829951 ep1).
        self.sg_navmesh_action_prediction = self._env_bool(
            "LIFELONG_SG_NAVMESH_ACTION_PREDICTION", True
        )
        # M0 remains the global routing map, but an M0 free cell is not
        # necessarily a pose where Habitat can place the agent (for example a
        # floor return underneath a sink/counter).  Intersect only the terminal
        # candidate set with the current live navmesh island.  This does not
        # expose a navmesh route to Theta*: it prevents the global planner from
        # optimizing toward a physically nonexistent endpoint.
        self.sg_navmesh_target_validation = self._env_bool(
            "LIFELONG_SG_NAVMESH_TARGET_VALIDATION", True
        )
        # V8 treats the retrieved object centre as a semantic measurement, not
        # as a robot pose.  The independently smoked post-hit repair searches
        # the complete success neighbourhood, intersects it with the current
        # navmesh island, then asks the persisted world/trajectory planner to
        # reach the closest admissible terminal.
        self.sg_post_hit_max_object_distance_m = float(
            os.environ.get(
                "LIFELONG_SG_POST_HIT_MAX_OBJECT_DISTANCE_M", "3.0"
            )
        )
        self.sg_post_hit_max_navmesh_snap_m = float(
            os.environ.get(
                "LIFELONG_SG_POST_HIT_MAX_NAVMESH_SNAP_M", "0.75"
            )
        )
        self.sg_post_hit_max_safe_snap_m = float(
            os.environ.get(
                "LIFELONG_SG_POST_HIT_MAX_SAFE_SNAP_M", "1.0"
            )
        )
        self.sg_post_hit_annulus_step_m = float(
            os.environ.get(
                "LIFELONG_SG_POST_HIT_ANNULUS_STEP_M", "0.10"
            )
        )
        self.sg_post_hit_angular_samples = int(
            os.environ.get(
                "LIFELONG_SG_POST_HIT_ANGULAR_SAMPLES", "64"
            )
        )
        self.sg_post_hit_max_planner_candidates = int(
            os.environ.get(
                "LIFELONG_SG_POST_HIT_MAX_PLANNER_CANDIDATES", "24"
            )
        )
        self.sg_require_traversable = self._env_bool("LIFELONG_SG_REQUIRE_TRAVERSABLE", True)
        self.sg_align_heading = self._env_bool("LIFELONG_SG_ALIGN_HEADING", True)
        self.sg_stop_at_waypoint = self._env_bool("LIFELONG_SG_STOP_AT_WAYPOINT", True)
        self.sg_heading_tolerance_deg = float(os.environ.get("LIFELONG_SG_HEADING_TOLERANCE_DEG", "15"))
        self.sg_query_source = os.environ.get("LIFELONG_SG_QUERY_SOURCE", "todo").strip().lower()
        self.sg_dual_anchor_enabled = self._env_bool(
            "LIFELONG_SG_DUAL_ANCHOR_ENABLE", False
        )
        self.level3_formal_suffix_enabled = self._env_bool(
            "MEMORY_MODULE_LEVEL3_FORMAL_SUFFIX_ENABLE", False
        )
        self.sg_policy_query_specs = self._load_sg_policy_query_specs()
        self._sg_blocked_by_task = {}
        self._sg_skip_next_task_key = None
        self._sg_nav_progress = None
        self._sg_instruction_reached_episodes = set()
        self._sg_instruction_query_attempts = {}
        self._sg_instruction_exhaustion_logged = set()
        self._w2m = None
        self._sg_memory_traversable_cache = None
        self._sg_memory_traversable_cache_key = None
        self._sg_memory_obstacle_cache = None
        self._sg_lavira_memory_source = None
        self._sg_lavira_memory_source_path = None
        self._sg_observed_obstacle = None
        self._sg_path_replans = 0
        self._sg_path_plan_active = False
        self._sg_world_position_xz = None
        self._sg_world_heading_rad = None
        self._sg_world_plan_window = None
        self._sg_world_safe_mask = None
        self._sg_world_trajectory_probe_mask = None
        self._sg_world_collision_blocked_cells = set()
        self._sg_world_collision_segments = []
        self._sg_dual_anchor_plan = None
        self._sg_dual_anchor_state = None
        self._sg_dual_anchor_instruction = None
        self._sg_dual_anchor_plan_path = None
        self._sg_level3_lavira_fallback = None
        self.sg_vlm_router_enabled = self._env_bool(
            "LIFELONG_SG_VLM_ROUTER_ENABLE", False
        )
        self._sg_single_recall_plans = {}
        self._sg_exploration_adapter = HabitatSGExplorationAdapter(
            enabled=self._env_bool("LIFELONG_SG_EXPLORATION_ENABLE", False),
            maximum_actions=int(
                os.environ.get("LIFELONG_SG_EXPLORATION_MAX_ACTIONS", "100")
            ),
            maximum_distance_m=float(
                os.environ.get("LIFELONG_SG_EXPLORATION_MAX_DISTANCE_M", "20.0")
            ),
            maximum_scan_turns=int(
                os.environ.get("LIFELONG_SG_EXPLORATION_SCAN_TURNS", "12")
            ),
            maximum_frontiers=int(
                os.environ.get("LIFELONG_SG_EXPLORATION_MAX_FRONTIERS", "4")
            ),
            frontier_goal_tolerance_m=self.sg_nav_goal_tolerance_m,
        )

    @staticmethod
    def _clear_grounding_dino_image_cache(segment_module) -> None:
        wrapper = getattr(segment_module, "grounding_dino_model", None)
        model = getattr(wrapper, "model", wrapper)
        unset_image_tensor = getattr(model, "unset_image_tensor", None)
        if callable(unset_image_tensor):
            unset_image_tensor()

    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """Serialize GPU vision and recover one known GroundingDINO cache race."""
        adapter = getattr(self, "inherit_sg", None)
        lock = getattr(adapter, "gpu_inference_lock", None)
        if lock is None:
            return super()._get_sem_pred(rgb)

        with lock:
            try:
                return super()._get_sem_pred(rgb)
            except RuntimeError as exc:
                message = str(exc)
                if "invalid for input of size 0" not in message or "shape" not in message:
                    raise
                self._clear_grounding_dino_image_cache(self.segment_module)
                logger.warning(
                    "[INHerit-SG] cleared stale GroundingDINO image features "
                    "and retrying semantic inference once: %s",
                    message,
                )
                return super()._get_sem_pred(rgb)

    def _load_inherit_sg_episode_memory_map(self):
        """Load the session-scoped warm memory contract once per worker."""

        path = os.environ.get("INHERIT_SG_EPISODE_MEMORY_MAP", "").strip()
        if not path:
            return None
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"INHERIT_SG_EPISODE_MEMORY_MAP does not exist: {path}"
            )
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != (
            "unilavira_level1_warm_episode_memory_map_v3"
        ):
            raise RuntimeError(
                f"Unsupported warm episode-memory map schema: {path}"
            )
        episodes = payload.get("episodes")
        if not isinstance(episodes, dict) or not episodes:
            raise RuntimeError(f"Warm episode-memory map is empty: {path}")
        normalized = {}
        branch_ids = set()
        for episode_id, raw in episodes.items():
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"Invalid warm memory binding for episode {episode_id}"
                )
            binding = copy.deepcopy(raw)
            binding["episode_id"] = str(episode_id)
            for key in ("session_id", "role", "base_m0_id", "branch_id"):
                value = str(binding.get(key) or "").strip()
                if not value:
                    raise RuntimeError(
                        f"Warm memory binding {episode_id} has no {key}"
                    )
                binding[key] = value
            if not binding["role"].startswith("ep"):
                raise RuntimeError(
                    f"Invalid warm role for {episode_id}: {binding['role']}"
                )
            if binding["branch_id"] in branch_ids:
                raise RuntimeError(
                    f"Duplicate warm branch_id: {binding['branch_id']}"
                )
            branch_ids.add(binding["branch_id"])
            normalized[str(episode_id)] = binding
        if not self._inherit_sg_preload_root:
            raise RuntimeError(
                "INHERIT_SG_PRELOAD_ROOT is required with "
                "INHERIT_SG_EPISODE_MEMORY_MAP"
            )
        logger.info(
            "[INHerit-SG-WARM] loaded independent EP0-branch bindings: "
            f"{len(normalized)} episode(s) from {path}"
        )
        return normalized

    def _bind_inherit_sg_episode_memory(self, episode_id, scene_id):
        """Bind one query to a private branch of its immutable EP0 M0."""

        bindings = self._inherit_sg_episode_memory_map
        if bindings is None:
            return
        episode_key = str(episode_id)
        binding = bindings.get(episode_key)
        if binding is None:
            raise RuntimeError(
                f"No immutable M0 binding for query episode {episode_key}"
            )
        preload_dir = os.path.join(
            os.path.abspath(self._inherit_sg_preload_root),
            "sessions",
            binding["session_id"],
        )
        required = (
            "memory_kb.json",
            "builder_state.pkl",
            "world_metric_map.npz",
            "trajectory_graph.npz",
            "lavira_observed_free_memory.npz",
        )
        missing = [
            name for name in required
            if not os.path.isfile(os.path.join(preload_dir, name))
        ]
        if missing:
            raise FileNotFoundError(
                "Warm query M0 is incomplete for "
                f"{episode_key}/{binding['session_id']}: {missing}"
            )

        previous = getattr(self, "inherit_sg", None)
        shared_models = getattr(previous, "shared_models", None)
        adapter = OnlineINHeritSGBuilder(
            self.config,
            width=self.width,
            height=self.height,
            save_dir=self._inherit_sg_output_root,
            device=self._inherit_sg_device,
        )
        adapter.shared_models = shared_models
        adapter.preload_dir = preload_dir
        # Constructor already reads INHERIT_SG_PRELOAD_QUERY_ONLY /
        # WORLD_MAP_ONLY.  Official L1 warm leaves both off and builds SAM3
        # observations.  The 55-episode explore replay sets both on so the
        # session stays model-free and only the navigation map is updated.
        if adapter.preload_query_only:
            adapter.async_enabled = False
            adapter.shared_models = None
        elif adapter.preload_world_map_only:
            raise RuntimeError(
                "INHERIT_SG_PRELOAD_WORLD_MAP_ONLY=1 requires "
                "INHERIT_SG_PRELOAD_QUERY_ONLY=1"
            )
        adapter.output_root = os.path.join(
            self._inherit_sg_output_root,
            "branches",
            binding["session_id"],
            f"{binding['role']}_{episode_key}",
        )
        self.inherit_sg = adapter
        self._inherit_sg_current_memory_binding = copy.deepcopy(binding)
        self._inherit_sg_current_memory_binding.update(
            {
                "preload_dir": preload_dir,
                "output_root": adapter.output_root,
                "scene_id": str(scene_id),
                "memory_input": "immutable_ep0_m0",
                "memory_scope": "independent_episode_branch_from_ep0",
            }
        )
        logger.info(
            "[INHerit-SG-WARM] bound independent EP0 branch: "
            f"episode={episode_key}, role={binding['role']}, "
            f"session={binding['session_id']}, base_m0={binding['base_m0_id']}, "
            f"memory_input=immutable_ep0_m0, preload={preload_dir}, "
            f"output={adapter.output_root}, reused_models={adapter.shared_models is not None}, "
            f"query_only={bool(adapter.preload_query_only)}, "
            f"world_map_only={bool(adapter.preload_world_map_only)}, "
            "reused_episode_memory=False"
        )

    def _load_sg_policy_query_specs(self):
        """Load instruction-compiled query fields without any GT/provenance."""

        path = os.environ.get("LIFELONG_SG_QUERY_SPEC_PATH", "").strip()
        required = self._env_bool("LIFELONG_SG_QUERY_SPEC_REQUIRED", False)
        if not path:
            if required:
                raise RuntimeError(
                    "LIFELONG_SG_QUERY_SPEC_REQUIRED=1 but no query-spec path "
                    "was provided"
                )
            return None
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Policy query-spec file is missing: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "level3_policy_query_specs_v1":
            raise RuntimeError(f"Unsupported policy query-spec schema: {path}")
        queries = payload.get("queries")
        if not isinstance(queries, dict) or not queries:
            raise RuntimeError(f"Policy query-spec file has no queries: {path}")
        vocabulary_path = os.environ.get(
            "INHERIT_SG_QUERY_VOCABULARY",
            "",
        ).strip()
        if not vocabulary_path:
            raise RuntimeError(
                "Policy query specs require INHERIT_SG_QUERY_VOCABULARY"
            )
        with open(vocabulary_path, "rb") as handle:
            vocabulary_raw = handle.read()
        vocabulary = json.loads(vocabulary_raw.decode("utf-8"))
        vocabulary_sha256 = hashlib.sha256(vocabulary_raw).hexdigest()
        if (
            payload.get("vocabulary_version")
            != vocabulary.get("vocabulary_version")
            or payload.get("vocabulary_sha256") != vocabulary_sha256
        ):
            raise RuntimeError(
                "Policy query specs were not compiled from the active runtime "
                "query vocabulary"
            )
        for episode_id, query in queries.items():
            if str(query.get("episode_id")) != str(episode_id):
                raise RuntimeError(
                    f"Policy query-spec key/episode mismatch for {episode_id}"
                )
            if not bool(query.get("policy_visible_only")):
                raise RuntimeError(
                    f"Policy query for {episode_id} is not visibility-certified"
                )
            forbidden = (
                bool(query.get("contains_gt_coordinates"))
                or bool(query.get("contains_memory_node_id"))
                or bool(query.get("contains_episode_provenance"))
            )
            if forbidden:
                raise RuntimeError(
                    f"Policy query for {episode_id} contains privileged fields"
                )
            if not query.get("target_object") or not query.get(
                "anchor_clause_normalized"
            ):
                raise RuntimeError(
                    f"Policy query for {episode_id} is incomplete"
                )
            branch_contract = query.get("instruction_branch_contract")
            if branch_contract is not None and branch_contract != (
                "sg_reach_uses_continuation_sg_failure_uses_full_instruction"
            ):
                raise RuntimeError(
                    f"Policy query for {episode_id} has an unsupported "
                    f"instruction branch contract: {branch_contract!r}"
                )
        logger.info(
            "[INHerit-SG-FIRST] loaded policy-visible Level-3 query specs: "
            f"{len(queries)} episode(s), complete-memory scope, "
            "episode-provenance filter disabled"
        )
        return payload

    @staticmethod
    def _sg_normalize_query_text(value):
        text = str(value or "").lower().replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _sg_split_anchor_instruction(instruction):
        """Return the policy-visible SG clause and its normal-VLN suffix."""

        text = str(instruction or "").strip()
        boundary = re.search(
            (
                r"\s*(?:[.;]|\b(?:then|after\s+that|finally)\b|"
                r"\bnext\b(?!\s+to\b))\s*"
            ),
            text,
            flags=re.IGNORECASE,
        )
        if boundary is None:
            return text, ""
        anchor = text[: boundary.start()].strip().rstrip(",").strip()
        continuation = text[boundary.end() :].strip()
        return anchor, continuation

    def _sg_policy_query_for_task(self, task_text, full_instruction=None):
        payload = self.sg_policy_query_specs
        if payload is None:
            return None, None
        episode_id = str(getattr(self, "current_episode_id", ""))
        query = (payload.get("queries") or {}).get(episode_id)
        if query is None:
            return None, f"no policy query for episode {episode_id}"
        expected = self._sg_normalize_query_text(
            query.get("anchor_clause_normalized")
        )
        actual = self._sg_normalize_query_text(task_text)
        if expected != actual:
            return None, (
                "policy query text mismatch: "
                f"expected={expected!r}, actual={actual!r}"
            )
        if (
            full_instruction is not None
            and "continuation_instruction_normalized" in query
        ):
            _, runtime_continuation = self._sg_split_anchor_instruction(
                full_instruction
            )
            expected_continuation = self._sg_normalize_query_text(
                query.get("continuation_instruction_normalized")
            )
            actual_continuation = self._sg_normalize_query_text(
                runtime_continuation
            )
            if expected_continuation != actual_continuation:
                return None, (
                    "policy continuation text mismatch: "
                    f"expected={expected_continuation!r}, "
                    f"actual={actual_continuation!r}"
                )
        return query, None

    def _init_envs(self) -> None:
        super()._init_envs()
        self._wrap_envs_for_inherit_sg()

    def _maps_initialization(self):
        obs, full_pose = super()._maps_initialization()
        self._start_inherit_sg_episode(obs[0])
        return obs, full_pose

    def rollout(self):
        completed = False
        try:
            result = super().rollout()
            completed = True
            return result
        finally:
            self._finish_inherit_sg_episode(abort=not completed)

    def reset(self) -> None:
        self._finish_inherit_sg_episode()
        self._sg_exploration_adapter.reset()
        self._sg_stop_on_miss_requested = False
        self._sg_memory_traversable_cache = None
        self._sg_memory_traversable_cache_key = None
        self._sg_memory_obstacle_cache = None
        self._sg_world_plan_window = None
        self._sg_world_safe_mask = None
        self._sg_world_trajectory_probe_mask = None
        self._sg_world_collision_blocked_cells = set()
        self._sg_world_collision_segments = []
        self._sg_dual_anchor_plan = None
        self._sg_dual_anchor_state = None
        self._sg_dual_anchor_instruction = None
        self._sg_dual_anchor_plan_path = None
        self._sg_level3_lavira_fallback = None
        return super().reset()

    def _wrap_envs_for_inherit_sg(self):
        if isinstance(self.envs, _SGEnvProxy):
            return
        self.envs = _SGEnvProxy(
            self.envs,
            adapter_getter=lambda: getattr(self, "inherit_sg", None),
            step_getter=lambda: getattr(self, "current_step", 0),
            observation_observer=self._capture_chain_step_output,
        )

    def _start_inherit_sg_episode(self, initial_obs):
        try:
            episode = self.envs.current_episodes()[0]
            episode_id = episode.episode_id
            scene_id = getattr(episode, "scene_id", "scene")
            if not self._inherit_sg_episode_allowed(episode_id, scene_id):
                return
            self._bind_inherit_sg_episode_memory(episode_id, scene_id)
            self.inherit_sg.start_episode(episode_id, scene_id=scene_id)
            self.inherit_sg.process_observation(initial_obs, step=0)
        except Exception as exc:
            logger.warning(f"[INHerit-SG] start hook failed: {exc}\n{traceback.format_exc()}")
            if getattr(getattr(self, "inherit_sg", None), "fail_closed", False):
                raise

    def _finish_inherit_sg_episode(self, abort=False):
        try:
            if getattr(self, "inherit_sg", None) is not None:
                if abort:
                    self.inherit_sg.abort_episode()
                else:
                    self.inherit_sg.finish_episode()
                    self._save_lavira_traversable_memory()
        except Exception as exc:
            logger.warning(f"[INHerit-SG] finish hook failed: {exc}\n{traceback.format_exc()}")
            if getattr(getattr(self, "inherit_sg", None), "fail_closed", False):
                raise

    def _calculate_metric(
        self, infos, submitted_answer=None, is_timeout=False
    ):
        """Persist the neutral dynamic-query trajectory before SG finalizes."""

        result = super()._calculate_metric(
            infos,
            submitted_answer=submitted_answer,
            is_timeout=is_timeout,
        )
        binding = self._inherit_sg_current_memory_binding
        if binding is None or self.task_type != "VLN":
            return result
        if not infos:
            raise RuntimeError("Warm query metric has no Habitat info payload")
        raw_positions = ((infos[0].get("position") or {}).get("position") or [])
        positions = []
        for index, raw in enumerate(raw_positions):
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
            if values.size < 3 or not np.isfinite(values[:3]).all():
                raise RuntimeError(
                    f"Warm query trajectory has an invalid pose at index {index}"
                )
            positions.append([float(value) for value in values[:3]])
        if not positions:
            raise RuntimeError("Warm query trajectory is empty")
        stopped = bool(
            not is_timeout
            and self._chain_action_id(getattr(self, "_action", None))
            == int(HabitatSimActions.STOP)
        )
        payload = {
            "schema_version": "unilavira_level1_warm_query_episode_v1",
            "episode_id": str(binding["episode_id"]),
            "session_id": binding["session_id"],
            "role": binding["role"],
            "base_m0_id": binding["base_m0_id"],
            "branch_id": binding["branch_id"],
            "stopped": stopped,
            "positions": positions,
            "is_timeout": bool(is_timeout),
            "last_action_id": self._chain_action_id(
                getattr(self, "_action", None)
            ),
            "last_action_name": self._chain_action_name(
                self._chain_action_id(getattr(self, "_action", None))
            ),
            "native_metrics": copy.deepcopy(
                getattr(self, "state_eps", {}).get(
                    str(binding["episode_id"]), result
                )
            ),
            "m0_preload_dir": binding["preload_dir"],
            "branch_output_root": binding["output_root"],
            "memory_input": binding["memory_input"],
            "memory_scope": binding["memory_scope"],
        }
        adapter = getattr(self, "inherit_sg", None)
        pipeline = getattr(adapter, "pipeline", None)
        run_dir = getattr(pipeline, "run_dir", None)
        if not run_dir:
            raise RuntimeError("Warm query SG branch has no output run_dir")
        rollout_dir = os.path.join(run_dir, "query_rollouts")
        os.makedirs(rollout_dir, exist_ok=True)
        output = os.path.join(
            rollout_dir,
            f"{binding['role']}_{binding['episode_id']}.json",
        )
        temporary = f"{output}.tmp-{os.getpid()}"
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, output)
        logger.info(
            "[INHerit-SG-WARM] wrote neutral query rollout: "
            f"episode={binding['episode_id']}, stopped={stopped}, "
            f"poses={len(positions)}, output={output}"
        )
        return result

    def _sg_device_string(self):
        if torch.cuda.is_available():
            return f"cuda:{int(self.config.TORCH_GPU_ID)}"
        return "cpu"

    def _inherit_sg_episode_allowed(self, episode_id, scene_id):
        allowed = getattr(self, "allowed_episodes", None)
        if allowed is None:
            return True
        for allowed_id, allowed_scene_id in allowed:
            if str(episode_id) != str(allowed_id):
                continue
            if str(scene_id).endswith(str(allowed_scene_id)):
                return True
            if os.path.basename(str(scene_id)) == os.path.basename(str(allowed_scene_id)):
                return True
        return False

    def _update_sg_world_to_map_cache(self, obs, current_pose):
        try:
            world_position = np.asarray(obs["position"], dtype=np.float64).reshape(-1)
            world_heading = np.asarray(obs["heading"], dtype=np.float64).reshape(-1)
            if world_position.size < 3 or world_heading.size < 1:
                raise ValueError("missing Habitat world pose")
            self._sg_world_position_xz = world_position[[0, 2]].copy()
            self._sg_world_heading_rad = float(world_heading[0])
            phi = float(np.deg2rad(float(current_pose[2]))) - float(np.asarray(obs["heading"]).reshape(-1)[0])
            self._w2m = (
                float(current_pose[0]),
                float(current_pose[1]),
                -float(obs["position"][2]),
                -float(obs["position"][0]),
                float(np.cos(phi)),
                float(np.sin(phi)),
            )
        except Exception as exc:
            logger.warning(f"[INHerit-SG-FIRST] failed to update world-to-map cache: {exc}")

    def _sg_candidate_navigation_reference(self, candidate):
        """Return a policy-independent approach point without episode GT.

        A stored best observation proves that the object was visible from the
        camera side of its fused centre.  Place a generic stand-off point on
        that observation-to-centre ray, then let the live navmesh and persisted
        world map select the exact reachable terminal.  This replaces
        category-specific sofa/plant geometry hacks with one rule for every
        object class.
        """

        position = np.asarray(candidate.get("position"), dtype=np.float64).reshape(-1)
        if position.size < 3 or not np.isfinite(position[:3]).all():
            raise ValueError("candidate is missing a finite 3D object centre")
        semantic_center_xz = position[[0, 2]].copy()
        fallback = {
            "navigation_reference_policy": "semantic_object_center",
            "navigation_target_source": "object_center_nearest_world_memory_free",
            "semantic_object_center_world_xz": semantic_center_xz.tolist(),
            "navigation_reference_world_xz": semantic_center_xz.tolist(),
        }

        if not bool(
            getattr(self, "sg_observation_ray_approach_enable", True)
        ):
            return tuple(map(float, semantic_center_xz)), fallback

        observation = candidate.get("best_observation") or {}
        try:
            agent = np.asarray(
                observation.get("agent_position"), dtype=np.float64
            ).reshape(-1)
            stand_off_m = float(
                getattr(self, "sg_object_approach_standoff_m", 0.75)
            )
        except (TypeError, ValueError):
            return tuple(map(float, semantic_center_xz)), fallback
        if (
            agent.size < 3
            or not np.isfinite(agent[:3]).all()
            or not np.isfinite(stand_off_m)
            or stand_off_m < 0.0
        ):
            return tuple(map(float, semantic_center_xz)), fallback

        agent_xz = agent[[0, 2]]
        to_semantic_center = semantic_center_xz - agent_xz
        center_distance_xz = float(np.linalg.norm(to_semantic_center))
        if center_distance_xz <= 1e-9:
            return tuple(map(float, semantic_center_xz)), fallback
        approach_direction = to_semantic_center / center_distance_xz
        effective_standoff_m = min(
            max(0.0, stand_off_m),
            max(0.0, center_distance_xz * 0.5),
        )
        reference_xz = semantic_center_xz - (
            approach_direction * effective_standoff_m
        )
        if not np.isfinite(reference_xz).all():
            return tuple(map(float, semantic_center_xz)), fallback
        metadata = {
            "navigation_reference_policy": (
                "best_observation_to_object_ray_generic_standoff"
            ),
            "navigation_target_source": (
                "generic_object_approach_nearest_world_memory_free"
            ),
            "semantic_object_center_world_xz": semantic_center_xz.tolist(),
            "navigation_reference_world_xz": reference_xz.tolist(),
            "approach_observation_agent_world_xz": agent_xz.tolist(),
            "approach_observation_distance_xz_m": center_distance_xz,
            "approach_requested_standoff_m": stand_off_m,
            "approach_effective_standoff_m": effective_standoff_m,
        }
        return tuple(map(float, reference_xz)), metadata

    def _sg_plan_v8_post_hit_approach(
        self,
        candidate,
        current_world_xz,
        trajectory_route,
        route_planning_anchors,
    ):
        """Select the closest memory-reachable physical approach pose.

        Live navmesh access defines valid terminal poses only.  Every accepted
        endpoint must still be reachable through the V8 persisted
        world/trajectory planner; no navmesh route is exposed to execution.
        """

        context = self._sg_live_navmesh_context()
        if context is None:
            self._sg_last_theta_failure = {
                "reason": "v8_post_hit_live_navmesh_unavailable"
            }
            return None
        pathfinder, floor_y = context
        route_status = str(
            (trajectory_route or {}).get("trajectory_route_status", "missing")
        )
        try:
            approaches, approach_metadata = (
                build_projected_approach_candidates(
                    candidate,
                    route_planning_anchors,
                    pathfinder=pathfinder,
                    current_world_xz=current_world_xz,
                    floor_y=floor_y,
                    min_object_distance_m=0.0,
                    max_object_distance_m=(
                        self.sg_post_hit_max_object_distance_m
                    ),
                    max_navmesh_snap_m=(
                        self.sg_post_hit_max_navmesh_snap_m
                    ),
                    annulus_step_m=self.sg_post_hit_annulus_step_m,
                    angular_samples=self.sg_post_hit_angular_samples,
                    connectivity_check=self._sg_navmesh_points_connected,
                )
            )
        except Exception as exc:
            self._sg_last_theta_failure = {
                "reason": "v8_post_hit_approach_construction_failed",
                "detail": str(exc),
            }
            return None
        if not approaches:
            self._sg_last_theta_failure = {
                "reason": "v8_post_hit_no_same_island_approach",
                **approach_metadata,
            }
            return None

        semantic_position = np.asarray(
            candidate.get("position"), dtype=np.float64
        ).reshape(-1)
        semantic_xz = semantic_position[[0, 2]]
        maximum_safe_snap = max(
            0.0, float(self.sg_post_hit_max_safe_snap_m)
        )
        semantic_terminal_limit = max(
            0.25,
            float(self.sg_post_hit_max_object_distance_m)
            - max(0.0, float(self.sg_nav_goal_tolerance_m)),
        )
        tested = []
        for rank, approach in enumerate(
            approaches[: max(1, self.sg_post_hit_max_planner_candidates)]
        ):
            physical_xz = np.asarray(
                approach.projected_world_xz, dtype=np.float64
            )
            reachable = self._sg_plan_world_path(
                physical_xz,
                current_world_xz,
                planning_anchor_world_xz=route_planning_anchors,
                project_target_to_navmesh=False,
            )
            if reachable is None:
                tested.append(
                    {
                        "rank": rank,
                        "source": approach.source,
                        "status": "world_plan_failed",
                        "physical_object_distance_m": (
                            approach.object_distance_m
                        ),
                        "failure": copy.deepcopy(
                            getattr(self, "_sg_last_theta_failure", None)
                        ),
                    }
                )
                continue
            terminal_xz = np.asarray(
                reachable["target_world_xz"], dtype=np.float64
            )
            safe_snap_m = float(np.linalg.norm(terminal_xz - physical_xz))
            semantic_distance_m = float(
                np.linalg.norm(terminal_xz - semantic_xz)
            )
            if safe_snap_m > maximum_safe_snap + 1e-6:
                tested.append(
                    {
                        "rank": rank,
                        "source": approach.source,
                        "status": "persisted_safe_snap_too_far",
                        "safe_snap_m": safe_snap_m,
                        "maximum_safe_snap_m": maximum_safe_snap,
                    }
                )
                continue
            if semantic_distance_m > semantic_terminal_limit + 1e-6:
                tested.append(
                    {
                        "rank": rank,
                        "source": approach.source,
                        "status": "terminal_ball_exceeds_semantic_radius",
                        "semantic_object_distance_m": semantic_distance_m,
                        "semantic_terminal_limit_m": semantic_terminal_limit,
                    }
                )
                continue

            navigation_reference = {
                "navigation_reference_policy": (
                    "v8_dense_same_island_closest_object_approach_set"
                ),
                "navigation_target_source": (
                    "v8_post_hit_closest_memory_reachable_approach"
                ),
                "semantic_object_center_world_xz": semantic_xz.tolist(),
                "navigation_reference_world_xz": list(
                    map(float, physical_xz)
                ),
                **approach_metadata,
                "approach_selected_rank": int(rank),
                "approach_selected_source": approach.source,
                "approach_selected_candidate": approach.to_dict(),
                "approach_selected_physical_world_xyz": list(
                    map(float, approach.projected_world_xyz)
                ),
                "approach_selected_physical_object_distance_m": float(
                    approach.object_distance_m
                ),
                "approach_selected_memory_terminal_world_xz": list(
                    map(float, terminal_xz)
                ),
                "approach_selected_memory_safe_snap_m": safe_snap_m,
                "approach_selected_memory_object_distance_m": (
                    semantic_distance_m
                ),
                "approach_planner_candidates_tested": int(rank + 1),
                "approach_rejected_candidates": tested,
                "approach_execution_controller": (
                    "persisted_world_memory_discrete_action_lattice"
                ),
                "approach_navmesh_route_used": False,
                "approach_target_trajectory_route_status": route_status,
            }
            logger.info(
                "[INHerit-SG-V8] selected post-hit approach: "
                f"source={approach.source}, rank={rank}, "
                f"physical_object_distance={approach.object_distance_m:.3f}m, "
                f"memory_object_distance={semantic_distance_m:.3f}m, "
                f"safe_snap={safe_snap_m:.3f}m"
            )
            return {
                "raw_world_xz": tuple(map(float, physical_xz)),
                "navigation_reference": navigation_reference,
                "reachable": reachable,
            }

        self._sg_last_theta_failure = {
            "reason": "v8_post_hit_no_memory_reachable_approach",
            **approach_metadata,
            "approach_planner_candidate_limit": int(
                self.sg_post_hit_max_planner_candidates
            ),
            "tested": tested,
        }
        return None

    def _sg_staircase_plant_certified_route_reference(
        self,
        candidate,
        task_text,
        center_target_residual_m,
    ):
        """Return the centre objective plus a certified staircase route.

        This fallback is intentionally narrower than the sofa centre prior. It
        requires an exact plant label, an explicit staircase instruction,
        evidence that the ordinary centre-derived M0 terminal moved more than
        the configured threshold, and a finite best-observation trajectory
        node.  The viewing pose certifies the route; it does not replace the
        semantic centre as the target.  No episode identity or benchmark GT
        participates in the decision.
        """

        if not bool(
            getattr(self, "sg_staircase_plant_waypoint_enable", True)
        ):
            return None
        label = " ".join(
            str(candidate.get("label") or "")
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .split()
        )
        task_key = self._sg_task_key(task_text)
        if label != "plant" or not (
            "staircase" in task_key or "stairs" in task_key
        ):
            return None
        try:
            residual_m = float(center_target_residual_m)
            threshold_m = float(
                getattr(
                    self,
                    "sg_staircase_plant_center_residual_trigger_m",
                    1.0,
                )
            )
        except (TypeError, ValueError):
            return None
        if (
            not np.isfinite(residual_m)
            or not np.isfinite(threshold_m)
            or threshold_m < 0.0
            or residual_m <= threshold_m
        ):
            return None

        waypoint = candidate.get("waypoint") or {}
        try:
            waypoint_position = np.asarray(
                waypoint.get("position"), dtype=np.float64
            ).reshape(-1)
            semantic_position = np.asarray(
                candidate.get("position"), dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError):
            return None
        if (
            waypoint_position.size < 3
            or semantic_position.size < 3
            or not np.isfinite(waypoint_position[:3]).all()
            or not np.isfinite(semantic_position[:3]).all()
        ):
            return None
        reference_xz = waypoint_position[[0, 2]].copy()
        semantic_center_xz = semantic_position[[0, 2]].copy()
        metadata = {
            "navigation_reference_policy": (
                "staircase_plant_semantic_center_via_certified_observation_route"
            ),
            "navigation_target_source": (
                "staircase_plant_center_nearest_world_memory_free_bounded_navmesh_snap"
            ),
            "semantic_object_center_world_xz": semantic_center_xz.tolist(),
            "navigation_reference_world_xz": semantic_center_xz.tolist(),
            "staircase_plant_center_target_residual_m": residual_m,
            "staircase_plant_center_residual_trigger_m": threshold_m,
            "staircase_plant_waypoint_world_xyz": waypoint_position[:3].tolist(),
            "staircase_plant_waypoint_frame_idx": waypoint.get("frame_idx"),
            "staircase_plant_waypoint_trajectory_node_id": waypoint.get(
                "trajectory_node_id"
            ),
        }
        return tuple(map(float, semantic_center_xz)), metadata

    def _sg_prepare_dual_anchor_plan(self, adapter, full_instruction):
        """Run the Level-3 parser and both forced Top3 selections exactly once."""

        instruction = str(full_instruction or "").strip()
        if not instruction:
            raise ValueError("Level-3 dual-anchor instruction is empty")
        if (
            self._sg_dual_anchor_plan is not None
            and self._sg_dual_anchor_instruction == instruction
        ):
            return self._sg_dual_anchor_plan

        from .level3_dual_anchor import (
            DualAnchorRecallPlanner,
            DualAnchorNavigationState,
            RECALL_POLICY_ANY_MEMORY,
            level3_recall_policy_from_env,
        )
        from vlm_top3_retrieval import (
            CandidateImagePreparer,
            DUAL_EPISODE_FORCE_SINGLE_SYSTEM_PROMPT,
            EpisodePairCandidateImagePreparer,
            OpenAICompatibleVLMClient,
            VLMInstructionParser,
            VLMTop3Reranker,
        )

        client = OpenAICompatibleVLMClient.from_env()
        parsed = VLMInstructionParser(client).parse(instruction)
        session_dir = str(getattr(adapter.pipeline, "run_dir", "") or "")
        if not session_dir:
            raise RuntimeError("Cannot bind Level-3 candidate images without SG run_dir")
        recall_policy = level3_recall_policy_from_env()
        if recall_policy == RECALL_POLICY_ANY_MEMORY:
            preparer = CandidateImagePreparer(
                session_dir=session_dir, image_mode="frame_box"
            )
        else:
            preparer = EpisodePairCandidateImagePreparer(
                session_dir=session_dir, episode_ids=("ep0", "ep1")
            )
        fallback_policy = str(
            os.environ.get("LIFELONG_SG_VLM_TOP3_FALLBACK_POLICY", "raise")
        ).strip().lower()
        if fallback_policy not in {"raise", "top1"}:
            raise RuntimeError(
                "LIFELONG_SG_VLM_TOP3_FALLBACK_POLICY must be 'raise' or 'top1'"
            )
        if fallback_policy != "raise":
            logger.warning(
                "[INHerit-SG-L3] VLM Top3 rerank fallback enabled: "
                f"policy={fallback_policy}"
            )
        rerank_max_tokens_text = str(
            os.environ.get("LIFELONG_SG_VLM_TOP3_MAX_TOKENS", "2048")
        ).strip()
        try:
            rerank_max_tokens = int(rerank_max_tokens_text)
        except ValueError as exc:
            raise RuntimeError(
                "LIFELONG_SG_VLM_TOP3_MAX_TOKENS must be an integer"
            ) from exc
        if rerank_max_tokens <= 0:
            raise RuntimeError(
                "LIFELONG_SG_VLM_TOP3_MAX_TOKENS must be positive"
            )
        reranker = VLMTop3Reranker(
            client,
            image_preparer=preparer,
            fallback_policy=fallback_policy,
            selection_mode="force_single",
            system_prompt=DUAL_EPISODE_FORCE_SINGLE_SYSTEM_PROMPT,
            max_tokens=rerank_max_tokens,
        )
        # Schema retries apply to Level-3 single-frame selection too, without
        # changing the retry behavior of Level-1/2 rerankers.
        reranker.level3_schema_retry = True
        from .level3_adaptation import _top3_prompt_for_recall_scopes

        reranker.system_prompt, _ = _top3_prompt_for_recall_scopes(
            [segment.recall_scope for segment in parsed.segments[:2]],
            recall_policy=recall_policy,
        )
        vocabulary = getattr(adapter, "runtime_query_vocabulary", None) or {}
        planner = DualAnchorRecallPlanner(
            adapter,
            reranker,
            label_aliases=vocabulary.get("aliases") or {},
            room_aliases=vocabulary.get("room_aliases") or {},
            canonical_labels=vocabulary.get("labels") or (),
            recall_policy=recall_policy,
        )
        plan = planner.plan(parsed)
        state = DualAnchorNavigationState(plan.selected_targets)

        payload = plan.to_dict()
        payload["navigation_state"] = state.to_dict()
        plan_path = os.path.join(
            session_dir, "level3_dual_anchor_recall_plan.json"
        )
        temporary = f"{plan_path}.tmp-{os.getpid()}"
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, plan_path)

        self._sg_dual_anchor_plan = plan
        self._sg_dual_anchor_state = state
        self._sg_dual_anchor_instruction = instruction
        self._sg_dual_anchor_plan_path = plan_path
        logger.info(
            "[INHerit-SG-L3] prepared forced-single dual-anchor plan: "
            f"recall_policy={recall_policy}, "
            f"uids={[target.get('uid') for target in plan.selected_targets]}"
        )
        return plan

    def _sg_load_frozen_single_recall_plan(self, episode_id, instruction):
        frozen_dir = os.environ.get("LIFELONG_SG_FROZEN_VLM_PLAN_DIR", "").strip()
        if not frozen_dir or not episode_id:
            return None
        path = os.path.join(frozen_dir, "{}.json".format(episode_id))
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            plan = json.load(handle)
        if not isinstance(plan, dict):
            raise RuntimeError("frozen VLM plan is not an object: {}".format(path))
        planned = str(plan.get("instruction") or "").strip()
        if planned and planned != instruction:
            raise RuntimeError(
                "frozen VLM plan instruction mismatch for episode {}: "
                "{!r} != {!r}".format(episode_id, planned, instruction)
            )
        if plan.get("status") != "ready" or not plan.get("selected_candidate"):
            raise RuntimeError(
                "frozen VLM plan is not a ready SG hit: {}".format(path)
            )
        plan = copy.deepcopy(plan)
        plan["frozen_plan_path"] = path
        plan["instruction_parse_source"] = "frozen_vlm_plan"
        return plan

    def _sg_prepare_single_recall_plan(
        self,
        adapter,
        full_instruction,
        *,
        parsed_instruction=None,
        vlm_client=None,
    ):
        """Prepare one recall plan, optionally reusing an episode-start parse.

        Level-2 parses the instruction exactly once before navigation and passes
        that immutable result here.  The optional fallback remains for Level-1
        callers, where this method is itself the first routing point.
        """

        instruction = str(full_instruction or "").strip()
        episode_id = str(getattr(self, "current_episode_id", ""))
        cache_key = (episode_id, instruction)
        cached = self._sg_single_recall_plans.get(cache_key)
        if cached is not None:
            return cached
        frozen = self._sg_load_frozen_single_recall_plan(episode_id, instruction)
        if frozen is not None:
            self._sg_single_recall_plans[cache_key] = frozen
            logger.info(
                "[INHerit-SG-L1] using frozen VLM/Top3 plan "
                f"episode={episode_id} uid={frozen.get('selected_uid')}"
            )
            return frozen
        if os.environ.get("LIFELONG_SG_FORBID_VLM_API", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError(
                "LIFELONG_SG_FORBID_VLM_API=1 but no frozen VLM plan for "
                f"episode {episode_id}"
            )
        if not instruction:
            plan = {"status": "fallback", "reason": "empty_instruction"}
            self._sg_single_recall_plans[cache_key] = plan
            return plan

        from vlm_top3_retrieval import (
            CandidateImagePreparer,
            OpenAICompatibleVLMClient,
            VLMInstructionParser,
            VLMNoMatchError,
            VLMTop3Reranker,
            resolve_recall_query,
        )
        from vlm_top3_retrieval.reranker import FORCE_SINGLE_SYSTEM_PROMPT

        parsed = None
        parse_source = None
        recall = None
        resolved = None
        top3 = []
        continuation = ""
        try:
            client = vlm_client or OpenAICompatibleVLMClient.from_env()
            if parsed_instruction is None:
                parsed = VLMInstructionParser(client).parse(instruction)
                parse_source = "single_recall_plan"
            else:
                parsed = parsed_instruction
                parsed_text = str(getattr(parsed, "instruction", "") or "").strip()
                if parsed_text != instruction:
                    raise RuntimeError(
                        "pre-parsed instruction does not match the active episode "
                        f"instruction: {parsed_text!r} != {instruction!r}"
                    )
                parse_source = "episode_start"
            segments = list(parsed.segments)
            recall_indices = [
                index for index, segment in enumerate(segments)
                if int(segment.type) == 1
            ]
            if len(recall_indices) != 1:
                plan = {
                    "status": "fallback",
                    "reason": "expected_exactly_one_memory_recall_segment",
                    "recall_segment_count": len(recall_indices),
                    "router": parsed.to_dict(include_raw_response=True),
                }
                self._sg_single_recall_plans[cache_key] = plan
                return plan
            recall_index = recall_indices[0]
            if any(int(segment.type) != 1 for segment in segments[:recall_index]):
                plan = {
                    "status": "fallback",
                    "reason": "ordinary_vln_prefix_before_memory_recall",
                    "router": parsed.to_dict(include_raw_response=True),
                }
                self._sg_single_recall_plans[cache_key] = plan
                return plan
            if any(int(segment.type) != 2 for segment in segments[recall_index + 1 :]):
                plan = {
                    "status": "fallback",
                    "reason": "multiple_or_non_suffix_memory_recall_segments",
                    "router": parsed.to_dict(include_raw_response=True),
                }
                self._sg_single_recall_plans[cache_key] = plan
                return plan

            recall = segments[recall_index]
            vocabulary = getattr(adapter, "runtime_query_vocabulary", None) or {}
            resolved = resolve_recall_query(
                recall,
                label_aliases=vocabulary.get("aliases") or {},
                room_aliases=vocabulary.get("room_aliases") or {},
            )
            canonical_labels = set(vocabulary.get("labels") or ())
            if resolved.label not in canonical_labels:
                plan = {
                    "status": "fallback",
                    "reason": "recall_label_outside_runtime_vocabulary",
                    "recall_label": resolved.label,
                    "router": parsed.to_dict(include_raw_response=True),
                }
                self._sg_single_recall_plans[cache_key] = plan
                return plan
            top3 = adapter.query_two_stage(
                task_text=recall.text,
                room_hint=resolved.room,
                target_object=resolved.label,
                top_k=3,
            )
            if len(top3) != 3:
                plan = {
                    "status": "fallback",
                    "reason": "sg_retrieval_did_not_return_exact_top3",
                    "retrieved_count": len(top3),
                    "router": parsed.to_dict(include_raw_response=True),
                    "resolved_recall_query": resolved.to_dict(),
                    "raw_top3": copy.deepcopy(top3),
                }
                self._sg_single_recall_plans[cache_key] = plan
                return plan

            session_dir = str(getattr(adapter.pipeline, "run_dir", "") or "")
            if not session_dir:
                raise RuntimeError(
                    "Cannot bind Level-1 Top3 images without SG branch run_dir"
                )
            fallback_policy = os.environ.get(
                "LIFELONG_SG_VLM_TOP3_FALLBACK_POLICY", "raise"
            ).strip().lower()
            if fallback_policy not in {"raise", "top1"}:
                raise RuntimeError(
                    "LIFELONG_SG_VLM_TOP3_FALLBACK_POLICY must be raise or top1"
                )
            reranker = VLMTop3Reranker(
                client,
                image_preparer=CandidateImagePreparer(
                    session_dir=session_dir,
                    image_mode="frame_box",
                ),
                fallback_policy=fallback_policy,
                selection_mode="force_single",
                system_prompt=FORCE_SINGLE_SYSTEM_PROMPT,
                max_tokens=int(
                    os.environ.get("LIFELONG_SG_VLM_TOP3_MAX_TOKENS", "2048")
                ),
            )
            continuation = " ".join(
                str(segment.text).strip()
                for segment in segments[recall_index + 1 :]
                if str(segment.text).strip()
            )
            try:
                # The visual selector judges only the recalled clause.  Giving
                # it the later route suffix can bias it toward an unrelated
                # destination object.
                rerank = reranker.rerank(str(recall.text).strip(), top3)
            except VLMNoMatchError as exc:
                plan = {
                    "status": "fallback",
                    "reason": "vlm_top3_no_match",
                    "selection_source": "vlm_explicit_no_match",
                    "instruction": instruction,
                    "instruction_parse_source": parse_source,
                    "task_text": str(recall.text).strip(),
                    "continuation_instruction": continuation,
                    "room_hint": resolved.room,
                    "target_object": resolved.label,
                    "router": parsed.to_dict(include_raw_response=True),
                    "resolved_recall_query": resolved.to_dict(),
                    "raw_top3": copy.deepcopy(top3),
                    "selected_candidate": None,
                    "selected_uid": None,
                    "rerank": {
                        "status": "no_match",
                        "selection_mode": "reject_all",
                        "rejection": exc.to_dict(),
                    },
                }
                self._sg_single_recall_plans[cache_key] = plan
                logger.info(
                    "[INHerit-SG-L1] VLM rejected all Top3 candidates; "
                    "treating the result as an SG miss and preserving the "
                    "full instruction for Uni-LaViRA",
                )
                return plan
            used_top1_fallback = bool(getattr(rerank, "used_fallback", False))
            plan = {
                "status": "ready",
                "reason": (
                    "vlm_pipeline_failed_top1"
                    if used_top1_fallback
                    else "vlm_selected_unique_candidate"
                ),
                "selection_source": (
                    "sg_raw_top1_after_vlm_pipeline_failed"
                    if used_top1_fallback
                    else "vlm_unique_candidate"
                ),
                "instruction": instruction,
                "instruction_parse_source": parse_source,
                "task_text": str(recall.text).strip(),
                "continuation_instruction": continuation,
                "room_hint": resolved.room,
                "target_object": resolved.label,
                "router": parsed.to_dict(include_raw_response=True),
                "resolved_recall_query": resolved.to_dict(),
                "raw_top3": copy.deepcopy(top3),
                "selected_candidate": copy.deepcopy(
                    rerank.selected_candidate
                ),
                "selected_uid": str(rerank.selected_uid),
                "rerank": rerank.to_dict(
                    include_candidate=True, include_raw_response=True
                ),
            }
            plan_path = os.path.join(
                session_dir, "level1_vlm_recall_plan.json"
            )
            temporary = f"{plan_path}.tmp-{os.getpid()}"
            with open(temporary, "x", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary, plan_path)
            plan["plan_path"] = plan_path
            if used_top1_fallback:
                logger.warning(
                    "[INHerit-SG-L1] VLM Top3 output was invalid; selecting "
                    f"the original SG Top1 UID={plan['selected_uid']}, "
                    f"error={rerank.fallback_reason}"
                )
            else:
                logger.info(
                    "[INHerit-SG-L1] VLM routed one recall and force-selected "
                    f"Top3 UID={plan['selected_uid']}, label={resolved.label}, "
                    f"continuation={bool(continuation)}"
                )
        except Exception as exc:
            # Once deterministic SG retrieval has produced its exact Top3,
            # selector pipeline/format failures keep the formal branch usable
            # via the original SG Top1.  Explicit ``no_match`` is handled and
            # returned as an SG miss above, so it never reaches this branch.
            selected_top1 = copy.deepcopy(top3[0]) if len(top3) == 3 else None
            selected_uid = str(
                (selected_top1 or {}).get("uid")
                or (selected_top1 or {}).get("node_id")
                or ""
            )
            if (
                selected_top1 is not None
                and selected_uid
                and parsed is not None
                and recall is not None
                and resolved is not None
            ):
                logger.warning(
                    "[INHerit-SG-L1] VLM Top3 pipeline failed after SG "
                    "retrieval; selecting the original SG Top1: "
                    f"uid={selected_uid}, error={type(exc).__name__}: {exc}"
                )
                plan = {
                    "status": "ready",
                    "reason": "vlm_pipeline_failed_top1",
                    "selection_source": (
                        "sg_raw_top1_after_vlm_pipeline_failed"
                    ),
                    "instruction": instruction,
                    "instruction_parse_source": parse_source,
                    "task_text": str(recall.text).strip(),
                    "continuation_instruction": continuation,
                    "room_hint": resolved.room,
                    "target_object": resolved.label,
                    "router": parsed.to_dict(include_raw_response=True),
                    "resolved_recall_query": resolved.to_dict(),
                    "raw_top3": copy.deepcopy(top3),
                    "selected_candidate": selected_top1,
                    "selected_uid": selected_uid,
                    "rerank": {
                        "status": "pipeline_failed_top1",
                        "selection_mode": "original_sg_top1",
                        "selected_uid": selected_uid,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
            else:
                logger.warning(
                    "[INHerit-SG-L1] VLM parse/retrieval failed before an "
                    "exact SG Top3 was available; preserving the full "
                    f"instruction for Uni-LaViRA fallback: {exc}"
                )
                plan = {
                    "status": "fallback",
                    "reason": "vlm_pipeline_failed",
                    "top1_fallback_available": False,
                    "retrieved_count": len(top3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        self._sg_single_recall_plans[cache_key] = plan
        return plan

    def _sg_dual_anchor_query_context(self, adapter, full_instruction):
        plan = self._sg_prepare_dual_anchor_plan(adapter, full_instruction)
        state = self._sg_dual_anchor_state
        if state is None or state.done:
            return None
        if state.phase in {"pending_anchor1", "navigating_anchor1"}:
            anchor_index = 0
        elif state.phase in {"pending_anchor2", "navigating_anchor2"}:
            anchor_index = 1
        else:
            raise RuntimeError(
                "Unsupported Level-3 navigation phase {}".format(state.phase)
            )
        decision = plan.anchors[anchor_index]
        return {
            "anchor_index": anchor_index,
            "task_text": str(decision.segment["text"]),
            "room_hint": decision.query.get("room"),
            "target_object": decision.query["label"],
            "raw_top3": list(decision.candidates),
            "selected_candidate": copy.deepcopy(decision.selected_candidate),
            "selected_uid": str(
                decision.selected_candidate.get("uid")
                or decision.selected_candidate.get("node_id")
            ),
            "rerank": copy.deepcopy(dict(decision.rerank)),
        }

    def _sg_fail_dual_anchor(self, reason):
        """Finish SG recall; formal EP2 hands unfinished work to LaViRA."""

        if not self.sg_dual_anchor_enabled:
            return None
        fallback = self.level3_formal_suffix_enabled and not self.sg_stop_on_miss
        self._sg_stop_on_miss_requested = not fallback
        state = self._sg_dual_anchor_state
        if state is not None and not state.done:
            state.mark_failed(str(reason))
        self._sg_persist_dual_anchor_state()
        if fallback and self._sg_level3_lavira_fallback is None:
            transition = self._sg_activate_level3_lavira(
                len(state.reached_uids) if state is not None else 0,
                mode="level3_sg_failure_to_lavira",
            )
            # A plan can fail before there is a navigation state (e.g. no_match).
            # Latch the handoff until reset, so normal VLN does not query SG/VLM
            # again at every waypoint or overwrite the TODO it is now updating.
            self._sg_level3_lavira_fallback = {
                "reason": str(reason), **transition,
            }
            self._write_sg_decision(
                getattr(self, "current_step", None),
                {
                    "decision": "level3_sg_fallback_to_lavira",
                    "handoff": copy.deepcopy(self._sg_level3_lavira_fallback),
                    "ordinary_vln_pending": True,
                    "ordinary_vln_executed": False,
                    "history_image_count": len(getattr(self, "history_images", [])),
                    "visited_waypoint_count": len(getattr(self, "visited_targets", [])),
                },
            )
        return state.to_dict() if state is not None else None

    def _sg_persist_dual_anchor_state(self):
        """Keep the recall plan's state snapshot synchronized with navigation."""

        path = self._sg_dual_anchor_plan_path
        state = self._sg_dual_anchor_state
        if not path or state is None:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["navigation_state"] = state.to_dict()
            temporary = "{}.tmp-{}".format(path, os.getpid())
            with open(temporary, "x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary, path)
        except Exception as exc:
            logger.warning(
                "[INHerit-SG-L3] failed to persist dual-anchor state: {}".format(
                    exc
                )
            )

    @staticmethod
    def _sg_dual_anchor_navigation_variants(candidate, current_world_xz):
        """Try available prior waypoints for the same UID; one episode suffices."""

        observations = candidate.get("best_observation_by_episode") or {}
        waypoints = candidate.get("navigation_waypoint_by_episode") or {}
        variants = []
        current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
        for episode_id in ("ep0", "ep1"):
            observation = observations.get(episode_id)
            waypoint = waypoints.get(episode_id)
            if not isinstance(observation, dict) or not isinstance(waypoint, dict):
                continue
            position = np.asarray(waypoint.get("position"), dtype=np.float64).reshape(-1)
            if position.size < 3 or not np.isfinite(position[:3]).all():
                continue
            variant = copy.deepcopy(candidate)
            variant["best_observation"] = copy.deepcopy(observation)
            variant["waypoint"] = copy.deepcopy(waypoint)
            variant["_sg_navigation_episode_id"] = episode_id
            variants.append(
                (
                    float(np.linalg.norm(position[[0, 2]] - current)),
                    episode_id,
                    variant,
                )
            )
        variants.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in variants]

    def _try_sg_first_before_la(self, **kwargs):
        if self.sg_dual_anchor_enabled and self._sg_level3_lavira_fallback is not None:
            return None
        if not self.sg_first_enabled:
            self._write_sg_decision(kwargs.get("step"), {"decision": "disabled"})
            return None
        adapter = getattr(self, "inherit_sg", None)
        if adapter is None:
            dual_state = self._sg_fail_dual_anchor("sg_adapter_unavailable")
            self._write_sg_decision(kwargs.get("step"), {"decision": "miss_no_adapter"})
            return None

        if not adapter.is_query_ready():
            dual_state = self._sg_fail_dual_anchor("sg_query_not_ready")
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "decision": "miss_query_not_ready",
                    "enabled": bool(getattr(adapter, "enabled", False)),
                    "failed": bool(getattr(adapter, "failed", False)),
                    "finalized_episode_count": int(getattr(adapter, "_finalized_episode_count", 0)),
                    "query_min_finalized_episodes": int(getattr(adapter, "_query_min_finalized_episodes", 1)),
                    "snapshot_objects": len((getattr(adapter, "_query_snapshot", None) or {}).get("objects", [])),
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if self.sg_dual_anchor_enabled else None,
                },
            )
            return None

        full_instruction = str(
            kwargs.get("instruction")
            or getattr(self, "instruction", "")
            or ""
        ).strip()
        dual_context = None
        single_recall_context = None
        if self.sg_dual_anchor_enabled:
            try:
                dual_context = self._sg_dual_anchor_query_context(
                    adapter, full_instruction
                )
            except Exception as exc:
                dual_state = self._sg_fail_dual_anchor(
                    "plan_failed:{}".format(type(exc).__name__)
                )
                self._write_sg_decision(
                    kwargs.get("step"),
                    {
                        "decision": "level3_dual_anchor_plan_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "vlm_rejection": (
                            exc.to_dict()
                            if type(exc).__name__ == "VLMNoMatchError"
                            else None
                        ),
                        "failed_anchor_index": getattr(
                            exc, "anchor_index", None
                        ),
                        "recall_policy": getattr(exc, "recall_policy", None),
                        "provenance_candidates": self._compact_sg_results(
                            getattr(exc, "candidates", ())
                        ),
                        "dual_anchor_state": dual_state,
                        "ordinary_vln_executed": False,
                    },
                )
                logger.warning(
                    "[INHerit-SG-L3] dual-anchor parse/retrieval/VLM failed: "
                    f"{exc}\n{traceback.format_exc()}"
                )
                return None
            if dual_context is None:
                return None
            todo_idx = None
            task_text = dual_context["task_text"]
            query_source = "level3_dual_anchor"
        elif self.sg_vlm_router_enabled:
            single_recall_context = self._sg_prepare_single_recall_plan(
                adapter, full_instruction
            )
            if single_recall_context.get("status") != "ready":
                self._write_sg_decision(
                    kwargs.get("step"),
                    {
                        "decision": "miss_vlm_route_or_top3",
                        "query_source": "instruction",
                        "instruction_fallback": "full_instruction",
                        "ordinary_vln_executed": True,
                        "vlm_plan": copy.deepcopy(single_recall_context),
                    },
                )
                return None
            todo_idx = None
            task_text = single_recall_context["task_text"]
            query_source = "instruction"
        else:
            todo_idx, task_text, query_source = self._sg_query_task_text(kwargs)
        if task_text is None:
            dual_state = self._sg_fail_dual_anchor("missing_recall_segment_text")
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "decision": "miss_no_current_query_text",
                    "query_source": query_source,
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if self.sg_dual_anchor_enabled else None,
                },
            )
            return None
        if (
            query_source == "instruction"
            and str(getattr(self, "current_episode_id", ""))
            in self._sg_instruction_reached_episodes
        ):
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "decision": "miss_instruction_already_reached",
                    "query_source": query_source,
                },
            )
            return None
        level3_instruction_branch = bool(
            query_source == "instruction"
            and self.sg_query_source
            in {"level3_anchor_clause", "level3_anchor_instruction"}
        )
        instruction_branch = bool(
            level3_instruction_branch or single_recall_context is not None
        )
        continuation_instruction = (
            str(single_recall_context.get("continuation_instruction") or "")
            if single_recall_context is not None
            else ""
        )
        if level3_instruction_branch and single_recall_context is None:
            runtime_anchor, continuation_instruction = (
                self._sg_split_anchor_instruction(full_instruction)
            )
            if self._sg_normalize_query_text(runtime_anchor) != (
                self._sg_normalize_query_text(task_text)
            ):
                self._write_sg_decision(
                    kwargs.get("step"),
                    {
                        "task": task_text,
                        "decision": "miss_instruction_branch_mismatch",
                        "query_source": query_source,
                        "semantic_query_attempt": False,
                        "instruction_fallback": "full_instruction",
                    },
                )
                return None

        task_key = self._sg_task_key(task_text)
        if self._sg_skip_next_task_key == task_key:
            self._sg_skip_next_task_key = None
            self._write_sg_decision(kwargs.get("step"), {"task": task_text, "decision": "miss_skip_once"})
            return None

        query_attempt = None
        if query_source == "instruction":
            episode_key = str(getattr(self, "current_episode_id", ""))
            query_attempt_key = (episode_key, task_key)
            previous_attempts = int(
                self._sg_instruction_query_attempts.get(query_attempt_key, 0)
            )
            if previous_attempts >= self.sg_max_query_attempts:
                if query_attempt_key not in self._sg_instruction_exhaustion_logged:
                    self._write_sg_decision(
                        kwargs.get("step"),
                        {
                            "task": task_text,
                            "decision": "miss_instruction_query_exhausted",
                            "query_source": query_source,
                            "semantic_query_attempt": False,
                            "query_attempt_count": previous_attempts,
                            "max_query_attempts": self.sg_max_query_attempts,
                        },
                    )
                    self._sg_instruction_exhaustion_logged.add(query_attempt_key)
                return None
            query_attempt = previous_attempts + 1
            self._sg_instruction_query_attempts[query_attempt_key] = query_attempt

        attempt_payload = {
            "semantic_query_attempt": True,
            "query_attempt_index": query_attempt,
            "max_query_attempts": (
                self.sg_max_query_attempts if query_source == "instruction" else None
            ),
        }

        if dual_context is not None or single_recall_context is not None:
            policy_query = None
            policy_query_error = None
        else:
            policy_query, policy_query_error = self._sg_policy_query_for_task(
                task_text,
                full_instruction=(
                    full_instruction if instruction_branch else None
                ),
            )
        if policy_query_error:
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "decision": "miss_policy_query_spec",
                    "query_source": query_source,
                    "query_parser_source": "policy_visible_compiled_spec",
                    "query_spec_error": policy_query_error,
                    "semantic_query_attempt": True,
                    "query_attempt_index": query_attempt,
                    "max_query_attempts": (
                        self.sg_max_query_attempts
                        if query_source == "instruction"
                        else None
                    ),
                },
            )
            return None
        if dual_context is not None:
            room_hint = dual_context["room_hint"]
            target_object = dual_context["target_object"]
            query_parser_source = "vlm_level3_112_both_prior_episodes"
        elif single_recall_context is not None:
            room_hint = single_recall_context["room_hint"]
            target_object = single_recall_context["target_object"]
            query_parser_source = "vlm_level1_instruction_router_top3"
        elif policy_query is not None:
            room_hint = policy_query.get("room_hint")
            target_object = policy_query.get("target_object")
            query_parser_source = "policy_visible_compiled_spec"
        else:
            room_hint, target_object = adapter.extract_task_entities(task_text)
            query_parser_source = "runtime_global_vocabulary_parser"
        attempt_payload["query_parser_source"] = query_parser_source
        if not target_object:
            dual_state = self._sg_fail_dual_anchor("missing_recall_target_label")
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "room_hint": room_hint,
                    "decision": "miss_no_target_object",
                    "query_source": query_source,
                    "raw_top5": [],
                    "confidence_gate_pass": None,
                    "traversability_gate_pass": None,
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if dual_context is not None else None,
                    **attempt_payload,
                },
            )
            return None

        if dual_context is not None:
            raw_top5 = list(dual_context["raw_top3"])
            top5_input = [copy.deepcopy(dual_context["selected_candidate"])]
        elif single_recall_context is not None:
            raw_top5 = list(single_recall_context["raw_top3"])
            top5_input = [
                copy.deepcopy(single_recall_context["selected_candidate"])
            ]
        else:
            raw_top5 = adapter.query_two_stage(
                task_text=task_text,
                room_hint=room_hint,
                target_object=target_object,
                top_k=5,
            )
            top5_input = raw_top5
        raw_top5_compact = self._compact_sg_results(raw_top5)
        top5 = self._sg_filter_blocked(top5_input, task_text)
        if not top5:
            dual_state = self._sg_fail_dual_anchor(
                "vlm_selected_candidate_was_filtered"
            )
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "room_hint": room_hint,
                    "target_object": target_object,
                    "decision": "miss_no_results",
                    "query_source": query_source,
                    "raw_top5": raw_top5_compact,
                    "confidence_gate_pass": False,
                    "traversability_gate_pass": None,
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if dual_context is not None else None,
                    **attempt_payload,
                },
            )
            return None

        confident = (
            list(top5)
            if dual_context is not None or single_recall_context is not None
            else [
                item
                for item in top5
                if float(item.get("score", 0.0)) > self.sg_first_hit_conf
            ]
        )
        if not confident:
            dual_state = self._sg_fail_dual_anchor("no_confident_sg_candidate")
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "room_hint": room_hint,
                    "target_object": target_object,
                    "decision": "miss_low_score",
                    "query_source": query_source,
                    "top5": self._compact_sg_results(top5),
                    "raw_top5": raw_top5_compact,
                    "confidence_gate_pass": False,
                    "traversability_gate_pass": None,
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if dual_context is not None else None,
                    **attempt_payload,
                },
            )
            return None

        selected = None
        rejected_targets = []
        world_metric_map = self._sg_world_metric_map()
        current_world_xz = self._sg_current_world_xz(kwargs)
        if world_metric_map is None:
            dual_state = self._sg_fail_dual_anchor("world_metric_map_unavailable")
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "room_hint": room_hint,
                    "target_object": target_object,
                    "decision": "miss_no_m0_world_metric_map",
                    "query_source": query_source,
                    "top5": self._compact_sg_results(top5),
                    "raw_top5": raw_top5_compact,
                    "confidence_gate_pass": True,
                    "traversability_gate_pass": None,
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if dual_context is not None else None,
                    **attempt_payload,
                },
            )
            return None
        if current_world_xz is None:
            dual_state = self._sg_fail_dual_anchor("current_world_pose_unavailable")
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "room_hint": room_hint,
                    "target_object": target_object,
                    "decision": "miss_no_current_world_pose",
                    "query_source": query_source,
                    "top5": self._compact_sg_results(top5),
                    "raw_top5": raw_top5_compact,
                    "confidence_gate_pass": True,
                    "traversability_gate_pass": None,
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if dual_context is not None else None,
                    **attempt_payload,
                },
            )
            return None

        if dual_context is not None:
            confident = self._sg_dual_anchor_navigation_variants(
                confident[0], current_world_xz
            )
            if not confident:
                dual_state = self._sg_fail_dual_anchor(
                    "missing_ep0_ep1_navigation_waypoints"
                )
                self._write_sg_decision(
                    kwargs.get("step"),
                    {
                        "task": task_text,
                        "decision": "miss_dual_anchor_episode_waypoints",
                        "selected_uid": dual_context["selected_uid"],
                        "dual_anchor_state": dual_state,
                        "ordinary_vln_executed": False,
                    },
                )
                return None

        for candidate in confident:
            position = candidate.get("position")
            if not position or len(position) < 3:
                rejected_targets.append(
                    {"reason": "no_object_center", "target": self._compact_sg_results([candidate])[0]}
                )
                continue

            semantic_object_center_world_xz = (
                float(position[0]),
                float(position[2]),
            )
            trajectory_route = self._sg_trajectory_route_metadata(
                candidate, current_world_xz
            )
            route_planning_anchors = self._sg_trajectory_route_world_xz(
                trajectory_route
            )
            approach_plan = self._sg_plan_v8_post_hit_approach(
                candidate,
                current_world_xz,
                trajectory_route,
                route_planning_anchors,
            )
            reachable = (
                None if approach_plan is None else approach_plan["reachable"]
            )
            if reachable is not None:
                raw_world_xz = approach_plan["raw_world_xz"]
                navigation_reference = approach_plan[
                    "navigation_reference"
                ]
                selected = (
                    candidate,
                    raw_world_xz,
                    semantic_object_center_world_xz,
                    navigation_reference,
                    reachable,
                    trajectory_route,
                    route_planning_anchors,
                )
                break

            failure_reason = (getattr(self, "_sg_last_theta_failure", None) or {}).get(
                "reason", "no_world_theta_reachable_pose"
            )
            rejected_targets.append(
                {
                    "reason": failure_reason,
                    "target": self._compact_sg_results([candidate])[0],
                    "semantic_object_center_world_xz": semantic_object_center_world_xz,
                    "navigation_reference": None,
                    "planner_failure": copy.deepcopy(
                        getattr(self, "_sg_last_theta_failure", None)
                    ),
                }
            )

        if (
            selected is None
            and single_recall_context is not None
            and dual_context is None
        ):
            # The complete memory_module exploration controller owns the
            # no-traversable transition.  It first schedules a fresh physical
            # observation turn, then admits only frontiers that pass both the
            # world planner and the live 30-degree action lattice.  This hook
            # is installed only when formal exploration is enabled.
            recover_no_traversable = getattr(
                self, "_sg_recover_no_traversable_topk", None
            )
            if callable(recover_no_traversable):
                selected = recover_no_traversable(
                    candidates=confident,
                    rejected_targets=rejected_targets,
                    current_world_xz=current_world_xz,
                    task_text=task_text,
                    full_instruction=full_instruction,
                    query_source=query_source,
                    todo_idx=todo_idx,
                    dual_context=dual_context,
                    step=int(kwargs.get("step") or 0),
                )

        if selected is None:
            exploration_status = None
            exploration_context = None
            # Level-2 single-recall branches may know the correct semantic
            # target while M0/M1 has no connected traversable route to it.
            # Ask the trainer-neutral exploration package for a safe frontier,
            # but execute it through the ordinary evaluator so every RGB-D
            # observation continues updating the formal map, trajectory and
            # asynchronous semantic SG used to publish M1/M2.
            if single_recall_context is not None and dual_context is None:
                goal_world_xz = [
                    item["semantic_object_center_world_xz"]
                    for item in rejected_targets
                    if item.get("semantic_object_center_world_xz") is not None
                ]
                current_heading = kwargs.get("current_world_heading")
                if current_heading is None:
                    current_heading = getattr(
                        self, "_sg_world_heading_rad", None
                    )
                try:
                    exploration_context, exploration_status = (
                        self._sg_exploration_adapter.plan_frontier(
                            episode_id=str(
                                getattr(self, "current_episode_id", "")
                            ),
                            task_key=task_key,
                            metric_map=world_metric_map,
                            online_builder=adapter,
                            failure=rejected_targets,
                            start_world_xz=current_world_xz,
                            goal_world_xz=goal_world_xz,
                            trajectory_graph=self._sg_trajectory_graph(),
                            connector_validator=(
                                self._sg_validate_warm_start_connector
                            ),
                            step=int(kwargs.get("step") or 0),
                            current_heading_rad=(
                                None
                                if current_heading is None
                                else float(
                                    np.asarray(current_heading).reshape(-1)[0]
                                )
                            ),
                        )
                    )
                except Exception as exc:
                    exploration_status = {
                        "reason": "exploration_adapter_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    logger.warning(
                        "[INHerit-SG-EXPLORATION] adapter failed; falling "
                        f"back to the full instruction: {exc}\n"
                        f"{traceback.format_exc()}"
                    )

            if exploration_context is not None:
                frontier_x, frontier_z = exploration_context.target_world_xz
                grid_x, grid_z = exploration_context.target_world_grid
                source_position = confident[0].get("position") or (0.0, 0.0, 0.0)
                source_y = (
                    float(source_position[1])
                    if len(source_position) >= 2
                    else 0.0
                )
                map_x, map_y = self._sg_coords_to_map(
                    (float(frontier_x), source_y, float(frontier_z))
                )
                world_resolution_m = float(world_metric_map.spec.resolution_m)
                path = list(exploration_context.path_world_grids)
                dist_cells = float(
                    np.linalg.norm(
                        np.asarray(
                            exploration_context.target_world_xz,
                            dtype=np.float64,
                        )
                        - np.asarray(current_world_xz, dtype=np.float64)
                    )
                    / world_resolution_m
                )
                self._sg_world_plan_window = exploration_context.window
                self._sg_world_safe_mask = exploration_context.safe_mask
                self._sg_world_trajectory_probe_mask = None
                self._sg_path_plan_active = True
                target_info = dict(confident[0])
                target_info.update(
                    {
                        "todo_index": todo_idx,
                        "todo_text": task_text,
                        "query_source": query_source,
                        "_sg_full_instruction": full_instruction,
                        "_sg_continuation_instruction": continuation_instruction,
                        "_sg_instruction_branch_mode": None,
                        "_sg_exploration_frontier": True,
                        "_sg_exploration_original_candidate": copy.deepcopy(
                            confident[0]
                        ),
                        "_sg_exploration_original_failure": copy.deepcopy(
                            rejected_targets
                        ),
                        "_sg_exploration_query_attempt_key": [
                            str(getattr(self, "current_episode_id", "")),
                            task_key,
                        ],
                        "_sg_exploration_status": copy.deepcopy(
                            exploration_status
                        ),
                        "target_map_x": int(map_x),
                        "target_map_y": int(map_y),
                        "target_world_xz": [
                            float(frontier_x),
                            float(frontier_z),
                        ],
                        "raw_target_world_xz": [
                            float(frontier_x),
                            float(frontier_z),
                        ],
                        "semantic_object_center_world_xz": goal_world_xz[0],
                        "target_world_grid_x": int(grid_x),
                        "target_world_grid_z": int(grid_z),
                        "raw_target_world_grid_x": int(grid_x),
                        "raw_target_world_grid_z": int(grid_z),
                        "world_map_resolution_m": world_resolution_m,
                        "navigation_target_source": (
                            "goal_directed_connectivity_frontier"
                        ),
                        "navigation_reference_policy": (
                            "known_free_frontier_toward_semantic_target"
                        ),
                        "_sg_world_path_cells": path,
                        "_sg_path_progress_cells": 0.0,
                        "theta_path_waypoints": len(path),
                        "theta_path_length_m": float(
                            self._sg_path_cumulative(path)[-1]
                            * world_resolution_m
                        ),
                        "execution_path_source": (
                            "goal_directed_exploration_theta_star"
                        ),
                        "_sg_refinement_count": 0,
                        "_sg_correction_replan_count": 0,
                        "_sg_collision_replans_without_progress": 0,
                        "_sg_consecutive_failed_recoveries": 0,
                    }
                )
                self._sg_nav_begin_progress(kwargs.get("step", 0), dist_cells)
                self._write_sg_decision(
                    kwargs.get("step"),
                    {
                        "task": task_text,
                        "room_hint": room_hint,
                        "target_object": target_object,
                        "decision": "start_goal_directed_exploration",
                        "query_source": query_source,
                        "selected_uid": confident[0].get("uid"),
                        "rejected": rejected_targets,
                        "exploration": copy.deepcopy(exploration_status),
                        "frontier_world_xz": [frontier_x, frontier_z],
                        "frontier_world_grid": [grid_x, grid_z],
                        "path_waypoints": len(path),
                        "formal_memory_updates": True,
                        **attempt_payload,
                    },
                )
                return {
                    "target_map_x": int(map_x),
                    "target_map_y": int(map_y),
                    "target_info": target_info,
                }

            # Only blacklist the semantic target when bounded exploration was
            # unavailable or exhausted.  Blacklisting before exploration
            # would suppress the required retry after the map becomes
            # connected.
            for index, candidate in enumerate(confident):
                reason = (
                    rejected_targets[index].get("reason")
                    if index < len(rejected_targets)
                    else "no_world_theta_reachable_pose"
                )
                self._sg_block_target_for_task(candidate, task_text, reason)
            dual_state = None
            if dual_context is not None:
                dual_state = self._sg_fail_dual_anchor(
                    "selected_object_has_no_reachable_episode_waypoint"
                )
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "task": task_text,
                    "room_hint": room_hint,
                    "target_object": target_object,
                    "decision": "miss_no_traversable_topk",
                    "query_source": query_source,
                    "top5": self._compact_sg_results(top5),
                    "raw_top5": raw_top5_compact,
                    "rejected": rejected_targets,
                    "exploration": copy.deepcopy(exploration_status),
                    "confidence_gate_pass": True,
                    "traversability_gate_pass": False,
                    "dual_anchor_state": dual_state,
                    "ordinary_vln_executed": False if dual_context is not None else None,
                    **attempt_payload,
                },
            )
            return None

        (
            top1,
            raw_world_xz,
            semantic_object_center_world_xz,
            navigation_reference,
            reachable,
            trajectory_route,
            route_planning_anchors,
        ) = selected
        exploration_completion = None
        if self._sg_exploration_adapter.matches(
            str(getattr(self, "current_episode_id", "")), task_key
        ) and self._sg_exploration_adapter.retry_pending:
            exploration_completion = {
                "scan_memory_update": (
                    self._sg_exploration_adapter.scan_memory_update(
                        world_metric_map, adapter
                    )
                ),
                **self._sg_exploration_adapter.complete_target_retry(),
            }
        grid_x, grid_z = reachable["target_grid"]
        snapped_world_xz = tuple(reachable["target_world_xz"])
        map_x, map_y = self._sg_coords_to_map(
            (snapped_world_xz[0], float(top1["position"][1]), snapped_world_xz[1])
        )
        world_resolution_m = float(world_metric_map.spec.resolution_m)
        dist_cells = float(
            np.linalg.norm(
                np.asarray(snapped_world_xz, dtype=np.float64) - current_world_xz
            )
            / world_resolution_m
        )
        trajectory_guidance = self._sg_compose_trajectory_guided_world_path(
            reachable,
            trajectory_route,
            current_world_xz,
        )
        object_center_residual_m = float(
            np.linalg.norm(
                np.asarray(snapped_world_xz, dtype=np.float64)
                - np.asarray(raw_world_xz, dtype=np.float64)
            )
        )
        semantic_object_center_residual_m = float(
            np.linalg.norm(
                np.asarray(snapped_world_xz, dtype=np.float64)
                - np.asarray(
                    semantic_object_center_world_xz, dtype=np.float64
                )
            )
        )
        navigation_target_source = str(
            navigation_reference["navigation_target_source"]
        )
        path_selection = self._sg_select_execution_world_path(
            reachable,
            trajectory_guidance,
            allow_shortcut=True,
        )
        execution_path = list(path_selection["path"])
        execution_path_source = str(
            path_selection["execution_path_source"]
        )
        path_selection_metadata = copy.deepcopy(path_selection)
        path_selection_metadata.pop("path", None)

        target_info = dict(top1)
        target_info.update(
            {
                "todo_index": todo_idx,
                "todo_text": task_text,
                "query_source": query_source,
                "_sg_full_instruction": full_instruction,
                "_sg_continuation_instruction": continuation_instruction,
                "_sg_instruction_branch_mode": (
                    "full_instruction_after_sg_reach_completed_recall_todo"
                    if instruction_branch
                    else None
                ),
                "_sg_level1_vlm_selected_uid": (
                    single_recall_context.get("selected_uid")
                    if single_recall_context is not None
                    else None
                ),
                "_sg_level1_vlm_rerank": (
                    copy.deepcopy(single_recall_context.get("rerank"))
                    if single_recall_context is not None
                    else None
                ),
                "target_map_x": int(map_x),
                "target_map_y": int(map_y),
                "target_world_xz": [float(snapped_world_xz[0]), float(snapped_world_xz[1])],
                "raw_target_world_xz": [float(raw_world_xz[0]), float(raw_world_xz[1])],
                "semantic_object_center_world_xz": [
                    float(semantic_object_center_world_xz[0]),
                    float(semantic_object_center_world_xz[1]),
                ],
                "navigation_reference_world_xz": [
                    float(raw_world_xz[0]),
                    float(raw_world_xz[1]),
                ],
                "target_world_grid_x": int(grid_x),
                "target_world_grid_z": int(grid_z),
                "raw_target_world_grid_x": int(reachable["raw_target_grid"][0]),
                "raw_target_world_grid_z": int(reachable["raw_target_grid"][1]),
                "world_map_resolution_m": world_resolution_m,
                "navigation_target_source": navigation_target_source,
                "navigation_reference_policy": navigation_reference[
                    "navigation_reference_policy"
                ],
                "reachable_object_center_residual_m": object_center_residual_m,
                "reachable_navigation_reference_residual_m": object_center_residual_m,
                "reachable_semantic_object_center_residual_m": semantic_object_center_residual_m,
                "_sg_best_object_center_residual_m": object_center_residual_m,
                "_sg_best_object_center_target_world_xz": [
                    float(snapped_world_xz[0]),
                    float(snapped_world_xz[1]),
                ],
                "_sg_route_planning_anchors_world_xz": [
                    [float(point[0]), float(point[1])]
                    for point in route_planning_anchors
                ],
                "_sg_world_path_cells": execution_path,
                "_sg_path_progress_cells": 0.0,
                "theta_path_waypoints": len(execution_path),
                "theta_path_length_m": float(
                    self._sg_path_cumulative(execution_path)[-1]
                    * world_resolution_m
                ),
                "execution_path_source": execution_path_source,
                "path_selection": copy.deepcopy(path_selection_metadata),
                "trajectory_guidance": copy.deepcopy(trajectory_guidance),
                "trajectory_corridor": copy.deepcopy(
                    reachable.get("trajectory_corridor")
                ),
                "navmesh_target_validation": copy.deepcopy(
                    reachable.get("navmesh_target_validation")
                ),
                "_sg_refinement_count": 0,
                "_sg_correction_replan_count": 0,
                "_sg_collision_replans_without_progress": 0,
                "_sg_consecutive_failed_recoveries": 0,
                "_sg_dual_anchor_index": (
                    int(dual_context["anchor_index"])
                    if dual_context is not None
                    else None
                ),
                "_sg_dual_anchor_selected_uid": (
                    dual_context["selected_uid"]
                    if dual_context is not None
                    else None
                ),
                "_sg_dual_anchor_rerank": (
                    copy.deepcopy(dual_context["rerank"])
                    if dual_context is not None
                    else None
                ),
                "_sg_navigation_episode_id": top1.get(
                    "_sg_navigation_episode_id"
                ),
            }
        )
        target_info.update(
            {
                key: copy.deepcopy(value)
                for key, value in navigation_reference.items()
                if key.startswith("approach_")
                or key.startswith("sofa_")
                or key.startswith("staircase_plant_")
            }
        )
        if execution_path_source == (
            "certified_trajectory_then_world_theta_star"
        ):
            target_info["_sg_trajectory_suffix_start_progress_cells"] = float(
                trajectory_guidance["trajectory_suffix_start_progress_cells"]
            )
        target_info.update(trajectory_route)
        self._sg_path_plan_active = True
        matched_keyframe_path = self._save_sg_match_keyframe(top1, kwargs.get("step"))
        if matched_keyframe_path:
            target_info["matched_keyframe_path"] = matched_keyframe_path
        self._sg_nav_begin_progress(kwargs.get("step", 0), dist_cells)
        self._write_sg_decision(
            kwargs.get("step"),
            {
                "task": task_text,
                "room_hint": room_hint,
                "target_object": target_object,
                "decision": "hit_sg_nav",
                "query_source": query_source,
                "top5": self._compact_sg_results(top5),
                "raw_top5": raw_top5_compact,
                "selected_uid": top1.get("uid"),
                "selected_label": top1.get("label"),
                "selected_score": top1.get("score"),
                "navigation_target_source": navigation_target_source,
                "navigation_reference_policy": navigation_reference[
                    "navigation_reference_policy"
                ],
                "object_center_world": top1.get("position"),
                "raw_world_xz": raw_world_xz,
                "semantic_object_center_world_xz": semantic_object_center_world_xz,
                "navigation_reference_world_xz": raw_world_xz,
                "navigation_reference": copy.deepcopy(navigation_reference),
                "raw_world_grid": reachable["raw_target_grid"],
                "snapped_world_xz": snapped_world_xz,
                "snapped_world_grid": (grid_x, grid_z),
                "visualization_map": (map_x, map_y),
                "planning_window_origin_world_grid": reachable["window_origin"],
                "planning_window_padding_cells": int(
                    reachable.get("planning_padding_cells", 0)
                ),
                "planning_route_anchor_count": int(
                    reachable.get("planning_route_anchor_count", 0)
                ),
                "planner": "world_metric_theta_star_with_certified_trajectory_corridor",
                "execution_path_source": execution_path_source,
                "path_selection": copy.deepcopy(path_selection_metadata),
                "theta_path_waypoints": len(execution_path),
                "theta_raw_waypoints": len(reachable["raw_path"]),
                "theta_path_length_m": float(
                    self._sg_path_cumulative(execution_path)[-1]
                    * world_resolution_m
                ),
                "theta_expanded_states": int(reachable["expanded_states"]),
                "theta_clearance_cells": float(reachable["clearance_cells"]),
                "trajectory_corridor": copy.deepcopy(
                    reachable.get("trajectory_corridor")
                ),
                "navmesh_target_validation": copy.deepcopy(
                    reachable.get("navmesh_target_validation")
                ),
                "warm_start_connector": (
                    reachable.get("trajectory_corridor") or {}
                ).get("warm_start_connector_status"),
                "dist_m": dist_cells * world_resolution_m,
                "trajectory_route_node_ids": target_info.get("trajectory_route_node_ids"),
                "trajectory_guidance": copy.deepcopy(trajectory_guidance),
                "matched_keyframe_path": matched_keyframe_path,
                "confidence_gate_pass": True,
                "traversability_gate_pass": True,
                "instruction_branch_mode": (
                    "full_instruction_after_sg_reach_completed_recall_todo"
                    if instruction_branch
                    else None
                ),
                "continuation_instruction": (
                    continuation_instruction
                    if instruction_branch
                    else None
                ),
                "exploration_completion": copy.deepcopy(
                    exploration_completion
                ),
                "level1_vlm_selected_uid": (
                    single_recall_context.get("selected_uid")
                    if single_recall_context is not None
                    else None
                ),
                **attempt_payload,
            },
        )
        if dual_context is not None:
            state = self._sg_dual_anchor_state
            if state is None:
                raise RuntimeError("Level-3 dual-anchor state disappeared")
            if state.phase in {"pending_anchor1", "pending_anchor2"}:
                activated = state.next_target()
                activated_uid = str(
                    activated.get("uid") or activated.get("node_id")
                )
                if activated_uid != dual_context["selected_uid"]:
                    raise RuntimeError(
                        "Level-3 state target differs from VLM-selected UID"
                    )
        return {"target_map_x": int(map_x), "target_map_y": int(map_y), "target_info": target_info}

    def _sg_world_metric_map(self):
        adapter = getattr(self, "inherit_sg", None)
        recorder = getattr(adapter, "world_memory", None)
        metric_map = getattr(recorder, "metric_map", None)
        if metric_map is None:
            return None
        if int(metric_map.stats().get("free_cells", 0)) <= 0:
            return None
        return metric_map

    def _sg_trajectory_graph(self):
        adapter = getattr(self, "inherit_sg", None)
        recorder = getattr(adapter, "world_memory", None)
        return getattr(recorder, "trajectory", None)

    def _sg_current_world_xz(self, kwargs=None):
        kwargs = kwargs or {}
        obs = kwargs.get("obs")
        if isinstance(obs, dict) and obs.get("position") is not None:
            position = np.asarray(obs["position"], dtype=np.float64).reshape(-1)
            if position.size >= 3 and np.isfinite(position[[0, 2]]).all():
                return position[[0, 2]].copy()
        position = kwargs.get("current_world_position")
        if position is not None:
            position = np.asarray(position, dtype=np.float64).reshape(-1)
            if position.size >= 3:
                position = position[[0, 2]]
            elif position.size >= 2:
                position = position[:2]
            if position.size == 2 and np.isfinite(position).all():
                return position.copy()
        cached = getattr(self, "_sg_world_position_xz", None)
        if cached is None:
            return None
        cached = np.asarray(cached, dtype=np.float64).reshape(-1)
        if cached.size != 2 or not np.isfinite(cached).all():
            return None
        return cached.copy()

    @staticmethod
    def _sg_window_point(window, global_point):
        return np.asarray(
            [
                float(global_point[0]) - float(window.origin_grid_x),
                float(global_point[1]) - float(window.origin_grid_z),
            ],
            dtype=np.float64,
        )

    def _sg_register_world_collision_block(
        self, failed_endpoint_grid, actual_position_grid=None
    ):
        """Add persistent negative free-space evidence for a rejected forward.

        Habitat validates the swept agent body, while M0 only contains a 2-D
        occupancy estimate.  Treating a rejected action as one bad endpoint
        lets the next replan cut through the same physical boundary one cell
        over.  Record the unexecuted tail of the attempted segment instead.
        The first clearance radius around the actual pose is kept unblocked so
        the recovery planner can still leave its current valid state.
        """

        endpoint = np.asarray(failed_endpoint_grid, dtype=np.float64).reshape(-1)
        if endpoint.size != 2 or not np.isfinite(endpoint).all():
            return None
        cell = tuple(map(int, np.rint(endpoint)))
        blocked_cells = {cell}
        segment_start = None
        if actual_position_grid is not None:
            actual = np.asarray(actual_position_grid, dtype=np.float64).reshape(-1)
            if actual.size == 2 and np.isfinite(actual).all():
                delta = endpoint - actual
                length = float(np.linalg.norm(delta))
                if length > 1e-6:
                    clearance_cells = float(
                        os.environ.get(
                            "LIFELONG_SG_DISCRETE_CLEARANCE_CELLS", "2.5"
                        )
                    )
                    protected_tail_start = min(
                        length, max(0.0, clearance_cells) + 0.75
                    )
                    sample_count = max(1, int(np.ceil(length * 2.0)))
                    for index in range(sample_count + 1):
                        distance = length * index / sample_count
                        if distance + 1e-6 < protected_tail_start:
                            continue
                        point = actual + delta * (distance / length)
                        blocked_cells.add(tuple(map(int, np.rint(point))))
                    segment_start = tuple(map(float, actual))
                    self._sg_world_collision_segments.append(
                        {
                            "actual_grid": segment_start,
                            "expected_grid": tuple(map(float, endpoint)),
                            "evidence_cells": len(blocked_cells),
                        }
                    )
        self._sg_world_collision_blocked_cells.update(blocked_cells)
        logger.info(
            "[INHerit-SG-V7] registered runtime collision segment at world "
            f"actual={segment_start}, endpoint={cell}, "
            f"new_evidence_cells={len(blocked_cells)}, "
            f"blocked_cells={len(self._sg_world_collision_blocked_cells)}"
        )
        return cell

    def _sg_apply_world_collision_blocks(
        self, safe_mask, window, clearance_cells
    ):
        blocked = getattr(self, "_sg_world_collision_blocked_cells", set())
        if not blocked:
            return 0
        radius = int(np.ceil(max(0.0, float(clearance_cells))))
        applied = 0
        for global_x, global_z in blocked:
            local_x, local_z = window.global_to_local(global_x, global_z)
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dz * dz > float(clearance_cells) ** 2:
                        continue
                    x = local_x + dx
                    z = local_z + dz
                    if 0 <= z < safe_mask.shape[0] and 0 <= x < safe_mask.shape[1]:
                        if bool(safe_mask[z, x]):
                            applied += 1
                        safe_mask[z, x] = False
        return int(applied)

    def _sg_plan_world_path(
        self,
        raw_target_world_xz,
        current_world_xz,
        exact=False,
        planning_anchor_world_xz=None,
        project_target_to_navmesh=True,
        navmesh_terminal_snap_tolerance_m=0.0,
    ):
        """Keep the ordinary plan first; widen its observed-map window on failure."""
        from membridge.navigation.map_connection import (
            GEOMETRIC_FAILURES, expanded_trajectory_anchors,
        )
        kwargs = dict(exact=exact, planning_anchor_world_xz=planning_anchor_world_xz,
                      project_target_to_navmesh=project_target_to_navmesh,
                      navmesh_terminal_snap_tolerance_m=navmesh_terminal_snap_tolerance_m)
        self._sg_last_world_window_retry = None
        result = self._sg_plan_world_path_once(raw_target_world_xz, current_world_xz, **kwargs)
        if result is not None or not self._env_bool('LIFELONG_SG_WORLD_FULL_WINDOW_RETRY', True):
            return result
        failure = copy.deepcopy(getattr(self, '_sg_last_theta_failure', None))
        if (failure or {}).get('reason') not in GEOMETRIC_FAILURES:
            return None
        anchors = expanded_trajectory_anchors(
            self, planning_anchor_world_xz, current_world_xz, raw_target_world_xz)
        if anchors is None:
            return None
        # Exactly one retry, through identical safety, terminal, and memory
        # planning code. The existing maximum-window-cell cap still applies.
        kwargs['planning_anchor_world_xz'] = anchors
        result = self._sg_plan_world_path_once(raw_target_world_xz, current_world_xz, **kwargs)
        audit = dict(status='connected' if result is not None else 'failed',
                     anchor_count=len(anchors), initial_failure=failure,
                     policy='ordinary_failure_then_observed_trajectory_window')
        self._sg_last_world_window_retry = audit
        if result is not None:
            result['world_window_retry'] = copy.deepcopy(audit)
        else:
            self._sg_last_theta_failure = dict(
                self._sg_last_theta_failure or {}, world_window_retry=audit)
        return result

    def _sg_plan_world_path_once(
        self,
        raw_target_world_xz,
        current_world_xz,
        exact=False,
        planning_anchor_world_xz=None,
        project_target_to_navmesh=True,
        navmesh_terminal_snap_tolerance_m=0.0,
    ):
        """Plan only through acquisition-observed M0 free space.

        The dense array exists only for this query.  Its origin is converted
        back to global grid coordinates before the path leaves this method, so
        no episode-relative raster coordinate becomes persistent state.
        """

        self._sg_last_theta_failure = None
        metric_map = self._sg_world_metric_map()
        if metric_map is None:
            self._sg_last_theta_failure = {"reason": "missing_world_metric_map"}
            return None
        current = np.asarray(current_world_xz, dtype=np.float64).reshape(-1)
        raw_target = np.asarray(raw_target_world_xz, dtype=np.float64).reshape(-1)
        if (
            current.size != 2
            or raw_target.size != 2
            or not np.isfinite(current).all()
            or not np.isfinite(raw_target).all()
        ):
            self._sg_last_theta_failure = {"reason": "invalid_world_plan_endpoints"}
            return None

        spec = metric_map.spec
        object_center_target = raw_target.copy()
        if bool(project_target_to_navmesh):
            raw_target, navmesh_target_projection = (
                self._sg_project_target_to_current_navmesh_island(
                    object_center_target,
                    current,
                )
            )
        else:
            raw_target = object_center_target.copy()
            navmesh_target_projection = {
                "status": "skipped_for_preprojected_approach_or_certified_target",
                "raw_object_center_world_xz": list(
                    map(float, object_center_target)
                ),
                "projected_world_xz": list(map(float, raw_target)),
                "projection_distance_m": 0.0,
            }
        start_global = spec.world_to_cell(float(current[0]), float(current[1]))
        object_center_target_global = spec.world_to_cell(
            float(object_center_target[0]),
            float(object_center_target[1]),
        )
        raw_target_global = spec.world_to_cell(
            float(raw_target[0]), float(raw_target[1])
        )
        start_state = metric_map.cell_state(*start_global, floor_id=0)

        padding = max(
            int(self.sg_snap_radius_cells),
            int(os.environ.get("LIFELONG_SG_WORLD_WINDOW_PADDING_CELLS", "40")),
        )
        planning_anchors = []
        if planning_anchor_world_xz is not None:
            try:
                anchor_array = np.asarray(
                    planning_anchor_world_xz, dtype=np.float64
                ).reshape(-1, 2)
            except (TypeError, ValueError):
                anchor_array = np.empty((0, 2), dtype=np.float64)
            planning_anchors = [
                tuple(map(float, point))
                for point in anchor_array
                if np.isfinite(point).all()
            ]
        if planning_anchors:
            anchor_cells = [
                spec.world_to_cell(point[0], point[1])
                for point in planning_anchors
            ]
            endpoint_min_x = min(start_global[0], raw_target_global[0])
            endpoint_max_x = max(start_global[0], raw_target_global[0])
            endpoint_min_z = min(start_global[1], raw_target_global[1])
            endpoint_max_z = max(start_global[1], raw_target_global[1])
            route_overhang = max(
                0,
                endpoint_min_x - min(point[0] for point in anchor_cells),
                max(point[0] for point in anchor_cells) - endpoint_max_x,
                endpoint_min_z - min(point[1] for point in anchor_cells),
                max(point[1] for point in anchor_cells) - endpoint_max_z,
            )
            clearance_margin = int(
                np.ceil(
                    float(
                        os.environ.get(
                            "LIFELONG_SG_DISCRETE_CLEARANCE_CELLS", "2.5"
                        )
                    )
                )
            )
            route_margin = int(
                os.environ.get("LIFELONG_SG_WORLD_ROUTE_MARGIN_CELLS", "2")
            )
            padding = max(
                padding,
                int(route_overhang) + clearance_margin + route_margin,
            )
        max_window_cells = int(
            os.environ.get("LIFELONG_SG_WORLD_MAX_WINDOW_CELLS", "4000000")
        )
        try:
            window = metric_map.planning_window(
                start_global,
                raw_target_global,
                floor_id=0,
                padding_cells=padding,
                max_window_cells=max_window_cells,
            )
        except Exception as exc:
            self._sg_last_theta_failure = {
                "reason": "world_planning_window_failed",
                "detail": str(exc),
                "start_world_grid": tuple(map(int, start_global)),
                "raw_target_world_grid": tuple(map(int, raw_target_global)),
            }
            return None

        clearance_cells = float(
            os.environ.get("LIFELONG_SG_DISCRETE_CLEARANCE_CELLS", "2.5")
        )
        trajectory_corridor = None
        trajectory_centerline = None
        trajectory_corridor_metadata = {
            "trajectory_corridor_status": "disabled",
            "warm_start_connector_status": "disabled",
        }
        if bool(getattr(self, "sg_trajectory_corridor_enable", True)):
            graph = self._sg_trajectory_graph()
            if graph is None:
                trajectory_corridor_metadata = {
                    "trajectory_corridor_status": "missing_trajectory_graph",
                    "warm_start_connector_status": "missing_trajectory_graph",
                }
            else:
                try:
                    (
                        trajectory_corridor,
                        trajectory_centerline,
                        corridor_stats,
                    ) = graph.certified_masks(
                        window,
                        spec,
                        clearance_cells=clearance_cells,
                        floor_id=0,
                        warm_start_world_xz=current,
                        max_terminal_connector_m=float(
                            getattr(
                                self,
                                "sg_warm_start_trajectory_connect_m",
                                0.35,
                            )
                        ),
                        connector_validator=(
                            self._sg_validate_warm_start_connector
                        ),
                    )
                    trajectory_corridor_metadata = {
                        "trajectory_corridor_status": "applied",
                        **corridor_stats,
                    }
                except Exception as exc:
                    logger.warning(
                        "[INHerit-SG-V7] failed to materialize certified "
                        f"trajectory corridor: {exc}"
                    )
                    trajectory_corridor = None
                    trajectory_corridor_metadata = {
                        "trajectory_corridor_status": "failed",
                        "trajectory_corridor_error": str(exc),
                        "warm_start_connector_status": "failed",
                    }

        base_safe_mask = window.safe_mask(
            clearance_cells,
            certified_free=trajectory_corridor,
        )
        trajectory_execution_slack_cells = 0.0
        if trajectory_corridor is not None:
            trajectory_execution_slack_cells = (
                self._sg_trajectory_execution_slack_cells(spec)
            )
        agent_radius_m = float(
            getattr(
                getattr(
                    self.config.TASK_CONFIG.SIMULATOR, "AGENT_0", None
                ),
                "RADIUS",
                0.10,
            )
        )
        trajectory_slack_obstacle_clearance_cells = (
            agent_radius_m / float(spec.resolution_m)
        )
        safe_mask = window.safe_mask(
            clearance_cells,
            certified_free=trajectory_corridor,
            certified_centers=trajectory_centerline,
            certified_center_slack_cells=trajectory_execution_slack_cells,
            certified_slack_obstacle_clearance_cells=(
                trajectory_slack_obstacle_clearance_cells
            ),
        )
        trajectory_execution_slack_added_cells = int(
            np.count_nonzero(np.logical_and(safe_mask, ~base_safe_mask))
        )
        trajectory_corridor_metadata.update(
            {
                "trajectory_execution_slack_cells": float(
                    trajectory_execution_slack_cells
                ),
                "trajectory_execution_slack_added_cells": int(
                    trajectory_execution_slack_added_cells
                ),
                "trajectory_slack_obstacle_clearance_cells": float(
                    trajectory_slack_obstacle_clearance_cells
                ),
            }
        )
        collision_overlay_cells = self._sg_apply_world_collision_blocks(
            safe_mask, window, clearance_cells
        )
        # Keep a second planning layer containing only ordinary M0 free space.
        # It deliberately excludes the full acquisition corridor: the only
        # certified cells that may be added later are the bounded warm-start
        # connector needed to enter this ordinary clearance-safe component.
        ordinary_safe_mask = window.safe_mask(clearance_cells)
        ordinary_collision_overlay_cells = self._sg_apply_world_collision_blocks(
            ordinary_safe_mask, window, clearance_cells
        )
        trajectory_probe_mask = None
        trajectory_probe_tube_cells = 0.0
        trajectory_probe_collision_overlay_cells = 0
        if trajectory_centerline is not None:
            trajectory_probe_tube_cells = float(
                os.environ.get(
                    "LIFELONG_SG_TRAJECTORY_PROBE_TUBE_CELLS", "1.0"
                )
            )
            distance_to_centerline = cv2.distanceTransform(
                np.logical_not(
                    np.asarray(trajectory_centerline, dtype=bool)
                ).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            trajectory_probe_mask = (
                distance_to_centerline
                <= max(0.0, trajectory_probe_tube_cells) + 1e-6
            )
            # The trajectory certificate may override stale probabilistic
            # occupancy, but never collision evidence observed in this warm
            # run.  Use the same footprint dilation as the normal safe mask.
            trajectory_probe_collision_overlay_cells = (
                self._sg_apply_world_collision_blocks(
                    trajectory_probe_mask,
                    window,
                    clearance_cells,
                )
            )
        trajectory_corridor_metadata.update(
            {
                "trajectory_probe_tube_cells": float(
                    trajectory_probe_tube_cells
                ),
                "trajectory_probe_cells": int(
                    0
                    if trajectory_probe_mask is None
                    else np.count_nonzero(trajectory_probe_mask)
                ),
                "trajectory_probe_collision_overlay_cells": int(
                    trajectory_probe_collision_overlay_cells
                ),
            }
        )
        start = window.global_to_local(*start_global)
        raw_target_local = window.global_to_local(*raw_target_global)
        certified_start = bool(
            trajectory_corridor is not None
            and trajectory_corridor[start[1], start[0]]
        )
        if start_state != "free" and not certified_start:
            self._sg_last_theta_failure = {
                "reason": "warm_start_not_free_in_world_memory",
                "start_world_xz": tuple(map(float, current)),
                "start_world_grid": tuple(map(int, start_global)),
                "start_world_state": start_state,
                "warm_start_connector": trajectory_corridor_metadata.get(
                    "warm_start_connector_status"
                ),
                **trajectory_corridor_metadata,
            }
            return None
        if not bool(safe_mask[start[1], start[0]]):
            self._sg_last_theta_failure = {
                "reason": "warm_start_not_clearance_safe_in_world_memory",
                "start_world_xz": tuple(map(float, current)),
                "start_world_grid": tuple(map(int, start_global)),
                "clearance_cells": float(clearance_cells),
                "warm_start_connector": trajectory_corridor_metadata.get(
                    "warm_start_connector_status"
                ),
                **trajectory_corridor_metadata,
            }
            return None

        target_validator, navmesh_target_validation = (
            self._sg_world_navmesh_target_validator(
                window,
                current,
                terminal_snap_tolerance_m=(
                    navmesh_terminal_snap_tolerance_m
                ),
            )
        )
        target, reachable_cells, raw_distance = self._sg_reachable_target(
            safe_mask,
            start,
            raw_target_local,
            exact,
            target_validator=target_validator,
        )
        if target is None:
            self._sg_last_theta_failure = {
                "reason": "no_world_reachable_target_within_snap_radius",
                "start_world_grid": tuple(map(int, start_global)),
                "raw_target_world_grid": tuple(
                    map(int, object_center_target_global)
                ),
                "navmesh_projected_target_world_grid": tuple(
                    map(int, raw_target_global)
                ),
                "reachable_cells": int(reachable_cells),
                "raw_distance_cells": float(raw_distance),
                "snap_radius_cells": int(self.sg_snap_radius_cells),
                "navmesh_target_projection": copy.deepcopy(
                    navmesh_target_projection
                ),
                "navmesh_target_validation": copy.deepcopy(
                    navmesh_target_validation
                ),
                **trajectory_corridor_metadata,
            }
            return None

        target_global_for_validation = tuple(
            map(int, window.local_to_global(*target))
        )
        navmesh_target_validation["selected_world_grid"] = list(
            target_global_for_validation
        )
        navmesh_target_validation["selected_world_xz"] = list(
            map(float, spec.cell_center_world(*target_global_for_validation))
        )
        navmesh_target_validation["projection"] = copy.deepcopy(
            navmesh_target_projection
        )

        raw_path_local, expanded = self._sg_theta_star(safe_mask, start, target)
        if not raw_path_local:
            self._sg_last_theta_failure = {
                "reason": "world_theta_star_failed",
                "start_world_grid": tuple(map(int, start_global)),
                "target_world_grid": window.local_to_global(*target),
                "expanded_states": int(expanded),
            }
            return None
        path_local = self._sg_string_pull_path(safe_mask, raw_path_local)
        path_global = [
            tuple(map(float, window.local_to_global(int(x), int(z))))
            for x, z in path_local
        ]
        raw_path_global = [
            tuple(map(int, window.local_to_global(x, z)))
            for x, z in raw_path_local
        ]
        target_global = tuple(map(int, window.local_to_global(*target)))
        cumulative = self._sg_path_cumulative(path_global)

        shortcut_candidates = []
        direct_connector_status = "not_attempted"
        direct_connector_path = []
        direct_safe_mask = None
        connector_radius_cells = max(
            0,
            int(
                np.ceil(
                    float(
                        getattr(
                            self,
                            "sg_warm_start_trajectory_connect_m",
                            0.35,
                        )
                    )
                    / float(spec.resolution_m)
                    - 1e-9
                )
            ),
        )
        direct_safe_mask, direct_connector_path = (
            self._sg_connect_start_to_safe_mask(
                ordinary_safe_mask,
                safe_mask,
                start,
                connector_radius_cells,
            )
        )
        if direct_safe_mask is None:
            direct_connector_status = "no_bounded_connector_to_observed_free"
        else:
            direct_connector_status = (
                "start_already_observed_free"
                if len(direct_connector_path) <= 1
                else "bounded_terminal_connector_applied"
            )
            if bool(direct_safe_mask[target[1], target[0]]):
                direct_raw_path, direct_expanded = self._sg_theta_star(
                    direct_safe_mask, start, target
                )
                if direct_raw_path:
                    direct_local_path = self._sg_string_pull_path(
                        direct_safe_mask, direct_raw_path
                    )
                    direct_global_path = [
                        tuple(
                            map(
                                float,
                                window.local_to_global(int(x), int(z)),
                            )
                        )
                        for x, z in direct_local_path
                    ]
                    shortcut_candidates.append(
                        {
                            "source": "observed_free_theta_star",
                            "path": direct_global_path,
                            "path_length_cells": float(
                                self._sg_path_cumulative(direct_global_path)[-1]
                            ),
                            "expanded_states": int(direct_expanded),
                            "terminal_connector_status": direct_connector_status,
                            "terminal_connector_length_cells": float(
                                self._sg_path_cumulative(direct_connector_path)[-1]
                                if direct_connector_path
                                else 0.0
                            ),
                        }
                    )

        # The union Theta* result may take a shorter hybrid through ordinary
        # free space and a small portion of the certified centreline.  Accept
        # it as a shortcut only when every continuous segment is covered by
        # ordinary safe cells/the bounded connector or by the narrow
        # trajectory probe tube.  This rejects shallow raster cuts across the
        # wider swept corridor that the discrete action lattice cannot replay.
        shortcut_support_mask = (
            ordinary_safe_mask.copy()
            if direct_safe_mask is None
            else direct_safe_mask.copy()
        )
        if trajectory_probe_mask is not None:
            shortcut_support_mask |= np.asarray(
                trajectory_probe_mask, dtype=bool
            )
        hybrid_supported = all(
            self._sg_float_segment_is_safe(
                shortcut_support_mask,
                self._sg_window_point(window, left),
                self._sg_window_point(window, right),
            )
            for left, right in zip(path_global[:-1], path_global[1:])
        )
        if hybrid_supported:
            shortcut_candidates.append(
                {
                    "source": "certified_hybrid_theta_star",
                    "path": list(path_global),
                    "path_length_cells": float(cumulative[-1]),
                    "expanded_states": int(expanded),
                    "terminal_connector_status": direct_connector_status,
                }
            )

        self._sg_world_plan_window = window
        self._sg_world_safe_mask = safe_mask
        self._sg_world_trajectory_probe_mask = trajectory_probe_mask
        selected_target_world_xz = np.asarray(
            spec.cell_center_world(*target_global), dtype=np.float64
        )
        object_center_distance_cells = float(
            np.linalg.norm(
                selected_target_world_xz - object_center_target
            )
            / float(spec.resolution_m)
        )
        return {
            "target_grid": target_global,
            "target_world_xz": tuple(map(float, selected_target_world_xz)),
            "raw_target_grid": tuple(
                map(int, object_center_target_global)
            ),
            "navmesh_projected_target_grid": tuple(
                map(int, raw_target_global)
            ),
            "navmesh_projected_target_world_xz": tuple(
                map(float, raw_target)
            ),
            "path": path_global,
            "raw_path": raw_path_global,
            "path_length_cells": float(cumulative[-1]),
            "expanded_states": int(expanded),
            "reachable_cells": int(reachable_cells),
            "clearance_cells": float(clearance_cells),
            "runtime_collision_overlay_cells": int(collision_overlay_cells),
            "ordinary_collision_overlay_cells": int(
                ordinary_collision_overlay_cells
            ),
            "raw_distance_cells": float(object_center_distance_cells),
            "trajectory_corridor": trajectory_corridor_metadata,
            "window_origin": (
                int(window.origin_grid_x),
                int(window.origin_grid_z),
            ),
            "window_shape": tuple(map(int, safe_mask.shape)),
            "planning_padding_cells": int(padding),
            "planning_route_anchor_count": len(planning_anchors),
            "shortcut_candidates": shortcut_candidates,
            "shortcut_candidate_count": len(shortcut_candidates),
            "direct_terminal_connector_status": direct_connector_status,
            "direct_terminal_connector_path": [
                [int(x), int(z)] for x, z in direct_connector_path
            ],
            "navmesh_target_validation": copy.deepcopy(
                navmesh_target_validation
            ),
            "safe_mask": safe_mask,
            "window": window,
            "_exploration_trajectory_probe_mask": trajectory_probe_mask,
        }

    def _sg_trajectory_route_world_xz(self, route_metadata):
        if str(route_metadata.get("trajectory_route_status", "")) != "connected":
            return []
        graph = self._sg_trajectory_graph()
        if graph is None:
            return []
        anchors = []

        def append_anchor(point):
            if anchors and np.linalg.norm(
                np.asarray(point, dtype=np.float64)
                - np.asarray(anchors[-1], dtype=np.float64)
            ) <= 1e-6:
                return
            anchors.append(point)

        connector = route_metadata.get("trajectory_connector_world_xz")
        if connector is not None:
            try:
                connector = tuple(
                    map(float, np.asarray(connector).reshape(-1)[:2])
                )
            except (TypeError, ValueError):
                return []
            if len(connector) != 2 or not np.isfinite(connector).all():
                return []
            append_anchor(connector)
        for node_id in route_metadata.get("trajectory_route_node_ids") or []:
            try:
                node = graph.nodes[int(node_id)]
                point = (
                    float(node.position_xyz[0]),
                    float(node.position_xyz[2]),
                )
            except (IndexError, TypeError, ValueError):
                return []
            if not np.isfinite(point).all():
                return []
            append_anchor(point)
        return anchors

    def _sg_trajectory_route_metadata(self, candidate, current_world_xz):
        recorder = getattr(getattr(self, "inherit_sg", None), "world_memory", None)
        graph = getattr(recorder, "trajectory", None)
        if graph is None:
            return {"trajectory_route_status": "missing_trajectory_graph"}
        current = np.asarray(
            [float(current_world_xz[0]), 0.0, float(current_world_xz[1])],
            dtype=np.float64,
        )
        floor_nodes = [node for node in graph.nodes if int(node.floor_id) == 0]
        connector_limit = float(
            getattr(self, "sg_warm_start_trajectory_connect_m", 0.35)
        )
        waypoint = candidate.get("waypoint") or {}
        target_node = waypoint.get("trajectory_node_id")
        if target_node is None:
            target_node = (candidate.get("best_observation") or {}).get(
                "trajectory_node_id"
            )
        if not floor_nodes or target_node is None:
            return {
                "trajectory_route_status": "missing_route_endpoint",
                "trajectory_start_node_id": None,
                "trajectory_target_node_id": target_node,
            }

        current_xz = current[[0, 2]]
        attachments = []
        for node in floor_nodes:
            node_xz = np.asarray(node.position_xyz, dtype=np.float64)[[0, 2]]
            attachments.append(
                {
                    "distance": float(np.linalg.norm(current_xz - node_xz)),
                    "kind": "node",
                    "connector": node_xz,
                    "start_nodes": [int(node.node_id)],
                }
            )
        floor_ids = {int(node.node_id) for node in floor_nodes}
        for left, right, _ in graph.edges:
            if int(left) not in floor_ids or int(right) not in floor_ids:
                continue
            left_xz = np.asarray(
                graph.nodes[int(left)].position_xyz, dtype=np.float64
            )[[0, 2]]
            right_xz = np.asarray(
                graph.nodes[int(right)].position_xyz, dtype=np.float64
            )[[0, 2]]
            delta = right_xz - left_xz
            denominator = float(np.dot(delta, delta))
            if denominator <= 1e-12:
                continue
            fraction = float(
                np.clip(
                    np.dot(current_xz - left_xz, delta) / denominator,
                    0.0,
                    1.0,
                )
            )
            projected = left_xz + fraction * delta
            attachments.append(
                {
                    "distance": float(
                        np.linalg.norm(current_xz - projected)
                    ),
                    "kind": "edge",
                    "connector": projected,
                    "start_nodes": [int(left), int(right)],
                }
            )
        attachment = min(
            attachments,
            key=lambda item: (
                item["distance"],
                0 if item["kind"] == "node" else 1,
                item["start_nodes"],
            ),
        )
        start_distance = float(attachment["distance"])
        if start_distance > connector_limit:
            return {
                "trajectory_route_status": (
                    "nearest_certified_trajectory_too_far"
                ),
                "trajectory_start_node_id": None,
                "trajectory_start_distance_m": start_distance,
                "trajectory_target_node_id": int(target_node),
                "trajectory_connector_limit_m": connector_limit,
                "trajectory_connector_kind": attachment["kind"],
            }
        connector_valid, connector_audit = (
            self._sg_validate_warm_start_connector(
                current_xz, attachment["connector"]
            )
        )
        if not connector_valid:
            return {
                "trajectory_route_status": "local_connector_validation_failed",
                "trajectory_start_node_id": None,
                "trajectory_start_distance_m": start_distance,
                "trajectory_target_node_id": int(target_node),
                "trajectory_connector_kind": attachment["kind"],
                "trajectory_connector_validation": connector_audit,
            }
        route_options = []
        route_error = None
        for possible_start in attachment["start_nodes"]:
            try:
                possible_route = graph.shortest_path(
                    possible_start, int(target_node)
                )
            except (IndexError, TypeError, ValueError) as exc:
                route_error = str(exc)
                continue
            if possible_route is None:
                continue
            connector_to_node = float(
                np.linalg.norm(
                    np.asarray(
                        graph.nodes[possible_start].position_xyz,
                        dtype=np.float64,
                    )[[0, 2]]
                    - attachment["connector"]
                )
            )
            route_length = sum(
                float(
                    np.linalg.norm(
                        np.asarray(graph.nodes[right].position_xyz)[[0, 2]]
                        - np.asarray(graph.nodes[left].position_xyz)[[0, 2]]
                    )
                )
                for left, right in zip(
                    possible_route[:-1], possible_route[1:]
                )
            )
            route_options.append(
                (
                    connector_to_node + route_length,
                    int(possible_start),
                    possible_route,
                )
            )
        if not route_options:
            return {
                "trajectory_route_status": (
                    "invalid_route_endpoint" if route_error else "disconnected"
                ),
                "trajectory_route_error": route_error,
                "trajectory_start_node_id": None,
                "trajectory_target_node_id": int(target_node),
            }
        _, start_node, route = min(
            route_options, key=lambda item: (item[0], item[1])
        )
        return {
            "trajectory_route_status": "connected",
            "trajectory_start_node_id": start_node,
            "trajectory_start_distance_m": start_distance,
            "trajectory_target_node_id": int(target_node),
            "trajectory_route_node_ids": list(map(int, route)),
            "trajectory_connector_kind": attachment["kind"],
            "trajectory_connector_world_xz": list(
                map(float, attachment["connector"])
            ),
            "trajectory_connector_validation": connector_audit,
        }

    def _sg_compose_trajectory_guided_world_path(
        self, reachable, route_metadata, current_world_xz
    ):
        """Follow persisted continuous acquisition edges before the map suffix.

        The certified corridor may be only one safe grid cell wide after the
        normal clearance erosion.  A free-space Theta* shortcut can enter that
        corridor at a shallow offset that is graph-connected but impossible for
        Habitat's 0.25 m / 30 degree action lattice to execute.  The acquisition
        graph contains the continuous path that was actually executed, so warm
        navigation uses that path in reverse up to the target's observation
        node, then Theta* only for the final object-center suffix.
        """

        status = str(route_metadata.get("trajectory_route_status", "missing"))
        if status != "connected":
            return {"trajectory_guidance_status": status}
        graph = self._sg_trajectory_graph()
        if graph is None:
            return {"trajectory_guidance_status": "missing_trajectory_graph"}
        route_ids = list(route_metadata.get("trajectory_route_node_ids") or [])
        if not route_ids:
            return {"trajectory_guidance_status": "empty_trajectory_route"}
        window = reachable.get("window")
        safe_mask = reachable.get("safe_mask")
        metric_map = self._sg_world_metric_map()
        if window is None or safe_mask is None or metric_map is None:
            return {"trajectory_guidance_status": "missing_world_plan"}

        resolution_m = float(metric_map.spec.resolution_m)
        current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
        path = [tuple((current / resolution_m - 0.5).tolist())]
        connector = route_metadata.get("trajectory_connector_world_xz")
        if connector is not None:
            connector = np.asarray(connector, dtype=np.float64).reshape(-1)
            if connector.size != 2 or not np.isfinite(connector).all():
                return {
                    "trajectory_guidance_status": (
                        "invalid_trajectory_connector"
                    )
                }
            connector_point = tuple(
                (connector / resolution_m - 0.5).tolist()
            )
            if (
                np.linalg.norm(
                    np.asarray(connector_point) - np.asarray(path[-1])
                )
                > 1e-6
            ):
                path.append(connector_point)
        try:
            route_nodes = [graph.nodes[int(node_id)] for node_id in route_ids]
        except (IndexError, TypeError, ValueError):
            return {"trajectory_guidance_status": "invalid_trajectory_route"}
        for node in route_nodes:
            point = (
                float(node.position_xyz[0]) / resolution_m - 0.5,
                float(node.position_xyz[2]) / resolution_m - 0.5,
            )
            if np.linalg.norm(np.asarray(point) - np.asarray(path[-1])) > 1e-6:
                path.append(point)

        # Keep the phase boundary in the same cumulative world-grid coordinate
        # system used by the controller.  The stall guard can then reset when a
        # valid acquisition-route detour hands off to the object-center suffix.
        trajectory_suffix_start_progress_cells = float(
            self._sg_path_cumulative(path)[-1]
        )

        route_end = route_nodes[-1]
        route_end_global = metric_map.spec.world_to_cell(
            float(route_end.position_xyz[0]),
            float(route_end.position_xyz[2]),
        )
        route_end_local = window.global_to_local(*route_end_global)
        target_global = tuple(map(int, reachable["target_grid"]))
        target_local = window.global_to_local(*target_global)
        raw_suffix, expanded = self._sg_theta_star(
            safe_mask, route_end_local, target_local
        )
        if not raw_suffix:
            return {
                "trajectory_guidance_status": "trajectory_suffix_theta_failed",
                "trajectory_suffix_expanded_states": int(expanded),
            }
        suffix = self._sg_string_pull_path(safe_mask, raw_suffix)
        for local_x, local_z in suffix:
            point = tuple(
                map(float, window.local_to_global(int(local_x), int(local_z)))
            )
            if np.linalg.norm(np.asarray(point) - np.asarray(path[-1])) > 1e-6:
                path.append(point)

        for segment_index, (left, right) in enumerate(zip(path[:-1], path[1:])):
            if not self._sg_float_segment_is_safe(
                safe_mask,
                self._sg_window_point(window, left),
                self._sg_window_point(window, right),
            ):
                return {
                    "trajectory_guidance_status": "trajectory_segment_not_safe",
                    "trajectory_route_waypoints": len(route_nodes),
                    "failed_segment_index": int(segment_index),
                    "failed_segment_start": list(map(float, left)),
                    "failed_segment_end": list(map(float, right)),
                }
        return {
            "trajectory_guidance_status": "applied",
            "trajectory_route_waypoints": len(route_nodes),
            "trajectory_suffix_waypoints": len(suffix),
            "trajectory_suffix_start_progress_cells": (
                trajectory_suffix_start_progress_cells
            ),
            "trajectory_suffix_expanded_states": int(expanded),
            "path_length_cells": float(self._sg_path_cumulative(path)[-1]),
            "path": path,
        }

    def _sg_select_execution_world_path(
        self,
        reachable,
        trajectory_guidance,
        *,
        allow_shortcut=True,
    ):
        """Select the shortest certified path without sacrificing fallback.

        The full acquisition route is retained as the reliable recovery
        backbone.  A shortcut is eligible only if ``_sg_plan_world_path``
        proved that every segment lies in ordinary observed-free space (plus
        the bounded terminal connector) or the narrow certified probe tube.
        Small savings are ignored so a one-cell raster difference cannot buy
        unnecessary execution risk.
        """

        baseline_path = list(reachable.get("path") or [])
        baseline_length = float(
            reachable.get(
                "path_length_cells",
                self._sg_path_cumulative(baseline_path)[-1]
                if baseline_path
                else float("inf"),
            )
        )
        selected = {
            "path": baseline_path,
            "path_length_cells": baseline_length,
            "execution_path_source": "world_theta_star",
            "selection_reason": "world_theta_star_fallback",
            "shortcut_savings_m": 0.0,
        }

        guided_applied = bool(
            trajectory_guidance.get("trajectory_guidance_status") == "applied"
            and len(trajectory_guidance.get("path") or []) >= 2
        )
        if guided_applied:
            guided_path = list(trajectory_guidance["path"])
            guided_length = float(
                trajectory_guidance.get(
                    "path_length_cells",
                    self._sg_path_cumulative(guided_path)[-1],
                )
            )
            selected = {
                "path": guided_path,
                "path_length_cells": guided_length,
                "execution_path_source": (
                    "certified_trajectory_then_world_theta_star"
                ),
                "selection_reason": "certified_trajectory_fallback",
                "shortcut_savings_m": 0.0,
            }
        else:
            guided_length = float("inf")

        enabled = bool(
            getattr(self, "sg_shortest_safe_path_enable", True)
        )
        candidates = []
        if enabled and bool(allow_shortcut):
            for candidate in reachable.get("shortcut_candidates") or []:
                path = list(candidate.get("path") or [])
                length = float(candidate.get("path_length_cells", float("inf")))
                if len(path) < 2 or not np.isfinite(length):
                    continue
                candidates.append((length, candidate))
        candidate_summaries = [
            {
                "source": str(candidate.get("source")),
                "path_length_cells": float(length),
            }
            for length, candidate in sorted(
                candidates,
                key=lambda item: (
                    item[0],
                    str(item[1].get("source"))
                    != "observed_free_theta_star",
                ),
            )
        ]
        if not candidates:
            selected["shortcut_candidates"] = candidate_summaries
            selected["shortcut_allowed"] = bool(allow_shortcut)
            return selected

        shortcut_length, shortcut = min(
            candidates,
            key=lambda item: (
                item[0],
                str(item[1].get("source"))
                != "observed_free_theta_star",
            ),
        )
        metric_map = self._sg_world_metric_map()
        resolution_m = float(
            metric_map.spec.resolution_m if metric_map is not None else 0.05
        )
        comparison_length = (
            guided_length if guided_applied else baseline_length
        )
        savings_m = float(
            max(0.0, comparison_length - shortcut_length) * resolution_m
        )
        minimum_savings_m = max(
            0.0,
            float(getattr(self, "sg_shortcut_min_savings_m", 0.25)),
        )
        if (
            not np.isfinite(comparison_length)
            or not guided_applied
            or savings_m + 1e-9 >= minimum_savings_m
        ):
            selected = {
                "path": list(shortcut["path"]),
                "path_length_cells": float(shortcut_length),
                "execution_path_source": str(shortcut["source"]),
                "selection_reason": "shortest_safe_path",
                "shortcut_savings_m": float(savings_m),
                "terminal_connector_status": shortcut.get(
                    "terminal_connector_status"
                ),
            }
        selected["shortcut_candidates"] = candidate_summaries
        selected["shortcut_allowed"] = bool(allow_shortcut)
        selected["shortcut_minimum_savings_m"] = float(minimum_savings_m)
        return selected

    def _sg_remaining_trajectory_route_metadata(
        self, target_info, current_world_xz
    ):
        """Rejoin only the unconsumed suffix of the selected acquisition route."""

        graph = self._sg_trajectory_graph()
        route_ids = list(target_info.get("trajectory_route_node_ids") or [])
        if graph is None or not route_ids:
            return {"trajectory_route_status": "missing_active_trajectory_route"}
        current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
        candidates = []
        for route_index, node_id in enumerate(route_ids):
            try:
                node = graph.nodes[int(node_id)]
                node_xz = np.asarray(node.position_xyz, dtype=np.float64)[
                    [0, 2]
                ]
            except (IndexError, TypeError, ValueError):
                return {"trajectory_route_status": "invalid_active_trajectory_route"}
            candidates.append(
                (float(np.linalg.norm(current - node_xz)), route_index)
            )
        distance_m, route_index = min(candidates)
        connector_limit_m = max(
            float(getattr(self, "sg_path_deviation_m", 0.35)),
            float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE),
        ) + float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
        if distance_m > connector_limit_m:
            return {
                "trajectory_route_status": "active_trajectory_route_too_far",
                "trajectory_start_distance_m": float(distance_m),
                "trajectory_connector_limit_m": float(connector_limit_m),
            }
        remaining_ids = list(map(int, route_ids[route_index:]))
        if not remaining_ids:
            return {"trajectory_route_status": "active_trajectory_route_exhausted"}
        return {
            "trajectory_route_status": "connected",
            "trajectory_start_node_id": int(remaining_ids[0]),
            "trajectory_start_distance_m": float(distance_m),
            "trajectory_target_node_id": int(remaining_ids[-1]),
            "trajectory_route_node_ids": remaining_ids,
            "trajectory_route_consumed_nodes": int(route_index),
        }

    def _sg_safe_navigation_mask(self, free_mask):
        free_mask = np.asarray(free_mask, dtype=bool)
        if free_mask.ndim != 2 or not free_mask.any():
            return None, 0.0
        obstacle_mask = getattr(self, "_sg_memory_obstacle_cache", None)
        clearance_cells = float(
            os.environ.get("LIFELONG_SG_DISCRETE_CLEARANCE_CELLS", "2.5")
        )
        if (
            clearance_cells > 0
            and obstacle_mask is not None
            and np.asarray(obstacle_mask).shape == free_mask.shape
        ):
            distance_to_obstacle = cv2.distanceTransform(
                np.logical_not(np.asarray(obstacle_mask, dtype=bool)).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            safe_mask = np.logical_and(free_mask, distance_to_obstacle > clearance_cells)
        else:
            clearance_cells = 0.0
            safe_mask = free_mask.copy()
        collision_mask = getattr(self, "collision_map", None)
        if collision_mask is not None and np.asarray(collision_mask).shape == safe_mask.shape:
            safe_mask = np.logical_and(safe_mask, np.logical_not(np.asarray(collision_mask, dtype=bool)))
        return safe_mask, clearance_cells

    def _sg_execution_navigation_mask(self, free_mask):
        """Known M0 free space plus collision feedback for one-step execution."""
        free_mask = np.asarray(free_mask, dtype=bool)
        if free_mask.ndim != 2 or not free_mask.any():
            return None
        mask = free_mask.copy()
        collision_mask = getattr(self, "collision_map", None)
        if collision_mask is not None and np.asarray(collision_mask).shape == mask.shape:
            mask = np.logical_and(mask, np.logical_not(np.asarray(collision_mask, dtype=bool)))
        return mask

    @staticmethod
    def _sg_connect_start_to_safe_mask(
        safe_mask, execution_mask, start, max_radius_cells
    ):
        """Connect the exact current cell to nearby clearance-safe known space."""
        safe_mask = np.asarray(safe_mask, dtype=bool).copy()
        execution_mask = np.asarray(execution_mask, dtype=bool)
        if safe_mask.shape != execution_mask.shape or safe_mask.ndim != 2:
            return None, []
        start = (int(start[0]), int(start[1]))
        height, width = safe_mask.shape
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            return None, []
        if not execution_mask[start[1], start[0]]:
            return None, []
        if safe_mask[start[1], start[0]]:
            return safe_mask, [start]

        radius = max(0, int(max_radius_cells))
        queue = deque([start])
        parent = {start: None}
        depth = {start: 0}
        endpoint = None
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while queue:
            current = queue.popleft()
            if current != start and safe_mask[current[1], current[0]]:
                endpoint = current
                break
            if depth[current] >= radius:
                continue
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if neighbor in parent or not execution_mask[ny, nx]:
                    continue
                parent[neighbor] = current
                depth[neighbor] = depth[current] + 1
                queue.append(neighbor)

        if endpoint is None:
            return None, []
        connector = []
        current = endpoint
        while current is not None:
            connector.append(current)
            current = parent[current]
        connector.reverse()
        for x, y in connector:
            safe_mask[y, x] = True
        return safe_mask, connector

    @staticmethod
    def _sg_grid_line_is_safe(mask, start, end):
        """Conservative supercover visibility between integer grid-cell centers."""
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        height, width = mask.shape

        def safe(x, y):
            return 0 <= x < width and 0 <= y < height and bool(mask[y, x])

        if not safe(x0, y0) or not safe(x1, y1):
            return False
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        x, y = x0, y0
        ix = iy = 0
        while ix < dx or iy < dy:
            lhs = (1 + 2 * ix) * dy
            rhs = (1 + 2 * iy) * dx
            if lhs == rhs:
                if not safe(x + sx, y) or not safe(x, y + sy):
                    return False
                x += sx
                y += sy
                ix += 1
                iy += 1
            elif lhs < rhs:
                x += sx
                ix += 1
            else:
                y += sy
                iy += 1
            if not safe(x, y):
                return False
        return True

    @staticmethod
    def _sg_float_segment_is_safe(mask, start, end):
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        distance = float(np.hypot(x1 - x0, y1 - y0))
        samples = max(1, int(np.ceil(distance * 2.0)))
        height, width = mask.shape
        for index in range(samples + 1):
            fraction = index / samples
            x = int(round(x0 + (x1 - x0) * fraction))
            y = int(round(y0 + (y1 - y0) * fraction))
            if not (0 <= x < width and 0 <= y < height and mask[y, x]):
                return False
        return True

    def _sg_reachable_target(
        self,
        safe_mask,
        start,
        raw_target,
        exact,
        target_validator=None,
    ):
        height, width = safe_mask.shape
        raw_x, raw_y = int(raw_target[0]), int(raw_target[1])
        visited = np.zeros_like(safe_mask, dtype=bool)
        queue = deque([start])
        visited[start[1], start[0]] = True
        cells = []
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        while queue:
            x, y = queue.popleft()
            cells.append((x, y))
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if visited[ny, nx] or not safe_mask[ny, nx]:
                    continue
                if dx and dy and (not safe_mask[y, nx] or not safe_mask[ny, x]):
                    continue
                visited[ny, nx] = True
                queue.append((nx, ny))

        if exact:
            if (
                0 <= raw_x < width
                and 0 <= raw_y < height
                and visited[raw_y, raw_x]
                and (
                    target_validator is None
                    or bool(target_validator((raw_x, raw_y)))
                )
            ):
                return (raw_x, raw_y), len(cells), 0.0
            return None, len(cells), float("inf")
        if not cells:
            return None, 0, float("inf")
        maximum_distance = float(self.sg_snap_radius_cells)
        ranked = sorted(
            (
                (point[0] - raw_x) ** 2 + (point[1] - raw_y) ** 2,
                point[1],
                point[0],
            )
            for point in cells
            if abs(point[0] - raw_x) <= maximum_distance
            and abs(point[1] - raw_y) <= maximum_distance
            and (point[0] - raw_x) ** 2 + (point[1] - raw_y) ** 2
            <= maximum_distance * maximum_distance + 1e-6
        )
        if not ranked:
            closest = min(
                cells,
                key=lambda point: (
                    (point[0] - raw_x) ** 2
                    + (point[1] - raw_y) ** 2,
                    point[1],
                    point[0],
                ),
            )
            return (
                None,
                len(cells),
                float(np.hypot(closest[0] - raw_x, closest[1] - raw_y)),
            )
        for distance_squared, target_y, target_x in ranked:
            target = (int(target_x), int(target_y))
            if target_validator is not None and not bool(
                target_validator(target)
            ):
                continue
            return target, len(cells), float(np.sqrt(distance_squared))
        return None, len(cells), float(np.sqrt(ranked[0][0]))

    def _sg_theta_star(self, safe_mask, start, goal):
        if start == goal:
            return [start], 0
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        max_states = int(os.environ.get("LIFELONG_SG_THETA_MAX_STATES", "250000"))
        g_score = {start: 0.0}
        parent = {start: start}
        open_heap = [(float(np.hypot(goal[0] - start[0], goal[1] - start[1])), 0.0, start[1], start[0])]
        closed = set()
        expanded = 0
        height, width = safe_mask.shape

        while open_heap and expanded < max_states:
            _, queued_g, y, x = heapq.heappop(open_heap)
            current = (x, y)
            if current in closed or queued_g != g_score.get(current):
                continue
            closed.add(current)
            expanded += 1
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path, expanded

            for dx, dy in neighbors:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor
                if not (0 <= nx < width and 0 <= ny < height and safe_mask[ny, nx]):
                    continue
                if dx and dy and (not safe_mask[y, nx] or not safe_mask[ny, x]):
                    continue
                anchor = parent[current]
                if anchor != current and self._sg_grid_line_is_safe(safe_mask, anchor, neighbor):
                    candidate_parent = anchor
                else:
                    candidate_parent = current
                candidate_g = g_score[candidate_parent] + float(
                    np.hypot(neighbor[0] - candidate_parent[0], neighbor[1] - candidate_parent[1])
                )
                if candidate_g + 1e-9 >= g_score.get(neighbor, float("inf")):
                    continue
                g_score[neighbor] = candidate_g
                parent[neighbor] = candidate_parent
                heuristic = float(np.hypot(goal[0] - nx, goal[1] - ny))
                heapq.heappush(open_heap, (candidate_g + heuristic, candidate_g, ny, nx))
        return None, expanded

    def _sg_string_pull_path(self, safe_mask, path):
        if len(path) <= 2:
            return list(path)
        result = [path[0]]
        anchor_index = 0
        while anchor_index < len(path) - 1:
            next_index = len(path) - 1
            while next_index > anchor_index + 1:
                if self._sg_grid_line_is_safe(safe_mask, path[anchor_index], path[next_index]):
                    break
                next_index -= 1
            result.append(path[next_index])
            anchor_index = next_index
        return result

    @staticmethod
    def _sg_path_cumulative(path):
        cumulative = [0.0]
        for start, end in zip(path[:-1], path[1:]):
            cumulative.append(cumulative[-1] + float(np.hypot(end[0] - start[0], end[1] - start[1])))
        return cumulative

    def _sg_plan_theta_path(self, free_mask, raw_target, current_pose, exact=False):
        self._sg_last_theta_failure = None
        if current_pose is None:
            self._sg_last_theta_failure = {"reason": "missing_current_pose"}
            return None
        safe_mask, clearance_cells = self._sg_safe_navigation_mask(free_mask)
        if safe_mask is None:
            self._sg_last_theta_failure = {"reason": "missing_safe_mask"}
            return None
        resolution_cm = float(self.resolution)
        start = (
            int(round(float(current_pose[0]) * 100.0 / resolution_cm)),
            int(round(float(current_pose[1]) * 100.0 / resolution_cm)),
        )
        height, width = safe_mask.shape
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            self._sg_last_theta_failure = {
                "reason": "start_outside_map",
                "start_map": start,
            }
            return None
        execution_mask = self._sg_execution_navigation_mask(free_mask)
        max_connector_radius = int(
            os.environ.get("LIFELONG_SG_START_CONNECT_RADIUS_CELLS", "6")
        )
        safe_mask, start_connector = self._sg_connect_start_to_safe_mask(
            safe_mask,
            execution_mask,
            start,
            max_connector_radius,
        )
        if safe_mask is None:
            known_free_at_start = bool(
                execution_mask is not None
                and execution_mask.shape == (height, width)
                and execution_mask[start[1], start[0]]
            )
            self._sg_last_theta_failure = {
                "reason": "start_not_connected_to_safe_space",
                "start_map": start,
                "known_free_at_start": known_free_at_start,
                "max_connector_radius_cells": max_connector_radius,
            }
            return None
        target, reachable_cells, raw_distance = self._sg_reachable_target(
            safe_mask, start, raw_target, exact
        )
        if target is None:
            self._sg_last_theta_failure = {
                "reason": "no_reachable_target_within_snap_radius",
                "start_map": start,
                "raw_target_map": tuple(map(int, raw_target)),
                "reachable_cells": int(reachable_cells),
                "raw_distance_cells": float(raw_distance),
            }
            return None
        raw_path, expanded = self._sg_theta_star(safe_mask, start, target)
        if not raw_path:
            self._sg_last_theta_failure = {
                "reason": "theta_star_failed",
                "start_map": start,
                "target_map": target,
                "expanded_states": int(expanded),
            }
            return None
        path = self._sg_string_pull_path(safe_mask, raw_path)
        cumulative = self._sg_path_cumulative(path)
        return {
            "target_map": (int(target[0]), int(target[1])),
            "path": [(float(x), float(y)) for x, y in path],
            "raw_path": [(int(x), int(y)) for x, y in raw_path],
            "path_length_cells": float(cumulative[-1]),
            "expanded_states": int(expanded),
            "reachable_cells": int(reachable_cells),
            "clearance_cells": float(clearance_cells),
            "raw_distance_cells": float(raw_distance),
            "start_connector_path": [
                (int(x), int(y)) for x, y in start_connector
            ],
            "start_connector_length_cells": float(
                max(0, len(start_connector) - 1)
            ),
            "safe_mask": safe_mask,
        }

    @staticmethod
    def _sg_point_on_path(path, cumulative, progress):
        if not path:
            return None
        progress = min(max(float(progress), 0.0), cumulative[-1])
        for index in range(len(path) - 1):
            if progress > cumulative[index + 1]:
                continue
            segment = cumulative[index + 1] - cumulative[index]
            if segment <= 1e-9:
                return np.asarray(path[index + 1], dtype=np.float64)
            fraction = (progress - cumulative[index]) / segment
            start = np.asarray(path[index], dtype=np.float64)
            end = np.asarray(path[index + 1], dtype=np.float64)
            return start + fraction * (end - start)
        return np.asarray(path[-1], dtype=np.float64)

    def _sg_path_progress_and_lookahead(self, path, position, previous_progress, safe_mask):
        cumulative = self._sg_path_cumulative(path)
        projected_progress, cross_track = self._sg_project_position_to_path(
            path, position, previous_progress, cumulative=cumulative
        )
        if projected_progress is None:
            progress, cross_track = previous_progress, float("inf")
        else:
            progress = max(float(previous_progress), float(projected_progress))

        lookahead_cells = self.sg_path_lookahead_m * 100.0 / self.resolution
        minimum_cells = float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE) * 100.0 / self.resolution
        desired = min(cumulative[-1], progress + lookahead_cells)
        candidate_progress = desired
        lookahead = None
        while candidate_progress >= min(cumulative[-1], progress + minimum_cells) - 1e-6:
            candidate = self._sg_point_on_path(path, cumulative, candidate_progress)
            if candidate is not None and self._sg_float_segment_is_safe(safe_mask, position, candidate):
                lookahead = candidate
                break
            candidate_progress -= minimum_cells
        if lookahead is None:
            lookahead = self._sg_point_on_path(
                path,
                cumulative,
                min(cumulative[-1], progress + minimum_cells),
            )
        return progress, cross_track, lookahead, cumulative[-1]

    @staticmethod
    def _sg_project_position_to_path(path, position, minimum_progress, cumulative=None):
        if cumulative is None:
            cumulative = INHeritSGZeroShotVlnEvaluatorMP._sg_path_cumulative(path)
        best = None
        for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
            start = np.asarray(start, dtype=np.float64)
            end = np.asarray(end, dtype=np.float64)
            delta = end - start
            length_sq = float(np.dot(delta, delta))
            fraction = 0.0 if length_sq <= 1e-9 else float(np.clip(np.dot(position - start, delta) / length_sq, 0.0, 1.0))
            projection = start + fraction * delta
            progress = cumulative[index] + fraction * float(np.sqrt(length_sq))
            if progress + 1.0 < float(minimum_progress):
                continue
            distance = float(np.linalg.norm(position - projection))
            key = (distance, -progress)
            if best is None or key < best[0]:
                best = (key, progress, distance)
        if best is None:
            return None, float("inf")
        return float(best[1]), float(best[2])

    def _sg_replan_theta_path(self, target_info, target, current_pose, reason):
        memory_traversable = self._sg_memory_traversable_mask()
        replanned = self._sg_plan_theta_path(memory_traversable, target, current_pose, exact=True)
        if replanned is None:
            self._sg_path_plan_active = False
            return None
        self._sg_path_replans += 1
        target_info["_sg_correction_replan_count"] = int(
            target_info.get("_sg_correction_replan_count", 0)
        ) + 1
        target_info["_sg_path_cells"] = list(replanned["path"])
        target_info["_sg_path_progress_cells"] = 0.0
        logger.info(
            "[INHerit-SG-FIRST] replanned Theta* SG path: "
            f"reason={reason}, waypoints={len(replanned['path'])}, "
            f"length={replanned['path_length_cells'] * self.resolution / 100.0:.3f}m"
        )
        return replanned

    def _sg_world_path_progress_and_lookahead(
        self, path, position, previous_progress, window, safe_mask
    ):
        cumulative = self._sg_path_cumulative(path)
        projected_progress, cross_track = self._sg_project_position_to_path(
            path, position, previous_progress, cumulative=cumulative
        )
        if projected_progress is None:
            progress, cross_track = previous_progress, float("inf")
        else:
            progress = max(float(previous_progress), float(projected_progress))

        metric_map = self._sg_world_metric_map()
        resolution_m = float(metric_map.spec.resolution_m)
        lookahead_cells = self.sg_path_lookahead_m / resolution_m
        minimum_cells = (
            float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
            / resolution_m
        )
        desired = min(cumulative[-1], progress + lookahead_cells)
        minimum_progress = min(cumulative[-1], progress + minimum_cells)
        candidate_progress = desired
        lookahead = None
        position_local = self._sg_window_point(window, position)
        while candidate_progress >= minimum_progress - 1e-6:
            candidate = self._sg_point_on_path(path, cumulative, candidate_progress)
            candidate_local = self._sg_window_point(window, candidate)
            if candidate is not None and self._sg_float_segment_is_safe(
                safe_mask, position_local, candidate_local
            ):
                lookahead = candidate
                break
            candidate_progress -= minimum_cells
        if lookahead is None:
            lookahead = self._sg_point_on_path(path, cumulative, minimum_progress)
        return progress, cross_track, lookahead, cumulative[-1]

    def _sg_replan_world_path(self, target_info, current_world_xz, reason):
        target_world = target_info.get("target_world_xz")
        if target_world is None:
            self._sg_path_plan_active = False
            return None
        immediate_reverse_replan = str(reason).startswith(
            "immediate_reverse_without_progress"
        )
        if immediate_reverse_replan:
            immediate_reverse_count = int(
                target_info.get("_sg_consecutive_immediate_reverse_replans", 0)
            ) + 1
            target_info["_sg_consecutive_immediate_reverse_replans"] = (
                immediate_reverse_count
            )
            if immediate_reverse_count >= 2:
                target_info["_sg_force_local_action_recovery"] = True
        else:
            target_info["_sg_consecutive_immediate_reverse_replans"] = 0
        # Runtime collision evidence can split a free-space component or show
        # that the old snapped cell was on the wrong side of a real boundary.
        # In that case keep the object centre as the objective, but recompute
        # the closest point in the *current reachable component*.  Ordinary
        # cross-track replans retain the already selected exact terminal cell.
        resnap_reasons = (
            "collision_displacement_",
            "no_safe_discrete_forward",
            "no_progressing_discrete_forward",
            "local_action_lattice_unreachable",
        )
        # An immediate reverse is controller-phase evidence, not evidence that
        # the selected terminal cell became invalid.  Resnapping here can move
        # the objective away after the agent has already reached its attainable
        # terminal neighbourhood (829901 ep2).  Replan exactly to the existing
        # target and leave target changes to geometry/safety recovery reasons.
        allow_target_resnap = str(reason).startswith(resnap_reasons)
        plan_target = target_world
        exact = True
        if allow_target_resnap:
            plan_target = target_info.get("raw_target_world_xz", target_world)
            exact = False
        remaining_route = self._sg_remaining_trajectory_route_metadata(
            target_info, current_world_xz
        )
        remaining_route_anchors = self._sg_trajectory_route_world_xz(
            remaining_route
        )
        planning_anchors = (
            remaining_route_anchors
            if remaining_route_anchors
            else target_info.get("_sg_route_planning_anchors_world_xz")
        )
        replanned = self._sg_plan_world_path(
            plan_target,
            current_world_xz,
            exact=exact,
            planning_anchor_world_xz=planning_anchors,
        )
        if replanned is None:
            self._sg_path_plan_active = False
            target_info["_sg_consecutive_failed_recoveries"] = int(
                target_info.get("_sg_consecutive_failed_recoveries", 0)
            ) + 1
            return None
        resolution_m = float(target_info["world_map_resolution_m"])
        effective_object_center_residual_m = None
        if allow_target_resnap:
            candidate_residual_m = float(
                np.linalg.norm(
                    np.asarray(
                        replanned["target_world_xz"], dtype=np.float64
                    )
                    - np.asarray(
                        target_info.get(
                            "raw_target_world_xz", replanned["target_world_xz"]
                        ),
                        dtype=np.float64,
                    )
                )
            )
            best_residual_m = float(
                target_info.get(
                    "_sg_best_object_center_residual_m",
                    target_info.get(
                        "reachable_object_center_residual_m",
                        candidate_residual_m,
                    ),
                )
            )
            max_degradation_m = float(
                os.environ.get(
                    "LIFELONG_SG_MAX_TARGET_RESNAP_DEGRADATION_M", "0.50"
                )
            )
            if candidate_residual_m > best_residual_m + max_degradation_m:
                rejected_target = tuple(map(float, replanned["target_world_xz"]))
                preserved_target = target_info.get(
                    "_sg_best_object_center_target_world_xz"
                )
                preserved = None
                if preserved_target is not None:
                    preserved = self._sg_plan_world_path(
                        preserved_target,
                        current_world_xz,
                        exact=True,
                        planning_anchor_world_xz=planning_anchors,
                    )
                if preserved is None:
                    preserved_failure = copy.deepcopy(
                        getattr(self, "_sg_last_theta_failure", None)
                    )
                    self._sg_last_theta_failure = {
                        "reason": "target_resnap_fidelity_degraded",
                        "candidate_target_world_xz": rejected_target,
                        "candidate_object_center_residual_m": candidate_residual_m,
                        "best_object_center_residual_m": best_residual_m,
                        "max_degradation_m": max_degradation_m,
                        "preserved_target_world_xz": preserved_target,
                        "preserved_target_planner_failure": preserved_failure,
                    }
                    target_info["_sg_rejected_target_resnap"] = copy.deepcopy(
                        self._sg_last_theta_failure
                    )
                    self._sg_path_plan_active = False
                    logger.warning(
                        "[INHerit-SG-V7] rejected degraded target resnap: "
                        f"candidate_residual={candidate_residual_m:.3f}m, "
                        f"best_residual={best_residual_m:.3f}m, "
                        f"limit={best_residual_m + max_degradation_m:.3f}m"
                    )
                    return None
                replanned = preserved
                effective_object_center_residual_m = best_residual_m
                replanned["target_preserved_for_fidelity"] = True
                logger.info(
                    "[INHerit-SG-V7] preserved best object-center target "
                    "instead of degraded current-component resnap: "
                    f"rejected_residual={candidate_residual_m:.3f}m, "
                    f"preserved_residual={best_residual_m:.3f}m"
                )
            else:
                effective_object_center_residual_m = candidate_residual_m

        trajectory_guidance = self._sg_compose_trajectory_guided_world_path(
            replanned,
            remaining_route,
            current_world_xz,
        )
        active_source = str(
            target_info.get("execution_path_source", "")
        )
        shortcut_sources = {
            "observed_free_theta_star",
            "certified_hybrid_theta_star",
        }
        shortcut_failure_reasons = (
            "collision_displacement_",
            "no_safe_discrete_forward",
            "no_progressing_discrete_forward",
            "local_action_lattice_unreachable",
        )
        shortcut_execution_failed = bool(
            active_source in shortcut_sources
            and str(reason).startswith(shortcut_failure_reasons)
        )
        allow_shortcut = not bool(
            target_info.get("_sg_shortcut_fallback_engaged", False)
            or active_source
            == "certified_trajectory_then_world_theta_star"
            or shortcut_execution_failed
        )
        path_selection = self._sg_select_execution_world_path(
            replanned,
            trajectory_guidance,
            allow_shortcut=allow_shortcut,
        )
        replanned["path"] = list(path_selection["path"])
        replanned["path_length_cells"] = float(
            path_selection["path_length_cells"]
        )
        selected_source = str(path_selection["execution_path_source"])
        path_selection_metadata = copy.deepcopy(path_selection)
        path_selection_metadata.pop("path", None)
        target_info["path_selection"] = path_selection_metadata
        target_info["execution_path_source"] = selected_source
        target_info["trajectory_guidance"] = copy.deepcopy(
            trajectory_guidance
        )
        if shortcut_execution_failed:
            target_info["_sg_shortcut_fallback_engaged"] = True
            target_info["_sg_shortcut_failure_reason"] = str(reason)
        if selected_source == (
            "certified_trajectory_then_world_theta_star"
        ):
            target_info.update(remaining_route)
            target_info["_sg_route_planning_anchors_world_xz"] = [
                [float(point[0]), float(point[1])]
                for point in remaining_route_anchors
            ]
            target_info["_sg_trajectory_suffix_start_progress_cells"] = float(
                trajectory_guidance[
                    "trajectory_suffix_start_progress_cells"
                ]
            )
        else:
            target_info.pop(
                "_sg_trajectory_suffix_start_progress_cells", None
            )
        self._sg_path_replans += 1
        # This is diagnostic total work, not a failure budget.  A successfully
        # installed path must be allowed to execute before recovery can fail.
        target_info["_sg_correction_replan_count"] = int(
            target_info.get("_sg_correction_replan_count", 0)
        ) + 1
        if allow_target_resnap:
            target_grid = tuple(map(int, replanned["target_grid"]))
            target_world_xz = tuple(map(float, replanned["target_world_xz"]))
            target_info["target_world_grid_x"] = target_grid[0]
            target_info["target_world_grid_z"] = target_grid[1]
            target_info["target_world_xz"] = list(target_world_xz)
            raw_distance_m = (
                effective_object_center_residual_m
                if effective_object_center_residual_m is not None
                else float(replanned.get("raw_distance_cells", 0.0))
                * resolution_m
            )
            target_info["reachable_object_center_residual_m"] = raw_distance_m
            if bool(replanned.get("target_preserved_for_fidelity")):
                if target_info.get("navigation_reference_policy") == (
                    "sofa_best_observation_ray_plus_half_depth"
                ):
                    target_info["navigation_target_source"] = (
                        "sofa_observation_ray_center_preserved_best_world_free"
                    )
                else:
                    target_info["navigation_target_source"] = (
                        "object_center_preserved_best_world_free"
                    )
            else:
                if target_info.get("navigation_reference_policy") == (
                    "sofa_best_observation_ray_plus_half_depth"
                ):
                    target_info["navigation_target_source"] = (
                        "sofa_observation_ray_center_nearest_current_reachable_world_free"
                    )
                else:
                    target_info["navigation_target_source"] = (
                        "object_center_nearest_current_reachable_world_free"
                    )
            target_info["reachable_navigation_reference_residual_m"] = (
                raw_distance_m
            )
            semantic_center = target_info.get(
                "semantic_object_center_world_xz"
            )
            if semantic_center is not None:
                target_info[
                    "reachable_semantic_object_center_residual_m"
                ] = float(
                    np.linalg.norm(
                        np.asarray(target_world_xz, dtype=np.float64)
                        - np.asarray(semantic_center, dtype=np.float64)
                    )
                )
            best_residual_m = float(
                target_info.get(
                    "_sg_best_object_center_residual_m", raw_distance_m
                )
            )
            if raw_distance_m < best_residual_m - 1e-9:
                target_info["_sg_best_object_center_residual_m"] = (
                    raw_distance_m
                )
                target_info["_sg_best_object_center_target_world_xz"] = [
                    float(target_world_xz[0]),
                    float(target_world_xz[1]),
                ]
        target_info["_sg_world_path_cells"] = list(replanned["path"])
        target_info["_sg_path_progress_cells"] = 0.0
        target_info.pop("_sg_local_action_queue", None)
        target_info.pop("_sg_local_action_goal_tolerance_cells", None)
        target_info.pop("_sg_local_action_goal_kind", None)
        target_info.pop("_sg_local_action_goal_world_grid", None)
        target_info.pop("_sg_local_action_goal_path_progress_cells", None)
        target_info.pop("_sg_local_action_recovery_active", None)
        # A replan installs a different path coordinate system.  Progress must
        # reset, while a successfully recomposed certified route keeps its new
        # phase boundary.  Plain Theta* replans have no such boundary.
        if selected_source != (
            "certified_trajectory_then_world_theta_star"
        ):
            target_info.pop("_sg_trajectory_suffix_start_progress_cells", None)
        self._sg_reset_world_path_stall_guard(target_info, clear_phase=True)
        # A successful replan supersedes the old path's one-step anti-oscillation
        # history.  Keeping that origin would reject the first deliberate
        # corrective backtrack on the new path as an "immediate reverse", then
        # abort even though Theta* found a valid route.  Once the corrective
        # MOVE_FORWARD executes, the normal pending-forward bookkeeping installs
        # a fresh guard and still prevents ping-pong on the new path.
        target_info.pop("_sg_last_forward_world_grid_origin", None)
        target_info["_sg_runtime_collision_block_count"] = len(
            self._sg_world_collision_blocked_cells
        )
        target_info["_sg_runtime_collision_overlay_cells"] = int(
            replanned.get("runtime_collision_overlay_cells", 0)
        )
        # A previous failed correction may have disabled the generic SG stall
        # guard.  Installing a valid new world path reactivates path-scoped
        # progress accounting; otherwise the legacy straight-line watchdog can
        # abort a certified detour while it is still executing (829951 ep1).
        self._sg_path_plan_active = True
        logger.info(
            "[INHerit-SG-V7] replanned world Theta* path: "
            f"reason={reason}, waypoints={len(replanned['path'])}, "
            f"length={replanned['path_length_cells'] * resolution_m:.3f}m, "
            f"target_resnapped={allow_target_resnap}, "
            "execution_path_source="
            f"{target_info.get('execution_path_source')}, "
            "target_world_xz="
            f"{tuple(map(float, target_info['target_world_xz']))}, "
            "object_center_residual_m="
            f"{float(target_info.get('reachable_object_center_residual_m', 0.0)):.3f}, "
            f"collision_blocks={len(self._sg_world_collision_blocked_cells)}, "
            "collision_overlay_cells="
            f"{int(replanned.get('runtime_collision_overlay_cells', 0))}"
        )
        return replanned

    @staticmethod
    def _sg_discrete_heading_lattice(current_heading, turn_rad):
        """Return the headings Habitat can reach from the measured yaw phase."""

        current_heading = float(current_heading)
        turn_rad = float(turn_rad)
        if (
            not np.isfinite(current_heading)
            or not np.isfinite(turn_rad)
            or turn_rad <= 0
        ):
            return []
        num_headings = int(round(2.0 * np.pi / turn_rad))
        return [
            float(
                np.arctan2(
                    np.sin(current_heading + offset * turn_rad),
                    np.cos(current_heading + offset * turn_rad),
                )
            )
            for offset in range(num_headings)
        ]

    def _sg_trajectory_execution_slack_cells(self, spec):
        """Bound the raster tube needed by Habitat's discrete action lattice.

        A certified acquisition edge is continuous, while MOVE_FORWARD can
        only use headings separated by TURN_ANGLE.  The worst heading phase is
        half a turn, producing ``step * sin(turn / 2)`` lateral error.  Add a
        half-cell for raster rounding and round up.  For the benchmark's
        0.25-m step, 30-degree turn, and 0.05-m cells this is two cells, not the
        old empirical one-cell allowance.

        An explicit environment value remains available for ablations.
        """

        configured = os.environ.get(
            "LIFELONG_SG_TRAJECTORY_EXECUTION_SLACK_CELLS"
        )
        if configured is not None:
            return max(0.0, float(configured))
        resolution_m = float(spec.resolution_m)
        if not np.isfinite(resolution_m) or resolution_m <= 0:
            return 0.0
        forward_step_m = float(
            self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        )
        turn_rad = np.deg2rad(
            float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        )
        lateral_error_m = abs(forward_step_m) * abs(
            np.sin(turn_rad * 0.5)
        )
        raster_rounding_m = resolution_m * 0.5
        return float(
            max(
                0,
                int(
                    np.ceil(
                        (lateral_error_m + raster_rounding_m)
                        / resolution_m
                        - 1e-9
                    )
                ),
            )
        )

    def _sg_live_navmesh_context(self):
        """Return the live Habitat pathfinder and current floor height."""

        try:
            envs = getattr(self, "envs", None)
            if envs is None:
                return None
            habitat_env = envs.call_at(0, "_env")
            simulator = habitat_env._sim
            state = simulator.get_agent_state()
            pathfinder = simulator.pathfinder
            if pathfinder is None or not bool(pathfinder.is_loaded):
                return None
            return pathfinder, float(state.position[1])
        except Exception as exc:
            logger.warning(
                "[INHerit-SG-V7] live navmesh unavailable: "
                f"{exc}"
            )
            return None

    def _sg_validate_warm_start_connector(
        self, start_world_xz, end_world_xz
    ):
        """Certify one bounded branch connector with the live simulator.

        This validates only the short straight join into persisted trajectory
        evidence.  It neither exposes nor consumes a navmesh route.
        """

        start = np.asarray(start_world_xz, dtype=np.float64).reshape(-1)
        end = np.asarray(end_world_xz, dtype=np.float64).reshape(-1)
        metadata = {
            "status": "invalid",
            "distance_m": None,
            "same_island": None,
            "try_step_residual_m": None,
        }
        if (
            start.size != 2
            or end.size != 2
            or not np.isfinite(start).all()
            or not np.isfinite(end).all()
        ):
            return False, metadata
        distance_m = float(np.linalg.norm(end - start))
        metadata["distance_m"] = distance_m
        maximum = float(
            getattr(self, "sg_warm_start_trajectory_connect_m", 0.35)
        )
        if distance_m > maximum + 1e-6:
            metadata["status"] = "distance_exceeds_bound"
            return False, metadata
        if not bool(
            getattr(self, "sg_navmesh_target_validation", True)
        ):
            metadata["status"] = "live_validation_disabled"
            return True, metadata
        context = self._sg_live_navmesh_context()
        if context is None:
            metadata["status"] = "live_navmesh_unavailable"
            return False, metadata
        pathfinder, floor_y = context
        start_point = np.asarray(
            [float(start[0]), float(floor_y), float(start[1])],
            dtype=np.float32,
        )
        end_point = np.asarray(
            [float(end[0]), float(floor_y), float(end[1])],
            dtype=np.float32,
        )
        try:
            if not bool(pathfinder.is_navigable(start_point)):
                metadata["status"] = "start_not_navigable"
                return False, metadata
            if not bool(pathfinder.is_navigable(end_point)):
                metadata["status"] = "endpoint_not_navigable"
                return False, metadata
            if hasattr(pathfinder, "get_island"):
                same_island = int(pathfinder.get_island(start_point)) == int(
                    pathfinder.get_island(end_point)
                )
            else:
                same_island = self._sg_navmesh_points_connected(
                    pathfinder, start_point, end_point
                )
            metadata["same_island"] = bool(same_island)
            if same_island is not True:
                metadata["status"] = "different_or_unverified_island"
                return False, metadata
            stepped = np.asarray(
                pathfinder.try_step(start_point, end_point),
                dtype=np.float64,
            ).reshape(-1)
            if stepped.size < 3 or not np.isfinite(stepped[:3]).all():
                metadata["status"] = "try_step_invalid"
                return False, metadata
            residual_m = float(np.linalg.norm(stepped[[0, 2]] - end))
            metadata["try_step_residual_m"] = residual_m
            if residual_m > 0.05 + 1e-6:
                metadata["status"] = "try_step_blocked"
                return False, metadata
        except Exception as exc:
            metadata["status"] = "navmesh_validation_failed"
            metadata["detail"] = str(exc)
            return False, metadata
        metadata["status"] = "validated"
        return True, metadata

    def _sg_project_target_to_current_navmesh_island(
        self,
        raw_target_world_xz,
        current_world_xz,
    ):
        """Project an object centre onto the current island's terminal set.

        ``PathFinder.snap_point`` supplies only the nearest standable endpoint;
        the returned point is subsequently intersected with M0 safe cells and
        routed by world-map Theta*.  No navmesh path is requested here.
        """

        raw_target = np.asarray(
            raw_target_world_xz, dtype=np.float64
        ).reshape(-1)
        current = np.asarray(current_world_xz, dtype=np.float64).reshape(-1)
        metadata = {"status": "disabled"}
        if raw_target.size != 2 or not np.isfinite(raw_target).all():
            metadata["status"] = "invalid_raw_target"
            return raw_target, metadata
        if not bool(
            getattr(self, "sg_navmesh_target_validation", True)
        ):
            return raw_target.copy(), metadata
        context = self._sg_live_navmesh_context()
        if context is None:
            metadata["status"] = "live_navmesh_unavailable"
            return raw_target.copy(), metadata
        if current.size != 2 or not np.isfinite(current).all():
            metadata["status"] = "invalid_current_world_pose"
            return raw_target.copy(), metadata
        pathfinder, floor_y = context
        current_point = np.asarray(
            [float(current[0]), float(floor_y), float(current[1])],
            dtype=np.float32,
        )
        raw_point = np.asarray(
            [float(raw_target[0]), float(floor_y), float(raw_target[1])],
            dtype=np.float32,
        )
        try:
            if not bool(pathfinder.is_navigable(current_point)):
                metadata["status"] = "current_pose_not_navigable"
                return raw_target.copy(), metadata
            has_island_api = hasattr(pathfinder, "get_island")
            current_island = (
                int(pathfinder.get_island(current_point))
                if has_island_api
                else None
            )
            snapped = np.asarray(
                pathfinder.snap_point(raw_point, current_island)
                if has_island_api
                else pathfinder.snap_point(raw_point),
                dtype=np.float64,
            ).reshape(-1)
            if snapped.size < 3 or not np.isfinite(snapped[:3]).all():
                metadata["status"] = "snap_returned_invalid_point"
                return raw_target.copy(), metadata
            if not bool(pathfinder.is_navigable(snapped[:3])):
                metadata["status"] = "snapped_point_not_navigable"
                return raw_target.copy(), metadata
            if has_island_api:
                if int(pathfinder.get_island(snapped[:3])) != current_island:
                    metadata["status"] = "snapped_point_on_other_island"
                    return raw_target.copy(), metadata
                connectivity_mode = "island_id"
            else:
                connected = self._sg_navmesh_points_connected(
                    pathfinder,
                    current_point,
                    snapped[:3],
                )
                if connected is not True:
                    metadata["status"] = (
                        "snapped_point_connectivity_unavailable"
                        if connected is None
                        else "snapped_point_disconnected"
                    )
                    return raw_target.copy(), metadata
                connectivity_mode = "shortest_path_boolean"
        except Exception as exc:
            metadata["status"] = "snap_failed"
            metadata["detail"] = str(exc)
            return raw_target.copy(), metadata
        projected = snapped[[0, 2]].astype(np.float64)
        metadata.update(
            {
                "status": "applied",
                "current_island": (
                    None if current_island is None else int(current_island)
                ),
                "connectivity_mode": connectivity_mode,
                "raw_object_center_world_xz": list(map(float, raw_target)),
                "projected_world_xz": list(map(float, projected)),
                "projection_distance_m": float(
                    np.linalg.norm(projected - raw_target)
                ),
            }
        )
        return projected, metadata

    @staticmethod
    def _sg_navmesh_points_connected(pathfinder, start_point, end_point):
        """Return navmesh connectivity without retaining or using its route."""

        try:
            import habitat_sim

            query = habitat_sim.ShortestPath()
            query.requested_start = np.asarray(
                start_point, dtype=np.float32
            )
            query.requested_end = np.asarray(end_point, dtype=np.float32)
            return bool(pathfinder.find_path(query))
        except Exception:
            return None

    def _sg_world_navmesh_target_validator(
        self,
        window,
        current_world_xz,
        terminal_snap_tolerance_m=0.0,
    ):
        """Validate terminal M0 cells against the current navmesh island.

        The returned predicate is intentionally used only by
        ``_sg_reachable_target``.  Theta* still sees the persisted world map and
        acquisition certificate, never a navmesh route or navmesh occupancy
        raster.
        """

        metadata = {
            "status": "disabled",
            "candidate_cells_checked": 0,
            "rejected_not_navigable": 0,
            "rejected_other_island": 0,
            "accepted_bounded_snap": 0,
            "rejected_snap_too_far": 0,
        }
        try:
            terminal_snap_tolerance_m = max(
                0.0, float(terminal_snap_tolerance_m)
            )
        except (TypeError, ValueError):
            terminal_snap_tolerance_m = 0.0
        metadata["terminal_snap_tolerance_m"] = float(
            terminal_snap_tolerance_m
        )
        if not bool(
            getattr(self, "sg_navmesh_target_validation", True)
        ):
            return None, metadata
        context = self._sg_live_navmesh_context()
        if context is None:
            metadata["status"] = "live_navmesh_unavailable"
            return None, metadata
        metric_map = self._sg_world_metric_map()
        if metric_map is None:
            metadata["status"] = "world_metric_map_unavailable"
            return None, metadata
        pathfinder, floor_y = context
        current = np.asarray(current_world_xz, dtype=np.float64).reshape(-1)
        if current.size != 2 or not np.isfinite(current).all():
            metadata["status"] = "invalid_current_world_pose"
            return None, metadata
        current_point = np.asarray(
            [float(current[0]), float(floor_y), float(current[1])],
            dtype=np.float32,
        )
        try:
            if not bool(pathfinder.is_navigable(current_point)):
                metadata["status"] = "current_pose_not_navigable"
                return None, metadata
            has_island_api = hasattr(pathfinder, "get_island")
            current_island = (
                int(pathfinder.get_island(current_point))
                if has_island_api
                else None
            )
        except Exception as exc:
            metadata["status"] = "current_island_query_failed"
            metadata["detail"] = str(exc)
            return None, metadata

        spec = metric_map.spec
        metadata["status"] = "applied"
        metadata["current_island"] = (
            None if current_island is None else int(current_island)
        )
        metadata["connectivity_mode"] = (
            "island_id" if has_island_api else "shortest_path_boolean"
        )

        def validate(local_cell):
            metadata["candidate_cells_checked"] += 1
            global_cell = tuple(
                map(int, window.local_to_global(*local_cell))
            )
            world_x, world_z = spec.cell_center_world(*global_cell)
            point = np.asarray(
                [float(world_x), float(floor_y), float(world_z)],
                dtype=np.float32,
            )
            try:
                validation_point = point
                if not bool(pathfinder.is_navigable(point)):
                    if terminal_snap_tolerance_m <= 0.0:
                        metadata["rejected_not_navigable"] += 1
                        return False
                    snapped = np.asarray(
                        pathfinder.snap_point(point, current_island)
                        if has_island_api
                        else pathfinder.snap_point(point),
                        dtype=np.float64,
                    ).reshape(-1)
                    if (
                        snapped.size < 3
                        or not np.isfinite(snapped[:3]).all()
                        or not bool(pathfinder.is_navigable(snapped[:3]))
                    ):
                        metadata["rejected_not_navigable"] += 1
                        return False
                    snap_distance_m = float(
                        np.linalg.norm(snapped[[0, 2]] - point[[0, 2]])
                    )
                    if snap_distance_m > terminal_snap_tolerance_m + 1e-6:
                        metadata["rejected_snap_too_far"] += 1
                        return False
                    validation_point = snapped[:3]
                if has_island_api:
                    if (
                        int(pathfinder.get_island(validation_point))
                        != current_island
                    ):
                        metadata["rejected_other_island"] += 1
                        return False
                else:
                    connected = self._sg_navmesh_points_connected(
                        pathfinder,
                        current_point,
                        validation_point,
                    )
                    if connected is not True:
                        metadata["rejected_other_island"] += 1
                        return False
                if validation_point is not point:
                    metadata["accepted_bounded_snap"] += 1
                    metadata["selected_bounded_snap_distance_m"] = float(
                        snap_distance_m
                    )
                    metadata["selected_bounded_snap_world_xyz"] = list(
                        map(float, validation_point)
                    )
            except Exception:
                metadata["rejected_not_navigable"] += 1
                return False
            return True

        return validate, metadata

    def _sg_navmesh_context(self):
        """Return navmesh context for live one-step action prediction.

        This evaluator uses ThreadedVectorEnv, so ``call_at(..., '_env')``
        returns the live environment object in-process.  The context is
        acquired once per controller/planner call and reused for all candidate
        actions; ``PathFinder.try_step`` is read-only and applies the same
        navmesh sliding used by Habitat's discrete MOVE_FORWARD action.
        """

        if not bool(
            getattr(self, "sg_navmesh_action_prediction", False)
        ):
            return None
        return self._sg_live_navmesh_context()

    @staticmethod
    def _sg_predict_forward_endpoint(
        position,
        heading,
        step_cells,
        resolution_m,
        navmesh_context=None,
    ):
        """Predict Habitat's slid endpoint for one discrete forward action."""

        position = np.asarray(position, dtype=np.float64).reshape(2)
        nominal = position + float(step_cells) * np.asarray(
            [-np.sin(float(heading)), -np.cos(float(heading))],
            dtype=np.float64,
        )
        if navmesh_context is None:
            return nominal, False
        try:
            pathfinder, floor_y = navmesh_context
            start_world = np.asarray(
                [
                    (position[0] + 0.5) * float(resolution_m),
                    float(floor_y),
                    (position[1] + 0.5) * float(resolution_m),
                ],
                dtype=np.float32,
            )
            nominal_world = np.asarray(
                [
                    (nominal[0] + 0.5) * float(resolution_m),
                    float(floor_y),
                    (nominal[1] + 0.5) * float(resolution_m),
                ],
                dtype=np.float32,
            )
            predicted_world = np.asarray(
                pathfinder.try_step(start_world, nominal_world),
                dtype=np.float64,
            ).reshape(-1)
            if predicted_world.size < 3 or not np.isfinite(
                predicted_world[:3]
            ).all():
                return nominal, False
            predicted = predicted_world[[0, 2]] / float(resolution_m) - 0.5
            return predicted, True
        except Exception:
            return nominal, False

    @staticmethod
    def _sg_local_lattice_key(position, heading_index):
        """Key continuous action endpoints without coarsening replay poses.

        The old quarter-cell key merged poses up to 1.25 cm apart.  Because a
        merged state's parent can be replaced independently of its children,
        the reconstructed action sequence could then differ from the poses
        whose segments passed the clearance check.  A mill-cell key keeps
        numerically equivalent lattice poses merged while preserving the exact
        30-degree/0.25-m action geometry needed for safe replay.
        """

        return (
            int(round(float(position[0]) * 1000.0)),
            int(round(float(position[1]) * 1000.0)),
            int(heading_index),
        )

    def _sg_plan_local_action_sequence(
        self,
        position,
        current_heading,
        target,
        window,
        safe_mask,
        target_info,
    ):
        """A* directly in Habitat's TURN/TURN/FORWARD action lattice.

        Theta* is retained as the long-range planner.  This bounded local
        search is the contract between continuous world geometry and the
        simulator's fixed 30-degree/0.25-m action space.  It may temporarily
        move away from the target; immediate reversal is a cost, not a hard
        impossibility constraint.
        """

        metric_map = self._sg_world_metric_map()
        if metric_map is None or window is None or safe_mask is None:
            return None
        resolution_m = float(metric_map.spec.resolution_m)
        step_cells = (
            float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
            / resolution_m
        )
        turn_rad = np.deg2rad(
            float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        )
        headings = self._sg_discrete_heading_lattice(current_heading, turn_rad)
        if not headings or step_cells <= 0:
            return None
        navmesh_context = self._sg_navmesh_context()

        start_position = np.asarray(position, dtype=np.float64).reshape(2)
        target_position = np.asarray(target, dtype=np.float64).reshape(2)
        start_distance = float(np.linalg.norm(start_position - target_position))
        radius_cells = max(
            step_cells * 2.0,
            float(getattr(self, "sg_local_action_radius_m", 2.5)) / resolution_m,
        )
        if start_distance > radius_cells + step_cells:
            target_info["_sg_local_action_status"] = "target_outside_local_radius"
            return None
        tolerance_cells = max(
            0.5,
            float(getattr(self, "sg_local_action_goal_tolerance_m", 0.15))
            / resolution_m,
        )
        target_info["_sg_local_action_goal_tolerance_cells"] = float(
            tolerance_cells
        )

        num_headings = len(headings)
        start_key = self._sg_local_lattice_key(start_position, 0)
        positions = {start_key: start_position}
        parents = {start_key: None}
        parent_actions = {}
        depths = {start_key: 0}
        g_score = {start_key: 0.0}
        tie = 0

        def heuristic(candidate):
            distance = float(np.linalg.norm(candidate - target_position))
            return max(0.0, distance - tolerance_cells) / step_cells

        open_heap = [(heuristic(start_position), 0.0, tie, start_key)]
        expanded = 0
        max_states = max(
            100,
            int(getattr(self, "sg_local_action_max_states", 12000)),
        )
        max_actions = int(
            os.environ.get("LIFELONG_SG_LOCAL_ACTION_MAX_ACTIONS", "96")
        )
        previous_forward_origin = target_info.get(
            "_sg_last_forward_world_grid_origin"
        )
        certified_route_path = None
        certified_route_cumulative = None
        certified_route_progress = float(
            target_info.get("_sg_path_progress_cells", 0.0)
        )
        if str(target_info.get("execution_path_source", "")) == (
            "certified_trajectory_then_world_theta_star"
        ):
            active_path = target_info.get("_sg_world_path_cells") or []
            suffix_start = target_info.get(
                "_sg_trajectory_suffix_start_progress_cells"
            )
            if len(active_path) >= 2 and (
                suffix_start is None
                or certified_route_progress < float(suffix_start) - 1e-6
            ):
                certified_route_path = active_path
                certified_route_cumulative = self._sg_path_cumulative(
                    active_path
                )
        goal_key = None

        while open_heap and expanded < max_states:
            _, queued_g, _, state_key = heapq.heappop(open_heap)
            if queued_g != g_score.get(state_key):
                continue
            state_position = positions[state_key]
            heading_index = int(state_key[2])
            depth = int(depths[state_key])
            expanded += 1
            if float(np.linalg.norm(state_position - target_position)) <= tolerance_cells:
                goal_key = state_key
                break
            if depth >= max_actions:
                continue

            successors = (
                (
                    int(HabitatSimActions.TURN_LEFT),
                    state_position,
                    (heading_index + 1) % num_headings,
                    0.20,
                ),
                (
                    int(HabitatSimActions.TURN_RIGHT),
                    state_position,
                    (heading_index - 1) % num_headings,
                    0.20,
                ),
            )
            for action, successor_position, successor_heading, action_cost in successors:
                successor_key = self._sg_local_lattice_key(
                    successor_position, successor_heading
                )
                candidate_g = queued_g + action_cost
                if candidate_g + 1e-9 >= g_score.get(successor_key, float("inf")):
                    continue
                g_score[successor_key] = candidate_g
                positions[successor_key] = successor_position
                parents[successor_key] = state_key
                parent_actions[successor_key] = action
                depths[successor_key] = depth + 1
                tie += 1
                heapq.heappush(
                    open_heap,
                    (
                        candidate_g + heuristic(successor_position),
                        candidate_g,
                        tie,
                        successor_key,
                    ),
                )

            forward_heading = headings[heading_index]
            forward_position, _ = self._sg_predict_forward_endpoint(
                state_position,
                forward_heading,
                step_cells,
                resolution_m,
                navmesh_context,
            )
            if (
                float(np.linalg.norm(forward_position - state_position))
                < step_cells * 0.5
            ):
                continue
            if float(np.linalg.norm(forward_position - start_position)) > radius_cells:
                continue
            if not self._sg_float_segment_is_safe(
                safe_mask,
                self._sg_window_point(window, state_position),
                self._sg_window_point(window, forward_position),
            ):
                continue
            # A recovery for a certified acquisition route must remain close
            # enough to rejoin that same route.  An unconstrained local A*
            # could otherwise find a 40--80 action loop through generic M0
            # free space, then leave the robot too far from every remaining
            # trajectory node and silently downgrade the next replan to plain
            # Theta* (829904/829951 ep1).
            if certified_route_path is not None:
                _, route_cross_track = self._sg_project_position_to_path(
                    certified_route_path,
                    forward_position,
                    certified_route_progress,
                    cumulative=certified_route_cumulative,
                )
                maximum_route_cross_track_cells = max(
                    step_cells,
                    float(getattr(self, "sg_path_deviation_m", 0.35))
                    / resolution_m,
                )
                if (
                    not np.isfinite(route_cross_track)
                    or float(route_cross_track)
                    > maximum_route_cross_track_cells + 1e-6
                ):
                    continue
            current_distance = float(
                np.linalg.norm(state_position - target_position)
            )
            forward_distance = float(
                np.linalg.norm(forward_position - target_position)
            )
            temporary_retreat = max(0.0, forward_distance - current_distance)
            reverse_penalty = 0.0
            if previous_forward_origin is not None:
                reverse_distance = float(
                    np.linalg.norm(
                        forward_position
                        - np.asarray(previous_forward_origin, dtype=np.float64)
                    )
                )
                if reverse_distance <= max(0.5, step_cells * 0.25):
                    reverse_penalty = 1.5
            forward_cost = (
                1.0
                + 0.10 * temporary_retreat / max(step_cells, 1e-6)
                + reverse_penalty
            )
            forward_key = self._sg_local_lattice_key(
                forward_position, heading_index
            )
            candidate_g = queued_g + forward_cost
            if candidate_g + 1e-9 < g_score.get(forward_key, float("inf")):
                g_score[forward_key] = candidate_g
                positions[forward_key] = forward_position
                parents[forward_key] = state_key
                parent_actions[forward_key] = int(
                    HabitatSimActions.MOVE_FORWARD
                )
                depths[forward_key] = depth + 1
                tie += 1
                heapq.heappush(
                    open_heap,
                    (
                        candidate_g + heuristic(forward_position),
                        candidate_g,
                        tie,
                        forward_key,
                    ),
                )

        if goal_key is None:
            target_info["_sg_local_action_status"] = (
                "unreachable_within_bounded_action_lattice"
            )
            target_info["_sg_local_action_expanded_states"] = int(expanded)
            return None

        actions = []
        cursor = goal_key
        while parents[cursor] is not None:
            actions.append(int(parent_actions[cursor]))
            cursor = parents[cursor]
        actions.reverse()

        # The queue is executed later from the simulator's continuous pose,
        # not from hashed search-state representatives.  Re-simulate the
        # reconstructed actions from the exact start and require every forward
        # segment to satisfy the same safety contract.  This makes a returned
        # plan replay-safe by construction even if future state-key changes
        # accidentally introduce an alias.
        replay_position = np.asarray(
            start_position, dtype=np.float64
        ).copy()
        replay_heading_index = 0
        for action_index, action in enumerate(actions):
            if action == int(HabitatSimActions.TURN_LEFT):
                replay_heading_index = (
                    replay_heading_index + 1
                ) % num_headings
                continue
            if action == int(HabitatSimActions.TURN_RIGHT):
                replay_heading_index = (
                    replay_heading_index - 1
                ) % num_headings
                continue
            if action != int(HabitatSimActions.MOVE_FORWARD):
                target_info["_sg_local_action_status"] = (
                    "invalid_action_in_reconstructed_plan"
                )
                return None
            replay_heading = headings[replay_heading_index]
            replay_endpoint, _ = self._sg_predict_forward_endpoint(
                replay_position,
                replay_heading,
                step_cells,
                resolution_m,
                navmesh_context,
            )
            if (
                float(np.linalg.norm(replay_endpoint - replay_position))
                < step_cells * 0.5
            ):
                target_info["_sg_local_action_status"] = (
                    "reconstructed_plan_failed_navmesh_displacement_replay"
                )
                target_info["_sg_local_action_replay_failure_index"] = int(
                    action_index
                )
                return None
            if not self._sg_float_segment_is_safe(
                safe_mask,
                self._sg_window_point(window, replay_position),
                self._sg_window_point(window, replay_endpoint),
            ):
                target_info["_sg_local_action_status"] = (
                    "reconstructed_plan_failed_exact_safety_replay"
                )
                target_info["_sg_local_action_replay_failure_index"] = int(
                    action_index
                )
                return None
            replay_position = replay_endpoint
        replay_goal_distance = float(
            np.linalg.norm(replay_position - target_position)
        )
        if replay_goal_distance > tolerance_cells + 1e-6:
            target_info["_sg_local_action_status"] = (
                "reconstructed_plan_missed_goal_on_exact_replay"
            )
            target_info["_sg_local_action_replay_goal_distance_cells"] = (
                replay_goal_distance
            )
            return None

        target_info["_sg_local_action_status"] = "planned"
        target_info["_sg_local_action_expanded_states"] = int(expanded)
        target_info["_sg_local_action_plan_length"] = len(actions)
        return {
            "actions": actions,
            "expanded_states": int(expanded),
            "goal_tolerance_cells": float(tolerance_cells),
            "goal_distance_cells": replay_goal_distance,
        }

    @staticmethod
    def _sg_discrete_candidate_advances(
        progress_gain,
        goal_gain,
        candidate_cross_track,
        minimum_advance_cells,
        maximum_progress_cross_track_cells,
    ):
        """Decide whether one executable forward endpoint makes progress.

        Projection onto a polyline is only meaningful near that polyline.  A
        candidate far from the active path can otherwise project onto a later
        segment and appear to gain cumulative progress while physically moving
        away from both the lookahead and final goal (829904 ep1).  Goal-distance
        progress remains valid regardless of cross-track distance; path-only
        progress is accepted only inside the controller's deviation corridor.
        """

        return bool(
            float(goal_gain) >= float(minimum_advance_cells)
            or (
                float(progress_gain) >= float(minimum_advance_cells)
                and float(candidate_cross_track)
                <= float(maximum_progress_cross_track_cells) + 1e-6
            )
        )

    def _sg_local_action_recovery_targets(
        self,
        target_info,
        position,
        final_target,
    ):
        """Return terminal or path-local goals for bounded action recovery.

        The action-lattice planner is deliberately local.  When the terminal
        object target is farther away than its search radius, recover against
        progressively nearer points on the already certified world path.  On
        the next evaluator cycle the long-range Theta* controller resumes from
        the attained subgoal.
        """

        metric_map = self._sg_world_metric_map()
        if metric_map is None:
            return []
        resolution_m = float(metric_map.spec.resolution_m)
        step_cells = (
            float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
            / resolution_m
        )
        radius_cells = max(
            step_cells * 2.0,
            float(getattr(self, "sg_local_action_radius_m", 2.5))
            / resolution_m,
        )
        position = np.asarray(position, dtype=np.float64).reshape(2)
        final_target = np.asarray(final_target, dtype=np.float64).reshape(2)
        final_distance = float(np.linalg.norm(position - final_target))
        if final_distance <= radius_cells + step_cells:
            return [
                {
                    "target": final_target,
                    "kind": "final_target",
                    "path_progress_cells": None,
                }
            ]

        path = target_info.get("_sg_world_path_cells") or []
        if len(path) < 2:
            target_info["_sg_local_action_status"] = (
                "missing_world_path_for_local_subgoal"
            )
            return []
        cumulative = self._sg_path_cumulative(path)
        reported_progress = float(
            target_info.get("_sg_path_progress_cells", 0.0)
        )
        projected_progress, projected_cross_track = (
            self._sg_project_position_to_path(
                path,
                position,
                reported_progress,
                cumulative=cumulative,
            )
        )
        maximum_cross_track_cells = max(
            step_cells,
            float(getattr(self, "sg_path_deviation_m", 0.35))
            / resolution_m,
        )
        if (
            projected_progress is not None
            and float(projected_cross_track) <= maximum_cross_track_cells + 1e-6
        ):
            path_progress = max(
                reported_progress, float(projected_progress)
            )
        else:
            path_progress = reported_progress
        remaining_cells = max(0.0, float(cumulative[-1]) - path_progress)
        if remaining_cells <= 1e-6:
            target_info["_sg_local_action_status"] = (
                "world_path_exhausted_before_final_target"
            )
            return []

        lookahead_cells = max(
            step_cells,
            float(getattr(self, "sg_path_lookahead_m", 1.0))
            / resolution_m,
        )
        # Leave at least two forward steps of radial margin for a discrete
        # detour around the obstacle that triggered recovery.
        maximum_advance = max(
            step_cells,
            radius_cells - 2.0 * step_cells,
        )
        nominal_advance = min(
            remaining_cells,
            lookahead_cells,
            maximum_advance,
        )
        advances = [
            nominal_advance,
            nominal_advance * 0.75,
            nominal_advance * 0.50,
            step_cells,
        ]
        tolerance_cells = max(
            0.5,
            float(getattr(self, "sg_local_action_goal_tolerance_m", 0.15))
            / resolution_m,
        )
        candidates = []
        seen = set()
        for advance in advances:
            advance = min(remaining_cells, max(step_cells, float(advance)))
            candidate_progress = min(
                float(cumulative[-1]), path_progress + advance
            )
            candidate = self._sg_point_on_path(
                path, cumulative, candidate_progress
            )
            if candidate is None:
                continue
            candidate = np.asarray(candidate, dtype=np.float64).reshape(2)
            candidate_distance = float(np.linalg.norm(candidate - position))
            if (
                candidate_distance <= tolerance_cells + 1e-6
                or candidate_distance > radius_cells + step_cells + 1e-6
            ):
                continue
            key = tuple(np.round(candidate, decimals=4))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "target": candidate,
                    "kind": "world_path_subgoal",
                    "path_progress_cells": float(candidate_progress),
                }
            )
        if not candidates:
            target_info["_sg_local_action_status"] = (
                "no_bounded_world_path_subgoal"
            )
        return candidates

    def _sg_take_local_action(
        self, target_info, position, current_heading, window, safe_mask
    ):
        queue = target_info.get("_sg_local_action_queue")
        if not queue:
            return None
        action = int(queue[0])
        metric_map = self._sg_world_metric_map()
        if metric_map is None:
            target_info.pop("_sg_local_action_queue", None)
            return None
        if action == int(HabitatSimActions.MOVE_FORWARD):
            resolution_m = float(metric_map.spec.resolution_m)
            step_cells = (
                float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
                / resolution_m
            )
            endpoint, navmesh_predicted = self._sg_predict_forward_endpoint(
                position,
                current_heading,
                step_cells,
                resolution_m,
                self._sg_navmesh_context(),
            )
            actual_step_cells = float(
                np.linalg.norm(
                    endpoint - np.asarray(position, dtype=np.float64)
                )
            )
            if navmesh_predicted and actual_step_cells < step_cells * 0.5:
                target_info.pop("_sg_local_action_queue", None)
                target_info["_sg_local_action_status"] = (
                    "queued_forward_navmesh_displacement_too_small"
                )
                return None
            if not self._sg_float_segment_is_safe(
                safe_mask,
                self._sg_window_point(window, position),
                self._sg_window_point(window, endpoint),
            ):
                target_info.pop("_sg_local_action_queue", None)
                target_info["_sg_local_action_status"] = "queued_forward_became_unsafe"
                logger.warning(
                    "[INHerit-SG-V7] discarded local action queue because "
                    "the next exact forward segment became unsafe: "
                    f"position={tuple(map(float, position))}, "
                    f"heading={float(current_heading):.6f}, "
                    f"endpoint={tuple(map(float, endpoint))}, "
                    "goal_kind="
                    f"{target_info.get('_sg_local_action_goal_kind')}, "
                    "goal_world_grid="
                    f"{target_info.get('_sg_local_action_goal_world_grid')}"
                )
                return None
            target_info["_sg_pending_forward_world_grid_position"] = tuple(
                map(float, position)
            )
            target_info["_sg_pending_forward_world_grid_endpoint"] = tuple(
                map(float, endpoint)
            )
        queue.pop(0)
        if not queue:
            target_info.pop("_sg_local_action_queue", None)
        logger.info(
            "[INHerit-SG-V7] local action-lattice action="
            f"{action}, remaining_actions={len(queue)}"
        )
        return action

    def _sg_start_local_action_recovery(
        self,
        target_info,
        position,
        current_heading,
        target,
        window,
        safe_mask,
        reason,
    ):
        recovery_targets = self._sg_local_action_recovery_targets(
            target_info,
            position,
            target,
        )
        plan = None
        selected_recovery_target = None
        for recovery_target in recovery_targets:
            plan = self._sg_plan_local_action_sequence(
                position,
                current_heading,
                recovery_target["target"],
                window,
                safe_mask,
                target_info,
            )
            if plan is not None:
                selected_recovery_target = recovery_target
                break
            logger.info(
                "[INHerit-SG-V7] bounded local recovery target rejected: "
                f"reason={reason}, kind={recovery_target['kind']}, "
                "goal_world_grid="
                f"{tuple(map(float, recovery_target['target']))}, "
                f"status={target_info.get('_sg_local_action_status')}"
            )
        if plan is None:
            target_info["_sg_consecutive_failed_recoveries"] = int(
                target_info.get("_sg_consecutive_failed_recoveries", 0)
            ) + 1
            return None
        target_info["_sg_local_action_goal_kind"] = selected_recovery_target[
            "kind"
        ]
        target_info["_sg_local_action_goal_world_grid"] = list(
            map(float, selected_recovery_target["target"])
        )
        path_progress = selected_recovery_target.get("path_progress_cells")
        if path_progress is None:
            target_info.pop(
                "_sg_local_action_goal_path_progress_cells", None
            )
        else:
            target_info["_sg_local_action_goal_path_progress_cells"] = float(
                path_progress
            )
        target_info["_sg_consecutive_failed_recoveries"] = 0
        target_info["_sg_consecutive_immediate_reverse_replans"] = 0
        target_info.pop("_sg_force_local_action_recovery", None)
        target_info["_sg_local_action_recovery_active"] = True
        # Execute local recovery as receding-horizon control.  Habitat can
        # slide/snap a nominal 0.25-m move on the navmesh, especially beside
        # furniture.  A long open-loop queue then diverges from the continuous
        # poses whose segments were certified by A*.  Keep the subgoal, but
        # execute only the first action and replan from the measured simulator
        # pose on the next evaluator cycle.
        target_info["_sg_local_action_queue"] = list(plan["actions"][:1])
        target_info["_sg_local_action_receding_plan_length"] = len(
            plan["actions"]
        )
        logger.info(
            "[INHerit-SG-V7] planned bounded local action recovery: "
            f"reason={reason}, actions={len(plan['actions'])}, "
            f"expanded={plan['expanded_states']}, "
            f"goal_kind={selected_recovery_target['kind']}, "
            "goal_world_grid="
            f"{tuple(map(float, selected_recovery_target['target']))}, "
            f"goal_distance_cells={plan['goal_distance_cells']:.2f}"
        )
        action = self._sg_take_local_action(
            target_info,
            position,
            current_heading,
            window,
            safe_mask,
        )
        if action is not None:
            return action
        # The current pose is already inside the attainable terminal set.  A
        # harmless turn yields one evaluator cycle so reached_reason can close
        # SG navigation through the normal reached/STOP path.
        return int(HabitatSimActions.TURN_LEFT)

    def _get_sg_world_path_action(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        current_world_xz = self._sg_current_world_xz(kwargs)
        if current_world_xz is None:
            return None
        metric_map = self._sg_world_metric_map()
        if metric_map is None:
            return None
        resolution_m = float(metric_map.spec.resolution_m)
        position = np.asarray(current_world_xz, dtype=np.float64) / resolution_m - 0.5
        target = (
            int(target_info["target_world_grid_x"]),
            int(target_info["target_world_grid_z"]),
        )
        path = target_info.get("_sg_world_path_cells") or []
        window = getattr(self, "_sg_world_plan_window", None)
        safe_mask = getattr(self, "_sg_world_safe_mask", None)

        previous_progress = float(target_info.get("_sg_path_progress_cells", 0.0))
        need_replan = len(path) < 2 or window is None or safe_mask is None
        reason = "missing_world_path"
        pending_forward_pose = target_info.pop(
            "_sg_pending_forward_world_grid_position", None
        )
        pending_forward_endpoint = target_info.pop(
            "_sg_pending_forward_world_grid_endpoint", None
        )
        if pending_forward_pose is not None:
            actual_displacement = float(
                np.linalg.norm(
                    position - np.asarray(pending_forward_pose, dtype=np.float64)
                )
            )
            expected_step = (
                float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
                / resolution_m
            )
            if actual_displacement < expected_step * 0.5:
                need_replan = True
                reason = f"collision_displacement_{actual_displacement:.2f}_world_cells"
                target_info.pop("_sg_local_action_queue", None)
                collisions_without_progress = int(
                    target_info.get("_sg_collision_replans_without_progress", 0)
                ) + 1
                target_info["_sg_collision_replans_without_progress"] = (
                    collisions_without_progress
                )
                if pending_forward_endpoint is not None:
                    self._sg_register_world_collision_block(
                        pending_forward_endpoint,
                        actual_position_grid=position,
                    )
                # max_refinements now bounds consecutive *executed* collision
                # recoveries.  The Nth successful replan is allowed to run; an
                # (N+1)th collision without intervening progress is terminal.
                if collisions_without_progress > int(self.sg_max_refinements):
                    target_info["_sg_pending_abort_reason"] = (
                        "repeated_collision_without_progress_"
                        f"{collisions_without_progress}"
                    )
                    return None
            else:
                target_info["_sg_last_forward_world_grid_origin"] = tuple(
                    map(float, pending_forward_pose)
                )

        # A local action-lattice plan deliberately leaves the continuous
        # Theta* path while it executes a sequence that is attainable under
        # Habitat's fixed turn/forward action set.  Cross-track replanning is
        # therefore invalid during that sequence: it used to clear the queue
        # after only a few actions and deterministically restart the same
        # recovery loop until the episode step budget was exhausted.  Pending
        # forward collision feedback above still has priority and invalidates
        # the queue before we reach this point.
        local_recovery_active = bool(
            target_info.get("_sg_local_action_recovery_active")
        )
        if not need_replan and local_recovery_active:
            local_goal = target_info.get("_sg_local_action_goal_world_grid")
            tolerance_cells = float(
                target_info.get(
                    "_sg_local_action_goal_tolerance_cells",
                    max(
                        0.5,
                        float(
                            getattr(
                                self,
                                "sg_local_action_goal_tolerance_m",
                                0.15,
                            )
                        )
                        / resolution_m,
                    ),
                )
            )
            local_goal_reached = False
            if local_goal is not None:
                local_goal_reached = bool(
                    np.linalg.norm(
                        position
                        - np.asarray(local_goal, dtype=np.float64)
                    )
                    <= tolerance_cells + 1e-6
                )
            if local_goal_reached:
                logger.info(
                    "[INHerit-SG-V7] reached bounded local recovery goal; "
                    "resuming world Theta* control: goal_kind="
                    f"{target_info.get('_sg_local_action_goal_kind')}, "
                    f"goal_world_grid={local_goal}"
                )
                target_info.pop("_sg_local_action_queue", None)
                target_info.pop("_sg_local_action_recovery_active", None)
                target_info.pop("_sg_local_action_goal_kind", None)
                target_info.pop("_sg_local_action_goal_world_grid", None)
                target_info.pop(
                    "_sg_local_action_goal_path_progress_cells", None
                )
                target_info.pop(
                    "_sg_local_action_goal_tolerance_cells", None
                )
                self._sg_reset_world_path_stall_guard(target_info)
                local_recovery_active = False
            elif local_goal is not None:
                rolling_plan = self._sg_plan_local_action_sequence(
                    position,
                    float(
                        np.asarray(
                            kwargs.get(
                                "current_world_heading",
                                getattr(
                                    self, "_sg_world_heading_rad", 0.0
                                ),
                            )
                        ).reshape(-1)[0]
                    ),
                    np.asarray(local_goal, dtype=np.float64),
                    window,
                    safe_mask,
                    target_info,
                )
                if rolling_plan is not None and rolling_plan["actions"]:
                    target_info["_sg_local_action_queue"] = [
                        int(rolling_plan["actions"][0])
                    ]
                    target_info[
                        "_sg_local_action_receding_plan_length"
                    ] = len(rolling_plan["actions"])
                    queued_action = self._sg_take_local_action(
                        target_info,
                        position,
                        float(
                            np.asarray(
                                kwargs.get(
                                    "current_world_heading",
                                    getattr(
                                        self,
                                        "_sg_world_heading_rad",
                                        0.0,
                                    ),
                                )
                            ).reshape(-1)[0]
                        ),
                        window,
                        safe_mask,
                    )
                    if queued_action is not None:
                        logger.info(
                            "[INHerit-SG-V7] receding-horizon local "
                            f"recovery replanned {len(rolling_plan['actions'])} "
                            "actions from measured pose"
                        )
                        return int(queued_action)
                logger.info(
                    "[INHerit-SG-V7] receding-horizon local recovery "
                    "could not continue from measured pose; replanning "
                    "the world path: status="
                    f"{target_info.get('_sg_local_action_status')}"
                )
                target_info.pop("_sg_local_action_queue", None)
                target_info.pop("_sg_local_action_recovery_active", None)
                target_info.pop("_sg_local_action_goal_kind", None)
                target_info.pop("_sg_local_action_goal_world_grid", None)
                target_info.pop(
                    "_sg_local_action_goal_path_progress_cells", None
                )
                target_info.pop(
                    "_sg_local_action_goal_tolerance_cells", None
                )
                need_replan = True
                reason = "local_action_lattice_unreachable_receding_horizon"

        if (
            not need_replan
            and not local_recovery_active
            and target_info.get("_sg_local_action_queue")
        ):
            queued_action = self._sg_take_local_action(
                target_info,
                position,
                float(np.asarray(
                    kwargs.get(
                        "current_world_heading",
                        getattr(self, "_sg_world_heading_rad", 0.0),
                    )
                ).reshape(-1)[0]),
                window,
                safe_mask,
            )
            if queued_action is not None:
                return int(queued_action)

        if not need_replan:
            progress, cross_track, lookahead, total = (
                self._sg_world_path_progress_and_lookahead(
                    path, position, previous_progress, window, safe_mask
                )
            )
            deviation_m = cross_track * resolution_m
            if deviation_m > self.sg_path_deviation_m or lookahead is None:
                need_replan = True
                reason = f"cross_track_{deviation_m:.3f}m"
        if need_replan:
            replanned = self._sg_replan_world_path(
                target_info, current_world_xz, reason
            )
            if replanned is None or len(replanned["path"]) < 2:
                planner_reason = (
                    getattr(self, "_sg_last_theta_failure", None) or {}
                ).get("reason", "world_theta_replan_failed")
                target_info["_sg_pending_abort_reason"] = (
                    f"{reason}_after_runtime_collision_overlay_{planner_reason}"
                    if reason.startswith("collision_displacement_")
                    else f"{reason}_{planner_reason}"
                )
                return None
            path = target_info["_sg_world_path_cells"]
            target = (
                int(target_info["target_world_grid_x"]),
                int(target_info["target_world_grid_z"]),
            )
            window = replanned["window"]
            safe_mask = replanned["safe_mask"]
            progress, cross_track, lookahead, total = (
                self._sg_world_path_progress_and_lookahead(
                    path, position, 0.0, window, safe_mask
                )
            )

        target_info["_sg_path_progress_cells"] = float(progress)
        target_info["_sg_path_cross_track_m"] = float(cross_track * resolution_m)
        target_info["_sg_path_remaining_m"] = float(
            (total - progress) * resolution_m
        )
        delta = np.asarray(lookahead, dtype=np.float64) - position
        desired_heading = float(np.arctan2(-delta[0], -delta[1]))
        supplied_heading = kwargs.get("current_world_heading")
        if supplied_heading is None:
            supplied_heading = getattr(self, "_sg_world_heading_rad", None)
        if supplied_heading is None:
            target_info["_sg_pending_abort_reason"] = "missing_current_world_heading"
            return None
        current_heading = float(np.asarray(supplied_heading).reshape(-1)[0])
        heading_error = float(
            np.arctan2(
                np.sin(desired_heading - current_heading),
                np.cos(desired_heading - current_heading),
            )
        )
        turn_rad = np.deg2rad(
            float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        )
        heading_lattice = self._sg_discrete_heading_lattice(
            current_heading, turn_rad
        )
        num_headings = len(heading_lattice)
        if num_headings == 0:
            target_info["_sg_pending_abort_reason"] = "invalid_discrete_heading_lattice"
            return None
        step_cells = (
            float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
            / resolution_m
        )
        queued_action = self._sg_take_local_action(
            target_info,
            position,
            current_heading,
            window,
            safe_mask,
        )
        if queued_action is not None:
            return int(queued_action)
        alternatives = []
        certified_probe_alternatives = []
        trajectory_probe_mask = getattr(
            self, "_sg_world_trajectory_probe_mask", None
        )
        current_goal_distance = float(
            np.hypot(position[0] - target[0], position[1] - target[1])
        )
        minimum_advance_cells = max(
            0.25, self.sg_nav_stall_eps_m / resolution_m
        )
        maximum_progress_cross_track_cells = max(
            step_cells,
            self.sg_path_deviation_m / resolution_m,
        )
        previous_forward_origin = target_info.get(
            "_sg_last_forward_world_grid_origin"
        )
        position_local = self._sg_window_point(window, position)
        navmesh_context = self._sg_navmesh_context()
        # Habitat TURN_LEFT/TURN_RIGHT preserve the episode's initial heading
        # phase.  If an episode starts at -15 degrees with a 30-degree turn,
        # the reachable headings are -15 + k*30 degrees, not k*30 degrees.
        # Anchor every simulated forward action to the measured current heading
        # so the segment checked against safe_mask is the segment Habitat will
        # actually execute.
        for turn_offset, candidate_heading in enumerate(heading_lattice):
            candidate_end, navmesh_predicted = (
                self._sg_predict_forward_endpoint(
                    position,
                    candidate_heading,
                    step_cells,
                    resolution_m,
                    navmesh_context,
                )
            )
            if (
                navmesh_predicted
                and float(np.linalg.norm(candidate_end - position))
                < step_cells * 0.5
            ):
                continue
            candidate_local = self._sg_window_point(window, candidate_end)
            clearance_safe = self._sg_float_segment_is_safe(
                safe_mask, position_local, candidate_local
            )
            trajectory_probe_aligned = bool(
                not clearance_safe
                and navmesh_predicted
                and trajectory_probe_mask is not None
                and self._sg_float_segment_is_safe(
                    trajectory_probe_mask,
                    position_local,
                    candidate_local,
                )
            )
            if not clearance_safe and not trajectory_probe_aligned:
                continue
            counterclockwise = turn_offset
            clockwise = (-turn_offset) % num_headings
            turn_steps = min(clockwise, counterclockwise)
            endpoint_error = float(np.linalg.norm(candidate_end - lookahead))
            heading_alignment = abs(
                float(
                    np.arctan2(
                        np.sin(desired_heading - candidate_heading),
                        np.cos(desired_heading - candidate_heading),
                    )
                )
            )
            candidate_progress, candidate_cross_track = (
                self._sg_project_position_to_path(path, candidate_end, progress)
            )
            progress_gain = (
                float(candidate_progress) - float(progress)
                if candidate_progress is not None
                else float("-inf")
            )
            candidate_goal_distance = float(
                np.hypot(candidate_end[0] - target[0], candidate_end[1] - target[1])
            )
            goal_gain = current_goal_distance - candidate_goal_distance
            # A stored acquisition centreline certifies the physical route,
            # but it does not predict Habitat's current navmesh slide.  Relax
            # the conservative occupancy clearance only for a live-predicted
            # endpoint that remains tightly aligned with that certificate.
            probe_cross_track_limit_cells = float(
                os.environ.get(
                    "LIFELONG_SG_TRAJECTORY_PROBE_MAX_CROSS_TRACK_CELLS",
                    "2.0",
                )
            )
            certified_probe_safe = bool(
                trajectory_probe_aligned
                and np.isfinite(candidate_cross_track)
                and float(candidate_cross_track)
                <= probe_cross_track_limit_cells + 1e-6
            )
            if not clearance_safe and not certified_probe_safe:
                continue
            advances = self._sg_discrete_candidate_advances(
                progress_gain,
                goal_gain,
                candidate_cross_track,
                minimum_advance_cells,
                maximum_progress_cross_track_cells,
            )
            reverses_last_forward = bool(
                previous_forward_origin is not None
                and np.linalg.norm(
                    candidate_end
                    - np.asarray(previous_forward_origin, dtype=np.float64)
                )
                <= max(0.5, step_cells * 0.25)
            )
            score = (
                endpoint_error
                + self.sg_path_turn_cost_cells * turn_steps
                + 0.1 * float(candidate_cross_track)
            )
            candidate_info = {
                "score": float(score),
                "turn_offset": turn_offset,
                "candidate_heading": candidate_heading,
                "navmesh_predicted": bool(navmesh_predicted),
                "certified_probe": bool(certified_probe_safe),
                "progress_gain": progress_gain,
                "goal_gain": goal_gain,
                "advances": advances,
                "reverses_last_forward": reverses_last_forward,
                "heading_alignment": heading_alignment,
                "candidate_end": candidate_end,
            }
            if certified_probe_safe:
                certified_probe_alternatives.append(candidate_info)
            else:
                alternatives.append(candidate_info)

        # While replaying an acquisition route, a live-navmesh-predicted
        # endpoint inside the narrow trajectory probe is the faithful reverse
        # action for that route.  Do not discard it merely because some other
        # heading lands in ordinary M0 free space: on a folded staircase path,
        # that unrelated endpoint can project onto a later nearby segment and
        # look like progress (849011).  Score both sets together during the
        # certified-route phase; endpoint error then keeps the controller on
        # the active route.  Outside that phase the conservative ordinary-free
        # preference remains unchanged.
        trajectory_suffix_start = target_info.get(
            "_sg_trajectory_suffix_start_progress_cells"
        )
        certified_route_phase = bool(
            target_info.get("execution_path_source")
            == "certified_trajectory_then_world_theta_star"
            and trajectory_suffix_start is not None
            and float(progress) + 1e-6 < float(trajectory_suffix_start)
        )
        if certified_route_phase and certified_probe_alternatives:
            target_info["_sg_certified_route_probe_candidate_count"] = int(
                len(certified_probe_alternatives)
            )
            alternatives.extend(certified_probe_alternatives)
            certified_probe_alternatives = []

        if not alternatives:
            replan_attempted = bool(
                kwargs.get("_sg_discrete_replan_attempted", False)
            )
            if not replan_attempted:
                replanned = self._sg_replan_world_path(
                    target_info, current_world_xz, "no_safe_discrete_forward"
                )
                if replanned is not None and len(replanned["path"]) >= 2:
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["_sg_discrete_replan_attempted"] = True
                    return self._get_sg_world_path_action(**retry_kwargs)
            probe_advancing = [
                item
                for item in certified_probe_alternatives
                if item["advances"] and not item["reverses_last_forward"]
            ]
            if probe_advancing:
                alternatives = probe_advancing
            else:
                planner_reason = (
                    getattr(self, "_sg_last_theta_failure", None) or {}
                ).get(
                    "reason",
                    "no_safe_discrete_forward_after_world_theta_replan",
                )
                recovery_action = self._sg_start_local_action_recovery(
                    target_info,
                    position,
                    current_heading,
                    target,
                    window,
                    safe_mask,
                    f"no_safe_discrete_forward_{planner_reason}",
                )
                if recovery_action is not None:
                    return int(recovery_action)
                target_info["_sg_pending_abort_reason"] = (
                    "local_action_lattice_unreachable_after_"
                    f"no_safe_discrete_forward_{planner_reason}"
                )
                return None

        advancing = [
            item
            for item in alternatives
            if item["advances"] and not item["reverses_last_forward"]
        ]
        if not advancing:
            replan_attempted = bool(
                kwargs.get("_sg_discrete_replan_attempted", False)
            )
            replan_reason = (
                "immediate_reverse_without_progress"
                if any(item["reverses_last_forward"] for item in alternatives)
                else "no_progressing_discrete_forward"
            )
            if not replan_attempted:
                replanned = self._sg_replan_world_path(
                    target_info, current_world_xz, replan_reason
                )
                if replanned is not None and len(replanned["path"]) >= 2:
                    if bool(
                        target_info.pop(
                            "_sg_force_local_action_recovery", False
                        )
                    ):
                        recovery_action = self._sg_start_local_action_recovery(
                            target_info,
                            position,
                            current_heading,
                            (
                                int(target_info["target_world_grid_x"]),
                                int(target_info["target_world_grid_z"]),
                            ),
                            replanned["window"],
                            replanned["safe_mask"],
                            "repeated_immediate_reverse_cycle",
                        )
                        if recovery_action is not None:
                            return int(recovery_action)
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["_sg_discrete_replan_attempted"] = True
                    return self._get_sg_world_path_action(**retry_kwargs)
            advancing = [
                item
                for item in certified_probe_alternatives
                if item["advances"] and not item["reverses_last_forward"]
            ]
            if not advancing:
                recovery_action = self._sg_start_local_action_recovery(
                    target_info,
                    position,
                    current_heading,
                    target,
                    window,
                    safe_mask,
                    f"{replan_reason}_after_world_theta_replan",
                )
                if recovery_action is not None:
                    return int(recovery_action)
                target_info["_sg_pending_abort_reason"] = (
                    "local_action_lattice_unreachable_after_"
                    f"{replan_reason}_after_world_theta_replan"
                )
                return None

        selected = min(advancing, key=lambda item: item["score"])
        selected_offset = int(selected["turn_offset"])
        if selected_offset == 0:
            action = HabitatSimActions.MOVE_FORWARD
            target_info["_sg_pending_forward_world_grid_position"] = tuple(
                map(float, position)
            )
            target_info["_sg_pending_forward_world_grid_endpoint"] = tuple(
                map(float, selected["candidate_end"])
            )
            if bool(selected.get("certified_probe")):
                target_info["_sg_certified_probe_forward_count"] = int(
                    target_info.get("_sg_certified_probe_forward_count", 0)
                ) + 1
        else:
            counterclockwise = selected_offset
            clockwise = (-selected_offset) % num_headings
            action = (
                HabitatSimActions.TURN_LEFT
                if counterclockwise <= clockwise
                else HabitatSimActions.TURN_RIGHT
            )

        logger.info(
            "[INHerit-SG-V7] world path action="
            f"{int(action)}, world_xz=({current_world_xz[0]:.4f},{current_world_xz[1]:.4f}), "
            f"heading={current_heading:.4f}rad, lookahead_grid=({lookahead[0]:.2f},{lookahead[1]:.2f}), "
            f"heading_error={heading_error:.4f}rad, "
            f"selected_heading={selected['candidate_heading']:.4f}rad, "
            f"navmesh_predicted={selected.get('navmesh_predicted', False)}, "
            f"certified_probe={selected.get('certified_probe', False)}, "
            f"progress_gain={selected['progress_gain']:.2f}cells, "
            f"goal_gain={selected['goal_gain']:.2f}cells, "
            f"cross_track={cross_track * resolution_m:.3f}m, "
            f"remaining={(total - progress) * resolution_m:.3f}m"
        )
        return int(action)

    def _get_sg_path_action(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        if target_info.get("_sg_world_path_cells") is not None:
            return self._get_sg_world_path_action(**kwargs)
        current_pose = np.asarray(kwargs.get("current_pose"), dtype=np.float64).reshape(-1)
        if current_pose.size < 3:
            return None
        position = current_pose[:2] * 100.0 / self.resolution
        target = (int(kwargs.get("target_map_x")), int(kwargs.get("target_map_y")))
        path = target_info.get("_sg_path_cells") or []
        safe_mask = self._sg_execution_navigation_mask(self._sg_memory_traversable_mask())
        if safe_mask is None:
            return None
        px, py = int(round(position[0])), int(round(position[1]))
        if 0 <= py < safe_mask.shape[0] and 0 <= px < safe_mask.shape[1]:
            safe_mask[py, px] = True

        previous_progress = float(target_info.get("_sg_path_progress_cells", 0.0))
        need_replan = len(path) < 2
        reason = "missing_path"
        pending_forward_pose = target_info.pop("_sg_pending_forward_pose_cells", None)
        if pending_forward_pose is not None:
            actual_displacement = float(
                np.linalg.norm(position - np.asarray(pending_forward_pose, dtype=np.float64))
            )
            expected_step = (
                float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
                * 100.0
                / self.resolution
            )
            if actual_displacement < expected_step * 0.5:
                need_replan = True
                reason = f"collision_displacement_{actual_displacement:.2f}_cells"
            else:
                target_info["_sg_last_forward_origin_cells"] = tuple(
                    map(float, pending_forward_pose)
                )
        if not need_replan:
            progress, cross_track, lookahead, total = self._sg_path_progress_and_lookahead(
                path, position, previous_progress, safe_mask
            )
            deviation_m = cross_track * self.resolution / 100.0
            if deviation_m > self.sg_path_deviation_m or lookahead is None:
                need_replan = True
                reason = f"cross_track_{deviation_m:.3f}m"
        if need_replan:
            replanned = self._sg_replan_theta_path(target_info, target, current_pose, reason)
            if replanned is None or len(replanned["path"]) < 2:
                return None
            path = target_info["_sg_path_cells"]
            progress, cross_track, lookahead, total = self._sg_path_progress_and_lookahead(
                path, position, 0.0, replanned["safe_mask"]
            )
            safe_mask = replanned["safe_mask"]

        target_info["_sg_path_progress_cells"] = float(progress)
        target_info["_sg_path_cross_track_m"] = float(cross_track * self.resolution / 100.0)
        target_info["_sg_path_remaining_m"] = float((total - progress) * self.resolution / 100.0)
        delta = np.asarray(lookahead, dtype=np.float64) - position
        desired_heading = float(np.rad2deg(np.arctan2(delta[1], delta[0])))
        current_heading = float(current_pose[2])
        heading_error = float((desired_heading - current_heading + 180.0) % 360.0 - 180.0)
        turn_deg = float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        num_headings = int(round(360.0 / turn_deg))
        current_bin = int(round(current_heading / turn_deg)) % num_headings
        step_cells = float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE) * 100.0 / self.resolution
        alternatives = []
        current_goal_distance = float(
            np.hypot(position[0] - target[0], position[1] - target[1])
        )
        minimum_advance_cells = max(
            0.25, self.sg_nav_stall_eps_m * 100.0 / self.resolution
        )
        previous_forward_origin = target_info.get("_sg_last_forward_origin_cells")
        for candidate_bin in range(num_headings):
            candidate_heading = candidate_bin * turn_deg
            candidate_rad = np.deg2rad(candidate_heading)
            candidate_end = position + step_cells * np.asarray(
                [np.cos(candidate_rad), np.sin(candidate_rad)], dtype=np.float64
            )
            if not self._sg_float_segment_is_safe(safe_mask, position, candidate_end):
                continue
            clockwise = (current_bin - candidate_bin) % num_headings
            counterclockwise = (candidate_bin - current_bin) % num_headings
            turn_steps = min(clockwise, counterclockwise)
            endpoint_error = float(np.linalg.norm(candidate_end - lookahead))
            heading_alignment = abs(
                (desired_heading - candidate_heading + 180.0) % 360.0 - 180.0
            )
            candidate_progress, candidate_cross_track = self._sg_project_position_to_path(
                path, candidate_end, progress
            )
            progress_gain = (
                float(candidate_progress) - float(progress)
                if candidate_progress is not None
                else float("-inf")
            )
            candidate_goal_distance = float(
                np.hypot(candidate_end[0] - target[0], candidate_end[1] - target[1])
            )
            goal_gain = current_goal_distance - candidate_goal_distance
            advances = bool(
                progress_gain >= minimum_advance_cells
                or goal_gain >= minimum_advance_cells
            )
            reverses_last_forward = bool(
                previous_forward_origin is not None
                and np.linalg.norm(
                    candidate_end
                    - np.asarray(previous_forward_origin, dtype=np.float64)
                )
                <= max(0.5, step_cells * 0.25)
            )
            score = (
                endpoint_error
                + self.sg_path_turn_cost_cells * turn_steps
                + 0.1 * float(candidate_cross_track)
            )
            alternatives.append(
                {
                    "score": float(score),
                    "endpoint_error": endpoint_error,
                    "turn_steps": turn_steps,
                    "heading_alignment": heading_alignment,
                    "candidate_bin": candidate_bin,
                    "progress_gain": progress_gain,
                    "goal_gain": goal_gain,
                    "advances": advances,
                    "reverses_last_forward": reverses_last_forward,
                }
            )
        if not alternatives:
            replanned = self._sg_replan_theta_path(
                target_info, target, current_pose, "no_safe_discrete_forward"
            )
            if replanned is None:
                return None
            target_info["_sg_pending_abort_reason"] = "no_safe_discrete_forward_after_theta_replan"
            return None

        advancing = [
            item
            for item in alternatives
            if item["advances"] and not item["reverses_last_forward"]
        ]
        if not advancing:
            replan_attempted = bool(kwargs.get("_sg_discrete_replan_attempted", False))
            replan_reason = (
                "immediate_reverse_without_progress"
                if any(item["reverses_last_forward"] for item in alternatives)
                else "no_progressing_discrete_forward"
            )
            if not replan_attempted:
                replanned = self._sg_replan_theta_path(
                    target_info, target, current_pose, replan_reason
                )
                if replanned is not None and len(replanned["path"]) >= 2:
                    return self._get_sg_path_action(
                        **kwargs, _sg_discrete_replan_attempted=True
                    )
            target_info["_sg_pending_abort_reason"] = (
                f"{replan_reason}_after_theta_replan"
            )
            logger.info(
                "[INHerit-SG-FIRST] no advancing non-reverse discrete action: "
                f"reason={replan_reason}, pose=({position[0]:.2f},{position[1]:.2f})"
            )
            return int(HabitatSimActions.TURN_LEFT)

        selected = min(advancing, key=lambda item: item["score"])
        selected_bin = int(selected["candidate_bin"])
        selected_heading = selected_bin * turn_deg
        if selected_bin == current_bin:
            action = HabitatSimActions.MOVE_FORWARD
            target_info["_sg_pending_forward_pose_cells"] = tuple(position.tolist())
        else:
            counterclockwise = (selected_bin - current_bin) % num_headings
            clockwise = (current_bin - selected_bin) % num_headings
            action = (
                HabitatSimActions.TURN_LEFT
                if counterclockwise <= clockwise
                else HabitatSimActions.TURN_RIGHT
            )

        logger.info(
            "[INHerit-SG-FIRST] path SG action="
            f"{int(action)}, pose=({current_pose[0]:.6f},{current_pose[1]:.6f},{current_heading:.3f}), "
            f"lookahead=({lookahead[0]:.2f},{lookahead[1]:.2f}), "
            f"heading_error={heading_error:.2f}, selected_heading={selected_heading:.1f}, "
            f"progress_gain={selected['progress_gain']:.2f}cells, "
            f"goal_gain={selected['goal_gain']:.2f}cells, "
            f"cross_track={cross_track * self.resolution / 100.0:.3f}m, "
            f"remaining={(total - progress) * self.resolution / 100.0:.3f}m"
        )
        return int(action)

    def _sg_activate_continuation_instruction(self, target_info):
        """Hand EP1 to normal VLN with full language and completed recall state."""

        branch_mode = target_info.get("_sg_instruction_branch_mode")
        if branch_mode != "full_instruction_after_sg_reach_completed_recall_todo":
            return None
        full_instruction = str(
            target_info.get("_sg_full_instruction") or ""
        ).strip()
        continuation = str(
            target_info.get("_sg_continuation_instruction") or ""
        ).strip()
        if not full_instruction or not continuation:
            raise RuntimeError(
                "SG reached a Level-3 EP1 anchor without a complete "
                "full-instruction/continuation branch"
            )
        active_instruction = str(
            getattr(self, "instruction", "") or ""
        ).strip()
        if self._sg_normalize_query_text(active_instruction) != (
            self._sg_normalize_query_text(full_instruction)
        ):
            raise RuntimeError(
                "Refusing the EP1 suffix transition because normal VLN's "
                "active instruction no longer matches the preserved complete "
                "instruction"
            )

        # Keep the complete instruction visible to every normal-VLN action
        # prompt.  For the one-time TODO initialization, however, derive only
        # the still-pending suffix and prepend the recall clause as completed.
        agent = getattr(self, "agent", None)
        if agent is not None:
            if hasattr(agent, "todo_list"):
                agent.todo_list = None
            if hasattr(agent, "last_object"):
                agent.last_object = ""
            if hasattr(agent, "todo_verification_feedback"):
                agent.todo_verification_feedback = ""
            if hasattr(agent, "stair"):
                agent.stair = False
            agent.todo_generation_instruction_override = continuation
            agent.todo_list_completed_prefix = [
                {
                    "content": str(
                        target_info.get("todo_text") or "Recall the requested object"
                    ).strip(),
                    "status": "completed",
                    "result": "Reached by memory recall navigation: {}".format(
                        target_info.get("label", "object")
                    ),
                }
            ]
        if hasattr(self, "stop_feedback"):
            self.stop_feedback = ""
        if hasattr(self, "consecutive_stop_failures"):
            self.consecutive_stop_failures = 0

        self.instruction = full_instruction
        visualizer = getattr(self, "visualizer", None)
        if visualizer is not None and hasattr(visualizer, "instruction"):
            visualizer.instruction = full_instruction
        logger.info(
            "[INHerit-SG-FIRST] EP1 anchor reached; normal VLN keeps the "
            "complete instruction while TODO initialization marks recall "
            f"completed and derives pending items from: {continuation}"
        )
        return {
            "mode": "full_instruction_with_completed_recall_todo",
            "fallback_on_sg_failure": "full_instruction",
            "active_instruction": full_instruction,
            "continuation_instruction": continuation,
            "completed_recall_todo": copy.deepcopy(
                getattr(agent, "todo_list_completed_prefix", [])
            ),
        }

    def _sg_activate_level3_suffix(self, target_info):
        """Hand the formal Level-3 suffix to normal VLN after Anchor 2."""

        if not self.level3_formal_suffix_enabled:
            return None
        return self._sg_activate_level3_lavira(
            2, mode="level3_anchor2_to_ordinary_vln_suffix"
        )

    def _sg_activate_level3_lavira(self, completed_count, *, mode):
        """Keep EP2 history and full language; initialize only unfinished TODOs."""

        plan = self._sg_dual_anchor_plan
        full_instruction = str(
            self._sg_dual_anchor_instruction or getattr(self, "instruction", "") or ""
        ).strip()
        state = self._sg_dual_anchor_state
        reached = list(state.reached_uids) if state is not None else []
        if completed_count not in (0, 1, 2) or completed_count != len(reached):
            raise RuntimeError("Level-3 TODO completion must match actual SG arrivals")
        if completed_count and plan is None:
            raise RuntimeError("Cannot mark SG TODOs complete without a recall plan")
        continuation = full_instruction
        if completed_count:
            expected = [
                str(anchor.selected_candidate.get("uid") or anchor.selected_candidate.get("node_id"))
                for anchor in plan.anchors[:completed_count]
            ]
            if reached != expected:
                raise RuntimeError("Level-3 completed TODO UIDs do not match reached anchors")
            pending = [str(anchor.segment.get("text") or "").strip()
                       for anchor in plan.anchors[completed_count:]]
            pending.append(str((plan.ordinary_vln_segment or {}).get("text") or "").strip())
            continuation = " ".join(text for text in pending if text)
        if not full_instruction or not continuation:
            raise RuntimeError("Level-3 LaViRA handoff is incomplete")
        active_instruction = str(getattr(self, "instruction", "") or "").strip()
        if self._sg_normalize_query_text(active_instruction) != (
            self._sg_normalize_query_text(full_instruction)
        ):
            raise RuntimeError(
                "Refusing Level-3 suffix handoff after the full instruction changed"
            )

        agent = getattr(self, "agent", None)
        completed = []
        for index, anchor in enumerate(plan.anchors[:completed_count] if plan else (), 1):
            completed.append(
                {
                    "content": str(anchor.segment.get("text") or "").strip(),
                    "status": "completed",
                    "result": "Completed by SG navigation; reached Anchor {}: {}".format(
                        index,
                        anchor.selected_candidate.get("label", "object"),
                    ),
                }
            )
        if agent is not None:
            if hasattr(agent, "todo_list"):
                agent.todo_list = None
            if hasattr(agent, "last_object"):
                agent.last_object = ""
            if hasattr(agent, "todo_verification_feedback"):
                agent.todo_verification_feedback = ""
            if hasattr(agent, "stair"):
                agent.stair = False
            agent.todo_generation_instruction_override = continuation
            agent.todo_list_completed_prefix = copy.deepcopy(completed)
        if hasattr(self, "stop_feedback"):
            self.stop_feedback = ""
        if hasattr(self, "consecutive_stop_failures"):
            self.consecutive_stop_failures = 0
        self.instruction = full_instruction
        visualizer = getattr(self, "visualizer", None)
        if visualizer is not None and hasattr(visualizer, "instruction"):
            visualizer.instruction = full_instruction
        logger.info(
            "[MEMORY-L3] LaViRA handoff: mode=%s completed_sg_todos=%d "
            "history=%d waypoints=%d pending=%s",
            mode, completed_count, len(getattr(self, "history_images", [])),
            len(getattr(self, "visited_targets", [])), continuation,
        )
        return {
            "mode": mode,
            "active_instruction": full_instruction,
            "continuation_instruction": continuation,
            "completed_recall_todos": completed,
        }

    def _on_sg_target_reached(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        if target_info.get("_sg_exploration_frontier"):
            current_world_xz = self._sg_current_world_xz(kwargs)
            metric_map = self._sg_world_metric_map()
            if current_world_xz is None or metric_map is None:
                raise RuntimeError(
                    "exploration frontier reached without current world memory"
                )
            scan = self._sg_exploration_adapter.frontier_reached(
                measured_world_xz=current_world_xz,
                metric_map=metric_map,
                online_builder=self.inherit_sg,
                step=int(kwargs.get("step") or 0),
            )
            query_attempt_key = target_info.get(
                "_sg_exploration_query_attempt_key"
            )
            if (
                isinstance(query_attempt_key, (list, tuple))
                and len(query_attempt_key) == 2
            ):
                query_attempt_key = tuple(map(str, query_attempt_key))
                self._sg_instruction_query_attempts[query_attempt_key] = 0
                self._sg_instruction_exhaustion_logged.discard(
                    query_attempt_key
                )
            self._sg_nav_reset_progress()
            self._sg_path_plan_active = False
            self._sg_world_plan_window = None
            self._sg_world_safe_mask = None
            self._sg_world_trajectory_probe_mask = None
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "decision": "reached_exploration_frontier_scan_then_retry",
                    "target_uid": target_info.get("uid"),
                    "frontier_world_xz": target_info.get("target_world_xz"),
                    "reach_reason": kwargs.get("reach_reason"),
                    "scan": scan,
                    "formal_memory_update_mode": (
                        "world_map_and_trajectory_sync_semantic_sg_async"
                    ),
                    "retry_uses_cached_episode_start_vlm_plan": True,
                },
            )
            logger.info(
                "[INHerit-SG-EXPLORATION] reached frontier; the evaluator's "
                f"next {scan['scan_turns']}-view panorama will scan through "
                "the online memory pipeline, then retry the cached SG target"
            )
            # The normal decision loop always acquires a full panorama before
            # invoking SG again.  Returning another explicit turn queue here
            # would scan twice (24 turns with the stock 30-degree setup).
            return []
        dual_anchor_index = target_info.get("_sg_dual_anchor_index")
        dual_anchor_active = isinstance(dual_anchor_index, int)
        dual_anchor_final = False
        if dual_anchor_active:
            state = self._sg_dual_anchor_state
            if state is None:
                raise RuntimeError("Reached a Level-3 anchor without navigation state")
            reached_uid = str(
                target_info.get("uid") or target_info.get("node_id") or ""
            )
            selected_uid = str(
                target_info.get("_sg_dual_anchor_selected_uid") or ""
            )
            if not reached_uid or reached_uid != selected_uid:
                raise RuntimeError(
                    "Reached SG UID differs from the Level-3 VLM selection"
                )
            state.mark_reached(reached_uid)
            self._sg_persist_dual_anchor_state()
            dual_anchor_final = state.phase in {"complete", "recall_complete"}
        todo_idx = target_info.get("todo_index")
        if isinstance(todo_idx, int) and isinstance(getattr(self.agent, "todo_list", None), list):
            if 0 <= todo_idx < len(self.agent.todo_list):
                label = target_info.get("label", "object")
                self.agent.todo_list[todo_idx]["status"] = "completed"
                self.agent.todo_list[todo_idx]["result"] = f"Reached by INHerit-SG SG-first: {label}"
                logger.info(f"[INHerit-SG-FIRST] TODO idx={todo_idx} completed locally by SG_NAV: {label}")
        instruction_transition = None
        if dual_anchor_active and dual_anchor_final:
            self._sg_instruction_reached_episodes.add(
                str(getattr(self, "current_episode_id", ""))
            )
            instruction_transition = self._sg_activate_level3_suffix(target_info)
        elif target_info.get("query_source") == "instruction":
            self._sg_instruction_reached_episodes.add(str(getattr(self, "current_episode_id", "")))
            if (
                target_info.get("_sg_instruction_branch_mode")
                == "full_instruction_after_sg_reach_completed_recall_todo"
                and str(
                    target_info.get("_sg_continuation_instruction") or ""
                ).strip()
            ):
                instruction_transition = (
                    self._sg_activate_continuation_instruction(target_info)
                )
        self._sg_nav_reset_progress()
        self._sg_path_plan_active = False
        self._sg_world_plan_window = None
        self._sg_world_safe_mask = None
        self._sg_world_trajectory_probe_mask = None
        final_actions = []
        waypoint = target_info.get("waypoint") or {}
        target_heading = waypoint.get("heading_rad")
        current_heading = kwargs.get("current_heading")
        heading_delta = None
        num_turns = 0
        align_at_this_target = bool(
            not dual_anchor_active or dual_anchor_final
        )
        if self.sg_align_heading and align_at_this_target and target_heading is not None and current_heading is not None:
            heading_delta = float(
                np.arctan2(
                    np.sin(float(target_heading) - float(current_heading)),
                    np.cos(float(target_heading) - float(current_heading)),
                )
            )
            turn_angle_deg = float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
            if abs(np.rad2deg(heading_delta)) > self.sg_heading_tolerance_deg and turn_angle_deg > 0:
                num_turns = int(round(abs(np.rad2deg(heading_delta)) / turn_angle_deg))
                turn_action = HabitatSimActions.TURN_LEFT if heading_delta > 0 else HabitatSimActions.TURN_RIGHT
                final_actions.extend([turn_action] * num_turns)
        instruction_branch_mode = target_info.get(
            "_sg_instruction_branch_mode"
        )
        instruction_has_continuation = bool(
            str(
                target_info.get("_sg_continuation_instruction") or ""
            ).strip()
        )
        if dual_anchor_active:
            stop_at_this_target = bool(
                dual_anchor_final and instruction_transition is None
            )
        elif instruction_branch_mode == (
            "full_instruction_after_sg_reach_completed_recall_todo"
        ):
            # A pure recall ends at the selected SG target.  Recall+route keeps
            # the full instruction for Uni-LaViRA and records the recall item
            # as completed in its TODO state.
            stop_at_this_target = not instruction_has_continuation
        else:
            stop_at_this_target = bool(self.sg_stop_at_waypoint)
        if stop_at_this_target:
            final_actions.append(HabitatSimActions.STOP)
        if dual_anchor_active and not dual_anchor_final:
            terminal_decision = "reached_dual_anchor1_continue_to_anchor2"
        elif dual_anchor_active and instruction_transition is not None:
            terminal_decision = "reached_dual_anchor2_handoff_vln_suffix"
        elif dual_anchor_active and dual_anchor_final:
            terminal_decision = "reached_dual_anchor2_stop"
        elif stop_at_this_target:
            terminal_decision = "reached_align_and_stop"
        elif instruction_transition is not None:
            terminal_decision = "reached_handoff_full_instruction_completed_recall"
        else:
            terminal_decision = "reached_mark_todo_completed"
        self._write_sg_decision(
            kwargs.get("step"),
            {
                "decision": terminal_decision,
                "target": self._compact_sg_results([target_info])[0] if target_info else None,
                "distance_to_target": float(kwargs.get("distance_to_target", 0.0)),
                "reached_agent_world_position": kwargs.get(
                    "current_world_position"
                ),
                "reach_reason": kwargs.get("reach_reason") or "exact_cell",
                "best_distance_to_target_cells": target_info.get("_sg_best_distance_cells"),
                "refinement_count": int(target_info.get("_sg_refinement_count", 0)),
                "correction_replan_count": int(
                    target_info.get("_sg_correction_replan_count", 0)
                ),
                "collision_replans_without_progress": int(
                    target_info.get("_sg_collision_replans_without_progress", 0)
                ),
                "navigation_target_world_xz": target_info.get("target_world_xz"),
                "raw_object_center_world_xz": target_info.get(
                    "semantic_object_center_world_xz",
                    target_info.get("raw_target_world_xz"),
                ),
                "navigation_reference_world_xz": target_info.get(
                    "navigation_reference_world_xz",
                    target_info.get("raw_target_world_xz"),
                ),
                "navigation_reference_policy": target_info.get(
                    "navigation_reference_policy"
                ),
                "reachable_object_center_residual_m": target_info.get(
                    "reachable_object_center_residual_m"
                ),
                "reachable_navigation_reference_residual_m": target_info.get(
                    "reachable_navigation_reference_residual_m"
                ),
                "reachable_semantic_object_center_residual_m": target_info.get(
                    "reachable_semantic_object_center_residual_m"
                ),
                "navigation_target_source": target_info.get(
                    "navigation_target_source"
                ),
                "current_heading_rad": current_heading,
                "target_heading_rad": target_heading,
                "heading_delta_rad": heading_delta,
                "num_turns": num_turns,
                "stop_requested": stop_at_this_target,
                "dual_anchor_index": dual_anchor_index,
                "dual_anchor_state": (
                    self._sg_dual_anchor_state.to_dict()
                    if dual_anchor_active
                    else None
                ),
                "ordinary_vln_handoff": bool(
                    dual_anchor_active and instruction_transition is not None
                ),
                "instruction_transition": instruction_transition,
            },
        )
        return final_actions

    def _on_sg_nav_aborted(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        reason = kwargs.get("reason", "unknown")
        if target_info.get("_sg_exploration_frontier"):
            transition = (
                self._sg_exploration_adapter.frontier_navigation_aborted(
                    reason=str(reason), step=int(kwargs.get("step") or 0)
                )
            )
            query_attempt_key = target_info.get(
                "_sg_exploration_query_attempt_key"
            )
            if (
                isinstance(query_attempt_key, (list, tuple))
                and len(query_attempt_key) == 2
            ):
                query_attempt_key = tuple(map(str, query_attempt_key))
                self._sg_instruction_query_attempts[query_attempt_key] = 0
                self._sg_instruction_exhaustion_logged.discard(
                    query_attempt_key
                )
            self._sg_nav_reset_progress()
            self._sg_path_plan_active = False
            self._sg_world_plan_window = None
            self._sg_world_safe_mask = None
            self._sg_world_trajectory_probe_mask = None
            self._write_sg_decision(
                kwargs.get("step"),
                {
                    "decision": "abort_exploration_frontier_retry_target",
                    "reason": str(reason),
                    "transition": transition,
                    "instruction_fallback": "full_instruction_on_exhaustion",
                },
            )
            return
        task_text = target_info.get("todo_text")
        dual_anchor_index = target_info.get("_sg_dual_anchor_index")
        dual_anchor_active = isinstance(dual_anchor_index, int)
        dual_anchor_state = None
        if dual_anchor_active:
            dual_anchor_state = self._sg_fail_dual_anchor(
                "navigation_aborted:{}".format(reason)
            )
        instruction_fallback = (
            "full_instruction_with_completed_sg_todos"
            if dual_anchor_active and self._sg_level3_lavira_fallback is not None
            else None
        )
        if target_info.get("_sg_instruction_branch_mode") == (
            "full_instruction_after_sg_reach_completed_recall_todo"
        ):
            full_instruction = str(
                target_info.get("_sg_full_instruction") or ""
            ).strip()
            active_instruction = str(
                getattr(self, "instruction", "") or ""
            ).strip()
            if (
                not full_instruction
                or self._sg_normalize_query_text(active_instruction)
                != self._sg_normalize_query_text(full_instruction)
            ):
                raise RuntimeError(
                    "SG navigation aborted after the EP1 full instruction "
                    "was lost or changed"
                )
            instruction_fallback = "full_instruction"
            logger.info(
                "[INHerit-SG-FIRST] EP1 SG navigation failed; preserving "
                "the complete instruction for normal VLN fallback."
            )
        self._sg_block_target_for_task(target_info, task_text, reason)
        self._sg_skip_next_task_key = self._sg_task_key(task_text)
        self._sg_nav_reset_progress()
        self._sg_path_plan_active = False
        self._sg_world_plan_window = None
        self._sg_world_safe_mask = None
        self._sg_world_trajectory_probe_mask = None
        self._write_sg_decision(
            kwargs.get("step"),
            {
                "decision": (
                    "abort_dual_anchor_nav"
                    if dual_anchor_active
                    else "abort_sg_nav"
                ),
                "reason": reason,
                "target": self._compact_sg_results([target_info])[0] if target_info else None,
                "distance_to_target": float(kwargs.get("distance_to_target", 0.0)),
                "correction_replan_count": int(
                    target_info.get("_sg_correction_replan_count", 0)
                ),
                "collision_replans_without_progress": int(
                    target_info.get("_sg_collision_replans_without_progress", 0)
                ),
                "navigation_target_world_xz": target_info.get("target_world_xz"),
                "raw_object_center_world_xz": target_info.get(
                    "semantic_object_center_world_xz",
                    target_info.get("raw_target_world_xz"),
                ),
                "navigation_reference_world_xz": target_info.get(
                    "navigation_reference_world_xz",
                    target_info.get("raw_target_world_xz"),
                ),
                "navigation_reference_policy": target_info.get(
                    "navigation_reference_policy"
                ),
                "reachable_object_center_residual_m": target_info.get(
                    "reachable_object_center_residual_m"
                ),
                "reachable_navigation_reference_residual_m": target_info.get(
                    "reachable_navigation_reference_residual_m"
                ),
                "reachable_semantic_object_center_residual_m": target_info.get(
                    "reachable_semantic_object_center_residual_m"
                ),
                "instruction_fallback": instruction_fallback,
                "dual_anchor_index": dual_anchor_index,
                "dual_anchor_state": dual_anchor_state,
                "ordinary_vln_executed": False if dual_anchor_active else None,
            },
        )

    def _sg_nav_begin_progress(self, step, dist_to_tgt):
        self._sg_nav_progress = {
            "best_dist": float(dist_to_tgt),
            "stall_count": 0,
            "last_step": int(step),
        }

    def _sg_nav_reset_progress(self):
        self._sg_nav_progress = None

    @staticmethod
    def _sg_reset_world_path_stall_guard(target_info, clear_phase=False):
        """Reset stall state whose coordinate system is the active world path."""

        target_info.pop("_sg_world_stall_path_progress_cells", None)
        target_info.pop("_sg_previous_world_grid_cell", None)
        target_info["_sg_non_improving_moves"] = 0
        pending = str(target_info.get("_sg_pending_abort_reason", ""))
        if pending.startswith("no_progress_before_exact_world_target_best_"):
            target_info.pop("_sg_pending_abort_reason", None)
        if clear_phase:
            target_info.pop("_sg_world_stall_phase", None)

    def _sg_world_nav_reached_reason(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        current_world_xz = self._sg_current_world_xz(kwargs)
        metric_map = self._sg_world_metric_map()
        if current_world_xz is None or metric_map is None:
            return None
        resolution_m = float(metric_map.spec.resolution_m)
        current_grid_position = (
            np.asarray(current_world_xz, dtype=np.float64) / resolution_m - 0.5
        )
        current_grid_cell = metric_map.spec.world_to_cell(
            float(current_world_xz[0]), float(current_world_xz[1])
        )
        target_grid = (
            int(target_info["target_world_grid_x"]),
            int(target_info["target_world_grid_z"]),
        )
        dist_cells = float(
            np.linalg.norm(
                current_grid_position - np.asarray(target_grid, dtype=np.float64)
            )
        )
        if tuple(current_grid_cell) == target_grid:
            target_info["_sg_best_distance_cells"] = dist_cells
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))
            return "exact_world_grid_cell"
        goal_tolerance_m = max(
            0.0,
            float(
                getattr(
                    self,
                    "sg_nav_goal_tolerance_m",
                    getattr(self, "sg_local_action_goal_tolerance_m", 0.15),
                )
            ),
        )
        distance_m = dist_cells * resolution_m
        if distance_m <= goal_tolerance_m + 1e-6:
            target_info["_sg_best_distance_cells"] = dist_cells
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))
            return (
                "within_discrete_world_goal_tolerance_"
                f"{distance_m:.2f}m"
            )
        local_goal_tolerance = target_info.get(
            "_sg_local_action_goal_tolerance_cells"
        )
        if (
            local_goal_tolerance is not None
            and dist_cells <= float(local_goal_tolerance) + 1e-6
        ):
            target_info["_sg_best_distance_cells"] = dist_cells
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))
            return (
                "nearest_reachable_discrete_action_pose_"
                f"{dist_cells * resolution_m:.2f}m"
            )

        best_dist = float(target_info.get("_sg_best_distance_cells", float("inf")))
        eps_cells = self.sg_nav_stall_eps_m / resolution_m
        distance_meaningfully_improved = dist_cells < best_dist - eps_cells
        closer = dist_cells < best_dist - 1e-6
        if closer:
            best_dist = dist_cells
            target_info["_sg_best_distance_cells"] = best_dist
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))

        # A bounded action-lattice recovery intentionally leaves the active
        # Theta* polyline while it executes a finite, prevalidated detour.  Do
        # not let the old path's stall guard abort that detour before its queue
        # reaches the local subgoal.  The queue is independently bounded by
        # LIFELONG_SG_LOCAL_ACTION_MAX_ACTIONS; after its final action the next
        # evaluator cycle resumes normal world-path progress accounting from a
        # clean baseline.
        if (
            target_info.get("_sg_local_action_queue")
            or target_info.get("_sg_local_action_recovery_active")
        ):
            self._sg_reset_world_path_stall_guard(target_info)
            target_info["_sg_world_stall_progress_source"] = (
                "bounded_local_action_recovery"
            )
            return None

        # A certified acquisition path is intentionally allowed to move away
        # from the final object while it follows a safe historical detour.  The
        # old guard counted those moves against straight-line target distance
        # and could abort just before the final suffix (824904 ep2).  Measure
        # progress in the active planned path's cumulative coordinate instead.
        path = target_info.get("_sg_world_path_cells") or []
        meaningfully_progressed = distance_meaningfully_improved
        recovery_meaningfully_progressed = distance_meaningfully_improved
        if len(path) >= 2:
            cumulative = self._sg_path_cumulative(path)
            minimum_progress = max(
                float(target_info.get("_sg_path_progress_cells", 0.0)),
                float(
                    target_info.get(
                        "_sg_world_stall_path_progress_cells", 0.0
                    )
                ),
            )
            projected_progress, _ = self._sg_project_position_to_path(
                path,
                current_grid_position,
                minimum_progress,
                cumulative=cumulative,
            )
            path_progress = (
                minimum_progress
                if projected_progress is None
                else max(minimum_progress, float(projected_progress))
            )

            suffix_start = target_info.get(
                "_sg_trajectory_suffix_start_progress_cells"
            )
            if suffix_start is None:
                phase = "world_theta"
            elif path_progress + 1e-6 >= float(suffix_start):
                phase = "trajectory_suffix"
            else:
                phase = "trajectory_route"
            if target_info.get("_sg_world_stall_phase") != phase:
                self._sg_reset_world_path_stall_guard(target_info)
                target_info["_sg_world_stall_phase"] = phase

            stall_progress = target_info.get(
                "_sg_world_stall_path_progress_cells"
            )
            if stall_progress is None:
                target_info["_sg_world_stall_path_progress_cells"] = float(
                    path_progress
                )
                meaningfully_progressed = True
            else:
                path_gain = path_progress - float(stall_progress)
                meaningfully_progressed = path_gain >= eps_cells - 1e-6
                if meaningfully_progressed:
                    recovery_meaningfully_progressed = True
                    target_info["_sg_world_stall_path_progress_cells"] = float(
                        path_progress
                    )
            target_info["_sg_world_stall_progress_source"] = "planned_path"
        else:
            target_info["_sg_world_stall_progress_source"] = (
                "target_distance_fallback"
            )

        if meaningfully_progressed:
            target_info["_sg_non_improving_moves"] = 0
        if recovery_meaningfully_progressed:
            target_info["_sg_collision_replans_without_progress"] = 0
            target_info["_sg_consecutive_failed_recoveries"] = 0

        previous_cell = target_info.get("_sg_previous_world_grid_cell")
        if previous_cell is not None and tuple(previous_cell) != tuple(current_grid_cell):
            if not meaningfully_progressed:
                target_info["_sg_non_improving_moves"] = int(
                    target_info.get("_sg_non_improving_moves", 0)
                ) + 1
        target_info["_sg_previous_world_grid_cell"] = tuple(
            map(int, current_grid_cell)
        )

        best_dist = float(target_info.get("_sg_best_distance_cells", dist_cells))
        at_best = dist_cells <= best_dist + 1e-6
        best_m = best_dist * resolution_m
        stalled = int(target_info.get("_sg_non_improving_moves", 0)) >= int(
            self.sg_nav_stall_steps
        )
        forward_step_m = float(
            self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        )
        path_remaining_m = float(
            target_info.get("_sg_path_remaining_m", float("inf"))
        )
        at_discrete_path_end = (
            path_remaining_m <= forward_step_m + self.sg_nav_stall_eps_m
        )
        if at_discrete_path_end and at_best:
            window = getattr(self, "_sg_world_plan_window", None)
            safe_mask = getattr(self, "_sg_world_safe_mask", None)
            has_safe_improving_forward = False
            if window is not None and safe_mask is not None:
                current_local = self._sg_window_point(window, current_grid_position)
                step_cells = forward_step_m / resolution_m
                supplied_heading = kwargs.get("current_world_heading")
                if supplied_heading is None:
                    supplied_heading = getattr(self, "_sg_world_heading_rad", None)
                if supplied_heading is None:
                    target_info["_sg_pending_abort_reason"] = (
                        "missing_current_world_heading_at_path_end"
                    )
                    return None
                current_heading = float(
                    np.asarray(supplied_heading).reshape(-1)[0]
                )
                turn_rad = np.deg2rad(
                    float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
                )
                headings = self._sg_discrete_heading_lattice(
                    current_heading, turn_rad
                )
                for heading in headings:
                    candidate = current_grid_position + step_cells * np.asarray(
                        [-np.sin(heading), -np.cos(heading)], dtype=np.float64
                    )
                    candidate_local = self._sg_window_point(window, candidate)
                    if not self._sg_float_segment_is_safe(
                        safe_mask, current_local, candidate_local
                    ):
                        continue
                    candidate_distance = float(
                        np.linalg.norm(
                            candidate - np.asarray(target_grid, dtype=np.float64)
                        )
                    )
                    if candidate_distance < dist_cells - 1e-6:
                        has_safe_improving_forward = True
                        break
            if not has_safe_improving_forward:
                return (
                    "no_safe_improving_discrete_forward_at_world_path_end_"
                    f"{dist_cells * resolution_m:.2f}m"
                )
        if stalled:
            target_info["_sg_pending_abort_reason"] = (
                f"no_progress_before_exact_world_target_best_{best_m:.2f}m"
            )
        return None

    def _sg_nav_reached_reason(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        if target_info.get("target_world_xz") is not None:
            return self._sg_world_nav_reached_reason(**kwargs)
        dist_cells = float(kwargs.get("distance_to_target", float("inf")))
        distance_m = dist_cells * self.resolution / 100.0
        agent_x = int(kwargs.get("agent_map_x"))
        agent_y = int(kwargs.get("agent_map_y"))
        current_pose = np.asarray(kwargs.get("current_pose"), dtype=np.float64).reshape(-1)
        turn_deg = float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)

        best_dist = float(target_info.get("_sg_best_distance_cells", float("inf")))
        eps_cells = self.sg_nav_stall_eps_m * 100.0 / self.resolution
        meaningfully_improved = dist_cells < best_dist - eps_cells
        closer = dist_cells < best_dist - 1e-6
        if closer:
            best_dist = dist_cells
            target_info["_sg_best_distance_cells"] = best_dist
            target_info["_sg_best_cell"] = (agent_x, agent_y)
        if meaningfully_improved:
            target_info["_sg_non_improving_moves"] = 0

        previous_cell = target_info.get("_sg_previous_cell")
        if previous_cell is not None and tuple(previous_cell) != (agent_x, agent_y):
            if not meaningfully_improved:
                target_info["_sg_non_improving_moves"] = int(
                    target_info.get("_sg_non_improving_moves", 0)
                ) + 1
        target_info["_sg_previous_cell"] = (agent_x, agent_y)

        best_dist = float(target_info.get("_sg_best_distance_cells", dist_cells))
        at_best = dist_cells <= best_dist + 1e-6
        best_m = best_dist * self.resolution / 100.0
        stalled = int(target_info.get("_sg_non_improving_moves", 0)) >= self.sg_nav_stall_steps
        forward_step_m = float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
        path_remaining_m = float(target_info.get("_sg_path_remaining_m", float("inf")))
        at_discrete_path_end = path_remaining_m <= forward_step_m + self.sg_nav_stall_eps_m
        if at_discrete_path_end and at_best and current_pose.size >= 2:
            target_x = float(target_info["target_map_x"])
            target_y = float(target_info["target_map_y"])
            current_xy = current_pose[:2] * 100.0 / self.resolution
            current_continuous_distance = float(
                np.hypot(current_xy[0] - target_x, current_xy[1] - target_y)
            )
            step_cells = (
                float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
                * 100.0
                / self.resolution
            )
            headings = np.deg2rad(np.arange(0.0, 360.0, turn_deg))
            safe_mask = self._sg_execution_navigation_mask(
                self._sg_memory_traversable_mask()
            )
            has_safe_improving_forward = False
            if safe_mask is not None:
                px, py = int(round(current_xy[0])), int(round(current_xy[1]))
                if 0 <= py < safe_mask.shape[0] and 0 <= px < safe_mask.shape[1]:
                    safe_mask[py, px] = True
                for heading in headings:
                    candidate = current_xy + step_cells * np.asarray(
                        [np.cos(heading), np.sin(heading)], dtype=np.float64
                    )
                    if not self._sg_float_segment_is_safe(
                        safe_mask, current_xy, candidate
                    ):
                        continue
                    candidate_distance = float(
                        np.hypot(candidate[0] - target_x, candidate[1] - target_y)
                    )
                    if candidate_distance < current_continuous_distance - 1e-6:
                        has_safe_improving_forward = True
                        break
            if not has_safe_improving_forward:
                return f"no_safe_improving_discrete_forward_at_path_end_{distance_m:.2f}m"
        if stalled:
            target_info["_sg_pending_abort_reason"] = (
                f"no_progress_before_exact_target_best_{best_m:.2f}m"
            )
        return None

    def _sg_nav_progress_abort_reason(self, step, dist_to_tgt, target_info=None):
        # SG path navigation records progress in _sg_nav_reached_reason.
        # Stagnation is an abort unless the discrete path-end test explicitly
        # established that no safe forward action can get closer.
        if target_info:
            reason = target_info.pop("_sg_pending_abort_reason", None)
            if reason:
                return reason
        if self._sg_path_plan_active:
            return None
        if self._sg_nav_progress is None:
            self._sg_nav_begin_progress(step, dist_to_tgt)
            return None
        resolution_m = float(
            (target_info or {}).get(
                "world_map_resolution_m", float(self.resolution) / 100.0
            )
        )
        eps_cells = self.sg_nav_stall_eps_m / resolution_m
        prog = self._sg_nav_progress
        if float(dist_to_tgt) < float(prog["best_dist"]) - eps_cells:
            prog["best_dist"] = float(dist_to_tgt)
            prog["stall_count"] = 0
            prog["last_step"] = int(step)
            return None
        prog["stall_count"] += 1
        prog["last_step"] = int(step)
        if prog["stall_count"] >= self.sg_nav_stall_steps:
            best_m = float(prog["best_dist"]) * resolution_m
            return f"stalled_{prog['stall_count']}_steps_no_closer_best_dist_{best_m:.2f}m"
        return None

    def _current_pending_todo(self):
        todo_list = getattr(self.agent, "todo_list", None)
        if not isinstance(todo_list, list):
            return None, None
        for idx, item in enumerate(todo_list):
            if item.get("status", "pending") != "completed":
                content = str(item.get("content", "")).strip()
                if content:
                    return idx, content
        return None, None

    def _sg_query_task_text(self, kwargs):
        source = self.sg_query_source
        if source in {"level3_anchor_clause", "level3_anchor_instruction"}:
            instruction = str(
                kwargs.get("instruction")
                or getattr(self, "instruction", "")
                or ""
            ).strip()
            # Level-3 EP1 is composed as ``return to the known anchor. continue
            # to the unknown goal``.  SG sees only the policy-visible anchor
            # clause.  A successful arrival switches normal VLN to the suffix;
            # every SG failure leaves this complete instruction untouched.
            anchor_clause, _ = self._sg_split_anchor_instruction(instruction)
            return None, anchor_clause or None, "instruction"
        if source in {"instruction", "raw_instruction", "episode_instruction"}:
            instruction = str(kwargs.get("instruction") or getattr(self, "instruction", "") or "").strip()
            return None, instruction or None, "instruction"
        todo_idx, task_text = self._current_pending_todo()
        return todo_idx, task_text, "todo"

    def _sg_coords_to_map(self, sg_position):
        fx_w = float(sg_position[0])
        fz_w = float(sg_position[2])
        w2m = getattr(self, "_w2m", None)
        if w2m is None:
            center = int(self.config.MAP.MAP_SIZE_CM / 2.0 / self.resolution)
            return (
                max(0, min(center, self.map_shape[0] - 1)),
                max(0, min(center, self.map_shape[1] - 1)),
            )
        ax_now, ay_now, sx_now, sy_now, c_phi, s_phi = w2m
        sx_f = -fz_w
        sy_f = -fx_w
        dsx = sx_f - sx_now
        dsy = sy_f - sy_now
        px_f = ax_now + (c_phi * dsx - s_phi * dsy)
        py_f = ay_now + (s_phi * dsx + c_phi * dsy)
        mx = int(px_f * 100.0 / self.resolution)
        my = int(py_f * 100.0 / self.resolution)
        mx = max(0, min(mx, self.map_shape[1] - 1))
        my = max(0, min(my, self.map_shape[0] - 1))
        return mx, my

    def _merge_sg_memory_traversable(self, traversable):
        memory_mask = self._sg_memory_traversable_mask()
        if memory_mask is None:
            return traversable
        current = np.asarray(traversable, dtype=bool)
        if current.shape != memory_mask.shape:
            return traversable
        return np.logical_or(current, memory_mask)

    def _sg_memory_traversable_mask(self):
        adapter = getattr(self, "inherit_sg", None)
        if adapter is None or not bool(getattr(adapter, "_preloaded_session", False)):
            return None
        lavira_memory = self._load_lavira_traversable_memory()
        if lavira_memory is not None:
            return self._project_lavira_traversable_memory(lavira_memory)

        pipeline = getattr(adapter, "pipeline", None)
        map_manager = getattr(pipeline, "map_manager", None)
        free_bin = getattr(map_manager, "free_bin", None)
        w2m = getattr(self, "_w2m", None)
        if free_bin is None or w2m is None:
            return None

        cache_key = tuple(round(float(value), 5) for value in w2m)
        if self._sg_memory_traversable_cache_key == cache_key:
            return self._sg_memory_traversable_cache

        free = np.asarray(free_bin) > 0
        free_v, free_u = np.where(free)
        if free_u.size == 0:
            return None
        resolution_m = float(getattr(map_manager, "resolution", 0.0))
        min_x = float(getattr(map_manager, "min_x", 0.0))
        min_z = float(getattr(map_manager, "min_z", 0.0))
        if resolution_m <= 0:
            return None

        world_x = free_u.astype(np.float64) * resolution_m + min_x
        world_z = free_v.astype(np.float64) * resolution_m + min_z
        map_x, map_y = self._sg_world_arrays_to_map(world_x, world_z, w2m)
        mask = np.zeros(self.map_shape, dtype=bool)
        valid = (
            (map_x >= 0)
            & (map_x < mask.shape[1])
            & (map_y >= 0)
            & (map_y < mask.shape[0])
        )
        mask[map_y[valid], map_x[valid]] = True

        occ_bin = getattr(map_manager, "occ_bin", None)
        if occ_bin is not None:
            occ_v, occ_u = np.where(np.asarray(occ_bin) > 0)
            if occ_u.size:
                occ_x = occ_u.astype(np.float64) * resolution_m + min_x
                occ_z = occ_v.astype(np.float64) * resolution_m + min_z
                occ_map_x, occ_map_y = self._sg_world_arrays_to_map(occ_x, occ_z, w2m)
                occ_valid = (
                    (occ_map_x >= 0)
                    & (occ_map_x < mask.shape[1])
                    & (occ_map_y >= 0)
                    & (occ_map_y < mask.shape[0])
                )
                mask[occ_map_y[occ_valid], occ_map_x[occ_valid]] = False

        self._sg_memory_traversable_cache_key = cache_key
        self._sg_memory_traversable_cache = mask
        logger.info(f"[INHerit-SG-FIRST] projected {int(mask.sum())} frozen free-space cells into planner map")
        return mask

    def _save_lavira_traversable_memory(self):
        adapter = getattr(self, "inherit_sg", None)
        if adapter is None or bool(getattr(adapter, "preload_query_only", False)):
            return
        capture_only = bool(getattr(adapter, "capture_only", False))
        if capture_only:
            out_dir = getattr(adapter, "offline_rgbd_dir", None)
            output_name = "lavira_navigation_memory.npz"
        else:
            pipeline = getattr(adapter, "pipeline", None)
            out_dir = getattr(pipeline, "run_dir", None)
            output_name = "lavira_observed_free_memory.npz"
        observed_free = getattr(self, "floor", None)
        observed_obstacle = getattr(self, "_sg_observed_obstacle", None)
        w2m = getattr(self, "_w2m", None)
        if not out_dir or observed_free is None or observed_obstacle is None or w2m is None:
            return

        mask = np.asarray(observed_free, dtype=bool)
        obstacle_mask = np.asarray(observed_obstacle, dtype=bool)
        if mask.ndim != 2 or obstacle_mask.shape != mask.shape or not mask.any():
            return
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, output_name)
        np.savez_compressed(
            out_path,
            observed_free=mask,
            observed_obstacle=obstacle_mask,
            world_to_map=np.asarray(w2m, dtype=np.float64),
            resolution_cm=np.asarray(float(self.resolution), dtype=np.float64),
            source_episode_id=np.asarray(str(getattr(self, "current_episode_id", ""))),
        )
        logger.info(
            f"[INHerit-SG] saved ep0 LaViRA navigation memory: {out_path} "
            f"({int(mask.sum())} free, {int(obstacle_mask.sum())} obstacle cells)"
        )

    def _load_lavira_traversable_memory(self):
        preload_dir = os.environ.get("INHERIT_SG_PRELOAD_DIR", "").strip()
        if not preload_dir:
            return None
        path = os.path.join(preload_dir, "lavira_observed_free_memory.npz")
        if self._sg_lavira_memory_source_path == path:
            return self._sg_lavira_memory_source
        self._sg_lavira_memory_source_path = path
        if not os.path.isfile(path):
            logger.warning(
                f"[INHerit-SG-FIRST] frozen M0 has no LaViRA observed-free memory; "
                f"falling back to SG free-space: {path}"
            )
            self._sg_lavira_memory_source = None
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                source = {
                    "observed_free": np.asarray(data["observed_free"], dtype=bool),
                    "observed_obstacle": np.asarray(data["observed_obstacle"], dtype=bool),
                    "world_to_map": np.asarray(data["world_to_map"], dtype=np.float64),
                    "resolution_cm": float(np.asarray(data["resolution_cm"]).reshape(-1)[0]),
                }
            if (
                source["observed_free"].ndim != 2
                or source["observed_obstacle"].shape != source["observed_free"].shape
                or source["world_to_map"].shape != (6,)
            ):
                raise ValueError("invalid LaViRA observed-free memory schema")
            self._sg_lavira_memory_source = source
            logger.info(
                f"[INHerit-SG-FIRST] loaded ep0 LaViRA observed-free memory: {path} "
                f"({int(source['observed_free'].sum())} cells)"
            )
            return source
        except Exception as exc:
            logger.warning(f"[INHerit-SG-FIRST] failed to load {path}: {exc}")
            self._sg_lavira_memory_source = None
            return None

    def _project_lavira_traversable_memory(self, source):
        w2m = getattr(self, "_w2m", None)
        if w2m is None:
            return None
        cache_key = ("lavira",) + tuple(round(float(value), 5) for value in w2m)
        if self._sg_memory_traversable_cache_key == cache_key:
            return self._sg_memory_traversable_cache

        source_resolution_m = float(source["resolution_cm"]) / 100.0
        source_w2m = np.asarray(source["world_to_map"], dtype=np.float64)
        source_free = np.asarray(source["observed_free"], dtype=bool)
        source_obstacle = np.asarray(source["observed_obstacle"], dtype=bool)

        # Inverse rasterization samples every destination cell from M0. Forward
        # point splatting drops cells under rotation (notably at 30 degrees),
        # creating checkerboard holes that are not physical obstacles.
        destination_y, destination_x = np.indices(self.map_shape, dtype=np.float64)
        destination_px = destination_x * float(self.resolution) / 100.0
        destination_py = destination_y * float(self.resolution) / 100.0
        ax_now, ay_now, sx_now, sy_now, c_phi, s_phi = map(float, w2m)
        delta_px = destination_px - ax_now
        delta_py = destination_py - ay_now
        dsx = c_phi * delta_px + s_phi * delta_py
        dsy = -s_phi * delta_px + c_phi * delta_py
        world_x = -(dsy + sy_now)
        world_z = -(dsx + sx_now)

        source_ax, source_ay, source_sx, source_sy, source_c, source_s = source_w2m
        source_dsx = -world_z - source_sx
        source_dsy = -world_x - source_sy
        source_px = source_ax + source_c * source_dsx - source_s * source_dsy
        source_py = source_ay + source_s * source_dsx + source_c * source_dsy
        source_x = np.rint(source_px / source_resolution_m).astype(np.int32)
        source_y = np.rint(source_py / source_resolution_m).astype(np.int32)
        valid = (
            (source_x >= 0)
            & (source_x < source_free.shape[1])
            & (source_y >= 0)
            & (source_y < source_free.shape[0])
        )

        mask = np.zeros(self.map_shape, dtype=bool)
        obstacle_mask = np.zeros(self.map_shape, dtype=bool)
        mask[valid] = source_free[source_y[valid], source_x[valid]]
        obstacle_mask[valid] = source_obstacle[source_y[valid], source_x[valid]]
        mask = np.logical_and(mask, np.logical_not(obstacle_mask))
        self._sg_memory_obstacle_cache = obstacle_mask
        self._sg_memory_traversable_cache_key = cache_key
        self._sg_memory_traversable_cache = mask
        logger.info(
            f"[INHerit-SG-FIRST] inverse-projected {int(mask.sum())} ep0 LaViRA "
            "observed-free cells into planner map"
        )
        return mask

    def _sg_world_arrays_to_map(self, world_x, world_z, w2m):
        ax_now, ay_now, sx_now, sy_now, c_phi, s_phi = w2m
        sx_f = -np.asarray(world_z, dtype=np.float64)
        sy_f = -np.asarray(world_x, dtype=np.float64)
        dsx = sx_f - sx_now
        dsy = sy_f - sy_now
        px_f = ax_now + (c_phi * dsx - s_phi * dsy)
        py_f = ay_now + (s_phi * dsx + c_phi * dsy)
        return (
            np.rint(px_f * 100.0 / self.resolution).astype(np.int32),
            np.rint(py_f * 100.0 / self.resolution).astype(np.int32),
        )

    def _process_map(self, step, full_map):
        traversable, floor, frontiers = super()._process_map(step, full_map)
        explored = np.asarray(full_map[1], dtype=bool)
        self._sg_observed_obstacle = np.logical_and(
            explored,
            np.logical_not(np.asarray(floor, dtype=bool)),
        )
        adapter = getattr(self, "inherit_sg", None)
        if adapter is not None and self._w2m is not None:
            adapter.update_world_lavira_occupancy_snapshot(
                np.asarray(floor, dtype=bool),
                self._sg_observed_obstacle,
                self._w2m,
                self.resolution,
            )
        # V7 never reprojects frozen M0 into LaViRA's episode-relative raster.
        # NORMAL navigation keeps its original local map; SG_NAV uses the
        # absolute world metric map through _sg_plan_world_path().
        return traversable, floor, frontiers

    def _snap_to_traversable(self, map_x, map_y, max_radius_cells=40):
        trav = getattr(self, "traversable", None)
        if trav is None:
            return int(map_x), int(map_y)
        return self._snap_to_mask(
            map_x,
            map_y,
            trav,
            max_radius_cells=max_radius_cells,
            mask_name="planner traversable",
        )

    def _snap_to_mask(self, map_x, map_y, mask, max_radius_cells=40, mask_name="traversable"):
        if mask is None:
            return int(map_x), int(map_y)
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            return int(map_x), int(map_y)
        h, w = mask.shape
        mx = int(max(0, min(map_x, w - 1)))
        my = int(max(0, min(map_y, h - 1)))
        if mask[my, mx]:
            return mx, my
        for radius in range(1, int(max_radius_cells) + 1):
            y0, y1 = max(0, my - radius), min(h, my + radius + 1)
            x0, x1 = max(0, mx - radius), min(w, mx + radius + 1)
            hits = np.argwhere(mask[y0:y1, x0:x1])
            if hits.size:
                cand_y = hits[:, 0] + y0
                cand_x = hits[:, 1] + x0
                d2 = (cand_y - my) ** 2 + (cand_x - mx) ** 2
                idx = int(np.argmin(d2))
                logger.info(
                    f"[INHerit-SG-FIRST] snapped target on {mask_name} ({mx},{my}) -> "
                    f"({int(cand_x[idx])},{int(cand_y[idx])}) r={radius}"
                )
                return int(cand_x[idx]), int(cand_y[idx])
        return mx, my

    @staticmethod
    def _is_mask_cell(mask, map_x, map_y):
        if mask is None:
            return False
        mask = np.asarray(mask)
        if mask.ndim != 2:
            return False
        h, w = mask.shape
        mx, my = int(map_x), int(map_y)
        return 0 <= mx < w and 0 <= my < h and bool(mask[my, mx])

    def _is_traversable_cell(self, map_x, map_y):
        trav = getattr(self, "traversable", None)
        return self._is_mask_cell(trav, map_x, map_y)

    def _save_sg_match_keyframe(self, target, step):
        observation = (target or {}).get("best_observation") or {}
        source_path = observation.get("rgb_path")
        if not source_path or not os.path.isfile(source_path):
            adapter = getattr(self, "inherit_sg", None)
            preload_dir = str(getattr(adapter, "preload_dir", "") or "")
            frame_idx = observation.get("frame_idx")
            if preload_dir and isinstance(frame_idx, int) and frame_idx >= 0:
                portable_path = os.path.join(
                    preload_dir, "frames", "rgb", f"{frame_idx:06d}.png"
                )
                if os.path.isfile(portable_path):
                    source_path = portable_path
                    observation["rgb_path"] = portable_path
        if not source_path or not os.path.isfile(source_path):
            logger.warning(f"[INHerit-SG-FIRST] matched keyframe is unavailable: {source_path}")
            return None

        episode_id = str(getattr(self, "current_episode_id", "unknown"))
        node_id = str((target or {}).get("uid") or (target or {}).get("node_id") or "object")
        frame_idx = int(observation.get("frame_idx", -1))
        out_dir = os.path.join(self.save_dir, "traj_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"sg_match_{episode_id}_{node_id}_frame{frame_idx:06d}.png",
        )

        with Image.open(source_path) as source:
            rendered = source.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        bbox = observation.get("bbox_xyxy")
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=5)
            label = f"{node_id}: {(target or {}).get('label', 'object')}"
            text_box = draw.textbbox((x1, max(0, y1 - 20)), label)
            draw.rectangle(text_box, fill=(255, 0, 0))
            draw.text((x1, max(0, y1 - 20)), label, fill=(255, 255, 255))
        rendered.save(out_path)
        logger.info(
            f"[INHerit-SG-FIRST] saved matched keyframe for step {step}: {out_path}"
        )
        return out_path

    def _sg_task_key(self, task_text):
        return " ".join(str(task_text or "").lower().split())

    def _sg_target_key(self, target_info):
        if not target_info:
            return None
        uid = target_info.get("uid") or target_info.get("node_id")
        if uid:
            return f"uid:{uid}"
        pos = target_info.get("position") or []
        if len(pos) >= 3:
            pos_key = ",".join(f"{float(v):.2f}" for v in pos[:3])
        else:
            pos_key = "no_pos"
        return f"{target_info.get('label', '')}:{pos_key}"

    def _sg_filter_blocked(self, results, task_text):
        if not results:
            return []
        blocked = self._sg_blocked_by_task.get(self._sg_task_key(task_text), set())
        if not blocked:
            return results
        return [item for item in results if self._sg_target_key(item) not in blocked]

    def _sg_block_target_for_task(self, target_info, task_text, reason):
        task_key = self._sg_task_key(task_text)
        target_key = self._sg_target_key(target_info)
        if not task_key or not target_key:
            return
        self._sg_blocked_by_task.setdefault(task_key, set()).add(target_key)
        logger.info(
            f"[INHerit-SG-FIRST] blocked target {target_key} for task='{task_key}' due to {reason}"
        )

    def _compact_sg_results(self, results):
        compact = []
        for item in results or []:
            compact.append(
                {
                    "uid": item.get("uid") or item.get("node_id"),
                    "label": item.get("label"),
                    "score": item.get("score"),
                    "object_score": item.get("object_score"),
                    "room_score": item.get("room_score"),
                    "room_label": item.get("room_label"),
                    "position": item.get("position"),
                    "waypoint": item.get("waypoint"),
                    "best_observation_score": item.get("best_observation_score"),
                    "waypoint_source": (item.get("waypoint") or {}).get("source"),
                    "target_map": (
                        item.get("target_map_x"),
                        item.get("target_map_y"),
                    ) if item.get("target_map_x") is not None else None,
                }
            )
        return compact

    def _write_sg_decision(self, step, payload):
        try:
            adapter = getattr(self, "inherit_sg", None)
            pipeline = getattr(adapter, "pipeline", None)
            out_dir = (
                getattr(pipeline, "run_dir", None)
                or getattr(adapter, "output_root", None)
                or os.environ.get("INHERIT_SG_OUTPUT_DIR")
            )
            if not out_dir:
                return
            os.makedirs(out_dir, exist_ok=True)
            row = {
                "episode_id": str(getattr(self, "current_episode_id", "")),
                "step": int(step) if step is not None else None,
            }
            row.update(payload or {})
            if (
                self.sg_stop_on_miss
                and str(row.get("decision", "")).startswith("miss_")
            ):
                self._sg_stop_on_miss_requested = True
            with open(os.path.join(out_dir, "sg_first_decisions.jsonl"), "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        default=self._sg_json_default,
                    )
                    + "\n"
                )
        except Exception as exc:
            logger.warning(f"[INHerit-SG-FIRST] failed to write decision log: {exc}")

    @staticmethod
    def _sg_json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, set):
            return sorted(value, key=str)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    @staticmethod
    def _env_bool(name, default):
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() not in {"0", "false", "no", "off"}


# The smooth-navigation experiment is an explicit opt-in overlay.  Installing
# it here (after the evaluator class is complete) keeps the registry bound to
# this warm-aware class while preserving the reviewed base controller as the
# fail-safe action source on every step.
if os.environ.get("LIFELONG_SG_SMOOTH_NAV_ENABLE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from .smooth_navigation.patch import (
        install_smooth_navigation as _install_smooth_navigation,
    )

    _install_smooth_navigation(
        INHeritSGZeroShotVlnEvaluatorMP,
        HabitatSimActions,
        logger,
    )


# Exploration is installed outside smooth navigation.  Directional-probe
# turns/forwards therefore remain exact exploration-owned actions; when the
# original SG target becomes reachable, its normal path controller still uses
# the already-installed smooth-navigation layer.  No legacy SG module is
# imported at runtime.
if os.environ.get("LIFELONG_SG_EXPLORATION_ENABLE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from .full_exploration import (
        install_goal_directed_exploration as _install_goal_directed_exploration,
    )

    _install_goal_directed_exploration(
        INHeritSGZeroShotVlnEvaluatorMP,
        HabitatSimActions,
        logger,
    )
