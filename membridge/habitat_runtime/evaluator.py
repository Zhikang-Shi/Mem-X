import copy
import gzip
import json
import math
import os
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict

import numpy as np
from PIL import Image
from fastdtw import fastdtw
from skimage.morphology import binary_closing
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

from habitat import logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

# Import Habitat visualization utilities

from .prompts import *  # VLN prompts re-exported as module-level for backward compat
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.env.env_utils import construct_envs
from vlnce_baselines.utils.misc import get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.utils.constant import base_classes, map_channels
from .utils.depth_utils import get_world_xz_from_pixel
from .utils.visualization import LaViRAVisualizer

import sys
# Dynamic path resolution for Docker/local compatibility
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

NAVDP_ROOT = os.path.join(PROJECT_ROOT, "navdp")
if NAVDP_ROOT not in sys.path:
    sys.path.append(NAVDP_ROOT)

# iplanner was removed from the codebase. The guards below leave downstream
# `if self.use_iplanner:` branches unreachable (default USE_IPLANNER=False).
IPlannerAgent = None

try:
    from policy_agent import NavDP_Agent
except ImportError as _e:
    logger.warning(f"NavDP import failed ({_e}); --use-navdp will fail loudly when used.")
    NavDP_Agent = None

import warnings

warnings.filterwarnings('ignore')

from .agent import VLMReasoningAgent



@baseline_registry.register_trainer(name="ZS-Evaluator-mp")
class ZeroShotVlnEvaluatorMP(BaseTrainer):
    def __init__(self, config, r2r, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self.r2r = r2r
        self.device = get_device(config.TORCH_GPU_ID)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.resolution = config.MAP.MAP_RESOLUTION
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)

        self.trans = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH),
                                             interpolation=Image.NEAREST)
                                         ])

        self.classes = []
        self.current_episode_id = None
        self.floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.collision_count = 0
        self.base_classes = copy.deepcopy(base_classes)


        self.visualize = True
        _rgb_base = getattr(config, 'RGB_SAVE_DIR', './saved_rgb_images')
        # Namespace by exp_name (the per-run timestamp appended to EVAL_CKPT_PATH_DIR
        # in run_mp) so concurrent tasks / re-runs never overwrite each other's frames.
        _exp_name = os.path.basename(config.EVAL_CKPT_PATH_DIR.rstrip('/'))
        self.save_dir = os.path.join(_rgb_base, _exp_name) if _exp_name else _rgb_base
        # When DEBUG_LOGGING is on, also forward the debug_log_dir to the visualizer
        # so per-step topdown_map.png + rgb_obs.png land alongside the LA/VA prompts.
        _debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        _debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs') if _debug_logging else None
        self.visualizer = LaViRAVisualizer(None, self.visualize, self.save_dir, self.width, self.height,
                                           debug_log_dir=_debug_log_dir)

        self.visited_targets = []  # List of targets the agent has identified/visited
        self.history_images = [] # Initialize history images
        self.current_step = 0  # Track current step for navigation decisions

        # Optional episode-chaining output.  Habitat vector environments may
        # auto-reset immediately after ``done``; therefore the next episode's
        # start must be captured from step outputs, never queried afterwards.
        self._chain_terminal_state_path = os.environ.get(
            "LIFELONG_CHAIN_TERMINAL_STATE_PATH", ""
        ).strip()
        self._chain_terminal_candidate = None
        self._chain_episode_identity = None

        # Distance thresholds for target management (in map units)
        # 0.75 meter threshold (75 cm / resolution)
        self.target_reached_threshold = getattr(config, 'TARGET_REACHED_THRESHOLD', 75.0 / self.resolution)
        # self.agent = VLMReasoningAgent(self.visualizer)
        
        # Statistics counters
        self.total_backtracks = 0
        self.total_waypoints = 0

        self.use_continuous_history = getattr(config.EVAL, 'USE_CONTINUOUS_HISTORY', True)
        self.history_interval = getattr(config.EVAL, 'HISTORY_INTERVAL', 2)
        
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.config.freeze()

        # Initialize agent with specified configuration
        # use_guideline=False, use_working_memory=False as requested
        # self.agent = VLMReasoningAgent(self.visualizer, use_guideline=False, use_working_memory=False)
        debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs')
        self.task_type = getattr(config, 'TASK_TYPE', 'VLN')
        self.use_todo_list = getattr(config.EVAL, 'USE_TODO_LIST', True)
        self.backtrack_second_chance = getattr(config.EVAL, 'BACKTRACK_SECOND_CHANCE', True)
        self.agent = VLMReasoningAgent(
            self.visualizer,
            task_type=self.task_type,
            use_guideline=True,
            use_working_memory=False,
            allow_move_behind=False,
            debug_logging=debug_logging,
            log_dir=debug_log_dir,
            use_todo_list=self.use_todo_list,
            backtrack_second_chance=self.backtrack_second_chance,
        )

        # The IPlanner backend was removed; this flag stays False so the inert
        # `if self.use_iplanner:` guards below never fire. Use FMM or NavDP instead.
        self.use_iplanner = False
        if self.use_iplanner:
            intrinsics = self._get_camera_intrinsics()
            intrinsics_tensor = torch.from_numpy(intrinsics).float().to(self.device)
            
            # Construct dynamic paths for iplanner model and config
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "iplanner", "plannernet.pt")
            config_path = os.path.join(project_root, "iplanner", "configs", "iplanner.yaml")
            
            logger.info(f"Loading iplanner from: {model_path} with config: {config_path}")
            
            self.iplanner_agent = IPlannerAgent(
                image_intrinsic=intrinsics_tensor,
                model_path=model_path,
                model_config_path=config_path,
                device=self.device
            )
            self.navdp_traj_global = None
            self.iplanner_update_freq = 5
            self.last_iplanner_step = -999


        # Initialize NavDP Agent
        self.use_navdp = getattr(config, 'USE_NAVDP', False)
        # logger.info(f"DEBUG: USE_NAVDP is set to {self.use_navdp}")
        if self.use_navdp:
            if NavDP_Agent is None:
                raise ImportError(
                    "USE_NAVDP=True but NavDP_Agent failed to import. "
                    "Likely missing `diffusers` (see navdp/requirements.txt). "
                    "Run: pip install diffusers==0.33.1"
                )
            intrinsics = self._get_camera_intrinsics()
            model_path = os.path.join(NAVDP_ROOT, "navdp-cross-modal.ckpt")
            # logger.info(f"Loading NavDP from: {model_path}")

            self.navdp_agent = NavDP_Agent(
                image_intrinsic=intrinsics,
                navi_model=model_path,
                device=self.device
            )
            self.navdp_agent.reset(batch_size=1, threshold=0.1)

        self.use_fmm = getattr(config, 'USE_FMM', True)
        
        # Validation Logic
        if self.use_navdp and self.use_iplanner:
             raise ValueError("Configuration Error: use_navdp and use_iplanner cannot both be True.")
        
        if not self.use_fmm and not self.use_navdp and not self.use_iplanner:
            raise ValueError("Configuration Error: use_fmm, use_navdp, and use_iplanner cannot all be False.")

    def _init_envs(self) -> None:
        # logger.info("start to initialize environments")

        # Check if we have specific allowed episodes loaded from assignment file
        specific_episodes = getattr(self, 'allowed_episodes', None)
        if specific_episodes:
            # self.allowed_episodes is a set of tuples (id, scene_id)
            specific_episodes = list(specific_episodes)

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
            specific_episodes_allowed=specific_episodes,
        )
        logger.info(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()
        # logger.info("initializing environments finished!")

    def _update_sg_world_to_map_cache(self, obs, current_pose):
        return

    def _try_sg_first_before_la(self, **kwargs):
        return None

    def _on_sg_target_reached(self, **kwargs):
        return

    def _on_sg_nav_aborted(self, **kwargs):
        return

    def _sg_nav_progress_abort_reason(self, step, dist_to_tgt, target_info=None):
        return None

    def _sg_nav_reached_reason(self, **kwargs):
        return None

    def _get_sg_path_action(self, **kwargs):
        return None

    def _collect_val_traj(self) -> None:
        if not self.r2r:
            role = self.config.TASK_CONFIG.DATASET.ROLES
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        
        gt_path = None
        if self.r2r:
            gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)
        else:
            gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split, role=role[0])
            
        self.gt_path = gt_path if os.path.exists(gt_path) else None
        if self.gt_path is None:
            logger.warning(f"GT data not found at {gt_path}, skipping NDTW calculation.")

    def _load_post_rollout_gt_entry(self, episode_id):
        """Load GT only after action selection has finished for the episode."""
        if not getattr(self, "gt_path", None):
            return None
        with gzip.open(self.gt_path) as handle:
            gt_data = json.load(handle)
        return gt_data.get(str(episode_id))

    def _calculate_metric(self, infos: List, submitted_answer: str = None, is_timeout: bool = False):
        """Dispatch metric calculation by task_type."""
        if self.task_type == "EQA":
            result = self._save_eqa_metric(
                infos, submitted_answer=submitted_answer, is_timeout=is_timeout
            )
        else:
            result = self._calculate_trajectory_metric(infos)
        self._write_chain_terminal_state(is_timeout=is_timeout)
        return result

    @staticmethod
    def _chain_heading_to_xyzw(heading_rad):
        """Convert the yaw-only Habitat heading sensor value to XYZW."""
        half = 0.5 * float(heading_rad)
        return [0.0, float(math.sin(half)), 0.0, float(math.cos(half))]

    @staticmethod
    def _chain_action_id(action):
        if isinstance(action, dict):
            action = action.get("action")
        try:
            return int(action)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _chain_action_name(action_id):
        for name in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            value = getattr(HabitatSimActions, name, None)
            if value is not None and action_id == int(value):
                return name.lower()
        return None if action_id is None else f"action_{action_id}"

    @staticmethod
    def _chain_canonical_scene_id(scene_id):
        value = str(scene_id).replace("\\", "/")
        marker = "data/scene_datasets/"
        if marker in value:
            return value.split(marker, 1)[1]
        return value.lstrip("/")

    def _begin_chain_terminal_capture(self, observation):
        if not self._chain_terminal_state_path:
            return
        episode = self.envs.current_episodes()[0]
        runtime_scene_id = str(episode.scene_id)
        self._chain_episode_identity = {
            "episode_id": str(episode.episode_id),
            "scene_id": self._chain_canonical_scene_id(runtime_scene_id),
            "runtime_scene_id": runtime_scene_id,
        }
        self._chain_terminal_candidate = None
        self._capture_chain_observation(
            observation,
            step=0,
            action=None,
            done=False,
            source="initial_observation",
        )

    def _capture_chain_observation(
        self, observation, step, action, done, source, position_override=None,
        heading_override=None,
    ):
        if not self._chain_terminal_state_path:
            return
        position_value = (
            position_override
            if position_override is not None
            else observation.get("position")
        )
        heading_value = (
            heading_override
            if heading_override is not None
            else observation.get("heading")
        )
        position = np.asarray(position_value, dtype=np.float64).reshape(-1)
        heading = np.asarray(heading_value, dtype=np.float64).reshape(-1)
        if (
            position.size < 3
            or heading.size < 1
            or not np.isfinite(position[:3]).all()
            or not np.isfinite(heading[0])
        ):
            raise RuntimeError("Cannot capture a finite Habitat terminal pose")
        action_id = self._chain_action_id(action)
        heading_rad = float(heading[0])
        self._chain_terminal_candidate = {
            "position": [float(value) for value in position[:3]],
            "rotation_xyzw": self._chain_heading_to_xyzw(heading_rad),
            "heading_rad": heading_rad,
            "step": int(step),
            "last_action": {
                "id": action_id,
                "name": self._chain_action_name(action_id),
            },
            "done": bool(done),
            "pose_source": str(source),
        }
        return self._chain_terminal_candidate

    def _capture_chain_step_output(self, observation, action, done, info, step):
        """Cache a post-action pose without trusting an auto-reset observation."""
        if not self._chain_terminal_state_path:
            return None
        if not done:
            self._capture_chain_observation(
                observation,
                step=step,
                action=action,
                done=False,
                source="post_action_observation",
            )
            return self._chain_terminal_candidate

        previous = self._chain_terminal_candidate
        if previous is None:
            raise RuntimeError("Episode ended before a chaining pose was captured")
        trajectory = ((info or {}).get("position") or {}).get("position")
        if trajectory is None:
            trajectory = []
        has_terminal_trajectory = len(trajectory) > 0
        final_position = trajectory[-1] if has_terminal_trajectory else previous["position"]
        returned_position = np.asarray(
            observation.get("position", []), dtype=np.float64
        ).reshape(-1)
        final_position_array = np.asarray(final_position, dtype=np.float64).reshape(-1)

        # A non-auto-reset output is authoritative, including its heading.  If
        # Habitat already reset, use the terminal trajectory position and apply
        # the final discrete turn to the last trusted heading.  MOVE/STOP do not
        # change yaw.
        returned_is_terminal = (
            returned_position.size >= 3
            and final_position_array.size >= 3
            and np.isfinite(returned_position[:3]).all()
            and np.linalg.norm(returned_position[:3] - final_position_array[:3]) <= 1e-4
        )
        if returned_is_terminal:
            self._capture_chain_observation(
                observation,
                step=step,
                action=action,
                done=True,
                source="terminal_post_action_observation",
                position_override=final_position_array[:3],
            )
            return self._chain_terminal_candidate

        heading = float(previous["heading_rad"])
        action_id = self._chain_action_id(action)
        turn_angle = math.radians(
            float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        )
        if action_id == int(HabitatSimActions.TURN_LEFT):
            heading += turn_angle
        elif action_id == int(HabitatSimActions.TURN_RIGHT):
            heading -= turn_angle
        heading = math.atan2(math.sin(heading), math.cos(heading))
        self._capture_chain_observation(
            observation,
            step=step,
            action=action,
            done=True,
            source=(
                "info_trajectory_terminal_with_action_heading"
                if has_terminal_trajectory
                else "pre_action_fallback_with_action_heading"
            ),
            position_override=final_position_array[:3],
            heading_override=[heading],
        )
        return self._chain_terminal_candidate

    def _write_chain_terminal_state(self, is_timeout=False):
        if not self._chain_terminal_state_path:
            return
        if self._chain_terminal_candidate is None or self._chain_episode_identity is None:
            raise RuntimeError("Terminal-state output requested without a captured pose")
        episode_id = self._chain_episode_identity["episode_id"]
        metrics = copy.deepcopy(getattr(self, "state_eps", {}).get(episode_id))
        if metrics is None:
            # Habitat sometimes preserves numeric episode IDs as dict keys.
            metrics = copy.deepcopy(
                getattr(self, "state_eps", {}).get(
                    int(episode_id) if episode_id.isdigit() else episode_id
                )
            )
        if not isinstance(metrics, dict):
            raise RuntimeError(f"No final metrics available for chained episode {episode_id}")
        candidate = copy.deepcopy(self._chain_terminal_candidate)
        last_action_name = (candidate.get("last_action") or {}).get("name")
        if is_timeout:
            termination_reason = "max_episode_steps"
        elif last_action_name == "stop":
            termination_reason = "agent_stop"
        elif candidate.get("done"):
            termination_reason = "environment_done"
        else:
            termination_reason = "rollout_ended"
        payload = {
            "schema": "lifelong_episode_terminal_state_v1",
            **self._chain_episode_identity,
            **candidate,
            "termination_reason": termination_reason,
            "is_timeout": bool(is_timeout),
            "metrics": metrics,
        }
        output = os.path.abspath(self._chain_terminal_state_path)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        temporary = os.path.join(
            os.path.dirname(output), f".{os.path.basename(output)}.tmp"
        )
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, output)
        logger.info(
            f"Wrote chained terminal state for episode {episode_id} to {output}: "
            f"position={payload['position']}, heading={payload['heading_rad']:.6f}"
        )

    def _save_eqa_metric(self, infos: List[Dict], submitted_answer: str = None, is_timeout: bool = False):
        if not infos:
            return
        info = infos[0]
        curr_eps = self.envs.current_episodes()
        ep_id = curr_eps[0].episode_id
        if ep_id in getattr(self, 'state_eps', {}):
            return

        question_text = curr_eps[0].question.question_text
        ground_truth_answer = curr_eps[0].question.answer_text
        predicted_answer = submitted_answer

        # Guarantee a 100% answer rate. If the agent reached this point without
        # submitting an answer (step-budget timeout: it never converged to a
        # confident STOP), force a best-guess Oracle-QA from the most recent
        # panorama so the episode is scored on its answer instead of auto-failed.
        if predicted_answer is None and self.visited_targets:
            try:
                pano = self.visited_targets[-1].get('panorama_frames', [])
                if pano:
                    predicted_answer = self.agent.query_llm_oracle(pano, self.instruction)
                    logger.info(f"[EQA forced-answer on timeout] ep {ep_id} -> {predicted_answer!r}")
            except Exception as e:
                logger.error(f"EQA forced-answer (timeout) failed: {e}")

        try:
            trajectory_list = info.get('position', {}).get('position', [])
            trajectory_json_string = json.dumps(trajectory_list)
        except Exception as e:
            logger.error(f"Failed to serialize trajectory: {e}")
            trajectory_json_string = "[]"

        metric = {
            'success': 0.0,
            'steps_taken': info.get('steps_taken', 0),
            'episode_id': ep_id,
            'question': question_text,
            'ground_truth': ground_truth_answer,
            'predicted_answer': predicted_answer if predicted_answer is not None else "TIMEOUT (No Answer)",
            'trajectory_json': trajectory_json_string,
        }

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.EVAL_CKPT_PATH_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )

        # Timeout only affects the recorded step count; the answer (forced
        # above if the agent never stopped) is still scored by exact match.
        # Only a genuinely missing answer scores 0.
        if is_timeout:
            metric['steps_taken'] = getattr(self, 'max_step', info.get('steps_taken', 0))

        if predicted_answer is not None:
            try:
                clean_gt = str(ground_truth_answer).strip()
                clean_pred = str(predicted_answer).strip()
                metric['success'] = 1.0 if clean_pred == clean_gt else 0.0
                logger.info(f"--- EQA JUDGMENT (Episode {ep_id}{' / TIMEOUT' if is_timeout else ''}) ---")
                logger.info(f"  Ground Truth    : {ground_truth_answer!r}")
                logger.info(f"  Submitted Answer: {predicted_answer!r}")
                logger.info(f"  SUCCESS : {metric['success']}")
            except Exception as e:
                logger.error(f"EQA judgement failure: {e}")
                metric['success'] = 0.0
        else:
            logger.info(f"--- Episode {ep_id} TIMEOUT, no answer available. Success: 0.0 ---")

        self.state_eps[ep_id] = metric
        try:
            with open(fname, "w") as f:
                json.dump(self.state_eps, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics to {fname}: {e}")

    def _calculate_trajectory_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id

        metric = {}
        metric['steps_taken'] = info.get('steps_taken', 0)
        metric['collision_count'] = int(self.collision_count)
        
        # Get success distance from config
        success_distance = getattr(self.config.TASK_CONFIG.TASK, 'SUCCESS_DISTANCE', 
                                 getattr(self.config.TASK_CONFIG.TASK.SUCCESS, 'SUCCESS_DISTANCE', 3.0))

        gt_entry = self._load_post_rollout_gt_entry(ep_id)
        if gt_entry is not None:
            gt_path = np.asarray(gt_entry['locations'], dtype=np.float64)
            if gt_path.ndim != 2 or len(gt_path) < 2:
                raise RuntimeError(f"Episode {ep_id} has an invalid post-rollout GT path")
            self.ref_path = gt_entry['locations']
            pred_path = np.array(info['position']['position'])
            distances = np.array(info['position']['distance'])
            if pred_path.ndim != 2 or len(pred_path) < 1 or len(distances) < 1:
                raise RuntimeError(f"Episode {ep_id} has an invalid predicted trajectory")
            if os.environ.get("LAVIRA_SAVE_DENSE_TRAJECTORY", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }:
                # Keep the raw rollout poses alongside the scalar metrics so
                # constrained waypoint metrics can be recomputed offline.  The
                # fixed metric-key allowlist ignores this field during normal
                # aggregate reporting.
                metric["trajectory"] = pred_path.tolist()
            gt_length = distances[0]
            dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
            
            metric['distance_to_goal'] = distances[-1]
            metric['success'] = 1. if distances[-1] <= success_distance else 0.
            metric['oracle_success'] = 1. if (distances <= success_distance).any() else 0.
            metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
            metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
            metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
            metric['sdtw'] = metric['ndtw'] * metric['success']
        else:
            if os.environ.get("LAVIRA_METRICS_FAIL_CLOSED", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }:
                raise RuntimeError(f"No post-rollout GT entry for episode {ep_id}")
            # Use environment provided metrics if available
            metric['distance_to_goal'] = info.get('distance_to_goal', 0.0)
            metric['success'] = info.get('success', 0.0)
            metric['oracle_success'] = info.get('oracle_success', 0.0)
            metric['path_length'] = info.get('path_length', 0.0)
            metric['spl'] = info.get('spl', 0.0)
            metric['ndtw'] = info.get('ndtw', 0.0)
            metric['sdtw'] = info.get('sdtw', 0.0)
            
        self.state_eps[ep_id] = metric
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        logger.info(f'ep{ep_id}:{self.state_eps[ep_id]}')

        # Visualize trajectory
        self.draw_agent_pos_and_ref_path(
            success=bool(metric['success']),
            success_distance=float(success_distance),
        )

        # Rename the episode directory with success status
        try:
            is_success = bool(metric['success'])
            status_suffix = "_success" if is_success else "_failed"
            
            # The current save directory for this episode
            current_dir = os.path.join(self.save_dir, str(ep_id))
            
            if os.path.exists(current_dir):
                # The new directory name
                new_dir = os.path.join(self.save_dir, f"{ep_id}{status_suffix}")
                
                if os.path.exists(new_dir):
                    import shutil
                    shutil.rmtree(new_dir)
                
                os.rename(current_dir, new_dir)
                logger.info(f"Renamed episode directory to: {new_dir}")
        except Exception as e:
            logger.info(f"Error renaming episode directory: {e}")

    def _initialize_policy(self) -> None:
        # logger.info("start to initialize policy")
        self.segment_module = GroundedSAM(self.config, self.device)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        self.visualizer.update_map(self.mapping_module)

        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()

    def _concat_obs(self, obs: Observations) -> np.ndarray:
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)  # (h, w, c)->(c, h, w)

        return state

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)  # [3, h, w]
        rgb = rgb[:, :, ::-1]  # RGB to BGR
        depth = state[:, :, 3:4]  # [1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH

        sem_seg_pred = self._get_sem_pred(rgb)  # [num_detected_classes, h, w]
        depth = self._preprocess_depth(depth, min_depth, max_depth)  # [1, h, w]

        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        ds = env_frame_width // self.map_args.FRAME_WIDTH  # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8)))  # resize
            depth = depth[ds // 2::ds, ds // 2::ds]  # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)  # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)  # (4+num_detected_classes, h, w)

        return state

    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        cls2 = self.classes.copy()
        if self.agent.stair:
            cls2 = ["stairs"]
        elif self.current_step <= 12:
            cls2.append('stairs')
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=cls2)
        if self.visualize:
            pass
            # save_path = os.path.join(self.save_dir, str(self.current_episode_id), f'step{self.current_step}_mask.png')
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cv2.imwrite(save_path, annotated_images)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        # logger.info("current step detected classes (before filtering): ", labels)

        class_names = self._process_labels(labels)
        masks = self._process_masks(masks, class_names)

        return masks.transpose(1, 2, 0)

    def _process_labels(self, labels: List[str]) -> List:
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)

        return class_names

    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        if masks.shape[0] > 0:  # Check if there are any masks
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx)  # dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)

            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]

            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))

        return final_masks

    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        # logger.info('max:',depth.max())
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99  # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 1.0  # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0

        return depth

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)

        return state  # state.shape=(c,h,w)

    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                                 [(0, max_channels - state.shape[0]),
                                  (0, 0),
                                  (0, 0)],
                                 mode='constant')
                          for state in n_states], axis=0)

        return torch.from_numpy(batch).to(self.device)




    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        # logger.info(f'{navigable_index}, {not_navigable_index}')
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)

        obstacles = one_step_full_map[0, ...].astype(bool)
        explored_area = one_step_full_map[1, ...].astype(bool)

        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=disk(kernel_size))

        return floor

    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int = 3) -> tuple:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)

        obstacles = full_map[0, ...].astype(bool)
        explored_area = full_map[1, ...].astype(bool)

        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)

        selem = disk(3)
        obstacles_closed = binary_closing(obstacles, footprint=selem)
        objects_closed = binary_closing(objects, footprint=selem)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=selem)

        untraversable = np.logical_or(objects_closed, obstacles_closed)
        # ensure stairs override untraversable
        untraversable[navigable_closed == 1] = 0
        # untraversable[stairs_mask == 1] = 0
        untraversable = remove_small_objects(untraversable, min_size=64)
        untraversable = binary_closing(untraversable, footprint=disk(3))
        traversable = np.logical_not(untraversable)

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=selem)
        traversable = np.logical_or(floor, traversable)

        explored_area = binary_closing(explored_area, footprint=selem)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversable, floor, frontiers.astype(np.uint8)

    def _maps_initialization(self):
        obs = self.envs.reset()  # type(obs): list
        
        # Handle instruction extraction for different tasks (VLN / ObjectNav / EQA)
        current_episode = self.envs.current_episodes()[0]
        if self.task_type == "EQA" and hasattr(current_episode, 'question'):
            self.instruction = current_episode.question.question_text
        elif 'instruction' in obs[0]:
            self.instruction = obs[0]['instruction']['text']
        elif hasattr(current_episode, 'object_category'):
            self.instruction = f"Find the {current_episode.object_category}"
        elif 'objectgoal' in obs[0]:
            self.instruction = f"Find the object with ID {obs[0]['objectgoal'][0]}"
        else:
            self.instruction = "Explore the environment"
            
        self.destination = "goal"
        self.classes = self.base_classes.copy()
        self.current_episode_id = self.envs.current_episodes()[0].episode_id

        self.visualizer._save_rgb_frame(obs[0], 0, None, self.current_episode_id)

        # Initialize trajectory tracking
        self.pos_list = []
        # GT is intentionally unavailable until post-rollout scoring.
        self.ref_path = []

        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses, self.current_step)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)

        return obs, full_pose



    def reset(self) -> None:
        self.classes = []
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.collision_count = 0
        self.base_classes = copy.deepcopy(base_classes)

        # Reset target tracking
        self.visited_targets = []
        self.history_images = []
        self.current_step = 0
        self.backtrack_steps = 0
        self.last_submitted_answer = None  # EQA: set when agent submits final answer
        self._eqa_done_requested = False  # EQA: set True when Oracle-QA fires (STOP alone doesn't end EQA)

        self.policy.reset()
        self.mapping_module.reset()
        self.agent.reset()
        # self.agent.model.reset_stats() # Do not reset stats between episodes!

    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix for depth projection"""
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        width = self.width
        height = self.height
        vfov = 2 * np.arctan(height / width * np.tan(hfov / 2))

        fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
        fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
        cx = width / 2
        cy = height / 2

        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
        return intrinsics

    def get_panorama(self, obs: Observations, step: int):
        """Collect a panorama by turning left in 12 steps of 30 degrees each."""

        panorama_frames = []

        for turn_step in range(1, 12 + 1):
            turn_action = [{"action": HabitatSimActions.TURN_LEFT}]  # 30 deg per step
            turn_outputs = self.envs.step(turn_action)
            turn_obs, _, turn_dones, turn_infos = [list(x) for x in zip(*turn_outputs)]

            if turn_dones[0]:
                # Return signal that episode is done - caller should handle this
                return {'turn_direction': 'episode_done', 'episode_finished': True}

            panorama_frames.append({
                'rgb': turn_obs[0]['rgb'].copy(),
                'depth': turn_obs[0]['depth'].copy(),
                'angle': turn_step * 30 % 360,
                'step': turn_step
            })

            # Update map state for consistency (not persisted).
            batch_obs = self._batch_obs(turn_obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in turn_obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses, self.current_step)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step + turn_step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
        panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]

        return panorama_frames[::3]

    def draw_agent_pos_and_ref_path(
        self,
        success: bool = False,
        success_distance: float = 3.0,
    ):
        """
        Visualize the agent and post-rollout reference paths using Matplotlib.

        The reference path is populated only after action selection has finished.
        Deriving the plotted goal from its last point keeps this diagnostic code
        from reading Habitat episode goals during policy execution.
        """
        if not getattr(self, "visualize", False):
            return None

        pos_list = getattr(self, "pos_list", None) or []
        ref_path = getattr(self, "ref_path", None) or []
        
        # Extract data for plotting
        # Habitat coordinates: Y is up. We visualize X-Z plane (Top-down).
        agent_x = [p[0] for p in pos_list]
        agent_z = [p[2] for p in pos_list]
        
        ref_x = [p[0] for p in ref_path]
        ref_z = [p[2] for p in ref_path]
        
        episode_goal_positions = [ref_path[-1]] if ref_path else []
        radius_list = [
            (ref_path[-1], success_distance)
        ] if ref_path else []

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # A goal position may be a navigable point snapped from an object center.
        for obj in episode_goal_positions:
            plt.scatter(obj[0], obj[2], c='red', s=200, marker='*', label='Episode Goal')
            
        # Plot Radius zones
        for center, rad in radius_list:
             circle = plt.Circle((center[0], center[2]), rad, color='green', fill=True, alpha=0.2)
             plt.gca().add_patch(circle)

        # Plot Reference Path
        if ref_x:
            plt.plot(ref_x, ref_z, c='green', linewidth=2, label='Reference Path', linestyle='-')
            # Start and End of Reference Path
            plt.scatter(ref_x[0], ref_z[0], c='orange', s=100, marker='o', label='Ref Start')
            plt.scatter(ref_x[-1], ref_z[-1], c='blue', s=100, marker='X', label='Ref Goal')

        # Plot Agent Trajectory
        if agent_x:
            plt.plot(agent_x, agent_z, c='red', linewidth=2, label='Agent Path', linestyle='-')
            if agent_x:
                plt.scatter(agent_x[0], agent_z[0], c='blue', s=100, marker='^', label='Agent Start')
                plt.scatter(agent_x[-1], agent_z[-1], c='red', s=80, marker='v', label='Agent End')

        plt.title(f'Episode {getattr(self, "current_episode_id", "unknown")} Trajectory')
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters) - Top Down View')
        
        # Avoid duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        episode_id = str(getattr(self, "current_episode_id", "unknown"))
        # out_dir = os.path.join(self.save_dir, episode_id)
        # Modified to save to traj_viz directory
        out_dir = os.path.join(self.save_dir, "traj_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"traj_viz_success_{success}_{episode_id}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved trajectory visualization to: {out_path}")

        return out_path


    def rollout(self):
        """
        Execute a whole episode using bounding box target navigation
        """
        if self.use_navdp:
            self.navdp_agent.reset_env(0)

        obs, full_pose = self._maps_initialization()
        self._begin_chain_terminal_capture(obs[0])

        # Check allowed episodes
        if getattr(self, 'allowed_episodes', None) is not None:
             current_ep = self.envs.current_episodes()[0]
             
             # Robust matching for (id, scene_id) to handle path differences
             is_allowed = False
             for allowed_id, allowed_scene_id in self.allowed_episodes:
                 if str(current_ep.episode_id) == allowed_id:
                     # Check Scene match (basename or suffix)
                     # allowed_scene_id is usually relative, current_ep.scene_id is absolute
                     if current_ep.scene_id.endswith(allowed_scene_id) or \
                        os.path.basename(current_ep.scene_id) == os.path.basename(allowed_scene_id):
                         is_allowed = True
                         break
             
             if not is_allowed:
                 ep_info = f"{current_ep.episode_id} ({os.path.basename(current_ep.scene_id)})"
                 logger.info(f"Skipping episode {ep_info} (Not in assignment)")
                 return



        dones = [False] * self.config.NUM_ENVIRONMENTS
        infos = [{}] * self.config.NUM_ENVIRONMENTS


        # --- Initialization ---
        self._action = None
        action_list = []
        going_to_stop = False
        panorama_got = False
        navigate_or_not = False
        collided = 0
        search_destination = False
        current_pose = full_pose[0] if full_pose is not None else None

        self.latest_la_output = {}

        target_map_x, target_map_y = None, None
        is_backtracking = False  # Flag to indicate if current target is a backtrack target
        backtracking_only = False  # True when we first return to an old waypoint before re-deciding.
        current_la_action = 'NAVIGATE'  # Track LA model's decision ('NAVIGATE' or 'STOP')
        waypoint = None

        # New: Track consecutive stop failures
        self.stop_feedback = ""
        self.consecutive_stop_failures = 0

        # Target step tracking: re-navigate if the agent has been pursuing the same
        # target for more than max_steps_to_target steps (timeout escape hatch).
        max_steps_to_target = 15
        target_set_step = None
        navigation_mode = "NORMAL"
        sg_target_info = None

        full_map = self.mapping_module.get_full_map()
        step = 0
        action_step = 0
        last_step = action_step
        while step < self.max_step:
            pos = self.envs.call_at(0,'_env')._sim.get_agent_state().position
            self.pos_list.append(pos)
            if self._action != 1:
                last_step = action_step
            # =================================================================
            # 1. ANALYZE STATE for step N
            #    (consume the result of the previous step's action)
            # =================================================================
            if dones[0]:
                self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None))
                return
            self.visualizer.instruction = self.instruction
            self.visualizer.destination = self.destination
            self.visualizer._action = self._action

            logger.info(f"\nepisode:{self.current_episode_id}, step:{step}")

            gray_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)

            # Update pose / position.
            last_pose = current_pose
            current_pose = full_pose[0]
            self.current_step = step
            self.visualizer.sync(step, self.current_episode_id)

            position = current_pose[:2] * 100 / self.resolution
            agent_map_x, agent_map_y = int(position[0]), int(position[1])
            self._update_sg_world_to_map_cache(obs[0], current_pose)

            # Prepare iplanner trajectory for visualization, if available.
            navdp_traj_list = None
            if getattr(self, 'navdp_traj_global', None) is not None:
                navdp_traj_list = [(pt[0], pt[1]) for pt in self.navdp_traj_global]

            self.visualizer._save_rgb_frame(obs[0], step, self.visited_targets, self.current_episode_id, (target_map_x, target_map_y), todo_list=self.agent._format_todo_for_prompt() if self.agent.use_todo_list else None, la_output=self.latest_la_output, navdp_traj=navdp_traj_list)

            # Save continuous history images
            if self.use_continuous_history and step % self.history_interval == 0:
                rgb_to_save = obs[0]['rgb'].copy()
                if isinstance(rgb_to_save, np.ndarray):
                    if rgb_to_save.dtype != np.uint8:
                        rgb_to_save = (rgb_to_save * 255).astype(np.uint8)
                    img_save = Image.fromarray(rgb_to_save)
                else:
                    img_save = rgb_to_save
                
                self.history_images.append({'step': step, 'image': img_save})

            # =================================================================
            # 2. PLAN/DECIDE for step N (4-step navigation cycle):
            #    Step 1: no target  -> spin to collect panorama
            #    Step 2: panorama done -> query LLM for the next target
            #    Step 3: have target -> navigate toward it
            #    Step 4: target reached or timed out -> reset and go back to Step 1
            # =================================================================

            # High-level decisions only happen when action_list is empty.
            if not action_list:
                # Target timeout check.
                if (
                    navigation_mode != "SG_NAV"
                    and target_map_x is not None
                    and target_map_y is not None
                    and target_set_step is not None
                ):
                    steps_since_target_set = step - target_set_step
                    if steps_since_target_set >= max_steps_to_target:
                        if backtracking_only:
                            logger.info("Backtrack-only return timed out, resetting navigation state.")

                        if len(self.visited_targets) > 0:
                            self.visited_targets.pop()

                        # Step 4: reset state, prepare to restart.
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        backtracking_only = False
                        sg_target_info = None
                        navigation_mode = "NORMAL"

                # Distance-to-target check.
                if target_map_x is not None and target_map_y is not None:
                    world_target_xz = (
                        (sg_target_info or {}).get("target_world_xz")
                        if navigation_mode == "SG_NAV"
                        else None
                    )
                    if world_target_xz is not None:
                        current_world_xz = np.asarray(
                            obs[0]["position"], dtype=np.float64
                        ).reshape(-1)[[0, 2]]
                        world_resolution_m = float(
                            sg_target_info["world_map_resolution_m"]
                        )
                        distance_to_target = float(
                            np.linalg.norm(
                                current_world_xz
                                - np.asarray(world_target_xz, dtype=np.float64)
                            )
                            / world_resolution_m
                        )
                    else:
                        distance_to_target = np.sqrt((target_map_x - agent_map_x) ** 2 + (target_map_y - agent_map_y) ** 2)

                    # SG targets are already snapped to a specific M0 observed-free
                    # cell. The subclass may accept a tightly bounded practical
                    # arrival when discrete Habitat dynamics cannot land on the
                    # exact integer cell.
                    sg_reach_reason = None
                    if navigation_mode == "SG_NAV":
                        sg_reach_reason = self._sg_nav_reached_reason(
                            step=step,
                            distance_to_target=distance_to_target,
                            agent_map_x=agent_map_x,
                            agent_map_y=agent_map_y,
                            current_pose=current_pose,
                            current_world_position=obs[0]["position"],
                            current_world_heading=obs[0]["heading"],
                            target_info=sg_target_info,
                        )
                        target_reached = sg_reach_reason is not None
                    else:
                        target_reached = distance_to_target < self.target_reached_threshold

                    if target_reached:
                        reached_actions = None
                        if backtracking_only:
                            logger.info("Reached backtrack waypoint. Returning to normal decision flow.")
                        if navigation_mode == "SG_NAV":
                            reached_actions = self._on_sg_target_reached(
                                step=step,
                                target_info=sg_target_info,
                                distance_to_target=distance_to_target,
                                current_world_position=np.asarray(
                                    obs[0]["position"], dtype=np.float64
                                ).reshape(-1).tolist(),
                                current_heading=float(np.asarray(obs[0]["heading"]).reshape(-1)[0]),
                                reach_reason=sg_reach_reason,
                            )

                        # Deduplicate waypoints that ended up too close to the just-reached one.
                        if len(self.visited_targets) > 0:
                            dist_calc = lambda target: np.sqrt(
                                (target['world_coords'][0] - self.visited_targets[-1]['world_coords'][0]) ** 2 + (
                                            target['world_coords'][1] - self.visited_targets[-1]['world_coords'][
                                        1]) ** 2) if 'world_coords' in target else float('inf')
                            for target in self.visited_targets[:-1]:
                                if dist_calc(target) < self.target_reached_threshold:
                                    self.visited_targets.pop()
                                    break

                        # Step 4: reset state, prepare for next target.
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        is_backtracking = False
                        backtracking_only = False
                        current_la_action = 'NAVIGATE'
                        sg_target_info = None
                        navigation_mode = "NORMAL"
                        if reached_actions:
                            action_list.extend(reached_actions)
                            continue

                # Step 1: no target and not spinning yet -> spin to collect panorama and decide direction.
                if target_map_x is None and not panorama_got and going_to_stop:
                    # Double-check stop: acquire a fresh panorama and verify with the LA model.
                    panorama_frames = self.get_panorama(obs[0], step)
                    step += 12 # get_panorama takes 12 steps
                    
                    if 'episode_finished' in panorama_frames:
                        break
                        
                    should_stop, stop_response = self.agent.double_check_stop(self.instruction, panorama_frames, self.visited_targets, episode_id=self.current_episode_id, step=step, history_images=self.history_images)
                    self.latest_la_output = stop_response
                    
                    if should_stop:
                        action_list.append(0)  # STOP action
                        self.consecutive_stop_failures = 0 # Success
                        if self.task_type == "EQA":
                            # EQA: STOP doesn't terminate the episode in habitat
                            # (ANSWER is a separate action). Oracle-QA once, mark
                            # done, let the post-step block exit the rollout.
                            try:
                                self.last_submitted_answer = self.agent.query_llm_oracle(panorama_frames, self.instruction)
                            except Exception as e:
                                logger.error(f"EQA oracle failed: {e}")
                                self.last_submitted_answer = None
                            self._eqa_done_requested = True
                    else:
                        going_to_stop = False
                        navigate_or_not = False
                        panorama_got = False # Ensure we re-enter Step 1 logic properly
                        current_la_action = 'NAVIGATE' # Ensure we default back to navigate
                        
                        # Handle rejection feedback
                        self.consecutive_stop_failures += 1
                        self.stop_feedback = f"System: Your previous decision to STOP was rejected by the double-check mechanism because the target was not clearly visible or not close enough (Failure count: {self.consecutive_stop_failures}). Please navigate closer to the target or verify your position."
                        
                        if self.consecutive_stop_failures >= 3:
                            logger.info(f"Forcing STOP after {self.consecutive_stop_failures} consecutive double-check rejections.")
                            action_list.append(0) # STOP action
                            # Reset flags to prevent loop
                            self.stop_feedback = ""
                            self.consecutive_stop_failures = 0
                            if self.task_type == "EQA":
                                try:
                                    self.last_submitted_answer = self.agent.query_llm_oracle(panorama_frames, self.instruction)
                                except Exception as e:
                                    logger.error(f"EQA oracle failed: {e}")
                                    self.last_submitted_answer = None
                                self._eqa_done_requested = True
                        
                elif target_map_x is None and not navigate_or_not:
                    # Step 1: collect panorama and decide direction.
                    current_rgb = obs[0]['rgb'].copy()

                    panorama_frames = self.get_panorama(obs[0], step)
                    step += 12 # get_panorama takes 12 steps

                    if 'episode_finished' in panorama_frames:
                        break

                    # Create a new waypoint anchored at the current map position.
                    waypoint_id = len(self.visited_targets)
                    self.visited_targets.append({
                        'step': step,
                        'init_image': Image.fromarray(current_rgb) if isinstance(current_rgb,
                                                                                 np.ndarray) else current_rgb,
                        'panorama_frames': panorama_frames,
                        'world_coords': (agent_map_x, agent_map_y),
                        'full_pose': current_pose  # full (x, y, heading)
                    })
                    self.total_waypoints += 1

                    self.visualizer._save_waypoint_panorama_rgb(panorama_frames, waypoint_id, step)

                    sg_decision = self._try_sg_first_before_la(
                        instruction=self.instruction,
                        step=step,
                        obs=obs[0],
                        current_pose=current_pose,
                        agent_map_x=agent_map_x,
                        agent_map_y=agent_map_y,
                        visited_targets=self.visited_targets,
                        target_reached_threshold=self.target_reached_threshold,
                    )
                    if getattr(self, "_sg_stop_on_miss_requested", False):
                        logger.info(
                            "[INHerit-SG-FIRST] SG miss in navigation-only validation; "
                            "stopping episode before LA fallback."
                        )
                        self._sg_stop_on_miss_requested = False
                        write_decision = getattr(self, "_write_sg_decision", None)
                        if callable(write_decision):
                            write_decision(
                                step,
                                {
                                    "decision": "stop_on_sg_miss",
                                    "terminal_reason": "navigation_only_validation",
                                },
                            )
                        self._action = 0
                        outputs = self.envs.step([{"action": self._action}])
                        step += 1
                        obs, _, dones, infos = [list(x) for x in zip(*outputs)]
                        continue
                    if sg_decision:
                        target_map_x = int(sg_decision["target_map_x"])
                        target_map_y = int(sg_decision["target_map_y"])
                        target_set_step = step
                        sg_target_info = sg_decision.get("target_info")
                        navigation_mode = "SG_NAV"
                        self.latest_la_output = {
                            "action": "SG_NAV",
                            "reasoning": "INHerit-SG two-stage retrieval hit before LA; skipped LA.",
                            "sg_target": sg_target_info,
                        }
                        if self.visited_targets:
                            self.visited_targets[-1].update({
                                "progress_analysis": "INHerit-SG first hit",
                                "reasoning": "Skipped LA and navigated to SG target.",
                                "direction_decision": "sg_nav",
                            })
                        panorama_got = False
                        navigate_or_not = False
                        is_backtracking = False
                        backtracking_only = False
                        current_la_action = "NAVIGATE"
                        logger.info(
                            f"[INHerit-SG-FIRST] accepted target "
                            f"({target_map_x}, {target_map_y}) before LA; skipping LA."
                        )
                        continue

                    # Combine stop feedback and todo verification feedback
                    combined_feedback = self.stop_feedback
                    todo_feedback = getattr(self.agent, 'todo_verification_feedback', "")
                    if todo_feedback:
                        combined_feedback = (combined_feedback + "\n" + todo_feedback).strip()
                        # Consume the feedback (don't repeat it forever)
                        self.agent.todo_verification_feedback = ""

                    decision = self.agent.navigate_or_backtrack(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        feedback=combined_feedback,
                        episode_id=self.current_episode_id,
                        step=step,
                        history_images=self.history_images if self.use_continuous_history else None
                    )
                    self.latest_la_output = decision

                    handoff = getattr(self, "_sg_level3_lavira_fallback", None)
                    if handoff is not None:
                        self._write_sg_decision(step, {
                            "decision": "level3_lavira_fallback_decision",
                            "ordinary_vln_executed": True,
                            "fallback_reason": handoff["reason"],
                            "lavira_action": decision.get("action"),
                            "history_image_count": len(self.history_images),
                            "visited_waypoint_count": len(self.visited_targets),
                            "todo_list": self.agent.todo_list,
                        })

                    # If LA decides to NAVIGATE (and not STOP), clear the feedback
                    if decision.get('action') == 'NAVIGATE' or decision.get('action') == 'BACKTRACK':
                        self.stop_feedback = ""
                        # If we had stop failures but now LA decided to move, we can reset the counter
                        # self.consecutive_stop_failures = 0 # Optional: reset or keep counting? Reset seems fair.
                        if decision.get('action') == 'NAVIGATE':
                             # Only reset if it's a new navigation decision, backtrack might still be "stuck" logic
                             pass

                    if decision.get('action') == 'STOP':
                        current_la_action = 'STOP'

                    if decision.get('action', 'NAVIGATE') == 'BACKTRACK':
                        self.total_backtracks += 1
                        target_waypoint_id = decision.get('waypoint', 0)
                        if isinstance(target_waypoint_id, int) and target_waypoint_id < len(self.visited_targets) - 1:
                            visited_targets_up_to_backtrack = self.visited_targets[:target_waypoint_id + 1]
                            failed_path = self.visited_targets[target_waypoint_id + 1:]
                            backtrack_point = visited_targets_up_to_backtrack[-1]

                            if self.agent.backtrack_second_chance:
                                logger.info(f"Initiating Backtrack Re-planning to Waypoint {target_waypoint_id}")

                                # 2. Replan
                                new_direction, replan_response = self.agent.replan_at_backtrack(
                                    self.instruction,
                                    visited_targets_up_to_backtrack,
                                    failed_path,
                                    episode_id=self.current_episode_id,
                                    step=step,
                                    history_images=self.history_images if self.use_continuous_history else None
                                )
                                self.latest_la_output = replan_response
                                logger.info(f"Re-planned direction: {new_direction}")

                                # 3. Get corresponding image from backtrack point
                                panorama_frames = backtrack_point['panorama_frames']
                                direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                                target_angle = direction_map.get(new_direction, 0)
                                frame_idx = target_angle // 90

                                if frame_idx < len(panorama_frames):
                                    # 4. Get BBox from VA model
                                    dir_rgb = panorama_frames[frame_idx]['rgb']
                                    dir_depth = panorama_frames[frame_idx]['depth']

                                    bbox = self.agent.query_llm(
                                        instruction=self.instruction,
                                        visited_targets=visited_targets_up_to_backtrack,
                                        rgb_image=dir_rgb,
                                        depth_image=dir_depth,
                                        width=self.width,
                                        height=self.height,
                                        current_step=self.current_step,
                                        progress_analysis="Backtracking and Replanning",
                                        episode_id=self.current_episode_id
                                    )

                                    bbox_save_path = os.path.join(self.save_dir, str(self.current_episode_id), f'bbox_step{step}.png')
                                    os.makedirs(os.path.dirname(bbox_save_path), exist_ok=True)

                                    d_img = dir_depth
                                    if len(d_img.shape) == 3:
                                        d_img = d_img[:, :, 0]

                                    dummy_depth = np.expand_dims(d_img, axis=2) # (H,W,1)
                                    processed_depth = self._preprocess_depth(dummy_depth, 0.1, 5.0) / 100.0

                                    if not self.agent.stair:
                                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), min(int(bbox.get('y2', 0)), self.height - 1))
                                    else:
                                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y1', 0)))

                                    base_pose = backtrack_point.get('full_pose')
                                    if base_pose is None:
                                        target_map_x, target_map_y = backtrack_point['world_coords']
                                    else:
                                        frame_angle = panorama_frames[frame_idx]['angle']

                                        if isinstance(base_pose, torch.Tensor):
                                            view_pose = base_pose.clone().cpu().numpy()
                                        else:
                                            view_pose = base_pose.copy()

                                        if view_pose.ndim == 2:
                                            view_pose = view_pose[0]

                                        view_pose[2] += frame_angle
                                        view_pose[2] = view_pose[2] % 360

                                        target = get_world_xz_from_pixel(
                                            pixel_coords=coords,
                                            depth_image=processed_depth,
                                            full_pose=view_pose,
                                            camera_intrinsics=self._get_camera_intrinsics(),
                                        )

                                        new_target_x = int(target[0] * 100.0 / self.resolution)
                                        new_target_y = int(target[1] * 100.0 / self.resolution)
                                        new_target_x = max(0, min(new_target_x, self.map_shape[0] - 1))
                                        new_target_y = max(0, min(new_target_y, self.map_shape[1] - 1))

                                        target_map_x, target_map_y = new_target_x, new_target_y

                                revisit_waypoint = {
                                    'step': step,
                                    'init_image': backtrack_point['init_image'],
                                    'panorama_frames': backtrack_point['panorama_frames'],
                                    'world_coords': backtrack_point['world_coords'],
                                    'full_pose': backtrack_point['full_pose'],
                                    'is_revisit': True,
                                    'original_waypoint_id': target_waypoint_id,
                                    'backtrack_from_waypoint_id': len(self.visited_targets) - 1
                                }

                                self.visited_targets.append(revisit_waypoint)
                                self.total_waypoints += 1

                                self.visited_targets[-1]['direction_decision'] = new_direction
                                self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb, np.ndarray) else dir_rgb
                                self.visited_targets[-1]['bbox'] = bbox
                                self.visited_targets[-1]['progress_analysis'] = "Backtracking Re-plan"

                                is_backtracking = False # Treat as new target
                                backtracking_only = False
                            else:
                                logger.info(f"Initiating Backtrack-only return to Waypoint {target_waypoint_id}")
                                if backtrack_point.get('world_coords') is not None:
                                    target_map_x, target_map_y = backtrack_point['world_coords']
                                else:
                                    decision['action'] = 'NAVIGATE'
                                    panorama_got = True
                                    continue

                                revisit_waypoint = {
                                    'step': step,
                                    'init_image': backtrack_point['init_image'],
                                    'panorama_frames': backtrack_point['panorama_frames'],
                                    'world_coords': backtrack_point['world_coords'],
                                    'full_pose': backtrack_point['full_pose'],
                                    'is_revisit': True,
                                    'original_waypoint_id': target_waypoint_id,
                                    'backtrack_from_waypoint_id': len(self.visited_targets) - 1,
                                    'progress_analysis': "Backtracking Only"
                                }

                                self.visited_targets.append(revisit_waypoint)
                                self.total_waypoints += 1
                                is_backtracking = True
                                backtracking_only = True

                            target_set_step = step

                        else:
                            # logger.info("Invalid waypoint ID for backtrack, continuing with navigation")
                            decision['action'] = 'NAVIGATE'
                        panorama_got = True
                    if decision.get('action', 'NAVIGATE') == 'NAVIGATE':
                        navigate_or_not = True
                        direction = decision.get('direction', 'forward')

                        if decision.get('stop_signal', False):
                            current_la_action = 'STOP'
                        else:
                            current_la_action = 'NAVIGATE'

                        progress_analysis = decision.get('progress_analysis', '')
                        reasoning = decision.get('reasoning', '')

                        self.visited_targets[-1].update({
                            'progress_analysis': progress_analysis,
                            'reasoning': reasoning,
                            'direction_decision': direction
                        })

                        # Pull the panorama frame matching the chosen direction for visualization.
                        direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                        target_angle = direction_map.get(direction, 0)
                        frame_idx = target_angle // 90

                        if frame_idx < len(panorama_frames):
                            dir_rgb = panorama_frames[frame_idx]['rgb']
                            self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb,
                                             np.ndarray) else dir_rgb
                            self.visited_targets[-1]['turn_action'] = f"turn {direction}"

                        # Each TURN_* action is 30 deg; chain them to reach the target heading.
                        if direction == 'left':
                            action_list.extend([2] * 3)  # TURN_LEFT 90 deg (3 * 30 deg)
                        elif direction == 'right':
                            action_list.extend([3] * 3)  # TURN_RIGHT 90 deg
                        elif direction == 'behind':
                            action_list.extend([2] * 6)  # TURN_LEFT 180 deg (6 * 30 deg)
                        # forward needs no turn

                        panorama_got = True

                # Step 2: panorama done, no queued actions -> ask the VA for a target bbox.
                elif target_map_x is None and panorama_got and not action_list:
                    progress_analysis = self.visited_targets[-1].get('progress_analysis', '')

                    bbox = self.agent.query_llm(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        rgb_image=obs[0]['rgb'],
                        depth_image=obs[0]['depth'],
                        width=self.width,
                        height=self.height,
                        current_step=self.current_step,
                        progress_analysis=progress_analysis,
                        planned_action=current_la_action,
                        episode_id=self.current_episode_id
                    )

                    self.visited_targets[-1].update({
                        'description': bbox.get('target', 'unknown target'),
                        'bbox': bbox,
                        'llm_reasoning': bbox.get('reasoning', ''),
                        'llm_progress': bbox.get('progress', '')
                    })

                    # If the current action is STOP, we are approaching the target.
                    # Once we reach this target (in Step 4), we should then trigger the double check.
                    if current_la_action == 'STOP':
                        going_to_stop = True

                    # Convert the VA bbox into a world coordinate target.
                    depth_image = self._preprocess_depth(obs[0]['depth'], 0.1, 5.0) / 100.0
                    if not self.agent.stair:
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), min(int(bbox.get('y2', 0)), self.height - 1))
                    else:
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y1', 0)))

                    # Pixel coords -> world coords (with retry loop, see body below).
                    while True:
                        target = get_world_xz_from_pixel(
                            pixel_coords=coords,
                            depth_image=depth_image,
                            full_pose=current_pose,
                            camera_intrinsics=self._get_camera_intrinsics(),
                        )
                        new_target_x = int(target[0] * 100.0 / self.resolution)
                        new_target_y = int(target[1] * 100.0 / self.resolution)
                        new_target_x = max(0, min(new_target_x, self.map_shape[0] - 1))
                        new_target_y = max(0, min(new_target_y, self.map_shape[1] - 1))

                        if self.traversable[new_target_y, new_target_x] == 1 or depth_image.max() < 0.1:
                            target_map_x, target_map_y = new_target_x, new_target_y
                            target_set_step = step
                            # logger.info(f"Target set at map coordinates: ({target_map_x}, {target_map_y}) at step {step}")

                            waypoint = np.array([target_map_y, target_map_x])

                            # Controller Selection Logic (Step 2)
                            use_special_controller = False
                            selected_controller = None # 'iplanner', 'navdp'

                            # Determine if we should use a special controller (NavDP/iPlanner)
                            # Logic:
                            # 1. if use_fmm is False -> MUST use special controller (NavDP or iPlanner, whichever is True)
                            # 2. if use_fmm is True -> use special controller ONLY IF (stair is True AND one of them is True)
                            
                            has_special_controller = self.use_navdp or self.use_iplanner
                            
                            if not self.use_fmm:
                                # FMM disabled, must use special controller
                                if self.use_iplanner: selected_controller = 'iplanner'
                                elif self.use_navdp: selected_controller = 'navdp'
                                use_special_controller = True
                            else:
                                # FMM enabled
                                if has_special_controller and self.agent.stair:
                                     # Hybrid mode: Use special controller for stairs
                                     if self.use_iplanner: selected_controller = 'iplanner'
                                     elif self.use_navdp: selected_controller = 'navdp'
                                     use_special_controller = True
                                else:
                                     # Use FMM (Normal case or stairs but no special controller enabled)
                                     use_special_controller = False

                            if use_special_controller:
                                traj_np = None  # Initialize to avoid undefined variable error
                                if selected_controller == 'iplanner':
                                    logger.info(f"DEBUG: Step 2 - Using iPlanner (stair={self.agent.stair})")
                                    # Calculate local goal from target_map_x, target_map_y
                                    # Convert map indices to world meters
                                    tx = target_map_x * self.resolution / 100.0
                                    ty = target_map_y * self.resolution / 100.0
                                
                                    ax = current_pose[0] # x in meters
                                    ay = current_pose[1] # y in meters (z in world)
                                    ah = np.deg2rad(current_pose[2]) # heading in radians (converted from degrees)
                                
                                    # Relative vector in world frame
                                    dx = tx - ax
                                    dy = ty - ay
                                    
                                    # Rotate to robot frame (X-forward, Y-left, Z-up)
                                    x_body = dx * np.cos(ah) + dy * np.sin(ah)   # Forward distance
                                    y_body = -dx * np.sin(ah) + dy * np.cos(ah)  # Left distance
                                    z_body = 0.0                                 # Up distance
                                    
                                    local_goal = torch.tensor([[x_body, y_body, z_body]]).to(self.device)
                                    
                                    # Call iplanner (Force update)
                                    min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                                    max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                                    depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                                    
                                    dep_tensor = torch.from_numpy(depth_meters).to(self.device).unsqueeze(0) # (1, H, W, 1)
                                    keypoints, traj, fear = self.iplanner_agent.step_pointgoal(dep_tensor, local_goal)
                                    traj_np = None  # Fix: initialize to avoid undefined variable
                                    
                                    # Store Global Trajectory
                                    traj_np = traj[0].cpu().numpy()
                                    global_traj = []
                                    for pt in traj_np:
                                        xb, yb, zb = pt
                                        w_dx = xb * np.cos(ah) - yb * np.sin(ah)
                                        w_dy = xb * np.sin(ah) + yb * np.cos(ah)
                                        wx = ax + w_dx
                                        wy = ay + w_dy
                                        global_traj.append([wx, wy, zb])
                                    self.navdp_traj_global = np.array(global_traj)
                                    self.last_iplanner_step = step

                                    if traj_np is not None:  # Check if traj_np is defined
                                        # Visualization (Project traj_np to depth)
                                        if self.visualize:
                                            try:
                                                # Use traj_np (Local Frame)
                                                viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                                if viz_depth is not None:
                                                    if len(viz_depth.shape) == 2:
                                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                                    elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                                    
                                                    intrinsics = self._get_camera_intrinsics()
                                                    fx, fy = intrinsics[0,0], intrinsics[1,1]
                                                    cx, cy = intrinsics[0,2], intrinsics[1,2]
                                                    h, w = viz_depth.shape[:2]
                                                    
                                                    valid_points = []
                                                    for pt in traj_np:
                                                        x, y, z = pt # x=forward, y=left, z=up
                                                        
                                                        xc = -y
                                                        yc = -z 
                                                        zc = x
                                                        
                                                        if x > 0.1:
                                                            u = fx * (-y) / x + cx
                                                            u = int(u)
                                                            v = int(fy * (self.config.MAP.AGENT_HEIGHT - z) / x + cy)
                                                            
                                                            if 0 <= u < w and 0 <= v < h:
                                                                valid_points.append((u, v))

                                                    # Draw lines
                                                    for i in range(len(valid_points)-1):
                                                        cv2.line(viz_depth, valid_points[i], valid_points[i+1], (0, 0, 255), 2) # Red
                                                    for pt in valid_points:
                                                        cv2.circle(viz_depth, pt, 3, (0, 255, 0), -1) # Green
                                                        
                                                    # Save
                                                    episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                                    os.makedirs(episode_dir, exist_ok=True)
                                                    cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                                            except Exception as e:
                                                logger.info(f"Error visualizing iplanner: {e}")

                                        # Direct Control Action
                                        # Find target point
                                        lookahead_dist = 1.0
                                        target_pt = None
                                        for pt in traj_np:
                                            dist = np.linalg.norm(pt[:2]) # x, y
                                            if dist > lookahead_dist:
                                                target_pt = pt
                                                break
                                        if target_pt is None and len(traj_np) > 0:
                                            target_pt = traj_np[-1]
                                        
                                        action_id = HabitatSimActions.MOVE_FORWARD
                                        if target_pt is not None:
                                            x, y = target_pt[:2]
                                            angle = np.arctan2(y, x) # y is left, x is forward. +angle = left.
                                            # Threshold 15 deg = 0.26 rad
                                            if angle > 0.26:
                                                action_id = HabitatSimActions.TURN_LEFT
                                            elif angle < -0.26:
                                                action_id = HabitatSimActions.TURN_RIGHT
                                            else:
                                                action_id = HabitatSimActions.MOVE_FORWARD
                                        
                                        action_list.append(action_id)
                                        break # Skip FMM
                                elif selected_controller == 'navdp':
                                    # logger.info(f"DEBUG: Step 2 - Using NavDP (stair={self.agent.stair})")
                                    tx = target_map_x * self.resolution / 100.0
                                    ty = target_map_y * self.resolution / 100.0
                                    ax = current_pose[0]
                                    ay = current_pose[1]
                                    ah = np.deg2rad(current_pose[2])
                                    
                                    dx = tx - ax
                                    dy = ty - ay
                                    
                                    x_body = dx * np.cos(ah) + dy * np.sin(ah)
                                    y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                                    z_body = 0.0
                                    
                                    goals = np.array([[x_body, y_body, z_body]])
                                    rgb_img = obs[0]['rgb']
                                    min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                                    max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                                    depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                                    depth_img = depth_meters
                                    
                                    if depth_img.ndim == 2:
                                        depth_img = depth_img[:, :, np.newaxis]
                                    images = rgb_img[np.newaxis, ...]
                                    depths = depth_img[np.newaxis, ...]
                                    
                                    next_point, traj, val, mask = self.navdp_agent.step_pointgoal(goals, images, depths)
                                    
                                    # Visualization
                                    local_traj = next_point[0]
                                    global_traj = []
                                    for pt in local_traj:
                                        lb_x, lb_y = pt[:2]
                                        wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                                        wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                                        global_traj.append([wx, wy])
                                    self.navdp_traj_global = np.array(global_traj)

                                    x, y = local_traj[0][:2]
                                    angle = np.arctan2(y, x)
                                    if angle > 0.26:
                                        action_id = HabitatSimActions.TURN_LEFT
                                    elif angle < -0.26:
                                        action_id = HabitatSimActions.TURN_RIGHT
                                    else:
                                        action_id = HabitatSimActions.MOVE_FORWARD
                                    
                                    action_list.append(action_id)
                                    break


                            # Emit the first navigation action immediately so the agent doesn't idle.
                            if not use_special_controller and self.use_fmm:
                                # Add Visualization for FMM
                                if self.visualize:
                                    try:
                                        viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                        if viz_depth is not None:
                                            if len(viz_depth.shape) == 2:
                                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                            elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                            
                                            intrinsics = self._get_camera_intrinsics()
                                            fx, fy = intrinsics[0,0], intrinsics[1,1]
                                            cx, cy = intrinsics[0,2], intrinsics[1,2]
                                            h, w = viz_depth.shape[:2]

                                            # Draw Goal
                                            ax = current_pose[0]
                                            ay = current_pose[1]
                                            ah = np.deg2rad(current_pose[2])
                                            
                                            tx_viz = target_map_x * self.resolution / 100.0
                                            ty_viz = target_map_y * self.resolution / 100.0
                                            dx_viz = tx_viz - ax
                                            dy_viz = ty_viz - ay
                                            gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                                            gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                                            gz = 0.0
                                            
                                            if gx > 0.1:
                                                gu = fx * (-gy) / gx + cx
                                                gv = fy * (self.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                                gu, gv = int(gu), int(gv)
                                                if 0 <= gu < w and 0 <= gv < h:
                                                    # Draw Yellow Star for Goal
                                                    cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                                    cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                                            episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                            os.makedirs(episode_dir, exist_ok=True)
                                            cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                                    except Exception as e:
                                        logger.info(f"Error in FMM viz: {e}")

                                navigation_action = self.policy._get_action(
                                    current_pose, waypoint, full_map[0], self.traversable,
                                    self.collision_map, step, self.current_episode_id,
                                    self.detected_classes, search_destination
                                )
                                action_list.append(navigation_action)
                            elif not use_special_controller and not self.use_fmm:
                                logger.warning("Step 2 - FMM is disabled and no other planner took over! This should not happen due to init validation.")
                            break
                        depth_image = depth_image - 0.1

                    panorama_got = False  # reset so the next Step 1 re-spins

                # Step 3: have a target and no queued actions -> keep planning toward it.
                elif target_map_x is not None and target_map_y is not None and not action_list:
                    if navigation_mode == "SG_NAV":
                        world_target_xz = (sg_target_info or {}).get("target_world_xz")
                        if world_target_xz is not None:
                            current_world_xz = np.asarray(
                                obs[0]["position"], dtype=np.float64
                            ).reshape(-1)[[0, 2]]
                            dist_to_tgt = float(
                                np.linalg.norm(
                                    current_world_xz
                                    - np.asarray(world_target_xz, dtype=np.float64)
                                )
                                / float(sg_target_info["world_map_resolution_m"])
                            )
                        else:
                            dist_to_tgt = np.sqrt((target_map_x - agent_map_x) ** 2 + (target_map_y - agent_map_y) ** 2)
                        abort_reason = self._sg_nav_progress_abort_reason(
                            step, dist_to_tgt, target_info=sg_target_info
                        )
                        if abort_reason:
                            sg_recovered = self._on_sg_nav_aborted(
                                step=step,
                                target_info=sg_target_info,
                                reason=abort_reason,
                                distance_to_target=dist_to_tgt,
                            )
                            if sg_recovered is True:
                                # The complete exploration controller replaced
                                # the failed SG path with a scan/frontier/probe
                                # state in-place.  Keep SG_NAV and its target
                                # metadata alive for the next physical action.
                                continue
                            panorama_got = False
                            navigate_or_not = False
                            target_map_x, target_map_y = None, None
                            target_set_step = None
                            sg_target_info = None
                            navigation_mode = "NORMAL"
                            continue
                    waypoint = np.array([target_map_y, target_map_x])
                    
                    # Controller Selection Logic (Step 3)
                    use_special_controller = False
                    selected_controller = None
                    sg_path_action = None

                    if navigation_mode == "SG_NAV":
                        sg_path_action = self._get_sg_path_action(
                            step=step,
                            current_pose=current_pose,
                            current_world_position=obs[0]["position"],
                            current_world_heading=obs[0]["heading"],
                            target_map_x=target_map_x,
                            target_map_y=target_map_y,
                            target_info=sg_target_info,
                        )
                        if sg_path_action is None:
                            abort_reason = (sg_target_info or {}).pop(
                                "_sg_pending_abort_reason",
                                "world_path_controller_unavailable",
                            )
                            sg_recovered = self._on_sg_nav_aborted(
                                step=step,
                                target_info=sg_target_info,
                                reason=abort_reason,
                                distance_to_target=dist_to_tgt,
                            )
                            if sg_recovered is True:
                                continue
                            panorama_got = False
                            navigate_or_not = False
                            target_map_x, target_map_y = None, None
                            target_set_step = None
                            sg_target_info = None
                            navigation_mode = "NORMAL"
                            continue
                        selected_controller = "sg_path"
                        use_special_controller = True
                    
                    has_special_controller = self.use_navdp or self.use_iplanner
                    
                    if selected_controller == "sg_path":
                        pass
                    elif not self.use_fmm:
                        if self.use_iplanner: selected_controller = 'iplanner'
                        elif self.use_navdp: selected_controller = 'navdp'
                        use_special_controller = True
                    else:
                        if has_special_controller and self.agent.stair:
                             if self.use_iplanner: selected_controller = 'iplanner'
                             elif self.use_navdp: selected_controller = 'navdp'
                             use_special_controller = True
                        else:
                             use_special_controller = False

                    if selected_controller == "sg_path":
                        action_list.append(sg_path_action)
                    elif selected_controller == 'iplanner':
                        traj_local_np = None
                        
                        ax = current_pose[0]
                        ay = current_pose[1]
                        ah = np.deg2rad(current_pose[2])
                        
                        if step - self.last_iplanner_step >= self.iplanner_update_freq or self.navdp_traj_global is None:
                            # Replan
                            tx = target_map_x * self.resolution / 100.0
                            ty = target_map_y * self.resolution / 100.0
                            
                            dx = tx - ax
                            dy = ty - ay
                            
                            x_body = dx * np.cos(ah) + dy * np.sin(ah)
                            y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                            z_body = 0.0
                            
                            local_goal = torch.tensor([[x_body, y_body, z_body]]).to(self.device)
                            
                            min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                            max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                            depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                            dep_tensor = torch.from_numpy(depth_meters).to(self.device).unsqueeze(0)
                            
                            keypoints, traj, fear = self.iplanner_agent.step_pointgoal(dep_tensor, local_goal)
                            
                            # Update Global
                            traj_local_np = traj[0].cpu().numpy()
                            global_traj = []
                            for pt in traj_local_np:
                                xb, yb, zb = pt
                                w_dx = xb * np.cos(ah) - yb * np.sin(ah)
                                w_dy = xb * np.sin(ah) + yb * np.cos(ah)
                                wx = ax + w_dx
                                wy = ay + w_dy
                                global_traj.append([wx, wy, zb])
                            self.navdp_traj_global = np.array(global_traj)
                            self.last_iplanner_step = step
                            logger.info(f"DEBUG: iPlanner generated trajectory with {len(global_traj)} points in Step 3")
                        else:
                            # Use stored global traj -> local
                            if self.navdp_traj_global is not None and len(self.navdp_traj_global) > 0:
                                local_traj = []
                                for gpt in self.navdp_traj_global:
                                    gx, gy, gz = gpt
                                    dx = gx - ax
                                    dy = gy - ay
                                    # Inverse rotate
                                    xb = dx * np.cos(ah) + dy * np.sin(ah)
                                    yb = -dx * np.sin(ah) + dy * np.cos(ah)
                                    local_traj.append([xb, yb, gz])
                                traj_local_np = np.array(local_traj)
                            else:
                                traj_local_np = np.array([])

                        # Unified Visualization on VLM Map (Update current frame)
                        if self.visualize and self.navdp_traj_global is not None:
                            try:
                                traj_world_list = [(pt[0], pt[1]) for pt in self.navdp_traj_global]
                                self.visualizer._save_rgb_frame(
                                    obs[0], 
                                    step, 
                                    self.visited_targets, 
                                    self.current_episode_id, 
                                    (target_map_x, target_map_y), 
                                    todo_list=self.agent._format_todo_for_prompt() if self.agent.use_todo_list else None,
                                    navdp_traj=traj_world_list
                                )
                            except Exception as e:
                                logger.info(f"Error updating VLM map viz in Step 3: {e}")

                        # Visualization (Project traj_local_np to depth)
                        if self.visualize:
                            try:
                                viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                if viz_depth is not None:
                                    if len(viz_depth.shape) == 2:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    
                                    intrinsics = self._get_camera_intrinsics()
                                    fx, fy = intrinsics[0,0], intrinsics[1,1]
                                    cx, cy = intrinsics[0,2], intrinsics[1,2]
                                    h, w = viz_depth.shape[:2]
                                    
                                    valid_points = []
                                    if traj_local_np is not None:
                                        for pt in traj_local_np:
                                            x, y, z = pt
                                            if x > 0.1:
                                                u = fx * (-y) / x + cx
                                                v = fy * (self.config.MAP.AGENT_HEIGHT - z) / x + cy
                                                u = int(u)
                                                v = int(v)
                                                if 0 <= u < w and 0 <= v < h:
                                                    valid_points.append((u, v))
                                    
                                    for i in range(len(valid_points)-1):
                                        cv2.line(viz_depth, valid_points[i], valid_points[i+1], (0, 0, 255), 2)
                                    for pt in valid_points:
                                        cv2.circle(viz_depth, pt, 3, (0, 255, 0), -1)
                                        
                                    # Draw Goal
                                    # Recalculate local goal for visualization
                                    tx_viz = target_map_x * self.resolution / 100.0
                                    ty_viz = target_map_y * self.resolution / 100.0
                                    dx_viz = tx_viz - ax
                                    dy_viz = ty_viz - ay
                                    gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                                    gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                                    gz = 0.0
                                    
                                    if gx > 0.1:
                                        gu = fx * (-gy) / gx + cx
                                        gv = fy * (self.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                        gu, gv = int(gu), int(gv)
                                        if 0 <= gu < w and 0 <= gv < h:
                                            # Draw Yellow Star for Goal
                                            cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                            cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                                    episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                    os.makedirs(episode_dir, exist_ok=True)
                                    cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                            except Exception as e:
                                logger.info(f"Error in iplanner viz: {e}")

                        # Action
                        lookahead_dist = 0.1
                        target_pt = None
                        if traj_local_np is not None:
                            for pt in traj_local_np:
                                dist = np.linalg.norm(pt[:2])
                                if dist > lookahead_dist:
                                    target_pt = pt
                                    break
                            if target_pt is None and len(traj_local_np) > 0:
                                target_pt = traj_local_np[-1]
                        
                        action_id = HabitatSimActions.MOVE_FORWARD
                        if target_pt is not None:
                            x, y = target_pt[:2]
                            angle = np.arctan2(y, x)
                            if angle > 0.26:
                                action_id = HabitatSimActions.TURN_LEFT
                            elif angle < -0.26:
                                action_id = HabitatSimActions.TURN_RIGHT
                            else:
                                action_id = HabitatSimActions.MOVE_FORWARD
                        
                        action_list.append(action_id)
                    elif selected_controller == 'navdp':
                        tx = target_map_x * self.resolution / 100.0
                        ty = target_map_y * self.resolution / 100.0
                        ax = current_pose[0]
                        ay = current_pose[1]
                        ah = np.deg2rad(current_pose[2])
                        
                        dx = tx - ax
                        dy = ty - ay
                        x_body = dx * np.cos(ah) + dy * np.sin(ah)
                        y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                        z_body = 0.0
                        
                        goals = np.array([[x_body, y_body, z_body]])
                        rgb_img = obs[0]['rgb']
                        min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                        max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                        depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                        depth_img = depth_meters
                        # NavDP Frequency Control
                        # Initialize variables if not present
                        if not hasattr(self, 'last_navdp_step'):
                            self.last_navdp_step = -999
                        if not hasattr(self, 'navdp_local_traj'):
                            self.navdp_local_traj = None

                        # Execute NavDP planning every 5 steps or if no trajectory exists
                        if step - self.last_navdp_step >= 5 or self.navdp_local_traj is None or len(self.navdp_local_traj) == 0:
                            # logger.info(f"DEBUG: Executing NavDP planning at step {step}")
                            
                            if depth_img.ndim == 2:
                                depth_img = depth_img[:, :, np.newaxis]
                            images = rgb_img[np.newaxis, ...]
                            depths = depth_img[np.newaxis, ...]
                            
                            next_point, traj, val, mask = self.navdp_agent.step_pointgoal(goals, images, depths)
                            
                            # Store the new trajectory and update timestamp
                            self.navdp_local_traj = next_point[0] # (N, 3)
                            self.last_navdp_step = step
                            
                            # Update Global Visualization (Optional, but good for debugging)
                            local_traj = self.navdp_local_traj
                            global_traj = []
                            for pt in local_traj:
                                lb_x, lb_y = pt[:2]
                                wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                                wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                                global_traj.append([wx, wy])
                            self.navdp_traj_global = np.array(global_traj)
                        
                        # Let's use the GLOBAL trajectory method for stability across steps.
                        if self.navdp_traj_global is not None and len(self.navdp_traj_global) > 0:
                            # Find a lookahead point on the global trajectory relative to CURRENT pose
                            target_pt = None
                            lookahead_dist = 0.5
                            
                            # Convert global trajectory back to CURRENT local frame
                            current_local_traj = []
                            for gpt in self.navdp_traj_global:
                                gx, gy = gpt
                                dx = gx - ax
                                dy = gy - ay
                                # Inverse rotate to get local x, y
                                lx = dx * np.cos(ah) + dy * np.sin(ah)
                                ly = -dx * np.sin(ah) + dy * np.cos(ah)
                                current_local_traj.append([lx, ly])
                            
                            # Now find the target point in current local frame
                            for pt in current_local_traj:
                                dist = np.linalg.norm(pt[:2])
                                if dist > lookahead_dist:
                                    target_pt = pt
                                    break
                            
                            if target_pt is None and len(current_local_traj) > 0:
                                target_pt = current_local_traj[-1]
                        else:
                            # Fallback
                            target_pt = None

                        if target_pt is not None:
                            x, y = target_pt[:2]
                            angle = np.arctan2(y, x)
                            if angle > 0.26:
                                action_id = HabitatSimActions.TURN_LEFT
                            elif angle < -0.26:
                                action_id = HabitatSimActions.TURN_RIGHT
                            else:
                                action_id = HabitatSimActions.MOVE_FORWARD
                        else:
                            action_id = HabitatSimActions.MOVE_FORWARD
                        
                        action_list.append(action_id)
                    elif not use_special_controller and self.use_fmm:
                        logger.info(f"DEBUG: Step 3 - Using FMM. use_iplanner={self.use_iplanner}, use_navdp={self.use_navdp}, stair={self.agent.stair}")
                        # Clear special controller trajectories to avoid stale visualization
                        self.navdp_traj_global = None
                        self.navdp_local_traj = None
                        
                        # Add Visualization for FMM
                        if self.visualize:
                            try:
                                viz_depth = self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                if viz_depth is not None:
                                    if len(viz_depth.shape) == 2:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                        viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                    
                                    intrinsics = self._get_camera_intrinsics()
                                    fx, fy = intrinsics[0,0], intrinsics[1,1]
                                    cx, cy = intrinsics[0,2], intrinsics[1,2]
                                    h, w = viz_depth.shape[:2]

                                    # Draw Goal
                                    ax = current_pose[0]
                                    ay = current_pose[1]
                                    ah = np.deg2rad(current_pose[2])
                                    
                                    tx_viz = target_map_x * self.resolution / 100.0
                                    ty_viz = target_map_y * self.resolution / 100.0
                                    dx_viz = tx_viz - ax
                                    dy_viz = ty_viz - ay
                                    gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                                    gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                                    gz = 0.0
                                    
                                    if gx > 0.1:
                                        gu = fx * (-gy) / gx + cx
                                        gv = fy * (self.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                        gu, gv = int(gu), int(gv)
                                        if 0 <= gu < w and 0 <= gv < h:
                                            # Draw Yellow Star for Goal
                                            cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                            cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                                    episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
                                    os.makedirs(episode_dir, exist_ok=True)
                                    cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                            except Exception as e:
                                logger.info(f"Error in FMM viz: {e}")

                        navigation_action = self.policy._get_action(
                            current_pose, waypoint, full_map[0], self.traversable,
                            self.collision_map, step, self.current_episode_id,
                            self.detected_classes, search_destination
                        )
                        action_list.append(navigation_action)
                    else:
                        logger.warning("Step 3 - FMM is disabled and no other planner took over!")
                    # logger.info(f"Added navigation action: {navigation_action}")


            # Execute only when there is a queued action.
            if action_list:
                # =================================================================
                # 3. ACT for step N: execute the chosen action.
                # =================================================================
                self._action = action_list[0]
                action_list.pop(0)
                actions = [{"action": self._action}]

                outputs = self.envs.step(actions)
                step += 1

                # =================================================================
                # 4. UPDATE for step N+1: refresh map and pose from the new obs.
                # =================================================================
                obs, _, dones, infos = [list(x) for x in zip(*outputs)]

                if not dones[0]:
                    batch_obs = self._batch_obs(obs)
                    poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
                    self.mapping_module(batch_obs, poses, self.current_step)
                    full_map, full_pose, one_step_full_map = \
                        self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
                    self.mapping_module.one_step_full_map.fill_(0.)
                    self.mapping_module.one_step_local_map.fill_(0.)

                    self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
                    self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

                    last_pose = current_pose
                    current_pose = full_pose[0]
                    current_action = self._action
                    if last_pose is not None and current_pose is not None:
                        displacement = calculate_displacement(last_pose, current_pose, self.resolution)
                        expected_forward_cells = (
                            float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
                            * 100.0
                            / self.resolution
                        )
                        if (
                            current_action == HabitatSimActions.MOVE_FORWARD
                            and displacement < expected_forward_cells * 0.5
                        ):
                            self.collision_count += 1
                        if displacement < 0.2 * 100 / self.resolution:
                            collided += 1
                        else:
                            collided = 0
                            replan = False
                        if collided >= 30:
                            fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                                 f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                            with open(fname, "a") as f:
                                f.writelines(
                                    f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")

                    if last_pose is not None and current_action is not None and current_action == 1:
                        collision_map = collision_check_fmm(last_pose, current_pose, self.resolution,
                                                            self.mapping_module.map_shape)
                        self.collision_map = np.logical_or(self.collision_map, collision_map)
                    self.traversable[self.collision_map == 1] = 0
                else:
                    self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None))
                    return
                # For EQA, STOP action does not set dones[0]=True; exit here
                # after the Oracle QA branch has set the done-requested flag.
                if getattr(self, '_eqa_done_requested', False):
                    self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None))
                    return
            else:
                # action_list is empty: skip ACT and let the next iteration plan again.
                pass
            action_step += 1
        self._calculate_metric(infos, submitted_answer=getattr(self, 'last_submitted_answer', None), is_timeout=True)

    def eval(self):
        # Load detailed assignments if available. File is namespaced by exp_name
        # (derived from the unique EVAL_CKPT_PATH_DIR) to allow parallel runs.
        exp_name = os.path.basename(self.config.EVAL_CKPT_PATH_DIR.rstrip('/'))
        assignment_file = f"data/logs/running_log/worker_{self.local_rank}_assignments_{exp_name}.json"
        if not os.path.exists(assignment_file):
            # Backward compat: fall back to the un-namespaced filename
            assignment_file = f"data/logs/running_log/worker_{self.local_rank}_assignments.json"
        if os.path.exists(assignment_file):
            with open(assignment_file, 'r') as f:
                assignments = json.load(f)
                # Set of (id, scene_id) tuples for fast lookup
                self.allowed_episodes = set((str(item['id']), item['scene_id']) for item in assignments)
            logger.info(f"Loaded {len(self.allowed_episodes)} specific episodes to process from {assignment_file}")

        # Sampling logic
        if self.config.EVAL.EPISODE_COUNT > -1:
            pass # Sampling is handled in run_mp.py, here we just use the assigned episodes


        # self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        self.agent.reset()

        # Reset model usage stats
        self.agent.model.reset_stats()

        eps_to_eval = sum(self.envs.number_of_episodes)

        self.state_eps = {}
        t1 = time.time()
        
        logger.info(f"Worker {self.local_rank}: Starting eval loop with {eps_to_eval} episodes")
        
        for i in tqdm(range(eps_to_eval), desc=f"Worker {self.local_rank}"):
            self.rollout()
            self.reset()

        self.envs.close()

        # Print final statistics
        logger.info("=== FINAL MODEL USAGE STATISTICS ===")
        final_stats = self.agent.model.print_usage_stats()

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)

        # Calculate and print average metrics
        # avg_metrics = defaultdict(float)
        # num_episodes = len(self.state_eps)
        # if num_episodes > 0:
        #     for metrics in self.state_eps.values():
        #         for k, v in metrics.items():
        #             if isinstance(v, (int, float)):
        #                 avg_metrics[k] += v

        #     for k in avg_metrics:
        #         avg_metrics[k] /= num_episodes

        #     logger.info("Average Metrics:")
        #     logger.info(json.dumps(dict(avg_metrics), indent=2))
        # else:
        #     logger.info("No episodes evaluated.")

        # Save model usage stats to separate file
        stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                   f"model_usage_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(stats_fname, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Model usage statistics saved to: {stats_fname}")

        # Save navigation statistics (backtracks, waypoints)
        nav_stats = {
            'total_backtracks': self.total_backtracks,
            'total_waypoints': self.total_waypoints
        }
        nav_stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                       f"nav_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(nav_stats_fname, "w") as f:
            json.dump(nav_stats, f, indent=2)
        logger.info(f"Navigation statistics saved to: {nav_stats_fname}")

        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        logger.info("test time: %d", t2 - t1)

    def eval_dynamic(self, ep_queue):
        """Dynamic work-stealing eval: pop episodes from a shared mp.Queue.

        Each worker reuses its trainer/policy/model across episodes but
        rebuilds the habitat env per episode (so EPISODES_ALLOWED can be
        re-targeted to a single id). The shared queue lets fast workers
        grab the next pending episode instead of being stuck with a fixed
        slice.
        """
        import queue as _queue
        # Heavy initialisation done once per worker
        self._initialize_policy()
        self._collect_val_traj()
        self.agent.model.reset_stats()
        self.state_eps = {}
        t1 = time.time()
        ep_count = 0
        logger.info(f"Worker {self.local_rank}: dynamic queue eval starting")

        while True:
            try:
                ep_info = ep_queue.get(timeout=2)
            except _queue.Empty:
                logger.info(f"Worker {self.local_rank}: queue empty, exiting after {ep_count} eps")
                break
            ep_id = str(ep_info['id'])
            scene_id = ep_info['scene_id']

            # Re-target the dataset filter to a single ep id and rebuild env.
            self.config.defrost()
            self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [ep_id]
            self.config.freeze()
            self.allowed_episodes = {(ep_id, scene_id)}
            if hasattr(self, 'envs') and self.envs is not None:
                try:
                    self.envs.close()
                except Exception as e:
                    logger.warning(f"envs.close() failed: {e}")
            self._init_envs()
            self.agent.reset()

            try:
                self.rollout()
            except Exception as exc:
                logger.exception(
                    f"Worker {self.local_rank} ep {ep_id} rollout failed"
                )
                raise RuntimeError(f"Episode {ep_id} rollout failed") from exc
            try:
                self.reset()
            except Exception as exc:
                logger.exception(
                    f"Worker {self.local_rank} ep {ep_id} reset failed"
                )
                raise RuntimeError(f"Episode {ep_id} reset failed") from exc
            ep_count += 1

        # Final cleanup + stats persistence (mirrors eval())
        if hasattr(self, 'envs') and self.envs is not None:
            try:
                self.envs.close()
            except Exception:
                pass

        final_stats = self.agent.model.print_usage_stats()
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        stem = f"{split}_r{self.local_rank}_w{self.world_size}"
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"stats_ep_ckpt_{stem}.json"), "w") as f:
            json.dump(self.state_eps, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"model_usage_stats_{stem}.json"), "w") as f:
            json.dump(final_stats, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"nav_stats_{stem}.json"), "w") as f:
            json.dump({'total_backtracks': self.total_backtracks, 'total_waypoints': self.total_waypoints}, f, indent=2)
        logger.info(f"Worker {self.local_rank}: ran {ep_count} eps in {time.time() - t1:.0f}s")
